$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '1'
$repo = 'D:\LLM\Backend\src\llama.cpp-my'
$srv  = "$repo\build\bin\Release\llama-server.exe"
$Model = '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf'
$Port = 8943
$out = '<TEMP>\v100\t32-repro315b'
Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $out | Out-Null
$dir = "$out\tree"; $save = "$out\slot-save"
New-Item -ItemType Directory -Force $save | Out-Null
$prompt = [IO.File]::ReadAllText('<TEMP>\v100\t32-stage3\soak-mismatch-round315-prompt.txt', (New-Object System.Text.UTF8Encoding($false)))

$sargs = @('-m',$Model,'-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','16384',
          '--port',"$Port",'--cache-ram','0','--no-context-shift','--host','127.0.0.1','--cache-idle-slots',
          '--slot-save-path',$save,'--slot-prompt-similarity','0',
          '--kv-tree','--tree-chunk','512','--tree-checkpoint-anchor-step','4096',
          '--tree-ram','64','--tree-disk',$dir,'--tree-disk-limit','512')
$p = Start-Process -FilePath $srv -ArgumentList $sargs -NoNewWindow -PassThru -RedirectStandardOutput "$out\srv-out.txt" -RedirectStandardError "$out\srv-err.txt"
for ($i=0; $i -lt 120; $i++) { Start-Sleep -Milliseconds 1000; if ($p.HasExited) { throw "server exited" }; try { Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3 | Out-Null; break } catch {} }

function Ask($prompt, $cache, $tag) {
    $body = @{prompt=$prompt; n_predict=32; temperature=0; top_k=1; seed=42; cache_prompt=$cache; stream=$false; n_probs=5; message_delimiters=@(@{role='user'; delimiter='User:'})}
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -TimeoutSec 300 -Body ($body | ConvertTo-Json -Compress -Depth 4)
    [IO.File]::WriteAllText("$out\$tag.txt", $r.content, (New-Object System.Text.UTF8Encoding($false)))
    Write-Output "[$tag] prompt_n=$($r.timings.prompt_n) pred=$($r.timings.predicted_n) content=$($r.content -replace "`n",'\n')"
    return $r
}

$r1 = Ask $prompt $true  'req1-cold'
$r2 = Ask $prompt $true  'req2-tree'
$r3 = Ask $prompt $false 'req3-full'
$r4 = Ask $prompt $true  'req4-tree'
Write-Output "tree2 == full3: $($r2.content -eq $r3.content)"
Write-Output "tree2 == tree4: $($r2.content -eq $r4.content)"
Write-Output "cold1 == full3: $($r1.content -eq $r3.content)"
Stop-Process -Id $p.Id -Force; $p.WaitForExit(10000) | Out-Null
Write-Output '=== restore/heal lines ==='
Get-Content "$out\srv-err.txt" -Encoding UTF8 | Select-String 'kv tree: restored|restore miss|rebuilt|captured heal' | ForEach-Object { $_.Line }