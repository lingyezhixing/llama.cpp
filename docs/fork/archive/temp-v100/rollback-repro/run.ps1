$ErrorActionPreference = 'Stop'
$bin   = 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\llama-server.exe'
$model = '<models>\Qwen3.5-0.8B-MTP\Qwen3.5-0.8B-UD-Q4_K_XL.gguf'
$dir   = '<TEMP>\v100\rollback-repro'
$port  = 18399
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$env:CUDA_VISIBLE_DEVICES = '1'

# deterministic filler: each sentence is similar in length
$sA = 'Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima number %d end of sentence. '
$sB = 'Mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu number %d end of sentence. '
$A = (1000..1349 | ForEach-Object { $sA -f $_ }) -join ''   # ~ 4500 tokens
$B = (2000..2249 | ForEach-Object { $sB -f $_ }) -join ''   # ~ 3500 tokens

$err = Join-Path $dir 'srv.err.log'
$out = Join-Path $dir 'srv.out.log'
Remove-Item $err,$out -Force -ErrorAction SilentlyContinue

$argList = @('-m', $model, '-ngl', '99', '-c', '16384', '-fa', 'on',
             '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
             '--kv-tree', '--tree-ram', '512', '--tree-disk', (Join-Path $dir 'kv-tree'), '--tree-disk-limit', '4096',
             '--ctx-checkpoints', '32', '--checkpoint-min-step', '2048',
             '-np', '1', '--no-warmup', '-lv', '4', '--host', '127.0.0.1', '--port', "$port")
$p = Start-Process -FilePath $bin -ArgumentList $argList -RedirectStandardError $err -RedirectStandardOutput $out -PassThru

$ok = $false
for ($i = 0; $i -lt 180; $i++) {
    Start-Sleep -Milliseconds 1000
    if ($p.HasExited) { break }
    try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { $ok = $true; break } } catch { }
}
Write-Output ("served = {0}" -f $ok)
if (-not $ok) { Write-Output 'server failed to start'; Get-Content $err | Select-Object -Last 20; exit 1 }

function Ask([string]$name, [string]$prompt, [int]$n) {
    $body = @{ prompt = $prompt; n_predict = $n; temperature = 0; seed = 7; cache_prompt = $true } | ConvertTo-Json -Depth 5
    $t0 = Get-Date
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600
        $txt = $resp.content
    } catch { $txt = "REQUEST_FAILED: $_" }
    $dt = ((Get-Date) - $t0).TotalSeconds
    Write-Output ("[{0}] {1:N1}s -> '{2}'" -f $name, $dt, ($txt -replace "`r?`n", '\n'))
}

Ask 'R1 long (A+B)'      ($A + $B) 4
Start-Sleep -Seconds 1
Ask 'R2 rollback (A)'    $A        4     # f_keep ~ 0.56 -> stock path
Start-Sleep -Seconds 1
Ask 'R3 rollback (A) #2' $A        4
Start-Sleep -Seconds 1
Ask 'R4 extend (A+B2)'   ($A + $B) 4

Start-Sleep -Seconds 2
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }

Write-Output ''
Write-Output '=== interesting log lines ==='
Select-String -Path $err -Pattern 'kv tree: (restored|restore miss|parked|rebuilt|heal)|restored context checkpoint|forcing full prompt re-processing|need to evaluate at least 1 token|n_past was set|failed to trim|GGML_ABORT|cache reuse|created context checkpoint|erasing old context checkpoint|erasing context checkpoint too close|superseding context checkpoint|checking checkpoint|selected slot by|stop processing' |
  ForEach-Object { $_.Line -replace '^\d+\.\d+\.\d+\.\d+ ', '' } | Select-Object -Last 80
