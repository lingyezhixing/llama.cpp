$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '1'
$repo = 'D:\LLM\Backend\src\llama.cpp-my'
$srv  = "$repo\build\bin\Release\llama-server.exe"
$Model = '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf'
$Port = 8941
$out = '<TEMP>\v100\t32-stock-repro'
New-Item -ItemType Directory -Force $out | Out-Null

$sargs = @('-m',$Model,'-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','16384',
          '--port',"$Port",'--cache-ram','0','--no-context-shift','--host','127.0.0.1','--cache-idle-slots')
$p = Start-Process -FilePath $srv -ArgumentList $sargs -NoNewWindow -PassThru -RedirectStandardOutput "$out\out.txt" -RedirectStandardError "$out\err.txt"
for ($i=0; $i -lt 120; $i++) { Start-Sleep -Milliseconds 1000; if ($p.HasExited) { throw "server exited early" }; try { Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3 | Out-Null; break } catch {} }

function TokCount($text) { (Invoke-RestMethod -Uri "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType 'application/json' -TimeoutSec 60 -Body (@{content=$text} | ConvertTo-Json -Compress)).tokens.Count }
function Filler($salt, $n_tok) {
    $unit = "$salt filler sentence for the kv tree acceptance test number "
    $ut = TokCount $unit
    $t = ''; $cur = 0
    while ($cur -lt $n_tok) { $need = [int][math]::Ceiling(($n_tok - $cur) / $ut); $t += ($unit * $need); $cur = TokCount $t }
    return $t
}
function Req($prompt) {
    $body = @{prompt=$prompt; n_predict=32; temperature=0; top_k=1; seed=42; cache_prompt=$true; stream=$false; message_delimiters=@(@{role='user'; delimiter='User:'})}
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -TimeoutSec 300 -Body ($body | ConvertTo-Json -Compress -Depth 4)
}

$sys = Filler 'system' 512
$pA = "$sys`nUser: " + (Filler 'alpha' 3200) + "`nUser turn 1: continue`nAssistant:"
$pB = "$sys`nUser: " + (Filler 'beta' 3200) + "`nUser turn 2: continue`nAssistant:"
Write-Output "tokens A=$(TokCount $pA) B=$(TokCount $pB)"

$rA = Req $pA
Write-Output "A ok: prompt_n=$($rA.timings.prompt_n) pred=$($rA.timings.predicted_n)"
try {
    $rB = Req $pB
    Write-Output "B ok: prompt_n=$($rB.timings.prompt_n) pred=$($rB.timings.predicted_n)"
} catch {
    Write-Output "B THREW: $($_.Exception.Message)"
}
$alive = -not $p.HasExited
Write-Output "server alive after B: $alive"
if ($alive) { Stop-Process -Id $p.Id -Force; $p.WaitForExit(10000) | Out-Null }
Write-Output '=== err tail ==='
Get-Content "$out\err.txt" -Encoding UTF8 | Select-Object -Last 12