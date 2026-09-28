param([string]$BinDir, [int]$Replay, [string]$Mode = 'prefill0', [int]$Ctx = 16384, [int]$Long = 1, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }

$port = 8498
$m    = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$sargs = @('-m', $m, '-np', '1', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', "$Ctx", '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')
$log = Join-Path $env:TEMP "t24_edge_$Replay`_$Mode.log"
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED"; exit 1 }

function Ask($ids, $n, $cache) {
    $body = @{ prompt = $ids; n_predict = $n; cache_prompt = $cache; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; ignore_eos = $true; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 1800
    return $r
}
function Health() {
    Start-Sleep -Seconds 1
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; return $h.status } catch { return 'DEAD' }
}
function Tokenize($text) {
    $tb = @{ content = $text } | ConvertTo-Json
    return (Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType 'application/json').tokens
}

$res = @()
if ($Mode -eq 'prefill0') {
    $P = if ($Long -eq 1) { ('The quick brown fox jumps over the lazy dog. ' * 500) } else { 'The capital of France is' }
    $ids = Tokenize $P
    "== replay=$Replay mode=prefill0 (prompt=$($ids.Count) tok) =="
    try { $a1 = Ask $ids 0 $true; "  A1 n_predict=0   : prompt_n=$($a1.timings.prompt_n) pred=$($a1.timings.predicted_n)" } catch { "  A1 EXCEPTION: $($_.Exception.Message)" }
    "  health after A1: $(Health)"
    try { $a2 = Ask $ids 8 $true; "  A2 same prompt   : prompt_n=$($a2.timings.prompt_n) pred=$($a2.timings.predicted_n) tok=$($a2.tokens -join ',')"; $res += ('A2,' + ($a2.tokens -join ',')) } catch { "  A2 EXCEPTION: $($_.Exception.Message)"; $res += 'A2,EXCEPTION' }
    "  health after A2: $(Health)"
} else {
    $P = 'The capital of France is'
    $ids = Tokenize $P
    "== replay=$Replay mode=retry (prompt=$($ids.Count) tok) =="
    $b1 = Ask $ids 64 $true
    $T = @($b1.tokens)
    "  B1 gen=$($T.Count) tok, prompt_n=$($b1.timings.prompt_n)"
    $res += ('B1,' + ($T -join ','))
    # retry: drop the last 60 generated tokens
    $short = @($ids) + @($T[0..($T.Count-61)])
    try { $b2 = Ask $short 8 $true; "  B2 retry(-60)    : prompt_n=$($b2.timings.prompt_n) pred=$($b2.timings.predicted_n) tok=$($b2.tokens -join ',')"; $res += ('B2,' + ($b2.tokens -join ',')) } catch { "  B2 EXCEPTION: $($_.Exception.Message)"; $res += 'B2,EXCEPTION' }
    "  health after B2: $(Health)"
    # retry: drop the whole reply
    try { $b3 = Ask $ids 8 $true; "  B3 retry(all)    : prompt_n=$($b3.timings.prompt_n) pred=$($b3.timings.predicted_n) tok=$($b3.tokens -join ',')"; $res += ('B3,' + ($b3.tokens -join ',')) } catch { "  B3 EXCEPTION: $($_.Exception.Message)"; $res += 'B3,EXCEPTION' }
    "  health after B3: $(Health)"
}
$res | Set-Content -LiteralPath $OutFile -Encoding ASCII

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
"done -> $OutFile"
