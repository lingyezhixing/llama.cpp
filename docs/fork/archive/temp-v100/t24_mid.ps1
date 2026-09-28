param([string]$BinDir, [int]$Replay, [int]$Cut = 6, [int]$Gen = 16, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }

$port = 8488
$m    = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$sargs = @('-m', $m, '-np', '1', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '8192', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')
$log = Join-Path $env:TEMP "t24_mid_$Replay`_$Cut.log"
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED (replay=$Replay)"; Get-Content $log -Tail 10; exit 1 }

function Ask($ids, $n) {
    $body = @{ prompt = $ids; n_predict = $n; cache_prompt = $true; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 600
    return ,$r.tokens
}
function Tokenize($text) {
    $tb = @{ content = $text } | ConvertTo-Json
    return (Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType 'application/json').tokens
}

$ids = Tokenize 'The capital of France is'
"== replay=$Replay cut=$Cut gen=$Gen =="
$T1 = Ask $ids $Gen
"  R1 : $($T1.Count) tok"
$X = @((Tokenize ' zebra')[0])
$new = @($ids) + @($T1[0..($Cut-1)]) + $X
"  new prompt: ids=$($ids.Count) + T1[0..$($Cut-1)] + X -> $($new.Count) tok (trim = $($Gen - $Cut) tokens)"
try {
    $T2 = Ask $new 8
    "  R2 : OK, $($T2.Count) tok : $($T2 -join ',')"
    $state = 'ok'
} catch {
    $state = "ERROR: $($_.Exception.Message)"
    "  R2 : FAILED - $state"
}
Start-Sleep -Seconds 2
try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; $alive = "$($h.status)"; if ($h.status -ne 'ok') { $alive = "$($h.status): $($h.error)" } } catch { $alive = 'DEAD' }
"  server after: $alive"
"$Replay,$Cut,$alive,$state" | Set-Content -LiteralPath $OutFile -Encoding ASCII

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
