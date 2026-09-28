param([string]$BinDir, [int]$Replay, [int]$Max = 4500, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }

$port = 8486
$m    = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$sargs = @('-m', $m, '-np', '1', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '4096', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')
$log = Join-Path $env:TEMP "t24_long_$Replay.log"
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED (replay=$Replay)"; Get-Content $log -Tail 10; exit 1 }

$body = @{
    prompt = 'Write a long detailed essay about the history of computing. Be verbose.'
    n_predict = $Max; temperature = 0; seed = 42; ignore_eos = $true; return_tokens = $true; cache_prompt = $false
} | ConvertTo-Json
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 3600
    ($r.tokens -join ',') | Set-Content -LiteralPath $OutFile -Encoding ASCII
    "replay=$Replay tokens=$($r.tokens.Count) stop_type=$($r.stop_type) truncated=$($r.truncated)"
} catch {
    "replay=$Replay REQUEST ERROR: $($_.Exception.Message)"
    if (Test-Path $OutFile) { } 
}

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
