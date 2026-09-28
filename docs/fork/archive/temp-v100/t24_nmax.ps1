param([string]$BinDir, [int]$Replay, [int]$Nmax = 2, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }

$port = 8485
$m    = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$sargs = @('-m', $m, '-np', '1', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '8192', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', "$Nmax",
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')
$log = Join-Path $env:TEMP "t24_nmax$Nmax`_$Replay.log"
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED (nmax=$Nmax replay=$Replay)"; Get-Content $log -Tail 10; exit 1 }

$body = @{
    prompt = 'The quick brown fox jumps over the lazy dog. Count from 1 to 40 slowly.'
    n_predict = 64; temperature = 0; seed = 42; ignore_eos = $true; return_tokens = $true; cache_prompt = $false
} | ConvertTo-Json
$r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 900
($r.tokens -join ',') | Set-Content -LiteralPath $OutFile -Encoding ASCII
"nmax=$Nmax replay=$Replay tokens=$($r.tokens.Count) draft_n=$($r.timings.draft_n) accepted=$($r.timings.draft_n_accepted)"

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
