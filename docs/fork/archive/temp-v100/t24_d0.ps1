param([string]$BinDir, [string]$Mode, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue

$port = 8482
$m = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'

$args = @('-m', $m, '-np', '1', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '4096', '-ctv', 'q8_0')
$args += @('--host', '127.0.0.1', '--port', "$port", '-a', 't20')
if ($Mode -ne 'base') {
    $args += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', '3')
    if ($Mode -eq 'on') { $env:GGML_CUDA_GDN_REPLAY = '1' }
}

$log = Join-Path $env:TEMP ("t24_d0_" + $Mode + ".log")
$exe = Join-Path $BinDir 'llama-server.exe'
$p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')

$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3
        if ($h.status -eq 'ok') { $ok = $true; break }
    } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) {
    "SERVER FAILED ($Mode)"
    Get-Content $log -Tail 12
    Get-Content ($log + '.err') -Tail 12
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

$prompt = 'Write a long detailed essay about the history of computing. Be verbose.'
$body = @{
    prompt       = $prompt
    n_predict    = 1000
    cache_prompt = $false
    temperature  = 0.0
    top_k        = 1
    min_p        = 0.0
    seed         = 42
    stream       = $false
    return_tokens = $true
} | ConvertTo-Json

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 3600
$sw.Stop()

($r.tokens -join ',') | Set-Content -LiteralPath $OutFile -Encoding ASCII
"$Mode : tokens=$($r.tokens.Count) seconds=$([math]::Round($sw.Elapsed.TotalSeconds,1)) tps=$([math]::Round($r.timings.predicted_per_second,2))"

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
