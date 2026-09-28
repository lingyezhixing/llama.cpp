param([string]$BinDir, [int]$Replay, [int]$Np = 1, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }

$port = 8491
$m    = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$sargs = @('-m', $m, '-np', "$Np", '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '8192', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')
$log = Join-Path $env:TEMP "t24_fresh_$Replay`_$Np.log"
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED"; exit 1 }

function Ask($slot, $prompt, $n, $cache) {
    $body = @{ prompt = $prompt; n_predict = $n; id_slot = $slot; cache_prompt = $cache; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 1800
    return ,$r.tokens
}

$P = 'The capital of France is'
"== replay=$Replay np=$Np =="
$A = Ask 0 $P 8 $false
$B = Ask 0 $P 8 $false           # same prompt again, no cache: full clear + fresh prefill
$C = Ask 1 $P 8 $false           # another slot (np > 1): a fresh sequence reusing a freed cell
"  A: $($A -join ',')"
"  B: $($B -join ',')"
"  C: $($C -join ',')"
"  A==B: $(($A -join ',') -eq ($B -join ','))   A==C: $(($A -join ',') -eq ($C -join ','))"

$lines = @()
$lines += 'A,' + ($A -join ',')
$lines += 'B,' + ($B -join ',')
$lines += 'C,' + ($C -join ',')
$lines | Set-Content -LiteralPath $OutFile -Encoding ASCII

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
