param([string]$BinDir, [int]$Replay, [int]$Check = 0, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }
if ($Check  -eq 1) { $env:GGML_CUDA_GDN_REPLAY_CHECK = '1' }

$port = 8481
$m    = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$sargs = @('-m', $m, '-np', '2', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '8192', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')
$log = Join-Path $env:TEMP "t24_np2_$Replay`_$Check.log"
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) {
    "SERVER FAILED (replay=$Replay)"
    Get-Content $log -Tail 15
    Get-Content ($log + '.err') -Tail 15
    exit 1
}

function Ask($slot, $ids, $n) {
    $body = @{ prompt = $ids; n_predict = $n; id_slot = $slot; cache_prompt = $true; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 1800
    return ,$r.tokens
}

function Tokenize($text) {
    $tb = @{ content = $text } | ConvertTo-Json
    return (Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType 'application/json').tokens
}

$idsA = Tokenize 'The capital of France is'
$idsB = Tokenize 'List the first ten prime numbers in order:'

$A = @(); $B = @(); $hA = @($idsA); $hB = @($idsB)
"== replay=$Replay =="
foreach ($round in 1..3) {
    $t = Ask 0 $hA 32; $A += $t; $hA += $t; "  slot0 r$round : $($t.Count) tok [$(($t | Select-Object -First 8) -join ',')...]"
    $t = Ask 1 $hB 32; $B += $t; $hB += $t; "  slot1 r$round : $($t.Count) tok [$(($t | Select-Object -First 8) -join ',')...]"
}

$out = @()
$out += ($A -join ',')
$out += ($B -join ',')
$out | Set-Content -LiteralPath $OutFile -Encoding ASCII

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
"done: replay=$Replay -> $OutFile"
