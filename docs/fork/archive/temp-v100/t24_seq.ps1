param([string]$BinDir, [int]$Replay, [int]$Check = 0, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }
if ($Check  -eq 1) { $env:GGML_CUDA_GDN_REPLAY_CHECK = '1' }

$port = 8482
$m    = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$sargs = @('-m', $m, '-np', '1', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '8192', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')
$log = Join-Path $env:TEMP "t24_seq_$Replay`_$Check.log"
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) {
    "SERVER FAILED (replay=$Replay check=$Check)"
    Get-Content $log -Tail 15
    Get-Content ($log + '.err') -Tail 15
    exit 1
}

function Ask($ids, $n) {
    $body = @{ prompt = $ids; n_predict = $n; cache_prompt = $true; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 1800
    return ,$r.tokens
}
function Tokenize($text) {
    $tb = @{ content = $text } | ConvertTo-Json
    return (Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType 'application/json').tokens
}

$P1 = 'The capital of France is'
$P2 = 'What are the main differences between TCP and UDP? Answer briefly:'
$ids1 = Tokenize $P1
$ids2 = Tokenize $P2
$X    = @((Tokenize ' Berlin')[0])

$res = @()
"== replay=$Replay check=$Check =="
# R1: fresh prefill + 4 tokens
$T1 = Ask $ids1 4; $res += ,$T1; "  R1 : $($T1 -join ',')"
# R2: continuation (0 trim)
$T2 = Ask (@($ids1) + @($T1)) 4; $res += ,$T2; "  R2 : $($T2 -join ',')"
# R3: replace the last 2 generated tokens of R2 by X (trim = 2, partial rollback)
$short = @($ids1) + @($T1) + @($T2[0..1]) + $X
$T3 = Ask $short 4; $res += ,$T3; "  R3 : $($T3 -join ',')"
# R4: continuation after the partial rollback
$T4 = Ask (@($short) + @($T3)) 4; $res += ,$T4; "  R4 : $($T4 -join ',')"
# R5: unrelated prompt (full reset)
$T5 = Ask $ids2 4; $res += ,$T5; "  R5 : $($T5 -join ',')"
# R6: continuation of the unrelated prompt
$T6 = Ask (@($ids2) + @($T5)) 4; $res += ,$T6; "  R6 : $($T6 -join ',')"

$lines = @()
foreach ($t in $res) { $lines += ($t -join ',') }
$lines | Set-Content -LiteralPath $OutFile -Encoding ASCII

if ($Check -eq 1) {
    $chk = Select-String -Path ($log + '.err') -Pattern 'fold did not reproduce' -ErrorAction SilentlyContinue
    "  fold check errors: $($chk.Count)"
}
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
"done: replay=$Replay -> $OutFile"
