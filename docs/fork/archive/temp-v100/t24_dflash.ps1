param([string]$BinDir, [int]$Replay, [int]$Check = 0, [string]$Mode = 'prod', [int]$N = 128, [int]$Rounds = 1, [int]$Ctx = 16384, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }
if ($Check  -eq 1) { $env:GGML_CUDA_GDN_REPLAY_CHECK = '1' }

$port  = 8494
$tgt   = '<models>\Qwen3.6-35B-A3B-Heretic\Qwen3.6-35B-A3B-Heretic-APEX-I-Quality.gguf'
$draft = '<models>\Qwen3.6-35B-A3B-Heretic\Qwen3.6-35B-A3B-DFlash-Q8_0.gguf'

$sargs = @('-m', $tgt, '-md', $draft,
           '--spec-type', 'draft-dflash', '--spec-draft-n-max', '6',
           '-np', '1', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', "$Ctx", '-ctv', 'q8_0', '-ub', '1024',
           '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0',
           '--presence-penalty', '0', '--repeat-penalty', '1.0', '--jinja',
           '--host', '127.0.0.1', '--port', "$port", '-a', 'dflash')
$log = Join-Path $env:TEMP "t24_dflash_$Replay`_$Check`_$Mode.log"
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 400; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED (replay=$Replay mode=$Mode)"; Get-Content "$log.err" -Tail 15; Get-Content $log -Tail 15; exit 1 }

function Ask($ids, $n) {
    $body = @{ prompt = $ids; n_predict = $n; cache_prompt = $true;
               temperature = ($(if ($Mode -eq 'greedy') { 0.0 } else { 0.6 })); top_k = ($(if ($Mode -eq 'greedy') { 1 } else { 20 })); top_p = 0.95; min_p = 0.0;
               seed = 42; ignore_eos = $true; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 3600
    return $r
}
function Tokenize($text) {
    $tb = @{ content = $text } | ConvertTo-Json
    return (Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType 'application/json').tokens
}

$P = 'The capital of France is'

"== dflash replay=$Replay mode=$Mode N=$N rounds=$Rounds =="
$ids  = Tokenize $P
$hist = @($ids)
$out  = @()
$stats = @()
foreach ($round in 1..$Rounds) {
    $r = Ask $hist $N
    $out += $r.tokens
    $hist += $r.tokens
    $stats += "r${round}:draft=$($r.timings.draft_n)/acc=$($r.timings.draft_n_accepted)/eval=$($r.timings.predicted_n)/tps=$([math]::Round($r.timings.predicted_per_second,2))"
    "  round $round : $($r.tokens.Count) tok, draft_n=$($r.timings.draft_n) accepted=$($r.timings.draft_n_accepted) tps=$([math]::Round($r.timings.predicted_per_second,2))"
}

$lines = @()
$lines += ("tokens," + ($out -join ','))
$lines += ("stats," + ($stats -join ';'))
$lines | Set-Content -LiteralPath $OutFile -Encoding ASCII

if ($Check -eq 1) {
    $c = @(Select-String -Path "$log.err" -Pattern 'fold did not reproduce' -ErrorAction SilentlyContinue).Count
    "  fold check mismatch lines: $c"
}
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
"done -> $OutFile"
