param([string]$BinDir, [int]$Replay, [int]$Check = 0, [int]$Np = 2, [string]$Slots = '0,1',
      [int]$Rounds = 2, [int]$N = 200, [string]$Mode = 'greedy', [int]$Ctx = 16384, [int]$Cache = 1, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }
if ($Check  -eq 1) { $env:GGML_CUDA_GDN_REPLAY_CHECK = '1' }

$port  = 8496
$tgt   = '<models>\Qwen3.6-35B-A3B-Heretic\Qwen3.6-35B-A3B-Heretic-APEX-I-Quality.gguf'
$draft = '<models>\Qwen3.6-35B-A3B-Heretic\Qwen3.6-35B-A3B-DFlash-Q8_0.gguf'

$sargs = @('-m', $tgt, '-md', $draft,
           '--spec-type', 'draft-dflash', '--spec-draft-n-max', '6',
           '-np', "$Np", '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', "$Ctx", '-ctv', 'q8_0', '-ub', '1024',
           '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0',
           '--presence-penalty', '0', '--repeat-penalty', '1.0', '--jinja',
           '--host', '127.0.0.1', '--port', "$port", '-a', 'dflash')
$log = Join-Path $env:TEMP "t24_dflash_np_$Replay`_$Check`_$Np`_$($Slots -replace ',','_')`_$Mode.log"
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 400; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED (replay=$Replay np=$Np slots=$Slots)"; Get-Content "$log.err" -Tail 12; exit 1 }

function Ask($slot, $ids, $n) {
    $body = @{ prompt = $ids; n_predict = $n; id_slot = $slot; cache_prompt = ($Cache -eq 1);
               temperature = ($(if ($Mode -eq 'greedy') { 0.0 } else { 0.6 })); top_k = ($(if ($Mode -eq 'greedy') { 1 } else { 20 })); top_p = 0.95; min_p = 0.0;
               seed = 42; ignore_eos = $true; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 3600
    return $r
}
function Tokenize($text) {
    $tb = @{ content = $text } | ConvertTo-Json
    return (Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType 'application/json').tokens
}

$P = @(
    'The capital of France is',
    'List the first ten prime numbers in order:',
    'Explain in one sentence why the sky is blue.',
    'Compute 17 * 23 step by step.'
)

$use = $Slots.Split(',') | ForEach-Object { [int]$_ }
$hist = @(); $out = @(); $stats = @()
for ($i = 0; $i -lt 4; $i++) { $hist += ,@(); $out += ,@(); $stats += ,@(); if ($use -contains $i) { $hist[$i] = @(Tokenize $P[$i]) } }

"== dflash np=$Np slots=$Slots replay=$Replay mode=$Mode N=$N rounds=$Rounds =="
foreach ($round in 1..$Rounds) {
    for ($i = 0; $i -lt 4; $i++) {
        if (-not ($use -contains $i)) { continue }
        $r = Ask $i $hist[$i] $N
        $out[$i] += $r.tokens
        $hist[$i] += $r.tokens
        $stats[$i] += "r${round}:$($r.timings.draft_n)/$($r.timings.draft_n_accepted)"
        "  slot$i r$round : $($r.tokens.Count) tok, draft=$($r.timings.draft_n) acc=$($r.timings.draft_n_accepted)"
    }
}

$lines = @()
for ($i = 0; $i -lt 4; $i++) { $lines += ("s$i," + ($out[$i] -join ',')) }
for ($i = 0; $i -lt 4; $i++) { $lines += ("st$i," + ($stats[$i] -join ';')) }
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
