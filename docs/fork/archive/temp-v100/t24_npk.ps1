param([string]$BinDir, [int]$Replay, [int]$Np = 4, [string]$Slots = '0,1,2,3', [int]$Rounds = 1, [int]$N = 16, [int]$Cache = 0, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }

$port = 8492
$m    = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$sargs = @('-m', $m, '-np', "$Np", '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '8192', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')
$log = Join-Path $env:TEMP ("t24_npk_" + $Replay + "_" + $Np + "_" + ($Slots -replace ',','_') + ".log")
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED (np=$Np slots=$Slots)"; Get-Content "$log.err" -Tail 8; exit 1 }

function Ask($slot, $ids, $n) {
    $body = @{ prompt = $ids; n_predict = $n; id_slot = $slot; cache_prompt = ($Cache -eq 1); temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; ignore_eos = $true; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 1800
    return ,$r.tokens
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
$hist = @(); $out = @()
for ($i = 0; $i -lt 4; $i++) { $hist += ,@(); $out += ,@(); if ($use -contains $i) { $hist[$i] = @(Tokenize $P[$i]) } }

"== np=$Np slots=$Slots replay=$Replay rounds=$Rounds cache=$Cache =="
foreach ($round in 1..$Rounds) {
    for ($i = 0; $i -lt 4; $i++) {
        if (-not ($use -contains $i)) { continue }
        $t = Ask $i $hist[$i] $N
        $out[$i] += $t
        $hist[$i] += $t
        "  slot$i r$round : $($t.Count) tok"
    }
}

$lines = @()
for ($i = 0; $i -lt 4; $i++) { $lines += ("s$i," + ($out[$i] -join ',')) }
$lines | Set-Content -LiteralPath $OutFile -Encoding ASCII

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
"done -> $OutFile"
