param([string]$BinDir, [int]$Replay, [int]$N = 32, [int]$Stagger = 30, [int]$Check = 0, [int]$Dflash = 0, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }
if ($Check  -eq 1) { $env:GGML_CUDA_GDN_REPLAY_CHECK = '1' }

$port = 8490
$m    = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$sargs = @('-m', $m, '-np', '2', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '8192', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')
if ($Dflash -eq 1) {
    $m = '<models>\Qwen3.6-35B-A3B-Heretic\Qwen3.6-35B-A3B-Heretic-APEX-I-Quality.gguf'
    $sargs = @('-m', $m, '-md', '<models>\Qwen3.6-35B-A3B-Heretic\Qwen3.6-35B-A3B-DFlash-Q8_0.gguf',
               '--spec-type', 'draft-dflash', '--spec-draft-n-max', '6',
               '-np', '2', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '16384', '-ctv', 'q8_0', '-ub', '1024',
               '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--min-p', '0', '--jinja',
               '--host', '127.0.0.1', '--port', "$port", '-a', 'dflash')
}
$log = Join-Path $env:TEMP "t24_conc_$Replay`_$Stagger`_$Check`_$Dflash.log"
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 2; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED (replay=$Replay)"; Get-Content "$log.err" -Tail 10; exit 1 }

Add-Type -AssemblyName System.Net.Http

$client = New-Object System.Net.Http.HttpClient
$client.Timeout = [TimeSpan]::FromMinutes(30)

$PA = 'The capital of France is'
$PB = 'List the first ten prime numbers in order:'

function MkReq($prompt, $slot, $n) {
    $body = @{ prompt = $prompt; n_predict = $n; id_slot = $slot; cache_prompt = $false; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; ignore_eos = $true; stream = $false; return_tokens = $true } | ConvertTo-Json
    return New-Object System.Net.Http.StringContent($body, [System.Text.Encoding]::UTF8, 'application/json')
}

# both requests in flight at the same time (staggered by 30 ms)
$t1 = $client.PostAsync("http://127.0.0.1:$port/completion", (MkReq $PA 0 $N))
Start-Sleep -Milliseconds $Stagger
$t2 = $client.PostAsync("http://127.0.0.1:$port/completion", (MkReq $PB 1 $N))
[System.Threading.Tasks.Task]::WaitAll(@($t1, $t2))

$r1 = $t1.Result.Content.ReadAsStringAsync().Result | ConvertFrom-Json
$r2 = $t2.Result.Content.ReadAsStringAsync().Result | ConvertFrom-Json

"== replay=$Replay concurrent =="
"  slot0: $($r1.tokens.Count) tok, draft_n=$($r1.timings.draft_n) accepted=$($r1.timings.draft_n_accepted)"
"  slot1: $($r2.tokens.Count) tok, draft_n=$($r2.timings.draft_n) accepted=$($r2.timings.draft_n_accepted)"

$lines = @()
$lines += ($r1.tokens -join ',')
$lines += ($r2.tokens -join ',')
$lines | Set-Content -LiteralPath $OutFile -Encoding ASCII

if ($Check -eq 1) { $cc = @(Select-String -Path ("$log.err") -Pattern 'fold did not reproduce' -ErrorAction SilentlyContinue).Count; "  fold check mismatch lines: $cc" }
$client.Dispose()
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
"done: replay=$Replay -> $OutFile"
