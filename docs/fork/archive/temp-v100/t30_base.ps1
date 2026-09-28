param(
    [string]$BinDir = 'D:\LLM\Backend\llama.cpp-t24',
    [string]$Model  = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf',
    [int]$Ctx = 135168,
    [int]$N = 150,
    [int]$Rounds = 3,
    [int]$Replay = 1,
    [int]$NMax = 3,
    [int]$Port = 8510,
    [string]$Out = "$env:TEMP\v100\t30_base.jsonl"
)

$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }

$tasks = @(
    @{ id = 'd0';      file = $null;                                 },
    @{ id = 'd32768';  file = "$env:TEMP\v100\t20_prompt32k.txt"     },
    @{ id = 'd131072'; file = "$env:TEMP\v100\t20_prompt128k.txt"    }
)

$sargs = @('-m', $Model, '-np', '1', '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-ub', '512',
           '-c', "$Ctx", '--seed', '42',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', "$NMax",
           '--host', '127.0.0.1', '--port', "$Port", '-a', 't30')
$log = "$env:TEMP\v100\t30_base_$Replay.log"
Remove-Item $log, "$log.err" -ErrorAction SilentlyContinue
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$ok = $false
for ($i = 0; $i -lt 400; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 3; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) {
    "SERVER FAILED"; Get-Content "$log.err" -Tail 20; Get-Content $log -Tail 20
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; exit 1
}
"server ready (replay=$Replay n_max=$NMax ctx=$Ctx)"

function Ask($txt, $n) {
    $body = @{ prompt = $txt; n_predict = $n; cache_prompt = $true;
               temperature = 0.0; top_k = 1; top_p = 1.0; min_p = 0.0;
               seed = 42; ignore_eos = $true; stream = $false; return_tokens = $true } | ConvertTo-Json
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 3600
}

Remove-Item $Out -ErrorAction SilentlyContinue
$texts = @{}
foreach ($t in $tasks) { if ($t.file) { $texts[$t.id] = [System.IO.File]::ReadAllText($t.file, [System.Text.Encoding]::UTF8) } }

foreach ($round in 1..$Rounds) {
    $order = @()
    for ($i = 0; $i -lt $tasks.Count; $i++) { $order += $tasks[($i + $round - 1) % $tasks.Count] }
    foreach ($t in $order) {
        $txt = if ($t.file) { $texts[$t.id] } else { 'The capital of France is' }
        $vramFile = "$env:TEMP\v100\_vram_t30.txt"
        Remove-Item $vramFile -ErrorAction SilentlyContinue
        $vs = Start-Process -FilePath 'nvidia-smi' -PassThru -WindowStyle Hidden -RedirectStandardOutput $vramFile `
                -ArgumentList '-i', '1', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-lms', '500'
        $w0 = Get-Date
        $r = Ask $txt $N
        $wall = ((Get-Date) - $w0).TotalSeconds
        Stop-Process -Id $vs.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 300
        $peak = 0
        if (Test-Path $vramFile) { foreach ($ln in Get-Content $vramFile) { $v = 0; if ([int]::TryParse($ln.Trim(), [ref]$v)) { if ($v -gt $peak) { $peak = $v } } } }
        if ($peak -eq 0) { $peak = [int](& nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits) }
        $tm = $r.timings
        $acc = 0.0; if ($tm.draft_n -gt 0) { $acc = [math]::Round($tm.draft_n_accepted / $tm.draft_n, 4) }
        $rec = [ordered]@{
            round = $round; id = $t.id; replay = $Replay
            prompt_n = $tm.prompt_n; prompt_ms = [math]::Round($tm.prompt_ms, 1)
            gen_n = $tm.predicted_n; gen_ms = [math]::Round($tm.predicted_ms, 1)
            tps = [math]::Round($tm.predicted_per_second, 3)
            draft_n = $tm.draft_n; draft_acc = $tm.draft_n_accepted; accept = $acc
            vram_peak_mb = $peak; wall_s = [math]::Round($wall, 1)
        }
        Add-Content -Path $Out -Value ($rec | ConvertTo-Json -Compress) -Encoding UTF8
        $tokensFile = "$env:TEMP\v100\t30_tokens_$($t.id)_r$round`_replay$Replay.txt"
        ($r.tokens -join ',') | Set-Content -LiteralPath $tokensFile -Encoding ASCII
        "[{0}] r{1} {2,-8} prompt_n={3,-7} gen={4,-4} tps={5,-7} acc={6,-6} vram={7}MB wall={8}s" -f `
            (Get-Date -Format 'HH:mm:ss'), $round, $t.id, $tm.prompt_n, $tm.predicted_n, `
            ([math]::Round($tm.predicted_per_second, 2)), $acc, $peak, [math]::Round($wall, 1)
    }
}

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
"done -> $Out"
