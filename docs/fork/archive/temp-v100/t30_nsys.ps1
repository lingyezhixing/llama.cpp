param(
    [string]$BinDir = 'D:\LLM\Backend\llama.cpp-t24',
    [string]$Model  = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf',
    [int]$Ctx = 135168,
    [int]$N = 150,
    [int]$Port = 8512,
    [int]$Duration = 460,
    [string]$Tag = 'd128k'
)

$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '1'
$env:GGML_CUDA_GDN_REPLAY = '1'
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe'
$rep  = "$env:TEMP\v100\t30_$Tag"
$log  = "$env:TEMP\v100\t30_${Tag}_srv.log"

$sargs = @('-m', $Model, '-np', '1', '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-ub', '512',
           '-c', "$Ctx", '--seed', '42', '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--host', '127.0.0.1', '--port', "$Port", '-a', 't30nsys')
$nargs = @('profile', '--trace=cuda', '--sample=none', '--cpuctxsw=none',
           '--duration', "$Duration", '-o', $rep, '--force-overwrite', 'true',
           (Join-Path $BinDir 'llama-server.exe')) + $sargs

$t0 = Get-Date
$p = Start-Process -FilePath $nsys -ArgumentList $nargs -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput ($log + '.nsys') -RedirectStandardError ($log + '.err')
"nsys pid $($p.Id), server under profile, waiting for health..."

$ok = $false
for ($i = 0; $i -lt 400; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED"; Get-Content ($log + '.err') -Tail 20; exit 1 }
$ready = ((Get-Date) - $t0).TotalSeconds
"server ready after $([math]::Round($ready,1))s"

$txt = [System.IO.File]::ReadAllText("$env:TEMP\v100\t20_prompt128k.txt", [System.Text.Encoding]::UTF8)
function Ask($promptTxt, $n) {
    $body = @{ prompt = $promptTxt; n_predict = $n; cache_prompt = $true;
               temperature = 0.0; top_k = 1; seed = 42; ignore_eos = $true;
               stream = $false; return_tokens = $true } | ConvertTo-Json
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 3600
}

$out = "$env:TEMP\v100\t30_$Tag.jsonl"
Remove-Item $out -ErrorAction SilentlyContinue
foreach ($round in 1..2) {
    $w0 = Get-Date
    $r = Ask $txt $N
    $w = ((Get-Date) - $w0).TotalSeconds
    $tm = $r.timings
    $rec = [ordered]@{ round = $round; prompt_n = $tm.prompt_n; prompt_ms = [math]::Round($tm.prompt_ms,1);
                       gen_n = $tm.predicted_n; gen_ms = [math]::Round($tm.predicted_ms,1);
                       tps = [math]::Round($tm.predicted_per_second,3); draft_n = $tm.draft_n;
                       draft_acc = $tm.draft_n_accepted; wall_s = [math]::Round($w,1);
                       since_start_s = [math]::Round(((Get-Date) - $t0).TotalSeconds,1) }
    Add-Content -Path $out -Value ($rec | ConvertTo-Json -Compress) -Encoding UTF8
    "round $round : prompt_n=$($tm.prompt_n) gen=$($tm.predicted_n) tps=$([math]::Round($tm.predicted_per_second,2)) wall=$([math]::Round($w,1))s since_start=$([math]::Round(((Get-Date) - $t0).TotalSeconds,1))s"
}

"waiting for nsys to finish capture (duration ${Duration}s)..."
$deadline = $t0.AddSeconds($Duration + 180)
while ((Get-Date) -lt $deadline) {
    if (Test-Path "$rep.nsys-rep") { break }
    Start-Sleep -Seconds 5
}
Start-Sleep -Seconds 5
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
"report: $(if (Test-Path "$rep.nsys-rep") { (Get-Item "$rep.nsys-rep").Length } else { 'MISSING' }) bytes"
Get-Content ($log + '.err') -Tail 5 | ForEach-Object { "nsys: $_" }
"done"
