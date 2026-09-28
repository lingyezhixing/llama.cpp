param(
    [string]$BinDir = 'D:\LLM\Backend\llama.cpp-t24',
    [string]$Model  = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf',
    [int]$NMax = 3,          # 0 = no speculative decoding
    [int]$Ctx = 135168,
    [int]$N = 150,
    [int]$Port = 8530,
    [string]$Tag = '',
    [int]$Prod = 0
)

$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '1'
$env:GGML_CUDA_GDN_REPLAY = '1'

if (-not $Tag) { $Tag = if ($NMax -gt 0) { "mtp$NMax" } else { 'nospec' } }
if ($Prod -eq 1) { $Tag = "${Tag}prod" }

$sargs = @('-m', $Model, '-np', '1', '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-ub', '512',
           '-c', "$Ctx", '--seed', '42', '-lv', '4',
           '--host', '127.0.0.1', '--port', "$Port", '-a', 't31')
if ($NMax -gt 0) {
    $sargs += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', "$NMax")
}
$log = "$env:TEMP\v100\t31_${Tag}.err"
Remove-Item $log -ErrorAction SilentlyContinue
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput "$env:TEMP\v100\t31_$Tag.log" -RedirectStandardError $log
$ok = $false
for ($i = 0; $i -lt 400; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 3; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED"; Get-Content $log -Tail 20; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; exit 1 }
"[$Tag] server ready (n_max=$NMax ctx=$Ctx)"

function Ask($txt, $n) {
    $tmpp = 0.0; $tk = 1; $tp = 1.0; $mn = 0.0
    if ($Prod -eq 1) { $tmpp = 0.6; $tk = 20; $tp = 0.95; $mn = 0.0 }
    $body = @{ prompt = $txt; n_predict = $n; cache_prompt = $true;
               temperature = $tmpp; top_k = $tk; top_p = $tp; min_p = $mn; seed = 42; ignore_eos = $true;
               stream = $false; return_tokens = $false } | ConvertTo-Json
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 3600
}

$d128 = [System.IO.File]::ReadAllText("$env:TEMP\v100\t20_prompt128k.txt", [System.Text.Encoding]::UTF8)
$out = "$env:TEMP\v100\t31_phaseA_$Tag.jsonl"
Remove-Item $out -ErrorAction SilentlyContinue
$plan = @(
    @{ id = 'd0';      txt = 'The capital of France is' },
    @{ id = 'd131072'; txt = $d128 },
    @{ id = 'd131072'; txt = $d128 }
)
$nreq = 0
foreach ($t in $plan) {
    $nreq++
    $w0 = Get-Date
    $r = Ask $t.txt $N
    $wall = ((Get-Date) - $w0).TotalSeconds
    $tm = $r.timings
    $rec = [ordered]@{
        tag = $Tag; n_max = $NMax; req = $nreq; id = $t.id
        prompt_n = $tm.prompt_n; prompt_ms = [math]::Round($tm.prompt_ms, 1)
        gen_n = $tm.predicted_n; gen_ms = [math]::Round($tm.predicted_ms, 1)
        tps = [math]::Round($tm.predicted_per_second, 3)
        draft_n = $tm.draft_n; draft_acc = $tm.draft_n_accepted
        accept = $(if ($tm.draft_n -gt 0) { [math]::Round($tm.draft_n_accepted / $tm.draft_n, 4) } else { 0 })
        wall_s = [math]::Round($wall, 1)
    }
    Add-Content -Path $out -Value ($rec | ConvertTo-Json -Compress) -Encoding UTF8
    "[{0}] req{1} {2,-8} prompt_n={3,-7} gen={4,-4} tps={5,-7} acc={6}" -f `
        (Get-Date -Format 'HH:mm:ss'), $nreq, $t.id, $tm.prompt_n, $tm.predicted_n, `
        ([math]::Round($tm.predicted_per_second, 2)), $rec.accept
}

"NOTE: server kept alive for later requests; stop it manually if needed (pid $($p.Id))"
"NOTE: SPC stats for this run are in $log (parse with t31_spc.py)"
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
"done -> $out"

