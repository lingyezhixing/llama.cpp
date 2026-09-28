param(
    [string]$BinDir = 'D:\LLM\Backend\llama.cpp-t30',
    [string]$Model  = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf',
    [string]$Tag    = 'A0',
    [int]$Ctx = 135168,
    [int]$N = 150,
    [int]$Port = 8520,
    [string]$Out = "$env:TEMP\v100\t30_ab.jsonl"
)

$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '1'
$env:GGML_CUDA_GDN_REPLAY = '1'

$sargs = @('-m', $Model, '-np', '1', '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-ub', '512',
           '-c', "$Ctx", '--seed', '42', '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--host', '127.0.0.1', '--port', "$Port", '-a', 't30ab')
$log = "$env:TEMP\v100\t30_ab_$Tag.log"
Remove-Item $log, "$log.err" -ErrorAction SilentlyContinue
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError "$log.err"
$ok = $false
for ($i = 0; $i -lt 400; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 3; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED"; Get-Content "$log.err" -Tail 20; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; exit 1 }

function Ask($txt, $n) {
    $body = @{ prompt = $txt; n_predict = $n; cache_prompt = $true;
               temperature = 0.0; top_k = 1; top_p = 1.0; seed = 42; ignore_eos = $true;
               stream = $false; return_tokens = $true } | ConvertTo-Json
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 3600
}

$envs = "replay=$($env:GGML_CUDA_GDN_REPLAY) vec=$($env:GGML_CUDA_FA_VEC_VERIFY) pb=$($env:GGML_CUDA_FATTN_VERIFY_PB) nbatch=$($env:GGML_CUDA_FATTN_VERIFY_NBATCH)"
"[$Tag] $envs"

$d128 = [System.IO.File]::ReadAllText("$env:TEMP\v100\t20_prompt128k.txt", [System.Text.Encoding]::UTF8)
$plan = @(
    @{ id = 'd0';      txt = 'The capital of France is' },
    @{ id = 'd131072'; txt = $d128 },
    @{ id = 'd131072'; txt = $d128 }
)

foreach ($t in $plan) {
    $w0 = Get-Date
    $r = Ask $t.txt $N
    $wall = ((Get-Date) - $w0).TotalSeconds
    $tm = $r.timings
    $rec = [ordered]@{
        tag = $Tag; vec = $env:GGML_CUDA_FA_VEC_VERIFY; pb = $env:GGML_CUDA_FATTN_VERIFY_PB; nbatch = $env:GGML_CUDA_FATTN_VERIFY_NBATCH
        id = $t.id; prompt_n = $tm.prompt_n; gen_n = $tm.predicted_n
        tps = [math]::Round($tm.predicted_per_second, 3)
        draft_n = $tm.draft_n; draft_acc = $tm.draft_n_accepted
        accept = $(if ($tm.draft_n -gt 0) { [math]::Round($tm.draft_n_accepted / $tm.draft_n, 4) } else { 0 })
        wall_s = [math]::Round($wall, 1)
    }
    Add-Content -Path $Out -Value ($rec | ConvertTo-Json -Compress) -Encoding UTF8
    ($r.tokens -join ',') | Set-Content -LiteralPath "$env:TEMP\v100\t30_ab_tok_${Tag}_$($t.id)_$(Get-Random).txt" -Encoding ASCII
    "[{0}] {1,-8} prompt_n={2,-7} gen={3,-4} tps={4,-7} acc={5,-6} draft={6}/{7} wall={8}s" -f `
        (Get-Date -Format 'HH:mm:ss'), $t.id, $tm.prompt_n, $tm.predicted_n, `
        ([math]::Round($tm.predicted_per_second, 2)), $rec.accept, $tm.draft_n_accepted, $tm.draft_n, [math]::Round($wall, 1)
}

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
"done -> $Out"
