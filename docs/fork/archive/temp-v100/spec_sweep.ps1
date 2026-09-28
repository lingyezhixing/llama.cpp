param(
    [string]$TasksFile,
    [string]$Out = "$env:TEMP\v100\spec_out.jsonl",
    [string]$ServerDir = 'D:\LLM\Backend\llama.cpp-my',
    [string]$Model = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf',
    [int]$Ctx = 135168,
    [int]$NPredict = 150,
    [int]$Port = 8088,
    [string]$PromptDir = "$env:TEMP\v100",
    [string[]]$PromptFiles,   # for -Calibrate: file names
    [switch]$Calibrate
)

$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '1'

function Start-Srv($specType, $nMax) {
    $exe = Join-Path $ServerDir 'llama-server.exe'
    $a = @('-m', $Model, '-np', '1', '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-c', "$Ctx",
           '--host', '127.0.0.1', '--port', "$Port", '-a', 'q38')
    if ($specType) { $a += @('--spec-type', $specType) }
    if ($nMax -gt 0) { $a += @('--spec-draft-n-max', "$nMax") }
    $tag = "srv"
    $so = "$env:TEMP\v100\srv_${tag}_out.log"
    $se = "$env:TEMP\v100\srv_${tag}_err.log"
    $p = Start-Process -FilePath $exe -ArgumentList $a -WorkingDirectory $ServerDir -PassThru `
            -RedirectStandardOutput $so -RedirectStandardError $se -WindowStyle Hidden
    $t0 = Get-Date
    while ($true) {
        Start-Sleep -Milliseconds 1500
        if ($p.HasExited) {
            Write-Host "server exited early (code $($p.ExitCode)); tail of stderr:"
            Get-Content $se -Tail 20 | ForEach-Object { "  $_" }
            throw "server start failed"
        }
        try {
            $h = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
            if ($h.StatusCode -eq 200) { break }
        } catch { }
        if (((Get-Date) - $t0).TotalSeconds -gt 420) { throw "server health timeout" }
    }
    "server ready in $([math]::Round(((Get-Date)-$t0).TotalSeconds,1))s (spec='$specType' n_max=$nMax)"
    return $p
}

function Stop-Srv($p) {
    if ($p -and -not $p.HasExited) {
        Stop-Process -Id $p.Id -Force
        Start-Sleep -Seconds 2
    }
}

function Req($promptText, $cache, $nPred) {
    $body = @{
        prompt        = $promptText
        n_predict     = $nPred
        temperature   = 0
        top_k         = 1
        seed          = 42
        cache_prompt  = [bool]$cache
        ignore_eos    = $true
        return_tokens = $true
        stream        = $false
    } | ConvertTo-Json -Depth 4
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post `
        -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType "application/json" -TimeoutSec 3600
}

if ($Calibrate) {
    $srv = Start-Srv '' 0
    foreach ($f in $PromptFiles) {
        $txt = [System.IO.File]::ReadAllText((Join-Path $PromptDir $f), [System.Text.Encoding]::UTF8)
        $body = @{ content = $txt } | ConvertTo-Json -Depth 3
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/tokenize" -Method Post `
                -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType "application/json" -TimeoutSec 600
        $toks = @($r.tokens).Count
        "{0,-24} chars={1,-8} tokens={2}" -f $f, $txt.Length, $toks
    }
    Stop-Srv $srv
    exit 0
}

$tasks = Get-Content $TasksFile -Encoding UTF8 -Raw | ConvertFrom-Json
$prevTag = $null
$srv = $null

foreach ($t in $tasks) {
    $tag = $t.tag
    if ($tag -ne $prevTag) {
        Stop-Srv $srv
        $srv = Start-Srv $t.spec_type $t.n_max
        $prevTag = $tag
    }
    $txt = [System.IO.File]::ReadAllText((Join-Path $PromptDir $t.prompt), [System.Text.Encoding]::UTF8)

    $vramFile = "$env:TEMP\v100\_vram_sample.txt"
    Remove-Item $vramFile -ErrorAction SilentlyContinue
    $vs = Start-Process -FilePath 'nvidia-smi' -PassThru -WindowStyle Hidden -RedirectStandardOutput $vramFile `
            -ArgumentList '-i', '1', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-lms', '500'

    $w0 = Get-Date
    $resp = Req $txt $t.cache $NPredict
    $wall = ((Get-Date) - $w0).TotalSeconds

    Stop-Process -Id $vs.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 300
    $peak = 0
    if (Test-Path $vramFile) {
        foreach ($ln in Get-Content $vramFile) { $v = 0; if ([int]::TryParse($ln.Trim(), [ref]$v)) { if ($v -gt $peak) { $peak = $v } } }
    }
    if ($peak -eq 0) {
        $peak = [int](& nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)
    }

    $tm = $resp.timings
    $acc = 0.0
    if ($tm.draft_n -gt 0) { $acc = [math]::Round($tm.draft_n_accepted / $tm.draft_n, 3) }
    $rec = [ordered]@{
        tag = $tag; spec_type = $t.spec_type; n_max = $t.n_max; prompt = $t.prompt; label = $t.label
        cache = [bool]$t.cache; prompt_n = $tm.prompt_n; gen_n = $tm.predicted_n
        tps = [math]::Round($tm.predicted_per_second, 2); prompt_ms = [math]::Round($tm.prompt_ms, 1)
        draft_n = $tm.draft_n; draft_acc = $tm.draft_n_accepted; accept = $acc
        vram_peak_mb = $peak; wall_s = [math]::Round($wall, 1)
        tokens = ($resp.tokens -join ' ')
    }
    $line = ($rec | ConvertTo-Json -Compress)
    Add-Content -Path $Out -Value $line -Encoding UTF8
    "[{0}] {1,-22} {2,-9} prompt_n={3,-7} gen={4,-4} tps={5,-7} acc={6,-6} vram={7}MB wall={8}s" -f `
        (Get-Date -Format 'HH:mm:ss'), $tag, $t.label, $tm.prompt_n, $tm.predicted_n, ([math]::Round($tm.predicted_per_second,2)), $acc, $peak, [math]::Round($wall,1)
}

Stop-Srv $srv
"done -> $Out"
