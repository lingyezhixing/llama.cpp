param(
    [string]$QuestionsFile = "$env:TEMP\v100\t20_qtest.json",
    [string]$OutDir = "$env:TEMP\v100\t20_quality",
    [int]$MaxTokens = 131072,
    [int]$PortBase = 8600,
    [int]$Ctx = 65536,
    [switch]$SkipExisting
)
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "1"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$questions = Get-Content -LiteralPath $QuestionsFile -Raw -Encoding UTF8 | ConvertFrom-Json

$arms = @(
    @{ id = "stock-nomtp"; dir = "D:\LLM\Backend\llama.cpp";     env = @(); spec = $false },
    @{ id = "stock-mtp";   dir = "D:\LLM\Backend\llama.cpp";     env = @(); spec = $true  },
    @{ id = "fix-nomtp";   dir = "D:\LLM\Backend\llama.cpp-t20"; env = @("GGML_CUDA_FA_SMALL_BATCH_VEC=1", "GGML_CUDA_GDN_VEC4=1"); spec = $false },
    @{ id = "fix-mtp";     dir = "D:\LLM\Backend\llama.cpp-t20"; env = @("GGML_CUDA_FA_SMALL_BATCH_VEC=1", "GGML_CUDA_GDN_VEC4=1"); spec = $true  }
)

$port = $PortBase
foreach ($a in $arms) {
    $port++
    foreach ($kv in @("GGML_CUDA_FA_SMALL_BATCH_VEC", "GGML_CUDA_GDN_VEC4", "GGML_CUDA_FATTN_PB_FORCE", "GGML_CUDA_FA_FORCE")) {
        [Environment]::SetEnvironmentVariable($kv, $null)
    }
    foreach ($e in $a.env) { $parts = $e.Split("="); [Environment]::SetEnvironmentVariable($parts[0], $parts[1]) }

    $sargs = @(
        "-m", $m, "-dev", "cuda0", "-np", "1", "-ngl", "99", "-fa", "on", "--seed", "42",
        "-c", "$Ctx", "-ctv", "q8_0", "--jinja", "--host", "127.0.0.1", "--port", "$port", "-a", "qtest"
    )
    if ($a.spec) { $sargs += @("--spec-type", "draft-mtp", "--spec-draft-n-max", "3") }
    $exe = "$($a.dir)\llama-server.exe"
    $log = "$OutDir\srv_$($a.id).log"
    $errlog = "$OutDir\srv_$($a.id)_err.log"
    $p = Start-Process -FilePath $exe -ArgumentList $sargs -RedirectStandardOutput $log -RedirectStandardError $errlog -PassThru -WindowStyle Hidden
    $ok = $false
    for ($i = 0; $i -lt 300; $i++) {
        Start-Sleep -Seconds 2
        try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3 -UseBasicParsing; if ($r.Content -match "ok") { $ok = $true; break } } catch {}
        if ($p.HasExited) { break }
    }
    if (-not $ok) { Write-Output "$($a.id): SERVER FAILED"; if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; continue }
    Write-Output "=== $($a.id) server up (port $port) ==="

    $stats = @{}
    if ($SkipExisting) {
        $spath = "$OutDir\stats_$($a.id).json"
        if (Test-Path -LiteralPath $spath) {
            $prev = Get-Content -LiteralPath $spath -Raw -Encoding UTF8 | ConvertFrom-Json
            foreach ($p in $prev.PSObject.Properties) { $stats[$p.Name] = $p.Value }
        }
    }
    foreach ($q in $questions) {
        $rawPath = "$OutDir\raw_$($a.id)_$($q.id).json"
        if ($SkipExisting -and $stats.ContainsKey($q.id) -and (Test-Path -LiteralPath $rawPath)) {
            Write-Output ("{0}/{1}: skip (already done)" -f $a.id, $q.id)
            continue
        }
        $body = @{
            messages = @( @{ role = "user"; content = $q.question } )
            temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42
            max_tokens = $MaxTokens; stream = $false
        } | ConvertTo-Json -Depth 6
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType "application/json" -TimeoutSec 43200
        } catch {
            Write-Output "$($a.id)/$($q.id): REQUEST FAILED: $_"
            continue
        }
        $sw.Stop()
        $raw = $resp | ConvertTo-Json -Depth 12
        Set-Content -LiteralPath "$OutDir\raw_$($a.id)_$($q.id).json" -Value $raw -Encoding UTF8
        $msg = $resp.choices[0].message
        $content = $msg.content
        $reason = $null
        if ($msg.PSObject.Properties.Name -contains "reasoning_content") { $reason = $msg.reasoning_content }
        $rtok = 0; $ctok = 0
        if (-not [string]::IsNullOrEmpty($reason)) {
            try {
                $tb = @{ content = $reason; add_special = $false } | ConvertTo-Json -Depth 3
                $t = Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType "application/json" -TimeoutSec 300
                $rtok = @($t.tokens).Count
            } catch { $rtok = -1 }
        }
        if (-not [string]::IsNullOrEmpty($content)) {
            try {
                $tb = @{ content = $content; add_special = $false } | ConvertTo-Json -Depth 3
                $t = Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType "application/json" -TimeoutSec 300
                $ctok = @($t.tokens).Count
            } catch { $ctok = -1 }
        }
        $finish = $resp.choices[0].finish_reason
        $tps = $resp.timings.predicted_per_second
        $n = $resp.usage.completion_tokens
        $stats[$q.id] = @{ completion_tokens = $n; reasoning_tokens = $rtok; answer_tokens = $ctok; finish_reason = $finish; tps = [math]::Round($tps, 2); wall_s = [math]::Round($sw.Elapsed.TotalSeconds, 1) }
        Write-Output ("{0}/{1}: tokens={2} reason={3} answer={4} finish={5} tps={6:N2} wall={7:N1}s" -f $a.id, $q.id, $n, $rtok, $ctok, $finish, $tps, $sw.Elapsed.TotalSeconds)
    }
    $stats | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$OutDir\stats_$($a.id).json" -Encoding UTF8
    Stop-Process -Id $p.Id -Force
    Start-Sleep -Seconds 2
}
foreach ($kv in @("GGML_CUDA_FA_SMALL_BATCH_VEC", "GGML_CUDA_GDN_VEC4")) { [Environment]::SetEnvironmentVariable($kv, $null) }
Write-Output "QUALITY RUN DONE"
