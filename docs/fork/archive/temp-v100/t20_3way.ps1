param([int]$NPredict = 250, [int]$PortBase = 8520)
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "1"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$short = "Write a long detailed essay about the history of computing. Be verbose."
$long  = [System.IO.File]::ReadAllText("$env:TEMP\v100\t20_prompt8k.txt")

$arms = @(
    @{ id = "stock";  dir = "D:\LLM\Backend\llama.cpp";        env = @() },
    @{ id = "t19";    dir = "D:\LLM\Backend\llama.cpp-my";     env = @() },
    @{ id = "t20";    dir = "D:\LLM\Backend\llama.cpp-t20";    env = @() },
    @{ id = "t20fix"; dir = "D:\LLM\Backend\llama.cpp-t20";    env = @("GGML_CUDA_FA_SMALL_BATCH_VEC=1", "GGML_CUDA_GDN_VEC4=1") }
)
$depths = @(
    @{ id = "d0";    prompt = $short; ctx = 4096 },
    @{ id = "d8192"; prompt = $long;  ctx = 12288 }
)

$port = $PortBase
foreach ($dp in $depths) {
    foreach ($a in $arms) {
        $port++
        $tag = "3w_$($a.id)_$($dp.id)"
        foreach ($kv in @("GGML_CUDA_FA_SMALL_BATCH_VEC", "GGML_CUDA_GDN_VEC4", "GGML_CUDA_FATTN_PB_FORCE", "GGML_CUDA_FA_FORCE")) {
            [Environment]::SetEnvironmentVariable($kv, $null)
        }
        foreach ($e in $a.env) { $parts = $e.Split("="); [Environment]::SetEnvironmentVariable($parts[0], $parts[1]) }
        $exe = "$($a.dir)\llama-server.exe"
        $sargs = @(
            "-m", $m, "-dev", "cuda0", "-np", "1", "-ngl", "99", "-fa", "on", "--seed", "42",
            "-c", "$($dp.ctx)", "-ctv", "q8_0", "--host", "127.0.0.1", "--port", "$port", "-a", "t20"
        )
        $log = "$env:TEMP\v100\$tag.log"
        $errlog = "$env:TEMP\v100\${tag}_err.log"
        $p = Start-Process -FilePath $exe -ArgumentList $sargs -RedirectStandardOutput $log -RedirectStandardError $errlog -PassThru -WindowStyle Hidden
        $ok = $false
        for ($i = 0; $i -lt 300; $i++) {
            Start-Sleep -Seconds 2
            try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3 -UseBasicParsing; if ($r.Content -match "ok") { $ok = $true; break } } catch {}
            if ($p.HasExited) { break }
        }
        if (-not $ok) { Write-Output "$tag SERVER FAILED"; if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; continue }
        $body = @{ prompt = $dp.prompt; n_predict = $NPredict; cache_prompt = $false; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; stream = $false; return_tokens = $true } | ConvertTo-Json
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 1800
        $out = @{ tag = $tag; arm = $a.id; depth = $dp.id; env = ($a.env -join ";"); tokens = $resp.tokens; content = $resp.content; tps = $resp.timings.predicted_per_second; prompt_n = $resp.timings.prompt_n }
        $out | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath "$env:TEMP\v100\$tag.json" -Encoding UTF8
        Write-Output ("{0}: prompt_n={1} gen_n={2} tps={3:N2}" -f $tag, $resp.timings.prompt_n, $resp.tokens.Count, $resp.timings.predicted_per_second)
        Stop-Process -Id $p.Id -Force
        Start-Sleep -Seconds 2
    }
}
foreach ($kv in @("GGML_CUDA_FA_SMALL_BATCH_VEC", "GGML_CUDA_GDN_VEC4")) { [Environment]::SetEnvironmentVariable($kv, $null) }
Write-Output "3WAY DONE"
