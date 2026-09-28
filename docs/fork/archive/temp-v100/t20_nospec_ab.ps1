param([int]$NPredict = 250, [int]$PortBase = 8500)
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "1"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"

$prompts = @(
    @{ id = "p1_factual"; text = "What is the capital of France? Answer in one sentence, then list three interesting facts about the Eiffel Tower." },
    @{ id = "p2_creative"; text = "Write a long detailed essay about the history of computing. Be verbose." },
    @{ id = "p3_code"; text = "Write a Python function that computes the n-th Fibonacci number iteratively. Include a docstring and a usage example." },
    @{ id = "p4_math"; text = "Solve step by step: A train travels 240 km in 2.5 hours. What is its average speed? Then, if it continues at that speed, how far will it travel in 4 hours?" }
)
$builds = @(
    @{ id = "ours";  dir = "D:\LLM\Backend\llama.cpp-my" },
    @{ id = "stock"; dir = "D:\LLM\Backend\llama.cpp" }
)

$port = $PortBase
foreach ($pr in $prompts) {
    foreach ($b in $builds) {
        $port++
        $tag = "$($b.id)_$($pr.id)"
        $exe = "$($b.dir)\llama-server.exe"
        $sargs = @(
            "-m", $m, "-dev", "cuda0", "-np", "1", "-ngl", "99", "-fa", "on", "--seed", "42",
            "-c", "4096", "-ctv", "q8_0", "--host", "127.0.0.1", "--port", "$port", "-a", "t20"
        )
        $log = "$env:TEMP\v100\t20_ns_$tag.log"
        $errlog = "$env:TEMP\v100\t20_ns_${tag}_err.log"
        $p = Start-Process -FilePath $exe -ArgumentList $sargs -RedirectStandardOutput $log -RedirectStandardError $errlog -PassThru -WindowStyle Hidden
        $ok = $false
        for ($i = 0; $i -lt 200; $i++) {
            Start-Sleep -Seconds 2
            try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3 -UseBasicParsing; if ($r.Content -match "ok") { $ok = $true; break } } catch {}
            if ($p.HasExited) { break }
        }
        if (-not $ok) { Write-Output "$tag SERVER FAILED"; if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; continue }
        $body = @{ prompt = $pr.text; n_predict = $NPredict; cache_prompt = $false; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; stream = $false; return_tokens = $true } | ConvertTo-Json
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 900
        $out = @{ tag = $tag; build = $b.id; prompt_id = $pr.id; tokens = $resp.tokens; content = $resp.content; tps = $resp.timings.predicted_per_second }
        $out | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath "$env:TEMP\v100\t20_ns_$tag.json" -Encoding UTF8
        Write-Output ("{0}: n={1} tps={2:N2}" -f $tag, $resp.tokens.Count, $resp.timings.predicted_per_second)
        Stop-Process -Id $p.Id -Force
        Start-Sleep -Seconds 2
    }
}
Write-Output "NOSPEC AB DONE"
