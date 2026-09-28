param([string]$ExtraArgs = "", [int]$Port = 8399, [string]$Tag = "run", [int]$PromptReps = 400)
$ErrorActionPreference = "Continue"
$exe = "D:\LLM\Backend\llama.cpp-my\llama-server.exe"
$sargs = @(
  "-m","<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf",
  "-dev","cuda1",
  "--mmproj","<models>\Qwen3.8-27B\mmproj-F16.gguf",
  "-mmdev","cuda0",
  "--image-min-tokens","1024",
  "-np","1",
  "--jinja","-ngl","99","-fa","on","--seed","42","-c","184320","-ctv","q8_0",
  "--temp","1.0","--top-p","0.95","--top-k","20","--min-p","0","--presence-penalty","0","--repeat-penalty","1.0",
  "--load-mode","mlock","--host","127.0.0.1","--port","$Port","-a","test",
  "--chat-template-file","D:/LLM/Backend/Chat-Template/llama.cpp/Qwen3.5-chat_template.jinja"
)
if ($ExtraArgs -ne "" -and $ExtraArgs -ne "none") { $sargs += $ExtraArgs.Split(" ") }
$log = "$env:TEMP\v100\server_$Tag.log"
$p = Start-Process -FilePath $exe -ArgumentList $sargs -RedirectStandardOutput $log -RedirectStandardError "$env:TEMP\v100\server_${Tag}_err.log" -PassThru -WindowStyle Hidden
$ok = $false
for ($i=0; $i -lt 400; $i++) {
  Start-Sleep -Seconds 2
  try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 -UseBasicParsing; if ($r.Content -match "ok") { $ok = $true; break } } catch {}
  if ($p.HasExited) { break }
}
if (-not $ok) { Write-Output "SERVER FAILED TO START"; Get-Content "$env:TEMP\v100\server_${Tag}_err.log" -Tail 30; if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; exit 1 }
Write-Output "=== server up (tag=$Tag, pid=$($p.Id)) ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
$sentence = "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant blue mountains. "
$prompt = $sentence * $PromptReps
foreach ($run in 1..2) {
  $body = @{ prompt = $prompt; n_predict = 1; cache_prompt = $false; temperature = 0.0; stream = $false } | ConvertTo-Json
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 900
  $sw.Stop()
  Write-Output ("run {0}: wall={1:N2}s prompt_n={2} prompt_ms={3:N0} pp={4:N1} t/s  gen_n={5} gen_ms={6:N0}" -f $run, $sw.Elapsed.TotalSeconds, $resp.timings.prompt_n, $resp.timings.prompt_ms, $resp.timings.prompt_per_second, $resp.timings.predicted_n, $resp.timings.predicted_ms)
}
Write-Output "=== startup info ==="
Get-Content $log | Select-String -Pattern "graph splits|buffer size|KV self|compute buffer|n_ctx|offloaded|CUDA" | Select-Object -First 25
Stop-Process -Id $p.Id -Force

