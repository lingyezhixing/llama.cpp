param([string]$ExtraArgs = "", [int]$Port = 8399, [string]$Tag = "dec", [int]$NPredict = 128)
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
if (-not $ok) { Write-Output "SERVER FAILED TO START"; Get-Content "$env:TEMP\v100\server_${Tag}_err.log" -Tail 20; if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; exit 1 }
Write-Output "=== server up (tag=$Tag, pid=$($p.Id)) args_extra='$ExtraArgs' ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
$short = "Write a long detailed essay about the history of computing. Be verbose."
foreach ($run in 1..3) {
  $body = @{ prompt = $short; n_predict = $NPredict; cache_prompt = $false; temperature = 0.0; stream = $false } | ConvertTo-Json
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 900
  $sw.Stop()
  Write-Output ("decode run {0}: gen_n={1} gen_ms={2:N0} **tg={3:N2} t/s** (wall {4:N2}s)" -f $run, $resp.timings.predicted_n, $resp.timings.predicted_ms, $resp.timings.predicted_per_second, $sw.Elapsed.TotalSeconds)
}
Write-Output "=== spec info (if any) ==="
Get-Content "$env:TEMP\v100\server_${Tag}_err.log" | Select-String -Pattern "draft|spec|accept" | Select-Object -Last 6
Stop-Process -Id $p.Id -Force
