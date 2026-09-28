param(
  [string]$Tag = "none",
  [string]$ExtraArgs = "",
  [int]$Port = 8460,
  [int]$NPredict = 900,
  [int]$Ctx = 16384,
  [int]$NProbs = 0,
  [string]$EnvVars = "",
  [string]$PromptFile = "",
  [string]$Prompt = "Write a long detailed essay about the history of computing. Be verbose."
)
$ErrorActionPreference = "Continue"
if ($PromptFile -ne "") { $Prompt = [System.IO.File]::ReadAllText($PromptFile) }
$env:CUDA_VISIBLE_DEVICES = "1"
foreach ($kv in $EnvVars.Split(";")) {
  if ($kv -ne "") { $parts = $kv.Split("="); [Environment]::SetEnvironmentVariable($parts[0], $parts[1]) }
}
$exe = "D:\LLM\Backend\llama.cpp-my\llama-server.exe"
$sargs = @(
  "-m","<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf",
  "-dev","cuda0","-np","1","-ngl","99","-fa","on","--seed","42","-c","$Ctx","-ctv","q8_0",
  "--host","127.0.0.1","--port","$Port","-a","t20"
)
if ($ExtraArgs -ne "" -and $ExtraArgs -ne "none") { $sargs += $ExtraArgs.Split(" ") }
$log = "$env:TEMP\v100\t20_srv_$Tag.log"
$errlog = "$env:TEMP\v100\t20_srv_${Tag}_err.log"
$p = Start-Process -FilePath $exe -ArgumentList $sargs -RedirectStandardOutput $log -RedirectStandardError $errlog -PassThru -WindowStyle Hidden
$ok = $false
for ($i=0; $i -lt 400; $i++) {
  Start-Sleep -Seconds 2
  try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3 -UseBasicParsing; if ($r.Content -match "ok") { $ok = $true; break } } catch {}
  if ($p.HasExited) { break }
}
if (-not $ok) { Write-Output "SERVER FAILED TO START ($Tag)"; Get-Content $errlog -Tail 20; if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }; exit 1 }
Write-Output "=== server up (tag=$Tag, args_extra='$ExtraArgs') ==="
$body = @{ prompt = $Prompt; n_predict = $NPredict; cache_prompt = $false; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; stream = $false; return_tokens = $true; n_probs = $NProbs; ignore_eos = $true } | ConvertTo-Json
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 3600
$sw.Stop()
$out = @{ tag = $Tag; extra = $ExtraArgs; predicted_n = $resp.timings.predicted_n; predicted_ms = $resp.timings.predicted_ms; tps = $resp.timings.predicted_per_second; prompt_n = $resp.timings.prompt_n; wall_s = $sw.Elapsed.TotalSeconds; tokens = $resp.tokens; content = $resp.content; stop_type = $resp.stop_type }
$out | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath "$env:TEMP\v100\t20_srv_$Tag.json" -Encoding UTF8
$resp | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath "$env:TEMP\v100\t20_srv_${Tag}_raw.json" -Encoding UTF8
Write-Output ("gen_n={0} gen_ms={1:N0} **tg={2:N2} t/s** prompt_n={3} wall={4:N1}s stop={5}" -f $resp.timings.predicted_n, $resp.timings.predicted_ms, $resp.timings.predicted_per_second, $resp.timings.prompt_n, $sw.Elapsed.TotalSeconds, $resp.stop_type)
Get-Content $errlog | Select-String -Pattern "draft|accept|spec" | Select-Object -Last 5 | Select-Object -ExpandProperty Line
Stop-Process -Id $p.Id -Force
