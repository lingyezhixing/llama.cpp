param([string]$BinDir, [int]$Replay, [string]$OutFile)
$env:CUDA_VISIBLE_DEVICES = '1'
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1'; $env:T24_REPLAY = '1' } else { Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue; $env:T24_REPLAY = '0' }
$log  = Join-Path $env:TEMP ("t24_n1_" + $Replay + ".log")
$wrap = Join-Path $env:TEMP 'v100\t24_srv_run_n1.cmd'
$p = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', $wrap, $BinDir, $log) -PassThru -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 240; $i++) { Start-Sleep -Seconds 2; try { $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8479/health' -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; break } } catch {}; if ($p.HasExited) { break } }
if (-not $ok) { "SERVER FAILED (replay=$Replay)"; Get-Content $log -Tail 20; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; exit 1 }
$body = @{ prompt = 'The quick brown fox jumps over the lazy dog. Count from 1 to 40 slowly.'; n_predict = 64; temperature = 0; seed = 42; ignore_eos = $true; return_tokens = $true; cache_prompt = $false } | ConvertTo-Json
$r = Invoke-RestMethod -Uri 'http://127.0.0.1:8479/completion' -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 600
($r.tokens -join ',') | Set-Content -LiteralPath $OutFile -Encoding ASCII
"replay=$Replay tokens=$($r.tokens.Count)"
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort 8479 -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2

