$ErrorActionPreference = 'Stop'
$bin   = 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\llama-server.exe'
$model = '<models>\Qwen3.5-0.8B-MTP\Qwen3.5-0.8B-UD-Q4_K_XL.gguf'
$dir   = '<TEMP>\v100\diag-smoke'
$port  = 18799
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$env:CUDA_VISIBLE_DEVICES = '1'

$sent = 'Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango number %d end of sentence. '
$U1 = 'Context part one. ' + (((1000..1150) | ForEach-Object { $sent -f $_ }) -join '')

$err = Join-Path $dir 'srv.err.log'
$out = Join-Path $dir 'srv.out.log'
Remove-Item $err,$out -Force -ErrorAction SilentlyContinue
$argList = @('-m', $model, '-ngl', '99', '-c', '8192', '-fa', 'on',
             '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
             '--ctx-checkpoints', '16', '--checkpoint-min-step', '2048',
             '--kv-tree', '--tree-ram', '256', '--tree-disk', (Join-Path $dir 'kv-tree'), '--tree-disk-limit', '2048',
             '-np', '1', '--no-warmup', '--host', '127.0.0.1', '--port', "$port")
$p = Start-Process -FilePath $bin -ArgumentList $argList -RedirectStandardError $err -RedirectStandardOutput $out -PassThru
$ok = $false
for ($i = 0; $i -lt 120; $i++) { Start-Sleep -Milliseconds 1000; if ($p.HasExited) { break }; try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { $ok = $true; break } } catch { } }
Write-Output ("served = {0}" -f $ok)

function Chat([string]$name, $messages) {
    $body = @{ messages = $messages; max_tokens = 8; temperature = 0; seed = 42; cache_prompt = $true } | ConvertTo-Json -Depth 8
    try { $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 600; $txt = [string]$resp.choices[0].message.content } catch { $txt = 'FAILED' }
    [Console]::WriteLine(("[{0}] -> '{1}'" -f $name, ($txt -replace "`r?`n", ' ')))
}
$mU1 = @{role='user'; content=('/no_think ' + $U1)}
Chat 'R1 [U1]' @($mU1) | Out-Null
Chat 'R2 [U1] retry' @($mU1) | Out-Null
Chat 'R3 [U1] retry2' @($mU1) | Out-Null
Start-Sleep -Seconds 2
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }

Write-Output ''
Write-Output '=== diag lines ==='
Select-String -Path $err -Pattern '\[diag\]|\[dbg\]|-> ' | ForEach-Object { $_.Line -replace '^\d+\.\d+\.\d+\.\d+ ', '' } | Select-Object -Last 18
