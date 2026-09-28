$ErrorActionPreference = 'Stop'
$bin   = 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\llama-server.exe'
$model = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$dir   = '<TEMP>\v100\rollback-27b'
$port  = 18499
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$env:CUDA_VISIBLE_DEVICES = '1'

$sent = 'Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango number %d end of sentence. '
$F1 = (1000..2199 | ForEach-Object { $sent -f $_ }) -join ''   # ~ 24k tokens
$F2 = (3000..3799 | ForEach-Object { $sent -f $_ }) -join ''   # ~ 8k tokens

$err = Join-Path $dir 'srv.err.log'
$out = Join-Path $dir 'srv.out.log'
Remove-Item $err,$out -Force -ErrorAction SilentlyContinue

$argList = @('-m', $model, '-ngl', '99', '-c', '49152', '-fa', 'on',
             '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
             '--kv-tree', '--tree-ram', '2048', '--tree-disk', (Join-Path $dir 'kv-tree'), '--tree-disk-limit', '4096',
             '--ctx-checkpoints', '32', '--checkpoint-min-step', '8192',
             '-np', '1', '--no-warmup', '-lv', '4', '--host', '127.0.0.1', '--port', "$port")
$p = Start-Process -FilePath $bin -ArgumentList $argList -RedirectStandardError $err -RedirectStandardOutput $out -PassThru
Write-Output 'loading model ...'
$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Milliseconds 1000
    if ($p.HasExited) { break }
    try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { $ok = $true; break } } catch { }
}
Write-Output ("served = {0}" -f $ok)
if (-not $ok) { Write-Output 'server failed to start'; Get-Content $err | Select-Object -Last 20; exit 1 }

$script:lastA = ''
function Chat([string]$name, $messages, [int]$n) {
    $body = @{ messages = $messages; max_tokens = $n; temperature = 0; seed = 7; cache_prompt = $true } | ConvertTo-Json -Depth 8
    $t0 = Get-Date
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 900
        $txt = $resp.choices[0].message.content
    } catch { $txt = "REQUEST_FAILED: $_" }
    $dt = ((Get-Date) - $t0).TotalSeconds
    Write-Output ("[{0}] {1:N1}s -> '{2}'" -f $name, $dt, ($txt -replace "`r?`n", '\n'))
    return $txt
}

$A1 = Chat 'R1 prefill F1'    @(@{role='user'; content=$F1}) 8
Start-Sleep -Seconds 1
$A2 = Chat 'R2 extend +F2'    @(@{role='user'; content=$F1}, @{role='assistant'; content=$A1}, @{role='user'; content=$F2}) 8
Start-Sleep -Seconds 1
$A3 = Chat 'R3 rollback F1'   @(@{role='user'; content=$F1}) 8
Start-Sleep -Seconds 1
$A4 = Chat 'R4 rollback F1#2' @(@{role='user'; content=$F1}) 8

Start-Sleep -Seconds 2
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }

Write-Output ''
Write-Output '=== key log lines ==='
Select-String -Path $err -Pattern 'kv tree: (restored|restore miss|parked|rebuilt|heal|capture)|restored context checkpoint|forcing full prompt re-processing|need to evaluate at least 1 token|n_past was set|failed to trim|GGML_ABORT|checking checkpoint|created context checkpoint|erasing|superseding|selected slot by|stop processing|slot released|truncated' |
  ForEach-Object { $_.Line -replace '^\d+\.\d+\.\d+\.\d+ ', '' } | Select-Object -Last 100
