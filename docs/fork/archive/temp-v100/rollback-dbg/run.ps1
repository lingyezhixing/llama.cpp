$ErrorActionPreference = 'Stop'
$bin   = 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\llama-server.exe'
$model = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
$dir   = '<TEMP>\v100\rollback-dbg'
$port  = 18599
New-Item -ItemType Directory -Force -Path $dir | Out-Null
$env:CUDA_VISIBLE_DEVICES = '1'

$sent = 'Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango number %d end of sentence. '
function Blk([int]$a, [int]$n) { (($a..($a+$n-1)) | ForEach-Object { $sent -f $_ }) -join '' }
$U1 = 'Context part one. '   + (Blk 1000 350)
$U2 = 'Context part two. '   + (Blk 2000 350)
$U3 = 'Context part three. ' + (Blk 3000 350)

$err = Join-Path $dir 'srv.err.log'
$out = Join-Path $dir 'srv.out.log'
Remove-Item $err,$out -Force -ErrorAction SilentlyContinue

$argList = @('-m', $model, '-ngl', '99', '-fa', 'on', '-c', '184320', '-ctv', 'q8_0',
             '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
             '--ctx-checkpoints', '16', '--checkpoint-min-step', '16384',
             '--kv-tree', '--tree-ram', '4096', '--tree-disk', (Join-Path $dir 'kv-tree'), '--tree-disk-limit', '8192',
             '--jinja', '-np', '1', '--no-warmup', '-lv', '4', '--host', '127.0.0.1', '--port', "$port")
$p = Start-Process -FilePath $bin -ArgumentList $argList -RedirectStandardError $err -RedirectStandardOutput $out -PassThru
Write-Output 'loading 27B (ctx 184320) ...'
$ok = $false
for ($i = 0; $i -lt 420; $i++) {
    Start-Sleep -Milliseconds 1000
    if ($p.HasExited) { break }
    try { $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { $ok = $true; break } } catch { }
}
Write-Output ("served = {0}" -f $ok)
if (-not $ok) { Write-Output 'server failed to start'; Get-Content $err | Select-Object -Last 20; exit 1 }

$script:chat_n = 0
function Chat([string]$name, $messages) {
    $body = @{ messages = $messages; max_tokens = 16; temperature = 0; seed = 42; cache_prompt = $true } | ConvertTo-Json -Depth 8
    $t0 = Get-Date
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 1200
        $txt = [string]$resp.choices[0].message.content
    } catch { $txt = 'REQUEST_FAILED' }
    $dt = ((Get-Date) - $t0).TotalSeconds
    $hash = (Get-FileHash -InputStream ([IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes($txt))) -Algorithm SHA256).Hash.Substring(0,12)
    [Console]::WriteLine(("[{0}] {1,7:N1}s hash={2} -> '{3}'" -f $name, $dt, $hash, ($txt -replace "`r?`n", '\n')))
    return $txt
}

$mU1 = @{role='user'; content=$U1}
$A1 = Chat 'R1 [U1]' @($mU1)
$mA1 = @{role='assistant'; content=$A1}
$mU2 = @{role='user'; content=$U2}
$A2 = Chat 'R2 [U1,A1,U2]' @($mU1,$mA1,$mU2)
$mA2 = @{role='assistant'; content=$A2}
$mU3 = @{role='user'; content=$U3}
$conv3 = @($mU1,$mA1,$mU2,$mA2,$mU3)

$A3  = Chat 'R3  [..U3]'          $conv3
$A4  = Chat 'R4  retry[..U3] #1'  $conv3
$A5  = Chat 'R5  retry[..U3+x]'   @($mU1,$mA1,$mU2,$mA2, @{role='user'; content=($U3 + ' extra one')})
$A6  = Chat 'R6  rollback[..U2]'  @($mU1,$mA1,$mU2)
$A7  = Chat 'R7  retry[..U3] #2'  $conv3
$A8  = Chat 'R8  retry[..U3] #3'  $conv3
$A9  = Chat 'R9  rollback[..U2]#2' @($mU1,$mA1,$mU2)
$A10 = Chat 'R10 retry[..U3] #4'  $conv3

Start-Sleep -Seconds 2
if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }

Write-Output ''
Write-Output '=== checkpoint / restore / rollback log (last 140) ==='
Select-String -Path $err -Pattern '\[dbg\]|restored context checkpoint|forcing full prompt re-processing|need to evaluate at least 1 token|n_past was set|-> ROLLBACK|-> CLEARED|-> FAIL|restored: tail|created context checkpoint|erasing old|superseding|checking checkpoint|kv tree: (restored|restore miss|parked|rebuilt)|selected slot by|stop processing' |
  ForEach-Object { $_.Line -replace '^\d+\.\d+\.\d+\.\d+ ', '' } | Select-Object -Last 140
