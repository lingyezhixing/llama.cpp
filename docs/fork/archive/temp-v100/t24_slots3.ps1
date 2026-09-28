param([string]$BinDir, [int]$Replay, [int]$Check = 0, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }
if ($Check  -eq 1) { $env:GGML_CUDA_GDN_REPLAY_CHECK = '1' }

$port    = 8483
$slotdir = Join-Path $env:TEMP 'v100\slots3'
$m       = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
New-Item -ItemType Directory -Path $slotdir -Force | Out-Null
Remove-Item (Join-Path $slotdir '*') -Force -ErrorAction SilentlyContinue

$sargs = @('-m', $m, '-np', '1', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '8192', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--slot-save-path', $slotdir,
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')

function Start-Srv($tag) {
    $log = Join-Path $env:TEMP "t24_slots3_$Replay`_$Check`_$tag.log"
    $p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
    for ($i = 0; $i -lt 300; $i++) {
        Start-Sleep -Seconds 2
        try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { Start-Sleep -Seconds 2; return $p } } catch {}
        if ($p.HasExited) { break }
    }
    "SERVER FAILED (replay=$Replay check=$Check tag=$tag)"
    Get-Content $log -Tail 10
    Get-Content ($log + '.err') -Tail 10
    exit 1
}
function Stop-Srv($p) {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
}
function Ask($ids, $n) {
    $body = @{ prompt = $ids; n_predict = $n; cache_prompt = $true; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 1800
    return ,$r.tokens
}
function Save-Slot($file) {
    $sb = @{ filename = $file } | ConvertTo-Json
    return Invoke-RestMethod -Uri "http://127.0.0.1:$port/slots/0?action=save" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($sb)) -ContentType 'application/json'
}
function Load-Slot($file) {
    $sb = @{ filename = $file } | ConvertTo-Json
    return Invoke-RestMethod -Uri "http://127.0.0.1:$port/slots/0?action=restore" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($sb)) -ContentType 'application/json'
}

$P = 'The capital of France is'

# phase 1: prefill + 40 tokens + save
$srv = Start-Srv 'p1'
$tb = @{ content = $P } | ConvertTo-Json
$ids = (Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType 'application/json').tokens
$TA = Ask $ids 40
$sp = Save-Slot 'a.bin'
"== replay=$Replay check=$Check =="
"  TA   : $($TA.Count) tok, save n_saved=$($sp.n_saved)"
Stop-Srv $srv

# phase 2: reference on a fresh server (no restore)
$srv = Start-Srv 'p2'
$TREF1 = Ask (@($ids) + @($TA)) 8
$TREF2 = Ask (@($ids) + @($TA) + @($TREF1)) 8
$tbx = @{ content = ' Berlin' } | ConvertTo-Json
$X = @((Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tbx)) -ContentType 'application/json').tokens[0])
Stop-Srv $srv

# phase 3: restore a.bin, continue, save b.bin
$srv = Start-Srv 'p3'
$rp = Load-Slot 'a.bin'
$TB1 = Ask (@($ids) + @($TA)) 8
$sp = Save-Slot 'b.bin'
Stop-Srv $srv

# phase 4: restore b.bin (saved right after a restore), continue
$srv = Start-Srv 'p4'
$rp = Load-Slot 'b.bin'
$TB2 = Ask (@($ids) + @($TA) + @($TB1)) 8
Stop-Srv $srv

# phase 5: restore + partial trim (last token of TA replaced by X)
$short = @($ids) + @($TA[0..($TA.Count-2)]) + @($X)
$srv = Start-Srv 'p5'
$rp = Load-Slot 'a.bin'
$TC = Ask $short 8
Stop-Srv $srv

# phase 6: same prompt without restore (fresh prefill)
$srv = Start-Srv 'p6'
$TD = Ask $short 8
Stop-Srv $srv

$lines = @()
$lines += 'TREF1,' + ($TREF1 -join ',')
$lines += 'TB1,'   + ($TB1   -join ',')
$lines += 'TREF2,' + ($TREF2 -join ',')
$lines += 'TB2,'   + ($TB2   -join ',')
$lines += 'TC,'    + ($TC    -join ',')
$lines += 'TD,'    + ($TD    -join ',')
$lines | Set-Content -LiteralPath $OutFile -Encoding ASCII

"  TREF1==TB1 : $(($TREF1 -join ',') -eq ($TB1 -join ','))"
"  TREF2==TB2 : $(($TREF2 -join ',') -eq ($TB2 -join ','))"
"  TC==TD     : $(($TC -join ',') -eq ($TD -join ','))"
"done: replay=$Replay check=$Check -> $OutFile"
