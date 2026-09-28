param([string]$BinDir, [int]$Replay, [int]$Np = 4, [int]$Slot = 2, [int]$NA = 16, [int]$NB = 16, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }

$port    = 8493
$slotdir = Join-Path $env:TEMP 'v100\slots4'
$m       = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
New-Item -ItemType Directory -Path $slotdir -Force | Out-Null
Remove-Item (Join-Path $slotdir '*') -Force -ErrorAction SilentlyContinue

$sargs = @('-m', $m, '-np', "$Np", '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '8192', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--slot-save-path', $slotdir,
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20')

function Start-Srv($tag) {
    $log = Join-Path $env:TEMP "t24_np4sr_$Replay`_$tag.log"
    $p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
    for ($i = 0; $i -lt 300; $i++) {
        Start-Sleep -Seconds 2
        try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { Start-Sleep -Seconds 2; return $p } } catch {}
        if ($p.HasExited) { break }
    }
    "SERVER FAILED (tag=$tag)"; Get-Content "$log.err" -Tail 8; exit 1
}
function Stop-Srv($p) {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
}
function Ask($slot, $ids, $n) {
    $body = @{ prompt = $ids; n_predict = $n; id_slot = $slot; cache_prompt = $true; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 1800
    return ,$r.tokens
}

$P = 'Compute 17 * 23 step by step.'

$srv = Start-Srv 'p1'
$tb = @{ content = $P } | ConvertTo-Json
$ids = (Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType 'application/json').tokens
$T1 = Ask $Slot $ids $NA
$sb = @{ filename = 'np4.bin' } | ConvertTo-Json
$sp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/slots/$Slot`?action=save" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($sb)) -ContentType 'application/json'
Stop-Srv $srv

$full = @($ids) + @($T1)

$srv = Start-Srv 'p2'
$TREF = Ask $Slot $full $NB
Stop-Srv $srv

$srv = Start-Srv 'p3'
$rp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/slots/$Slot`?action=restore" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($sb)) -ContentType 'application/json'
$TB = Ask $Slot $full $NB
Stop-Srv $srv

"== np=$Np slot=$Slot replay=$Replay =="
"  save n_saved=$($sp.n_saved)  restore n_restored=$($rp.n_restored)"
"  TREF: $($TREF -join ',')"
"  TB  : $($TB -join ',')"
"  TREF==TB : $(($TREF -join ',') -eq ($TB -join ','))"
"TREF," + ($TREF -join ',') | Set-Content -LiteralPath $OutFile -Encoding ASCII
"TB," + ($TB -join ',') | Add-Content -LiteralPath $OutFile -Encoding ASCII
