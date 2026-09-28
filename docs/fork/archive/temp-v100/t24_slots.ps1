param([string]$BinDir, [int]$Replay, [int]$NA = 16, [int]$NB = 8)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }

$port    = 8484
$slotdir = Join-Path $env:TEMP 'v100\slots'
$m       = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
New-Item -ItemType Directory -Path $slotdir -Force | Out-Null
Remove-Item (Join-Path $slotdir '*') -Force -ErrorAction SilentlyContinue

function Start-Srv {
    $sargs = @('-m', $m, '-np', '1', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '4096', '-ctv', 'q8_0',
               '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
               '--slot-save-path', $slotdir,
               '--host', '127.0.0.1', '--port', "$port", '-a', 't20')
    $log = Join-Path $env:TEMP "t24_slots_$Replay.log"
    $p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
    for ($i = 0; $i -lt 300; $i++) {
        Start-Sleep -Seconds 2
        try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { Start-Sleep -Seconds 2; return $p } } catch {}
        if ($p.HasExited) { break }
    }
    "SERVER FAILED (replay=$Replay)"
    Get-Content $log -Tail 15
    Get-Content ($log + '.err') -Tail 15
    exit 1
}

function Stop-Srv($p) {
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 3
}

function Ask($prompt, $n) {
    $body = @{ prompt = $prompt; n_predict = $n; cache_prompt = $true; temperature = 0.0; top_k = 1; min_p = 0.0; seed = 42; stream = $false; return_tokens = $true } | ConvertTo-Json
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 1800
    return ,$r.tokens
}

$P = 'The capital of France is'

# 1) tokenize + first request + save slot
$srv = Start-Srv
$tb = @{ content = $P } | ConvertTo-Json
$ids = (Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType 'application/json').tokens
$TA = Ask $ids $NA
"saved: P=$($ids.Count) tokens, TA=$($TA -join ',')"
$sb = @{ filename = 't24slot.bin' } | ConvertTo-Json
$sp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/slots/0?action=save" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($sb)) -ContentType 'application/json'
"save: $($sp | ConvertTo-Json -Compress)"
Stop-Srv $srv

$full = @($ids) + @($TA)
$file = Join-Path $slotdir 't24slot.bin'
"slot file exists: $(Test-Path $file), size: $(if (Test-Path $file) { (Get-Item $file).Length } else { 0 })"

# 2) reference run (no restore)
$srv = Start-Srv
$TREF = Ask $full $NB
"reference: TREF=$($TREF -join ',')"
Stop-Srv $srv

# 3) restore run
$srv = Start-Srv
$rp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/slots/0?action=restore" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($sb)) -ContentType 'application/json'
"restore: $($rp | ConvertTo-Json -Compress)"
$TB = Ask $full $NB
"restored:  TB =$($TB -join ',')"
Stop-Srv $srv

"RESULT replay=$Replay : TREF==TB : $(($TREF -join ',') -eq ($TB -join ','))"
