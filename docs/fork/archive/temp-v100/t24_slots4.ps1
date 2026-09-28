param([string]$BinDir, [int]$Replay, [int]$R = 1, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue
if ($Replay -eq 1) { $env:GGML_CUDA_GDN_REPLAY = '1' }

$port    = 8487
$slotdir = Join-Path $env:TEMP 'v100\slots4'
$m       = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'
New-Item -ItemType Directory -Path $slotdir -Force | Out-Null
Remove-Item (Join-Path $slotdir '*') -Force -ErrorAction SilentlyContinue

$sargs = @('-m', $m, '-np', '1', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '8192', '-ctv', 'q8_0',
           '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3',
           '--slot-save-path', $slotdir,
           '--host', '127.0.0.1', '--port', "$port", '-a', 't20', '-v')

function Start-Srv($tag) {
    $log = Join-Path $env:TEMP "t24_slots4_$Replay`_r$R`_$tag.log"
    $p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')
    for ($i = 0; $i -lt 300; $i++) {
        Start-Sleep -Seconds 2
        try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { Start-Sleep -Seconds 2; return $p } } catch {}
        if ($p.HasExited) { break }
    }
    "SERVER FAILED (replay=$Replay r=$R tag=$tag)"; Get-Content $log -Tail 10; Get-Content ($log + '.err') -Tail 10; exit 1
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

# phase 1: prefill + 40 tokens + save a.bin (state has pending p at save)
$srv = Start-Srv 'p1'
$tb = @{ content = $P } | ConvertTo-Json
$ids = (Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tb)) -ContentType 'application/json').tokens
$TA = Ask $ids 40
$sp = Save-Slot 'a.bin'
$tbx = @{ content = ' Berlin' } | ConvertTo-Json
$X = @((Invoke-RestMethod -Uri "http://127.0.0.1:$port/tokenize" -Method Post -Body ([System.Text.Encoding]::UTF8.GetBytes($tbx)) -ContentType 'application/json').tokens[0])
Stop-Srv $srv

# trimmed prompt: TA minus its last R tokens, then X
$short = @($ids) + @($TA[0..($TA.Count-1-$R)]) + @($X)
"== replay=$Replay R=$R =="
"  TA=$($TA.Count) tok, short=$($short.Count) tok, X=$X"

# reference: fresh prefill of the trimmed prompt
$srv = Start-Srv 'ref'
$TD = Ask $short 8
Stop-Srv $srv

# restore a.bin then ask the trimmed prompt (trim = R -> partial rollback)
$srv = Start-Srv 'res'
$rp = Load-Slot 'a.bin'
$TC = Ask $short 8
Stop-Srv $srv

"  short  : $($short -join ',')"
"  TD     : $($TD -join ',')"
"  TC     : $($TC -join ',')"
"  TC==TD : $(($TC -join ',') -eq ($TD -join ','))"

"TD," + ($TD -join ',') | Set-Content -LiteralPath $OutFile -Encoding ASCII
"TC," + ($TC -join ',') | Add-Content -LiteralPath $OutFile -Encoding ASCII
"short," + ($short -join ',') | Add-Content -LiteralPath $OutFile -Encoding ASCII

