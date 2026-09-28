param(
    [string]$BinDir = 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release',
    [string]$Model  = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf',
    [int]$Np     = 2,
    [int]$Ctx    = 98304,
    [int]$Port   = 8560,
    [string]$Tag = 'np2'
)

$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '1'
$env:GGML_CUDA_GDN_REPLAY = '1'

$sargs = @('-m', $Model, '-np', "$Np", '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-ub', '512',
           '-c', "$Ctx", '--seed', '42', '-lv', '4', '--jinja',
           '--host', '127.0.0.1', '--port', "$Port", '-a', 't32np2')
$log = "$env:TEMP\v100\t32_np2_$Tag.err"
Remove-Item $log -ErrorAction SilentlyContinue
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput "$env:TEMP\v100\t32_np2_$Tag.log" -RedirectStandardError $log
$ok = $false
for ($i = 0; $i -lt 400; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 3; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED"; Get-Content $log -Tail 20; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; exit 1 }
"[np2] server ready (np=$Np ctx=$Ctx)"

$tools = @(@{ type = 'function'; function = @{ name = 'run'; description = 'Run a shell command';
              parameters = @{ type = 'object'; properties = @{ cmd = @{ type = 'string' } }; required = @('cmd') } } })

function Ask($msgs, $idSlot) {
    $body = @{ messages = $msgs; max_tokens = 1; cache_prompt = $true; tools = $tools; id_slot = $idSlot;
               temperature = 0.0; top_k = 1; seed = 42; stream = $false } | ConvertTo-Json -Depth 8
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 3600
}

function Vram() { $u = & nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits; return [int]$u }

$sys  = "You are a coding agent. Follow the user instruction."
$text = [System.IO.File]::ReadAllText("$env:TEMP\v100\t20_prompt128k.txt", [System.Text.Encoding]::UTF8)
$shared = $text.Substring(0, 120000)          # ~30K tokens shared prefix
$uniqA  = " UNIQUE-A section. " + $text.Substring(200000, 8000)
$uniqB  = " UNIQUE-B section. " + $text.Substring(300000, 8000)

$mA = @(); $mA += [ordered]@{ role = 'system'; content = $sys }; $mA += [ordered]@{ role = 'user'; content = ($shared + $uniqA) }
$mB = @(); $mB += [ordered]@{ role = 'system'; content = $sys }; $mB += [ordered]@{ role = 'user'; content = ($shared + $uniqB) }

foreach ($r in 0..2) {
    $grow = " round $r continued."
    if ($r -gt 0) {
        $mA += [ordered]@{ role = 'assistant'; content = "A reply $r" }
        $mA += [ordered]@{ role = 'user';      content = "A followup $r $grow" }
        $mB += [ordered]@{ role = 'assistant'; content = "B reply $r" }
        $mB += [ordered]@{ role = 'user';      content = "B followup $r $grow" }
    }
    foreach ($c in @(@('A', 0, $mA), @('B', 1, $mB))) {
        $w0 = Get-Date
        $resp = Ask $c[2] $c[1]
        $wall = ((Get-Date) - $w0).TotalSeconds
        $tm = $resp.timings
        "[{0}] {1}-round{2} slot={3} prompt_n={4,-7} prompt_ms={5,-9} vram_used={6} MiB" -f `
            (Get-Date -Format 'HH:mm:ss'), $c[0], $r, $c[1], $tm.prompt_n, [math]::Round($tm.prompt_ms,1), (Vram)
        Add-Content -Path "$env:TEMP\v100\t32_np2_progress.txt" -Encoding UTF8 -Value (
            "[{0}] {1}-round{2} slot={3} prompt_n={4,-7} prompt_ms={5,-9} vram_used={6} MiB" -f `
            (Get-Date -Format 'HH:mm:ss'), $c[0], $r, $c[1], $tm.prompt_n, [math]::Round($tm.prompt_ms,1), (Vram))
    }
}

"NOTE: log = $log (srv prompt cache lines: none expected)"
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
"done"
