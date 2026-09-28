param(
    [string]$BinDir = 'D:\LLM\Backend\llama.cpp-t24',
    [string]$Model  = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf',
    [string]$Scenario = 'A',        # A = long agent turn + stripped-thinking next turn; B = two-session rotation
    [int]$CacheRam = 600,           # MiB; -1 = no limit
    [int]$NMax = 0,                 # 0 = no speculative decoding (faster); 3 = production-like
    [int]$Steps = 20,
    [int]$Ctx = 135168,
    [int]$Port = 8540,
    [string]$Tag = ''
)

$ErrorActionPreference = 'Stop'
$env:CUDA_VISIBLE_DEVICES = '1'
$env:GGML_CUDA_GDN_REPLAY = '1'
$env:LLAMA_SERVER_SLOTS_DEBUG = '1'

if (-not $Tag) { $Tag = "t32_${Scenario}_cr$CacheRam" }

$sargs = @('-m', $Model, '-np', '1', '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-ub', '512',
           '-c', "$Ctx", '--seed', '42', '-lv', '4', '--cache-ram', "$CacheRam", '--jinja',
           '--host', '127.0.0.1', '--port', "$Port", '-a', 't32')
if ($NMax -gt 0) {
    $sargs += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', "$NMax")
}
$log = "$env:TEMP\v100\t32_$Tag.err"
Remove-Item $log -ErrorAction SilentlyContinue
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $sargs -PassThru `
        -WindowStyle Hidden -RedirectStandardOutput "$env:TEMP\v100\t32_$Tag.log" -RedirectStandardError $log
$ok = $false
for ($i = 0; $i -lt 400; $i++) {
    Start-Sleep -Seconds 2
    try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3; if ($h.status -eq 'ok') { $ok = $true; Start-Sleep -Seconds 3; break } } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) { "SERVER FAILED"; Get-Content $log -Tail 20; Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; exit 1 }
"[t32/$Scenario] server ready (cache_ram=$CacheRam MiB, n_max=$NMax, ctx=$Ctx)"

$script:tools = @(@{ type = 'function'; function = @{ name = 'run'; description = 'Run a shell command';
                     parameters = @{ type = 'object'; properties = @{ cmd = @{ type = 'string' } }; required = @('cmd') } } })

function Ask($msgs, $n) {
    $body = @{ messages = $msgs; max_tokens = $n; cache_prompt = $true; tools = $script:tools;
               temperature = 0.0; top_k = 1; top_p = 1.0; seed = 42; stream = $false } | ConvertTo-Json -Depth 8
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/v1/chat/completions" -Method Post `
            -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 3600
}

$out = "$env:TEMP\v100\t32_${Tag}.jsonl"
Remove-Item $out -ErrorAction SilentlyContinue
$script:nreq = 0
function Do-Ask($msgs, $label) {
    $script:nreq++
    $w0 = Get-Date
    $r = Ask $msgs 1
    $wall = ((Get-Date) - $w0).TotalSeconds
    $tm = $r.timings
    $txt = ''
    if ($r.choices -and $r.choices.Count -gt 0) { $txt = ($r.choices[0] | ConvertTo-Json -Compress -Depth 10) }
    $rec = [ordered]@{
        tag = $Tag; step = $label; req = $script:nreq
        prompt_n = $tm.prompt_n; prompt_ms = [math]::Round($tm.prompt_ms, 1)
        gen_n = $tm.predicted_n; gen_ms = [math]::Round($tm.predicted_ms, 1)
        text = $txt
        wall_s = [math]::Round($wall, 2)
    }
    Add-Content -Path $out -Value ($rec | ConvertTo-Json -Compress) -Encoding UTF8
    "[{0}] {1,-10} prompt_n={2,-7} prompt_ms={3,-9} wall={4}s" -f `
        (Get-Date -Format 'HH:mm:ss'), $label, $tm.prompt_n, $rec.prompt_ms, $rec.wall_s
}

if ($Scenario -eq 'A') {
    $prior = [System.IO.File]::ReadAllText("$env:TEMP\v100\t20_prompt32k.txt", [System.Text.Encoding]::UTF8)
    $sys   = "You are a coding agent. Follow the user instruction."
    $fox   = "The quick brown fox jumps over the lazy dog. "

    # turn k step 1: prompt ends with user message U1 (+ assistant header)
    $msgs = @()
    $msgs += [ordered]@{ role = 'system'; content = $sys }
    $msgs += [ordered]@{ role = 'user';   content = $prior }
    Do-Ask $msgs 'k1-step0'

    # turn k steps 1..N: each adds assistant content + tool result (role=tool)
    $orig = @()   # keep the tool-result texts so the final turn can reuse them verbatim
    for ($i = 1; $i -le $Steps; $i++) {
        $a = "analysis $i : " + ($fox * 2)
        $t = "tool result $i : " + ($fox * 2)
        $orig += $t
        $tc = @([ordered]@{ id = "call_$i"; type = 'function'; function = [ordered]@{ name = 'run'; arguments = (@{ cmd = "step $i" } | ConvertTo-Json -Compress) } })
        $msgs += [ordered]@{ role = 'assistant'; content = $a; tool_calls = $tc }
        $msgs += [ordered]@{ role = 'tool';      tool_call_id = "call_$i"; content = $t }
        Do-Ask $msgs ("k1-step$i")
    }

    # turn k+1: same tool results, but all assistant contents replaced (thinking stripped analogue)
    $fin = @()
    $fin += [ordered]@{ role = 'system'; content = $sys }
    $fin += [ordered]@{ role = 'user';   content = $prior }
    for ($i = 1; $i -le $Steps; $i++) {
        $tc = @([ordered]@{ id = "call_$i"; type = 'function'; function = [ordered]@{ name = 'run'; arguments = (@{ cmd = "step $i" } | ConvertTo-Json -Compress) } })
        $fin += [ordered]@{ role = 'assistant'; content = "done $i"; tool_calls = $tc }
        $fin += [ordered]@{ role = 'tool';      tool_call_id = "call_$i"; content = $orig[$i-1] }
    }
    $fin += [ordered]@{ role = 'user'; content = "Summarize the whole session briefly." }
    Do-Ask $fin 'k2-final'

} elseif ($Scenario -eq 'B') {
    $text  = [System.IO.File]::ReadAllText("$env:TEMP\v100\t20_prompt128k.txt", [System.Text.Encoding]::UTF8)
    $sys   = "You are a coding agent. Follow the user instruction."
    $comA  = $text.Substring(0, 4000)          # shared head (LCP ~1K tokens)
    $bodyA = $comA + " MARKER-A " + $text.Substring(4000, 26000)
    $bodyB = $comA + " MARKER-B " + $text.Substring(30000, 26000)
    $fox   = "The quick brown fox jumps over the lazy dog. "

    $mA = @(); $mA += [ordered]@{ role = 'system'; content = $sys }; $mA += [ordered]@{ role = 'user'; content = $bodyA }
    $mB = @(); $mB += [ordered]@{ role = 'system'; content = $sys }; $mB += [ordered]@{ role = 'user'; content = $bodyB }

    for ($r = 0; $r -lt 3; $r++) {
        $grow = " round $r : " + ($fox * 4)
        $mA += [ordered]@{ role = 'assistant'; content = "A reply $r" }
        $mA += [ordered]@{ role = 'user';      content = "A followup $r" + $grow }
        $mB += [ordered]@{ role = 'assistant'; content = "B reply $r" }
        $mB += [ordered]@{ role = 'user';      content = "B followup $r" + $grow }
        Do-Ask $mA ("A-round$r")
        Do-Ask $mB ("B-round$r")
    }
    Do-Ask $mA 'A-final'
    Do-Ask $mB 'B-final'
}

"NOTE: log = $log ; parse with t32_log.py"
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
"done -> $out"

