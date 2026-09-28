param([string]$BinDir, [string]$Mode, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue

$port = 8481
$m = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'

$args = @('-m', $m, '--jinja', '-ngl', '99', '-fa', 'on', '--seed', '42', '-c', '65536')
$args += @('--temp', '1.0', '--top-p', '0.95', '--top-k', '20', '--min-p', '0')
$args += @('--presence-penalty', '0', '--repeat-penalty', '1.0', '--load-mode', 'mlock')
$args += @('--host', '127.0.0.1', '--port', "$port", '-a', 'Qwen3.8-27B')
if ($Mode -ne 'base') {
    $args += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', '3')
    if ($Mode -eq 'on') { $env:GGML_CUDA_GDN_REPLAY = '1' }
}

$log = Join-Path $env:TEMP ("t24_fc_" + $Mode + ".log")
$exe = Join-Path $BinDir 'llama-server.exe'
$p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')

$ok = $false
for ($i = 0; $i -lt 300; $i++) {
    Start-Sleep -Seconds 2
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3
        if ($h.status -eq 'ok') { $ok = $true; break }
    } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) {
    "SERVER FAILED ($Mode)"
    Get-Content $log -Tail 12
    Get-Content ($log + '.err') -Tail 12
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

$lines = @()
$lines += '用贴吧暴躁老哥的口气和风格， 且使用文言文，  仿《过秦论》作《过美利坚论》'
$lines += '注意：'
$lines += '1、文言文要言辞犀利，且不带白话文句子'
$lines += '2、用非常硬核的文言文'
$lines += '3、知识面体现的要宽'
$lines += '4、插上想象力翅膀，豪放主义诗人的奔放程度。'
$lines += '5、用词要极具想象力，且非常奔放不羁，你要放飞自我。'
$q = $lines -join [Environment]::NewLine

$body = @{
    messages    = @(@{ role = 'user'; content = $q })
    temperature = 1.0
    top_p       = 0.95
    top_k       = 20
    min_p       = 0.0
    seed        = 42
    max_tokens  = 16384
    stream      = $false
} | ConvertTo-Json -Depth 6

$uri = "http://127.0.0.1:$port/v1/chat/completions"
$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$r = Invoke-RestMethod -Uri $uri -Method Post -Body $bytes -ContentType 'application/json' -TimeoutSec 7200
$sw.Stop()

$content = $r.choices[0].message.content
$ct = $r.usage.completion_tokens
Set-Content -LiteralPath $OutFile -Value $content -Encoding UTF8
$md5 = (Get-FileHash -LiteralPath $OutFile -Algorithm MD5).Hash

"$Mode : tokens=$ct chars=$($content.Length) seconds=$([math]::Round($sw.Elapsed.TotalSeconds,1)) md5=$md5"

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
