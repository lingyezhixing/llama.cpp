param([string]$BinDir, [string]$Mode, [string]$OutFile)

$env:CUDA_VISIBLE_DEVICES = '1'
Remove-Item Env:\GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue
Remove-Item Env:\GGML_CUDA_GDN_REPLAY_CHECK -ErrorAction SilentlyContinue

$port = 8480
$m = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf'

$args = @('-m', $m, '-np', '1', '-ngl', '99', '-fa', 'on', '-c', '16384', '-ctv', 'q8_0', '--port', "$port", '--host', '127.0.0.1')
if ($Mode -eq 'on') {
    $env:GGML_CUDA_GDN_REPLAY = '1'
    $args += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', '3')
} elseif ($Mode -eq 'off') {
    $args += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', '3')
}

$log = Join-Path $env:TEMP ("t24_multi_" + $Mode + ".log")
$p = Start-Process -FilePath (Join-Path $BinDir 'llama-server.exe') -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError ($log + '.err')

$ok = $false
for ($i = 0; $i -lt 240; $i++) {
    Start-Sleep -Seconds 2
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 3
        if ($h.status -eq 'ok') { $ok = $true; break }
    } catch {}
    if ($p.HasExited) { break }
}
if (-not $ok) {
    "SERVER FAILED ($Mode)"
    Get-Content $log -Tail 15
    Get-Content ($log + '.err') -Tail 15
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

$prompts = @(
    'The quick brown fox jumps over the lazy dog. Count from 1 to 40 slowly.',
    'Explain in three sentences why the sky is blue.',
    'Write a C++ function that reverses a singly linked list, then explain it briefly.',
    'List the first ten prime numbers and their squares.',
    'Translate the sentence "Good morning, how are you?" into French, German and Japanese.',
    'A farmer has 17 sheep and all but 9 run away. How many are left? Explain your reasoning.',
    'Write a short haiku about the ocean at night.',
    'What are the main differences between TCP and UDP? Answer with a small table.',
    'Compute 1234 * 5678 step by step.',
    'Summarize the plot of Romeo and Juliet in exactly two sentences.'
)

$results = @()
foreach ($pr in $prompts) {
    $body = @{
        prompt        = $pr
        n_predict     = 64
        temperature   = 0
        seed          = 42
        ignore_eos    = $true
        return_tokens = $true
        cache_prompt  = $false
    } | ConvertTo-Json

    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/completion" -Method Post -Body ([Text.Encoding]::UTF8.GetBytes($body)) -ContentType 'application/json' -TimeoutSec 900
    $results += ($r.tokens -join ',')
    "  $($Mode) prompt $($results.Count)/$($prompts.Count): $($r.tokens.Count) tokens"
}

$results | Set-Content -LiteralPath $OutFile -Encoding ASCII

Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
$conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3
"done: $Mode -> $OutFile"
