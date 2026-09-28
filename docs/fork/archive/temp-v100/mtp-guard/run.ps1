$ErrorActionPreference = 'Stop'
$bin   = 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\llama-server.exe'
$model = '<models>\Qwen3.5-0.8B-MTP\Qwen3.5-0.8B-UD-Q4_K_XL.gguf'
$dir   = '<TEMP>\v100\mtp-guard'
$port  = 18199
New-Item -ItemType Directory -Force -Path $dir | Out-Null

function Run-Case([string]$name, [int]$ngl, [bool]$replay) {
    $err = Join-Path $dir "$name.err.log"
    $out = Join-Path $dir "$name.out.log"
    Remove-Item $err, $out -Force -ErrorAction SilentlyContinue

    if ($replay) { $env:GGML_CUDA_GDN_REPLAY = '1' } else { Remove-Item Env:GGML_CUDA_GDN_REPLAY -ErrorAction SilentlyContinue }
    $env:CUDA_VISIBLE_DEVICES = '1'

    $argList = @('-m', $model, '-ngl', "$ngl", '-c', '2048', '-fa', 'on',
                 '--host', '127.0.0.1', '--port', "$port",
                 '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '--no-warmup')
    $p = Start-Process -FilePath $bin -ArgumentList $argList -RedirectStandardError $err -RedirectStandardOutput $out -PassThru

    $ok = $false
    for ($i = 0; $i -lt 180; $i++) {
        Start-Sleep -Milliseconds 1000
        if ($p.HasExited) { break }
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $ok = $true; break }
        } catch { }
    }

    $text = ''
    if ($ok) {
        $body = @{ messages = @(@{ role = 'user'; content = 'Count from 1 to 8, one number per line.' });
                    temperature = 0; max_tokens = 24; seed = 123 } | ConvertTo-Json -Depth 6
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/chat/completions" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 300
            $text = $resp.choices[0].message.content
        } catch { $text = "REQUEST_FAILED: $_" }
    }

    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    Start-Sleep -Seconds 1

    $exited   = $p.HasExited
    $exitCode = if ($exited) { $p.ExitCode } else { $null }
    $guard    = (Select-String -Path $err -Pattern 'ReplaySSM enabled|disabling replay' -ErrorAction SilentlyContinue | ForEach-Object { $_.Line.Trim() }) -join ' || '
    $mtp      = ((Select-String -Path $err -Pattern "adding speculative implementation 'draft-mtp'" -ErrorAction SilentlyContinue) | Measure-Object).Count
    $abort    = ((Select-String -Path $err -Pattern 'GGML_ABORT|terminate called|abort\(\)|0xC0000005' -ErrorAction SilentlyContinue) | Measure-Object).Count

    return [pscustomobject]@{
        case = $name; ngl = $ngl; replay = $replay; served = $ok
        mtp_lines = $mtp; aborts = $abort
        guard = $guard
        text_hash = (Get-FileHash -InputStream ([IO.MemoryStream]::new([Text.Encoding]::UTF8.GetBytes($text))) -Algorithm SHA256).Hash.Substring(0,16)
        text = $text
    }
}

$results = @()
$results += Run-Case 'p1-env-on-ngl99'  99 $true
$results += Run-Case 'p1-env-off-ngl99' 99 $false
$results += Run-Case 'p2-env-on-ngl5'    5 $true
$results += Run-Case 'p2-env-off-ngl5'   5 $false
$results += Run-Case 'p3-env-on-ngl0'    0 $true
$results += Run-Case 'p3-env-off-ngl0'   0 $false

$results | ForEach-Object {
    '[{0}] ngl={1} replay={2} served={3} mtp={4} aborts={5} hash={6}' -f $_.case, $_.ngl, $_.replay, $_.served, $_.mtp_lines, $_.aborts, $_.text_hash
    '    guard: ' + $_.guard
    '    text : ' + ($_.text -replace "`r?`n", ' \n ')
}
''
'PAIR CHECKS:'
'p1 (ngl 99): ' + $(if ($results[0].text_hash -eq $results[1].text_hash) { 'MATCH' } else { 'DIFF' })
'p2 (ngl  5): ' + $(if ($results[2].text_hash -eq $results[3].text_hash) { 'MATCH' } else { 'DIFF' })
'p3 (ngl  0): ' + $(if ($results[4].text_hash -eq $results[5].text_hash) { 'MATCH' } else { 'DIFF' })
