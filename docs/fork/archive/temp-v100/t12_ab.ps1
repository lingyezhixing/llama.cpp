$ErrorActionPreference = "Stop"
$t = "$env:TEMP\v100"
$dep = "D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll"
$bench = "D:\LLM\Backend\llama.cpp-my\llama-bench.exe"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$env:CUDA_VISIBLE_DEVICES = "1"
$log = "$t\t12_ab_log.txt"
if (Test-Path $log) { Remove-Item $log }

function Run-Point($name, $args, $round, $cfg) {
    $vramMax = 0
    $job = Start-Job -ScriptBlock {
        $max = 0
        while ($true) {
            $v = (nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | Select-String "^1,") -replace '.*,\s*',''
            if ($v -and [int]$v -gt $max) { $max = [int]$v }
            Start-Sleep -Milliseconds 400
        }
    }
    $out = & $bench -m $m -ngl 99 -fa on -ctv q8_0 -ub 512 @args -o csv 2>&1
    Stop-Job $job; Remove-Job $job -Force
    $lines = @($out | Where-Object { $_ -match '^pp|^tg' })
    $summary = @()
    foreach ($l in $lines) {
        $f = $l -split ','
        $summary += ("{0}={1}" -f $f[34], $f[36])
    }
    $s = "round=$round cfg=$cfg point=$name :: " + ($summary -join "  ")
    Add-Content -Path $log -Value $s
    Write-Output $s
}

$points = @(
    @{ n = 'P1_short';  a = @('-p', '512,4096,8192', '-n', '128', '-r', '3') },
    @{ n = 'P2_d32k';   a = @('-p', '4096', '-n', '0', '-d', '32768', '-r', '3') },
    @{ n = 'P3_pp32k';  a = @('-p', '32768', '-n', '0', '-r', '2') },
    @{ n = 'P4_d128k';  a = @('-p', '8192', '-n', '0', '-d', '131072', '-r', '1') },
    @{ n = 'P5_ub2048'; a = @('-ub', '2048', '-p', '4096', '-n', '0', '-r', '2') }
)

foreach ($round in 1..3) {
    foreach ($cfg in @('BASE', 'KV')) {
        Copy-Item "$t\ggml-cuda-T12-$cfg.dll" $dep -Force
        foreach ($p in $points) {
            Run-Point $p.n $p.a $round $cfg
        }
    }
}
Write-Output "DONE"
