$ErrorActionPreference = "Continue"
$t = "$env:TEMP\v100"
$dep = "D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll"
$bench = "D:\LLM\Backend\llama.cpp-my\llama-bench.exe"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe"
$env:CUDA_VISIBLE_DEVICES = "1"
$log = "$t\t16_sweep2_log.txt"
if (Test-Path $log) { Remove-Item $log }

$appArgs = @('-m', $m, '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-ub', '512', '-p', '4096', '-n', '0', '-d', '8192', '-r', '1', '-o', 'csv')

$configs = @(
    @{ n = 'pb4-grid768';        dll = 'env';  e = @{ GGML_CUDA_FATTN_PB = '4' } },
    @{ n = 'cfgA-Qreg';          dll = 'cfgA'; e = @{} },
    @{ n = 'cfgB-c64';           dll = 'cfgB'; e = @{} },
    @{ n = 'cfgC-c64-fa64';      dll = 'cfgC'; e = @{} },
    @{ n = 'cfgA-Qreg-pb2';      dll = 'cfgA'; e = @{ GGML_CUDA_FATTN_PB = '2' } }
)

foreach ($c in $configs) {
    Copy-Item "$t\ggml-cuda-T16-$($c.dll).dll" $dep -Force
    foreach ($k in $c.e.Keys) { Set-Item "Env:$k" $c.e[$k] }
    $rep = "$t\t16_$($c.n).nsys-rep"
    if (Test-Path $rep) { Remove-Item $rep -Force }
    $outfile = "$t\t16_$($c.n).out.txt"
    cmd /c "`"$nsys`" profile --trace=cuda --cuda-event-trace=false --cuda-graph-trace=node --force-overwrite=true -o `"$t\t16_$($c.n)`" `"$bench`" $($appArgs -join ' ') > `"$outfile`" 2>&1"
    foreach ($k in $c.e.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
    $ok = if (Test-Path $rep) { "OK" } else { "NO-REPORT" }
    $s = "$($c.n): $ok"
    Add-Content -Path $log -Value $s
    Write-Output $s
}
Write-Output "SWEEP2 DONE"
