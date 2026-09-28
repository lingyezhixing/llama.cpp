$ErrorActionPreference = "Continue"
$t = "$env:TEMP\v100"
$dep = "D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll"
$bench = "D:\LLM\Backend\llama.cpp-my\llama-bench.exe"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe"
$env:CUDA_VISIBLE_DEVICES = "1"

$appArgs = @('-m', $m, '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-ub', '512', '-p', '4096', '-n', '0', '-d', '8192', '-r', '1', '-o', 'csv')

$configs = @(
    @{ n = 'cfgB-c64';      dll = 'cfgB' },
    @{ n = 'cfgE-w8';       dll = 'cfgE' }
)

foreach ($c in $configs) {
    Copy-Item "$t\ggml-cuda-T16-$($c.dll).dll" $dep -Force
    $rep = "$t\t16_$($c.n).nsys-rep"
    if (Test-Path $rep) { Remove-Item $rep -Force }
    cmd /c "`"$nsys`" profile --trace=cuda --cuda-event-trace=false --cuda-graph-trace=node --force-overwrite=true -o `"$t\t16_$($c.n)`" `"$bench`" $($appArgs -join ' ') > `"$t\t16_$($c.n).out.txt`" 2>&1"
    $ok = if (Test-Path $rep) { "OK" } else { "NO-REPORT" }
    Write-Output "$($c.n): $ok"
    Start-Sleep -Seconds 10
}
Write-Output "DONE; restoring BASE"
Copy-Item "$t\ggml-cuda-T12-BASE.dll" $dep -Force
