$ErrorActionPreference = "Continue"
$t = "$env:TEMP\v100"
$dep = "D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll"
$bench = "D:\LLM\Backend\llama.cpp-my\llama-bench.exe"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$nsys = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe"
$env:CUDA_VISIBLE_DEVICES = "1"
$log = "$t\t16_sweep_log.txt"
if (Test-Path $log) { Remove-Item $log }

$appArgs = @('-m', $m, '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-ub', '512', '-p', '4096', '-n', '0', '-d', '8192', '-r', '1', '-o', 'csv')

$configs = @(
    @{ n = 'base-grid80';           dll = 'env';  e = @{} },
    @{ n = 'blocks192';             dll = 'env';  e = @{ GGML_CUDA_FATTN_BLOCKS = '192' } },
    @{ n = 'blocks96';              dll = 'env';  e = @{ GGML_CUDA_FATTN_BLOCKS = '96' } },
    @{ n = 'blocks160';             dll = 'env';  e = @{ GGML_CUDA_FATTN_BLOCKS = '160' } },
    @{ n = 'pb2-grid384';           dll = 'env';  e = @{ GGML_CUDA_FATTN_PB = '2' } },
    @{ n = 'pb4-grid768';           dll = 'env';  e = @{ GGML_CUDA_FATTN_PB = '4' } },
    @{ n = 'cfgA-combine64-Qreg';   dll = 'cfgA'; e = @{} },
    @{ n = 'cfgB-combine64';        dll = 'cfgB'; e = @{} },
    @{ n = 'cfgC-combine64-fa64';   dll = 'cfgC'; e = @{} }
)

foreach ($c in $configs) {
    Copy-Item "$t\ggml-cuda-T16-$($c.dll).dll" $dep -Force
    foreach ($k in $c.e.Keys) { Set-Item "Env:$k" $c.e[$k] }
    $rep = "$t\t16_$($c.n).nsys-rep"
    if (Test-Path $rep) { Remove-Item $rep -Force }
    $outfile = "$t\t16_$($c.n).out.txt"
    cmd /c "`"$nsys`" profile --trace=cuda --cuda-event-trace=false --cuda-graph-trace=node --force-overwrite=true -o `"$t\t16_$($c.n)`" `"$bench`" $($appArgs -join ' ') > `"$outfile`" 2>&1"
    foreach ($k in $c.e.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
    $tps = 'n/a'
    if (Test-Path $outfile) {
        $row = (Get-Content $outfile | Where-Object { $_ -match '^"' } | Select-Object -Last 1)
        if ($row) { $tps = ($row -split ',')[39].Trim('"') }
    }
    $ok = if (Test-Path $rep) { "OK" } else { "NO-REPORT" }
    $s = "$($c.n): $ok pp4096@d8192=$tps t/s"
    Add-Content -Path $log -Value $s
    Write-Output $s
}
Write-Output "SWEEP DONE"
