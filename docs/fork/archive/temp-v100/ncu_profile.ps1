$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "1"
$ncu   = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.1\ncu.bat"
$bench = "D:\LLM\Backend\llama.cpp-my\llama-bench.exe"
$model = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$dir   = "$env:TEMP\v100"
Remove-Item "$dir\ncu_done.txt" -ErrorAction SilentlyContinue

# run A: duration-only breakdown, short context (ub512)
& $ncu --target-processes all --metrics gpu__time_duration.sum --launch-count 4000 --csv --log-file "$dir\ncu_A_pp512.csv" $bench -m $model -p 512 -n 0 -ub 512 -fa on -ctv q8_0 -r 1 *> "$dir\ncu_A_stdout.txt"

# run B: duration-only breakdown, long context (depth 16k, ub512)
& $ncu --target-processes all --metrics gpu__time_duration.sum --launch-count 6000 --csv --log-file "$dir\ncu_B_pp512d16k.csv" $bench -m $model -p 512 -n 0 -ub 512 -d 16384 -fa on -ctv q8_0 -r 1 *> "$dir\ncu_B_stdout.txt"

# run C: efficiency metrics on the first ~500 kernels (warmup prefill region)
& $ncu --target-processes all --metrics gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed --launch-count 500 --csv --log-file "$dir\ncu_C_eff.csv" $bench -m $model -p 512 -n 0 -ub 512 -fa on -ctv q8_0 -r 1 *> "$dir\ncu_C_stdout.txt"

"done $(Get-Date -Format o)" | Out-File "$dir\ncu_done.txt"
