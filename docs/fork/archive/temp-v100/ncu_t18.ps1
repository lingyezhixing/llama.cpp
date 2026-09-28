$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "1"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.1\ncu.bat"
$dir = "$env:TEMP\v100"
Set-Location $dir
& $ncu --target-processes all --clock-control none --cache-control none `
    --kernel-name regex:flash_attn_ext_f16 --launch-skip 2 --launch-count 1 `
    --section SpeedOfLight --section Occupancy --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis `
    --log-file "$dir\ncu_t18.log" `
    "$dir\t18_fa_harness.exe" 5 35072 *> "$dir\ncu_t18_out.txt"
"done $(Get-Date -Format o)" | Out-File "$dir\ncu_t18_done.txt"
