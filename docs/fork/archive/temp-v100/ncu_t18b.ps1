$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "1"
$ncu = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.1\ncu.bat"
$dir = "$env:TEMP\v100"
Set-Location $dir
# profile the ncols=32 variant at grid=768 (PB=2): SKIP the first kernels so we catch a steady-state launch
& $ncu --target-processes all --clock-control none --cache-control none `
    --kernel-name regex:flash_attn_ext_f16 --launch-skip 25 --launch-count 1 `
    --section SpeedOfLight --section Occupancy --section WarpStateStats --section SchedulerStats --section MemoryWorkloadAnalysis `
    --log-file "$dir\ncu_t18b.log" `
    "$dir\t18_fa_nc32.exe" 20 35072 768 384 *> "$dir\ncu_t18b_out.txt"
"done $(Get-Date -Format o)" | Out-File "$dir\ncu_t18b_done.txt"
