param([int]$r = 2)
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "1"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$exe = "D:\LLM\Backend\llama.cpp-my\llama-bench.exe"
$log = "$env:TEMP\v100\t20_perf_pb.txt"

function Run-Pb($pb) {
    if ($pb -eq 0) { [Environment]::SetEnvironmentVariable("GGML_CUDA_FATTN_PB_FORCE", $null) }
    else { [Environment]::SetEnvironmentVariable("GGML_CUDA_FATTN_PB_FORCE", "$pb") }
    $out = & $exe -m $m -ngl 99 -fa on -ctv q8_0 -ub 512 -p 0 -n 128 -d 32768 -r $r -o csv 2>$null | Where-Object { $_ -match '^"' }
    if (-not $out) { Write-Output "pb=$pb FAIL"; return }
    foreach ($row in $out) {
        $f = $row -split ','
        $ts = $f[39].Trim('"')
        $line = "pb=$pb tg128 d32768 = $ts"
        Add-Content $log $line
        Write-Output $line
    }
}

Run-Pb 0
Run-Pb 2
Run-Pb 4
Run-Pb 6
Run-Pb 8
Run-Pb 0
[Environment]::SetEnvironmentVariable("GGML_CUDA_FATTN_PB_FORCE", $null)
Write-Output "PB SWEEP DONE"
