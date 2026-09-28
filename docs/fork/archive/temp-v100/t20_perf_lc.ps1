param([string]$depth = "0,32768", [int]$r = 3)
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "1"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$exe = "D:\LLM\Backend\llama.cpp-my\llama-bench.exe"
$log = "$env:TEMP\v100\t20_perf_lc.txt"

function Run-Cfg($name, $fix) {
    if ($fix) {
        [Environment]::SetEnvironmentVariable("GGML_CUDA_GDN_VEC4", "1")
        [Environment]::SetEnvironmentVariable("GGML_CUDA_FA_SMALL_BATCH_VEC", "1")
    } else {
        [Environment]::SetEnvironmentVariable("GGML_CUDA_GDN_VEC4", $null)
        [Environment]::SetEnvironmentVariable("GGML_CUDA_FA_SMALL_BATCH_VEC", $null)
    }
    $out = & $exe -m $m -ngl 99 -fa on -ctv q8_0 -ub 512 -p 0 -n 128 -d $depth -r $r -o csv 2>$null | Where-Object { $_ -match '^"' }
    if (-not $out) { Write-Output "$name FAIL"; return }
    foreach ($row in $out) {
        $f = $row -split ','
        $d = $f[35].Trim('"'); $ts = $f[39].Trim('"')
        $line = "$name tg128 d$d = $ts"
        Add-Content $log $line
        Write-Output $line
    }
}

Run-Cfg "base" $false
Run-Cfg "lc"   $true
Run-Cfg "lc"   $true
Run-Cfg "base" $false
[Environment]::SetEnvironmentVariable("GGML_CUDA_GDN_VEC4", $null)
[Environment]::SetEnvironmentVariable("GGML_CUDA_FA_SMALL_BATCH_VEC", $null)
Write-Output "LC PERF DONE"
