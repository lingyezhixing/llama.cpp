param([string]$depth = "0,4096,32768", [string]$which = "all", [int]$r = 2)
$ErrorActionPreference = "Continue"
$env:CUDA_VISIBLE_DEVICES = "1"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$exe = "D:\LLM\Backend\llama.cpp-my\llama-bench.exe"
$log = "$env:TEMP\v100\t20_perf.txt"

$configs = @(
    @{ n = "base"; env = @{} },
    @{ n = "gdn";  env = @{ GGML_CUDA_GDN_VEC4 = "1" } },
    @{ n = "pb";   env = @{ GGML_CUDA_FATTN_PB_FORCE = "1" } },
    @{ n = "both"; env = @{ GGML_CUDA_GDN_VEC4 = "1"; GGML_CUDA_FATTN_PB_FORCE = "1" } },
    @{ n = "all3"; env = @{ GGML_CUDA_GDN_VEC4 = "1"; GGML_CUDA_FATTN_PB_FORCE = "1"; GGML_CUDA_FA_SMALL_BATCH_VEC = "1" } }
)
if ($which -ne "all") {
    $configs = $configs | Where-Object { $_.n -in $which.Split(",") }
}

foreach ($c in $configs) {
    foreach ($k in @("GGML_CUDA_GDN_VEC4","GGML_CUDA_FATTN_PB_FORCE","GGML_CUDA_FA_SMALL_BATCH_VEC")) {
        [Environment]::SetEnvironmentVariable($k, $null)
    }
    foreach ($k in $c.env.Keys) { [Environment]::SetEnvironmentVariable($k, $c.env[$k]) }

    $out = & $exe -m $m -ngl 99 -fa on -ctv q8_0 -ub 512 -p 0 -n 128 -d $depth -r $r -o csv 2>$null | Where-Object { $_ -match '^"' }
    if (-not $out) { Write-Output "$($c.n) FAIL"; continue }
    foreach ($row in $out) {
        $f = $row -split ','
        $n = $f[34].Trim('"'); $d = $f[35].Trim('"'); $ts = $f[39].Trim('"')
        $line = "$($c.n) tg$n d$d = $ts"
        Add-Content $log $line
        Write-Output $line
    }
}
foreach ($k in @("GGML_CUDA_GDN_VEC4","GGML_CUDA_FATTN_PB_FORCE","GGML_CUDA_FA_SMALL_BATCH_VEC")) {
    [Environment]::SetEnvironmentVariable($k, $null)
}
Write-Output "PERF DONE"
