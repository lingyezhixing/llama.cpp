param([string]$ub = "512", [switch]$sanity)
$ErrorActionPreference = "Continue"
$t = "$env:TEMP\v100"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$env:CUDA_VISIBLE_DEVICES = "1"
$log = "$t\t19_matrix_ub$ub.txt"

$arms = @(
    @{ n = 'OURS';  dir = 'D:\LLM\Backend\llama.cpp-my'; sha = '7F1B9B2403438803' },
    @{ n = 'STOCK'; dir = 'D:\LLM\Backend\llama.cpp';    sha = '976E2CABF9EADC7D' }
)

function Run-Bench($arm, $tag, $extra, $r) {
    $h = (Get-FileHash "$($arm.dir)\ggml-cuda.dll" -Algorithm SHA256).Hash.Substring(0,16)
    if ($h -ne $arm.sha) { Write-Output "SHA MISMATCH $($arm.n): $h"; return }
    $out = & "$($arm.dir)\llama-bench.exe" -m $m -ngl 99 -fa on -ctv q8_0 -ub $ub @extra -r $r -o csv 2>$null | Where-Object { $_ -match '^"' }
    if (-not $out) { Add-Content $log "$tag $($arm.n) ub$ub FAIL/OOM"; Write-Output "$tag $($arm.n) ub$ub FAIL/OOM"; return }
    foreach ($row in $out) {
        $f = $row -split ','
        $p = $f[33].Trim('"'); $n = $f[34].Trim('"'); $d = $f[35].Trim('"'); $ts = $f[39].Trim('"')
        $line = "$tag $($arm.n) ub$ub pp$p tg$n d$d = $ts"
        Add-Content $log $line; Write-Output $line
    }
}

if ($sanity) {
    Run-Bench $arms[0] "sanity" @('-p','4096','-n','0','-d','32768') 1
    Run-Bench $arms[1] "sanity" @('-p','4096','-n','0','-d','32768') 1
    Write-Output "SANITY DONE"; exit
}

Run-Bench $arms[0] "short"    @('-p','512,4096,8192','-n','0') 3
Run-Bench $arms[1] "short"    @('-p','512,4096,8192','-n','0') 3
Run-Bench $arms[0] "short-tg" @('-p','0','-n','128','-d','0,4096,8192') 3
Run-Bench $arms[1] "short-tg" @('-p','0','-n','128','-d','0,4096,8192') 3

foreach ($round in 1..2) {
    $order = if ($round % 2 -eq 1) { @(0,1) } else { @(1,0) }
    foreach ($i in $order) { Run-Bench $arms[$i] "r${round}-pp32768" @('-p','32768','-n','0') 2 }
    foreach ($i in $order) { Run-Bench $arms[$i] "r${round}-tg32768" @('-p','0','-n','128','-d','32768') 2 }
    foreach ($i in $order) { Run-Bench $arms[$i] "r${round}-pp131072" @('-p','131072','-n','0') 2 }
    foreach ($i in $order) { Run-Bench $arms[$i] "r${round}-tg131072" @('-p','0','-n','128','-d','131072') 2 }
}
Write-Output "MATRIX ub$ub DONE"
