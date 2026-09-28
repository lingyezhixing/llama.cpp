param([string]$stage = "all")
$ErrorActionPreference = "Continue"
$t = "$env:TEMP\v100"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$env:CUDA_VISIBLE_DEVICES = "1"
$dir = "D:\LLM\Backend\llama.cpp-my"
$log = "$t\t19l_accept.txt"

$arms = @(
    @{ n = 'T19';  dll = "$t\ggml-cuda-T19.dll"  },
    @{ n = 'T19L'; dll = "$t\ggml-cuda-T19L.dll" }
)

function Use-Dll($arm) {
    Copy-Item $arm.dll "$dir\ggml-cuda.dll" -Force
    return (Get-FileHash "$dir\ggml-cuda.dll" -Algorithm SHA256).Hash.Substring(0,16)
}

function Run-Bench($arm, $tag, $extra, $r) {
    $h = Use-Dll $arm
    $out = & "$dir\llama-bench.exe" -m $m -ngl 99 -fa on -ctv q8_0 -ub 512 @extra -r $r -o csv 2>$null | Where-Object { $_ -match '^"' }
    if (-not $out) { $l = "$tag $($arm.n) FAIL/OOM (dll $h)"; Add-Content $log $l; Write-Output $l; return }
    foreach ($row in $out) {
        $f = $row -split ','
        $p = $f[33].Trim('"'); $n = $f[34].Trim('"'); $d = $f[35].Trim('"'); $ts = $f[39].Trim('"')
        $l = "$tag $($arm.n) pp$p tg$n d$d = $ts"
        Add-Content $log $l; Write-Output $l
    }
}

if ($stage -in @('short','all')) {
    Run-Bench $arms[0] 'pp-short' @('-p','512,4096,8192','-n','0') 3
    Run-Bench $arms[1] 'pp-short' @('-p','512,4096,8192','-n','0') 3
    Run-Bench $arms[1] 'tg-short' @('-p','0','-n','128','-d','0,4096,8192') 3
    Run-Bench $arms[0] 'tg-short' @('-p','0','-n','128','-d','0,4096,8192') 3
}

if ($stage -in @('mid','all')) {
    foreach ($round in 1..2) {
        $order = if ($round % 2 -eq 1) { @(0,1) } else { @(1,0) }
        foreach ($i in $order) { Run-Bench $arms[$i] "r${round}-pp32768" @('-p','32768','-n','0') 2 }
        foreach ($i in $order) { Run-Bench $arms[$i] "r${round}-tg32768" @('-p','0','-n','128','-d','32768') 2 }
    }
}

if ($stage -in @('long','all')) {
    Run-Bench $arms[0] 'pp131072' @('-p','131072','-n','0') 2
    Run-Bench $arms[1] 'pp131072' @('-p','131072','-n','0') 2
    Run-Bench $arms[1] 'tg131072' @('-p','0','-n','128','-d','131072') 2
    Run-Bench $arms[0] 'tg131072' @('-p','0','-n','128','-d','131072') 2
}

if ($stage -in @('ppl','all')) {
    foreach ($i in @(0,1)) {
        Use-Dll $arms[$i] | Out-Null
        $ppl = & "$dir\llama-perplexity.exe" -m $m -ngl 99 -fa on -ctv q8_0 -c 512 --chunks 8 --seed 42 2>&1 | Select-String 'Final estimate' | Select-Object -Last 1
        $l = "ppl $($arms[$i].n) = $ppl"
        Add-Content $log $l; Write-Output $l
    }
}

Write-Output "STAGE $stage DONE"
