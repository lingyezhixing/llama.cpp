param([string]$phase = "1")
$ErrorActionPreference = "Continue"
$t = "$env:TEMP\v100"
$dep = "D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll"
$bench = "D:\LLM\Backend\llama.cpp-my\llama-bench.exe"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$env:CUDA_VISIBLE_DEVICES = "1"
$log = "$t\t12t16_accept_log.txt"

$dlls = @{ A = 'ggml-cuda-T12-BASE.dll'; B = 'ggml-cuda-T12T16.dll' }
$base = @('-m', $m, '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0', '-ub', '512')

function Run-Point($cfg, $name, $extra) {
    Copy-Item "$t\$($dlls[$cfg])" $dep -Force
    $out = & $bench @base @extra -o csv 2>$null | Where-Object { $_ -match '^"' }
    foreach ($row in $out) {
        $f = $row -split ','
        $p = $f[33].Trim('"'); $n = $f[34].Trim('"'); $d = $f[35].Trim('"')
        $ts = [math]::Round([double]$f[39].Trim('"'), 2)
        $tag = if ($n -ne '0') { "tg$n" } else { "pp$p" }
        if ($d -ne '0') { $tag = "$tag@d$d" }
        $line = "phase$phase round=$($script:round) cfg=$cfg $name $tag = $ts"
        Add-Content -Path $log -Value $line
        Write-Output $line
    }
}

$short = @('-p', '512,4096,8192', '-n', '128', '-r', '3')
$ub2048 = @('-ub', '2048', '-p', '4096', '-n', '0', '-r', '2')
$d32k = @('-p', '4096', '-n', '0', '-d', '32768', '-r', '3')
$pp32k = @('-p', '32768', '-n', '0', '-r', '2')
$d128k = @('-p', '8192', '-n', '0', '-d', '131072', '-r', '1')

$plan = switch ($phase) {
    "1" { @{ rounds = 3; points = @(@{ n = 'short'; a = $short }, @{ n = 'ub2048'; a = $ub2048; rounds = 2 }) } }
    "2" { @{ rounds = 3; points = @(@{ n = 'depth32k'; a = $d32k }) } }
    "3" { @{ rounds = 3; points = @(@{ n = 'pp32768'; a = $pp32k }) } }
    "4" { @{ rounds = 2; points = @(@{ n = 'd128k'; a = $d128k }) } }
}

foreach ($p in $plan.points) {
    $r = if ($p.ContainsKey('rounds')) { $p.rounds } else { $plan.rounds }
    foreach ($round in 1..$r) {
        $script:round = $round
        $order = if ($round % 2 -eq 1) { @('A', 'B') } else { @('B', 'A') }
        foreach ($cfg in $order) { Run-Point $cfg $p.n $p.a }
    }
}
Write-Output "PHASE $phase DONE"
