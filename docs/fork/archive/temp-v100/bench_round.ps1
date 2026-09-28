param([string]$label)
$env:CUDA_VISIBLE_DEVICES = "1"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$bench = "D:\LLM\Backend\llama.cpp-my\llama-bench.exe"
$f = { param($pat) & $bench -m $m -ngl 99 -fa on -ctv q8_0 -ub 512 @pat 2>&1 | Select-String -Pattern "\|\s+(pp|tg)" | ForEach-Object { $_.Line.Trim() } }
Write-Output "### $label : depth32k"
& $f @("-p","4096","-d","32768","-n","0","-r","2")
Write-Output "### $label : pp32768"
& $f @("-p","32768","-n","0","-r","2")
Write-Output "### $label : short + tg128"
& $f @("-p","512,4096,8192","-n","128","-r","3")
Write-Output "### $label : ub2048 spot"
& "D:\LLM\Backend\llama.cpp-my\llama-bench.exe" -m $m -ngl 99 -fa on -ctv q8_0 -ub 2048 -p 4096 -n 0 -r 2 2>&1 | Select-String -Pattern "\|\s+pp" | ForEach-Object { $_.Line.Trim() }
