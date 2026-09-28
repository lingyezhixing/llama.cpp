$t = "$env:TEMP\v100"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$env:CUDA_VISIBLE_DEVICES = "1"
$prompt = "Write a short story about a robot who learns to paint:"
foreach ($arm in @(@('OURS', 'D:\LLM\Backend\llama.cpp-my'), @('STOCK', 'D:\LLM\Backend\llama.cpp'))) {
    $out = "$t\t19_gen_$($arm[0]).txt"
    & "$($arm[1])\llama-completion.exe" -m $m -ngl 99 -fa on -ctv q8_0 -ub 512 -p $prompt -n 200 --seed 42 --temp 0 2>&1 | Out-File -Encoding utf8 $out
    Write-Output "$($arm[0]) gen written -> $out"
}
