param(
  [string]$Tag,
  [int]$BatchA,
  [int]$NExtra,
  [string]$EnvVars = "",
  [string]$Extra = ""
)
$env:CUDA_VISIBLE_DEVICES = "1"
foreach ($kv in $EnvVars.Split(";")) {
  if ($kv -ne "") { $parts = $kv.Split("="); [Environment]::SetEnvironmentVariable($parts[0], $parts[1]) }
}
$exe = "D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\llama-batch-invariance.exe"
$m = "<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf"
$out = "$env:TEMP\v100\t20_binv_$Tag.txt"
$cmd = "$exe -m $m --device CUDA0 -ngl 99 --prefill 24 --batch-a $BatchA --nextra $NExtra --n-ctx 4096 --ctk f16 --ctv q8_0 --fa on $Extra > $out 2>&1"
cmd /c $cmd
Write-Output "=== $Tag (A=$BatchA B=$($BatchA+$NExtra) env='$EnvVars' extra='$Extra') ==="
Get-Content $out | Select-String -Pattern "^layer|^h_nextn|^logits|^layer inputs|^run A" | Select-Object -ExpandProperty Line
