param(
    [string]$BinDir = 'D:\LLM\Backend\llama.cpp-t24',
    [string]$Model  = '<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf',
    [string]$Prompt = "$env:TEMP\v100\t20_prompt128k.txt",
    [int]$N = 150,
    [int]$Ctx = 135168,
    [string]$Tag = 'cli128k'
)

$ErrorActionPreference = 'Continue'
$env:CUDA_VISIBLE_DEVICES = '1'
$env:GGML_CUDA_GDN_REPLAY = '1'
$nsys = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe'
$rep  = "$env:TEMP\v100\t30_$Tag"
$out  = "$env:TEMP\v100\t30_${Tag}_cli.log"

$cargs = @('-m', $Model, '-f', $Prompt, '-n', "$N", '-ngl', '99', '-fa', 'on', '-ctv', 'q8_0',
           '-ub', '512', '-c', "$Ctx", '--seed', '42', '--temp', '0', '--ignore-eos',
           '-st', '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '--no-warmup')
$nargs = @('profile', '--trace=cuda', '--cuda-event-trace=false', '--sample=none', '--cpuctxsw=none',
           '-o', $rep, '--force-overwrite', 'true',
           (Join-Path $BinDir 'llama-cli.exe')) + $cargs

$t0 = Get-Date
& $nsys @nargs *> $out
"exit=$LASTEXITCODE elapsed=$([math]::Round(((Get-Date)-$t0).TotalSeconds,1))s"
if (Test-Path "$rep.nsys-rep") { "report: $((Get-Item "$rep.nsys-rep").Length) bytes" } else { "report MISSING" }
Get-Content $out | Select-String -Pattern 'Generation:|Prompt:|tokens per second|error' | ForEach-Object { $_.Line.Trim() }
