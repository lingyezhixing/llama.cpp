$ErrorActionPreference = 'Continue'

$exe = 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe'
$models = [ordered]@{
  '2B' = '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf'
  '3B' = '<models>\Qwen2.5-Coder-3B-IQ4_XS.gguf'
}
$variants = @(
  @{ name='np1-f16';      args=@('-np','1','-c','512') },
  @{ name='np1-vq8';      args=@('-np','1','-c','512','-ctv','q8_0') },
  @{ name='np1-vq4';      args=@('-np','1','-c','512','-ctv','q4_0') },
  @{ name='np1-kvq8-vq8'; args=@('-np','1','-c','512','-ctk','q8_0','-ctv','q8_0') },
  @{ name='np3-f16';      args=@('-np','3','-c','512') },
  @{ name='np3-vq8';      args=@('-np','3','-c','512','-ctv','q8_0') },
  @{ name='np3-vq4';      args=@('-np','3','-c','512','-ctv','q4_0') },
  @{ name='np3-vq8-kvu';  args=@('-np','3','-c','512','-ctv','q8_0','-kvu') },
  @{ name='np3-vq4-kvu';  args=@('-np','3','-c','512','-ctv','q4_0','-kvu') }
)

$env:CUDA_VISIBLE_DEVICES = '0'

$outDir = '<TEMP>\v100\t32-matrix'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$archive = 'D:\LLM\Backend\v100-collab\artifacts\t32-stage1-correctness-matrix.txt'
$enc = New-Object System.Text.UTF8Encoding($false)
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine('T32 stage 0+1: correctness matrix (small models, CUDA device 0)')
[void]$sb.AppendLine('date: 2026-09-27')
[void]$sb.AppendLine("exe: $exe")
[void]$sb.AppendLine('')

$summary = @()
foreach ($m in $models.GetEnumerator()) {
  foreach ($v in $variants) {
    $tag = "$($m.Key)-$($v.name)"
    $out = Join-Path $outDir "$tag.out"
    $err = Join-Path $outDir "$tag.err"

    $argv = @('-m', $m.Value, '-ngl', '99', '-fa', 'on', '--mode', 'correctness') + $v.args
    $p = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -Wait -PassThru -RedirectStandardOutput $out -RedirectStandardError $err

    $checks = @(Select-String -Path $err -Pattern '^\[t32')
    $pass = @($checks | Where-Object { $_.Line -match ' PASS$' }).Count
    $fail = @($checks | Where-Object { $_.Line -match ' FAIL$' }).Count

    $line = "{0,-20} exit={1} pass={2} fail={3}" -f $tag, $p.ExitCode, $pass, $fail
    Write-Output $line
    $summary += $line

    [void]$sb.AppendLine("=== $tag : exit=$($p.ExitCode) pass=$pass fail=$fail ===")
    [void]$sb.AppendLine("cmd: test-t32-range.exe -m $($m.Value) -ngl 99 -fa on --mode correctness $($v.args -join ' ')")
    [void]$sb.AppendLine('')
    [void]$sb.AppendLine([System.IO.File]::ReadAllText($err, $enc).TrimEnd())
    [void]$sb.AppendLine('')
  }
}

[void]$sb.AppendLine('=== summary ===')
foreach ($l in $summary) { [void]$sb.AppendLine($l) }

[System.IO.File]::WriteAllText($archive, $sb.ToString(), $enc)
[System.IO.File]::WriteAllLines((Join-Path $outDir 'summary.txt'), $summary, $enc)
Write-Output "archive: $archive"
