$out = "$env:TEMP\v100\t30_ab.jsonl"
Remove-Item $out -ErrorAction SilentlyContinue
$configs = @(
    @{ tag = 'A0';  vec = $null; pb = $null;  nbatch = $null  },
    @{ tag = 'A3';  vec = '4';   pb = '160';  nbatch = '1024' },
    @{ tag = 'A2';  vec = $null; pb = '160';  nbatch = '1024' },
    @{ tag = 'A1';  vec = '4';   pb = $null;  nbatch = $null  },
    @{ tag = 'A0b'; vec = $null; pb = $null;  nbatch = $null  }
)
foreach ($c in $configs) {
    Remove-Item Env:\GGML_CUDA_FA_VEC_VERIFY -ErrorAction SilentlyContinue
    Remove-Item Env:\GGML_CUDA_FATTN_VERIFY_PB -ErrorAction SilentlyContinue
    Remove-Item Env:\GGML_CUDA_FATTN_VERIFY_NBATCH -ErrorAction SilentlyContinue
    if ($c.vec)    { $env:GGML_CUDA_FA_VEC_VERIFY      = $c.vec }
    if ($c.pb)     { $env:GGML_CUDA_FATTN_VERIFY_PB    = $c.pb }
    if ($c.nbatch) { $env:GGML_CUDA_FATTN_VERIFY_NBATCH = $c.nbatch }
    "===== config $($c.tag) vec=$($c.vec) pb=$($c.pb) nbatch=$($c.nbatch) $(Get-Date -Format HH:mm:ss) ====="
    & powershell -NoProfile -ExecutionPolicy Bypass -File "$env:TEMP\v100\t30_ab.ps1" -Tag $c.tag -Out $out 2>&1 |
        ForEach-Object { $_ }
}
"===== matrix done $(Get-Date -Format HH:mm:ss) ====="
Remove-Item Env:\GGML_CUDA_FA_VEC_VERIFY, Env:\GGML_CUDA_FATTN_VERIFY_PB, Env:\GGML_CUDA_FATTN_VERIFY_NBATCH -ErrorAction SilentlyContinue
