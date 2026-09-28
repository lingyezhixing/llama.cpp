$ErrorActionPreference = 'Stop'

$repo = 'D:\LLM\Backend\src\llama.cpp-my'
$dstRoot = Join-Path $repo 'docs\fork\archive'

function Copy-Selection {
    param(
        [string] $src,
        [string] $dst,
        [string[]] $includeExt,
        [int] $textCapKB,
        [int] $logCapKB,
        [string[]] $excludeDirs,
        [string[]] $bigExceptions
    )

    New-Item -ItemType Directory -Force -Path $dst | Out-Null

    $copied = 0
    $bytes  = 0
    $skipped = 0

    foreach ($f in (Get-ChildItem -LiteralPath $src -Recurse -File -Force)) {
        $rel = $f.FullName.Substring($src.Length + 1)

        $excluded = $false
        foreach ($d in $excludeDirs) {
            if ($rel -like "$d*") { $excluded = $true; break }
        }
        if ($excluded) { $skipped++; continue }

        $ext = $f.Extension.ToLower()
        $keep = $false

        if ($includeExt -contains $ext) {
            $keep = $true
        } elseif ($ext -eq '.txt') {
            $keep = $f.Length -le ($textCapKB * 1KB)
        } elseif ($ext -eq '.log' -or $ext -eq '.err') {
            $keep = $f.Length -le ($logCapKB * 1KB)
        }

        if ($keep -and $f.Length -gt ($textCapKB * 1KB) -and -not ($bigExceptions -contains $rel)) {
            $keep = $false
        }

        if (-not $keep) { $skipped++; continue }

        $target = Join-Path $dst $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
        Copy-Item -LiteralPath $f.FullName -Destination $target -Force
        $copied++
        $bytes += $f.Length
    }

    "{0}: copied {1} files, {2:N2} MB, skipped {3}" -f $dst, $copied, ($bytes / 1MB), $skipped
}

$textExt = @('.md', '.patch', '.diff', '.py', '.ps1', '.cmd', '.cu', '.csv', '.jsonl', '.json', '.png')

Copy-Selection `
    -src 'D:\LLM\Backend\v100-collab' `
    -dst (Join-Path $dstRoot 'v100-collab') `
    -includeExt $textExt `
    -textCapKB 256 `
    -logCapKB 64 `
    -excludeDirs @() `
    -bigExceptions @('artifacts\t32-stage4-soak-30.txt', 'artifacts\t32-merge-logs\srv-soak-final-err.txt')

Copy-Selection `
    -src '<TEMP>\v100' `
    -dst (Join-Path $dstRoot 'temp-v100') `
    -includeExt @('.md', '.ps1', '.cmd', '.py', '.json', '.jsonl', '.csv') `
    -textCapKB 256 `
    -logCapKB 64 `
    -excludeDirs @('t32-27b\', 't32-media-e2e\', 't32-stage3\', 't32-tree-smoke\', 'slots\', 'slots3\', 'slots4\', 'llmdb\', 'cleanup\') `
    -bigExceptions @()

$mtpSrc = (Get-ChildItem -LiteralPath 'D:\LLM\Backend' -Directory -Force | Where-Object { $_.Name -like 'MTP*' } | Select-Object -First 1).FullName
Write-Output ("mtp source: {0}" -f $mtpSrc)

Copy-Selection `
    -src $mtpSrc `
    -dst (Join-Path $dstRoot 'mtp-sealed-20260923') `
    -includeExt @('.md', '.py', '.ps1', '.cmd', '.cu', '.patch') `
    -textCapKB 256 `
    -logCapKB 64 `
    -excludeDirs @() `
    -bigExceptions @()

Write-Output '=== total size ==='
$all = Get-ChildItem -LiteralPath $dstRoot -Recurse -File -Force
"{0} files, {1:N2} MB" -f $all.Count, (($all | Measure-Object Length -Sum).Sum / 1MB)
Write-Output '=== top dirs ==='
Get-ChildItem -LiteralPath $dstRoot -Directory | ForEach-Object {
    $s = (Get-ChildItem -LiteralPath $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum
    "{0,-24} {1,8:N2} MB" -f $_.Name, ($s / 1MB)
}
