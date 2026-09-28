param(
    [string]$OutDir = "$env:TEMP\v100\t20_quality",
    [string]$MdDir = "D:\LLM\Backend",
    [string]$QuestionsFile = "$env:TEMP\v100\t20_qtest.json"
)
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Path $MdDir -Force | Out-Null

$questions = Get-Content -LiteralPath $QuestionsFile -Raw -Encoding UTF8 | ConvertFrom-Json

$arms = @(
    @{ id = "stock-nomtp"; file = "原版-nomtp.md";    title = "原版（上游）· 不开 MTP"; build = "上游 llama.cpp @ e6ab7c1a4, ggml-cuda 976E2CAB"; spec = "关闭" },
    @{ id = "stock-mtp";   file = "原版-mtp.md";      title = "原版（上游）· 开 MTP";   build = "上游 llama.cpp @ e6ab7c1a4, ggml-cuda 976E2CAB"; spec = "draft-mtp, n-max 3" },
    @{ id = "fix-nomtp";   file = "修复后-nomtp.md";  title = "修复后 · 不开 MTP";      build = "修复分支, ggml-cuda 8093C771 + 修复开关"; spec = "关闭" },
    @{ id = "fix-mtp";     file = "修复后-mtp.md";    title = "修复后 · 开 MTP";        build = "修复分支, ggml-cuda 8093C771 + 修复开关"; spec = "draft-mtp, n-max 3" }
)

foreach ($a in $arms) {
    $stats = $null
    $spath = Join-Path $OutDir "stats_$($a.id).json"
    if (Test-Path -LiteralPath $spath) { $stats = Get-Content -LiteralPath $spath -Raw -Encoding UTF8 | ConvertFrom-Json }

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("# $($a.title)")
    $lines.Add("")
    $lines.Add("- 构建: $($a.build)")
    $lines.Add("- 投机解码: $($a.spec)")
    $lines.Add("- 采样: greedy (temperature 0, top-k 1, seed 42)")
    $lines.Add("- 上下文: 131072 (最大输出不限); 聊天模板: 模型自带 (GGUF 内嵌)")
    $lines.Add("- 说明: 仅保留最终回答 (thinking 已剥离); 推理长度 = thinking 部分 token 数")
    $lines.Add("")
    $lines.Add("## 汇总")
    $lines.Add("")
    $lines.Add("| 题目 | 推理 (token) | 回答 (token) | 总输出 (token) | 用时 (s) |")
    $lines.Add("|---|---|---|---|---|")
    foreach ($q in $questions) {
        $s = $null
        if ($stats -and ($stats.PSObject.Properties.Name -contains $q.id)) { $s = $stats.$($q.id) }
        if ($s) {
            $notes = ""
            if ($s.finish_reason -ne "stop") { $notes = " *(" + $s.finish_reason + ")*" }
            $lines.Add("| $($q.id) | $($s.reasoning_tokens) | $($s.answer_tokens) | $($s.completion_tokens)$notes | $($s.wall_s) |")
        } else {
            $lines.Add("| $($q.id) | - | - | - | - |")
        }
    }
    $lines.Add("")
    foreach ($q in $questions) {
        $s = $null
        if ($stats -and ($stats.PSObject.Properties.Name -contains $q.id)) { $s = $stats.$($q.id) }
        $lines.Add("## $($q.id)")
        $lines.Add("")
        if ($s) { $lines.Add("推理 $($s.reasoning_tokens) token / 回答 $($s.answer_tokens) token / 总 $($s.completion_tokens) token (finish=$($s.finish_reason))") } else { $lines.Add("(无指标)") }
        $lines.Add("")
        $lines.Add("**问题**")
        $lines.Add("")
        $lines.Add($q.question)
        $lines.Add("")
        $lines.Add("**回答**")
        $lines.Add("")
        $path = Join-Path $OutDir "raw_$($a.id)_$($q.id).json"
        if (Test-Path -LiteralPath $path) {
            $d = Get-Content -LiteralPath $path -Raw -Encoding UTF8 | ConvertFrom-Json
            $content = $d.choices[0].message.content
            if ([string]::IsNullOrWhiteSpace($content)) { $content = "(空回答)" }
            $lines.Add($content)
        } else {
            $lines.Add("(缺失: 未生成)")
        }
        $lines.Add("")
        $lines.Add("---")
        $lines.Add("")
    }
    $outPath = Join-Path $MdDir $a.file
    Set-Content -LiteralPath $outPath -Value ($lines -join "`n") -Encoding UTF8
    Write-Output "wrote $outPath ($((Get-Item $outPath).Length) bytes)"
}
Write-Output "RENDER DONE"
