### Task 5: A/B 验收（校准 + 低重叠长会话 + 多短会话 + 3B 对照 + 反例 + 回归）

**Files:**
- Create: `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1`（频道产物）
- 产物：`t32-stage3-accept.txt`（各模式汇总）、`t32-stage3-calib.txt`、`t32-stage3-neg.txt`、`t32-stage3-3b.txt`（原始日志）

**Interfaces:**
- Consumes: Task 3/4 的日志契约 + `--tree-*` CLI。
- Produces: 验收结论（脚本内 PASS/FAIL 行）+ 原始日志归档。

- [ ] **Step 1: 写验收脚本（完整）**

用 `[System.IO.File]::WriteAllText(..., UTF8Encoding($false))` 写入 `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1`：

```powershell
param(
    [Parameter(Mandatory=$true)][string]$Mode,   # calib|ab|overlap|b|b3|neg|heal|ref
    [string]$Model = '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf',
    [int]$Port = 8933,
    [string]$OutDir = '<TEMP>\v100\t32-stage3'
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\LLM\Backend\src\llama.cpp-my'
$srv  = "$repo\build\bin\Release\llama-server.exe"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$env:CUDA_VISIBLE_DEVICES = '0'

$fails = 0
function Assert($cond, $what) {
    if ($cond) { Write-Output "PASS  $what" } else { Write-Output "FAIL  $what"; $script:fails++ }
}

# ---- prompt construction (target ~N tokens, measured via /tokenize) ----
function TokCount($text) {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType 'application/json' -Body (@{content=$text} | ConvertTo-Json -Compress)
    return $r.tokens.Count
}
function Filler($salt, $n_tok) {
    $unit = "$salt filler sentence for the kv tree acceptance test number "
    $ut = TokCount $unit
    $t = ''
    $cur = 0
    while ($cur -lt $n_tok) {
        $need = [int][math]::Ceiling(($n_tok - $cur) / $ut)
        $t += ($unit * $need)
        $cur = TokCount $t
    }
    return $t
}

function Start-Srv($tree, $ram, $diskdir, $idle = $true) {
    $args = @('-m',$Model,'-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','24576',
              '--port',"$Port",'--cache-ram','0',
              '--no-context-shift','--host','127.0.0.1')
    if ($idle) { $args += '--cache-idle-slots' } else { $args += '--no-cache-idle-slots' }
    if ($tree) {
        $args += @('--kv-tree','--tree-chunk','512','--tree-anchor-step','4096',
                   '--tree-ram',"$ram",'--tree-disk',$diskdir,'--tree-disk-limit','2048','--tree-debug')
    }
    $p = Start-Process -FilePath $srv -ArgumentList $args -NoNewWindow -PassThru `
         -RedirectStandardOutput "$OutDir\srv-$Mode-out.txt" -RedirectStandardError "$OutDir\srv-$Mode-err.txt"
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 1000
        try { Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -Method Get | Out-Null; return $p } catch {}
    }
    throw "server did not come up"
}

function Req($prompt, $cache) {
    $body = @{prompt=$prompt; n_predict=48; temperature=0; top_k=1; seed=42; cache_prompt=$cache; stream=$false}
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body ($body | ConvertTo-Json -Compress)
    return $r
}

function ReqDelim($prompt, $cache) {
    $body = @{prompt=$prompt; n_predict=32; temperature=0; top_k=1; seed=42; cache_prompt=$cache; stream=$false; message_delimiters=@(@{role='user'; delimiter='User:'})}
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -TimeoutSec 120 -Body ($body | ConvertTo-Json -Compress -Depth 4)
}

# ---- session types ----
# long sessions: shared system prompt only (low overlap -> f_keep < 0.5 -> tree park/restore)
$sys  = Filler 'system' 512
$baseA = Filler 'alpha' 16384
$baseB = Filler 'beta'  16384
# overlap sessions: share 12K (stock VRAM reuse; tree must stay out of the way)
$ovl  = Filler 'shared' 12288
$baseC = "$ovl" + (Filler 'gamma' 4096)
$baseD = "$ovl" + (Filler 'delta' 4096)
# short sessions for mode b
$sess = @()
foreach ($s in @('one','two','three','four')) { $sess += ("$sys" + (Filler $s 1536)) }

$results = @{}
$hashes  = @{}
function ContentHash($s) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $h = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($s))
    return ([System.BitConverter]::ToString($h)).Replace('-','').Substring(0,16)
}
function RunTurns($names, $bases, $rounds, $cache, $tag) {
    $hist = @{}
    foreach ($n in $names) { $hist[$n] = '' }
    for ($r = 1; $r -le $rounds; $r++) {
        for ($i = 0; $i -lt $names.Count; $i++) {
            $n = $names[$i]
            $prompt = "$sys`n" + $bases[$i] + $hist[$n] + "`nUser turn ${r}: continue`nAssistant:"
            $resp = Req $prompt $cache
            $hist[$n] += $resp.content
            $h = ContentHash $resp.content
            $hashes["$tag/$n/$r"] = $h
            $results["$tag/$n/$r"] = $resp.content
            Write-Output ("[$Mode/$tag] $n round ${r}: hash=$h prompt_n=$($resp.timings.prompt_n) pred=$($resp.timings.predicted_n) cached=$($resp.tokens_cached)")
        }
    }
}
function ComparePasses($names, $rounds) {
    for ($r = 1; $r -le $rounds; $r++) {
        foreach ($n in $names) {
            Assert ($hashes["tree/$n/$r"] -eq $hashes["full/$n/$r"]) "equal: $n round $r (tree reuse vs full prefill)"
        }
    }
}

switch ($Mode) {
    'ref' {
        $p = Start-Srv $false 0 '' $true
        RunTurns @('A','B') @($baseA,$baseB) 6 $false 'full'
        RunTurns @('C','D') @($baseC,$baseD) 3 $false 'full'
        RunTurns @('one','two','three','four') $sess 3 $false 'full'
        $p | Stop-Process -Force
    }
    'calib' {
        $dir = "$OutDir\tree-calib"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 4096 $dir
        RunTurns @('A','B') @($baseA,$baseB) 2 $true 'tree'
        $p | Stop-Process -Force
        $line = (Select-String -Path "$OutDir\srv-$Mode-err.txt" -Pattern 'kv tree: parked' | Select-Object -Last 1).Line
        Write-Output "CALIB LAST PARK: $line"
        $total = 0
        if ($line -match 'ram = (\d+) B, disk = (\d+) B') { $total = [int64]$matches[1] + [int64]$matches[2] }
        Write-Output "CALIB TOTAL BYTES AFTER 2 ROUNDS: $total"
        Assert ($total -gt 0) 'calib: tree stores data'
        Assert ((Select-String -Path "$OutDir\srv-$Mode-err.txt" -Pattern 'kv tree: restore miss').Count -ge 1) 'calib: cold restore miss is visible'
    }
    'ab' {
        $dir = "$OutDir\tree-ab"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $ram = [int]$env:T32_RAM_MIB; if ($ram -le 0) { $ram = 96 }
        $p = Start-Srv $true $ram $dir
        RunTurns @('A','B') @($baseA,$baseB) 6 $false 'full'
        RunTurns @('A','B') @($baseA,$baseB) 6 $true  'tree'
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        $parked = ($log | Select-String 'kv tree: parked').Count
        $restored = ($log | Select-String 'kv tree: restored').Count
        $miss = ($log | Select-String 'kv tree: restore miss').Count
        $diskmax = 0
        foreach ($l in $log) { if ($l -match 'ram = (\d+) B, disk = (\d+) B') { $d = [int64]$matches[2]; if ($d -gt $diskmax) { $diskmax = $d } } }
        Assert ($parked -ge 6) 'ab: park ran on every switch'
        Assert ($restored -ge 8) 'ab: later turns restored from the tree'
        Assert ($miss -eq 0)   'ab: every tree-pass turn restored from the tree'
        Assert ($diskmax -gt 0) 'ab: SSD tier actually used (disk > 0)'
        ComparePasses @('A','B') 6
    }
    'overlap' {
        $dir = "$OutDir\tree-ovl"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 96 $dir $false
        RunTurns @('C','D') @($baseC,$baseD) 3 $false 'full'
        RunTurns @('C','D') @($baseC,$baseD) 3 $true  'tree'
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        $parked = ($log | Select-String 'kv tree: parked').Count
        $restored = ($log | Select-String 'kv tree: restored').Count
        Assert ($parked -eq 0)    'overlap: no park for high-overlap switches (idle park off)'
        Assert ($restored -eq 0)  'overlap: no restore for high-overlap switches'
        ComparePasses @('C','D') 3
    }
    'b' {
        $dir = "$OutDir\tree-b"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 128 $dir
        RunTurns @('one','two','three','four') $sess 3 $false 'full'
        RunTurns @('one','two','three','four') $sess 3 $true  'tree'
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert ((($log | Select-String 'kv tree: restored').Count) -ge 4) 'b: short sessions restore from the tree'
        Assert ((($log | Select-String 'ref=') | Select-String 'ref=4').Count -ge 1) 'b: shared prefix stored once (refcount = 4)'
        ComparePasses @('one','two','three','four') 3
    }
    'b3' {
        $Model = '<models>\Qwen2.5-Coder-3B-IQ4_XS.gguf'
        $sys3 = Filler 'system' 256
        $b1 = Filler 'aa' 4096; $b2 = Filler 'bb' 4096
        $dir = "$OutDir\tree-b3"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 512 $dir
        RunTurns @('P','Q') @($b1,$b2) 3 $false 'full'
        RunTurns @('P','Q') @($b1,$b2) 3 $true  'tree'
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert (($log | Select-String 'kv tree: parked').Count -ge 1) 'b3: tree still parks pure-attention content (D11: no fork reuse)'
        Assert ((($log | Select-String 'failed to capture heal anchor').Count) -eq 0) 'b3: no failed captures'
        ComparePasses @('P','Q') 3
    }
    'neg' {
        $dir = "$OutDir\tree-neg"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 96 $dir
        RunTurns @('A','B') @($baseA,$baseB) 2 $true 'tree'
        $files = Get-ChildItem $dir -File
        Write-Output "NEG removing $($files.Count) block files"
        $files | Remove-Item -Force
        RunTurns @('A') @($baseA) 1 $true 'tree-after-loss'
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert ((($log | Select-String 'failed to read block|restore failed').Count) -ge 1) 'neg: SSD read failure is visible'
        Assert ($results.ContainsKey('tree-after-loss/A/1')) 'neg: request still succeeded after the failure'
    }
    'heal' {
        $dir = "$OutDir\tree-heal"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 512 $dir
        $h  = "User: " + ('filler ' * 480) + "`nAssistant: ok`nUser: " + ('filler ' * 600) + "`nAssistant: ok`nUser: "
        $tA = 'alpha-tail ' * 450
        $tF = 'gamma-fork ' * 450
        $reqs = @("$h$tA", "$h$tF", ("Zeta: " + ('omega ' * 1500)), "$h$tF")
        # D10: run the tree pass first on a fresh tree so the first fork restore happens with cache_prompt=true
        foreach ($pass in @(@($true,'tree'), @($false,'full'))) {
            $i = 0
            foreach ($q in $reqs) {
                $i++
                $r = ReqDelim $q $pass[0]
                $hsh = ContentHash $r.content
                $hashes["$($pass[1])/heal/$i"] = $hsh
                Write-Output ("[heal/$($pass[1])] req $i: hash=$hsh prompt_n=$($r.timings.prompt_n)")
            }
        }
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert ((($log | Select-String 'captured heal anchor').Count) -eq 1) 'heal: exactly one fork anchor captured'
        Assert ((($log | Select-String 'failed to capture heal anchor').Count) -eq 0) 'heal: no failed captures'
        for ($i = 1; $i -le 4; $i++) { Assert ($hashes["tree/heal/$i"] -eq $hashes["full/heal/$i"]) "heal: request $i identical (tree vs full prefill)" }
    }
}

Write-Output "RESULT $Mode: $fails failure(s)"
exit $fails
```

说明与调参（实施时允许按实测调，但必须保证断言语义不变）：
- `$ram` 由校准决定：`calib` 输出 `CALIB TOTAL BYTES`，阶段 3 验收取 `$env:T32_RAM_MIB = [int]($total/3/1MB)`（"精准溢出"：约 1/3 驻留、2/3 溢写 SSD）；预期 2x16K 总量 ~0.6-1.0 GB -> ram ~200-330 MiB。`ab` 默认 96 MiB 只是保守值。
- 预算按锚点大小定（重要）：2B 的 PARTIAL 锚点 ~20 MB/个；3B 纯 attention 的锚点 = 全量 KV（4K 会话 ~150 MB/个）。`b`=128 MiB、`b3`=512 MiB、`heal`=512 MiB 即为此定。`ab` 若出现"restore 命中数不足但 disk>0 成立"，允许把 ram 从 total/3 提到 total/2 再跑（保持 disk>0；在报告里记录调整与理由）。
- `ref` 用 `cache_prompt=false` + 树关，逐请求全量 prefill，仅作运行间确定性守门（见 Step 2）；主对比在模式内 `full` vs `tree` 两遍完成。
- `b` 的 refcount 证据：`dump()` 只在 `--tree-debug` 打开时打印；Start-Srv 在 `$tree` 为真时已追加 `--tree-debug`，`b` 模式断言日志里出现 `ref=4`。若日志量影响性能，可只对 `b` 保留 `--tree-debug`（其它模式去掉）。

- [ ] **Step 2: 运行 + 断言**

```powershell
$ps = 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1'
& powershell -ExecutionPolicy Bypass -File $ps -Mode calib | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-calib.txt'
$total = (Select-String -Path 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-calib.txt' -Pattern 'CALIB TOTAL BYTES: (\d+)').Matches.Groups[1].Value
$env:T32_RAM_MIB = [string][int]([int64]$total / 3 / 1MB)
& powershell -ExecutionPolicy Bypass -File $ps -Mode ab   | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab-ab.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode overlap | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab-overlap.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode b    | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab-b.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode ref  | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab-ref.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode b3   | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-3b.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode neg  | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-neg.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode heal | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-heal.txt'
```

输出一致性断言（脚本内为主）：
1. 每个模式（`ab`/`overlap`/`b`/`b3`/`heal`）在同一进程内先跑 `full` 轮（`cache_prompt=false`，纯全量 prefill）再跑 `tree` 轮（`cache_prompt=true`），`ComparePasses` 逐轮比较内容的 SHA256 前 16 位 —— 这是设计 §7.1(b)"逐位一致"判据在 server 层的等价物（token 级一致 => 同 hash）。
2. `ref` 模式跑两遍（第二遍输出到 `*-ref2.txt`），两遍的 `[ref/full] ... hash=` 行必须逐行相同（运行间确定性守门；若不稳定，先如实报告并记录，不作为其他断言的前提）。
3. 日志证据由脚本内 Assert 覆盖：`ab` 的 parked/restored/miss/disk 峰值；`overlap` 的 parked==0 且 restored==0；`b` 的 refcount=4；`heal` 的 captured >= 1；`neg` 的读失败可见且请求成功。

- [ ] **Step 3: 汇总归档 + 文档更新**

- 把 `t32-stage3-ab.ps1`、`t32-stage3-ab-*.txt`、`t32-stage3-*.txt`、各 `srv-*.txt` 关键日志复制到 `D:\LLM\Backend\v100-collab\artifacts\`（日志已直接写在那里则跳过）。
- RESULTS.md / STATUS.md 追加阶段 3 段（UTF-8 no-BOM 追加）：集成行为、验收数字（restore 次数、disk 峰值、ram 上界、逐 token 一致性结论）、D1 场景勘误、已知缺口（MTP spec）。
- 设计文档状态行改为"阶段 0-3 已完成；阶段 4 未开始"。

- [ ] **Step 4: 提交（若有代码修正）**

验收过程中发现的集成 bug 修复走 SDD fix round（本计划回写 + 修复提交，前缀 `server :`）；纯脚本/文档改动提交：

```powershell
git -C $repo add tools/server/server-context.cpp tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp common/common.h common/arg.cpp
git -C $repo commit -m "server : fix issues found by kv tree acceptance" -m "Assisted-by: opencode"
```

---

## Self-Review（写计划时的自查）

1. **Spec 覆盖**：
   - §3.4 接缝（launch save/load + idle）→ Task 3/4；`--kv-tree` 开关 + 默认关 → Task 2；旋钮（chunk/anchor-step/ram/disk/debug）→ Task 2；VRAM 零变化（不动 unified、不加 seq）→ 设计约束（代码未碰 KV 分配）✓。
   - §3.2/4 heal 自愈 → Task 4；§5 pin/淘汰复用阶段 2 + 流式恢复补强 → Task 1；§6 可见性（WRN/INF 日志契约）→ Task 3/4 日志表；§7 验收 → Task 5（D1 勘误已标注）。
   - §2.3 引擎回归：本阶段不改 `src/llama*`，无需重跑引擎测试（阶段 1 已覆盖）。
   - 终审延后项：`bytes_load` 少计 → Task 1；restore RAM 峰值 → Task 1（D5）；CLI 校验 → Task 2；restore miss INF → Task 3；drop-seq → Task 1+4；`cfg.debug` → Task 1。其余（锚点策略打磨、统计漂移、demote 评分、32-bit 上限、部分分支测试）留阶段 4 ✓。
2. **占位符扫描**：无 TBD/TODO；Task 1 中 `n_fail_guard()` 是误留，**已删除**（见勘误）；Task 2 中 `cache_ram_mib` 覆盖语句是误留，**已删除**。
3. **类型一致性**：`tree_heal`（llama_pos）、`prompt_park/prompt_restore_tree`、`drop_seq`、`set_block_payload`、`kv_tree_config` 字段名、日志契约字符串在 Task 1/2/3/4/5 中一致 ✓。
4. **已知风险**（实施时注意）：
   - `prompt_restore_tree` 后 `prompt.tokens` 被替换，`pos_next()` 与 KV 对齐依赖 `restore()` 内部的 `seq_rm(C,-1)`；Task 3 冒烟必须验证 A 第二次请求 `prompt_n` 下降。
   - heal 截断依赖"上一批 decode 已完成"（server 的批循环语义）；失败时 Task 4 冒烟会暴露。
   - `/tokenize` 端点若不存在，`TokCount` 改用 `Req` 加 `{n_predict:0}` 读 `timings.prompt_n`（实施时先 curl 验证）。
   - 脚本的 `ref` 模式需按 Step 2 说明扩展为 A/B+C/D+短会话三组。
