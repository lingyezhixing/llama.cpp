### Task 4: fork 验收模式 + 全回归 + 短 soak + 文档

**Files:**
- Modify (artifact): `artifacts\t32-stage3-ab.ps1` (`fork` 模式), `artifacts\t32-tree-storage-design.md` (状态行 + D25/D26/D27)
- 产物: `artifacts\t32-stage5-fork.txt`, `t32-stage5-{logic,model,accept}.txt`, `t32-stage5-{ab,overlap,b,b3,neg,heal,ref}.txt`, `t32-stage5-soak.txt`, `t32-stage5-logs\`

**Interfaces:**
- Consumes: Task 3 的 miss-heal 日志 `kv tree: captured heal anchor at %d`、Task 2 的 fork 间距.
- Produces: 阶段 5 验收证据; 无 git 提交 (脚本/文档是 artifact).

- [ ] **Step 1: 加 `fork` 模式**

在 `switch ($Mode)` 的 `soak` 分支之后加:

```powershell
        'fork' {
            $dir = "$OutDir\tree-fork"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            # no message delimiters -> no checkpoint guesses -> the first fork is a restore miss
            $p = Start-Srv $true 512 $dir $true 32768 1 '' 2048 $false 16384
            Build-Sys
            $base = Filler 'shared' 8192
            $a1 = Filler 'branchA1' 8192
            $a2 = Filler 'branchA2' 4096
            $b1 = $base + $a1 + $a2                  # fork 1 at ~len(base)
            $b2 = $base + (Filler 'branchB' 2048)    # diverges at fork 1
            $b3 = $base + $a1 + (Filler 'branchC' 2048)  # diverges at ~fork 1 + 8192 (>= fork_step)

            $hashes = @{}
            foreach ($pass in @(@($true,'tree'), @($false,'full'))) {
                $i = 0
                foreach ($q in @($b1, $b2, $b2, $b3, $b3)) {
                    $i++
                    $r = Req $q $pass[0]
                    $hashes["$($pass[1])/fork/$i"] = (ContentHash $r.content)
                    Write-Output ("[fork/$($pass[1])] req ${i}: prompt_n=$($r.timings.prompt_n) cached=$($r.tokens_cached)")
                }
            }
            Stop-Srv

            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $captured = @($log | Select-String 'captured heal anchor at (\d+)' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value })
            $restored = @($log | Select-String 'kv tree: restored (\d+) tokens' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value })
            Write-Output "FORK METRICS captured=[$($captured -join ',')] restored=[$($restored -join ',')]"
            Assert ($captured.Count -ge 2) 'fork: two fork anchors captured (miss heal + second fork)'
            Assert ($restored -contains $captured[0]) 'fork: the miss-heal anchor is reused'
            Assert ($restored -contains $captured[1]) 'fork: the second fork anchor is reused'
            for ($i = 1; $i -le 5; $i++) {
                Assert ($hashes["tree/fork/$i"] -eq $hashes["full/fork/$i"]) "fork: request $i identical (tree vs full prefill)"
            }
        }
```

注: `Start-Srv` 位置参数顺序 = tree, ram, diskdir, idle, anchor_step, np, slot_save, disk_mib, tree_debug, ctx; 两个分叉点相距 = len($a1) ≈ 8192 >= 默认 fork_step, 第二个锚点可落 (Filler 可能略超目标 token 数, 只会更大).

- [ ] **Step 2: 跑 fork 模式并修正**

```powershell
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode fork | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage5-fork.txt'
```

Expected: `RESULT fork: 0 failure(s)`, `captured` 至少含 miss-heal 的分叉点, `restored` 含该点; 5 项逐位一致. 按实测修正第二分叉的间距/长度.

- [ ] **Step 3: 全回归 + 短 soak (cuda1)**

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$art='D:\LLM\Backend\v100-collab\artifacts'
$repo='D:\LLM\Backend\src\llama.cpp-my'
$exe="$repo\build\bin\Release\test-t32-tree.exe"
& $exe --mode logic 2>&1 | Tee-Object "$art\t32-stage5-logic.txt" | Select-Object -Last 1
& $exe -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on --mode model --ram-mib 4096 -c 8192 2>&1 | Tee-Object "$art\t32-stage5-model.txt" | Select-Object -Last 1
& $exe -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on --mode accept --ram-mib 4096 -c 8192 2>&1 | Tee-Object "$art\t32-stage5-accept.txt" | Select-Object -Last 1
$env:T32_RAM_MIB='133'
foreach ($m in @('ab','overlap','b','b3','neg','heal','ref','fork')) {
    & powershell -ExecutionPolicy Bypass -File "$art\t32-stage3-ab.ps1" -Mode $m 2>&1 | Tee-Object "$art\t32-stage5-$m.txt" | Select-String 'RESULT|FAIL'
}
$env:T32_RAM_MIB=''
& powershell -ExecutionPolicy Bypass -File "$art\t32-stage3-ab.ps1" -Mode soak -Minutes 5 | Tee-Object "$art\t32-stage5-soak.txt" | Select-String 'SOAK METRICS|SOAK RESTART|RESULT'
```

Expected: 全部 0 FAIL / 0 failure(s); soak 绿.

- [ ] **Step 4: 归档 + 文档**

- 复制 srv 日志到 `artifacts\t32-stage5-logs\`.
- `t32-tree-storage-design.md`: 状态行改 `阶段 0-5 已完成 (... 分支 t32-stage4/t32-stage5 未合并/未 push); 磨损优化/持久化待后续`; §9 加 D25/D26/D27 记录 (双档间距/ miss-heal / 删除死代码).
- `RESULTS.md`/`STATUS.md` 追加阶段 5 段; `TASKS\T32-agent-session-reuse.md` 加阶段 5 Result.

- [ ] **Step 5: 收尾**

不合并; 分支 `t32-stage5` 保留, 与阶段 4 一起交用户决定 (整理并合并 / push / 保留).

---

## Self-Review

- **Spec 覆盖:** D25 (Task 1+2), D26 (Task 2+3), D27 (Task 2), D28 (Task 2), D29 (Task 1), D30 (全局). 用户诉求: 命名 (T1), 8K 默认 + 旋钮 (T1), 分叉点自动落锚 (T2+T3), 中断安全 (守卫已在, T2 测试覆盖 pos_max 不齐拒绝路径的既有测试保留), 剪枝只剪猜的 (D27 结构性证明 + 删除死代码).
- **Placeholder 扫描:** 无 TBD; fork 模式的第二个分叉间距已定死 (`$a1` ≈ 8192 >= 默认 fork_step).
- **类型一致性:** `cfg.fork_step`/`anchors_skipped_step`/`res.heal`/`tree_checkpoint_*` 在 h/cpp/server/tests/script 中一致; `stats_line` 字段名 `step_skips=` 与测试断言一致.
- **风险:** (1) Task 3 Step 2 的 model 测试在引擎已改后可能直接 PASS (无 server 参与) —— 记录实际 RED 证据 (若无可写"引擎先行已绿, server 接线由 Task 4 fork 模式验收"); (2) fork 模式的第二个分叉依赖共享长度 ≥ fork_step; 计划已给修正路径; (3) 重命名影响既有脚本, T1 验证含 heal 回归.
