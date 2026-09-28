### Task 5: 长跑 (30/60 min) + 全回归 + 归档 + 整理合并

**Files:**
- 产物: `artifacts\t32-stage4-soak-30.txt`, `t32-stage4-soak-60.txt`, `t32-stage4-{logic,model,accept,ab,overlap,b,b3,neg,heal,ref}.txt`, `t32-stage4-logs\`
- Modify: `artifacts\t32-tree-storage-design.md` (状态行 + §9 D12/D13 已实现), `D:\LLM\Backend\v100-collab\RESULTS.md`, `STATUS.md`, `TASKS\T32-agent-session-reuse.md`

**Interfaces:**
- Consumes: Task 1-4 全部.
- Produces: 阶段 4 终态 + 合并后的 master (未 push).

- [ ] **Step 1: 30 min 验收 soak**

```powershell
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode soak -Minutes 30 | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-soak-30.txt'
```

Expected: `RESULT soak: 0 failure(s)`; 若失败 -> 定位修复 (新提交) 后重跑.

- [ ] **Step 2: 全回归 (cuda1)**

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$repo='D:\LLM\Backend\src\llama.cpp-my'
$art='D:\LLM\Backend\v100-collab\artifacts'
$exe="$repo\build\bin\Release\test-t32-tree.exe"
& $exe --mode logic 2>&1 | Tee-Object "$art\t32-stage4-logic.txt" | Select-Object -Last 1
& $exe -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096 2>&1 | Tee-Object "$art\t32-stage4-model.txt" | Select-Object -Last 1
& $exe -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode accept --ram-mib 4096 2>&1 | Tee-Object "$art\t32-stage4-accept.txt" | Select-Object -Last 1
$env:T32_RAM_MIB='133'
foreach ($m in @('ab','overlap','b','b3','neg','heal','ref')) {
    & powershell -ExecutionPolicy Bypass -File "$art\t32-stage3-ab.ps1" -Mode $m 2>&1 | Tee-Object "$art\t32-stage4-$m.txt" | Select-String 'RESULT|FAIL'
}
$env:T32_RAM_MIB=''
```

Expected: 每个 `RESULT <mode>: 0 failure(s)`; logic/model/accept 0 FAIL.

- [ ] **Step 3: 60 min 最终 soak**

```powershell
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode soak -Minutes 60 | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-soak-60.txt'
```

Expected: `RESULT soak: 0 failure(s)`; 记录 `SOAK METRICS` 全量数字 (IO 基线).

- [ ] **Step 4: 归档 + 频道文档**

- 复制本轮 srv 日志到 `artifacts\t32-stage4-logs\`.
- `t32-tree-storage-design.md` 状态行改为: `阶段 0-4 已完成 (阶段 4: soak 长跑 + D12/D13, 分支 t32-stage4 未 push); 磨损优化/持久化待后续`.
- `RESULTS.md`/`STATUS.md` 追加阶段 4 段 (数字 + D14 磨损延后 + 已知缺口), `TASKS\T32-agent-session-reuse.md` 加阶段 4 Result.

- [ ] **Step 5: 整理提交 + 合并 (沿用阶段 3 惯例)**

```powershell
$repo='D:\LLM\Backend\src\llama.cpp-my'
$want = git -C $repo rev-parse 't32-stage4^{tree}'
git -C $repo checkout -b t32-stage4-tidy master
git -C $repo checkout t32-stage4 -- tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp
git -C $repo commit -m "server : fix kv tree counters, spacing scope and startup cleanup" -m "Assisted-by: opencode"
git -C $repo checkout t32-stage4 -- tests/test-t32-tree.cpp
git -C $repo commit -m "tests : add kv tree fix, wipe and anchor payload tests" -m "Assisted-by: opencode"
git -C $repo checkout t32-stage4 -- tools/server/server-context.cpp
git -C $repo commit -m "server : rebuild context checkpoints after a tree restore" -m "Assisted-by: opencode"
$t = git -C $repo rev-parse 'HEAD^{tree}'
if ($t -ne $want) { throw "tree mismatch: $t != $want" }
git -C $repo checkout master
git -C $repo merge --ff-only t32-stage4-tidy
git -C $repo branch -d t32-stage4-tidy; git -C $repo branch -D t32-stage4
```

(合并后复跑: logic + model + 5 min soak, 全绿; 分支删除前记录 head 到 STATUS.)

---

## Self-Review

- **Spec 覆盖:** D12 (Task 3), D13 (Task 2), D22/D23 (Task 1), 聚合指标 (Task 3d), soak/压力/正确性 (Task 4-5), 磨损延后 (D14, 无代码), D11 不做 (D16). 用户清单: 反复建树 (soak 轮换+分叉), 修剪 (小预算 churn), 整树删除 (erase + restart), RAM/SSD 调度 (demote/restore/diskmax 断言), 检查点更新 (D12 + rebuilt 断言) - 全部有任务对应.
- **Placeholder 扫描:** 无 TBD/TODO; 每步含实际代码/命令/期望输出.
- **类型一致性:** `kv_tree_restore_anchor.pos/data_tgt/data_dft` 在 h/cpp/测试/server 一致; `prompt_restore_tree(tree, tokens, n_ckpt_max)` 签名与调用点一致; `stats_line()` 字段名与 Task 4 日志解析不耦合 (soak 只 grep 既有日志).
- **风险点:** (1) heal 模式 `rebuilt >= 1` 依赖请求 4 恢复在 C=1024 且 487 锚点仍在 - 阶段 3 证据支持; 若实测为 0, 在 Task 3 Step 4 记录并调整断言口径 (改在 soak 断言, 已双保险). (2) soak RSS/句柄阈值是启发式, 首轮 smoke 后可按实测微调并记录. (3) `--slot-save-path` 目录与 tree 目录分离, 互不影响.
