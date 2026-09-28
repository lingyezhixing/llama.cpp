# T32 阶段 5 实施计划: 检查点命名 + 分叉锚点 (D25-D30)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让锚点分布从"碰运气"变成证据驱动: restore miss 时在分叉点落锚 (heal-on-miss); 分叉锚点使用独立且可调的 `--tree-checkpoint-fork-step` (默认 8192), 猜测型锚点保持 `--tree-checkpoint-anchor-step` (原 `--tree-anchor-step`, 默认 32768); 删除死代码; 计数可观测.

**Architecture:** 引擎侧: cfg 加 `fork_step`; `capture_anchor` 间距改用 `fork_step` 并在跳过分计时; `restore` 在 miss (`C < 0`) 且 `deep > 0` 时返回 `heal = deep`; 删除 `promote_prune` (死代码, 见 D27 证明). server 侧: `prompt_restore_tree` 在 miss 时也设置 `tree_heal`, 调用方在 `prompt_clear` 后保留它. 验收: 新 `fork` 模式 + 全回归 + 短 soak.

**Tech Stack:** C++ (llama.cpp server + 引擎模块), PowerShell 5.1 (验收), Ninja Multi-Config.

**Spec:** `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md` (§9) + 本计划 §Decisions. 设计文档状态行与 D25/D26 记录在 Task 4 更新.

---

## Global Constraints

- 分支: `t32-stage5`, base = `t32-stage4` head `42ee7a6f7` (阶段 4 未合并, 本阶段不合并). 用户已授权本计划在 `t32-stage5` 上自动提交; **不 push, 不建 PR, 不部署**.
- 测试设备: 一律 `CUDA_VISIBLE_DEVICES=1`. 模型: 2B 主用; 3B 仅回归 b3. 不跑 27B.
- 构建: 测试 `<TEMP>\v100\build_test_t32.cmd test-t32-tree`; server: 删 `build\bin\Release\llama-server-impl.dll` 后 `<TEMP>\v100\build_server.cmd`. 共享 build 树保持 `LLAMA_BUILD_TESTS=ON`.
- 日志契约 (grep 依赖): 既有行不改词形; 本阶段新增/改动: `capture skipped at %d: within fork_step` (原 `within anchor_step`), `kv tree stats:` 增加 `step_skips=` 字段.
- 命令环境: PowerShell 5.1; 写文本用 `[System.IO.File]::WriteAllText` + `UTF8Encoding($false)`; 后台 server 必须 `-RedirectStandardError`.
- 代码风格: ASCII, 注释极少, 复用现有模式.
- 历史计划文档 (`t32-tree-plan-stage0-1/2/3/4.md`) 不改; 设计文档与验收脚本同步改名.

## Decisions (D25-D30)

- **D25:** 双档间距. MESSAGE (park 采纳的检查点候选) 用 `--tree-checkpoint-anchor-step` (默认 32768); ONDEMAND (分叉捕获) 用 `--tree-checkpoint-fork-step` (默认 8192). 两者独立、非负、0 = 无最小间距. 理由 (用户): 8K 重算代价低, 更密存储扛不住; 小模型可手动调密, 大模型调稀疏.
- **D26:** heal-on-miss. `restore` 在 `C < 0 && deep > 0` 时返回 `heal = deep`; server 在 miss 时把 `tree_heal` 保留过 `prompt_clear` (仅 cache_prompt=true), 首次全量 prefill 在分叉点捕获锚点. 中断安全: server 仅在 `n_tokens == heal` 精确相等时捕获, 引擎再验 `pos_max == pos - 1`; 下次选槽 `tree_heal` 无条件清零 (server-context.cpp:1749). 判错代价: 无 (守卫完备).
- **D27:** 删除 `promote_prune` (死代码). 证明: `capture_anchor` 取 `prev` = 同链在 pos 之下最深的锚点, 这是构造使然 (`fork_step = 0` 时也成立), 故 `(prev, pos)` 之间不可能有锚点; 剪枝窗口恒为空, 与间距无关. 猜测锚点的取舍由淘汰评分完成 (heat==0 的猜测天然先淘汰). 判错代价: 少一个空函数, 无行为变化.
- **D28:** 计数: `kv_tree_stats.anchors_skipped_step` (间距跳过, capture + park 候选两处), `stats_line()` 加 `step_skips=`.
- **D29:** 重命名: CLI `--tree-anchor-step` -> `--tree-checkpoint-anchor-step` (env `LLAMA_ARG_TREE_CHECKPOINT_ANCHOR_STEP`), 新 `--tree-checkpoint-fork-step` (env `LLAMA_ARG_TREE_CHECKPOINT_FORK_STEP`); `common_params.tree_anchor_step` -> `tree_checkpoint_anchor_step`, 新增 `tree_checkpoint_fork_step`. `kv_tree_config` 字段保持 `anchor_step`, 新增 `fork_step` (内部短名). 脚本 `t32-stage3-ab.ps1` 同步改名. 判错代价: 老命令行失效 (树未部署, 可接受).
- **D30:** 分支 base = `42ee7a6f7`; 阶段 5 不合并, 结束时与阶段 4 一起交用户决定.

## 文件结构

- Modify: `common/common.h` — params 字段改名 + 新字段.
- Modify: `common/arg.cpp` — 两个选项 (改名 + 新增), env, 校验.
- Modify: `tools/server/server-kv-tree.h` — `cfg.fork_step`, stats 两个新计数.
- Modify: `tools/server/server-kv-tree.cpp` — capture 间距/fork_step, 计数, `restore` miss-heal, 删 `promote_prune`.
- Modify: `tools/server/server-context.cpp` — cfg 映射 + 启动日志, `prompt_restore_tree` miss-heal, 调用方保留 tree_heal.
- Modify: `tests/test-t32-tree.cpp` — `run_logic_fork`, `scenario_fork_miss`.
- Modify (artifact): `artifacts\t32-stage3-ab.ps1` — 改名 + 新 `fork` 模式.
- Modify (artifact): `artifacts\t32-tree-storage-design.md` — 状态行 + D25/D26/D27 记录.

---

### Task 1: 重命名 + 新旋钮 (D25/D29, 无行为变化)

**Files:**
- Modify: `common/common.h:638`, `common/arg.cpp:1755-1763`, `tools/server/server-context.cpp:1497-1518`, `tools/server/server-kv-tree.h:14-21`
- Modify (artifact): `artifacts\t32-stage3-ab.ps1` (1 处)

**Interfaces:**
- Produces: `params.tree_checkpoint_anchor_step` (默认 32768), `params.tree_checkpoint_fork_step` (默认 8192); `kv_tree_config.fork_step` (默认 8192); CLI/env 新名 (Task 2 用 `cfg.fork_step`, Task 4 脚本用新名).

- [ ] **Step 1: 改 common.h**

```cpp
    int32_t     tree_chunk      = 512;      // KV tree block size in tokens
    int32_t     tree_checkpoint_anchor_step = 32768;   // min spacing between kv tree checkpoint anchors
    int32_t     tree_checkpoint_fork_step   = 8192;    // min spacing between kv tree fork anchors
```

- [ ] **Step 2: 改 arg.cpp (两个选项)**

原 `--tree-anchor-step` 块替换为:

```cpp
        {
            {"--tree-checkpoint-anchor-step"}, "N",
            string_format("minimum spacing between kv tree checkpoint anchors in tokens (default: %d)", params.tree_checkpoint_anchor_step),
            [](common_params & params, int value) {
                if (value < 0) {
                    throw std::invalid_argument("tree-checkpoint-anchor-step must be non-negative");
                }
                params.tree_checkpoint_anchor_step = value;
            }
        ).set_env("LLAMA_ARG_TREE_CHECKPOINT_ANCHOR_STEP").set_examples({LLAMA_EXAMPLE_SERVER}),
        {
            {"--tree-checkpoint-fork-step"}, "N",
            string_format("minimum spacing between kv tree fork anchors in tokens (default: %d)", params.tree_checkpoint_fork_step),
            [](common_params & params, int value) {
                if (value < 0) {
                    throw std::invalid_argument("tree-checkpoint-fork-step must be non-negative");
                }
                params.tree_checkpoint_fork_step = value;
            }
        ).set_env("LLAMA_ARG_TREE_CHECKPOINT_FORK_STEP").set_examples({LLAMA_EXAMPLE_SERVER}),
```

- [ ] **Step 3: 改 server-context.cpp 映射与启动日志**

```cpp
            cfg.chunk       = params_base.tree_chunk;
            cfg.anchor_step = params_base.tree_checkpoint_anchor_step;
            cfg.fork_step   = params_base.tree_checkpoint_fork_step;
```
启动日志加 fork_step:

```cpp
            SRV_INF("kv tree enabled: chunk = %d, anchor_step = %d, fork_step = %d, ram = %d MiB, disk = %s (%d MiB)\n",
                    cfg.chunk, cfg.anchor_step, cfg.fork_step, params_base.tree_ram_mib,
                    cfg.disk_dir.empty() ? "off" : cfg.disk_dir.c_str(), params_base.tree_disk_mib);
```

- [ ] **Step 4: server-kv-tree.h 加 cfg 字段**

```cpp
struct kv_tree_config {
    int         chunk       = 512;
    int         anchor_step = 32768;
    int         fork_step   = 8192;
    ...
```

- [ ] **Step 5: 脚本改名**

`artifacts\t32-stage3-ab.ps1` 的 Start-Srv 里 `'--tree-anchor-step'` -> `'--tree-checkpoint-anchor-step'` (只此一处).

- [ ] **Step 6: 验证**

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$repo='D:\LLM\Backend\src\llama.cpp-my'
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
Remove-Item -Force "$repo\build\bin\Release\llama-server-impl.dll" -ErrorAction SilentlyContinue
& '<TEMP>\v100\build_server.cmd'
& "$repo\build\bin\Release\llama-server.exe" --help 2>&1 | Select-String 'tree-checkpoint'
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode heal | Select-String 'RESULT'
```

Expected: help 显示 `--tree-checkpoint-anchor-step` 与 `--tree-checkpoint-fork-step`, 不再有 `--tree-anchor-step`; heal `RESULT heal: 0 failure(s)`.

- [ ] **Step 7: 提交**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add common/common.h common/arg.cpp tools/server/server-context.cpp tools/server/server-kv-tree.h
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "common : rename kv tree checkpoint options and add the fork step" -m "Assisted-by: opencode"
```

---

### Task 2: 引擎: fork 间距 + miss-heal + 计数 + 删死代码 (D25/D26/D27/D28)

**Files:**
- Modify: `tools/server/server-kv-tree.h` (stats), `tools/server/server-kv-tree.cpp` (`capture_anchor` 间距, `restore` miss-heal, `promote_prune` 删除, `stats_line`)
- Test: `tests/test-t32-tree.cpp` (`run_logic_fork`, 注册)

**Interfaces:**
- Consumes: `cfg.fork_step` (Task 1).
- Produces: `kv_tree_stats.anchors_skipped_step`; `kv_tree_restore.heal = deep` on miss; `stats_line()` 含 `step_skips=` (Task 3/4 依赖).

- [ ] **Step 1: 写失败测试**

`tests/test-t32-tree.cpp` 的 `run_logic_fixes` 之后插入:

```cpp
static void run_logic_fork() {
    fprintf(stderr, "[t32-tree] logic: fork spacing and miss heal\n");

    // fork anchors use fork_step, guesses keep anchor_step
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 32768;
        cfg.fork_step   = 8192;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);
        const auto tok = make_tokens(10240, 0);

        {
            kv_tree_io_fake io;
            io.max_pos = 10239;
            check(tree.park(io, nullptr, tok, {}), "fork: park the chain");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 1023;
            check(tree.capture_anchor(io, nullptr, tok, 1024), "fork: the first capture is free");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 5119;
            check(!tree.capture_anchor(io, nullptr, tok, 5120), "fork: within fork_step is refused");
        }
        check_eq(tree.stats().anchors_skipped_step, 1, "fork: the spacing skip is counted");
        {
            kv_tree_io_fake io;
            io.max_pos = 9215;
            check(tree.capture_anchor(io, nullptr, tok, 9216), "fork: at fork_step it is stored");
        }
        check_eq(tree.stats().anchors_added, 3, "fork: tip + two fork anchors");
        check(tree.stats_line().find("step_skips=1") != std::string::npos, "fork: stats line has step_skips");
    }

    // a restore miss reports the divergence point
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 32768;
        cfg.fork_step   = 8192;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);

        const auto tok_a = make_tokens(4096, 0);
        auto tok_b = make_tokens(4096, 1);
        for (int i = 0; i < 2048; ++i) {
            tok_b[i] = tok_a[i];
        }

        {
            kv_tree_io_fake io;
            io.max_pos = 4095;
            check(tree.park(io, nullptr, tok_a, {}), "fork: park A");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 4095;
            const kv_tree_restore r = tree.restore(io, nullptr, tok_b);
            check_eq(r.C, -1, "fork: B misses (no anchor on the shared path)");
            check_eq(r.heal, 2048, "fork: the miss reports the divergence point");
        }
    }
}
```

`run_logic()` 追加 `run_logic_fork();`.

- [ ] **Step 2: 运行确认失败**

Run: build + `--mode logic`
Expected: FAIL (`anchors_skipped_step` 不存在/编译失败; `r.heal` 为 -1).

- [ ] **Step 3: 实现**

(3a) `server-kv-tree.h` stats 加字段:

```cpp
    int64_t anchors_skipped_step = 0;
```

(3b) `capture_anchor` 间距 (363-374) 替换:

```cpp
    llama_pos prev = -1;
    for (const auto & kv : anchors) {
        if (kv.second.pos >= pos || kv.second.pos <= prev) {
            continue;
        }
        if (std::find(chain.begin(), chain.end(), kv.first.first) != chain.end()) {
            prev = kv.second.pos;
        }
    }

    if (prev >= 0 && pos - prev < cfg.fork_step) {
        fprintf(stderr, "[kv-tree] capture skipped at %d: within fork_step\n", pos);
        st.anchors_skipped++;
        st.anchors_skipped_step++;
        return false;
    }
```

(3c) `park` 候选间距 (1174-1177) 加计数:

```cpp
        if (prev_kept >= 0 && c.pos - prev_kept < cfg.anchor_step) {
            st.anchors_skipped++;
            st.anchors_skipped_step++;
            continue;
        }
```

(3d) `restore` miss-heal (1313-1316):

```cpp
    if (C < 0) {
        st.restore_miss++;
        res.heal = m.deep;   // seed a fork anchor while the caller re-prefills
        return res;
    }
```

(3e) 删除 `promote_prune` 定义 (404-431) 与调用 (399) 与声明 (server-kv-tree.h:179), 附一行注释在 capture 成功处说明为什么不需要剪枝:

```cpp
    // no pruning needed: prev is the deepest same-chain anchor below pos, so no anchor can sit between them
```

(3f) `stats_line()` 加 `step_skips=%" PRId64` 与参数 `st.anchors_skipped_step` (放在 `skipped=` 之后).

- [ ] **Step 4: 运行确认通过**

Run: build + `--mode logic` -> 0 FAIL (64 + 7 = 71 checks).

- [ ] **Step 5: 提交**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : add fork anchor spacing and miss heal to the kv tree" -m "Assisted-by: opencode"
```

---

### Task 3: server 接线 (miss-heal) + 模型测试

**Files:**
- Modify: `tools/server/server-context.cpp:361-372` (prompt_restore_tree), `:1795-1801` (调用方)
- Test: `tests/test-t32-tree.cpp` (`scenario_fork_miss`, 注册)

**Interfaces:**
- Consumes: Task 2 的 `res.heal` on miss.
- Produces: server 在 miss 时也设 `tree_heal` (Task 4 fork 模式验收依赖日志 `captured heal anchor at <deep>` 出现在 miss 之后).

- [ ] **Step 1: 写失败测试 (model)**

`scenario_restore_anchors` 之后插入:

```cpp
static int scenario_fork_miss(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] scenario: fork miss\n");

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tok_a = make_tokens(4096, 3);
    auto tok_b = make_tokens(4096, 4);
    for (int i = 0; i < 2048; ++i) {
        tok_b[i] = tok_a[i];
    }

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    if (prefill(ctx, 0, tok_a, 0, 512) != 0) {
        return 1;
    }
    check(tree.park(io, nullptr, tok_a, {}), "fork-miss: park A");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    const kv_tree_restore r1 = tree.restore(io, nullptr, tok_b);
    check_eq(r1.C, -1, "fork-miss: the first visit misses");
    check_eq(r1.heal, 2048, "fork-miss: the miss reports the divergence point");

    // the server prefills to the heal position and captures there
    if (prefill(ctx, 0, std::vector<llama_token>(tok_b.begin(), tok_b.begin() + 2048), 0, 512) != 0) {
        return 1;
    }
    check(tree.capture_anchor(io, nullptr, tok_b, 2048), "fork-miss: capture at the divergence point");
    if (prefill(ctx, 0, tok_b, 2048, 512) != 0) {
        return 1;
    }

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    const kv_tree_restore r2 = tree.restore(io, nullptr, tok_b);
    check_eq(r2.C, 2048, "fork-miss: the second visit restores at the fork");
    check_eq(r2.heal, -1, "fork-miss: no heal on a full match");

    const auto base = run_baseline(ctx, tok_b, 42, 8);
    const auto r = run_tree_path(ctx, tree, io, tok_b, 42, 8);
    check_eq(r.res.C, 2048, "fork-miss: the tree path restores at the fork");
    check(r.gen == base, "fork-miss: tokens match the baseline");

    tree.dump();

    return 0;
}
```

`main` 的 model 分派加 `ret |= scenario_fork_miss(ctx, cfg);`.

- [ ] **Step 2: 运行确认失败**

Run: build + model mode
Expected: FAIL `fork-miss: the first visit misses` 之后的 heal 断言 (engine 已支持, 但 server 未接线不影响 harness; 此步实际为 GREEN 验证引擎路径) —— 若引擎 Task 2 已完成, 本测试应直接 PASS; 失败点应在 server 端行为 (下一步的验收模式). 记录实际结果.

- [ ] **Step 3: server 接线**

(3a) `prompt_restore_tree` miss 分支:

```cpp
        kv_tree_restore res = tree.restore(io_tgt, io_dft_ptr, tokens.get_tokens());
        if (res.C <= 0) {
            SLT_INF(*this, "kv tree: restore miss for %zu tokens, full prefill\n", tokens.size());
            tree_heal = res.heal;   // seed a fork anchor while the prompt is re-prefilled
            return false;
        }
```

(3b) 调用方 (get_available_slot) 保留 heal 过 prompt_clear:

```cpp
                if (tree) {
                    ret->prompt_park(*tree);

                    // no point restoring when the request will not reuse the prompt
                    if (!task.params.cache_prompt || !ret->prompt_restore_tree(*tree, task.tokens, params_base.n_ctx_checkpoints,
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS)) {
                        const llama_pos heal = task.params.cache_prompt ? ret->tree_heal : -1;
                        ret->prompt_clear();
                        ret->tree_heal = heal;
                    }
                }
```

- [ ] **Step 4: 运行确认通过**

Run: build test + model mode -> 0 FAIL (56 + 7 = 63 checks); rebuild server; heal 模式 `RESULT heal: 0 failure(s)`.

- [ ] **Step 5: 提交**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-context.cpp tests/test-t32-tree.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : seed fork anchors when a tree restore misses" -m "Assisted-by: opencode"
```

---

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
