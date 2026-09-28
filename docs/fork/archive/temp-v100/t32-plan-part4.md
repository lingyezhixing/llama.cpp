### Task 6: 淘汰次序 + pin + 拒绝 + 降级优先

**Files:**
- Modify: `tools/server/server-kv-tree.h` (加淘汰/删除方法声明)
- Modify: `tools/server/server-kv-tree.cpp` (`has_successor` / `remove_block` / `remove_seq` / `demote_one` / `evict_*` + 完整 `enforce_budget`)
- Modify: `tests/test-t32-tree.cpp` (`run_logic_evict`)

**Interfaces:**
- Consumes: Task 5 的 `enforce_budget`/`demote_*`/`remove_anchor`
- Produces: 淘汰次序 (锚点 -> 叶块 -> 整条叶序列 -> 拒绝), pin 纪律, `evicted_*`/`evict_refused` 计数

- [ ] **Step 1: 头文件加声明**

private 段加:

```cpp
    bool has_successor(const kv_tree_block & b) const;

    bool demote_one();

    bool evict_anchor_one();
    bool evict_block_one();
    bool evict_seq_one();

    void remove_seq(std::unordered_map<uint64_t, kv_tree_seq>::iterator it);
```

- [ ] **Step 2: `server-kv-tree.cpp` 加实现**

在 `settle` 之后插入:

```cpp
bool kv_tree::has_successor(const kv_tree_block & b) const {
    const auto it = blocks_at.find(b.pos1);
    return it != blocks_at.end() && !it->second.empty();
}

void kv_tree::remove_seq(std::unordered_map<uint64_t, kv_tree_seq>::iterator it) {
    kv_tree_seq & s = it->second;

    for (auto & kv : anchors) {
        bool in_chain = false;
        for (const uint64_t hash : s.chain) {
            if (hash == kv.first.first) {
                in_chain = true;
                break;
            }
        }
        if (in_chain && kv.second.refcount > 0) {
            kv.second.refcount--;
        }
    }

    for (auto rit = s.chain.rbegin(); rit != s.chain.rend(); ++rit) {
        const auto b = blocks.find(*rit);
        if (b == blocks.end()) {
            continue;
        }
        if (--b->second.refcount <= 0) {
            remove_block(b);
        }
    }

    seqs.erase(it);
    st.evicted_seqs++;
}

bool kv_tree::demote_one() {
    auto best = blocks.end();
    int64_t best_score = 0;

    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        const kv_tree_block & b = it->second;
        if (b.on_disk || b.data.empty() || b.pinned) {
            continue;
        }

        int64_t score = b.refcount * 1000000 + b.heat * 1000 + b.last_used;
        if (!has_successor(b)) {
            score -= 100000000;   // leaves go to disk first
        }

        if (best == blocks.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best != blocks.end() && demote_block(best->second)) {
        return true;
    }

    auto best_a = anchors.end();
    int64_t best_a_score = 0;

    for (auto it = anchors.begin(); it != anchors.end(); ++it) {
        const kv_tree_anchor & a = it->second;
        if (a.on_disk || a.data_tgt.empty() || a.pinned) {
            continue;
        }

        const int64_t score = a.refcount * 1000000 + a.heat * 1000 + a.last_used;

        if (best_a == anchors.end() || score < best_a_score) {
            best_a = it;
            best_a_score = score;
        }
    }

    return best_a != anchors.end() && demote_anchor(best_a->second);
}

bool kv_tree::evict_anchor_one() {
    auto best = anchors.end();
    int64_t best_score = 0;

    for (auto it = anchors.begin(); it != anchors.end(); ++it) {
        const kv_tree_anchor & a = it->second;
        if (a.pinned) {
            continue;
        }

        const int64_t score = a.refcount * 1000000 + a.heat * 1000 + a.last_used;

        if (best == anchors.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best == anchors.end()) {
        return false;
    }

    remove_anchor(best);
    st.evicted_anchors++;
    return true;
}

bool kv_tree::evict_block_one() {
    auto best = blocks.end();
    int64_t best_score = 0;

    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        const kv_tree_block & b = it->second;
        if (b.pinned || b.refcount > 1 || has_successor(b)) {
            continue;
        }

        const int64_t score = b.refcount * 1000000 + b.heat * 1000 + b.last_used;

        if (best == blocks.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best == blocks.end()) {
        return false;
    }

    remove_block(best);
    return true;
}

bool kv_tree::evict_seq_one() {
    auto best = seqs.end();
    int64_t best_score = 0;

    for (auto it = seqs.begin(); it != seqs.end(); ++it) {
        if (it->second.pinned) {
            continue;
        }

        const int64_t score = it->second.last_used;

        if (best == seqs.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best == seqs.end()) {
        return false;
    }

    remove_seq(best);
    return true;
}
```

用下面替换 Task 5 的 `enforce_budget`:

```cpp
bool kv_tree::enforce_budget() {
    while ((size_t) st.bytes_ram > cfg.ram_limit && demote_one()) {
    }

    // still short: evict in the order anchors -> leaf blocks -> whole leaf sequences
    while ((size_t) st.bytes_ram > cfg.ram_limit) {
        if (!evict_anchor_one() && !evict_block_one() && !evict_seq_one()) {
            break;
        }
    }

    if ((size_t) st.bytes_ram > cfg.ram_limit) {
        fprintf(stderr, "[kv-tree] eviction could not free enough ram (%" PRId64 " > %zu)\n", st.bytes_ram, cfg.ram_limit);
        st.evict_refused++;
        return false;
    }

    return true;
}
```

- [ ] **Step 3: harness 加 `run_logic_evict`**

在 `run_logic` 之前插入:

```cpp
static int run_logic_evict() {
    fprintf(stderr, "[t32-tree] logic: eviction\n");

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 20 * 1024;   // 5 chunks worth of fake payloads
    cfg.disk_dir    = "";          // no disk tier: eviction only

    kv_tree tree(cfg);

    const auto tok_a = make_tokens(1536, 0);
    const auto tok_b = make_tokens(1536, 1);

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_a, {}), "evict: park A");
        check_eq(tree.stats().blocks_ram, 3, "evict: A has 3 blocks");
    }

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_b, {}), "evict: park B forces eviction");
        check(tree.stats().evicted_blocks > 0, "evict: blocks were evicted");
        check_eq(tree.stats().park_refused, 0, "evict: park still succeeded");
    }

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        const auto r = tree.restore(io, nullptr, tok_a);
        check_eq(r.C, -1, "evict: A degrades to a full prefill (visible)");
    }

    {
        kv_tree_config cfg2 = cfg;
        cfg2.ram_limit = 4 * 1024;

        kv_tree tree2(cfg2);
        kv_tree_io_fake io;
        io.max_pos = 1535;

        check(!tree2.park(io, nullptr, tok_a, {}), "evict: park refused with a tiny budget");
        check_eq(tree2.stats().park_refused, 1, "evict: refusal counted");
        check_eq(tree2.stats().evict_refused, 1, "evict: refusal is visible");
        check_eq(tree2.stats().blocks_ram, 0, "evict: nothing was left behind");
    }

    return n_fail == 0 ? 0 : 1;
}
```

`run_logic` 的第一行改为:

```cpp
    fprintf(stderr, "[t32-tree] mode logic\n");

    run_logic_evict();
```

- [ ] **Step 4: 构建并跑 logic**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
```
Expected: 全部 PASS; `evict: park B forces eviction` 与 `evict: park refused with a tiny budget` 必须 PASS; `blocks_ram == 0` 表示拒绝后无残留.

- [ ] **Step 5: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : add kv tree eviction order and pin discipline" -m "Assisted-by: opencode"
```

---

### Task 7: 验收 harness (mini A/B) + 归档 + 文档

**Files:**
- Modify: `tests/test-t32-tree.cpp` (`run_accept` + main 接线)
- Modify: `D:\LLM\Backend\v100-collab\TASKS\T32-agent-session-reuse.md` (追加 `## Result (stage 2)`)
- Modify: `D:\LLM\Backend\v100-collab\RESULTS.md` / `STATUS.md` (追加阶段 2 段)
- Modify: `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md` (状态行)
- Create: `D:\LLM\Backend\v100-collab\artifacts\t32-stage2-model.txt` / `t32-stage2-accept.txt` (运行归档)

**Interfaces:**
- Consumes: Task 1-6 全部
- Produces: 阶段 2 出口证据 (mini A/B 全绿 + 归档)

- [ ] **Step 1: harness 加 `run_accept`**

在 `scenario_ssd` 之后插入:

```cpp
static int run_accept(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] mode accept\n");

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    // A-mini: two 4096-token sequences sharing 3072 tokens, six alternating restores
    const auto tok_a = make_fork_tokens(3072, 1024, 0);
    const auto tok_b = make_fork_tokens(3072, 1024, 1);

    prefill(ctx, 0, tok_a, 0, 512);

    const size_t seq_bytes = llama_state_seq_get_size_range_ext(ctx, 0, 0, 4096, 0);

    check(tree.park(io, nullptr, tok_a, {}), "accept: park A (4096)");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, tok_b, 0, 512);
    check(tree.park(io, nullptr, tok_b, {}), "accept: park B (4096)");

    const auto base_a = run_baseline(ctx, tok_a, 42, 8);
    const auto base_b = run_baseline(ctx, tok_b, 42, 8);

    for (int round = 0; round < 6; ++round) {
        const auto ra = run_tree_path(ctx, tree, io, tok_a, 42, 8);
        check_eq(ra.res.C, 4096, "accept: A restores at the tip");
        check(ra.gen == base_a, "accept: A tokens match the baseline");

        const auto rb = run_tree_path(ctx, tree, io, tok_b, 42, 8);
        check_eq(rb.res.C, 4096, "accept: B restores at the tip");
        check(rb.gen == base_b, "accept: B tokens match the baseline");
    }

    check_eq(tree.stats().tokens_reused, 12 * 4096, "accept: tokens reused");
    check_eq(tree.stats().blocks_ram + tree.stats().blocks_disk, 8, "accept: shared trunk stored once (6+2 blocks)");
    check(tree.stats().bytes_ram + tree.stats().bytes_disk <= (int64_t) seq_bytes * 2 * 10 / 16, "accept: stored bytes are far below two full copies");

    // B-mini: four 1024-token sessions sharing a 512-token prefix
    {
        kv_tree_config cfg2 = cfg;
        cfg2.ram_limit = 4096ull << 20;

        kv_tree tree2(cfg2);
        kv_tree_io_llama io2(ctx, 0);

        std::vector<std::vector<llama_token>> toks;
        for (int i = 0; i < 4; ++i) {
            toks.push_back(make_fork_tokens(512, 512, 10 + i));
        }

        for (int i = 0; i < 4; ++i) {
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
            prefill(ctx, 0, toks[i], 0, 512);
            check(tree2.park(io2, nullptr, toks[i], {}), "accept: park short session");
        }

        check_eq(tree2.stats().blocks_ram + tree2.stats().blocks_disk, 5, "accept: shared prefix stored once (1+4 blocks)");

        for (int i = 0; i < 4; ++i) {
            const auto base = run_baseline(ctx, toks[i], 42, 8);
            const auto r = run_tree_path(ctx, tree2, io2, toks[i], 42, 8);
            check_eq(r.res.C, 1024, "accept: short session restores at the tip");
            check(r.gen == base, "accept: short session tokens match the baseline");
        }
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}
```

`main` 的 model 分支之后加:

```cpp
    if (mode == "accept") {
        return run_accept(ctx, cfg);
    }
```

- [ ] **Step 2: 全量跑 + 归档**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
$out = & '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096 2>&1 | Out-String
[System.IO.File]::WriteAllText('D:\LLM\Backend\v100-collab\artifacts\t32-stage2-model.txt', $out, (New-Object System.Text.UTF8Encoding($false)))
$out = & '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 8192 --mode accept --ram-mib 4096 2>&1 | Out-String
[System.IO.File]::WriteAllText('D:\LLM\Backend\v100-collab\artifacts\t32-stage2-accept.txt', $out, (New-Object System.Text.UTF8Encoding($false)))
```
Expected: 三模式 exit 0; 归档里无 FAIL. 另外用 3B 纯 attention 模型做一次对照:
```powershell
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen2.5-Coder-3B-IQ4_XS.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```
Expected: tip / sparsify / ssd PASS; fork 场景里 "A' 无锚点 -> C=-1" 的判据对纯 attention 同样成立 (锚点仍由检查点提供, 行为一致).

- [ ] **Step 3: 频道文档**

`TASKS\T32-agent-session-reuse.md` 追加:

```
## Result (stage 2)

- 树模块 `tools/server/server-kv-tree.{h,cpp}` + harness `tests/test-t32-tree.cpp` 完成 (分支 `t32-stage2`)
- 模式: logic (fake IO 单测: 哈希链/去重/稀疏化/淘汰/拒绝) / model (2B 真机: tip/fork/自愈/sparsify/ssd) / accept (mini A/B)
- 结论: park/restore 逐 token 与基线一致 (np=1); 分叉按最深可用锚点恢复, 自愈后第二次免费;
  SSD 单份权威往返逐位一致; 淘汰次序 (锚点 -> 叶块 -> 叶序列 -> 拒绝) 与 pin 纪律生效, 拒绝可见
- 归档: artifacts/t32-stage2-model.txt, artifacts/t32-stage2-accept.txt
- 未接 server (`--kv-tree` 属阶段 3); 生产未动
```

`RESULTS.md` / `STATUS.md` 各追加一段 (格式同阶段 1): 完成内容 + 关键数字 (块数/复用 token/淘汰计数来自归档) + 未 push/未部署. spec 状态行改为:

```
状态: 阶段 0-2 已完成 (阶段 2 树模块 + harness, 分支 `t32-stage2`, 未合并/未 push); 阶段 3-4 未开始
```

- [ ] **Step 4: Commit**

```
git add tests/test-t32-tree.cpp
git commit -m "tests : add kv tree acceptance harness" -m "Assisted-by: opencode"
```

频道文档在 `D:\LLM\Backend\v100-collab` (非 git 仓库), 不进 commit.

---

## Self-Review

**Spec 覆盖检查 (§1-§6, 阶段 2 范围):**

| spec 条目 | 任务 |
|---|---|
| §1.2 块链/内容哈希/逐 token 校验/元数据 | T1 |
| §1.3 锚点 (tip/message/ondemand, 硬约束不凭空生成) | T2/T3/T4 |
| §2 引擎 API 使用 (range + PARTIAL_ONLY) | T2 (adapter) |
| §3.1 park (缺失段/末端锚点/收编/簿记) | T1/T2/T3 |
| §3.2 restore (C=最深锚点/装载/裁剪/状态/路径锚点) | T2/T3 |
| §3.2 自愈 (重放后捕获) | T4 |
| §3.3 降级 (位置不从 0 / SSD 读失败 / 预算不够) | T2 (前置检查) / T5 / T6 |
| §4 稀疏化 + 冲突留靠前 + 分叉提升/剪枝 | T3 / T4 |
| §5.1 统计 | T1-T6 (stats) |
| §5.2 RAM/SSD 放置 (单份权威/移动) | T5/T6 |
| §5.3 淘汰次序 + 拒绝 | T6 |
| §5.4 pin 纪律 | T5/T6 |
| §6 错误处理与可观测 (WRN + 计数) | T2-T6 |
| §7.3 反例 (无锚点/位置不从 0/SSD 读失败) | T3 (C=-1) / T2 (park 拒绝) / T5 |
| 独立 harness (短序列正确性) | T1-T7 |

未覆盖 (属阶段 3/4, 见 spec §9): server 集成/`--kv-tree`/A/B 真实场景/指标上报/持久化.

**Placeholder 扫描:** 无 TBD/TODO; 每个代码步骤都是完整函数或明确 old/new 替换.

**类型一致性:** 全计划统一使用 `kv_tree_block{hash,pos0,pos1,refcount,heat,last_used,on_disk,pinned,transient,bytes,path,tokens,data}`、`kv_tree_anchor{blk_hash,pos,kind,refcount,heat,last_used,on_disk,pinned,transient,bytes,path,data_tgt,data_dft,data_spec}`、`kv_tree_io` 六方法; 计数一律用 `bytes` 字段 (与 data vector 无关, SSD 上 vector 为空); 锚点键一律 `(blk_hash, pos)`; 恢复点一律 `res.C` / `res.heal`.

## Execution Handoff

Plan complete and saved to `D:\LLM\Backend\v100-collab\artifacts\t32-tree-plan-stage2.md`. 两种执行方式:

1. **Subagent-Driven (推荐)** - 每任务派新 subagent, 任务间双阶段审查 (与阶段 0/1 相同流程)
2. **Inline Execution** - 本会话内按 executing-plans 批量执行 + 检查点

执行前提: 分支 `t32-stage2` 从 master `52b7bf7de` 起; 提交按 Global Constraints 自动进行; push/部署/合并需另行批准.
