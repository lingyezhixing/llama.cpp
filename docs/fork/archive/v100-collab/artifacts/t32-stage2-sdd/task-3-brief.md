### Task 3: 检查点收编 + 稀疏化 + 分叉场景

**Files:**
- Modify: `tools/server/server-kv-tree.h` (加 `containing_block`)
- Modify: `tools/server/server-kv-tree.cpp` (`containing_block` + `park` 收编逻辑)
- Modify: `tests/test-t32-tree.cpp` (`scenario_fork` / `scenario_sparsify` + model 模式串联)

**Interfaces:**
- Consumes: Task 2 的 `store_anchor`/`restore`/harness 助手
- Produces: `kv_tree::containing_block`; park 的收编语义 (排序/`anchor_step`/冲突留靠前/锚点 refcount 覆盖)

- [ ] **Step 1: 头文件加 `containing_block`**

private 段 `store_anchor` 之后加:

```cpp
    uint64_t containing_block(llama_pos pos) const;
```

- [ ] **Step 2: `server-kv-tree.cpp` 加 `containing_block` 与收编逻辑**

在 `store_anchor` 之后插入:

```cpp
uint64_t kv_tree::containing_block(llama_pos pos) const {
    if (pos <= 0) {
        return 0;
    }

    auto it = blocks_at.upper_bound(pos);
    if (it == blocks_at.begin()) {
        return 0;
    }
    --it;

    for (const uint64_t hash : it->second) {
        const auto b = blocks.find(hash);
        if (b != blocks.end() && b->second.pos0 < pos && pos <= b->second.pos1) {
            return hash;
        }
    }

    return 0;
}
```

`park` 中, 在 `if (!store_anchor(tip, L, KV_TREE_ANCHOR_TIP, ...)) { ... }` 之后、`kv_tree_seq & s = seqs[tip];` 之前插入:

```cpp
    // adopt checkpoint candidates: sort by pos, greedy with anchor_step, conflict keeps the earlier one
    std::vector<kv_tree_anchor_in> cand = checkpoints;
    std::sort(cand.begin(), cand.end(), [](const kv_tree_anchor_in & a, const kv_tree_anchor_in & b) {
        return a.pos < b.pos;
    });

    std::vector<std::pair<uint64_t, llama_pos>> touched;
    touched.emplace_back(tip, L);

    llama_pos last_kept = -1;

    for (const kv_tree_anchor_in & c : cand) {
        if (c.pos <= 0 || c.pos > L || c.data_tgt.empty()) {
            continue;
        }

        const uint64_t blk = containing_block(c.pos);
        if (blk == 0) {
            st.anchors_skipped++;
            continue;
        }

        llama_pos prev = -1;
        for (const auto & kv : anchors) {
            if (kv.second.pos < c.pos && kv.second.pos > prev) {
                prev = kv.second.pos;
            }
        }

        const llama_pos prev_kept = prev > last_kept ? prev : last_kept;

        if (prev_kept >= 0 && c.pos - prev_kept < cfg.anchor_step) {
            st.anchors_skipped++;
            continue;
        }

        if (!store_anchor(blk, c.pos, KV_TREE_ANCHOR_MESSAGE, std::vector<uint8_t>(c.data_tgt), std::vector<uint8_t>(c.data_dft))) {
            st.anchors_skipped++;
            continue;
        }

        touched.emplace_back(blk, c.pos);
        last_kept = c.pos;
    }

    // anchors that already cover this sequence keep their refcount in sync
    for (auto & kv : anchors) {
        if (kv.second.pos > L) {
            continue;
        }

        bool in_chain = false;
        for (const uint64_t hash : h) {
            if (hash == kv.first.first) {
                in_chain = true;
                break;
            }
        }
        if (!in_chain) {
            continue;
        }

        bool was_touched = false;
        for (const auto & t : touched) {
            if (t.first == kv.first.first && t.second == kv.second.pos) {
                was_touched = true;
                break;
            }
        }
        if (!was_touched) {
            kv.second.refcount++;
            touched.emplace_back(kv.first.first, kv.second.pos);
        }
    }
```

注意: `park` 的签名里 `(void) checkpoints;` 要删掉.

- [ ] **Step 3: harness 加 `scenario_fork` 与 `scenario_sparsify`**

在 `scenario_tip` 之后插入:

```cpp
static std::vector<llama_token> make_fork_tokens(int n_shared, int n_tail, int salt) {
    std::vector<llama_token> t = make_tokens(n_shared, 0);
    for (int i = 0; i < n_tail; ++i) {
        t.push_back(10 + (i * 13 + salt) % 1000);
    }
    return t;
}

static int scenario_fork(llama_context * ctx, const kv_tree_config & cfg_in) {
    fprintf(stderr, "[t32-tree] scenario: fork\n");

    kv_tree_config cfg = cfg_in;
    cfg.anchor_step = 512;

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tok_a  = make_fork_tokens(1024, 1024, 0);
    const auto tok_b  = make_fork_tokens(1024, 1024, 1);
    const auto tok_a2 = make_fork_tokens(1024, 512,  2);

    prefill(ctx, 0, tok_a, 0, 512);

    const kv_tree_anchor_in ck = capture_partial(ctx, 0, 512);

    prefill(ctx, 0, tok_a, 512, 512);
    check(tree.park(io, nullptr, tok_a, { ck }), "fork: park A");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, tok_b, 0, 512);
    check(tree.park(io, nullptr, tok_b, {}), "fork: park B");

    {
        const auto base = run_baseline(ctx, tok_a, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tok_a, 42, 8);
        check_eq(r.res.C, 2048, "fork: A restores at the tip");
        check(r.gen == base, "fork: A tokens match the baseline");
    }
    {
        const auto base = run_baseline(ctx, tok_b, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tok_b, 42, 8);
        check_eq(r.res.C, 2048, "fork: B restores at the tip");
        check(r.gen == base, "fork: B tokens match the baseline");
    }
    {
        const auto base = run_baseline(ctx, tok_a2, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tok_a2, 42, 8);
        check_eq(r.res.C, 512, "fork: A' restores at the deepest usable anchor");
        check_eq(r.res.heal, 1024, "fork: A' asks for a heal at the fork point");
        check(r.gen == base, "fork: A' tokens match the baseline");
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

static int scenario_sparsify(llama_context * ctx, const kv_tree_config & cfg_in) {
    fprintf(stderr, "[t32-tree] scenario: sparsify\n");

    kv_tree_config cfg = cfg_in;
    cfg.anchor_step = 1024;

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tokens = make_tokens(2048, 0);

    prefill(ctx, 0, tokens, 0, 512);

    const kv_tree_anchor_in ck1 = capture_partial(ctx, 0, 512);

    prefill(ctx, 0, tokens, 512, 512);

    const kv_tree_anchor_in ck2 = capture_partial(ctx, 0, 1024);

    prefill(ctx, 0, tokens, 1024, 512);

    const kv_tree_anchor_in ck3 = capture_partial(ctx, 0, 1536);

    prefill(ctx, 0, tokens, 1536, 512);

    check(tree.park(io, nullptr, tokens, { ck1, ck2, ck3 }), "sparsify: park");

    {
        const std::vector<llama_token> short1(tokens.begin(), tokens.begin() + 1024);
        const auto base = run_baseline(ctx, short1, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, short1, 42, 8);
        check_eq(r.res.C, 512, "sparsify: [0, 1024) restores at 512");
        check(r.gen == base, "sparsify: [0, 1024) tokens match");
    }
    {
        const std::vector<llama_token> short2(tokens.begin(), tokens.begin() + 1536);
        const auto base = run_baseline(ctx, short2, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, short2, 42, 8);
        check_eq(r.res.C, 1536, "sparsify: [0, 1536) restores at 1536");
        check(r.gen == base, "sparsify: [0, 1536) tokens match");
    }
    {
        const auto base = run_baseline(ctx, tokens, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
        check_eq(r.res.C, 2048, "sparsify: full length restores at the tip");
        check(r.gen == base, "sparsify: full length tokens match");
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}
```

`main` 的 model 分支改为:

```cpp
    if (mode == "model") {
        int ret = 0;
        ret |= scenario_tip(ctx, cfg);
        ret |= scenario_fork(ctx, cfg);
        ret |= scenario_sparsify(ctx, cfg);
        return ret;
    }
```

- [ ] **Step 4: 构建并跑 model 模式**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```
Expected: tip / fork / sparsify 三场景全 PASS, exit 0. 若 fork 的 A' 出现 token 不匹配: 先看 logits 是否近并列 (打印 tokens 差异), 近并列属已知 T24 现象, 记录后继续; 否则是 bug.

- [ ] **Step 5: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : adopt checkpoint anchors in the kv tree" -m "Assisted-by: opencode"
```

---

