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
        if (b.on_disk || b.data.empty()) {
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
        if (a.on_disk || a.data_tgt.empty()) {
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

