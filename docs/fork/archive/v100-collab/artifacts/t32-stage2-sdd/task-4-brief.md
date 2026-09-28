### Task 4: 自愈捕获 + 分叉提升/剪枝

**Files:**
- Modify: `tools/server/server-kv-tree.h` (加 `capture_anchor` / `promote_prune` / `remove_anchor` / `last_promote` 成员)
- Modify: `tools/server/server-kv-tree.cpp` (`capture_anchor` / `promote_prune` / `remove_anchor` + `anchor_bytes` 助手)
- Modify: `tests/test-t32-tree.cpp` (`scenario_fork` 的 A' 块替换为带自愈的版本)

**Interfaces:**
- Consumes: Task 2/3 的 `containing_block`/`store_anchor`/`restore`
- Produces: `kv_tree::capture_anchor`; 自愈后同一分叉点第二次恢复免费

**设计注 (审阅重点):** spec §4 的 "删除 (前一锚点, P) 之间无分叉价值的中间锚点" 在本计划里实现为**保守版**: 维护 `last_promote` (上次提升位置), 提升 P 时删除 `(last_promote, P)` 间 `refcount<2 && kind != TIP && !pinned` 的锚点; 首次提升 (last_promote < 0) 只记录不删, 避免一次提升就清掉所有单序列检查点.

- [ ] **Step 1: 头文件加声明与成员**

public 段 `restore` 之后加:

```cpp
    // capture the state at pos (the io state must be exactly at pos) as a fork anchor;
    // tokens is the sequence content, used to attach the anchor to the right chain block
    bool capture_anchor(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens, llama_pos pos);
```

private 段加:

```cpp
    void promote_prune(llama_pos pos);

    void remove_anchor(std::map<std::pair<uint64_t, llama_pos>, kv_tree_anchor>::iterator it);
```

成员区加:

```cpp
    llama_pos last_promote = -1;
```

- [ ] **Step 2: `server-kv-tree.cpp` 加实现**

文件头 include 区加 `#include <filesystem>`; 在 `store_anchor` 之后插入:

```cpp
static size_t anchor_bytes(const kv_tree_anchor & a) {
    return a.bytes;
}

void kv_tree::remove_anchor(std::map<std::pair<uint64_t, llama_pos>, kv_tree_anchor>::iterator it) {
    kv_tree_anchor & a = it->second;

    if (a.transient) {
        st.bytes_ram -= (int64_t) a.bytes;
        st.anchors_ram--;
    }

    if (a.on_disk) {
        std::error_code ec;
        std::filesystem::remove(a.path, ec);
        st.bytes_disk -= (int64_t) a.bytes;
        st.anchors_disk--;
    } else {
        st.bytes_ram -= (int64_t) a.bytes;
        st.anchors_ram--;
    }

    anchors.erase(it);
}

bool kv_tree::capture_anchor(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens, llama_pos pos) {
    if (pos <= 0) {
        return false;
    }
    if (io_tgt.pos_max() != pos) {
        fprintf(stderr, "[kv-tree] capture refused at %d: sequence end %d\n", pos, io_tgt.pos_max());
        return false;
    }

    std::vector<uint64_t> chain;
    chain_hashes(tokens, cfg.chunk, chain);

    const uint64_t blk = containing_block(pos, &chain);
    if (blk == 0) {
        fprintf(stderr, "[kv-tree] capture refused at %d: no chain block contains this position\n", pos);
        return false;
    }

    const auto key = std::make_pair(blk, pos);

    const auto it = anchors.find(key);
    if (it != anchors.end()) {
        it->second.heat++;
        it->second.last_used = ++now;
        return true;
    }

    llama_pos prev = -1;
    for (const auto & kv : anchors) {
        if (kv.second.pos < pos && kv.second.pos > prev) {
            prev = kv.second.pos;
        }
    }

    if (prev >= 0 && pos - prev < cfg.anchor_step) {
        st.anchors_skipped++;
        return true;
    }

    std::vector<uint8_t> tgt;
    std::vector<uint8_t> dft;

    if (!io_tgt.get_partial(tgt)) {
        return false;
    }
    if (io_dft != nullptr) {
        io_dft->get_partial(dft);
    }

    if (!store_anchor(blk, pos, KV_TREE_ANCHOR_ONDEMAND, std::move(tgt), std::move(dft))) {
        return false;
    }

    promote_prune(pos);

    return true;
}

void kv_tree::promote_prune(llama_pos pos) {
    if (last_promote < 0) {
        last_promote = pos;
        return;
    }

    for (auto it = anchors.begin(); it != anchors.end(); ) {
        kv_tree_anchor & a = it->second;

        if (a.pos > last_promote && a.pos < pos && a.kind != KV_TREE_ANCHOR_TIP && a.refcount < 2 && !a.pinned) {
            remove_anchor(it++);
        } else {
            ++it;
        }
    }

    last_promote = pos;
}
```

- [ ] **Step 3: 替换 `scenario_fork` 的 A' 块**

用下面替换 Task 3 里 `// A' forks at 1024` 的那个 `{ ... }` 块:

```cpp
    {
        const auto base = run_baseline(ctx, tok_a2, 42, 8);

        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

        const auto r = tree.restore(io, nullptr, tok_a2);
        check_eq(r.res.C, 512, "fork: A' restores at the deepest usable anchor");
        check_eq(r.res.heal, 1024, "fork: A' asks for a heal at the fork point");

        prefill(ctx, 0, tok_a2, (int) r.res.C, 512);

        check(tree.capture_anchor(io, nullptr, tok_a2, 1024), "fork: heal capture at the fork point");

        prefill(ctx, 0, tok_a2, 1024, 512);

        const auto gen = generate(ctx, 0, 42, (llama_pos) tok_a2.size(), 8);
        check(gen == base, "fork: A' tokens match the baseline");

        const auto r2 = run_tree_path(ctx, tree, io, tok_a2, 42, 8);
        check_eq(r2.res.C, 1024, "heal: A' restores at the self-healed anchor");
        check_eq(r2.res.heal, -1, "heal: no second heal needed");
        check(r2.gen == base, "heal: tokens still match the baseline");
    }
```

- [ ] **Step 4: 构建并跑**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```
Expected: tip / fork / sparsify 全 PASS, 其中 `heal:` 两条必须 PASS (自愈锚点生效, 第二次恢复无重放).

- [ ] **Step 5: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : self-heal kv tree anchors at fork points" -m "Assisted-by: opencode"
```

---

