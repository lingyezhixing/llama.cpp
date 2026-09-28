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
    // no pruning needed here: the spacing rule keeps prev <= pos - fork_step,
    // so no anchor can sit between prev and the new one
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

