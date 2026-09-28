### Task 3: D12 锚点载荷返回 + 检查点表重建

**Files:**
- Modify: `tools/server/server-kv-tree.h:143-147` (struct), `:167` 附近 (无新方法)
- Modify: `tools/server/server-kv-tree.cpp:1274-1411` (restore)
- Modify: `tools/server/server-context.cpp:348-373` (prompt_restore_tree), `:1765` (调用点), `:990-1000` 附近 (`tree_ops` 成员 + 聚合日志)
- Modify: `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` (heal 断言加 `rebuilt`)
- Test: `tests/test-t32-tree.cpp` (新 `scenario_restore_anchors`, model 模式注册)

**Interfaces:**
- Consumes: 现有 `restore`, `load_payload`, `kv_tree_anchor`, `capture_partial`, `prefill`, `check`.
- Produces: `struct kv_tree_restore_anchor { llama_pos pos; std::vector<uint8_t> data_tgt; std::vector<uint8_t> data_dft; };`; `kv_tree_restore::anchors` 变为 `std::vector<kv_tree_restore_anchor>` (升序, 只含 ≤ C 的路径锚点); server 日志 `kv tree: rebuilt %zu context checkpoints` (Task 4/5 grep).

- [ ] **Step 1: 写失败测试 (engine, model 模式)**

`tests/test-t32-tree.cpp` 里 `scenario_unaligned` 之后插入:

```cpp
static int scenario_restore_anchors(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] scenario: restore anchors\n");

    const auto tokens = make_tokens(3072, 7);

    // park with a checkpoint candidate at 2048
    {
        kv_tree tree(cfg);
        kv_tree_io_llama io(ctx, 0);

        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        if (prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 2048), 0, 512) != 0) {
            return 1;
        }

        const kv_tree_anchor_in ck = capture_partial(ctx, 0, 2048);
        check(!ck.data_tgt.empty(), "restore-anchors: capture the partial state at 2048");

        if (prefill(ctx, 0, std::vector<llama_token>(tokens.begin() + 2048, tokens.end()), 2048, 512) != 0) {
            return 1;
        }

        check(tree.park(io, nullptr, tokens, { ck }), "restore-anchors: park with the checkpoint");

        const kv_tree_restore r = tree.restore(io, nullptr, tokens);
        check_eq(r.C, 3072, "restore-anchors: restore at the tip");
        check_eq((long long) r.anchors.size(), 2, "restore-anchors: two anchors on the path");

        bool found = false;
        bool sorted = true;
        llama_pos prev = -1;
        for (const auto & a : r.anchors) {
            if (a.pos <= prev) {
                sorted = false;
            }
            prev = a.pos;
            if (a.pos == 2048) {
                found = true;
                check(a.data_tgt == ck.data_tgt, "restore-anchors: the payload matches the parked checkpoint");
            }
        }
        check(found, "restore-anchors: the checkpoint anchor is returned");
        check(sorted, "restore-anchors: anchors are sorted by position");
    }

    // same with the payloads on disk
    {
        kv_tree_config cfg2 = cfg;
        cfg2.disk_dir   = (std::filesystem::temp_directory_path() / "t32-tree-anchors").string();
        cfg2.ram_limit  = 8 * 1024;    // force demotion
        cfg2.disk_limit = 1ull << 30;  // the real 2B KV for 3072 tokens is ~150 MB

        std::error_code ec;
        std::filesystem::remove_all(cfg2.disk_dir, ec);

        kv_tree tree(cfg2);
        kv_tree_io_llama io(ctx, 0);

        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        if (prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 2048), 0, 512) != 0) {
            return 1;
        }

        const kv_tree_anchor_in ck = capture_partial(ctx, 0, 2048);

        if (prefill(ctx, 0, std::vector<llama_token>(tokens.begin() + 2048, tokens.end()), 2048, 512) != 0) {
            return 1;
        }

        check(tree.park(io, nullptr, tokens, { ck }), "restore-anchors: park with a small ram budget");
        check(tree.stats().anchors_disk + tree.stats().blocks_disk > 0, "restore-anchors: data went to disk");

        const kv_tree_restore r = tree.restore(io, nullptr, tokens);
        check_eq(r.C, 3072, "restore-anchors: restore at the tip (ssd)");

        bool found = false;
        for (const auto & a : r.anchors) {
            if (a.pos == 2048) {
                found = true;
                check(a.data_tgt == ck.data_tgt, "restore-anchors: the disk payload matches the parked checkpoint");
            }
        }
        check(found, "restore-anchors: the disk anchor is returned");

        std::filesystem::remove_all(cfg2.disk_dir, ec);
    }

    return 0;
}
```

`main` 的 model 分派 (862-869) 加一行:

```cpp
        ret |= scenario_ssd(ctx, cfg);
        ret |= scenario_restore_anchors(ctx, cfg);
        ret |= scenario_unaligned(ctx, cfg);
```

- [ ] **Step 2: 运行确认失败**

Run: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree; if ($?) { $env:CUDA_VISIBLE_DEVICES='1'; & 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096 2>&1 | Select-String 'FAIL|restore-anchors' }`
Expected: 编译失败 (`r.anchors` 是 `std::vector<llama_pos>`, 无 `.pos/.data_tgt`).

- [ ] **Step 3: 实现 (engine + server + heal 断言)**

(3a) `server-kv-tree.h:143-147` 替换:

```cpp
struct kv_tree_restore_anchor {
    llama_pos pos = 0;
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
};

struct kv_tree_restore {
    llama_pos C = -1;              // restore point, -1 = caller must do a full prefill
    llama_pos heal = -1;           // capture an anchor when the prefill crosses this pos
    std::vector<kv_tree_restore_anchor> anchors; // path anchors up to C, ascending (D12)
};
```

(3b) `server-kv-tree.cpp` restore 的锚点收集 (1296-1311) 替换:

```cpp
    llama_pos C = -1;
    uint64_t  c_hash = 0;
    std::vector<std::pair<uint64_t, llama_pos>> path_anchors;

    for (const uint64_t hash : cand) {
        for (auto it = anchors.lower_bound(std::make_pair(hash, 0));
             it != anchors.end() && it->first.first == hash; ++it) {
            if (it->second.pos <= m.deep) {
                path_anchors.emplace_back(hash, it->second.pos);
                if (it->second.pos > C) {
                    C = it->second.pos;
                    c_hash = hash;
                }
            }
        }
    }
```

在 `if (ok) { ... }` 块之后、解 pin 之前 (1373 之后) 插入载荷收集:

```cpp
    std::vector<kv_tree_restore_anchor> out_anchors;
    if (ok) {
        for (const auto & key : path_anchors) {
            kv_tree_anchor & a = anchors.at(key);
            if (!load_payload(a)) {
                continue;   // payload unavailable: fewer rebuilt checkpoints
            }

            kv_tree_restore_anchor ra;
            ra.pos      = a.pos;
            ra.data_tgt = a.data_tgt;
            ra.data_dft = a.data_dft;
            out_anchors.push_back(std::move(ra));
        }
    }
```

尾部 (1402-1404) 替换:

```cpp
    res.C       = C;
    res.heal    = m.deep > C ? m.deep : -1;
    res.anchors = std::move(out_anchors);
```

(3c) `server-context.cpp:348-373` 替换 `prompt_restore_tree` 头部与重建逻辑:

```cpp
    bool prompt_restore_tree(kv_tree & tree, const server_tokens & tokens, int n_ckpt_max) {
        if (!lora.empty() || tokens.empty() || tokens.has_mtmd) {
            return false;
        }

        kv_tree_io_llama io_tgt(ctx_tgt, id);
        std::unique_ptr<kv_tree_io_llama> io_dft;
        kv_tree_io * io_dft_ptr = nullptr;
        if (ctx_dft != nullptr) {
            io_dft = std::make_unique<kv_tree_io_llama>(ctx_dft, id);
            io_dft_ptr = io_dft.get();
        }

        kv_tree_restore res = tree.restore(io_tgt, io_dft_ptr, tokens.get_tokens());
        if (res.C <= 0) {
            SLT_INF(*this, "kv tree: restore miss for %zu tokens, full prefill\n", tokens.size());
            return false;
        }

        prompt.tokens = server_tokens(llama_tokens(tokens.get_tokens().begin(), tokens.get_tokens().begin() + res.C), false);
        prompt.checkpoints.clear();

        if (n_ckpt_max > 0) {
            std::list<common_prompt_checkpoint> cks;

            for (auto & a : res.anchors) {
                if (a.pos <= 0 || a.pos >= res.C) {
                    continue;
                }

                common_prompt_checkpoint ck;
                ck.id_task = -1;
                ck.update_pos(a.pos, 0, a.pos - 1);
                ck.data_tgt = std::move(a.data_tgt);
                ck.data_dft = std::move(a.data_dft);
                cks.push_back(std::move(ck));
            }

            while ((int) cks.size() > n_ckpt_max) {
                cks.pop_front();
            }

            prompt.checkpoints = std::move(cks);

            if (!prompt.checkpoints.empty()) {
                SLT_INF(*this, "kv tree: rebuilt %zu context checkpoints\n", prompt.checkpoints.size());
            }
        }

        tree_heal = res.heal > res.C ? res.heal : -1;

        SLT_INF(*this, "kv tree: restored %d tokens (heal = %d)\n", (int) res.C, (int) tree_heal);
        return true;
    }
```

调用点 `server-context.cpp:1765`:

```cpp
                    if (!task.params.cache_prompt || !ret->prompt_restore_tree(*tree, task.tokens, params_base.n_ctx_checkpoints)) {
```

(3d) 聚合统计日志: 在 `server_context` 的 `std::unique_ptr<kv_tree> tree;` (994) 旁加成员:

```cpp
    int64_t tree_ops = 0;
```

在 `get_available_slot` 的 `if (update_cache) { ... }` 块之后 (1779 之后) 插入:

```cpp
            if (tree && ++tree_ops % 64 == 0) {
                SRV_INF("kv tree stats: %s\n", tree->stats_line().c_str());
            }
```

(3e) heal 验收断言 (`t32-stage3-ab.ps1` heal 段, 在 `HEAL METRICS` 行附近):

```powershell
            $rebuilt = ($log | Select-String 'rebuilt \d+ context checkpoints').Count
            Write-Output "HEAL REBUILT lines=$rebuilt"
            Assert ($rebuilt -ge 1) 'heal: context checkpoints rebuilt after the fork restore (D12)'
```

- [ ] **Step 4: 运行确认通过**

Run (harness):

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096 2>&1 | Select-String 'FAIL|scenario: restore anchors'
```

Expected: 0 FAIL; `restore-anchors` 全部 PASS.

Run (server + heal 验收):

```powershell
$repo='D:\LLM\Backend\src\llama.cpp-my'
Remove-Item -Force "$repo\build\bin\Release\llama-server-impl.dll" -ErrorAction SilentlyContinue
& '<TEMP>\v100\build_server.cmd'
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode heal | Select-String 'REBUILT|RESULT|FAIL'
```

Expected: `RESULT heal: 0 failure(s)` 且 `REBUILT lines>=1`.

- [ ] **Step 5: 提交 (两个)**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : return kv tree anchor payloads on restore" -m "Assisted-by: opencode"

git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-context.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : rebuild context checkpoints after a tree restore" -m "Assisted-by: opencode"
```

(脚本改动在频道 artifacts, 不入 git.)

---

