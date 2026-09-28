### Task 1: 模块小修 (D22/D23) + 统计行

**Files:**
- Modify: `tools/server/server-kv-tree.h:167` (stats 附近加声明)
- Modify: `tools/server/server-kv-tree.cpp:470-510` (write_disk), `:363-368` (capture_anchor prev), `:1165-1170` (park prev), 文件尾 (stats_line 实现)
- Test: `tests/test-t32-tree.cpp` (新 `run_logic_fixes`, 注册到 `run_logic`)

**Interfaces:**
- Consumes: 现有 `kv_tree_stats`, `kv_tree_io_fake`, `make_tokens`, `check`/`check_eq`.
- Produces: `std::string kv_tree::stats_line() const` (Task 4/5 日志解析依赖 `parks=` 等字段名); 修复后的计数语义 (Task 3 测试依赖 `disk_errors` 单计数).

- [ ] **Step 1: 写失败测试**

在 `tests/test-t32-tree.cpp` 的 `run_logic()` 之前插入 (放在 `run_logic_capture` 之后):

```cpp
static void run_logic_fixes() {
    fprintf(stderr, "[t32-tree] logic: fixes\n");

    // disk_errors must count one error per failed write, not two
    {
        const auto file = std::filesystem::temp_directory_path() / "t32-tree-diskerr";
        std::error_code ec;
        std::filesystem::remove(file, ec);
        {
            std::FILE * f = fopen(file.string().c_str(), "wb");
            if (f != nullptr) {
                fputc('x', f);
                fclose(f);
            }
        }

        kv_tree_config cfg;
        cfg.ram_limit  = 0;           // force demotion
        cfg.disk_limit = 1ull << 20;
        cfg.disk_dir   = file.string(); // a file, not a directory: every write fails

        kv_tree tree(cfg);

        kv_tree_io_fake io;
        io.max_pos = 1023;
        check(!tree.park(io, nullptr, make_tokens(1024, 0), {}), "fixes: park refused when the disk tier fails");
        check_eq(tree.stats().disk_errors, 2, "fixes: one disk error per failed write (block + anchor)");

        std::filesystem::remove(file, ec);
    }

    // anchor spacing must be scoped to the chain, not global
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 4096;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);

        const auto tok_a = make_tokens(512, 0);
        const auto tok_b = make_tokens(2048, 1);

        {
            kv_tree_io_fake io;
            io.max_pos = 511;
            check(tree.park(io, nullptr, tok_a, {}), "fixes: park chain A (tip anchor at 512)");
        }

        {
            kv_tree_anchor_in ck;
            ck.pos      = 1024;
            ck.data_tgt = kv_tree_io_fake::pattern(0, 64);

            kv_tree_io_fake io;
            io.max_pos = 2047;
            check(tree.park(io, nullptr, tok_b, { ck }), "fixes: park chain B with a candidate at 1024");
        }

        // chain B now has anchors at 1024 and 2048; chain A has one at 512
        check_eq(tree.stats().anchors_added, 3, "fixes: the cross-chain anchor does not suppress the candidate");
    }

    // capture spacing must be chain-scoped too, and same-chain spacing must still apply
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 4096;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);

        const auto tok_a = make_tokens(512, 0);
        const auto tok_b = make_tokens(2048, 1);

        {
            kv_tree_io_fake io;
            io.max_pos = 511;
            check(tree.park(io, nullptr, tok_a, {}), "fixes: park chain A");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 2047;
            check(tree.park(io, nullptr, tok_b, {}), "fixes: park chain B");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 1535;
            check(tree.capture_anchor(io, nullptr, tok_b, 1536), "fixes: cross-chain anchor does not block the capture");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 2559;
            check(!tree.capture_anchor(io, nullptr, tok_b, 2560), "fixes: same-chain spacing still refuses");
        }
    }

    // stats summary line
    {
        kv_tree_config cfg;
        cfg.ram_limit = 1ull << 20;

        kv_tree tree(cfg);

        kv_tree_io_fake io;
        io.max_pos = 1023;
        check(tree.park(io, nullptr, make_tokens(1024, 0), {}), "fixes: park for the stats line");

        const std::string s = tree.stats_line();
        check(s.find("parks=1") != std::string::npos, "fixes: stats line has parks");
        check(s.find("anchors=1") != std::string::npos, "fixes: stats line has anchors");
        check(s.find("disk_err=0") != std::string::npos, "fixes: stats line has disk_err");
    }
}
```

并在 `run_logic()` (284-287) 里追加调用:

```cpp
    run_logic_capture();
    run_logic_fixes();
```

- [ ] **Step 2: 运行确认失败**

Run: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree; if ($?) { & 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic 2>&1 | Select-String 'FAIL|fixes' }`
Expected: 编译失败 (`stats_line` 不存在) 或运行 FAIL: `fixes: one disk error per failed write` (got 4), `fixes: the cross-chain anchor does not suppress the candidate` (got 2), `fixes: cross-chain anchor does not block the capture` (FAIL).

- [ ] **Step 3: 实现修复与 stats_line**

`server-kv-tree.h` 在 `const kv_tree_stats & stats() const { return st; }` 之后加:

```cpp
    std::string stats_line() const;
```

`server-kv-tree.cpp` 三处:

(1) `write_disk` (470-510) 删掉两个自增 (调用方已计数):

```cpp
bool kv_tree::write_disk(const std::string & path, const std::vector<uint8_t> & buf) {
    const std::string tmp = path + ".tmp";

    {
        std::error_code ec;   // the tier subdirectories are created on demand
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
        if (ec) {
            fprintf(stderr, "[kv-tree] failed to create the directory for %s: %s\n", tmp.c_str(), ec.message().c_str());
            return false;
        }
    }

    std::FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        fprintf(stderr, "[kv-tree] failed to open %s for writing\n", tmp.c_str());
        return false;
    }
```

(2) `capture_anchor` 的 prev (363-368) 改为链作用域:

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
```

(3) `park` 候选采纳的 prev (1165-1170) 改为链作用域:

```cpp
        llama_pos prev = -1;
        for (const auto & kv : anchors) {
            if (kv.second.pos >= c.pos || kv.second.pos <= prev) {
                continue;
            }
            if (std::find(h.begin(), h.end(), kv.first.first) != h.end()) {
                prev = kv.second.pos;
            }
        }
```

(4) 文件尾 (dump 之前) 加实现:

```cpp
std::string kv_tree::stats_line() const {
    char buf[512];

    snprintf(buf, sizeof(buf),
             "parks=%" PRId64 " ok=%" PRId64 " refused=%" PRId64
             " restore=%" PRId64 " hits=%" PRId64 " miss=%" PRId64
             " anchors=%" PRId64 " skipped=%" PRId64 " reuse_tok=%" PRId64
             " store=%" PRId64 " load=%" PRId64
             " evicted=%" PRId64 "/%" PRId64 "/%" PRId64 " evict_refused=%" PRId64
             " disk_err=%" PRId64 " ram=%" PRId64 " disk=%" PRId64,
             st.park_calls, st.park_ok, st.park_refused,
             st.restore_calls, st.restore_hits, st.restore_miss,
             st.anchors_added, st.anchors_skipped, st.tokens_reused,
             st.bytes_store, st.bytes_load,
             st.evicted_anchors, st.evicted_blocks, st.evicted_seqs, st.evict_refused,
             st.disk_errors, st.bytes_ram, st.bytes_disk);

    return buf;
}
```

- [ ] **Step 4: 运行确认通过**

Run: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree; if ($?) { & 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic 2>&1 | Select-Object -Last 3 }`
Expected: `logic: N PASS, 0 FAIL` (N = 46 + 12 = 58).

- [ ] **Step 5: 提交**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : fix kv tree disk error counting and anchor spacing" -m "Assisted-by: opencode"
```

`stats_line` 与测试同属本任务, 但按仓库惯例拆第二个提交 (只含实现与断言行):

```powershell
# 注: 若与上一条同文件, 用 git add -p 不可行 -> 实际执行时把 stats_line 声明/实现/断言拆到本提交
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : add kv tree stats summary line" -m "Assisted-by: opencode"
```

(执行者注: 为让第二个提交干净, 可在 Step 3 时先只做修复, 提交后再加 `stats_line` + 其断言并提交. 两个提交都留在 `t32-stage4` 上.)

---

