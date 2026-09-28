### Task 1: 树模块增量（drop_seq / 流式恢复 + bytes_load / debug / ctor 校验）

**Files:**
- Modify: `tools/server/server-kv-tree.h`
- Modify: `tools/server/server-kv-tree.cpp`
- Test: `tests/test-t32-tree.cpp`

**Interfaces:**
- Consumes: 无（纯模块内改动）
- Produces:
  - `bool kv_tree::drop_seq(const std::vector<llama_token> & tokens)`（public）
  - `bool kv_tree::set_block_payload(kv_tree_io & io, kv_tree_block & b, bool append, std::vector<uint8_t> & scratch)`（private）
  - 不变：`park/restore/capture_anchor/stats/dump` 签名。

- [ ] **Step 1: 写失败测试（drop + 流式恢复）**

在 `tests/test-t32-tree.cpp` 的 `run_logic_evict()` 之后（:139 之后）插入两个函数：

```cpp
static void run_logic_drop() {
    fprintf(stderr, "[t32-tree] logic: drop_seq\n");

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 1 << 20;
    cfg.disk_dir    = "";

    kv_tree tree(cfg);

    const auto tok   = make_tokens(1536, 0);
    auto       tok_x = tok;
    for (int i = 0; i < 512; ++i) {
        tok_x.push_back((llama_token) (10 + (i * 7 + 99) % 1000));
    }
    auto tok_mid   = std::vector<llama_token>(tok.begin(), tok.begin() + 1400);
    auto tok_other = make_tokens(1536, 1);

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok, {}), "drop: park A");
    }

    check(!tree.drop_seq(tok_other),                    "drop: unrelated tokens are a no-op");
    check(tree.drop_seq(tok_mid),                       "drop: mid-tail tokens drop the sequence");
    check_eq(tree.stats().evicted_seqs, 1,              "drop: sequence was dropped");
    check_eq(tree.stats().blocks_ram, 0,                "drop: its blocks were released");
    check(!tree.drop_seq(tok),                          "drop: second drop is a no-op");

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok, {}), "drop: re-park A");
    }
    check(tree.drop_seq(tok_x),                         "drop: superstring drops the sequence");
    check_eq(tree.stats().evicted_seqs, 2,              "drop: sequence was dropped again");
}

static void run_logic_stream() {
    fprintf(stderr, "[t32-tree] logic: streaming restore\n");

    const auto dir = std::filesystem::temp_directory_path() / "t32-tree-logic-stream";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 4096;      // fits one fake block (4096 B), nothing more
    cfg.disk_limit  = 1 << 20;
    cfg.disk_dir    = dir.string();

    kv_tree tree(cfg);

    const auto tok = make_tokens(1536, 0);

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok, {}),                          "stream: park with spill");
        check_eq(tree.stats().blocks_disk, 3,                           "stream: all blocks spilled to disk");
        check((size_t) tree.stats().bytes_ram <= cfg.ram_limit,         "stream: park respects the ram budget");
    }

    {
        kv_tree_io_fake io;
        const auto r = tree.restore(io, nullptr, tok);
        check_eq(r.C, 1536,                                             "stream: restore hits the tip");
        check_eq(r.heal, -1,                                            "stream: no heal needed for an exact tip");
        check_eq(io.set_range_bytes, 3 * 4096,                          "stream: all blocks were written to the io");
        check_eq(io.set_partial_calls, 1,                               "stream: the anchor was written");
        check((size_t) tree.stats().bytes_ram <= cfg.ram_limit,         "stream: restore respects the ram budget");
        check_eq(tree.stats().bytes_load, 3 * 4096,                "stream: bytes_load counts payloads moved from disk");
        check_eq(tree.stats().blocks_disk, 3,                           "stream: streamed blocks stay on disk");
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}
```

说明：fake io 的载荷 = `(p1-p0)*8` 字节（:46-49），512 token 块 = 4096 B；`get_partial` = 64 B（:57-60）。预算 4096 B 下 park 时 3 个块必然全部降级到磁盘（降级顺序与哈希表遍历无关：两次降级后仍超预算），锚点 64 B 留 RAM。两个函数均用文件里已有的全局 `n_fail` 计数（:14），不需要额外封装。

- [ ] **Step 2: 接入 `run_logic()` 并跑失败**

`run_logic()`（:141-193）中 `run_logic_evict();`（:144）之后加：

```cpp
    run_logic_drop();
    run_logic_stream();
```

Run: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree`
Expected: 编译失败（`drop_seq` 不存在）或链接失败 - 这是预期的失败。

- [ ] **Step 3: 实现 drop_seq + 流式恢复 + 计数**

`tools/server/server-kv-tree.h`：public 区 `capture_anchor` 之后（:162 后）加：

```cpp
    // release the stored sequence that matches a prefix of tokens; no-op when nothing matches
    bool drop_seq(const std::vector<llama_token> & tokens);
```

private 区 `load_payload` 声明旁（:199-200）加：

```cpp
    bool set_block_payload(kv_tree_io & io, kv_tree_block & b, bool append, std::vector<uint8_t> & scratch);
```

`tools/server/server-kv-tree.cpp`：
1. 在 `remove_seq`（:665-693）之后加：

```cpp
bool kv_tree::drop_seq(const std::vector<llama_token> & tokens) {
    if (tokens.empty()) {
        return false;
    }

    const kv_tree_match m = match(tokens);

    uint64_t tip = 0;
    if (m.n_part > 0) {
        tip = m.part_hash;
    } else if (m.n_full > 0) {
        tip = m.path.back();
    } else {
        return false;
    }

    const auto it = seqs.find(tip);
    if (it == seqs.end()) {
        return false;
    }

    remove_seq(it);
    return true;
}
```

2. `load_payload` 计数（移自 restore 循环，避免重复计数；磁盘 -> RAM 的搬运才计数）：
   - `load_payload(kv_tree_block &)`（:550-576）的 `b.transient = true;` 之前加：
   ```cpp
   st.bytes_load += (int64_t) b.data.size();
   ```
   - `load_payload(kv_tree_anchor &)`（:578-606）的 `a.transient = true;` 之前加：
   ```cpp
   st.bytes_load += (int64_t) anchor_bytes(a);
   ```

3. 在 `load_payload(kv_tree_block &)` 之后加新 helper：

```cpp
// feed one block payload to the io; keep it resident only when it fits the ram budget,
// otherwise stream it from disk through a scratch buffer
bool kv_tree::set_block_payload(kv_tree_io & io, kv_tree_block & b, bool append, std::vector<uint8_t> & scratch) {
    if (b.on_disk && (size_t) st.bytes_ram + b.bytes > cfg.ram_limit) {
        std::vector<uint8_t> buf;
        if (!read_disk(b.path, buf)) {
            fprintf(stderr, "[kv-tree] failed to read block %016" PRIx64 " from disk\n", b.hash);
            st.disk_errors++;
            return false;
        }

        kv_env e;
        if (!envelope_parse(buf, e) || e.magic != T32_ENV_MAGIC_BLOCK || e.hash != b.hash || e.p0 != b.pos0 || e.p1 != b.pos1 || e.n_dft != 0 || e.n_spec != 0) {
            fprintf(stderr, "[kv-tree] block %016" PRIx64 " payload check failed, dropping it\n", b.hash);
            st.disk_errors++;
            return false;
        }

        scratch.assign(e.tgt, e.tgt + e.n_tgt);
        st.bytes_load += (int64_t) scratch.size();
        return io.set_range(scratch.data(), scratch.size(), append);
    }

    if (!load_payload(b)) {
        return false;
    }

    return io.set_range(b.data.data(), b.data.size(), append);
}
```

4. `restore()`（:1226-1245）替换两个块装载段：

```cpp
    std::vector<uint8_t> scratch;

    // blocks covering [0, C): the last one may overshoot and is trimmed below
    for (const uint64_t hash : m.path) {
        kv_tree_block & b = blocks[hash];
        if (b.pos0 >= C) {
            break;
        }
        if (!set_block_payload(io_tgt, b, b.pos0 != 0, scratch)) {
            fprintf(stderr, "[kv-tree] restore failed: cannot load block [%d, %d)\n", b.pos0, b.pos1);
            ok = false;
            break;
        }
    }

    if (ok && m.n_part > 0 && C > blocks[m.part_hash].pos0) {
        kv_tree_block & b = blocks[m.part_hash];
        if (!set_block_payload(io_tgt, b, true, scratch)) {
            fprintf(stderr, "[kv-tree] restore failed: cannot load the partial block [%d, %d)\n", b.pos0, b.pos1);
            ok = false;
        }
    }
```

（删除原 :1236 的 `st.bytes_load += (int64_t) b.data.size();`。）

5. `park()`（:931）与 `restore()`（:1167）函数体末尾（`return true;`/成功 `return res;` 之前）加：

```cpp
    if (cfg.debug) {
        dump();
    }
```

（park 里放在最后一个 `return true;` 之前；restore 里放在 `st.restore_hits++...` 之后、`return res;` 之前。）

6. 构造函数（文件内 `kv_tree::kv_tree(const kv_tree_config & cfg)`，实施时定位）加校验：

```cpp
    if (cfg.chunk <= 0 || cfg.anchor_step < 0) {
        fprintf(stderr, "[kv-tree] invalid config: chunk = %d, anchor_step = %d\n", cfg.chunk, cfg.anchor_step);
        GGML_ABORT("invalid kv tree config");
    }
```

- [ ] **Step 4: 跑测试通过**

Run: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree` 然后 `& "$repo\build\bin\Release\test-t32-tree.exe" --mode logic`
Expected: 全部 PASS（原 18 项 + 新 14 项），exit 0。注意 `run_logic_stream` 用磁盘临时目录（`%TEMP%\t32-tree-logic-stream`），结束后应已删除。

再跑模型模式回归（确认流式改动未破坏真机路径）：
Run: `CUDA_VISIBLE_DEVICES=0`（PowerShell: `$env:CUDA_VISIBLE_DEVICES='0'`）`test-t32-tree.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096`
Expected: 43/43 PASS（与阶段 2 最终 head 相同）。

- [ ] **Step 5: 提交**

```powershell
git -C $repo add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git -C $repo commit -m "server : stream disk payloads during kv tree restore" -m "Assisted-by: opencode"
```

---

