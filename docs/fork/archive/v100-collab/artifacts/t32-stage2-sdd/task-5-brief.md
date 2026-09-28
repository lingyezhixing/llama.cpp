### Task 5: SSD 层 (信封/文件/移动语义) + 预算降级

**Files:**
- Modify: `tools/server/server-kv-tree.h` (加 payload 存取/落盘/结算声明; block/anchor 加 `transient`)
- Modify: `tools/server/server-kv-tree.cpp` (信封 + 文件 I/O + `load_payload`/`demote_*`/`settle` + `enforce_budget` 降级 + `restore` 接入)
- Modify: `tests/test-t32-tree.cpp` (`scenario_ssd`)

**Interfaces:**
- Consumes: Task 2-4 全部
- Produces: SSD 单份权威移动语义; `st.bytes_disk`/`blocks_disk`/`anchors_disk`/`disk_errors`/`bytes_store` 计数

- [ ] **Step 1: 头文件**

`kv_tree_block` 加一个字段 (放在 `pinned` 之后; `kv_tree_anchor` 已在 Task 4 加过):

```cpp
    bool        transient = false;
```

private 段: 把 `bool make_room(size_t need_ram);` 换成

```cpp
    bool enforce_budget();

    void park_rollback(const std::vector<std::pair<uint64_t, std::vector<uint8_t>>> & blobs,
                       const kv_tree_match & m,
                       const std::vector<std::pair<uint64_t, llama_pos>> & touched,
                       uint64_t tip);

    void remove_block(std::unordered_map<uint64_t, kv_tree_block>::iterator it);
```

再加:

```cpp
    bool load_payload(kv_tree_block & b);
    bool load_payload(kv_tree_anchor & a);
    bool demote_block(kv_tree_block & b);
    bool demote_anchor(kv_tree_anchor & a);
    void settle();

    bool write_disk(const std::string & path, const std::vector<uint8_t> & buf);
    bool read_disk(const std::string & path, std::vector<uint8_t> & out);
    std::string block_path(uint64_t hash) const;
    std::string anchor_path(uint64_t blk_hash, llama_pos pos) const;
```

- [ ] **Step 2: `server-kv-tree.cpp` 加信封与文件 I/O**

文件头 include 区加 `#include <filesystem>`; 在 `chain_hashes` 之后插入:

```cpp
static const uint32_t T32_ENV_MAGIC_BLOCK  = 0x42323354u;  // "T32B"
static const uint32_t T32_ENV_MAGIC_ANCHOR = 0x41323354u;  // "T32A"
static const uint32_t T32_ENV_VERSION      = 1;

static void put_u32(std::vector<uint8_t> & v, uint32_t x) {
    uint8_t b[4];
    memcpy(b, &x, 4);
    v.insert(v.end(), b, b + 4);
}

static void put_i32(std::vector<uint8_t> & v, int32_t x) {
    put_u32(v, (uint32_t) x);
}

static void put_u64(std::vector<uint8_t> & v, uint64_t x) {
    uint8_t b[8];
    memcpy(b, &x, 8);
    v.insert(v.end(), b, b + 8);
}

struct kv_env {
    uint32_t magic   = 0;
    uint64_t hash    = 0;
    llama_pos p0     = 0;
    llama_pos p1     = 0;
    uint32_t n_tgt   = 0;
    uint32_t n_dft   = 0;
    uint32_t n_spec  = 0;
    const uint8_t * tgt  = nullptr;
    const uint8_t * dft  = nullptr;
    const uint8_t * spec = nullptr;
};

static std::vector<uint8_t> envelope_make(uint32_t magic, uint64_t hash, llama_pos p0, llama_pos p1,
                                          const std::vector<uint8_t> & tgt,
                                          const std::vector<uint8_t> & dft,
                                          const std::vector<uint8_t> & spec) {
    uint64_t ph = 0;
    if (!tgt.empty()) {
        ph = XXH64(tgt.data(), tgt.size(), 0);
    }
    if (!dft.empty()) {
        ph = XXH64(dft.data(), dft.size(), ph);
    }
    if (!spec.empty()) {
        ph = XXH64(spec.data(), spec.size(), ph);
    }

    std::vector<uint8_t> v;
    v.reserve(44 + tgt.size() + dft.size() + spec.size());

    put_u32(v, magic);
    put_u32(v, T32_ENV_VERSION);
    put_u64(v, hash);
    put_i32(v, (int32_t) p0);
    put_i32(v, (int32_t) p1);
    put_u32(v, (uint32_t) tgt.size());
    put_u32(v, (uint32_t) dft.size());
    put_u32(v, (uint32_t) spec.size());
    put_u64(v, ph);
    v.insert(v.end(), tgt.begin(), tgt.end());
    v.insert(v.end(), dft.begin(), dft.end());
    v.insert(v.end(), spec.begin(), spec.end());

    return v;
}

static bool envelope_parse(const std::vector<uint8_t> & buf, kv_env & e) {
    if (buf.size() < 44) {
        return false;
    }

    const uint8_t * p = buf.data();
    size_t off = 0;

    uint32_t version = 0;
    uint64_t ph = 0;

    memcpy(&e.magic,   p + off, 4); off += 4;
    memcpy(&version,   p + off, 4); off += 4;
    memcpy(&e.hash,    p + off, 8); off += 8;
    memcpy(&e.p0,      p + off, 4); off += 4;
    memcpy(&e.p1,      p + off, 4); off += 4;
    memcpy(&e.n_tgt,   p + off, 4); off += 4;
    memcpy(&e.n_dft,   p + off, 4); off += 4;
    memcpy(&e.n_spec,  p + off, 4); off += 4;
    memcpy(&ph,        p + off, 8); off += 8;

    if (version != T32_ENV_VERSION) {
        return false;
    }
    if (off + e.n_tgt + e.n_dft + e.n_spec != buf.size()) {
        return false;
    }

    e.tgt  = p + off; off += e.n_tgt;
    e.dft  = p + off; off += e.n_dft;
    e.spec = p + off;

    uint64_t check = 0;
    if (e.n_tgt > 0) {
        check = XXH64(e.tgt, e.n_tgt, 0);
    }
    if (e.n_dft > 0) {
        check = XXH64(e.dft, e.n_dft, check);
    }
    if (e.n_spec > 0) {
        check = XXH64(e.spec, e.n_spec, check);
    }

    return check == ph;
}
```

在 `containing_block` 之后插入:

```cpp
std::string kv_tree::block_path(uint64_t hash) const {
    char buf[32];
    snprintf(buf, sizeof(buf), "%016" PRIx64 ".bin", hash);
    return cfg.disk_dir + "/blocks/" + buf;
}

std::string kv_tree::anchor_path(uint64_t blk_hash, llama_pos pos) const {
    char buf[48];
    snprintf(buf, sizeof(buf), "%016" PRIx64 "_%d.bin", blk_hash, (int) pos);
    return cfg.disk_dir + "/anchors/" + buf;
}

bool kv_tree::write_disk(const std::string & path, const std::vector<uint8_t> & buf) {
    const std::string tmp = path + ".tmp";

    std::FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }

    const size_t n = fwrite(buf.data(), 1, buf.size(), f);
    const bool ok = fclose(f) == 0 && n == buf.size();

    if (!ok) {
        remove(tmp.c_str());
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path, ec);
    }
    if (ec) {
        remove(tmp.c_str());
        return false;
    }

    return true;
}

bool kv_tree::read_disk(const std::string & path, std::vector<uint8_t> & out) {
    std::FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }

    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return false;
    }

    out.resize((size_t) size);

    const size_t n = fread(out.data(), 1, out.size(), f);
    fclose(f);

    return n == out.size();
}

bool kv_tree::demote_block(kv_tree_block & b) {
    if (cfg.disk_dir.empty() || b.on_disk || b.data.empty()) {
        return false;
    }
    if (st.bytes_disk + (int64_t) b.data.size() > (int64_t) cfg.disk_limit) {
        return false;
    }

    const std::string path = block_path(b.hash);

    if (!write_disk(path, envelope_make(T32_ENV_MAGIC_BLOCK, b.hash, b.pos0, b.pos1, b.data, {}, {}))) {
        fprintf(stderr, "[kv-tree] failed to write block %016" PRIx64 " to disk\n", b.hash);
        st.disk_errors++;
        return false;
    }

    st.bytes_ram  -= (int64_t) b.data.size();
    st.bytes_disk += (int64_t) b.data.size();
    st.blocks_ram--;
    st.blocks_disk++;
    st.bytes_store += (int64_t) b.data.size();

    b.on_disk = true;
    b.path    = path;
    std::vector<uint8_t>().swap(b.data);

    return true;
}

bool kv_tree::demote_anchor(kv_tree_anchor & a) {
    if (cfg.disk_dir.empty() || a.on_disk || a.data_tgt.empty()) {
        return false;
    }

    const size_t size = anchor_bytes(a);
    if (st.bytes_disk + (int64_t) size > (int64_t) cfg.disk_limit) {
        return false;
    }

    const std::string path = anchor_path(a.blk_hash, a.pos);

    if (!write_disk(path, envelope_make(T32_ENV_MAGIC_ANCHOR, a.blk_hash, a.pos, a.pos, a.data_tgt, a.data_dft, a.data_spec))) {
        fprintf(stderr, "[kv-tree] failed to write anchor %016" PRIx64 "@%d to disk\n", a.blk_hash, a.pos);
        st.disk_errors++;
        return false;
    }

    st.bytes_ram  -= (int64_t) size;
    st.bytes_disk += (int64_t) size;
    st.anchors_ram--;
    st.anchors_disk++;
    st.bytes_store += (int64_t) size;

    a.on_disk = true;
    a.path    = path;
    std::vector<uint8_t>().swap(a.data_tgt);
    std::vector<uint8_t>().swap(a.data_dft);
    std::vector<uint8_t>().swap(a.data_spec);

    return true;
}

bool kv_tree::load_payload(kv_tree_block & b) {
    if (!b.on_disk) {
        return true;
    }

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

    b.data.assign(e.tgt, e.tgt + e.n_tgt);
    b.transient = true;

    st.bytes_ram += (int64_t) b.data.size();
    st.blocks_ram++;

    return true;
}

bool kv_tree::load_payload(kv_tree_anchor & a) {
    if (!a.on_disk) {
        return true;
    }

    std::vector<uint8_t> buf;
    if (!read_disk(a.path, buf)) {
        fprintf(stderr, "[kv-tree] failed to read anchor %016" PRIx64 "@%d from disk\n", a.blk_hash, a.pos);
        st.disk_errors++;
        return false;
    }

    kv_env e;
    if (!envelope_parse(buf, e) || e.magic != T32_ENV_MAGIC_ANCHOR || e.hash != a.blk_hash || e.p0 != a.pos || e.p1 != a.pos) {
        fprintf(stderr, "[kv-tree] anchor %016" PRIx64 "@%d payload check failed, dropping it\n", a.blk_hash, a.pos);
        st.disk_errors++;
        return false;
    }

    a.data_tgt.assign(e.tgt, e.tgt + e.n_tgt);
    a.data_dft.assign(e.dft, e.dft + e.n_dft);
    a.data_spec.assign(e.spec, e.spec + e.n_spec);
    a.transient = true;

    st.bytes_ram += (int64_t) anchor_bytes(a);
    st.anchors_ram++;

    return true;
}

void kv_tree::settle() {
    for (auto & p : blocks) {
        kv_tree_block & b = p.second;
        if (!b.transient) {
            continue;
        }

        b.transient = false;

        if ((size_t) st.bytes_ram <= cfg.ram_limit) {
            std::error_code ec;
            std::filesystem::remove(b.path, ec);

            st.bytes_disk -= (int64_t) b.data.size();
            st.blocks_disk--;
            b.on_disk = false;
            b.path.clear();
        } else {
            st.bytes_ram -= (int64_t) b.data.size();
            st.blocks_ram--;
            std::vector<uint8_t>().swap(b.data);
        }
    }

    for (auto & p : anchors) {
        kv_tree_anchor & a = p.second;
        if (!a.transient) {
            continue;
        }

        a.transient = false;

        const size_t size = anchor_bytes(a);

        if ((size_t) st.bytes_ram <= cfg.ram_limit) {
            std::error_code ec;
            std::filesystem::remove(a.path, ec);

            st.bytes_disk -= (int64_t) size;
            st.anchors_disk--;
            a.on_disk = false;
            a.path.clear();
        } else {
            st.bytes_ram -= (int64_t) size;
            st.anchors_ram--;
            std::vector<uint8_t>().swap(a.data_tgt);
            std::vector<uint8_t>().swap(a.data_dft);
            std::vector<uint8_t>().swap(a.data_spec);
        }
    }
}
```

- [ ] **Step 3: 预算模型改为 commit-then-enforce (pin + 回滚), 接入 restore**

**(3a)** cpp: 删掉 `make_room` 的定义, 在 `settle` 之后插入 (`remove_block` 在 Task 6 复用):

```cpp
bool kv_tree::enforce_budget() {
    for (auto & p : blocks) {
        if ((size_t) st.bytes_ram <= cfg.ram_limit) {
            break;
        }

        kv_tree_block & b = p.second;
        if (b.on_disk || b.data.empty()) {
            continue;
        }

        demote_block(b);
    }

    for (auto & p : anchors) {
        if ((size_t) st.bytes_ram <= cfg.ram_limit) {
            break;
        }

        kv_tree_anchor & a = p.second;
        if (a.on_disk || a.data_tgt.empty()) {
            continue;
        }

        demote_anchor(a);
    }

    return (size_t) st.bytes_ram <= cfg.ram_limit;
}

void kv_tree::remove_block(std::unordered_map<uint64_t, kv_tree_block>::iterator it) {
    kv_tree_block & b = it->second;

    // anchors whose containing block is gone can never be used again
    for (auto a = anchors.begin(); a != anchors.end(); ) {
        if (a->first.first == b.hash) {
            remove_anchor(a++);
        } else {
            ++a;
        }
    }

    if (b.transient) {
        st.bytes_ram -= (int64_t) b.bytes;
        st.blocks_ram--;
    }

    if (b.on_disk) {
        std::error_code ec;
        std::filesystem::remove(b.path, ec);
        st.bytes_disk -= (int64_t) b.bytes;
        st.blocks_disk--;
    } else {
        st.bytes_ram -= (int64_t) b.bytes;
        st.blocks_ram--;
    }

    auto & v = blocks_at[b.pos0];
    v.erase(std::remove(v.begin(), v.end(), b.hash), v.end());
    if (v.empty()) {
        blocks_at.erase(b.pos0);
    }

    blocks.erase(it);
    st.evicted_blocks++;
}

void kv_tree::park_rollback(const std::vector<std::pair<uint64_t, std::vector<uint8_t>>> & blobs,
                            const kv_tree_match & m,
                            const std::vector<std::pair<uint64_t, llama_pos>> & touched,
                            uint64_t tip) {
    for (const auto & p : blobs) {
        auto it = blocks.find(p.first);
        if (it != blocks.end()) {
            remove_block(it);
        }
    }

    for (const auto & key : touched) {
        auto it = anchors.find(key);
        if (it == anchors.end()) {
            continue;
        }
        if (it->second.refcount <= 1) {
            remove_anchor(it);
        } else {
            it->second.refcount--;
        }
    }

    for (const uint64_t hash : m.path) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.refcount--;
        }
    }
    if (m.n_part > 0) {
        auto it = blocks.find(m.part_hash);
        if (it != blocks.end()) {
            it->second.refcount--;
        }
    }

    seqs.erase(tip);
}
```

**(3b)** `park`: 删掉预检查块

```cpp
    size_t need = part_tgt.size() + part_dft.size();
    for (const auto & p : blobs) {
        need += p.second.size();
    }

    if (!make_room(need)) {
        fprintf(stderr, "[kv-tree] park refused: no room for %zu bytes\n", need);
        st.park_refused++;
        return false;
    }
```

并把结尾 (从 `st.park_ok++;` 到最后) 换成:

```cpp
    // pin the new payloads while the budget is enforced, then roll back if it cannot be met
    std::vector<uint64_t> pinned_blocks;
    for (const auto & p : blobs) {
        pinned_blocks.push_back(p.first);
    }
    for (const uint64_t hash : pinned_blocks) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = true;
        }
    }
    for (const auto & key : touched) {
        auto it = anchors.find(key);
        if (it != anchors.end()) {
            it->second.pinned = true;
        }
    }

    const bool fits = enforce_budget();

    for (const uint64_t hash : pinned_blocks) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = false;
        }
    }
    for (const auto & key : touched) {
        auto it = anchors.find(key);
        if (it != anchors.end()) {
            it->second.pinned = false;
        }
    }

    if (!fits) {
        fprintf(stderr, "[kv-tree] park refused: the budget cannot hold the sequence\n");
        park_rollback(blobs, m, touched, tip);
        st.park_refused++;
        return false;
    }

    st.park_ok++;
    return true;
```

**(3c)** `capture_anchor`: 把结尾

```cpp
    if (!store_anchor(blk, pos, KV_TREE_ANCHOR_ONDEMAND, std::move(tgt), std::move(dft))) {
        return false;
    }

    promote_prune(pos);

    return true;
```

换成

```cpp
    if (!store_anchor(blk, pos, KV_TREE_ANCHOR_ONDEMAND, std::move(tgt), std::move(dft))) {
        return false;
    }

    if (!enforce_budget() || anchors.count(key) == 0) {
        st.anchors_skipped++;
        return false;
    }

    promote_prune(pos);

    return true;
```

**(3d)** `restore` 里三处替换:

1. 整块装载: `if (!io_tgt.set_range(b.data.data(), b.data.size(), b.pos0 != 0)) {` 换成 `if (!load_payload(b) || !io_tgt.set_range(b.data.data(), b.data.size(), b.pos0 != 0)) {`
2. 部分块装载: `if (!io_tgt.set_range(b.data.data(), b.data.size(), true)) {` 换成 `if (!load_payload(b) || !io_tgt.set_range(b.data.data(), b.data.size(), true)) {`
3. 锚点装载: `const kv_tree_anchor & a = it->second;` 换成 `kv_tree_anchor & a = it->second;`, 且 `if (!io_tgt.set_partial(a.data_tgt.data(), a.data_tgt.size())) {` 换成 `if (!load_payload(a) || !io_tgt.set_partial(a.data_tgt.data(), a.data_tgt.size())) {`
4. 在两组 unpin 循环之后、`if (!ok) {` 之前插入一行:

```cpp
    settle();
```


- [ ] **Step 4: harness 加 `scenario_ssd`**

文件头 include 区加 `#include <filesystem>`; 在 `scenario_sparsify` 之后插入:

```cpp
static int scenario_ssd(llama_context * ctx, const kv_tree_config & cfg_in) {
    fprintf(stderr, "[t32-tree] scenario: ssd\n");

    kv_tree_config cfg = cfg_in;
    cfg.ram_limit  = 64 << 10;    // force everything to disk
    cfg.disk_limit = 64 << 20;
    cfg.disk_dir   = (std::filesystem::temp_directory_path() / "t32-tree-test").string();

    std::error_code ec;
    std::filesystem::remove_all(cfg.disk_dir, ec);
    std::filesystem::create_directories(cfg.disk_dir + "/blocks", ec);
    std::filesystem::create_directories(cfg.disk_dir + "/anchors", ec);

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tokens = make_tokens(1536, 0);

    prefill(ctx, 0, tokens, 0, 512);

    check(tree.park(io, nullptr, tokens, {}), "ssd: park");
    check(tree.stats().blocks_disk > 0, "ssd: blocks demoted to disk");
    check(tree.stats().anchors_disk > 0, "ssd: tip anchor demoted to disk");
    check_eq(tree.stats().blocks_ram, 0, "ssd: no blocks left in ram");

    {
        const auto base = run_baseline(ctx, tokens, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
        check_eq(r.res.C, 1536, "ssd: restore point is the tip");
        check(r.gen == base, "ssd: tokens match the baseline");
    }

    for (const auto & e : std::filesystem::directory_iterator(cfg.disk_dir + "/blocks")) {
        std::filesystem::remove(e.path());
    }

    const int64_t errors_before = tree.stats().disk_errors;

    {
        const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
        check_eq(r.res.C, -1, "ssd: restore misses when block files are gone");
        check(tree.stats().disk_errors > errors_before, "ssd: disk error counted");
    }

    std::filesystem::remove_all(cfg.disk_dir, ec);

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}
```

`main` 的 model 分支加 `ret |= scenario_ssd(ctx, cfg);`.

- [ ] **Step 5: 构建并跑**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```
Expected: tip / fork / sparsify / ssd 全 PASS, 其中 `ssd: restore point is the tip` 与 `ssd: tokens match the baseline` 必须 PASS (SSD 往返逐位一致).

- [ ] **Step 6: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : add ssd tier and single-copy payload moves to the kv tree" -m "Assisted-by: opencode"
```

---

