### Task 2: 信封缺省 + tip 锚点 + restore + llama adapter + 真机 tip 场景

**Files:**
- Modify: `tools/server/server-kv-tree.h` (加 `restore` 声明与 `store_anchor` 私有方法)
- Modify: `tools/server/server-kv-tree.cpp` (加 adapter / `store_anchor` / 改 `park` / 加 `restore`)
- Modify: `tests/test-t32-tree.cpp` (加 model 模式 + `scenario_tip`)

**Interfaces:**
- Consumes: Task 1 的全部类型与 `park`/`match`
- Produces: `kv_tree::restore`, `kv_tree::store_anchor`, `kv_tree_io_llama` 全部实现, harness 助手 `prefill`/`generate`/`run_baseline`/`run_tree_path`/`capture_partial`

- [ ] **Step 1: 头文件加声明**

在 `class kv_tree` 的 public 段 `park` 之后加:

```cpp
    // load the deepest anchor-covered prefix of tokens into the (cleared) sequence
    kv_tree_restore restore(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens);
```

在 private 段 `match` 之后加:

```cpp
    bool store_anchor(uint64_t blk_hash, llama_pos pos, int kind,
                      std::vector<uint8_t> && data_tgt, std::vector<uint8_t> && data_dft);
```

- [ ] **Step 2: `server-kv-tree.cpp` 加 adapter 与 `store_anchor`**

在 `kv_tree::kv_tree` 之前插入:

```cpp
kv_tree_io_llama::kv_tree_io_llama(llama_context * ctx, llama_seq_id seq_id) : ctx(ctx), seq_id(seq_id) {
}

bool kv_tree_io_llama::get_range(llama_pos p0, llama_pos p1, std::vector<uint8_t> & out) {
    const size_t size = llama_state_seq_get_size_range_ext(ctx, seq_id, p0, p1, 0);
    if (size == 0) {
        return false;
    }

    out.resize(size);

    return llama_state_seq_get_data_range_ext(ctx, out.data(), out.size(), seq_id, p0, p1, 0) == size;
}

bool kv_tree_io_llama::set_range(const uint8_t * data, size_t size, bool append) {
    return llama_state_seq_set_data_range_ext(ctx, data, size, seq_id, append, 0) == size;
}

bool kv_tree_io_llama::get_partial(std::vector<uint8_t> & out) {
    const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (size == 0) {
        return false;
    }

    out.resize(size);

    return llama_state_seq_get_data_ext(ctx, out.data(), out.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size;
}

bool kv_tree_io_llama::set_partial(const uint8_t * data, size_t size) {
    return llama_state_seq_set_data_ext(ctx, data, size, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size;
}

bool kv_tree_io_llama::seq_rm(llama_pos p0, llama_pos p1) {
    return llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1);
}

llama_pos kv_tree_io_llama::pos_max() {
    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
}
```

在 `make_room` 之后插入:

```cpp
bool kv_tree::store_anchor(uint64_t blk_hash, llama_pos pos, int kind,
                           std::vector<uint8_t> && data_tgt, std::vector<uint8_t> && data_dft) {
    if (pos <= 0 || blk_hash == 0) {
        return false;
    }

    const auto key = std::make_pair(blk_hash, pos);

    const auto it = anchors.find(key);
    if (it != anchors.end()) {
        it->second.refcount++;
        it->second.heat++;
        it->second.last_used = ++now;
        return true;
    }

    const size_t size = data_tgt.size() + data_dft.size();

    kv_tree_anchor & a = anchors[key];

    a.blk_hash  = blk_hash;
    a.pos       = pos;
    a.kind      = kind;
    a.refcount  = 1;
    a.last_used = ++now;
    a.bytes     = size;
    a.data_tgt  = std::move(data_tgt);
    a.data_dft  = std::move(data_dft);

    st.anchors_ram++;
    st.anchors_added++;
    st.bytes_ram += (int64_t) size;

    return true;
}
```

- [ ] **Step 3: 改 `park` (tip 锚点 + 预算一次算清)**

用下面完整函数替换 Task 1 的 `park`:

```cpp
bool kv_tree::park(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens,
                   const std::vector<kv_tree_anchor_in> & checkpoints) {
    (void) checkpoints;

    st.park_calls++;

    const llama_pos L = (llama_pos) tokens.size();

    if (L <= 0 || io_tgt.pos_max() != L - 1) {
        fprintf(stderr, "[kv-tree] park refused: sequence end %d, expected %d\n", io_tgt.pos_max(), L - 1);
        st.park_refused++;
        return false;
    }

    std::vector<uint64_t> h;
    chain_hashes(tokens, cfg.chunk, h);

    const uint64_t tip = h.back();

    if (seqs.count(tip) != 0) {
        seqs[tip].last_used = ++now;
        st.park_ok++;
        return true;
    }

    const kv_tree_match m = match(tokens);

    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> blobs;
    std::vector<size_t> idxs;

    for (size_t i = m.n_full; i < h.size(); ++i) {
        if (blocks.count(h[i]) != 0) {
            continue;
        }

        const size_t a = i * (size_t) cfg.chunk;
        const size_t b = std::min(tokens.size(), a + (size_t) cfg.chunk);

        std::vector<uint8_t> blob;
        if (!io_tgt.get_range((llama_pos) a, (llama_pos) b, blob)) {
            fprintf(stderr, "[kv-tree] park refused: failed to read range [%zu, %zu)\n", a, b);
            st.park_refused++;
            return false;
        }

        blobs.emplace_back(h[i], std::move(blob));
        idxs.push_back(i);
    }

    // tip anchor: the io state is exactly at L
    std::vector<uint8_t> part_tgt;
    std::vector<uint8_t> part_dft;

    if (!io_tgt.get_partial(part_tgt)) {
        fprintf(stderr, "[kv-tree] park refused: failed to capture the tip state\n");
        st.park_refused++;
        return false;
    }
    if (io_dft != nullptr) {
        io_dft->get_partial(part_dft);
    }

    size_t need = part_tgt.size() + part_dft.size();
    for (const auto & p : blobs) {
        need += p.second.size();
    }

    if (!make_room(need)) {
        fprintf(stderr, "[kv-tree] park refused: no room for %zu bytes\n", need);
        st.park_refused++;
        return false;
    }

    for (size_t k = 0; k < blobs.size(); ++k) {
        const size_t i = idxs[k];
        const size_t a = i * (size_t) cfg.chunk;
        const size_t b = std::min(tokens.size(), a + (size_t) cfg.chunk);

        kv_tree_block & nb = blocks[blobs[k].first];

        nb.hash      = blobs[k].first;
        nb.pos0      = (llama_pos) a;
        nb.pos1      = (llama_pos) b;
        nb.refcount  = 1;
        nb.last_used = ++now;
        nb.tokens.assign(tokens.begin() + a, tokens.begin() + b);
        nb.data      = std::move(blobs[k].second);

        blocks_at[nb.pos0].push_back(nb.hash);
        st.blocks_ram++;
        st.bytes_ram += (int64_t) nb.data.size();
    }

    for (const uint64_t hash : m.path) {
        kv_tree_block & b = blocks[hash];
        b.refcount++;
        b.heat++;
        b.last_used = ++now;
    }
    if (m.n_part > 0) {
        kv_tree_block & b = blocks[m.part_hash];
        b.refcount++;
        b.heat++;
        b.last_used = ++now;
    }

    if (!store_anchor(tip, L, KV_TREE_ANCHOR_TIP, std::move(part_tgt), std::move(part_dft))) {
        fprintf(stderr, "[kv-tree] park refused: cannot store the tip anchor at %d\n", L);
        st.park_refused++;
        return false;
    }

    kv_tree_seq & s = seqs[tip];
    s.chain     = h;
    s.len       = L;
    s.last_used = ++now;

    st.park_ok++;
    return true;
}
```

- [ ] **Step 4: 加 `restore`**

在 `park` 之后插入:

```cpp
kv_tree_restore kv_tree::restore(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens) {
    st.restore_calls++;

    io_tgt.seq_rm(-1, -1);
    if (io_dft != nullptr) {
        io_dft->seq_rm(-1, -1);
    }

    kv_tree_restore res;

    const kv_tree_match m = match(tokens);

    if (m.deep <= 0) {
        st.restore_miss++;
        return res;
    }

    std::vector<uint64_t> cand = m.path;
    if (m.n_part > 0) {
        cand.push_back(m.part_hash);
    }

    llama_pos C = -1;
    uint64_t  c_hash = 0;
    std::vector<llama_pos> path_anchors;

    for (const uint64_t hash : cand) {
        for (auto it = anchors.lower_bound(std::make_pair(hash, 0));
             it != anchors.end() && it->first.first == hash; ++it) {
            if (it->second.pos <= m.deep) {
                path_anchors.push_back(it->second.pos);
                if (it->second.pos > C) {
                    C = it->second.pos;
                    c_hash = hash;
                }
            }
        }
    }

    if (C < 0) {
        st.restore_miss++;
        return res;
    }

    for (const uint64_t hash : cand) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = true;
        }
    }
    for (auto & kv : anchors) {
        if (kv.second.pos <= C && std::find(cand.begin(), cand.end(), kv.first.first) != cand.end()) {
            kv.second.pinned = true;
        }
    }

    bool ok = true;

    // blocks covering [0, C): the last one may overshoot and is trimmed below
    for (const uint64_t hash : m.path) {
        kv_tree_block & b = blocks[hash];
        if (b.pos0 >= C) {
            break;
        }
        if (!io_tgt.set_range(b.data.data(), b.data.size(), b.pos0 != 0)) {
            fprintf(stderr, "[kv-tree] restore failed: cannot load block [%d, %d)\n", b.pos0, b.pos1);
            ok = false;
            break;
        }
        st.bytes_load += (int64_t) b.data.size();
    }

    if (ok && m.n_part > 0 && C > blocks[m.part_hash].pos0) {
        kv_tree_block & b = blocks[m.part_hash];
        if (!io_tgt.set_range(b.data.data(), b.data.size(), true)) {
            fprintf(stderr, "[kv-tree] restore failed: cannot load the partial block [%d, %d)\n", b.pos0, b.pos1);
            ok = false;
        }
    }

    if (ok) {
        io_tgt.seq_rm(C, -1);

        const auto it = anchors.find(std::make_pair(c_hash, C));
        if (it == anchors.end()) {
            ok = false;
        } else {
            const kv_tree_anchor & a = it->second;
            if (!io_tgt.set_partial(a.data_tgt.data(), a.data_tgt.size())) {
                fprintf(stderr, "[kv-tree] restore failed: cannot load the state at %d\n", C);
                ok = false;
            } else if (io_dft != nullptr && !a.data_dft.empty()) {
                io_dft->set_partial(a.data_dft.data(), a.data_dft.size());
            }
        }
    }

    for (const uint64_t hash : cand) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = false;
        }
    }
    for (auto & kv : anchors) {
        if (kv.second.pos <= C && std::find(cand.begin(), cand.end(), kv.first.first) != cand.end()) {
            kv.second.pinned = false;
        }
    }

    if (!ok) {
        io_tgt.seq_rm(-1, -1);
        st.restore_miss++;
        return res;
    }

    kv_tree_anchor & ca = anchors.at(std::make_pair(c_hash, C));
    ca.heat++;
    ca.last_used = ++now;

    st.restore_hits++;
    st.tokens_reused += C;

    res.C      = C;
    res.heal   = m.deep > C ? m.deep : -1;
    res.anchors = path_anchors;

    return res;
}
```

- [ ] **Step 5: harness 加 model 模式与 `scenario_tip`**

在 `run_logic` 之后插入 (helper 与场景), 并改 `main` 支持 model 模式:

```cpp
static int prefill(llama_context * ctx, llama_seq_id seq, const std::vector<llama_token> & tokens, int from, int ubatch) {
    llama_batch batch = llama_batch_init(ubatch, 0, 1);

    for (int i = from; i < (int) tokens.size(); ) {
        batch.n_tokens = 0;

        const int n = std::min(ubatch, (int) tokens.size() - i);

        for (int j = 0; j < n; ++j) {
            common_batch_add(batch, tokens[i + j], i + j, { seq }, j == n - 1);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[t32-tree] decode failed at token %d\n", i);
            llama_batch_free(batch);
            return 1;
        }

        i += n;
    }

    llama_batch_free(batch);
    return 0;
}

static std::vector<llama_token> generate(llama_context * ctx, llama_seq_id seq_id, llama_token first, llama_pos pos0, int n) {
    std::vector<llama_token> res;

    llama_sampler * smpl = llama_sampler_init_greedy();
    llama_batch batch = llama_batch_init(1, 0, 1);

    llama_token cur = first;
    llama_pos   pos = pos0;

    for (int i = 0; i < n; ++i) {
        batch.n_tokens = 0;
        common_batch_add(batch, cur, pos++, { seq_id }, true);

        if (llama_decode(ctx, batch) != 0) {
            res.clear();
            break;
        }

        cur = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, cur);
        res.push_back(cur);
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    return res;
}

static kv_tree_anchor_in capture_partial(llama_context * ctx, llama_seq_id seq, llama_pos pos) {
    kv_tree_anchor_in c;
    c.pos = pos;

    const size_t size = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (size == 0) {
        return c;
    }

    c.data_tgt.resize(size);

    const size_t n = llama_state_seq_get_data_ext(ctx, c.data_tgt.data(), c.data_tgt.size(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (n != size) {
        c.data_tgt.clear();
    }

    return c;
}

static std::vector<llama_token> run_baseline(llama_context * ctx, const std::vector<llama_token> & tokens, llama_token first, int n_gen) {
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    if (prefill(ctx, 0, tokens, 0, 512) != 0) {
        return {};
    }

    return generate(ctx, 0, first, (llama_pos) tokens.size(), n_gen);
}

struct path_result {
    std::vector<llama_token> gen;
    kv_tree_restore res;
};

static path_result run_tree_path(llama_context * ctx, kv_tree & tree, kv_tree_io & io,
                                 const std::vector<llama_token> & tokens, llama_token first, int n_gen) {
    path_result r;

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    r.res = tree.restore(io, nullptr, tokens);

    if (r.res.C >= 0 && prefill(ctx, 0, tokens, (int) r.res.C, 512) != 0) {
        return r;
    }

    r.gen = generate(ctx, 0, first, (llama_pos) tokens.size(), n_gen);
    return r;
}

static int scenario_tip(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] scenario: tip\n");

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tokens = make_tokens(1536, 0);

    if (prefill(ctx, 0, tokens, 0, 512) != 0) {
        return 1;
    }

    check(tree.park(io, nullptr, tokens, {}), "tip: park");
    check_eq(tree.stats().anchors_added, 1, "tip: one tip anchor");

    const auto base = run_baseline(ctx, tokens, 42, 8);
    check(!base.empty(), "tip: baseline generation");

    const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
    check_eq(r.res.C, 1536, "tip: restore point is the tip");
    check_eq(r.res.heal, -1, "tip: no heal needed");
    check(r.gen == base, "tip: tokens match the baseline");

    // the sequence now ends at L + 8, so parking it again must be refused
    check(!tree.park(io, nullptr, tokens, {}), "tip: park refused when the sequence ends past L");
    check_eq(tree.stats().park_refused, 1, "tip: refusal counted");

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}
```

`main` 改为 (完整替换):

```cpp
int main(int argc, char ** argv) {
    std::string mode = "logic";

    int    chunk       = 512;
    int    anchor_step = 32768;
    int    ram_mib     = 8192;
    int    disk_mib    = 65536;
    std::string disk;

    std::vector<char *> args;
    args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            continue;
        }
        if (arg == "--chunk" && i + 1 < argc) {
            chunk = atoi(argv[++i]);
            continue;
        }
        if (arg == "--anchor-step" && i + 1 < argc) {
            anchor_step = atoi(argv[++i]);
            continue;
        }
        if (arg == "--ram-mib" && i + 1 < argc) {
            ram_mib = atoi(argv[++i]);
            continue;
        }
        if (arg == "--disk" && i + 1 < argc) {
            disk = argv[++i];
            continue;
        }
        if (arg == "--disk-mib" && i + 1 < argc) {
            disk_mib = atoi(argv[++i]);
            continue;
        }

        args.push_back(argv[i]);
    }

    if (mode == "logic") {
        return run_logic();
    }

    common_params params;
    params.sampling.seed = 1234;
    params.n_parallel = 1;

    common_init();

    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        fprintf(stderr, "usage: %s -m model.gguf [common args] --mode <logic|model> [--chunk N] [--anchor-step N] [--ram-mib N] [--disk DIR] [--disk-mib N]\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();

    if (ctx == nullptr) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    kv_tree_config cfg;
    cfg.chunk       = chunk;
    cfg.anchor_step = anchor_step;
    cfg.ram_limit   = (size_t) ram_mib << 20;
    cfg.disk_limit  = (size_t) disk_mib << 20;
    cfg.disk_dir    = disk;

    if (mode == "model") {
        int ret = 0;
        ret |= scenario_tip(ctx, cfg);
        return ret;
    }

    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
```

- [ ] **Step 6: 构建并跑 tip 场景**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```
Expected: `scenario: tip`, `tip: park PASS`, `tip: one tip anchor PASS`, `tip: restore point is the tip PASS`, `tip: no heal needed PASS`, `tip: tokens match the baseline PASS`, exit 0. 另跑 `--mode logic` 确认无回归.

- [ ] **Step 7: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : add kv tree tip anchor and prefix restore" -m "Assisted-by: opencode"
```

---

