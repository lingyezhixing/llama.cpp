### Task 1: tree storage side (data model, mapping, park, match)

**Files:**
- Modify: `tools/server/server-kv-tree.h` (types + signatures)
- Modify: `tools/server/server-kv-tree.cpp` (mapping, chain split/hash, park, match, indexes, rollback)
- Modify: `tests/test-t32-tree.cpp` (media logic tests; existing call sites gain the media argument)

**Interfaces (produced for Task 2):**
- `struct kv_tree_media { int64_t idx; uint64_t id; int32_t n_tok; int32_t n_pos; };`
- `bool kv_tree::park(kv_tree_io &, kv_tree_io *, const std::vector<llama_token> &, const std::vector<kv_tree_media> &, const std::vector<kv_tree_anchor_in> &);`
- `kv_tree_match kv_tree::match(const std::vector<llama_token> &, const std::vector<kv_tree_media> &) const;`
- `kv_tree_anchor_in` field `pos` renamed to `int64_t tok`.
- block fields `int64_t tok0`, `std::vector<kv_tree_media> media`; index `blocks_by_pos0`.
- static helpers in the cpp: `pos_at`, `pos_last_cell`, `pos_anchor`, `block_end`, `chain_split`, `chain_hashes`.

**Steps:**

- [ ] **Step 1: write the mapping helpers and chain split/hash in server-kv-tree.cpp**

```cpp
static llama_pos pos_at(const std::vector<kv_tree_media> & media, int64_t t) {
    llama_pos p = (llama_pos) t;
    for (const auto & m : media) {
        if (m.idx >= t) {
            break;
        }
        p += m.n_pos - m.n_tok;
    }
    return p;
}

// position of the last KV cell of a sequence of L tokens
static llama_pos pos_last_cell(const std::vector<kv_tree_media> & media, size_t L) {
    if (L == 0) {
        return -1;
    }
    for (const auto & m : media) {
        if (m.idx <= (int64_t) L - 1 && (int64_t) L - 1 < m.idx + m.n_tok) {
            return pos_at(media, m.idx);   // media cells all carry the chunk start position
        }
    }
    return pos_at(media, (int64_t) L) - 1;
}

static llama_pos pos_anchor(const std::vector<kv_tree_media> & media, size_t L) {
    return pos_last_cell(media, L) + 1;
}

// end of the block that starts at token a: chunk-aligned, never cuts a media chunk
static size_t block_end(const std::vector<kv_tree_media> & media, size_t a, size_t L, int chunk) {
    size_t b = std::min(L, a + (size_t) chunk);
    for (const auto & m : media) {
        if (m.idx < (int64_t) b && (int64_t) b < m.idx + m.n_tok) {
            b = (size_t) (m.idx + m.n_tok);
            break;
        }
    }
    return b;
}

static void chain_split(const std::vector<kv_tree_media> & media, size_t L, int chunk, std::vector<size_t> & starts) {
    starts.clear();
    for (size_t a = 0; a < L; ) {
        starts.push_back(a);
        a = block_end(media, a, L, chunk);
    }
}

static void chain_hashes(const std::vector<llama_token> & tokens, const std::vector<kv_tree_media> & media,
                         const std::vector<size_t> & starts, std::vector<uint64_t> & h) {
    h.clear();
    uint64_t cur = 0;
    for (size_t i = 0; i < starts.size(); ++i) {
        const size_t a = starts[i];
        const size_t b = i + 1 < starts.size() ? starts[i + 1] : tokens.size();
        cur = XXH64(tokens.data() + a, (b - a) * sizeof(llama_token), cur);
        for (const auto & m : media) {
            if (m.idx < (int64_t) a) {
                continue;
            }
            if (m.idx >= (int64_t) b) {
                break;
            }
            std::vector<uint8_t> rec;
            put_i32(rec, (int32_t) (m.idx - (int64_t) a));
            put_u64(rec, m.id);
            put_i32(rec, m.n_tok);
            put_i32(rec, m.n_pos);
            cur = XXH64(rec.data(), rec.size(), cur);
        }
        h.push_back(cur);
    }
}
```

Note: `put_i32`/`put_u64` already exist above in the file.

- [ ] **Step 2: park**

- guard: `io_tgt.pos_max() != pos_last_cell(media, L)` (message prints both).
- replace the `i * chunk` slicing with `starts` from `chain_split(media, L, cfg.chunk, starts)`; `chain_hashes(tokens, media, starts, h)`.
- per block: `a = starts[i]`, `b = starts[i+1]` (or L for the last); `get_range(pos_at(media, a), pos_at(media, b), blob)`.
- stored block: `nb.tok0 = a; nb.pos0 = pos_at(media, a); nb.pos1 = pos_at(media, b); nb.tokens = [a,b); nb.media = entries with idx in [a,b);`
- `blocks_at[nb.tok0].push_back(hash); blocks_by_pos0[nb.pos0].push_back(hash);`
- tip anchor: `store_anchor(tip, L, pos_anchor(media, L), KV_TREE_ANCHOR_TIP, ...)` (signature gains `tok` before `pos`; see step 4).
- candidates: `apos = pos_anchor(media, c.tok)`; skip when `c.tok` falls inside a chunk; `containing_block(apos, &h)`.
- refcount sync loop condition: `kv.second.tok > (int64_t) L` -> continue.

- [ ] **Step 3: match**

```cpp
kv_tree_match kv_tree::match(const std::vector<llama_token> & tokens, const std::vector<kv_tree_media> & media) const {
    kv_tree_match m;

    std::vector<size_t> starts;
    chain_split(media, tokens.size(), cfg.chunk, starts);

    std::vector<uint64_t> h;
    chain_hashes(tokens, media, starts, h);

    const size_t n_full_req = starts.empty() ? 0 : starts.size() - 1;

    for (size_t i = 0; i < n_full_req; ++i) {
        const auto it = blocks.find(h[i]);
        if (it == blocks.end()) {
            break;
        }
        const kv_tree_block & b = it->second;
        const size_t a = starts[i];
        if (b.tok0 != (int64_t) a || a + b.tokens.size() > tokens.size()) {
            break;
        }
        if (memcmp(tokens.data() + a, b.tokens.data(), b.tokens.size() * sizeof(llama_token)) != 0) {
            break;
        }
        m.path.push_back(h[i]);
        m.n_full++;
        m.deep = (llama_pos) (a + b.tokens.size());
    }

    if (m.n_full == n_full_req && n_full_req < starts.size()) {
        const size_t tail_a = starts[n_full_req];
        const auto it = blocks_at.find((llama_pos) tail_a);
        if (it != blocks_at.end()) {
            for (const uint64_t hash : it->second) {
                const kv_tree_block & b = blocks.at(hash);

                size_t n = std::min(b.tokens.size(), tokens.size() - tail_a);
                size_t k = 0;
                while (k < n && b.tokens[k] == tokens[tail_a + k]) {
                    ++k;
                }

                // media identity and chunk-boundary clamp: never verify into the middle of a chunk
                for (const auto & bm : b.media) {
                    if (bm.idx < (int64_t) tail_a) {
                        continue;
                    }
                    if (bm.idx >= (int64_t) (tail_a + k)) {
                        break;
                    }
                    const kv_tree_media * rm = media_at(media, bm.idx);
                    if (rm == nullptr || rm->id != bm.id || rm->n_tok != bm.n_tok || rm->n_pos != bm.n_pos) {
                        k = (size_t) (bm.idx - (int64_t) tail_a);
                        break;
                    }
                    if ((int64_t) (tail_a + k) < bm.idx + bm.n_tok) {
                        k = (size_t) (bm.idx - (int64_t) tail_a);
                        break;
                    }
                }

                if (k > m.n_part) {
                    m.n_part = k;
                    m.part_hash = hash;
                }
            }
        }

        m.deep = std::max(m.deep, (llama_pos) (tail_a + m.n_part));
    }

    return m;
}
```

Add `media_at(media, idx)` (place it next to the other helpers):

```cpp
static const kv_tree_media * media_at(const std::vector<kv_tree_media> & media, int64_t idx) {
    const auto it = std::lower_bound(media.begin(), media.end(), idx,
            [](const kv_tree_media & m, int64_t v) { return m.idx < v; });
    if (it != media.end() && it->idx == idx) {
        return &*it;
    }
    return nullptr;
}
```

- [ ] **Step 4: signatures and mechanical updates**

- `store_anchor(uint64_t blk_hash, int64_t tok, llama_pos pos, int kind, ...)`; `kv_tree_anchor` gains `int64_t tok`; `store_anchor` sets `a.tok = tok`. For empty media the callers pass tok == pos, so text-only behavior is unchanged.
- `containing_block(pos, chain)`: look up `blocks_by_pos0.lower_bound(pos)`, then `--it`, then check `b.pos0 < pos && pos <= b.pos1` and chain membership.
- `remove_block`: erase from `blocks_at[b.tok0]` and `blocks_by_pos0[b.pos0]` (remove the hash from both vectors, erase empty keys).
- `park_rollback`: unchanged logic (keys are still `(hash, pos)`), but `touched` entries and `anchors` lookups stay as they are.
- `drop_seq(tokens, media)`, `restore(io, io_dft, tokens, media)`, `capture_anchor(io, io_dft, tokens, media, tok)`: add the parameter; Task 1 only needs them to compile with text-only behavior intact (empty media => token index == position). Task 2 implements their media paths.
  - For Task 1, in `restore` replace `m.deep` comparisons on anchors with `a.tok` (anchors now carry tok; set `a.tok = pos` at store time for empty media so text-only behavior is identical), `seq_rm(C, -1)` -> `seq_rm(pos_at(media, C), -1)`, block loop condition `b.pos0 >= C` -> `b.tok0 >= (int64_t) C`, partial-block condition `C > blocks[m.part_hash].pos0` -> `... .tok0`, and find the anchor by the selected key instead of `(c_hash, C)`.
  - `capture_anchor`: expected end = `pos_last_cell(media, tokens.size())`, anchor pos = end + 1, and use `containing_block(anchor pos)`; F1 fallback rewritten in token space (`b.tok0 < tok && tok <= b.tok0 + (int64_t) b.tokens.size()`, prefix compare `n = tok - b.tok0`).
- Update every existing call site in the harness to pass an empty media vector.

- [ ] **Step 5: harness logic tests for the storage side**

Add `run_logic_media()` and call it from `run_logic()`. Extend `kv_tree_io_fake` to record ranges: add `std::vector<std::pair<llama_pos, llama_pos>> ranges;` and push `(p0,p1)` in `get_range`. Tests (use `make_tokens` and overwrite a window with `LLAMA_TOKEN_NULL`):

1. identity: park 3000 tokens with one chunk `{idx=1000, id=0xAA, n_tok=1026, n_pos=32}`; `match` the same input -> `n_part` reaches the end; `match` with `id=0xBB` -> match stops at or before 1000; `match` with `n_tok=1025, n_pos=32` -> stops before 1000.
2. alignment: chunk at `idx=1000, n_tok=1026` (crosses 1024); park and assert the recorded ranges do not start or end strictly inside `[1000, 2026)`: every recorded `(p0,p1)` satisfies `p0 <= 1000 || p0 >= 2026` and `p1 <= 1000 || p1 >= 2026`.
3. round trip: park, then `restore` with the same input; `res.C > 0`; the fake io's `set_range_calls` > 0; `pos_at` sanity: the last range end equals `pos_at(media, L)`.
4. clamp: request = the first 1500 tokens (ends inside the chunk at 1000..2026) is impossible for real requests, so instead build the stored chain from 3000 tokens and the request from 1600 tokens with a chunk at `idx=1500, n_tok=100` -> the request's own chunk is whole; assert `n_part` never lands inside `[1500,1600)` unless the whole chunk is verified.
5. `pos_last_cell` guard: park a sequence ending with a chunk; assert it succeeds (the guard uses the chunk start position).

- [ ] **Step 6: build and run**

Run: `build_test_t32.cmd test-t32-tree` then `build\bin\Release\test-t32-tree.exe --mode logic`
Expected: all existing checks pass plus the new media checks, `FAIL count: 0`.

- [ ] **Step 7: commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "kv tree : media-aware blocks and matching"
```

---

