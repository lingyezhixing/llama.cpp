# KV tree media reuse - implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** make the kv tree store, match and restore prompts that contain media chunks, using chunk identity in the block hash and a token<->position mapping for M-RoPE.

**Architecture:** the tree keeps token indices as the matching unit and positions as the KV unit. Media spans (start token index, identity hash, n_tok, n_pos) travel with every call; the tree derives positions from them. Block boundaries are aligned so a media chunk never straddles a block.

**Tech Stack:** C++17, llama.cpp server (`tools/server`), test harness `tests/test-t32-tree.cpp`, PowerShell E2E scripts.

**Spec:** `D:\LLM\Backend\v100-collab\artifacts\t32-media-tree-spec.md` (read it first; the plan argues from it)

## Global Constraints

- Repo: `D:\LLM\Backend\src\llama.cpp-my`, branch `t32-media` (already checked out). Never push, never open a PR.
- Commits are allowed on this branch (standing user authorization for plan branches). Conventional commit style `area : lowercase summary`, no AI co-author trailers except `Assisted-by: <name>` is NOT needed here (private fork).
- ASCII only in code and comments: no emdash, no unicode arrows. Comments: short, 1-2 lines, ASD-STE100 style, only where non-obvious. No comment restating the code.
- Do not add new files under `tests/`; extend `tests/test-t32-tree.cpp`.
- Build test target: `<TEMP>\v100\build_test_t32.cmd test-t32-tree`
- Run logic tests: `build\bin\Release\test-t32-tree.exe --mode logic` (no model needed).
- Run model tests: `$env:CUDA_VISIBLE_DEVICES='0'; build\bin\Release\test-t32-tree.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on --mode model --ram-mib 4096 -c 8192` (night: device 0).
- Build server: delete `build\bin\Release\llama-server-impl.dll` then `<TEMP>\v100\build_server.cmd`.
- PowerShell: no heredoc; write files with `[System.IO.File]::WriteAllText($p, $s, (New-Object System.Text.UTF8Encoding($false)))`.
- Test models: 2B `<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf`, 0.8B `<models>\Qwen3.5-0.8B-MTP\Qwen3.5-0.8B-UD-Q4_K_XL.gguf` + `<models>\Qwen3.5-0.8B-MTP\mmproj-F16.gguf`.
- Never kill unrelated `llama-server` processes (ports 10002/10006 belong to the user).

---

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

### Task 2: tree retrieval side (restore, anchors, capture, drop)

**Files:**
- Modify: `tools/server/server-kv-tree.cpp`
- Modify: `tools/server/server-kv-tree.h` (`kv_tree_restore_anchor` gains `tok`)
- Modify: `tests/test-t32-tree.cpp`

**Interfaces:**
- `kv_tree_restore_anchor { int64_t tok; llama_pos pos; data_tgt; data_dft; }`
- `restore` returns `res.C` and `res.heal` as token indices; `res.anchors` carry `tok` and `pos`.

**Steps:**

- [ ] **Step 1: restore media path**

- anchor filter `it->second.tok <= m.deep`; pick the largest `tok`; keep `(c_hash, c_pos)`.
- block loading by `b.tok0 < C`; partial block when `m.n_part > 0 && C > blocks[m.part_hash].tok0`.
- trim `io_tgt.seq_rm(pos_at(media, C), -1)`.
- anchor state lookup `anchors.find(std::make_pair(c_hash, c_pos))`.
- `res.anchors`: `ra.tok = a.tok; ra.pos = a.pos;`
- miss path: `res.heal = m.deep` (token index, unchanged).

- [ ] **Step 2: capture_anchor media path**

- `tokens.size() == tok` contract; `end = pos_last_cell(media, tokens.size())`; refuse when `io.pos_max() != end`; `apos = end + 1`.
- chain over `(tokens, media)`; `containing_block(apos, &chain)`; F1 fallback in token space (as in Task 1 step 4).
- `prev` scan: same-chain anchors with `pos < apos`, kind != MESSAGE; `fork_step` compare `apos - prev`.
- `store_anchor(blk, tok, apos, KV_TREE_ANCHOR_ONDEMAND, ...)`.

- [ ] **Step 3: drop_seq media path**

- `match(tokens, media)`; tip selection unchanged.

- [ ] **Step 4: harness logic tests for retrieval**

1. capture/restore at a media-ending boundary: park 3000 tokens with a chunk at `idx=1000, n_tok=1026, n_pos=32`; `capture_anchor` at `tok=2026` (chunk end) -> returns true; then `restore` with a request of 2500 tokens and assert `res.C == 2026` or deeper, `res.anchors` non-empty, and the fake io recorded a `seq_rm(pos_at(media, C), -1)`-equivalent trim (assert `seq_rm` calls via a new recorded `trims` vector).
2. anchor `tok` vs `pos`: assert the stored anchor at the chunk end has `tok == 2026` and `pos == pos_at(media, 1000) + 1`.
3. text-only unchanged: existing checks stay green (empty media => tok == pos).
4. `drop_seq` with media: park, then `drop_seq(same input)` returns true; `drop_seq` with a different chunk id returns false.

- [ ] **Step 5: build and run**

Run: `build_test_t32.cmd test-t32-tree` then `test-t32-tree.exe --mode logic` -> 0 failures; then the 2B model mode with `CUDA_VISIBLE_DEVICES=0` -> 0 failures.

- [ ] **Step 6: commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "kv tree : media-aware restore and anchors"
```

---

### Task 3: server integration

**Files:**
- Modify: `tools/server/server-common.h` / `server-common.cpp` (`pos_last`, media map accessor)
- Modify: `tools/server/server-context.cpp` (spans helper, park/restore/erase/heal wiring)

**Interfaces:**
- `llama_pos server_tokens::pos_last() const;`
- `const std::map<size_t, mtmd::input_chunk_ptr> & server_tokens::media_map() const;`
- static `std::vector<kv_tree_media> tree_media_spans(const server_tokens & tokens);` in server-context.cpp (FNV-1a 64 over `mtmd_input_chunk_get_id`).

**Steps:**

- [ ] **Step 1: server_tokens helpers**

```cpp
llama_pos server_tokens::pos_last() const {
    if (tokens.empty()) {
        return -1;
    }
    if (has_mtmd && tokens.back() == LLAMA_TOKEN_NULL) {
        for (auto it = map_idx_to_media.rbegin(); it != map_idx_to_media.rend(); ++it) {
            const auto & chunk = it->second;
            const size_t n_tok = mtmd_input_chunk_get_n_tokens(chunk.get());
            if (it->first + n_tok == tokens.size()) {
                return pos_next((int64_t) it->first);
            }
        }
    }
    return pos_next() - 1;
}
```

- [ ] **Step 2: spans helper in server-context.cpp**

```cpp
static uint64_t fnv1a64(const char * s) {
    uint64_t h = 1469598103934665603ull;
    for (; s != nullptr && *s != '\0'; ++s) {
        h ^= (uint8_t) *s;
        h *= 1099511628211ull;
    }
    return h;
}

static std::vector<kv_tree_media> tree_media_spans(const server_tokens & tokens) {
    std::vector<kv_tree_media> out;
    for (const auto & e : tokens.media_map()) {
        const mtmd_input_chunk * chunk = e.second.get();
        kv_tree_media m;
        m.idx   = (int64_t) e.first;
        m.id    = fnv1a64(mtmd_input_chunk_get_id(chunk));
        m.n_tok = (int32_t) mtmd_input_chunk_get_n_tokens(chunk);
        m.n_pos = (int32_t) mtmd_input_chunk_get_n_pos(chunk);
        out.push_back(m);
    }
    return out;
}
```

- [ ] **Step 3: wire the four call sites**

- `prompt_park` (`:303`): drop `prompt.tokens.has_mtmd` from the guard; guard `p_max != prompt.tokens.pos_last()`; `tree.park(io_tgt, io_dft_ptr, prompt.tokens.get_tokens(), tree_media_spans(prompt.tokens), cks)`; `in.tok = c.n_tokens`.
- `prompt_restore_tree` (`:347`): drop `tokens.has_mtmd` from the guard; `tree.restore(io_tgt, io_dft_ptr, tokens.get_tokens(), tree_media_spans(tokens))`; on hit:
  ```cpp
  prompt.tokens = tokens.clone();
  prompt.tokens.keep_first(res.C);
  prompt.checkpoints.clear();
  if (n_ckpt_max > 0 && !prompt.tokens.has_media()) { ... existing rebuild ... }
  ```
  Add `bool server_tokens::has_media() const { return !map_idx_to_media.empty(); }` (private map access is fine inside the class).
- `SLOT_ERASE` (`:2866`): drop the `!slot->prompt.tokens.has_mtmd` guard; `tree->drop_seq(slot->prompt.tokens.get_tokens(), tree_media_spans(slot->prompt.tokens))`.
- heal capture (`:3635`): `tree->capture_anchor(io_h_tgt, io_h_dft_ptr, slot.prompt.tokens.get_tokens(), tree_media_spans(slot.prompt.tokens), hp)`.

- [ ] **Step 4: build the server and run the text-only regressions**

Run: delete `build\bin\Release\llama-server-impl.dll`; `build_server.cmd`; expect a clean link.
Run: logic + model harness again (0 failures).
Run a quick text-only server check on the 2B with the tree enabled (no mmproj), one 2-request conversation, expect `parked` / `restored` in the log and no warnings. Use the existing script `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` mode `soak` for a short run if convenient, or a direct server start + two requests.

- [ ] **Step 5: commit**

```
git add tools/server/server-common.h tools/server/server-common.cpp tools/server/server-context.cpp
git commit -m "server : wire media prompts into the kv tree"
```

---

### Task 4: end-to-end media verification on the 0.8B + mmproj

**Files:**
- Create: `D:\LLM\Backend\v100-collab\artifacts\t32-media-e2e.ps1`
- Create evidence: `D:\LLM\Backend\v100-collab\artifacts\t32-media-e2e-<timestamp>.txt`
- Append: `D:\LLM\Backend\v100-collab\RESULTS.md`, `STATUS.md`, `TASKS\T32-agent-session-reuse.md`

**Steps:**

- [ ] **Step 1: generate two small test images**

Use PowerShell + System.Drawing to write `t32-img-a.png` (e.g. 320x240 red/blue pattern) and `t32-img-b.png` (different pattern) under `<TEMP>\v100\`. Base64 them in the script.

- [ ] **Step 2: script**

Start `llama-server` (cuda0, port 18080) with:
`-m <models>\Qwen3.5-0.8B-MTP\Qwen3.5-0.8B-UD-Q4_K_XL.gguf --mmproj ...mmproj-F16.gguf -ngl 99 -fa on --image-min-tokens 1024 -np 1 -c 32768 -b 2048 -ub 512 --temp 0 --seed 42 --ctx-checkpoints 4 --kv-tree --tree-ram 2048 --tree-disk <TEMP>\v100\t32-media-tree --tree-checkpoint-anchor-step 8192 --tree-checkpoint-fork-step 4096 -a media-e2e`
Then:
1. request 1: chat completion with image A + a fixed question (greedy, `max_tokens` 32) -> record output + `timings.prompt_n`.
2. request 2: the same conversation plus one more user turn (image A repeated, same prefix) -> expect `prompt_n` much smaller than the full length and the log to show `kv tree: restored ... tokens`.
3. request 3: same positions but image B -> must NOT restore from A's chain (prompt_n full or a restore that stops before the image), and the log must not report a restore whose C is past the image start.
4. run the same three requests against a second server started WITHOUT `--kv-tree` and compare outputs token by token (greedy). If a near-tie shows up (logit gap < 0.05), record it as `cmp_tie` instead of a mismatch (see `artifacts\t32-stage3-ab.ps1` for the tie-aware comparison).
5. assert the tree log contains `parked` for a media prompt and `restored` for request 2.

- [ ] **Step 3: run, collect, archive**

Run the script with `CUDA_VISIBLE_DEVICES=0`; save the full log to the evidence file; include the tree stats lines (`kv tree stats:`), prompt_n values, and the comparison verdict.

- [ ] **Step 4: update the channel docs**

Append a section to `RESULTS.md`, `STATUS.md` and the T32 task file: what was built, the evidence file, the commands, the limitations, and the ruling list from the SDD ledger.

- [ ] **Step 5: commit** (repo changes only; the channel docs live outside the repo and are not committed)

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tools/server/server-common.h tools/server/server-common.cpp tools/server/server-context.cpp tests/test-t32-tree.cpp
git commit -m "kv tree : media reuse end to end"
```
(only if anything is still uncommitted; the E2E script and evidence are artifacts, not repo files)
