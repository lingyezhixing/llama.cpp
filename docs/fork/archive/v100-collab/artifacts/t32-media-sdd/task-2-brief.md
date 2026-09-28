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

