# KV tree media reuse - design spec

Status: approved for implementation (night session 2026-09-28, implementer-autonomous)
Branch: t32-media (from master d701c2c13)
Test model: <models>\Qwen3.5-0.8B-MTP (text: Qwen3.5-0.8B-UD-Q4_K_XL.gguf + mmproj-F16.gguf), device cuda0 at night

## Goal

Let the kv tree store, match and restore prompts that contain media chunks (images / audio / video),
so a media conversation gets the same cross-request KV reuse the tree already gives to text.

## Verified facts (from the code, file:line)

- `server_tokens` packs media as `LLAMA_TOKEN_NULL` placeholders; `map_idx_to_media` maps the start token
  index to the chunk. Token count != position count for M-RoPE (`server-common.h:150-158`).
- Media identity already exists: `mtmd_input_chunk_get_id()` returns the chunk id, set by
  `mtmd_helper_bitmap_init_from_buf` to the sha256 hex of the raw media bytes (`mtmd-helper.cpp:387`).
  It survives `mtmd_input_chunk_get_placeholder` (serialize/load, `mtmd.cpp:2412`) and `clone()`
  (`mtmd.cpp`, image tokens clone copies `id`). Empty id means "unknown identity".
- Position model: `server_tokens::pos_next` / `size_up_to_pos` (`server-common.cpp:389/432`).
  M-RoPE is selected from the text model rope type (`mtmd.cpp:549-565`; the 0.8B-MTP reports
  `rope type = 40`, `mrope sections = [11,11,10,0]`). For an image, all tokens share the temporal
  position `pos_0`; the next token advances by `n_pos = max(nx,ny)` (`mtmd.cpp:2470-2475`,
  `mtmd-helper.cpp:196`). Image cells therefore carry duplicate `pos` values.
- KV cells store the temporal pos (`llama-batch.cpp:93-103`). Range save/load filters cells by
  `pos in [p0,p1)` with a plain comparison (`llama-kv-cache.cpp:2104-2106`) and round-trips the
  M-RoPE x/y through `llama_kv_cell_ext` (`llama-kv-cache.cpp:2435-2438`). Duplicate pos is fine.
- The tree (`server-kv-tree.{h,cpp}`) assumes token index == position everywhere: blocks are
  512-token slices used directly as `get_range(a,b)` positions, `blocks_at` is keyed by `pos0`,
  `match()` hashes token chunks and memcmps tokens, anchors carry `pos` only, and `restore()`
  returns a `C` used both as token count and as position.
- Server gates that currently exclude media from the tree: `prompt_park` (`server-context.cpp:304`),
  `prompt_restore_tree` (`:348`), `SLOT_ERASE` drop (`:2866`). These guards are removed by this spec.
- Gates that STAY (upstream behavior, out of scope): checkpoint creation for mtmd prompts
  (`server-context.cpp:3858`), `n_cache_reuse` for mtmd (`:3423`), stock prompt cache unchanged.

## Design

### Data model

```cpp
// server-kv-tree.h
struct kv_tree_media {
    int64_t  idx   = 0;   // start token index of the chunk
    uint64_t id    = 0;   // opaque identity hash (server: FNV-1a 64 over mtmd_input_chunk_get_id)
    int32_t  n_tok = 0;   // mtmd_input_chunk_get_n_tokens
    int32_t  n_pos = 0;   // mtmd_input_chunk_get_n_pos
};
```

- Spans are sorted by `idx`, non-overlapping, each chunk is whole.
- `kv_tree_block` gains `int64_t tok0` (start token index) and `std::vector<kv_tree_media> media`
  (entries with `idx` in `[tok0, tok0 + tokens.size())`). `pos0/pos1` become real positions.
- `kv_tree_anchor` gains `int64_t tok` (token index of the captured state); `pos` stays the position.
- `kv_tree_anchor_in.pos` -> `int64_t tok`. `kv_tree_restore_anchor` gains `int64_t tok`.
- Indexes: `blocks_at` keyed by `tok0` (used by the match tail and the capture fallback);
  new `blocks_by_pos0` keyed by `pos0` (used by `containing_block`). Both maintained by park,
  remove_block and park_rollback.

### Position mapping (static helpers in server-kv-tree.cpp)

- `pos_at(media, t) = t + sum(n_pos - n_tok for entries with idx < t)`.
  Valid only at t = 0, t = L, chunk starts and chunk-aligned block boundaries (never inside a chunk).
- `pos_last_cell(media, L)`: if the last token belongs to a chunk (entry with `idx <= L-1 < idx+n_tok`,
  which always ends at L): `pos_at(entry.idx)`; else `pos_at(media, L) - 1`. Returns -1 when L == 0.
- `pos_anchor(media, L) = pos_last_cell(media, L) + 1` (the state "after" L tokens).
- `block_end(media, a, L, chunk)`: `b = min(a + chunk, L)`; for the first entry with
  `idx < b < idx + n_tok`: `b = idx + n_tok`. Chunks do not overlap, one adjustment suffices.
- `chain_split(media, L, chunk, starts)`: 0, block_end(...), ... all block starts.
- `chain_hashes(tokens, media, starts, h)`: per block, `XXH64` over the token bytes, then over a
  compact record per entry (`idx - a`, `id`, `n_tok`, `n_pos`), chained. Media identity is therefore
  part of the block hash.

### Matching (`match(tokens, media)`)

- Full blocks: hash lookup + `memcmp` tokens (the hash already binds media identity).
- Tail block: token walk against stored blocks starting at the request's last block start
  (`blocks_at`), then the media clamp: for each stored entry with `tail_a <= idx < tail_a + k`,
  require a request entry with the same `idx/id/n_tok/n_pos`; if missing or different,
  `k = idx - tail_a`; if `idx + n_tok > tail_a + k`, `k = idx - tail_a` (never cut inside a chunk).
- `m.deep` and `m.n_part` stay token indices and are always chunk-aligned.

### Park

- Guard: `io.pos_max() == pos_last_cell(media, L)`.
- Blocks `[starts[i], starts[i+1])`, `get_range(pos_at(a), pos_at(b))`, store `tok0/pos0/pos1/media`,
  update both indexes.
- Tip anchor: `store_anchor(tip, L, pos_anchor(media, L), TIP, ...)`.
- Candidates: `c.tok` -> `apos = pos_anchor(media, c.tok)`; skip a candidate whose `tok` falls inside
  a chunk; `containing_block(apos)` uses `blocks_by_pos0`; `fork_step` / anchor_step compare positions.
- Refcount sync loop compares `kv.second.tok <= L`.

### Restore

- `m = match(tokens, media)`; anchors with `a.tok <= m.deep`; `C = largest a.tok`, remember
  `(c_hash, c_pos)` of that anchor.
- Load blocks with `b.tok0 < C`, then the partial block when `m.n_part > 0 && C > part.tok0`;
  trim with `seq_rm(pos_at(media, C), -1)`; load the anchor state by `(c_hash, c_pos)`.
- `res.C = C` (token index), `res.heal = m.deep > C ? m.deep : -1`, `res.anchors` carry `tok` + `pos`.

### Capture / drop

- `capture_anchor(io, io_dft, tokens, media, tok)`: contract `tokens.size() == tok`; expected
  `io.pos_max() == pos_last_cell(media, tokens.size())`; anchor pos = expected + 1; chain over
  `(tokens, media)`; `containing_block(pos)`; F1 fallback in token space (block with
  `tok0 < tok <= tok0 + tokens.size()`, token prefix and media prefix equal); `prev` scan by pos;
  `fork_step` by positions.
- `drop_seq(tokens, media)`: `match(tokens, media)`.

### Server integration

- `server_tokens`: add `llama_pos pos_last() const` (position of the last KV cell, -1 when empty) and
  expose the media map through a const accessor for span building.
- `server-context.cpp`: static `tree_media_spans(const server_tokens &)` -> `vector<kv_tree_media>`
  using FNV-1a 64 over the chunk id string (nullptr id -> 0).
- `prompt_park`: drop the `has_mtmd` guard; guard `p_max == prompt.tokens.pos_last()`; pass spans;
  checkpoint candidates use `c.n_tokens` as `tok`.
- `prompt_restore_tree`: drop the `has_mtmd` guard; pass spans; on hit rebuild the slot prompt as
  `tokens.clone()` then `keep_first(res.C)` (preserves the media map and the has_mtmd flag);
  skip the checkpoint rebuild when the prompt contains media.
- `SLOT_ERASE`: drop the `has_mtmd` guard; pass spans to `drop_seq`.
- Heal capture: pass spans of `slot.prompt.tokens`; `tree_heal` stays a token index.
- Disk envelope format unchanged; media identity lives in RAM with the tokens (the tree is wiped on
  restart by design).

## Invariants

1. For prompts without media (empty span list) every path is behavior-identical to today.
2. `C`, `heal` and checkpoint `tok` are token indices; all KV I/O, anchor keys and `fork_step`
   distances are positions; the two are related only through the mapping helpers.
3. No block boundary and no restore point ever falls inside a media chunk.
4. Media identity (id, n_tok, n_pos) is part of the block hash; a request with a different chunk id
   cannot restore from that chain.
5. The tree stores no media bytes; the restored slot prompt is rebuilt from the incoming request.

## Tests

- Harness logic (fake io, synthetic spans): media park/restore round trip with `n_pos < n_tok`;
  identity mismatch (different id; same id, different n_tok/n_pos); straddling chunk alignment
  (block extension, no split); partial-match clamp at a chunk start; capture/restore at a
  media-ending boundary; `drop_seq` with media; all existing text-only checks stay green.
- Harness model (2B): unchanged, must stay green.
- Server E2E (0.8B-MTP + mmproj, cuda0, `--image-min-tokens 1024`, greedy): round 1 with an image
  parks; round 2 with the same conversation restores (prompt_n drops, logs show tree restore);
  a different image at the same position does not restore from the old chain; outputs vs a no-tree
  baseline are equal (tie-aware if a near-tie appears).
- Regressions: 2B tree acceptance script; text-only server run on the 0.8B.

## Limitations (accepted)

- A media chunk is atomic: one chunk belongs to one block; a very large video creates a very large
  block (budget granularity suffers).
- Stock checkpoints, `n_cache_reuse` and spec state (`data_spec`) stay media-gated.
- Media bytes are not persisted by the tree.
