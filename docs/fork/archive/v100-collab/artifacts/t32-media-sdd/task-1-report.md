# Task 1 report: tree storage side (data model, mapping, park, match)

Branch: t32-media. Commit: 5722fac1e "kv tree : media-aware blocks and matching" (base d701c2c13).
Not pushed, no PR created.

## What I implemented

All 7 steps of the brief.

server-kv-tree.h:
- new `kv_tree_media` (idx/id/n_tok/n_pos).
- `kv_tree_block` gains `int64_t tok0` and `std::vector<kv_tree_media> media`.
- `kv_tree_anchor` gains `int64_t tok`; `kv_tree_anchor_in.pos` renamed to `tok`;
  `kv_tree_restore_anchor` gains `tok`.
- new signatures for park/restore/capture_anchor/drop_seq/match; all take the media span vector.
- indexes: `blocks_at` re-keyed to `tok0`; new `blocks_by_pos0` keyed by `pos0`.

server-kv-tree.cpp:
- static helpers: `pos_at`, `pos_last_cell`, `pos_anchor`, `block_end`, `chain_split`,
  `chain_hashes` (token bytes + compact media record, chained), `media_at`.
- `match(tokens, media)`: full blocks by hash + tok0 check + memcmp; tail block via
  `blocks_at[tail_a]` token walk plus the media identity and chunk-boundary clamp.
- `park(...)`: guard uses `pos_last_cell(media, L)`; blocks come from `chain_split` and are
  stored with `tok0/pos0/pos1/media`; both indexes updated; tip anchor is
  `store_anchor(tip, L, pos_anchor(media, L), ...)`; candidates are position-mapped and
  skipped when their token index falls inside a chunk; the refcount sync loop compares
  anchor `tok`; ranges use `pos_at` at both ends.
- `store_anchor(hash, tok, pos, kind, ...)` sets `a.tok`.
- `containing_block` uses `blocks_by_pos0`; `has_successor` uses `blocks_at[b.tok0 + size]`;
  `remove_block` erases from both indexes.
- `capture_anchor(...)`: expected end = `pos_last_cell(media, tok)`, anchor pos = expected + 1,
  `containing_block(anchor pos)`, F1 fallback in token space (`b.tok0 < tok <= tok0 + size`,
  token prefix compare, media prefix compare in both directions), `prev` scan and fork_step
  by positions.
- `restore(...)`: anchor selection by `a.tok <= m.deep` with `C` a token index and `(c_hash, c_pos)`
  remembered; block loads gated by `tok0`; trim via `seq_rm(pos_at(media, C), -1)`; anchor loaded
  by its own key; out anchors carry `tok` + `pos`.
- `drop_seq(tokens, media)`.

tests/test-t32-tree.cpp:
- `kv_tree_io_fake` records `ranges` in `get_range`.
- all existing call sites gain the empty media vector; `capture_partial` sets `tok`.
- new `run_logic_media()` with 28 checks; called from `run_logic()`.

## Tests and exact results

Build: `<TEMP>\v100\build_test_t32.cmd test-t32-tree` - clean, no warnings.

Logic: `build\bin\Release\test-t32-tree.exe --mode logic` (stdout+stderr to a file)
- exit=0, 108 PASS, 0 FAIL (case-sensitive).
- 28 of the PASS lines are the new `media:` checks; all pre-existing text checks stay green.

Model: `$env:CUDA_VISIBLE_DEVICES='0'`; `build\bin\Release\test-t32-tree.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on --mode model --ram-mib 4096 -c 8192`
- exit=0, 65 PASS, 0 FAIL; all 7 scenarios ran (tip, fork, sparsify, ssd, restore anchors,
  fork miss, unaligned).

New media tests cover:
- identity in a block hash (different id / n_tok / n_pos does not match past the chunk),
- chunk-straddling block split (4 blocks: 0, 512, chunk end 2026, 2538; no range boundary
  inside the chunk position span; ranges end at `pos_at(L)`),
- park/restore round trip (C=3000, anchor tok=3000 pos=2006, trim at mapped pos 2005,
  4 block loads, tokens_reused=3000),
- tail-block identity clamp through the media comparison (the token walk passes, k clamps
  to 2000-1536=464, deep=2000),
- malformed truncated request (2024 tokens, chunk 2000..2048): n_part clamps to the chunk start,
  never verifies into the chunk,
- park guard uses the last KV cell (chunk 2000..3000 with n_pos=16: park succeeds at pos_max=2000,
  refused at 2015).

## Files changed

- tools/server/server-kv-tree.h
- tools/server/server-kv-tree.cpp
- tests/test-t32-tree.cpp
(3 files, +556/-184)

## Self-review findings (fixed before commit)

- First test expectations assumed the chunk at idx=1000 crosses the boundary at 512; it crosses
  1024, so the correct block starts are 0/512/2026/2538 (4 blocks). Test fixed; the module was right.
- My first `capture_anchor` draft added a strict `tokens.size() == tok` contract and derived the
  expected end from `tokens.size()`. That broke 8 pre-existing text checks because the harness
  (and the server pre-heal call pattern) passes the full token vector while the io state ends at
  tok. Reworked per controller decision 3 (see decisions below).
- Tail-id clamp expectation was written as 2048-1536; the correct clamped value is 2000-1536=464.
- All three files are ASCII only (checked byte-wise).

## Decisions made autonomously

1. `kv_tree::match` moved from private to public. Controller decision 2 requires passing a
   malformed (truncated) token vector "straight to match()" and asserting `n_part`; the harness
   includes only the header, so match must be callable from the test. No server caller is affected.
2. `capture_anchor` computes the expected io end from `pos_last_cell(media, tok)`, not from
   `tokens.size()`. Rationale: controller decision 3 (existing harness call sites must stay
   unchanged except the media argument) and the current server call site pass a token vector that
   can extend past the capture point; for empty media plus tok==pos this is byte-identical to the
   old `pos_max() == tok - 1` guard. The brief's `tokens.size()` wording only matches when
   tokens.size()==tok, which is the server's heal-capture case.
3. `capture_anchor` additionally refuses when `tok` falls strictly inside a media chunk
   (same defensive posture as the park candidate skip and the match clamp): the state after a
   partial chunk is not defined, and such an anchor could later be selected by restore. Not
   exercised by any current caller.
4. Removed the now-unused `chunk_hash` static helper (replaced by inline XXH64 in `chain_hashes`).
5. `has_successor` was not listed in the brief; it had to switch from `blocks_at[b.pos1]` to
   `blocks_at[b.tok0 + tokens.size()]` because `blocks_at` is now keyed by token index.
6. In `match`'s full-block loop I kept the brief's `b.tok0 != a` check as defense (a mismatched
   stored block breaks the chain instead of being silently skipped).
7. Test media spans are built with a local `pos_at` lambda (controller decision 1: do not export
   the cpp helpers); the alignment check asserts no recorded range endpoint lies strictly inside
   the chunk position span `[pos_at(idx), pos_at(idx+n_tok))`, which is the position-space
   translation of the brief's token-space intent.

## Concerns

- server-context.cpp still uses the old signatures and does not compile until Tasks 2/3. This is
  expected; the test target compiles only test-t32-tree.cpp and server-kv-tree.cpp. Do not build
  the server target or run the server on this commit.
- The anchor disk envelope is unchanged, so `tok` is not persisted. This is per spec ("disk
  envelope format unchanged"); anchors are never reloaded with their metadata from disk, only
  their payloads, and the tree is wiped on restart, so no information is lost.
- The parked block payload for a huge media chunk can span many positions; accepted limitation
  from the spec.

# Fix report (review follow-up)

Commit 2e341d9e9 "kv tree : walk back interleaved blocks in chain lookup" on t32-media.
Builds on 5722fac1e; not pushed.

## Changes

1. Important (plan-mandated): single-predecessor lookup missed covering blocks when chains
   interleave in the indexes.
   - `containing_block`: now walks predecessor keys of `blocks_by_pos0` backwards until a block
     on the given chain is found, then applies the range check. The first chain block found is
     the only possible cover (chain blocks have increasing, disjoint spans), documented with a
     comment; returning 0 when it fails the range check is sound for the same reason.
   - capture F1 fallback: now walks predecessor keys of `blocks_at` backwards until a block
     covers `tok` and passes the token prefix and media prefix checks. On a failed check it keeps
     walking (a later covering block can be from another chain), bounded by the map begin.
     Rationale recorded: unlike chain blocks, covering blocks from different chains are not
     unique in token space, so the "first hit" shortcut does not apply here.
2. Minor: header comments now match the token/position split: `kv_tree_match.deep` is a token
   index; `kv_tree_restore.C` and `heal` are token indices (positions are used for KV I/O only).

## Regression test (tests/test-t32-tree.cpp, run_logic_media)

Interleaved chains in one tree: a text chain of 3000 tokens is parked first (block keys at
multiples of 512), then a media chain with chunk {idx=1000, n_tok=1026, n_pos=32} is parked
with a checkpoint candidate at tok=2534. That candidate maps to position 1540, whose nearest
`blocks_by_pos0` predecessor is the text chain block at key 1536; the media chain block covering
1540 starts at key 1032. Checks: the candidate is adopted (`anchors_added == 3`), a restore of
the media chain returns two anchors, and the candidate anchor is present with
tok=2534 / pos=1540.

Verified the test catches the old behavior: with only server-kv-tree.cpp stashed to the previous
commit, logic mode fails exactly on this test
- "media: the interleaved candidate is adopted" FAIL (got 2, want 3)
- "media: both interleaved anchors are on the path" FAIL (got 1, want 2)
- "media: the interleaved candidate anchor is usable" FAIL
The stash was popped and the fix rebuilt before the final runs.

## Commands and results

- Build: `<TEMP>\v100\build_test_t32.cmd test-t32-tree` - clean; the
  only warnings are pre-existing ones from src/llama.cpp, none from the touched files.
- Logic: `build\bin\Release\test-t32-tree.exe --mode logic` - exit=0, 114 PASS, 0 FAIL
  (was 108 PASS before this fix; +6 checks from the new interleaved test).
- Model (2B): `$env:CUDA_VISIBLE_DEVICES='0'` then
  `build\bin\Release\test-t32-tree.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on --mode model --ram-mib 4096 -c 8192`
  - exit=0, 65 PASS, 0 FAIL (unchanged by the fix).
- ASCII check on the three files: no non-ASCII bytes.
