# Task 2 report: tree retrieval side (restore, anchors, capture, drop)

Branch: t32-media. Commit: 70157be75 "kv tree : media-aware restore and anchors" (base 2e341d9e9).
Not pushed, no PR created.

## What I implemented

Step 4 of the brief, plus verification of steps 1-3:

- Audited `restore`, `capture_anchor` and `drop_seq` in tools/server/server-kv-tree.cpp line by line
  against the brief and the spec. All of steps 1-3 were already present and correct at 2e341d9e9:
  Task 1 had to convert these functions for compilation (plan line 234) and did the full media
  conversion, reviewed clean. No code change was needed.
  - restore: anchors filtered by `a.tok <= m.deep`, largest `a.tok` selected with `(c_hash, c_pos)`;
    block loads gated by `b.tok0 < C`; partial block when `m.n_part > 0 && C > part.tok0`; trim
    `seq_rm(pos_at(media, C), -1)`; state loaded by `(c_hash, c_pos)`; `res.anchors` carry `tok`+`pos`;
    miss heal is `m.deep`.
  - capture_anchor: expected end `pos_last_cell(media, tok)` (Task 1 decision kept, not
    `tokens.size()`), refuse inside a chunk, `apos = end + 1`, chain over `(tokens, media)`,
    `containing_block(apos, &chain)`, F1 token-space fallback, `prev` scan by pos with
    kind != MESSAGE, `fork_step`, `store_anchor(blk, tok, apos, ONDEMAND)`.
  - drop_seq: `match(tokens, media)`, tip selection unchanged.
- tests/test-t32-tree.cpp: `kv_tree_io_fake` records `seq_rm` calls in a new `trims` vector;
  `run_logic_media()` gains the retrieval tests (brief step 4 items 1-4):
  1. boundary capture/restore: park 3000 tokens with chunk {idx=1000, n_tok=1026, n_pos=32};
     `capture_anchor` at tok=2026 returns true; `restore` of a 2500-token request returns C=2026,
     one path anchor, and the fake io recorded `seq_rm(pos_at(media, C), -1)`.
  2. anchor tok vs pos: the returned chunk-end anchor has tok=2026 and pos=pos_at(media,1000)+1=1001.
  3. text-only unchanged: every pre-existing check stays green (no assertion changes).
  4. `drop_seq`: same input drops the parked sequence; a different chunk id is a no-op.

## What I tested and exact results

Build: `<TEMP>\v100\build_test_t32.cmd test-t32-tree` - exit 0. The only
warning is the pre-existing C4297 in src/llama.cpp:75; nothing from the touched files.

Logic: `build\bin\Release\test-t32-tree.exe --mode logic` (stdout+stderr to a file)
- exit 0, 127 PASS, 0 FAIL. Before this task: 114 PASS (Task 1 fix report). +13 new checks.
- New checks all pass, e.g. "media: capture at the chunk end PASS",
  "media: restore uses the chunk-end anchor PASS (got 2026, want 2026)",
  "media: the anchor keeps the mapped position PASS (got 1001, want 1001)",
  "media: the trim used the mapped position PASS",
  "media: a different id does not drop the sequence PASS", "media: the same input drops the sequence PASS".

Model (2B): `$env:CUDA_VISIBLE_DEVICES='0'` then
`build\bin\Release\test-t32-tree.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on --mode model --ram-mib 4096 -c 8192`
- exit 0, 65 PASS, 0 FAIL; all 7 scenarios ran (tip, fork, sparsify, ssd, restore anchors, fork miss,
  unaligned). Same as the Task 1 baseline, as expected: no production code changed.

Mutation checks (tests have teeth; mutations were reverted before the final build, not committed):
- `seq_rm(pos_at(media, C), -1)` -> `seq_rm((llama_pos) C, -1)`: logic exits 1, 2 FAILs, one being the
  new "media: the trim used the mapped position".
- anchor filter `it->second.tok <= m.deep` -> `it->second.pos <= m.deep`: logic exits 1, 2 FAILs
  ("restore uses the chunk-end anchor" got 2530 want 2026; "the chunk-end anchor is on the path" got 2
  want 1). This is why the extra 2530 capture is in the test: without it the brief's checks do not
  catch a position-based filter.

## Files changed

- tests/test-t32-tree.cpp (+72; committed)
- tools/server/server-kv-tree.h, tools/server/server-kv-tree.cpp: staged as the brief asks, but with
  zero diff - the media paths were already there from Task 1.
Commit 70157be75 contains only the test change.

## Self-review findings

- Verified the committed tree is exactly the tested tree: rebuild after restoring the source from git
  gave 127 PASS logic and the model binary was built from the same content.
- ASCII-only and `git diff --check` clean.
- Trim assertion is guarded by `r.C >= 0` so a restore miss cannot false-pass against the initial
  `seq_rm(-1, -1)` clear call.
- Existing text checks untouched; the model mode regression stays green.
- Kept the new checks inside `run_logic_media()` to reuse its local `pos_at`/`add_media` lambdas
  instead of duplicating or refactoring helpers.

## Decisions made autonomously

1. Did not modify server-kv-tree.cpp/h. The brief steps 1-3 were already implemented at the base
   commit; I verified each bullet against the code and covered them with tests. The commit message is
   the brief's; the commit is tests-only.
2. Kept Task 1's capture contract (`pos_last_cell(media, tok)`, not the brief's `tokens.size()`)
   per the controller context.
3. Added one extra capture at tok=2530 to the boundary scenario so the anchor filter is checked to
   use token indices, not mapped positions (mutation-verified). This is one capture plus the existing
   assertions, not a new test function.
4. Capture order in the test is 2530 before 2026: the reverse order would trip the default
   fork_step (535 < 8192) and refuse the 2530 capture. Reordering fails visibly, not silently.

## Concerns

- The retrieval code lives in commits 5722fac1e/2e341d9e9, so this task's commit looks small. The
  controller may want the review to diff 2e341d9e9..70157be75 for tests and treat the code as
  already reviewed in Task 1.
- Model mode is text-only, so media retrieval coverage is logic-only here; the real media path is
  exercised by Task 4's server E2E (0.8B-MTP + mmproj).
- The fake io's `pos_max` semantics stay coarse (last position written), so tests assert the
  recorded call arguments rather than the resulting cell layout.
