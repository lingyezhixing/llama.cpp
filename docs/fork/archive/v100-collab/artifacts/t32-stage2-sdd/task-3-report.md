# Task 3 Report: checkpoint adoption + sparsify + fork

## Status: DONE

## What I implemented

1. `tools/server/server-kv-tree.h`
   - private: `uint64_t containing_block(llama_pos pos) const;` after `store_anchor`.
   - `blocks_at` changed from `std::unordered_map` to `std::map` (controller ruling 1).
2. `tools/server/server-kv-tree.cpp`
   - `containing_block` after `store_anchor`, using `blocks_at.lower_bound(pos)` then
     `--it` (ruling 1). The brief's `upper_bound` cannot compile on `unordered_map`
     and would also skip checkpoints on a chunk boundary.
   - `park`: removed `(void) checkpoints;`; inserted the brief's adoption block after
     the tip `store_anchor(...)` and before `kv_tree_seq & s = seqs[tip];` verbatim:
     sorted candidates, `anchor_step` greedy (conflict keeps the earlier one),
     `containing_block` attach, `store_anchor` per kept candidate, then the refcount
     sync pass that appends to `touched`. `touched` is kept exactly as written so
     Task 5 can consume it in its park tail.
3. `tests/test-t32-tree.cpp`
   - Inserted `make_fork_tokens`, `scenario_fork`, `scenario_sparsify` after
     `scenario_tip`; `main` model branch now chains tip, fork, sparsify.
   - Ruling 2 (capture position): before each capture the prefix [0, pos) is
     prefilled (sliced vector), then the full vector continues from pos with ubatch
     512. The batch boundaries [0,512),[512,1024),... are unchanged.
   - Runtime-required addition (see Concerns): `llama_memory_seq_rm(ctx, 0, -1, -1)`
     at the top of `scenario_fork` and `scenario_sparsify`, matching the existing
     pattern (`run_baseline`, park B). Scenarios are chained on one context and the
     model enforces M-RoPE positions (needs X < Y), so the first fork prefill at
     position 0 was rejected while seq 0 still held the tip scenario's end (X=1543).

## Commands run and results

Build (success, exit 0; only the pre-existing C4297 warning in `src/llama.cpp`):

```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
```

Logic mode (exit 0, 8/8 PASS):

```
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
[t32-tree] park A (3 chunks)                     PASS
[t32-tree] blocks after A                        PASS (got 3, want 3)
[t32-tree] park B2 (shares 2 chunks)             PASS
[t32-tree] blocks after B2 (dedup)               PASS (got 4, want 4)
[t32-tree] park A again (identical)              PASS
[t32-tree] blocks after A re-park (no duplicate) PASS (got 4, want 4)
[t32-tree] park C (partial tail block)           PASS
[t32-tree] blocks after C (tail is a new block)  PASS (got 5, want 5)
```

Model mode (Qwen3.5-2B, RTX, -fa on, c 4096, ram 4096 MiB; exit 0, 21/21 PASS):

```
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
[t32-tree] tip: park                                          PASS
[t32-tree] tip: one tip anchor                                PASS (got 1, want 1)
[t32-tree] tip: baseline generation                           PASS
[t32-tree] tip: restore point is the tip                      PASS (got 1536, want 1536)
[t32-tree] tip: no heal needed                                PASS (got -1, want -1)
[t32-tree] tip: tokens match the baseline                     PASS
[t32-tree] tip: park refused when the sequence ends past L    PASS
[t32-tree] tip: refusal counted                               PASS (got 1, want 1)
[t32-tree] fork: park A                                       PASS
[t32-tree] fork: park B                                       PASS
[t32-tree] fork: A restores at the tip                        PASS (got 2048, want 2048)
[t32-tree] fork: A tokens match the baseline                  PASS
[t32-tree] fork: B restores at the tip                        PASS (got 2048, want 2048)
[t32-tree] fork: B tokens match the baseline                  PASS
[t32-tree] fork: A' restores at the deepest usable anchor     PASS (got 512, want 512)
[t32-tree] fork: A' asks for a heal at the fork point         PASS (got 1024, want 1024)
[t32-tree] fork: A' tokens match the baseline                 PASS
[t32-tree] sparsify: park                                     PASS
[t32-tree] sparsify: [0, 1024) restores at 512                PASS (got 512, want 512)
[t32-tree] sparsify: [0, 1024) tokens match                   PASS
[t32-tree] sparsify: [0, 1536) restores at 1536               PASS (got 1536, want 1536)
[t32-tree] sparsify: [0, 1536) tokens match                   PASS
[t32-tree] sparsify: full length restores at the tip          PASS (got 2048, want 2048)
[t32-tree] sparsify: full length tokens match                 PASS
```

All generated sequences are bit-exact vs their fresh-prefill baselines; no T24
near-tie case occurred. Adoption semantics visible in `dump()`:

- fork: `anchor cffc5f188e0b1ff2@512 kind=1 ref=2` - checkpoint adopted on block
  [0,512) and refcount synced by park B; shared prefix blocks ref=2.
- sparsify: `anchors: 3 ram ... 3 added, 1 skipped` - ck2@1024 dropped by the
  anchor_step=1024 greedy, ck1@512 and ck3@1536 kept.

First model run evidence (before the seq_rm fix): decode failed at token 0 with
"the last position stored in the memory module ... is X = 1543 ... starting
position of Y = 0 ... required X < Y". This also confirmed that the model rejects
re-decoding existing positions, so the original second `prefill` call in the brief
would fail even on a clean sequence; the sequential prefix continuation avoids it.

## Files changed and commit

- `tools/server/server-kv-tree.h` (+5/-1)
- `tools/server/server-kv-tree.cpp` (+94/-3)
- `tests/test-t32-tree.cpp` (+114/-1)

Commit on `t32-stage2` (local only, nothing pushed):

```
0ca557660 server : adopt checkpoint anchors in the kv tree
```

Message body: `Assisted-by: opencode`.

## Self-review findings

- Diff vs brief: the adoption block, both scenarios (except the approved capture
  lines and the two scenario-start clears), `make_fork_tokens` and the main wiring
  are verbatim. Deviations: (1) std::map + lower_bound, (2) prefix prefill before
  capture, (3) seq_rm at scenario start. (1) and (2) were ruled by the controller;
  (3) is explained in Concerns.
- ASCII-only: 0 bytes > 127 in all three files. No new comments beyond the brief's
  two; no edits under `src/`; working tree clean after commit.
- `touched` is populated exactly as the brief writes it (tip entry first), ready for
  the Task 5 park tail.
- `blocks_at` as std::map stays API-compatible with later tasks (`operator[]`,
  `find`, `erase`) and gives deterministic iteration order.

## Issues or concerns

- Deviation 3 (seq_rm at the top of fork/sparsify) was not pre-approved; it was
  required at runtime because the brief assumed a fresh context. It is the same
  one-line pattern already used in this file. Controller review should confirm.
- `blocks_at` container type changed from the Task 1 header; later task snippets use
  only compatible operations, but the plan text still says `unordered_map`.
- The containment rule is (pos0, pos1], i.e. pos attaches to the block ending at
  pos; this is what makes fork A' C=512 and sparsify C=1536 work and is now
  documented by the `lower_bound` + condition pairing.
- No token mismatches observed, so no BLOCKED evidence was needed.

---

# Fix Report: filter adopted anchors by parked chain and count skips (round 1/5)

## What changed

Three review findings (all Important, plan-patched), in `tools/server/server-kv-tree.h` and
`tools/server/server-kv-tree.cpp`:

1. Invalid candidate guard now bumps `st.anchors_skipped++` before `continue`
   (expected filter, no WRN).
2. `containing_block` is chain-filtered: new signature
   `uint64_t containing_block(llama_pos pos, const std::vector<uint64_t> * chain) const;`
   It keeps the `lower_bound` lookup and now skips candidate hashes not present in the
   parked chain (`chain != nullptr` -> `std::find`), so an anchor can no longer attach to
   a divergent variant that restore cannot reach. Adoption calls it as
   `containing_block(c.pos, &h)`. The `blk == 0` branch now prints a WRN before counting:
   "candidate at %d skipped: no containing block on the parked chain". `capture_anchor`
   (Task 4) does not exist yet and was not touched.
3. `store_anchor` failure branch now prints "candidate at %d skipped: anchor store failed"
   before `st.anchors_skipped++` (unexpected failure -> WRN + counter).

Spacing skips keep the existing counter-only behavior (expected filter), per the ruling.

## Verification

Rebuild:

```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
```

Success; `test-t32-tree.exe` linked. Logic mode: 8/8 PASS, exit 0. Model mode: all 21
assertions PASS, exit 0, adoption outcomes unchanged (fork: 3 anchors added, 0 skipped,
A' C=512/heal=1024; sparsify: 3 added, 1 skipped by anchor_step, C=512/1536/2048; all
token sequences bit-exact vs baselines). No new WRN lines: both new warnings are on
paths that did not trigger.

```
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
[kv-tree] anchors: 3 ram, 0 disk, 3 added, 0 skipped        (fork)
[kv-tree] anchors: 3 ram, 0 disk, 3 added, 1 skipped        (sparsify)
exit=0
```

## Commit

New commit (no amend), on `t32-stage2`, local only:

```
5b843e2f6 server : filter adopted anchors by parked chain and count skips
```

Message body: `Assisted-by: opencode`. Diff: 2 files, +12/-5, ASCII only, no `src/` edits.
