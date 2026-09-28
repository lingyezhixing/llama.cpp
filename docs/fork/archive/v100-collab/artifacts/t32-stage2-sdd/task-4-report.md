# Task 4 Report: self-heal capture + fork promote/prune

## Status: DONE_WITH_CONCERNS (functional goal met; four brief defects corrected, see Deviations)

## What I implemented

1. `tools/server/server-kv-tree.h`
   - public: `capture_anchor(kv_tree_io &, kv_tree_io *, const std::vector<llama_token> &, llama_pos)`
     after `restore`, with the brief's two comment lines.
   - private: `promote_prune(llama_pos)` and
     `remove_anchor(std::map<std::pair<uint64_t, llama_pos>, kv_tree_anchor>::iterator)`.
   - members: `llama_pos last_promote = -1;`.
   - `kv_tree_anchor`: `bool transient = false;` after `pinned` (see Deviation D1).
2. `tools/server/server-kv-tree.cpp`
   - `#include <filesystem>`.
   - After `store_anchor`: `anchor_bytes` (brief verbatim; unused until T5), `remove_anchor`
     (transient/on_disk/RAM accounting exactly as briefed), `capture_anchor` (chain from
     `tokens`, `containing_block(pos, &chain)`, duplicate heat bump, anchor_step spacing,
     ONDEMAND store, `promote_prune`), `promote_prune` (verbatim, including the conservative
     first-promote-records-only cursor).
   - Corrected the exact-state guard to `io_tgt.pos_max() != pos - 1` (Deviation D2) and added
     WRN refusals on partial capture failure (Deviation D4).
3. `tests/test-t32-tree.cpp`
   - Replaced the fork A' block with the self-heal version: restore (assert C=512, heal=1024),
     replay [512, 1024), `capture_anchor(io, nullptr, tok_a2, 1024)`, replay [1024, 1536),
     generation compare, then `run_tree_path` again (assert C=1024, heal=-1, tokens match).
   - Two required transcription corrections: `r.C`/`r.heal` instead of `r.res.C`/`r.res.heal`
     (D3), and the first replay prefills the slice [0, 1024) (D4).

## Commands run and results

Build (exit 0; only the pre-existing C4297 warning in `src/llama.cpp`; no warnings in the
three changed files):

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

Model mode (Qwen3.5-2B, CUDA_VISIBLE_DEVICES=0, -ngl 99, -fa on, -c 4096, --ram-mib 4096;
exit 0, 28/28 PASS):

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
[t32-tree] fork: heal capture at the fork point               PASS
[t32-tree] fork: A' tokens match the baseline                 PASS
[t32-tree] heal: A' restores at the self-healed anchor        PASS (got 1024, want 1024)
[t32-tree] heal: no second heal needed                        PASS (got -1, want -1)
[t32-tree] heal: tokens still match the baseline              PASS
[t32-tree] sparsify: park                                     PASS
[t32-tree] sparsify: [0, 1024) restores at 512                PASS (got 512, want 512)
[t32-tree] sparsify: [0, 1024) tokens match                   PASS
[t32-tree] sparsify: [0, 1536) restores at 1536               PASS (got 1536, want 1536)
[t32-tree] sparsify: [0, 1536) tokens match                   PASS
[t32-tree] sparsify: full length restores at the tip          PASS (got 2048, want 2048)
[t32-tree] sparsify: full length tokens match                 PASS
```

fork dump evidence: `anchor 0fe844ccd2ec2fee@1024 kind=2 ref=1 heat=1` - the on-demand
(ONDEMAND) heal anchor attached to block [512, 1024) at the fork point; anchors 4 ram,
0 skipped; all four restores hit; generated sequences bit-exact vs fresh-prefill baselines.

## Files changed and commit

- `tools/server/server-kv-tree.h` (+11/-0)
- `tools/server/server-kv-tree.cpp` (+105/-0)
- `tests/test-t32-tree.cpp` (+20/-4)

Commit on `t32-stage2` (local only, nothing pushed):

```
e4a50cb83 server : self-heal kv tree anchors at fork points
```

Message body: `Assisted-by: opencode`.

## Self-review findings

- Completeness vs brief: all Step 1-3 constructs present. `promote_prune` and `remove_anchor`
  bodies are verbatim; `capture_anchor` is verbatim except `pos - 1` (D2) and the two WRN
  refusals (D4). Header declarations and member placement per brief.
- `anchor_bytes` is intentionally unused in T4 (brief-mandated; T5 uses it in
  `settle`/`load_payload`). MSVC /W3 does not emit C4505; the build shows no new warnings.
- The first model run confirms the fork A' path: the heal anchor is reachable on the second
  restore (`heal: A' restores at the self-healed anchor`, no second heal, tokens match).
- `last_promote` ends at 1024 after the model run; the prune loop is not entered (first
  promotion records only) and no deletion occurs, exactly the conservative spec 4 reading.
- ASCII-only: 0 bytes > 127 in all three files. No new comments beyond the brief's two
  header lines. No edits under `src/`. Working tree clean after commit.

## Deviations (all brief defects; each fix is minimal)

- D1 (compile blocker): the brief's `remove_anchor` reads `a.transient`, but
  `kv_tree_anchor` has no `transient` field in the T3 tree (the plan adds it in T5 Step 1).
  Fix: added `bool transient = false;` to `kv_tree_anchor` after `pinned` (the plan's exact
  T5 snippet text, default false, so T4 accounting is unchanged). I did not add the field to
  `kv_tree_block`; T5 still adds that one. T5/T6 use `remove_anchor` for transient payloads,
  so the branch must exist to avoid an accounting leak when they land.
- D2 (functional blocker): the brief checks `io_tgt.pos_max() != pos`, but this module stores
  anchor positions as the number of covered tokens (`park` tip check is `pos_max() != L - 1`;
  `capture_partial` uses the same count). With the brief's check no capture could ever
  succeed (state after 1024 tokens has pos_max 1023). Fix: `io_tgt.pos_max() != pos - 1`;
  the refusal WRN text is unchanged.
- D3 (compile blocker): the A' block declares `const auto r = tree.restore(...)` and then
  reads `r.res.C` / `r.res.heal`, but `restore` returns `kv_tree_restore` (members `C`,
  `heal`, `anchors`), not a `path_result`. Compile error C2039. Fix: `r.C` / `r.heal`.
- D4 (functional blocker + visibility): the harness `prefill` helper always runs from `from`
  to the end of the vector (Task 3 ruling 2 and its runtime evidence: this model rejects
  re-decoding existing positions). The brief's `prefill(ctx, 0, tok_a2, (int) r.res.C, 512)`
  prefills to 1536, so capture at 1024 would be refused and the following `prefill(..., 1024,
  512)` would re-decode positions 1024..1535 and fail. Fix (Task 3's established slice
  pattern): replay only the slice `[0, 1024)` first, capture, then continue with the full
  vector from 1024; batch boundaries [512,1024) / [1024,1536) unchanged. Also added the
  park-style WRN refusals when `io_tgt.get_partial` (or a provided `io_dft->get_partial`)
  fails, matching the T2 "never silent" ruling for the same failure class; both paths are
  dormant in stage 2 (io_dft is null and the model capture succeeds).

## Issues or concerns

- D1-D4 are transcription fixes, not design changes, but D4's WRN additions and D1's early
  `transient` field touch text T5 may patch; T5's (3c) patch only replaces the
  store/promote_prune ending, so the old-text match is unaffected. Controller review should
  confirm D1/D2/D4.
- `capture_anchor` still returns false silently if `store_anchor` fails; that path is
  unreachable in practice (pos > 0 and blk != 0 are checked first), so I left it verbatim.
- No token mismatches occurred; all heal assertions are green with the brief's expected
  values, so no BLOCKED evidence was needed.

---

# Fix Report: count capture refusals in the kv tree (round 1/5)

## What changed

1. Important (plan-mandated, ruling (d)): `kv_tree::capture_anchor` in
   `tools/server/server-kv-tree.cpp` now does `st.anchors_skipped++;` before `return false;`
   on both partial-capture refusal paths (`io_tgt.get_partial` failure and a provided
   `io_dft->get_partial` failure). One line each, after the existing WRN. The counters
   already existed in `kv_tree_stats`; no header change.
2. Folded Minor (controller ruling): `tests/test-t32-tree.cpp` A' block guards the replay
   prefill with `if (r.C >= 0) { ... }` so a restore miss (C = -1) can no longer make
   `prefill` read `tokens[-1]`.

Both fixes exactly as specified in the fix-round brief.

## Verification

Rebuild (exit 0, only the pre-existing C4297 warning in `src/llama.cpp`):

```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
```

Logic mode: 8/8 PASS, exit 0. Model mode (Qwen3.5-2B, CUDA_VISIBLE_DEVICES=0, -ngl 99,
-fa on, -c 4096, --ram-mib 4096): 28/28 PASS, exit 0, including `fork: heal capture at the
fork point`, `fork: A' tokens match the baseline`, `heal: A' restores at the self-healed
anchor`, `heal: no second heal needed`, `heal: tokens still match the baseline`; fork dump
still shows `anchor 0fe844ccd2ec2fee@1024 kind=2 ref=1 heat=1` and `0 skipped` (the new
counters only fire on the refusal paths, which do not trigger).

```
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
exit=0
```

## Commit

New commit (no amend), on `t32-stage2`, local only:

```
8d370c4ce server : count capture refusals in the kv tree
```

Message body: `Assisted-by: opencode`. Diff: 2 files, +5/-1, ASCII only, no `src/` edits.
