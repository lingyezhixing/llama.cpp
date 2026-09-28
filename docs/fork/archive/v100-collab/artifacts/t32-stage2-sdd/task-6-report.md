# Task 6 report: eviction order + pin + refusal + demote priority

Status: DONE_WITH_CONCERNS (one plan defect found, reproduced, fixed; see below)

## What was implemented

Transcribed the brief (Steps 1-3) faithfully into the three files, then fixed one plan defect (documented below).

### tools/server/server-kv-tree.h
- Private section, after `bool enforce_budget();`: added declarations `has_successor(...)`, `demote_one()`, `evict_anchor_one()`, `evict_block_one()`, `evict_seq_one()`, `remove_seq(...)` exactly as the brief lists them.

### tools/server/server-kv-tree.cpp
- Inserted after the `settle()` definition, in the brief's order: `has_successor`, `remove_seq`, `demote_one` (leaf penalty `-100000000`, blocks then anchors), `evict_anchor_one` (skip pinned), `evict_block_one` (skip pinned / refcount > 1 / has_successor), `evict_seq_one` (skip pinned, oldest `last_used`).
- Replaced the Task 5 demote-only `enforce_budget` with the brief's full version: demote loop -> eviction loop anchors -> leaf blocks -> whole leaf sequences -> explicit WRN + `evict_refused++` + `false`.
- Plan defect fix in `park` (2 added lines + comment): the new sequence record is pinned (`s.pinned = true`) together with its blocks/anchors while `enforce_budget` runs, and unpinned after (before the `!fits` rollback). See "Plan defect" below.

### tests/test-t32-tree.cpp
- Added `run_logic_evict()` before `run_logic()` verbatim (20 KiB then 4 KiB budgets, fake io, the 10 checks).
- `run_logic`: inserted `run_logic_evict();` after the existing `[t32-tree] mode logic` fprintf. Note: the brief said "change the first line to ..." but that line was already exactly that fprintf, so the only real change is the call insertion.

## Plan defect found and corrected

Brief's code, run verbatim, fails 3 of the 10 new checks. Reproduced output (first build, before the fix):

```
[t32-tree] evict: park refused with a tiny budget   FAIL
[t32-tree] evict: refusal counted                   FAIL (got 0, want 1)
[t32-tree] evict: refusal is visible                FAIL (got 0, want 1)
[t32-tree] evict: nothing was left behind           PASS (got 0, want 0)
LOGIC_EXIT=1
```

Cause: `park` pins its new blocks and anchors during `enforce_budget`, but not the new seq record. With the 4 KiB budget (one 4096-byte block alone equals the limit), `demote_one` cannot demote (empty `disk_dir`), `evict_anchor_one`/`evict_block_one` correctly skip all pinned payloads, then `evict_seq_one` evicts the unpinned sequence that park just added, `remove_seq` tears down its chain and everything reaches 0 bytes, so `enforce_budget` returns true and the park succeeds - contradicting the spec's "all pinned and still short -> refuse (explicit WRN + metric)" and the preflight ledger note ("park pins its own new payloads so eviction only touches older data", progress.md T6 self row).

Fix: pin the new seq for the duration of `enforce_budget` in `park` (`s.pinned = true` / `s.pinned = false`), mirroring the payload pin discipline. With all candidates pinned, the eviction loop breaks, `enforce_budget` WRNs and returns false, `park_rollback` removes the three blocks + tip anchor + seq: `park_refused == 1`, `evict_refused == 1`, `blocks_ram == 0`. No assertion was changed or weakened. The seq reference stays valid because `unordered_map::erase` of another seq does not invalidate it and no insertion happens during enforcement.

Rejected alternatives: (a) loosening the test - forbidden and wrong (spec 5.4 explicitly requires abandoning the work sequence, never sacrificing the target); (b) having `evict_seq_one` skip seqs by tip-anchor pin state - couples eviction to anchor placement and is not what "pin discipline" means; (c) skipping the newest seq - arbitrary.

## Commands and results

Build:
```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
-> [2/2] Linking CXX executable bin\Release\test-t32-tree.exe, BUILD_EXIT=0
   (only the pre-existing C4297 warning in src/llama.cpp)
```

Logic (`& '.\build\bin\Release\test-t32-tree.exe' --mode logic`, LOGIC_EXIT=0, 18/18 PASS):
```
[t32-tree] evict: park A                                                PASS
[t32-tree] evict: A has 3 blocks                                        PASS (got 3, want 3)
[t32-tree] evict: park B forces eviction                                PASS
[t32-tree] evict: blocks were evicted                                   PASS
[t32-tree] evict: park still succeeded                                  PASS (got 0, want 0)
[t32-tree] evict: A degrades to a full prefill (visible)                PASS (got -1, want -1)
[kv-tree] eviction could not free enough ram (12352 > 4096)
[kv-tree] park refused: the budget cannot hold the sequence
[t32-tree] evict: park refused with a tiny budget                       PASS
[t32-tree] evict: refusal counted                                       PASS (got 1, want 1)
[t32-tree] evict: refusal is visible                                    PASS (got 1, want 1)
[t32-tree] evict: nothing was left behind                               PASS (got 0, want 0)
[t32-tree] park A (3 chunks) ... blocks after C (tail is a new block)   PASS (8/8 original checks)
```

Model (small model, CUDA_VISIBLE_DEVICES=0, full log at `<user>\AppData\Local\Temp\opencode\test-t32-model.log`, MODEL_EXIT=0, 36/36 PASS, no regression):
```
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
tip:      park, one tip anchor, baseline, restore point 1536, no heal, tokens match, park refused past L, refusal counted          PASS (8/8)
fork:     park A/B, A/B tip restores, A' heal flow, self-healed anchor 1024, tokens match                                          PASS (13/13)
sparsify: park, 512/1536/2048 restores, tokens match                                                                               PASS (7/7)
ssd:      park, demotions to disk, 0 ram blocks, tip restore, tokens match, miss + disk error on removed files                     PASS (8/8)
```
The only "failed" strings in the log are the expected ssd negative-path messages (`[kv-tree] failed to read block ...` / `restore failed: cannot load block`) that the ssd test asserts via `disk_errors`.

## Files changed / commit

Commit `bf1e4562d` on `t32-stage2`: `server : add kv tree eviction order and pin discipline` (trailer `Assisted-by: opencode`, verified).

- tools/server/server-kv-tree.h (+10)
- tools/server/server-kv-tree.cpp (+179/-14)
- tests/test-t32-tree.cpp (+55)

## Self-review findings

- Diff vs brief: all six new methods, the full `enforce_budget`, the 20 `run_logic_evict` lines and the wiring match the brief text verbatim; the only intentional deviations are the two seq-pin lines + comment in `park` (defect fix above). Blank-line layout of the header declarations follows the brief snippet.
- ASCII only: 0 bytes > 127 in all three files. Comments: brief's comments kept, no extra commentary. No changes under `src/`. No other scope creep; `remove_block`/`remove_anchor`/`demote_*` untouched.
- Commit contains exactly the brief's Step 5 `git add` set; working tree clean afterwards.
- Behavioral note (not a defect, for final review triage): `has_successor` is position-adjacency based (`blocks_at` at `pos1`), independent of content/chain. After the leaf block A2 is evicted, A0/A1 still count as having successors because B's blocks sit at the same pos0 values; the eviction loop then moves to the whole-sequence step, which cleans them up. That matches the plan's order (blocks, then whole sequences) and the expected counters (`evicted_blocks == 3`, `evicted_seqs == 1`), but "no successor" means "no block starts where this one ends", not "no content-dependent child".
- `capture_anchor` still runs `enforce_budget` with an unpinned new anchor; eviction may drop it -> `anchors.count(key) == 0` -> counted as `anchors_skipped`. That is the Task 5 behavior and is consistent with anchors being the cheapest eviction class.

## Issues / concerns

- The seq-pin fix is a deviation from the brief's literal text. It is required for the brief's own assertions (`evict: park refused with a tiny budget` etc.) and for spec 5.4; evidence before/after is above. Plan text should be patched at the `park` tail.
- Minor (deferred): `remove_seq` inherits the T5 known minor that `remove_block` counts removals as `evicted_blocks`; `evicted_seqs` is also incremented here, so a whole-sequence eviction bumps both counters by design.
- Minor (deferred): `demote_one` does not skip pinned payloads (demotion of pinned payloads is allowed per spec, and this is what the brief specifies), while `evict_*` do skip pinned. Intentional asymmetry, matches the interface note.

---

# Fix round 1/5: leaf-sequence guard in `evict_seq_one`

Status: DONE

## Finding addressed

Review (Important, plan-mandated spec deviation): `evict_seq_one` had no leaf guard, contrary to spec 5.3 step 3 ("whole leaf sequences"). Any unpinned sequence was eligible, including a parent whose chain is a prefix of a live child, oldest `last_used` first, so a parent could be destroyed while its leaf child survived. Data safety already held (shared trunk blocks stay protected by refcount); this restores the spec's ordering.

## Change

Replaced `evict_seq_one` in `tools/server/server-kv-tree.cpp` with the controller's exact code verbatim: pins are still skipped, then a sequence is a candidate only if no other stored sequence has a strictly longer chain starting with the same hashes (`std::equal` prefix test), then the oldest `last_used` leaf is removed via `remove_seq`. `std::equal` needs `<algorithm>`, already included at the top of the cpp; no header or test changes. No assertion was touched. ASCII only (0 bytes > 127), no `src/` edits.

The existing logic eviction scenario still passes because A's and B's chains diverge at block 0 (neither is a prefix of the other) and tree2 has a single seq that is pinned during enforcement; both are leaves, so selection is unchanged. The model run is the regression evidence for the no-eviction paths.

## Commands and results

Build:
```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
-> BUILD_EXIT=0 (only the pre-existing C4297 warning in src/llama.cpp)
```

Logic (LOGIC_EXIT=0, 18/18 PASS):
```
evict: park A / A has 3 blocks / park B forces eviction / blocks were evicted /
park still succeeded / A degrades to a full prefill (visible) / park refused with a
tiny budget / refusal counted / refusal is visible / nothing was left behind      PASS
park A (3 chunks) ... blocks after C (tail is a new block)                        PASS (8/8)
```

Model (small model, CUDA_VISIBLE_DEVICES=0, log at `<user>\AppData\Local\Temp\opencode\test-t32-model-fix1.log`, MODEL_EXIT=0, 36/36 PASS, 0 FAIL):
```
tip PASS 8/8, fork PASS 13/13, sparsify PASS 7/7, ssd PASS 8/8
```
The only "failed" strings in the log are the expected ssd negative-path messages (`failed to read block` / `restore failed`) asserted by the ssd test.

## Commit

New commit (no amend) `19acf3fb2` on `t32-stage2`: `server : evict only leaf sequences in the kv tree`, trailer `Assisted-by: opencode`. One file changed (+19/-2). Working tree clean.

## Concerns

- None new. The prefix test is O(seqs^2 * chain) per eviction step; with the tiny number of stored sequences in stage 2 this is fine (final review can triage if sequence counts grow).

