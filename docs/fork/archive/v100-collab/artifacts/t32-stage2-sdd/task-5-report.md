# Task 5 report: SSD tier (envelope/files/move semantics) + budget degradation

Status: DONE

## What was implemented

Transcribed the brief faithfully into the three files.

### tools/server/server-kv-tree.h
- `kv_tree_block`: added `bool transient = false;` after `pinned` (anchor already had it from Task 4).
- Private section: removed `make_room(size_t)`; added `enforce_budget()`, `park_rollback(...)`, `remove_block(...)`, the two `load_payload` overloads, `demote_block`, `demote_anchor`, `settle()`, `write_disk`, `read_disk`, `block_path`, `anchor_path`.

### tools/server/server-kv-tree.cpp
- Envelope helpers after `chain_hashes`: magic constants, `put_u32/put_i32/put_u64`, `kv_env`, `envelope_make`, `envelope_parse` (payload XXH64 chained check, size/version validation).
- After `containing_block`: `block_path`/`anchor_path`, `write_disk` (tmp + atomic-ish rename), `read_disk`, `demote_block`, `demote_anchor` (envelope write, RAM->disk counter moves, `bytes_store`), `load_payload` for block and anchor (read + verify + `transient = true`, RAM counters up), `settle()` (resolves transient loads to a single authoritative copy: fit -> delete file and keep RAM; not fit -> drop RAM, keep disk).
- Deleted `make_room` definition.
- Added `enforce_budget()` (demote blocks then anchors until `bytes_ram <= ram_limit`, returns fit), `remove_block()` (removes dependent anchors via `remove_anchor`, tier-aware counters incl. transient double-count, unlinks disk file, cleans `blocks_at`, `evicted_blocks++`), `park_rollback()` (remove new blobs via `remove_block`, unwind touched anchor refcounts, decrement matched path refcounts, erase seq record).
- `park`: removed the old pre-check refusal block; after the seq record, added the pin / `enforce_budget` / unpin / `park_rollback` tail verbatim.
- `capture_anchor`: after `store_anchor` added `if (!enforce_budget() || anchors.count(key) == 0) { st.anchors_skipped++; return false; }` before `promote_prune`.
- `restore`: inserted `load_payload` in the whole-block, partial-block and anchor load paths; `kv_tree_anchor & a` is now non-const; `settle()` inserted after the two unpin loops and before `if (!ok)`.

### tests/test-t32-tree.cpp
- Added `#include <filesystem>`.
- Added `scenario_ssd` after `scenario_sparsify` (64 KiB RAM / 64 MiB disk / temp dir `t32-tree-test`, park 1536 tokens, assert disk demotion, SSD round-trip against baseline, delete block files, assert miss + `disk_errors` bump, remove temp dir).
- `main`: `ret |= scenario_ssd(ctx, cfg);`.

Deviation from the brief (plan defect, documented below): added one line in `scenario_ssd`:
`llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);` before the initial `prefill`.

## Plan defect corrected

The brief's `scenario_ssd` runs after `scenario_sparsify` on the same context but does not clear sequence 0 before its first prefill. `scenario_sparsify` ends with `generate` at position 2055, so the prefill at position 0 fails:

```
E init: the tokens of sequence 0 ... last position ... X = 2055 ... starting position ... Y = 0
E decode: failed to initialize batch
[t32-tree] decode failed at token 0
[kv-tree] park refused: sequence end 2055, expected 1535
```

`scenario_fork` and `scenario_sparsify` both call `llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1)` before their first prefill; `scenario_tip` runs first on a fresh context. The fix adds the same line to `scenario_ssd`. No assertion was changed or weakened.

## Commands and results

Build:
```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
-> [10/11] Linking CXX executable bin\Release\test-t32-tree.exe  (no errors; only a pre-existing C4297 warning in src/llama.cpp)
```

Logic:
```
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
-> 8/8 PASS, EXIT=0
```

Model (Qwen3.5-2B-UD-Q4_K_XL.gguf, -ngl 99 -fa on -c 4096 --ram-mib 4096, CUDA_VISIBLE_DEVICES=0):
```
tip:      park, one tip anchor, baseline generation, restore point is the tip (1536),
          no heal needed, tokens match the baseline, park refused past L, refusal counted  -> PASS
fork:     park A/B, A/B tip restores, heal flow, self-healed anchor, tokens match         -> PASS
sparsify: park, [0,1024)->512, [0,1536)->1536, full->2048, tokens match                   -> PASS
ssd:      park PASS
          blocks demoted to disk PASS
          tip anchor demoted to disk PASS
          no blocks left in ram PASS (got 0, want 0)
          restore point is the tip PASS (got 1536, want 1536)
          tokens match the baseline PASS (SSD round-trip bit-exact)
          restore misses when block files are gone PASS (got -1, want -1)
          disk error counted PASS
EXIT=0
```

Post-run `dump` shows the settled state is single-copy: `blocks: 0 ram, 3 disk`, `anchors: 0 ram, 1 disk`. Temp dir `%TEMP%\t32-tree-test` removed (`Test-Path` = False). No other disk location touched.

## Files changed / commit

Commit `0c9d8ffd9` on `t32-stage2`: `server : add ssd tier and single-copy payload moves to the kv tree` (Assisted-by: opencode)

- tools/server/server-kv-tree.h (+21/-1)
- tools/server/server-kv-tree.cpp (+526/-10)
- tests/test-t32-tree.cpp (+54/-0)

## Self-review findings

- Step-by-step diff against the brief: envelope helpers, file I/O, payload functions, `settle`, `enforce_budget`, `remove_block`, `park_rollback`, park tail, capture_anchor tail, restore edits all match the brief text verbatim.
- `#include <filesystem>` was already present in `server-kv-tree.cpp` (Task 4 `remove_anchor`); not duplicated.
- `make_room` fully removed (grep across *.h/*.cpp: no hits).
- ASCII only: 0 non-ASCII bytes in all three changed files.
- Comments are the brief's (minimal); no extra commentary added. No changes under `src/`. No scope creep: `remove_block`/`park_rollback`/`settle` are exactly the brief's code intended for Task 6 reuse.
- The only intentional deviation is the one `seq_rm` line in `scenario_ssd` (defect above).

## Issues / concerns

- `read_disk` uses `long` from `ftell`; on Windows that caps single files at 2 GiB. Brief-specified code, payloads here are ~6 MB, left as is.
- In `capture_anchor`, a failed `enforce_budget()` returns false but leaves the just-stored anchor in the map (recovers once RAM frees up; no rollback requested by the brief). Flagging only in case Task 6/T7 want different semantics.
