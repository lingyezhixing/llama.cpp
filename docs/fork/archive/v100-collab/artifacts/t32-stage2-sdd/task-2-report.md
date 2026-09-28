# Task 2 Report: 信封缺省 + tip 锚点 + restore + llama adapter + 真机 tip 场景

## Status: DONE

## What I implemented

Transcribed the brief (Steps 1-5) verbatim; no design improvisation.

1. `tools/server/server-kv-tree.h`
   - public: `kv_tree_restore restore(kv_tree_io &, kv_tree_io *, const std::vector<llama_token> &)` after `park`.
   - private: `bool store_anchor(uint64_t, llama_pos, int, std::vector<uint8_t> &&, std::vector<uint8_t> &&)` after `match`.
2. `tools/server/server-kv-tree.cpp`
   - Added the full `kv_tree_io_llama` adapter (get_range/set_range via range ext API; get_partial/set_partial via PARTIAL_ONLY; seq_rm/pos_max via llama_memory) before the `kv_tree` constructor.
   - Added `store_anchor` after `make_room` (dedup on (hash,pos), refcount/heat/last_used; new anchor sets bytes/kind/refcount and counters).
   - Replaced `park` with the brief version: budget computed once including the tip partial payload; per-seq dedup unchanged; tip anchor stored at (tip, L) before creating the seq; no `(void) io_dft`.
   - Added `restore` after `park`: clears both sequences, matches, collects candidate hashes (path + partial), selects deepest anchor at C <= deep, pins path blocks/anchors, loads blocks covering [0, C) (append except pos0==0), trims >= C, loads the partial anchor state, unpins, updates heat and stats, returns C/heal/anchors. Failure clears tgt and counts as miss.
3. `tests/test-t32-tree.cpp`
   - Added helpers `prefill`, `generate` (greedy), `capture_partial`, `run_baseline`, `struct path_result`, `run_tree_path`, and `scenario_tip` after `run_logic`.
   - Replaced `main` with the brief version: `--mode logic|model`, `--chunk`, `--anchor-step`, `--ram-mib`, `--disk`, `--disk-mib`; common param parsing and context init only for model mode.

## Commands run and results

### Build

```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
```

Result: success, `test-t32-tree.exe` linked (only pre-existing warning C4297 in src/llama.cpp, not touched).

### Logic mode (regression)

```
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
```

All PASS, exit 0:

```
[t32-tree] park A (3 chunks)                     PASS
[t32-tree] blocks after A                        PASS (got 3, want 3)
[t32-tree] park B2 (shares 2 chunks)             PASS
[t32-tree] blocks after B2 (dedup)               PASS (got 4, want 4)
[t32-tree] park A again (identical)              PASS
[t32-tree] blocks after A re-park (no duplicate) PASS (got 4, want 4)
[t32-tree] park C (partial tail block)           PASS
[t32-tree] blocks after C (tail is a new block)  PASS (got 5, want 5)
```

### Model mode (tip scenario, Qwen3.5-2B, RTX, -fa on, c 4096, ram 4096 MiB)

```
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```

All PASS, exit 0:

```
[t32-tree] tip: park                                          PASS
[t32-tree] tip: one tip anchor                                PASS (got 1, want 1)
[t32-tree] tip: baseline generation                           PASS
[t32-tree] tip: restore point is the tip                      PASS (got 1536, want 1536)
[t32-tree] tip: no heal needed                                PASS (got -1, want -1)
[t32-tree] tip: tokens match the baseline                     PASS
[kv-tree] park refused: sequence end 1543, expected 1535
[t32-tree] tip: park refused when the sequence ends past L    PASS
[t32-tree] tip: refusal counted                               PASS (got 1, want 1)
```

Stats: 3 blocks ram (39113796 bytes), 1 tip anchor, restore 1 hit, 1536 tokens reused. The 8 generated tokens after restore are bit-exact vs the fresh-prefill baseline, as expected at np=1.

## Files changed and commit

- `tools/server/server-kv-tree.h` (+6)
- `tools/server/server-kv-tree.cpp` (+226/-3)
- `tests/test-t32-tree.cpp` (+198/-1)

Commit (on `t32-stage2`, nothing pushed):
```
7e593248c server : add kv tree tip anchor and prefix restore
```
Message body: `Assisted-by: opencode`.

## Self-review findings

- Diff vs brief: verbatim; no edits under `src/`; working tree clean after commit.
- ASCII-only verified (0 bytes > 127 in all three files); `git diff --check` clean.
- Comments: only the brief's two comments added (tip anchor; blocks covering [0, C)); no new comments invented.
- Note (from the brief, kept as-is): the full-`park` replacement does not set `nb.bytes = nb.data.size()` for new blocks (Task 1's park did). `kv_tree_block::bytes` therefore stays 0 for RAM blocks, visible in `dump()` output (`ram 0 bytes` per block). Byte accounting through `st.bytes_ram` is unaffected, and `restore` uses `b.data.size()` directly, so behavior is correct; the field is presumably to be populated by the Task 5 SSD tier. Flagging only; no change made.
- `capture_partial` is defined per the brief but not yet called (helper set for later tasks); build shows no warning.
- The brief's deferred partial-tail refcount/heat behavior is unchanged, as instructed.

## Issues or concerns

- None blocking. The only observation is the `kv_tree_block::bytes == 0` note above, which is brief-specified behavior and harmless for this task.

---

# Fix Report: track block payload bytes (controller ruling)

## What changed

Plan defect confirmed: the Task 2 full-`park` replacement dropped
`nb.bytes = nb.data.size();` that Task 1's park had. Spec 5.1 and Tasks 5/6
(`remove_block`, `demote_*`, `settle`) account payload sizes via
`kv_tree_block::bytes`, so the field must be populated.

Edit (one line) in `tools/server/server-kv-tree.cpp`, in the park block commit
loop, right after `nb.data = std::move(blobs[k].second);`:

```cpp
nb.bytes     = nb.data.size();
```

`store_anchor` was re-checked: `a.bytes = size;` is already present (line 156),
no change needed there.

## Verification

Rebuild:

```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
```

Success; `test-t32-tree.exe` linked.

Logic mode:

```
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
```

All 8 assertions PASS, exit 0. `dump()` now prints per-block payload bytes:
4096 for each full 512-token fake chunk, 800 for the 100-token tail
(previously all `0 bytes`); `blocks: ... 17376 bytes ram` unchanged and
consistent.

Model mode:

```
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```

All 9 assertions PASS (`tip: park`, `tip: one tip anchor`,
`tip: baseline generation`, `tip: restore point is the tip`,
`tip: no heal needed`, `tip: tokens match the baseline`,
`tip: park refused when the sequence ends past L`, `tip: refusal counted`
and the implicit decode success), exit 0. `dump()` prints each block at
6303912 bytes; total `bytes_ram` 39113796 = 3 x 6303912 blocks + 20202060
anchor payload, consistent.

## Commit

New commit (no amend), on `t32-stage2`, nothing pushed:

```
7b59bb340 server : track block payload bytes in kv tree park
```

Message body: `Assisted-by: opencode`. Diff: 1 file, +1 line.

Constraints: ASCII only, no `src/` edits, no push, no subagents.

---

# Fix Report: report draft state failures (review round 1)

## What changed

Two review findings, both plan-mandated: the draft (`io_dft`) paths in `park`
and `restore` ignored adapter failures, silently storing/using an empty draft
blob and violating the "degradations/refusals never silent" constraint
(dormant in stage 2 because `io_dft` is null on every exercised path).

`tools/server/server-kv-tree.cpp`:

1. In `park`, the draft capture now checks the result, warns, counts the
   refusal and bails out (before `make_room`, so nothing is committed):

```cpp
    if (io_dft != nullptr && !io_dft->get_partial(part_dft)) {
        fprintf(stderr, "[kv-tree] park refused: failed to capture the tip draft state\n");
        st.park_refused++;
        return false;
    }
```

2. In `restore`, the draft load failure now warns and clears `ok`, so the
   restore is rolled back (sequence cleared, counted as miss) instead of
   silently counting as a hit:

```cpp
            } else if (io_dft != nullptr && !a.data_dft.empty()) {
                if (!io_dft->set_partial(a.data_dft.data(), a.data_dft.size())) {
                    fprintf(stderr, "[kv-tree] restore failed: cannot load the draft state at %d\n", C);
                    ok = false;
                }
            }
```

Diff: 1 file, +8/-3. Behavior unchanged when `io_dft == nullptr`.

## Verification

Rebuild:

```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
```

Success; `test-t32-tree.exe` linked.

Logic mode:

```
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
```

All 8 assertions PASS, exit 0 (unchanged output; block bytes still 4096/4096/4096/4096/800).

Model tip scenario:

```
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```

All 9 assertions PASS, exit 0, including `tip: tokens match the baseline`
(bit-exact vs baseline). No new stderr warnings: with `io_dft == nullptr` both
new branches are skipped, as intended.

## Commit

New commit (no amend), on `t32-stage2`, nothing pushed:

```
a2ebafb65 server : report draft state failures in kv tree park and restore
```

Message body: `Assisted-by: opencode`.

Constraints: ASCII only, no `src/` edits, no push, no subagents.
