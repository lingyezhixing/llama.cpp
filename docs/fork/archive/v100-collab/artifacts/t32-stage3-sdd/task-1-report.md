# Task 1 report: tree module increments (drop_seq / streaming restore + bytes_load / debug / ctor validation)

Commit: `421dd61b6` server : stream disk payloads during kv tree restore (branch `t32-stage3`)
Base: `4f2631088`

## What was implemented

Files: `tools/server/server-kv-tree.h`, `tools/server/server-kv-tree.cpp`, `tests/test-t32-tree.cpp`.

- `tests/test-t32-tree.cpp`: added `run_logic_drop()` and `run_logic_stream()` verbatim from the brief; wired both into `run_logic()` after `run_logic_evict()`.
- `server-kv-tree.h`: public `bool drop_seq(const std::vector<llama_token> &)`; private `bool set_block_payload(kv_tree_io &, kv_tree_block &, bool, std::vector<uint8_t> &)`.
- `server-kv-tree.cpp`:
  - `drop_seq()`: matches the request tokens against the stored blocks, resolves the tip hash (partial-block match first, else last fully matched block), and calls `remove_seq()`; no-op when nothing matches or the sequence is not stored.
  - `set_block_payload()`: disk blocks are streamed through a scratch buffer when they do not fit the RAM budget (`st.bytes_ram + b.bytes > cfg.ram_limit`), keeping them on disk; otherwise `load_payload()` + `set_range()` as before.
  - `restore()`: block loading now goes through `set_block_payload()`; the old `st.bytes_load` line in the loop is removed (no double count).
  - `bytes_load` accounting: added in `load_payload(kv_tree_block &)` (block moved disk -> RAM) and `load_payload(kv_tree_anchor &)` (anchor moved disk -> RAM, the deferred "bytes_load undercount" item).
  - `park()` and successful `restore()` end with `if (cfg.debug) { dump(); }`.
  - Constructor validates `cfg.chunk > 0` and `cfg.anchor_step >= 0`, logs `[kv-tree] invalid config: ...` and `GGML_ABORT`s otherwise.
- One deviation from the brief's step list (forced, see Self-review): `write_disk()` now creates its parent directory on demand. The brief's `run_logic_stream()` only creates the top-level temp dir, while the module writes into `disk_dir/blocks/` and `disk_dir/anchors/`; without this the brief's exact test can never pass. This also matters for Task 2/5: the plan's server code and scripts only create the top-level `--tree-disk` dir.

## What was tested and results

1. Logic: `build\bin\Release\test-t32-tree.exe --mode logic` -> all checks PASS, exit 0.
   - Pre-existing: eviction 10 + base logic 8 = 18 PASS.
   - New: drop_seq 9 PASS, streaming restore 10 PASS (37 checks total).
   - New assertions include `bytes_load == 3 * 4096` (blocks moved from disk; tip anchor stays RAM-resident in that scenario) and `blocks_disk == 3` after restore (streamed blocks stay on disk).
   - Temp dir `%TEMP%\t32-tree-logic-stream` removed after the run (verified `Test-Path` = False).
2. Model regression (CUDA_VISIBLE_DEVICES=0):
   `test-t32-tree.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096`
   -> 43/43 PASS (tip 8, fork 13, sparsify 7, ssd 8, unaligned 7), exit 0. The ssd scenario exercises the new streaming path on real 6.3 MB blocks (ram_limit 64 KiB): restore streams them, they remain on disk, tokens match the baseline.

## TDD evidence

RED (step 2, before implementation):
Command: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree`
Output:
```
[1/3] Building CXX object tests\CMakeFiles\test-t32-tree.dir\Release\test-t32-tree.cpp.obj
FAILED: [code=2] tests/CMakeFiles/test-t32-tree.dir/Release/test-t32-tree.cpp.obj
tests/test-t32-tree.cpp(166): error C2039: "drop_seq": 不是 "kv_tree" 的成员
...
ninja: build stopped: subcommand failed.
```
Expected: compile failure because `drop_seq` did not exist yet.

GREEN:
Command: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree` then `build\bin\Release\test-t32-tree.exe --mode logic`
Output (after the module implementation): build links cleanly; every `[t32-tree]` line is `PASS`; `EXIT=0`.
Command (model): see above; `EXIT=0`, 43 PASS lines.

An intermediate GREEN attempt exposed the `write_disk` gap (`failed to write block ... to disk`, park refused, 10 FAIL in the stream test); the parent-dir creation fixed it and the exact brief test passed unchanged.

## Files changed

```
 tests/test-t32-tree.cpp         | 84 +++++++++++++++++++++++++++++++++++++++++
 tools/server/server-kv-tree.cpp | 80 +++++++++++++++++++++++++++++++++++++--
 tools/server/server-kv-tree.h   |  4 ++
 3 files changed, 165 insertions(+), 3 deletions(-)
```

## Self-review findings

- Brief code blocks for tests and module are reproduced verbatim (values, names, log strings); only `write_disk` differs (documented above).
- No non-ASCII characters added; comments are ASCII and concise.
- `bytes_load` is counted exactly once per disk -> RAM move: block via `load_payload(block)` or scratch stream, anchor via `load_payload(anchor)`; the old count in the restore loop was removed.
- Scratch is reused across blocks in restore; no transient RAM is allocated for streamed blocks.
- No other callers of `load_payload(block)` exist in the module, so the new counting cannot double-count there.
- `cfg.debug` dumps only on full park success / successful restore, matching the brief's placement.
- No unrelated edits, no renames, no new files.

## Concerns

1. `write_disk()` parent-dir creation is not in the brief's step list (ruling R8 in the plan's preflight ledger assumed the test's `create_directories(dir)` was sufficient, but it only creates the top-level dir). I kept the brief's test unchanged and made the module create `blocks/`/`anchors/` on demand. Task 2/5 need exactly this behavior or the server SSD tier cannot write. Flagging for the controller in case they prefer the alternative (test-side subdir creation).
2. The plan says "原 18 项 + 新 14 项 = 32"; the actual check call count is 18 + 19 = 37 (drop 9, stream 10). No functional impact; the plan's arithmetic appears off.
