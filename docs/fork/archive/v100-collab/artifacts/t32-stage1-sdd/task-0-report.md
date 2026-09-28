# Task 0 Report: H2D bench harness + first engine measurement

Status: DONE_WITH_CONCERNS (functionally complete; one environment deviation, see Concerns)

## What I implemented

1. `tests/test-t32-range.cpp` (new, 137 lines): h2d-mode-only harness, copied verbatim from the
   brief's code block (verified identical via Compare-Object against the fenced block in
   `task-0-brief.md`). Provides `fill_context()` and `run_h2d()`; later tasks extend this file.
2. `tests/CMakeLists.txt`: registered `llama_build(test-t32-range.cpp)` after line 320
   (`set_tests_properties(test-state-restore-fragmented ...)`), keeping a blank line on each side.
   Diff is +2 lines (the `llama_build` line plus one blank separator).
3. `<TEMP>\v100\build_test_t32.cmd`: created exactly as the brief
   specifies (preferred temp path was writable).

No other repo file was touched. Step 7 (commit) intentionally skipped per dispatch instructions;
changes left uncommitted.

## Exact commands run and results

### Environment deviation found before building

`git status` clean at HEAD `ba41cccec` (branch `t32-stage1`). The shared build tree
`build\CMakeCache.txt` had `LLAMA_BUILD_TESTS:BOOL=OFF`, so `cmake --build build --target
test-t32-range` failed with `ninja: error: unknown target 'test-t32-range'` even after an
automatic reconfigure (Ninja Multi-Config resolves targets before manifest regen).

Workaround (one-time): `cmake -S . -B build -DLLAMA_BUILD_TESTS=ON` (configure OK, 2.1s),
build the target, then restore the original setting with `-DLLAMA_BUILD_TESTS=OFF` (configure OK).
End state: cache back to its original `LLAMA_BUILD_TESTS=OFF`, binary still present.

### Build

```
cmd /c "<TEMP>\v100\build_test_t32.cmd test-t32-range"
```

Result: `[36/36] Linking CXX executable bin\Release\test-t32-range.exe` - success
(test object compiled at step 28 with default warnings only). Note: flipping tests ON also caused
ggml/llama DLLs to relink with identical config; no source change.

### 27B h2d bench

```
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe `
    -m <models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf `
    -ngl 99 -fa on -np 2 -c 65536 -b 512 -ub 512 --mode h2d --n 32000
```

Exit code 0. Full stderr (stdout empty; 15 unused-tensor warnings omitted here but included in the
archive; first line was wrapped by PowerShell 2> as an ErrorRecord, content unchanged):

```
0.01.020.307 W model has unused tensor blk.64.attn_norm.weight (size = 20480 bytes) -- ignoring
... (15 blk.64 warnings total) ...
0.18.039.467 I cmn          init: llama threadpool init, n_threads = 8
[t32-h2d] fill: 32000 tokens in 40.2 s (796.0 t/s)
[t32-h2d] size: 2150.4 MiB
[t32-h2d] D2H:  0.81 s (2.60 GiB/s) [2254815072 bytes]
[t32-h2d] H2D:  0.89 s (2.37 GiB/s) [2254815072 bytes]
```

Sanity: size 2254815072 bytes = 2.100 GiB, 70463 B/token; D2H/H2D sizes equal the reported state
size and exit code is 0. Bandwidth 2.60/2.37 GiB/s, well above 1 GiB/s.

Control run (smaller model, raw cmd redirection to rule out capture loss; same behavior):
`Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 0 ... --n 64` -> exit 0, size 20.0 MiB, D2H 11.96 GiB/s,
H2D 11.31 GiB/s. Also confirmed stock `llama-cli.exe` from the same build prints no model-load
INFO logs either, so the short stderr is normal for this fork's default verbosity, not data loss.
Logs: `temp\v100\t32_small_h2d.{out,err}`, `temp\v100\t32_cli_ctrl.{out,err}`,
raw 27B capture `temp\v100\t32_stage0_h2d.err` (UTF-16LE), UTF-8 copy `t32_stage0_h2d.txt`.

## Files changed (uncommitted, repo)

- `tests/test-t32-range.cpp` - new (untracked)
- `tests/CMakeLists.txt` - modified, +2 lines (`llama_build(test-t32-range.cpp)` + blank line)
- `git diff --stat`: `tests/CMakeLists.txt | 2 ++` (plus the untracked new file)

No commits, no pushes. Build artifacts under `build\` changed as a side effect of building
(relinked DLLs, new test exe/obj); `build\bin\Release\test-t32-range.exe` exists.

## Channel archival

Succeeded:
- `D:\LLM\Backend\v100-collab\artifacts\t32-stage0-h2d.txt` created (command line, full stderr,
  4 numbers, 100K extrapolation, notes).
- `TASKS\T32-agent-session-reuse.md`: appended `## Result (stage 0)` section.
- Per the brief's Step 6 parenthetical, also replaced the spec appendix H2D row in
  `artifacts\t32-tree-storage-design.md` line 172: `~2s/GB` -> measured `2.37 GiB/s H2D /
  2.60 GiB/s D2H`, 100K extrapolation.

## Self-review findings and concerns

1. Concerns - build env deviation: brief assumed tests were configured in the shared build tree;
   it was `LLAMA_BUILD_TESTS=OFF`. I flipped it ON for the build and restored OFF afterwards.
   Consequence: any later rebuild of `test-t32-range` needs the same one-time flip (or the
   controller may choose to leave tests ON in this tree). Documented in the archive too.
2. Numbers vs brief's expectation: measured 32K state 2150.4 MiB (2.100 GiB, 70463 B/token) is
   ~10-16% above the brief's rough `~1860 MiB` / `1950000000 bytes` and slightly above the
   dispatch sanity band (1.8-2.0 GiB). The harness is internally consistent (get/set both return
   exactly the state size, exit 0), so this is a real model/config size, not a harness bug.
   100K extrapolation from measurement: 6.56 GiB -> H2D ~2.8 s / D2H ~2.5 s (brief's 5.0 GiB
   basis would give ~2.1 s / ~1.9 s). TASKS/spec notes record this deviation.
3. CMake registration added 2 lines (target + separator blank), not literally one line; matches
   surrounding style.
4. Bandwidth on this PCIe x4 slot is ~2.4-2.6 GiB/s, ~5x faster than the old conservative
   `~2s/GB`; still far below small-transfer rate (11 GiB/s at 20 MiB), consistent with many
   scattered per-tensor copies rather than one contiguous transfer.
5. No lint/typecheck available for this harness; verification was build + run + output check.

## Commit

User-approved commit: `82674b9f5` - "t32 : add range state API bench harness"
(2 files changed, 139 insertions: new `tests/test-t32-range.cpp`, `tests/CMakeLists.txt` +2).
Not pushed. `git status --short` clean after commit.
