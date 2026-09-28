# Task 2 Report: D13 startup wipe

## Status: DONE

## What was implemented

1. `tests/test-t32-tree.cpp`: added `run_logic_wipe()` (exactly as specified in the brief) and registered it in `run_logic()` after `run_logic_fixes()`.
   - Creates `%TEMP%\t32-tree-wipe` with `blocks/` and `anchors/` subdirs and 3 stale files (`aa.bin`, `bb.bin.tmp`, `cc.bin`).
   - Constructs `kv_tree` with `cfg.disk_dir` set and asserts all 3 files are gone.
   - Constructs `kv_tree` with a default (empty) `cfg.disk_dir` to prove no wipe work is attempted.
2. `tools/server/server-kv-tree.cpp:188-204` (ctor): when `cfg.disk_dir` is non-empty:
   - counts regular files recursively under `disk_dir` into `n_stale` (`std::error_code`, no-throw),
   - `remove_all` on `<disk_dir>/blocks` and `<disk_dir>/anchors`,
   - logs `[kv-tree] cleared %zu stale files from %s`.
   - The `chunk/anchor_step` validation block is unchanged. Task 1 code (`stats_line()`, counters) untouched.

## TDD evidence

### RED

Build: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree` -> exit 0.
Run: `& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic`

```
pass=61 fail=3 exit=1
[t32-tree] wipe: constructor with a stale disk dir                                  PASS
[t32-tree] wipe: block file removed                                                 FAIL
[t32-tree] wipe: tmp file removed                                                   FAIL
[t32-tree] wipe: anchor file removed                                                FAIL
[t32-tree] wipe: constructor without a disk dir                                     PASS
```

Full log saved at `<user>\AppData\Local\Temp\opencode\t32-task2-red.txt`.

### GREEN

Rebuilt with the same command (exit 0), reran `--mode logic`:

```
pass=64 fail=0 exit=0
[kv-tree] cleared 0 stale files from <user>\AppData\Local\Temp\t32-tree-logic-stream
[kv-tree] cleared 0 stale files from <user>\AppData\Local\Temp\t32-tree-diskerr
[kv-tree] cleared 3 stale files from <user>\AppData\Local\Temp\t32-tree-wipe
[t32-tree] wipe: constructor with a stale disk dir                                  PASS
[t32-tree] wipe: block file removed                                                 PASS
[t32-tree] wipe: tmp file removed                                                   PASS
[t32-tree] wipe: anchor file removed                                                PASS
[t32-tree] wipe: constructor without a disk dir                                     PASS
```

64 checks total (59 pre-existing + 5 new), 0 FAIL, exit 0.
Full log saved at `<user>\AppData\Local\Temp\opencode\t32-task2-green.txt`.

Note: the brief's Step 4 expected "61 checks" but the task context expects 64 (59 + 5); the measured result is 64. The "61" in the brief is stale/inconsistent, the 5 added checks are all present and passing.

## Server smoke test (cuda1, port 8937)

Rebuilt `llama-server-impl.dll` (deleted first) with `& '<TEMP>\v100\build_server.cmd` (exit 0; UI download timeout warnings are pre-existing env noise). Ran the brief's smoke snippet verbatim:

```
cleared_lines=1 files_left=0
```

Server log line:

```
[kv-tree] cleared 3 stale files from <TEMP>\v100\t32-stage4-wipe
```

`3` = the 2 planted stale files (`blocks/aa.bin`, `anchors/cc.bin`) plus `srv.txt`, the stderr redirect file that the smoke harness itself creates inside `$d` before the server starts. `remove_all` only targets `blocks/`/`anchors/`, so `srv.txt` survives (hence it is excluded from `files_left`). No `llama-server` process or listener left on port 8937 after the run.

## Files changed

- `tools/server/server-kv-tree.cpp` (+17)
- `tests/test-t32-tree.cpp` (+44)

Commit: `ce03a50f8 server : clear stale kv tree files on startup` (`Assisted-by: opencode`). Working tree clean after commit.

## Self-review findings

- Diff matches the brief's Step 1 and Step 3 code verbatim, aside from placement of `run_logic_wipe` (immediately before `run_logic`).
- Existing `fixes` test with `disk_dir` set to a *file* still passes: `recursive_directory_iterator` over a file with `error_code` yields an empty range and logs `cleared 0`; `remove_all` on `<file>/blocks` fails silently (ec) and does not delete the file. Verified in GREEN output.
- Empty `disk_dir` skips the wipe block entirely (no filesystem calls), verified by the fifth check.
- No throws: both iterator construction and `remove_all` use `std::error_code` overloads. The range-for's `operator++` is the non-ec overload, but at startup the directory is static and single-threaded, so no concurrent mutation.
- ASCII only, one concise comment (the brief's), matches surrounding style.
- Task 1 counters/`stats_line()` untouched.

## Concerns

- `n_stale` counts all regular files under `disk_dir`, not only `blocks/` + `anchors/` (per the brief). Any unrelated file a user puts in the per-server exclusive dir inflates the log count, while only `blocks/` and `anchors/` are deleted. Cosmetic only; the dir is documented as per-server exclusive.
- Fixed temp path `t32-tree-wipe` follows the existing test pattern (`t32-tree-diskerr`); concurrent test binaries in logic mode could race on it. Pre-existing convention, not introduced here.
- Brief's "61 checks" figure is stale; actual is 64 per the task context.
