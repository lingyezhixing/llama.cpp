# Task 2 Report: fork anchor spacing + miss heal + counters + dead code removal (D25/D26/D27/D28)

Status: DONE_WITH_CONCERNS (one non-blocking doc discrepancy: check count)

## Commit

- `9e7784270` `server : add fork anchor spacing and miss heal to the kv tree`
  (trailer `Assisted-by: opencode`), branch `t32-stage5`, base `ad81d0efc`.
- Files in commit: `tools/server/server-kv-tree.h`, `tools/server/server-kv-tree.cpp`, `tests/test-t32-tree.cpp`.

## TDD evidence

### RED (Step 1+2)

Test first: added `run_logic_fork` verbatim from the brief after `run_logic_fixes`,
and `run_logic_fork();` at the end of `run_logic()` after `run_logic_wipe()`.

Build command:
```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
```
Failing output (compile error, no binary produced):
```
[4/13] Building CXX object tests\CMakeFiles\test-t32-tree.dir\Release\test-t32-tree.cpp.obj
FAILED: [code=2] tests/CMakeFiles/test-t32-tree.dir/Release/test-t32-tree.cpp.obj
D:\LLM\Backend\src\llama.cpp-my\tests\test-t32-tree.cpp(425): error C2039: "anchors_skipped_step": is not a member of "kv_tree_stats"
D:\LLM\Backend\src\llama.cpp-my\tools\server\server-kv-tree.h(37): note: see declaration of "kv_tree_stats"
ninja: build stopped: subcommand failed.
```
(Message rendered in the console's Chinese locale; text above transcribed.)
This is exactly the expected RED: `anchors_skipped_step` absent, and `r.heal` is not
yet implemented.

### GREEN (Step 4)

Build command: same as above. Result:
```
[5/5] Linking CXX executable bin\Release\test-t32-tree.exe
```

Run command:
```
$env:CUDA_VISIBLE_DEVICES='1'; & 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic
```
Result: `74` check lines, `0` FAIL. All new checks:
```
[t32-tree] fork: park the chain                                                     PASS
[t32-tree] fork: the first capture is free                                          PASS
[t32-tree] fork: within fork_step is refused                                        PASS
[t32-tree] fork: the spacing skip is counted                                        PASS (got 1, want 1)
[t32-tree] fork: at fork_step it is stored                                          PASS
[t32-tree] fork: tip + two fork anchors                                             PASS (got 3, want 3)
[t32-tree] fork: stats line has step_skips                                          PASS
[t32-tree] fork: park A                                                             PASS
[t32-tree] fork: B misses (no anchor on the shared path)                            PASS (got -1, want -1)
[t32-tree] fork: the miss reports the divergence point                              PASS (got 2048, want 2048)
```

## Changes (all verbatim from the brief)

| File:line (post-commit) | Change |
|---|---|
| `tools/server/server-kv-tree.h:52` | `int64_t anchors_skipped_step = 0;` after `anchors_skipped` |
| `tools/server/server-kv-tree.h:189` | `promote_prune` declaration removed |
| `server-kv-tree.cpp:383-388` | capture spacing now `cfg.fork_step`, log `within fork_step`, increments `anchors_skipped_step` |
| `server-kv-tree.cpp:413-415` | 2-line comment replacing the `promote_prune` call (D27 rationale) |
| `server-kv-tree.cpp:1123-1167` | deleted `promote_prune` definition (29 lines) |
| `server-kv-tree.cpp:1161-1165` | park candidate spacing keeps `cfg.anchor_step`, only adds `anchors_skipped_step++` |
| `server-kv-tree.cpp:1300-1304` | restore miss: `res.heal = m.deep;` before return |
| `server-kv-tree.cpp:1424,1430` | `stats_line()` gains `step_skips=%" PRId64` after `skipped=` |
| `tests/test-t32-tree.cpp:396-465,518` | `run_logic_fork` + registration |

No other log wording changed. `park`'s candidate threshold is still `cfg.anchor_step`.
`git grep promote_prune` returns nothing.

## Self-review findings

- Fidelity: diff re-read line by line against the brief; all snippets match verbatim
  (values, wording, comment text). ASCII only.
- Heal semantics checked against the consumer: `tools/server/server-context.cpp:411`
  computes `tree_heal = res.heal > res.C ? res.heal : -1`. On a miss now `C=-1`,
  `heal=deep>0`, so `tree_heal=deep`, as Task 3/4 expect.
- D27 reasoning verified against the test outcome: after spacing, `capture_anchor`
  never stores an anchor with `pos - prev < fork_step`, so the old prune window
  `(prev, pos)` is empty; removing `promote_prune` changed no observable behavior
  (all pre-existing 64 logic checks still pass).
- `anchors_skipped_step` increments only at the two D25/D26 sites: capture spacing
  (fork_step) and park candidate adoption (anchor_step). Other `anchors_skipped++`
  sites are unchanged, as specified.
- Working tree clean after commit; only the three intended files in the commit.

## Concerns

- Brief/plan arithmetic mismatch (non-blocking, no code impact): the brief predicts
  `64 + 7 = 71` logic checks, but its `run_logic_fork` contains 10 checks. Verified:
  new checks = 10 (regex count of `check(`/`check_eq(` in the function), and the
  current-source baseline is 64 (stale prebuilt exe in `build/` showed 68 and does
  not match the current test source). Actual final: 74 checks, 0 FAIL. Plan text
  should read 64 + 10 = 74.
- Only `test-t32-tree` target was built (per task constraints); the additive header
  change does not touch `server-context.cpp` call sites, but the full `llama-server`
  target was not recompiled in this task.

## Exact commands run

```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='1'
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : add fork anchor spacing and miss heal to the kv tree" -m "Assisted-by: opencode"
```
