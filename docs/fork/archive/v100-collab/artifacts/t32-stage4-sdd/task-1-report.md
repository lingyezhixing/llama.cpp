# Task 1 report: module fixes (D22/D23) + stats line

Status: DONE

Commit: 98c613325 `server : fix kv tree disk error counting, anchor spacing and add stats line` (single commit, as directed by the controller; trailer `Assisted-by: opencode`). Branch `t32-stage4`, base `daf4186d3`. Not pushed.

## What was implemented

1. `tools/server/server-kv-tree.h`: declared `std::string stats_line() const;` after `stats()`.
2. `tools/server/server-kv-tree.cpp`:
   - `write_disk`: removed the two `st.disk_errors++` increments (create-directories failure, fopen failure). Callers `demote_block` / `demote_anchor` already increment exactly once, so each failed write now counts one error instead of two.
   - `capture_anchor`: `prev` (anchor spacing) is now scoped to the request chain (`std::find(chain...)`), same pattern as `promote_prune`; anchors from other chains no longer suppress a capture.
   - `park`: checkpoint-candidate adoption `prev` is now scoped to the parked chain `h`; a cross-chain anchor no longer suppresses a candidate.
   - `stats_line()`: one-line summary with the exact fields from the brief (`parks= ok= refused= restore= hits= miss= anchors= skipped= reuse_tok= store= load= evicted=a/b/c evict_refused= disk_err= ram= disk=`). Task 4/5 log parsing depends on these names.
3. `tests/test-t32-tree.cpp`: added `run_logic_fixes()` (verbatim from the brief) after `run_logic_capture()`, registered in `run_logic()`.

## TDD evidence

### RED

Step 1: added the test only, then built:
`& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree`
- `tests/test-t32-tree.cpp(389): error C2039: "stats_line": 不是 "kv_tree" 的成员` (not a member of kv_tree).

Step 2 (behavioral RED): added a temporary empty `stats_line` stub so the test compiles, rebuilt, then ran
`& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic`:
- `fixes: one disk error per failed write (block + anchor)` FAIL (got 4, want 2)
- `fixes: the cross-chain anchor does not suppress the candidate` FAIL (got 2, want 3)
- `fixes: cross-chain anchor does not block the capture` FAIL
- `fixes: stats line has parks` FAIL
- `fixes: stats line has anchors` FAIL
- `fixes: stats line has disk_err` FAIL

6 FAILs, all in the new `fixes` block; the stub was then replaced by the real implementation. This matches the brief's predicted failure modes.

### GREEN

`& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree` -> builds clean (no warnings).
`& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic` -> `exit=0 pass=59 fail=0` (case-sensitive count of PASS/FAIL assertion lines). The `fixes` block shows 13/13 PASS.

Note: the brief says "46 + 12 = 58"; the test code given in the brief contains 13 `check`/`check_eq` calls (2 + 3 + 4 + 4), and the pre-existing logic-mode count is 46, so the actual total is 59. Arithmetic slip in the brief, not a test problem.

## Files changed

- `tools/server/server-kv-tree.h` (+2)
- `tools/server/server-kv-tree.cpp` (+32/-4)
- `tests/test-t32-tree.cpp` (+116)

## Self-review findings

- `write_disk` has exactly two call sites (`demote_block`, `demote_anchor`), both increment `disk_errors` on `false`; no failed write can now go uncounted and none is double counted. Other write failure points inside `write_disk` (fwrite/rename) never counted there before, behavior unchanged.
- The chain-scoped `prev` loops follow the existing `promote_prune` pattern; `capture_anchor` already builds the token-hash `chain`, and `park` already has `h`.
- `stats_line` is ASCII-only, single snprintf, 512-byte buffer is ample (longest field magnitudes are int64).
- Test 3's second assertion (`fixes: same-chain spacing still refuses`) passes because no stored block contains pos 2560 with 2048 tokens (containment is checked before spacing), not because spacing refuses it. It is verbatim from the brief and does not weaken the fix, but it does not exercise the same-chain spacing path.

## Concerns

- The brief expected a two-commit split (fixes, then stats line); the controller explicitly required one commit, which is what was done.
- Check-count discrepancy: brief says 58, actual is 59 (13 new checks, not 12).
- The same-chain-spacing assertion described above is weak; a future task could capture at e.g. pos 2048+512 on chain B with io ending there so containment succeeds and spacing is actually reached.
- No GPU/model run was performed (out of scope for Task 1; `--mode logic` only).
