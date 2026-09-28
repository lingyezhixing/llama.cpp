# Task 1 report: 分叉锚点不被猜测压制 (D31)

Status: DONE_WITH_CONCERNS (plan's verbatim test values were broken; adapted minimally, see Concerns)

## Summary

- `capture_anchor`'s `prev` scan now skips `KV_TREE_ANCHOR_MESSAGE` anchors. Checkpoint guesses never suppress a fork (ONDEMAND) capture; fork-vs-fork spacing still applies at `cfg.fork_step`, and `park` candidate adoption still respects `anchor_step` against all anchors (unchanged).
- One commit on `t32-stage5b`: `8184bdc7c server : do not let checkpoint guesses suppress fork anchors` (`Assisted-by: opencode`). Only `tools/server/server-kv-tree.cpp` and `tests/test-t32-tree.cpp` are in the commit.
- Artifact `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` heal mode no longer passes `-fork_step 512`; comment updated. Artifact is outside the repo and is intentionally not committed (per brief).
- Acceptance: logic 80 PASS / 0 FAIL, heal `RESULT heal: 0 failure(s)` with `REBUILT lines=1`, model 65 PASS / 0 FAIL, fork `RESULT fork: 0 failure(s)`.

## Files changed

- `tools/server/server-kv-tree.cpp` (capture_anchor, lines 378-380): added
  ```cpp
  if (kv.second.kind == KV_TREE_ANCHOR_MESSAGE) {
      continue;   // guesses never suppress a fork anchor
  }
  ```
- `tests/test-t32-tree.cpp` (`run_logic_fork`): new "guesses never suppress a fork anchor" block, 5 checks.
- `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` (line 266-267): heal uses default fork_step.
- Server DLL rebuilt (`llama-server-impl.dll` deleted, then `build_server.cmd`), exit 0.

## TDD evidence

Build commands: `<TEMP>\v100\build_test_t32.cmd test-t32-tree`, run `test-t32-tree.exe --mode logic` (CUDA_VISIBLE_DEVICES=1).

Baseline before test: 75 PASS / 0 FAIL.

1. RED with brief's verbatim test values (`tok=4096`, park `max_pos=4095`, captures 5120 / 13312):
   ```
   [kv-tree] capture refused at 5120: no chain block contains this position
   [t32-tree] fork: the guess does not suppress the fork      FAIL
   [t32-tree] fork: the fork-vs-fork skip is counted          FAIL (got 0, want 1)
   exit=1
   ```
2. After implementing the fix, verbatim values still failed the same 2 checks (same "no chain block contains this position" refusals), proving the problem is the test values, not the fix:
   ```
   [t32-tree] fork: the guess does not suppress the fork      FAIL
   [t32-tree] fork: the fork-vs-fork skip is counted          FAIL (got 0, want 1)
   exit=1
   ```
3. Test values adapted minimally (see Concerns): `make_tokens(10240, 0)`, park `max_pos=10239`, last capture `9216` (`max_pos=9215`). Everything else verbatim (cfg: chunk 512 / anchor_step 32768 / fork_step 8192 / ram 1 MiB; guess pos 1024; capture 5120; all check texts; 5 checks).
   RED (fix temporarily removed and rebuilt):
   ```
   [t32-tree] fork: the guess does not suppress the fork      FAIL
   [t32-tree] fork: no spacing skip against the guess         FAIL (got 1, want 0)
   [t32-tree] fork: another fork anchor still suppresses (8192)  FAIL
   exit=1
   ```
4. GREEN (fix restored):
   ```
   PASS=80 FAIL=0, exit=0
   [t32-tree] fork: park with a guess at 1024                 PASS
   [t32-tree] fork: the guess does not suppress the fork      PASS
   [t32-tree] fork: no spacing skip against the guess         PASS (got 0, want 0)
   [t32-tree] fork: another fork anchor still suppresses (8192)  PASS
   [t32-tree] fork: the fork-vs-fork skip is counted          PASS (got 1, want 1)
   ```

## Acceptance outputs

Heal (artifact with default fork_step 8192), saved to `t32-stage5b-heal.txt`:
```
HEAL ANCHOR pos=1024 stored_in_dump=True
HEAL METRICS captured=1 stored=True notstored=0 failed=0 missed=0 parked=7 restored=2
HEAL REBUILT lines=1
RESULT heal: 0 failure(s)
```
This is the D31 proof end-to-end: with fork_step back at its default 8192, the guess at 487 no longer suppresses the fork (heal) capture at 1024, and the checkpoint rebuild happens (REBUILT >= 1).

Model (`--mode model --ram-mib 4096 -c 8192`, saved to `t32-stage5b-model.txt`): 65 PASS / 0 FAIL, exit 0.

Fork (saved to `t32-stage5b-fork.txt`): 5/5 tree-vs-full hashes identical, `RESULT fork: 0 failure(s)`.

Note: heal/model/fork logs were written with `*>` redirection instead of `Tee-Object`; file contents are equivalent.

## Self-review

- Fix is scoped to `capture_anchor` only. `park`'s candidate `prev` scan (uses `h`) is untouched, so forks still suppress later guesses via `anchor_step` as decided in D31.
- No stats, log strings, or knobs changed. No new files in the repo. Comments are ASCII and one line.
- No existing assertion was weakened; the new block adds 5 checks (baseline 75 -> 80).
- Verified via `git diff` that only the intended 2 files changed; working tree clean after commit; branch is `t32-stage5b`, `git push` not run.

## Concerns

1. The plan/brief test values cannot pass and were adapted. With `tok=4096` there is no stored block covering 5120 or 13312, so `capture_anchor` refuses with "no chain block contains this position" both before and after the fix (verified post-fix with verbatim values). Additionally the tip anchor at 4096 would itself suppress a fork at 5120 (distance 1024 < fork_step), and 13312 - 5120 = 8192 is not less than fork_step, so the expected refusal at 13312 is unreachable (the existing test asserts "at fork_step it is stored" for exactly this boundary). Adapted: sequence 10240 tokens, park `max_pos=10239`, last capture 9216 so a real fork-vs-fork (ONDEMAND@5120, distance 4096) spacing skip is exercised. All assertions and messages from the brief were kept.
2. Check count is 80, not the brief's 78 (75 + 3). The brief's own block contains 5 checks (3 `check` + 2 `check_eq`); baseline was 75. Acceptance is 0 FAIL.
3. The artifact script lives outside the repo, so its heal-mode change is not part of the commit (as the brief's commit command implies). The updated script was used for the heal/fork acceptance runs.
