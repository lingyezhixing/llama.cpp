# Task 3 Report: server wiring (miss-heal) + model test

Status: DONE_WITH_CONCERNS (all required checks green; 4 non-blocking observations below)

## Commit

- `04954b4689e4455e63c68526fa11ba9e38b4de4f` `server : seed fork anchors when a tree restore misses`
  (trailer `Assisted-by: opencode`), branch `t32-stage5`, base `9e7784270`.
- Files in commit: `tools/server/server-context.cpp`, `tests/test-t32-tree.cpp`. Worktree clean after commit.
- Artifact change outside the repo (not committable, see Deviation 2): `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1`.

## What changed

| File:line (post-commit) | Change |
|---|---|
| `tools/server/server-context.cpp:364` | miss path in `prompt_restore_tree`: `tree_heal = res.heal;` (with the brief's comment) before `return false` |
| `tools/server/server-context.cpp:1814-1816` | caller in `get_available_slot`: snapshot `heal` (only when `cache_prompt`), `prompt_clear()`, then restore `tree_heal = heal` |
| `tests/test-t32-tree.cpp:1040-1086` | new `scenario_fork_miss` (verbatim from the brief) |
| `tests/test-t32-tree.cpp:1242` | model dispatch: `ret |= scenario_fork_miss(ctx, cfg);` |
| `tests/test-t32-tree.cpp:718` | `scenario_fork`: added `cfg.fork_step = 512;` (see Deviation 1) |
| `artifacts\t32-stage3-ab.ps1` (outside repo) | `Start-Srv` gained trailing `$fork_step = 8192` param, passes `--tree-checkpoint-fork-step`; heal mode calls `-fork_step 512` (see Deviation 2) |

Log contract unchanged. `git grep` of the miss wording still shows the single existing
`kv tree: restore miss for %zu tokens, full prefill` line.

## Test evidence

### Model mode (Step 2 of the brief): engine path already green

The harness links `server-kv-tree` only, so the new model test cannot exercise the
server wiring. Per the task note this is recorded honestly: **engine path already green
(Task 2); server wiring is verified by the heal acceptance + Task 4's fork mode.**

First run (before the `scenario_fork` config adaptation below) exposed an interaction the
plan did not foresee: Task 2's D25 spacing (now `fork_step`, default 8192) refused the
pre-existing `scenario_fork` heal capture at 1024 (512 past the anchor at 512):

```
[kv-tree] capture skipped at 1024: within fork_step
[t32-tree] fork: heal capture at the fork point           FAIL
[t32-tree] heal: A' restores at the self-healed anchor    FAIL (got 512, want 1024)
[t32-tree] heal: no second heal needed                    FAIL (got 1024, want -1)
```

These 3 checks are not part of this task's new test, but the binding acceptance requires
0 FAIL, so `scenario_fork` now pins `cfg.fork_step = 512` next to its existing
`cfg.anchor_step = 512`, which reproduces its pre-Task-2 trajectory.

Final run (exact command from the constraints):

```
$env:CUDA_VISIBLE_DEVICES='1'
test-t32-tree.exe -m Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```

Result: exit 0, **64 PASS / 0 FAIL** (plan predicted 56 + 7 = 63; the new scenario has 8
checks, so actual is 56 + 8 = 64; same off-by-one class as Task 2's 71 vs 74).

New checks, all PASS:

```
fork-miss: park A                                     PASS
fork-miss: the first visit misses                     PASS (got -1, want -1)
fork-miss: the miss reports the divergence point      PASS (got 2048, want 2048)
fork-miss: capture at the divergence point            PASS
fork-miss: the second visit restores at the fork      PASS (got 2048, want 2048)
fork-miss: no heal on a full match                    PASS (got -1, want -1)
fork-miss: the tree path restores at the fork         PASS (got 2048, want 2048)
fork-miss: tokens match the baseline                  PASS
```

Additional diagnostic run with `-c 8192` (not the mandated command, extra evidence only):
64 PASS / 0 FAIL, fork-miss all PASS, no `failed to prepare attention ubatches` /
`failed to find a memory slot` errors. See Concern 3 about the 4096 run.

### Logic mode (constraint: unchanged 74/0)

Result: exit 0, 74 PASS / 0 FAIL.

### Server build

Deleted `build\bin\Release\llama-server-impl.dll`, then
`& '<TEMP>\v100\build_server.cmd'`:
`[6/6] Linking CXX executable bin\Release\llama-server.exe`, no errors. (UI asset
download timeout / "building without an embedded UI" is the known offline behavior.)

### Heal acceptance

First run with the unmodified script (default `fork_step = 8192`):

```
HEAL METRICS captured=0 stored=False notstored=0 failed=0 missed=0 parked=7 restored=2
FAIL  heal: exactly one heal capture
FAIL  heal: the captured anchor is genuinely stored (dump shows @pos kind=2)
RESULT heal: 2 failure(s)
```

Server log isolates the cause: `kv tree: restored 487 tokens (heal = 1024)` followed by
`[kv-tree] capture skipped at 1024: within fork_step` (487 + 512 = 1024; 1024 - 487 = 537
< 8192). The scenario was written for stage-3 semantics where `capture_anchor` used
`anchor_step` (512); after Task 2 only `fork_step` governs it. Diagnostic proof with
`$env:LLAMA_ARG_TREE_CHECKPOINT_FORK_STEP='512'` gave `RESULT heal: 0 failure(s)`, so the
gap is purely the script's server configuration.

After the script update (Deviation 2), the exact constraint command:

```
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode heal
```

Result (exit 0):

```
HEAL ANCHOR pos=1024 stored_in_dump=True
HEAL METRICS captured=1 stored=True notstored=0 failed=0 missed=0 parked=7 restored=2
HEAL REBUILT lines=1
PASS  heal: context checkpoints rebuilt after the fork restore (D12)
PASS  heal: exactly one heal capture
PASS  heal: the captured anchor is genuinely stored (dump shows @pos kind=2)
PASS  heal: no failed captures
PASS  heal: request 1..4 identical (tree vs full prefill)
RESULT heal: 0 failure(s)
```

## Deviations from the brief

1. **`scenario_fork` config (one line).** The brief's Step 4 expects 0 FAIL, but Task 2's
   D25 spacing silently broke 3 pre-existing model checks (Task 2 only ran logic mode).
   `cfg.fork_step = 512;` with a one-line comment reproduces the scenario's pre-stage-5
   intent (an explicit heal capture 512 past the park anchor). The scenario itself is not
   weakened; fork spacing semantics stay covered by `run_logic_fork`.
2. **Artifact script update (outside the repo).** Heal mode passes `anchor_step = 512`
   specifically so the heal pos clears the spacing rule (pre-existing comment); under
   stage 5 the same rule is `fork_step`, so heal mode now also passes `-fork_step 512`.
   `Start-Srv`'s new `$fork_step` parameter is appended last with default 8192, so every
   existing positional call (including Task 4's planned fork call at plan line 437) is
   unchanged. Without this, heal mode fails even though the repo code is correct. The
   heal acceptance green above depends on this file; Task 4 must keep `-fork_step 512`
   in heal mode when it edits the script.
3. **Caller snippet form.** The brief inlines the `ckpt_tail` expression; the tree has
   `const bool ckpt_tail = ...` (introduced by stage-4 commit c0619da14). Kept the
   existing local variable (brief line numbers `:1795-1801` also predate that commit).
   Behavior identical: the added lines are exactly the brief's `heal` snapshot/restore.
4. **Check arithmetic.** Plan says 63 (56 + 7); the brief's scenario contains 8 checks,
   actual 64 PASS / 0 FAIL.

## Self-review findings

- Miss path: `res.heal = m.deep` can be 0 when nothing matched; `tree_heal = 0` then
  reaches the capture block at `n_tokens == 0`, where `capture_anchor` refuses `pos <= 0`
  silently and the block clears `tree_heal = -1`. No log noise, no bad state.
- Caller: `prompt_restore_tree` is only called when `cache_prompt` is true (short
  circuit), so on the false branch the snapshot yields `-1` and the previous unconditional
  reset behavior is preserved; `tree_heal` is also reset at slot selection (`:1794`).
- Update-cache gate: non-completion tasks and the no-tree path never reach the new code;
  no behavior change there.
- Log contract: miss wording untouched; heal wording in the acceptance output is the
  existing `captured heal anchor at %d`.
- Style: ASCII only; 2 short comments added (one from the brief, one for the
  non-obvious test spacing); no other file touched; commit contains only the two
  brief-listed files.
- Fidelity: `scenario_fork_miss` and the miss-path line are verbatim from the brief.
- The heal acceptance verifies the success-path heal and the rebuilt server; the miss
  path set by this task is exercised end-to-end by Task 4's fork mode (as the task note
  anticipated). It has not been observed live in this task.

## Concerns

1. Heal mode's green result now depends on the artifact script change (Deviation 2);
   if Task 4 rewrites `Start-Srv` from the plan text, it must retain `-fork_step 512`
   (and the appended parameter) or heal mode regresses again.
2. The mandated `-c 4096` run makes `fork-miss: tokens match the baseline` vacuous: the
   scenario prefills exactly 4096 tokens, generation then decodes at position 4096 and
   fails (`failed to prepare attention ubatches`), so both `generate()` results are empty
   and compare equal. The `-c 8192` diagnostic shows the same check genuinely passing
   with real generation. No action taken (test kept verbatim, per instructions); Task 4's
   fork mode (ctx 16384) is the real generation-equality evidence for miss-heal.
3. The new model scenario proves engine behavior only; server miss-heal wiring is
   compile-verified and code-inspected here, functionally deferred to Task 4.
4. Plan arithmetic off by one (63 vs actual 64); doc-only.

## Exact commands run

```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='1'
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 8192 --mode model --ram-mib 4096   # diagnostic
Remove-Item 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\llama-server-impl.dll'
& '<TEMP>\v100\build_server.cmd'
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode heal
git add tools/server/server-context.cpp tests/test-t32-tree.cpp
git commit -m "server : seed fork anchors when a tree restore misses" -m "Assisted-by: opencode"
```
