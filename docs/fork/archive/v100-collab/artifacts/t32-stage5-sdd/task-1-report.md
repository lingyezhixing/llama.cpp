# Task 1 Report: rename + new knob (D25/D29, no behavior change)

Status: DONE

## Commit

- `ad81d0efc` `common : rename kv tree checkpoint options and add the fork step`
  (trailer `Assisted-by: opencode`), branch `t32-stage5`, base `42ee7a6f7`.
- Files in commit: `common/common.h`, `common/arg.cpp`, `tools/server/server-context.cpp`, `tools/server/server-kv-tree.h`.

## Changes

| File:line | Change |
|---|---|
| `common/common.h:638` | `tree_anchor_step` -> `tree_checkpoint_anchor_step = 32768` |
| `common/common.h:639` | new `tree_checkpoint_fork_step = 8192` |
| `common/arg.cpp:1754-1763` | `--tree-anchor-step` -> `--tree-checkpoint-anchor-step`, env `LLAMA_ARG_TREE_CHECKPOINT_ANCHOR_STEP` |
| `common/arg.cpp:1764-1773` | new `--tree-checkpoint-fork-step`, env `LLAMA_ARG_TREE_CHECKPOINT_FORK_STEP` |
| `tools/server/server-context.cpp:1498-1499` | `cfg.anchor_step` <- `tree_checkpoint_anchor_step`; `cfg.fork_step` <- `tree_checkpoint_fork_step` |
| `tools/server/server-context.cpp:1517-1518` | startup log now includes `fork_step = %d` |
| `tools/server/server-kv-tree.h:17` | new `kv_tree_config::fork_step = 8192` (stored only, unused in this task) |
| `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1:59` | `'--tree-anchor-step'` -> `'--tree-checkpoint-anchor-step'` (only change in that script) |

No other log wording changed. Old CLI option / env var / field name no longer exist (`git grep` returns none).

## Verification (Step 6)

1. Test target build:
   `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree`
   Result: builds clean, `[49/49] Linking CXX executable bin\Release\test-t32-tree.exe`
   (fresh exe timestamp 2026/9/27 19:53:50). No errors.

2. Server build (after deleting `build\bin\Release\llama-server-impl.dll`):
   `& '<TEMP>\v100\build_server.cmd`
   Result: builds clean, `[17/17] Linking CXX executable bin\Release\llama-server.exe`.
   Note: UI asset download timed out (offline environment) and CMake fell back to
   "building without an embedded UI" - pre-existing environment behavior, unrelated.

3. `llama-server.exe --help`:
   ```
   --tree-checkpoint-anchor-step N         minimum spacing between kv tree checkpoint anchors in tokens (default:
   --tree-checkpoint-fork-step N           minimum spacing between kv tree fork anchors in tokens (default: 8192)
   ```
   No `--tree-anchor-step` line (pattern `tree-anchor` matched nothing). The first
   line's `(default: ...` is only cut by console width wrapping.

4. Heal acceptance (`$env:CUDA_VISIBLE_DEVICES='1'`):
   `powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode heal`
   Result: `RESULT heal: 0 failure(s)`

5. Startup log from the heal run (confirms cfg mapping + new log field):
   ```
   kv tree enabled: chunk = 512, anchor_step = 512, fork_step = 8192, ram = 512 MiB, disk = ...\tree-heal (2048 MiB)
   ```

## Self-review findings

- Deviation from brief snippet (syntax only): the brief's arg.cpp replacement shows
  two comma-separated brace blocks in one call, which implies a two-argument
  `add_opt`. In this tree `add_opt` is a unary lambda (`common/arg.cpp:1432`,
  `[&](common_arg arg)`), so that nesting cannot compile. Implemented as two
  separate `add_opt(common_arg(...))` calls; all names, defaults, help strings,
  env vars, error messages and ordering are verbatim from the brief and match
  the adjacent one-option-per-`add_opt` style.
- Brief's Step 5 path (`artifacts\t32-stage3-ab.ps1`) is the v100-collab copy
  outside the repo; changed there, not committed (matches brief commit scope).
- Field alignment in `common.h` intentionally differs from the shorter existing
  fields (long names), exactly as the brief's snippet shows.
- `cfg.fork_step` is stored/logged only; no consumer yet (expected for this task).

## Concerns

- None blocking. Consumers must use the new option/env names from now on;
  old `--tree-anchor-step` / `LLAMA_ARG_TREE_ANCHOR_STEP` now fail to parse
  (intended per D29, tree not deployed).
