# T32 stage 2 - final fix report

Date: 2026-09-27
Branch: t32-stage2
Base: a9d5d9595 (tests : add kv tree acceptance harness)

## What changed

### Fix 1 - park early-return visibility
`tools/server/server-kv-tree.cpp`, `kv_tree::park`:
when the tip sequence is already stored, the early return now logs how many
checkpoint candidates were dropped and counts them in `anchors_skipped`.
Previously the candidates were dropped silently (park returned true, nothing
reported).

### Fix 2 - scope promote_prune to the capture chain
`tools/server/server-kv-tree.h` and `tools/server/server-kv-tree.cpp`:
- `promote_prune(llama_pos)` -> `promote_prune(const std::vector<uint64_t> & chain, llama_pos)`.
- Removed the `last_promote` member.
- The previous anchor is now the closest anchor below `pos` that sits on the
  same capture chain; pruning is limited to anchors strictly between that
  previous anchor and `pos` that are on the same chain (still skipping TIP
  anchors, refcount >= 2 and pinned anchors).
This prevents a heal on branch B from deleting branch A's exclusive anchors.

### Fix 3 - non-aligned restore scenario
`tests/test-t32-tree.cpp`: new `scenario_unaligned` (inserted after
`scenario_ssd`, wired into `main`'s model branch). It parks 1300 tokens
(2 full chunks + a 276-token tail) with a checkpoint at the non-aligned
position 1150, then:
- restores the 1200-token prefix: matches 2 full blocks, partially matches the
  stored tail (176 tokens), picks the anchor at 1150, loads the partial block
  (overshoots to 1300), trims with `seq_rm(1150, -1)`;
- restores the full 1300 tokens: anchors at the non-aligned tip.
Both paths are compared bit-exact against matched baselines (same batch
boundaries at the original fill).

## Commands and results

Build:
```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
```
Result: EXIT=0 (ggml/llama also rebuilt because CMake re-ran; no src/ edits).

Logic mode:
```
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
```
Result: EXIT=0, 18 PASS / 0 FAIL.

2B model (Qwen3.5-2B-UD-Q4_K_XL, CUDA_VISIBLE_DEVICES=0, -ngl 99 -fa on -c 4096 --ram-mib 4096):
```
EXIT=0 PASS=43 FAIL=0 (7 unaligned: checks, all PASS)
```
Includes "unaligned: tokens match the matched baseline" and
"unaligned: full length tokens match the matched baseline" with no loosened
assertions.

3B control (Qwen2.5-Coder-3B-IQ4_XS, same flags; pure attention, the unaligned
scenario also runs):
```
EXIT=0 PASS=43 FAIL=0 (7 unaligned: checks, all PASS)
```

## Commits

- `9fc69f7c1cbd7959bca6c0c113aa9af71981d895` server : report unadopted checkpoints and scope anchor pruning to the chain (module files)
- `e7eea21ead32cbb9746d8df986cf993b6ee66346` tests : cover the non-aligned kv tree restore path (harness)

Both carry `Assisted-by: opencode`. Local only, no push.

## Issues / notes

- The plan prose said "six new unaligned: checks"; the specified scenario
  actually contains 7 check lines (1 park + 3 partial-length + 3 full-length).
  All 7 pass; total is the previous 36 + 7 = 43.
- No token-comparison failure occurred, so no evidence collection / BLOCKED
  path was needed.
- `git` prints LF/CRLF warnings for the touched files; this is pre-existing
  `core.autocrlf` behavior, not a content issue.
