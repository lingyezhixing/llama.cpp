# Task 3 report: park/restore integrated into `get_available_slot`

Status: **DONE** (initially BLOCKED; resolved by the controller ruling applied
in the addendum at the end of this file).

Branch: `t32-stage3`, base/tip `df4225828` (clean before this task).
No commit was created (Step 4 not reached).

## What was implemented (verbatim per brief)

`tools/server/server-context.cpp` only:

- The `#include "server-kv-tree.h"` was already present (added by Task 2, ruling
  recorded in progress.md); no-op.
- Step 1: after `server_prompt prompt;` in `server_slot`:
  - `llama_pos tree_heal = -1;`
  - `bool prompt_park(kv_tree & tree) const`
  - `bool prompt_restore_tree(kv_tree & tree, const server_tokens & tokens)`
  All code, names, and log strings copied verbatim from the brief.
- Step 2: the `if (ret) { ... }` tail of `get_available_slot` replaced verbatim:
  `ret->tree_heal = -1;`, `update_cache = update_cache && (prompt_cache || tree);`,
  and the `if (tree) { park; restore-or-clear } else { stock prompt_cache path }`
  branch.

Diff: 85 insertions, 6 deletions in `tools/server/server-context.cpp` (uncommitted).
The working tree currently contains exactly the brief's code.

## Build

`Remove-Item llama-server-impl.dll` then `build_server.cmd` -> EXIT=0
(`server-context.cpp.obj` rebuilt, `llama-server-impl.dll` + `.exe` linked).

## Smoke result with the brief's code: FAIL (no park, no reuse)

Exact Step 3 command (log: `<TEMP>\v100\s3smoke-err.txt`):

```
0.25.059 I slot get_availabl: selected slot by LRU
0.25.059 I slot prompt_resto: kv tree: restore miss for 407 tokens, full prefill
0.25.775 I slot      release: stop processing: n_tokens = 454
0.25.815 I slot get_availabl: selected slot by LRU
0.25.815 W slot  prompt_park: kv tree: park skipped (seq range [453, 453], tokens 454)
0.25.815 I slot prompt_resto: kv tree: restore miss for 407 tokens, full prefill
... (both lines repeat for requests 3 and 4)
prompt_n A1 = 407, A2 = 407
```

- No `parked`, no `restored` line ever. `prompt_park` always exits at the
  position guard.
- All 4 requests succeed but always do a full 407-token prefill.

## Root cause (not a budget issue)

The smoke model `Qwen3.5-2B-UD-Q4_K_XL.gguf` is a hybrid SSM+attention model
(GGUF keys `qwen35.ssm.*`; the stage 3 plan itself calls it "2B hybrid",
`t32-tree-plan-stage3.md:21`).

- `llama_memory_hybrid::seq_pos_min` returns
  `std::max(mem_attn->seq_pos_min(seq), mem_recr->seq_pos_min(seq))`
  (`src/llama-memory-hybrid.cpp:172-174`).
- The recurrent part reports the position of its single state cell, i.e. the
  last processed position (`src/llama-memory-recurrent.cpp:495-509`), here 453.
- Therefore `llama_memory_seq_pos_min` on this model is always `p_max`
  (observed `[453, 453]`), never 0, and the brief's guard
  `if (p_min != 0 || p_max != tokens.size() - 1)` can never pass.
- There is no public API to query only the attention sub-memory's `pos_min`
  (`llama_get_memory` returns the hybrid wrapper; `get_mem_attn()` is in `src/`).
- `--tree-ram` cannot change this: the guard runs before `tree.park()`; the
  budget only affects the tree's internal eviction.

`tree.park`'s own invariant (`io_tgt.pos_max() == L - 1`, tip anchor `get_partial`,
range reads via `llama_state_seq_*_range_ext`) is hybrid-safe; only the caller's
guard is not.

## Diagnostic with a one-line guard fix: full expected result

To validate a fix before escalating, I temporarily changed the guard to
`if (p_max != (llama_pos) prompt.tokens.size() - 1)` (only the `p_min != 0`
term removed), rebuilt, and ran the exact Step 3 smoke. This edit was then
reverted; it is NOT in the working tree. Log:
`<TEMP>\v100\s3smoke-exp-err.txt`.

```
A2 output: 100% system: you are helpful. alpha ...
prompt_n A1 = 407, A2 = 4, B1 = 407, B2 = 4
restore miss for 407 tokens, full prefill            (A first)
parked 454 tokens, 1 checkpoint candidates, ram = 44686416 B, disk = 0 B
restore miss for 407 tokens, full prefill            (B first)
parked 454 tokens, 1 checkpoint candidates, ram = 60606180 B, disk = 28766652 B
restored 403 tokens (heal = 407)                     (A second)
parked 454 tokens, 1 checkpoint candidates, ram = 64888476 B, disk = 24484356 B
restored 403 tokens (heal = 407)                     (B second)
```

This matches every expected item of the brief: 4 requests OK, A2 `prompt_n` (4)
far below A1 (407), `parked` + `restored` lines present, `disk =` > 0 on the
second park, and `restore miss` only on the first request of each session.

## Off-path sanity (`--kv-tree` off, binary from the verbatim tree)

Same script without tree flags (log: `s3smoke-off-err.txt`): 4 requests OK,
`r1.content == r3.content` and `r2.content == r4.content`, `prompt_n` 407/407,
zero `kv tree` log lines. The stock path is unchanged for this model.

## Proposed amendment (needs a controller ruling)

Replace the guard:

```cpp
if (p_min != 0 || p_max != (llama_pos) prompt.tokens.size() - 1) {
```

with a check that holds for hybrid memory, e.g.:

```cpp
if (p_max != (llama_pos) prompt.tokens.size() - 1) {
```

`p_min` may be kept for the log line or dropped. The `p_min != 0` term only
detected a removed prefix; it never detected mid-sequence trims, and it cannot
be evaluated correctly on hybrid memory with the public API. A hybrid-aware
variant gated on `llama_n_rs_seq(ctx_tgt) == 0` is possible but adds a branch for
little extra safety.

The logged `[%d, %d]` values would then report the raw memory range (e.g.
`[453, 453]`) even on success; if that matters for the log contract, the skip
message can stay as-is and only the condition changes (the skip message is
unchanged either way).

With this ruling applied, I can re-apply the one-line change, rebuild, re-run
the exact Step 3 smoke (already validated above), and commit per Step 4.

## Files changed

```
 tools/server/server-context.cpp | 91 ++++++++++++++++++++++++++++++---
 1 file changed, 85 insertions(+), 6 deletions(-)
```

Uncommitted. No other file touched (no new files; `src/`, `server-kv-tree.*`,
`common/*`, `tests/*` untouched).

## Self-review findings

- Steps 1 and 2 are byte-for-byte the brief's code; log strings match the
  contract exactly.
- `tree_heal` is reset to -1 on every slot selection (Step 2) and set from the
  restore result; `prompt_restore_tree` clears `prompt.checkpoints`.
- `--kv-tree` off: `update_cache = update_cache && (prompt_cache || tree)` is
  equivalent to the stock expression, and the else-branch reproduces the stock
  flow (verified by the off-path run).
- Comments are ASCII, concise, English; no extra features or metrics work.
- Build reproducible; after the diagnostic revert the tree was rebuilt so the
  binary matches the sources again.

## Concerns

1. The plan's Task 3 guard is incompatible with the plan's own hybrid smoke
   model; Tasks 5's A/B scenarios run the same model and would hit the same
   failure until the guard is amended.
2. Because the fix was not authorized, this task is BLOCKED with no commit;
   the implementation is complete and verified modulo the one-line guard.

## Commands (verbatim)

Build (twice, verbatim and after revert):

```powershell
Remove-Item -Force 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\llama-server-impl.dll' -ErrorAction SilentlyContinue
& '<TEMP>\v100\build_server.cmd'
```

Smoke: Step 3 block of the brief, unmodified (port 8931, dir
`<TEMP>\v100\t32-tree-smoke`).

---

# Addendum: ruling applied, task completed

Status after addendum: **DONE**. Commit `376fa06ad` (branch `t32-stage3`).

## What changed vs the brief and why

Applied the controller ruling (D9) exactly: in `prompt_park`, dropped the
`p_min != 0` term and the `llama_memory_seq_pos_min` call, kept only the
`p_max != (llama_pos) prompt.tokens.size() - 1` check; added the three-line
comment from the ruling; warning string changed to
`kv tree: park skipped (seq end %d, tokens %zu)`.

Reason: `llama_memory_seq_pos_min` is not usable as a guard on hybrid memory -
it returns `max(attn_min, recr_min)` and the recurrent part reports its tail, so
`p_min` is never 0 on the 2B hybrid smoke model (root cause section above).
The server clears and reprocesses whenever a front removal is not supported, so
positions are 0-based whenever the KV is consistent with the token list.

Everything else in Steps 1 and 2 stays verbatim per the brief. Diff vs base
`df4225828`: 87 insertions, 6 deletions, only `tools/server/server-context.cpp`.

## Build (after amendment)

Deleted `build\bin\Release\llama-server-impl.dll`, ran `build_server.cmd` ->
EXIT=0.

## Step 3 smoke (exact brief command, `--tree-ram 64`)

Log: `<TEMP>\v100\s3smoke-err.txt`.

```
prompt_n A1 = 407, A2 = 4, B1 = 407, B2 = 4
restore miss for 407 tokens, full prefill            (A first)
parked 454 tokens, 1 checkpoint candidates, ram = 44686416 B, disk = 0 B
restore miss for 407 tokens, full prefill            (B first)
parked 454 tokens, 1 checkpoint candidates, ram = 60606180 B, disk = 28766652 B
restored 403 tokens (heal = 407)                     (A second)
parked 454 tokens, 1 checkpoint candidates, ram = 64888476 B, disk = 24484356 B
restored 403 tokens (heal = 407)                     (B second)
```

Checklist:
- 4 requests succeed. Yes.
- A2/B2 `prompt_n` (4) clearly below A1/B1 (407). Yes.
- `kv tree: parked` and `kv tree: restored` lines present. Yes.
- `disk =` > 0 on a parked line. Yes (B park 28766652 B; A2 park 24484356 B).
- `restore miss` only on the first request of each session. Yes (A1, B1).
- No `park skipped` lines. Yes (grep pattern `parked|restored|restore miss|park skipped`
  returned only the 7 lines above).

## Off-path check (`--kv-tree` absent)

Log: `<TEMP>\v100\s3smoke-off-err.txt`.

- 4 requests OK; `r1.content == r3.content`, `r2.content == r4.content`.
- `prompt_n` 407/407 (full prefill in both passes).
- zero `kv tree` log lines.

Direct A/B content check (both servers live, same prompt, greedy):
tree-on A2 content is byte-identical to the off-path content (251 chars each);
tree-on prompt_n 407/4, off-path 407. Logs:
`cmp-tree-err.txt`, `cmp-off-err.txt` in the same temp dir.

## Commit

`376fa06ad server : reuse kv tree across slot switches` with
`Assisted-by: opencode`, one file changed.

## Remaining concerns

None. The only deviation from the brief is the ruled amendment above.
