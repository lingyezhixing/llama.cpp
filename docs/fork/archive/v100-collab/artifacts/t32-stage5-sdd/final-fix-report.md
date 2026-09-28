# Final fix wave report: T32 stage 5 polish (branch `t32-stage5`)

Date: 2026-09-27
Base: `04954b468` (stage-5 whole-branch review verdict "Ready to merge: Yes", 6 minor findings)
Fix-wave head: `af5e4c934`
Scope: all 6 findings (comment rationale, test baseline guard, ctor validation, park-side step-skip
assertion, design-doc notes, script comment). No push, no deploy.

## Commit

| commit | files | message |
|---|---|---|
| `af5e4c934` | `tools/server/server-kv-tree.cpp` (5+/4-), `tests/test-t32-tree.cpp` (18+) | `server : polish kv tree fork notes and validation` |

Carries `Assisted-by: opencode`. The two artifact edits (`D:\LLM\Backend\v100-collab\artifacts\...`)
live outside the repo and are not part of the commit.

## Findings

### 1. Wrong rationale comment (comment only)

`tools/server/server-kv-tree.cpp:413` (was 413-414):

```cpp
// no pruning needed: prev is the deepest same-chain anchor below pos, so no anchor can sit between them
```

Same wording fixed in `artifacts\t32-tree-plan-stage5.md` (D27 and the Task 2 code snippet at
the old line 288) and `artifacts\t32-tree-storage-design.md` (§9 D27 note), both kept in their
original language style. The old text claimed the spacing rule (`prev <= pos - fork_step`) is the
reason; it is not - `prev` is the deepest same-chain anchor below `pos` by construction, which also
holds for `fork_step = 0`.

### 2. `scenario_fork_miss` baseline guard

`tests/test-t32-tree.cpp` (`scenario_fork_miss`): added, matching `scenario_tip`:

```cpp
const auto base = run_baseline(ctx, tok_b, 42, 8);
check(!base.empty(), "fork-miss: baseline generation");
```

### 3. ctor validation for `fork_step`

`tools/server/server-kv-tree.cpp:183`:

```cpp
if (cfg.chunk <= 0 || cfg.anchor_step < 0 || cfg.fork_step < 0) {
    fprintf(stderr, "[kv-tree] invalid config: chunk = %d, anchor_step = %d, fork_step = %d\n", cfg.chunk, cfg.anchor_step, cfg.fork_step);
    GGML_ABORT("invalid kv tree config");
}
```

### 4. Park-side `anchors_skipped_step` assertion

`tests/test-t32-tree.cpp` (`run_logic_fork`, first block, after the existing `step_skips=1` check):
one new `check_eq` on the `anchors_skipped_step` delta. The candidate is at 8192 with a non-empty
payload; it is skipped because the same-chain anchor at 1024 is 7168 tokens below it (< anchor_step
32768), so only the skip counter moves.

Deviation from the concrete suggestion: re-parking the *same* token vector hits `park()`'s
"sequence already stored" early return, which bumps the plain `anchors_skipped` counter and never
reaches the candidate loop. Re-parking the 9216-token prefix keeps the loop reachable and its tip
anchor `(blk@9216)` already exists (from the capture at 9216), so `anchors_added` stays 3 as
required. Only 1 new check was added, keeping logic at 74 + 1 = 75.

### 5. Design doc notes

`artifacts\t32-tree-storage-design.md` §9, appended after D27:

- tree workflows run with context shift effectively off (acceptance always passes `--no-context-shift`;
  the `pos_max` guard does not prove the cached window starts at 0);
- heal at `m.deep == tokens.size()` (all prompt tokens block-matched but no anchor covering them) is
  not captured (perf-only; same as the stage-3 heal-on-hit note).

### 6. Script cosmetic

`artifacts\t32-stage3-ab.ps1:2`, mode list comment: `...|soak` -> `...|soak|fork`.

## Verification

Test target build:

```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
[10/11] Linking CXX executable bin\Release\test-t32-tree.exe
```

Logic mode (expect 75 = 74 + 1, 0 FAIL):

```
build\bin\Release\test-t32-tree.exe --mode logic
exit 0, 75 PASS / 0 FAIL
[t32-tree] fork: the park candidate skip is counted                                 PASS (got 1, want 1)
```

Model mode (expect 65 = 64 + 1, 0 FAIL; device cuda1):

```
$env:CUDA_VISIBLE_DEVICES='1'
test-t32-tree.exe -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on --mode model --ram-mib 4096 -c 8192
exit 0, 65 PASS / 0 FAIL
[t32-tree] fork-miss: baseline generation                                           PASS
```

Logs: `<TEMP>\v100\t32-polish-{logic,model}.txt`.

Server rebuild (ctor change is compiled into the server too):

```
Remove-Item build\bin\Release\llama-server-impl.dll
& '<TEMP>\v100\build_server.cmd'
[5/5] Linking CXX executable bin\Release\llama-server.exe
(UI asset download timeout is the known offline behavior; assets unchanged)
```

Heal acceptance (expect 0 failures):

```
powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode heal
HEAL METRICS captured=1 stored=True notstored=0 failed=0 missed=0 parked=7 restored=2
RESULT heal: 0 failure(s)
```

Log: `<TEMP>\v100\t32-polish-heal.txt`.

## Concerns

- Finding 4 is asserted as a delta against a saved `step_skips_before` because the new park also
  runs after the existing absolute-value checks; any reordering of the block must move that capture
  with it.
- No behavior change (comment, validation, tests); the other acceptance modes were not re-run.
  `heal` was re-run because the server binary was rebuilt.
