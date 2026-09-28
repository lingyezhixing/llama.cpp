# Task 3 Report: D12 anchor payload return + checkpoint table rebuild

Status: DONE

Branch: `t32-stage4` (commits authorized, no push, no PR)

## What was implemented

**Engine (`kv_tree`)**
- New `struct kv_tree_restore_anchor { llama_pos pos; std::vector<uint8_t> data_tgt; std::vector<uint8_t> data_dft; }`.
- `kv_tree_restore::anchors` changed from `std::vector<llama_pos>` to `std::vector<kv_tree_restore_anchor>` (path anchors up to C, ascending).
- `kv_tree::restore` collects `(hash, pos)` path anchors, then after the restore loads succeed and before unpinning/settle, copies (not moves) each anchor payload into `out_anchors` via `load_payload`. Unavailable payloads are skipped (fewer rebuilt checkpoints), restore still succeeds.

**Server**
- `server_slot::prompt_restore_tree` takes `int n_ckpt_max`; after a successful restore it rebuilds `prompt.checkpoints` from `res.anchors`, filtering `pos <= 0 || pos >= C`, converting each into a `common_prompt_checkpoint` (`id_task = -1`, `update_pos(pos, 0, pos - 1)`, moved payloads), truncates to `n_ckpt_max` keeping the deepest anchors, and logs `kv tree: rebuilt %zu context checkpoints`.
- Call site in `get_available_slot` passes `params_base.n_ctx_checkpoints`.
- `server_context` gets `int64_t tree_ops = 0;`; after the cache-update block in `get_available_slot`, every 64 ops logs `kv tree stats: %s` via `tree->stats_line()`.

**Artifact (outside git)**
- `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` heal section: counts `rebuilt \d+ context checkpoints`, writes `HEAL REBUILT lines=$rebuilt`, asserts `>= 1` (D12).

## TDD evidence

### RED (Step 1-2)
Added `scenario_restore_anchors` (registered between `scenario_ssd` and `scenario_unaligned`) and built:

```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
> FAILED: [code=2] tests/CMakeFiles/test-t32-tree.dir/Release/test-t32-tree.cpp.obj
> ...\tests\test-t32-tree.cpp(906): error C2228: ".pos" the left side must be class/struct/union
> ninja: build stopped: subcommand failed.
```

Compile error is the expected RED: `r.anchors` was still `std::vector<llama_pos>`.

### Intermediate failure (test plumbing fix)
First run with the brief's literal prefill calls failed:

```
[kv-tree] park refused: sequence end 2047, expected 3071
restore-anchors: park with the checkpoint   FAIL
...
PASS=45 FAIL=10
```

Cause: the brief calls `prefill(ctx, 0, std::vector<llama_token>(tokens.begin() + 2048, tokens.end()), 2048, 512)`, but the harness `prefill` treats `from` as an index into the vector it is given (test-t32-tree.cpp:498), so a 1024-long slice with `from = 2048` decodes nothing. Fixed both occurrences in the new scenario to `prefill(ctx, 0, tokens, 2048, 512)`, the same pattern used by `scenario_fork` (test-t32-tree.cpp:660). No production code was affected.

### GREEN (Step 4, model mode, GPU 1)
```
$env:CUDA_VISIBLE_DEVICES='1'
build_test_t32.cmd test-t32-tree            (BUILD_EXIT=0)
test-t32-tree.exe -m Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -c 4096
    --mode model --ram-mib 4096
EXIT=0  PASS=55  FAIL=0
```

All 12 `restore-anchors` checks PASS, including the RAM path (`payload matches the parked checkpoint`, `two anchors on the path`, `sorted by position`) and the disk path (`data went to disk`, `disk payload matches the parked checkpoint`).

## Server build + heal acceptance

Server built after deleting `build\bin\Release\llama-server-impl.dll` (`BUILD_EXIT=0`; only pre-existing UI download timeouts).

```
t32-stage3-ab.ps1 -Mode heal
HEAL METRICS captured=1 stored=True notstored=0 failed=0 missed=0 parked=7 restored=2
HEAL REBUILT lines=1
PASS  heal: context checkpoints rebuilt after the fork restore (D12)
PASS  heal: exactly one heal capture
PASS  heal: the captured anchor is genuinely stored (dump shows @pos kind=2)
PASS  heal: no failed captures
PASS  heal: request 1..4 identical (tree vs full prefill)
RESULT heal: 0 failure(s)
EXIT=0
```

Server log evidence (fresh run, `<TEMP>\v100\t32-stage3\srv-heal-err.txt`):

```
kv tree: restored 487 tokens (heal = 1024)
kv tree: rebuilt 3 context checkpoints
kv tree: restored 2443 tokens (heal = 2447)
```

## Files changed

Commit `d7d02b908` `server : return kv tree anchor payloads on restore` (3 files, 119+/6-):
- `tools/server/server-kv-tree.h`
- `tools/server/server-kv-tree.cpp`
- `tests/test-t32-tree.cpp`

Commit `2575f41f0` `server : rebuild context checkpoints after a tree restore` (1 file, 37+/3-):
- `tools/server/server-context.cpp`

Both commits carry `Assisted-by: opencode`. Not in git: `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` (v100-collab is not a git repo).

## Self-review findings

- Placement follows the brief: payload collection happens after the C-anchor load succeeds and while the path anchors are still pinned, before unpin/`settle()`; data is copied so later demotion/eviction cannot invalidate the returned payloads.
- `anchors.at(key)` is safe: every key in `path_anchors` has `pos <= C` and is in `cand`, so it was pinned above and cannot be evicted before this loop.
- The rebuild filter `a.pos >= res.C` excludes the restore point itself, so the tip anchor does not become a mid-prompt checkpoint.
- `n_ckpt_max <= 0` leaves `prompt.checkpoints` empty, matching pre-existing behavior for that config.
- `common_prompt_checkpoint::update_pos` initializes `n_tokens/pos_min/pos_max`; `data_spec` stays default-empty, consistent with tree anchors.
- Acceptance log shows the rebuild path is exercised and behavioral equality with full prefill holds (4/4 request hashes).

## Concerns

- The brief's test code had a real bug (suffix prefill with an absolute `from`); deviation documented above. If the controller wants the literal brief form kept, the harness `prefill` would need a different contract - not done.
- The `kv tree stats:` aggregate log (every 64 ops) was compile-verified only; the heal run performs far fewer than 64 cache updates, so it never fired. No runtime evidence for that one line.
- No other concerns; `RESULT heal: 0 failure(s)` with `REBUILT lines=1`, so no assertion weakening was needed.

---

# Fix report: double `load_payload` on the disk-resident C anchor

Status: DONE (commit `569f05669`)

## Reviewer finding

`kv_tree::restore` loaded the anchor at `C` (server-kv-tree.cpp:1384) and then the new collection loop loaded it again. `load_payload(kv_tree_anchor &)` never cleared `on_disk`, so a second call re-read the file and added `anchor_bytes(a)` to `st.bytes_ram`/`st.anchors_ram`/`st.bytes_load` again. `settle()`/`remove_anchor()` subtract one copy, leaving a phantom `a.bytes` per disk-backed restore: inflated stats, biased demotion/eviction, and monotonic growth until `enforce_budget()` refuses permanently.

## What changed

- `tools/server/server-kv-tree.cpp` - `load_payload(kv_tree_anchor & a)` is now idempotent for resident payloads:

```cpp
bool kv_tree::load_payload(kv_tree_anchor & a) {
    if (!a.on_disk || a.transient) {
        return true;
    }
```

- `tests/test-t32-tree.cpp` - SSD variant of `scenario_restore_anchors` now runs a second `tree.restore(io, nullptr, tokens)` after capturing `tree.stats().bytes_ram` and asserts the value is unchanged:

```cpp
        const int64_t bytes_ram_after_first = tree.stats().bytes_ram;
        tree.restore(io, nullptr, tokens);
        check_eq(tree.stats().bytes_ram, bytes_ram_after_first, "restore-anchors: repeated disk restores do not inflate ram accounting");
```

Head commit: `569f05669 server : fix kv tree anchor payload double load` (`Assisted-by: opencode`), 2 files, 5+/1-.

## RED: assertion catches the defect (guard temporarily reverted)

Command (same model-mode invocation as below, with only `|| a.transient` removed):

```
[t32-tree] restore-anchors: repeated disk restores do not inflate ram accounting    FAIL (got 40404120, want 20202060)
EXIT=1  PASS=55 FAIL=1
```

The delta is exactly one anchor payload (20,202,060 B) leaked by the second disk-backed restore, confirming the finding and that the assertion pins it.

## GREEN: covering tests

Logic mode (no model):

```
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic
EXIT=0  PASS=64  FAIL=0
```

Model mode:

```
$env:CUDA_VISIBLE_DEVICES='1'
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
[t32-tree] restore-anchors: repeated disk restores do not inflate ram accounting    PASS (got 0, want 0)
EXIT=0  PASS=56  FAIL=0
```

Counts match the controller's expectation: logic 64, model 56 (55 + 1 new). The heal acceptance was not re-run because the fix and assertion are scoped to the anchor RAM accounting path covered by the model test; no server-context behavior changed.

Logs: `<user>\AppData\Local\Temp\opencode\t32-stage4-task3-fix-red.txt` (RED), `...fix-logic.txt`, `...fix-model.txt` (GREEN).

## Remaining concerns

- None new. The block-resident variant of `load_payload` keeps its old guard; no current caller double-loads a block within a restore, so it was left untouched per the minimal-fix ruling.

---

# Fix report: tail semantics for rebuilt checkpoints on recurrent contexts

Status: DONE (commit `c0619da14`)

## Root cause (controller-diagnosed)

At soak round ~137 the server aborted at `common/common.cpp:1580: failed to remove sequence 1 with p0=520, p1=-1`. A restored slot rebuilt a checkpoint (anchor at pos 521) with `pos_min = 0`, `pos_max = 520`. On the next request with a 520-token LCP reuse, the server's checkpoint-rollback search selected it because `cur.pos_min == 0`, set `n_past = 520`, and the partial `seq_rm(520, -1)` on a hybrid/recurrent context hit the recurrent rollback branch that cannot roll back one token -> `GGML_ABORT`. Stock checkpoints for recurrent contexts carry `pos_min = pos_max = tail`, so the filter never selects them for a partial `seq_rm` in this situation.

## What changed

- `tools/server/server-context.cpp` - `prompt_restore_tree` gains `bool ckpt_tail`; the rebuild loop uses `ck.update_pos(a.pos, ckpt_tail ? a.pos - 1 : 0, a.pos - 1)` with a 2-line comment on why tail semantics are needed.
- Call site in `get_available_slot` passes:
  `ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS`
  via a local `const bool ckpt_tail` (pure-attention `PART` keeps `pos_min = 0`, stock semantics).
- Log line and all other behavior unchanged.

Commit: `c0619da14 server : use tail semantics for rebuilt checkpoints on recurrent contexts` (`Assisted-by: opencode`), 1 file, 7+/3-.

## Verification (all cuda1)

Server rebuild (delete dll + `build_server.cmd`): `BUILD_EXIT=0` (only pre-existing UI download timeouts).

Logic mode:
```
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic
EXIT=0  PASS=64  FAIL=0
```

Model mode:
```
$env:CUDA_VISIBLE_DEVICES='1'
& '...\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
EXIT=0  PASS=56  FAIL=0
```

Heal acceptance:
```
t32-stage3-ab.ps1 -Mode heal
HEAL METRICS captured=1 stored=True notstored=0 failed=0 missed=0 parked=7 restored=2
HEAL REBUILT lines=1
RESULT heal: 0 failure(s)
EXIT=0
```

Soak regression (previous run aborted at round ~137):
```
t32-stage3-ab.ps1 -Mode soak -Minutes 5
SOAK METRICS rounds=293 parked=211 restored=94 rebuilt=79 failed=0 evict_refused=11 diskmax=536288092 cmp_ok=58 cmp_bad=0 rss_mb=1902->1976 handles=225->235 wr_mb=14667 rd_mb=5607
SOAK RESTART files_before=47 cleared_lines=1 files_after=0
RESULT soak: 0 failure(s)
EXIT=0
```

All 14 soak assertions PASS; no `failed to remove sequence` / abort line in the log. The D12 rebuild path ran 79 times past the old abort point with no failure.

Logs: `<user>\AppData\Local\Temp\opencode\t32-stage4-task3-soakfix-{logic,model,heal,soak}.txt`.

## Remaining concerns

- None. `ckpt_tail` semantics were not unit-tested directly (no server unit test exists); coverage is the soak + heal acceptance above, where the previously fatal request pattern runs repeatedly.

---

# Fix report: whole-branch review fixes (Important x1, Minor x2)

Status: DONE (commit `42ee7a6f7`)

## What changed

1. **Important - byte cap for rebuilt checkpoints** (`tools/server/server-context.cpp`). PART contexts ignore `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`, so each rebuilt checkpoint is a copy of nearly the whole prefix KV, uncounted by `kv_tree_stats` and uncapped by `--tree-ram`; `n_ctx_checkpoints` alone allowed multi-GB bursts (e.g. hybrid 27B ~162 MB x 32). The rebuild now tracks `ck_bytes` (`data_tgt.size() + data_dft.size()`) and, after the count cap, drops from the front while `ck_bytes > 256 MiB`, keeping the deepest anchors. Count-cap drops decrement `ck_bytes` too. No CLI option added (YAGNI).
2. **Minor - block `load_payload` transient guard** (`tools/server/server-kv-tree.cpp`): `if (!b.on_disk || b.transient)` for symmetry with the anchor overload.
3. **Minor - ctor wipe counting** (`tools/server/server-kv-tree.cpp`): replaced the throwing iterator count with `remove_all(dir + "/blocks", ec) + remove_all(dir + "/anchors", ec)`, which returns the removed count and drops the inaccurate regular-file tally.

Artifacts updated (outside git): `t32-tree-plan-stage4.md` D18 amended with the tail-semantics refinement (c0619da14) and a new D24 entry for the 256 MiB cap (header now D14-D24); `t32-tree-storage-design.md` §9 D12 note gained one sentence on the byte cap.

## Verification (cuda1)

Server rebuild (delete dll + `build_server.cmd`): `BUILD_EXIT=0` (only pre-existing UI download timeouts).

Logic mode:
```
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic
EXIT=0  PASS=64  FAIL=0
```

Model mode:
```
$env:CUDA_VISIBLE_DEVICES='1'
& '...\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
EXIT=0  PASS=56  FAIL=0
```

Heal acceptance:
```
t32-stage3-ab.ps1 -Mode heal
HEAL METRICS captured=1 stored=True notstored=0 failed=0 missed=0 parked=7 restored=2
HEAL REBUILT lines=1
RESULT heal: 0 failure(s)
EXIT=0
```

Soak (5 min):
```
t32-stage3-ab.ps1 -Mode soak -Minutes 5
SOAK METRICS rounds=305 parked=218 restored=95 rebuilt=80 failed=0 evict_refused=13 diskmax=536288092 cmp_ok=61 cmp_bad=0 rss_mb=1902->1918 handles=274->284 wr_mb=15351 rd_mb=5674
SOAK RESTART files_before=9 cleared_lines=1 files_after=0
RESULT soak: 0 failure(s)
EXIT=0
```

14/14 soak assertions PASS, no abort lines. Logs: `<user>\AppData\Local\Temp\opencode\t32-stage4-task3-final-{logic,model,heal,soak}.txt`.

## Remaining concerns

- The 256 MiB cap is not unit-tested (no server test); anchors in the 2B matrix are far below it, so the drop-shallowest loop never fires in the acceptance runs. The accounting path (count cap, byte tracking) is exercised on every rebuild.
- `remove_all` counts directories as well as files, so the "cleared N stale files" log is now a removed-entries count, which can include the `blocks`/`anchors` directories themselves (per the ruled fix).
- The 5-min soak ran on the hybrid 2B with small anchors, so the byte cap threshold itself is not stressed; it only bounds the PART-model burst case on paper.



