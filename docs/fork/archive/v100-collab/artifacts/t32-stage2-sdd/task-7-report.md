# Task 7 report: acceptance harness (mini A/B) + archives + docs

Status: BLOCKED (two expectation mismatches; assertions left untouched per task policy; no commit; channel docs not updated; full evidence and proposed remedies below)

## What was implemented

`tests/test-t32-tree.cpp` only (75 added lines, not committed):

- `run_accept()` inserted verbatim from the brief between `scenario_ssd` and `main` (A-mini: two 4096-token forks sharing 3072 tokens, 6 alternating restore rounds; B-mini: four 1024-token sessions sharing a 512-token prefix).
- `main`: `if (mode == "accept") { return run_accept(ctx, cfg); }` added after the model branch.

Archives created with the brief's exact commands (UTF-8, no BOM, verified):

- `D:\LLM\Backend\v100-collab\artifacts\t32-stage2-model.txt` (7314 bytes, SHA256 `C2BAFF3AB...04B77822`)
- `D:\LLM\Backend\v100-collab\artifacts\t32-stage2-accept.txt` (5914 bytes, SHA256 `63644A200...4D7B9CBF`)

Note: the brief's `2>&1 | Out-String` capture produces a 7-line PowerShell error-record header at the top of each archive (the first native stderr line is wrapped by `NativeCommandError`); the run output itself is complete and unmodified after that header.

## Commands and results

Build (exit 0, only the pre-existing C4297 warning in `src/llama.cpp`):

```
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
```

Logic (`--mode logic`): exit 0, 18/18 PASS.

Model, 2B (`Qwen3.5-2B-UD-Q4_K_XL.gguf`, `-ngl 99 -fa on -c 4096 --mode model --ram-mib 4096`): exit 0, 36/36 PASS, 0 FAIL (tip 8/8, fork 13/13, sparsify 7/7, ssd 8/8). Archive: `t32-stage2-model.txt`.

Accept, 2B (`-c 8192 --mode accept --ram-mib 4096`): exit 1, **40 PASS / 2 FAIL**. Archive: `t32-stage2-accept.txt` lines 36-37:

```
[t32-tree] accept: shared trunk stored once (6+2 blocks)     FAIL (got 10, want 8)
[t32-tree] accept: stored bytes are far below two full copies FAIL
```

All A/B restores (12/12 at C=4096, tokens bit-exact), `accept: tokens reused` (49152 = 12 x 4096), all four B-mini park/restores and `accept: shared prefix stored once (1+4 blocks)` (5) PASS.

3B pure-attention control (`Qwen2.5-Coder-3B-IQ4_XS.gguf`, `-c 4096 --mode model --ram-mib 4096`): exit 1, **30 PASS / 6 FAIL**. tip 8/8, fork 13/13, sparsify 7/7, ssd **2/8**. The 6 FAILs:

```
[kv-tree] eviction could not free enough ram (56642424 > 65536)
[kv-tree] park refused: the budget cannot hold the sequence
[t32-tree] ssd: park                                     FAIL
[t32-tree] ssd: blocks demoted to disk                   FAIL
[t32-tree] ssd: tip anchor demoted to disk               FAIL
[t32-tree] ssd: no blocks left in ram                    PASS (got 0, want 0)
[t32-tree] ssd: restore point is the tip                 FAIL (got -1, want 1536)
[t32-tree] ssd: tokens match the baseline                FAIL
[t32-tree] ssd: restore misses when block files are gone PASS (got -1, want -1)   <- vacuous (nothing was stored)
[t32-tree] ssd: disk error counted                       FAIL
```

## Finding 1: accept A-mini expectations are arithmetically wrong (plan defect)

Block count: actual 10, correct. The dump (`t32-tree-accept.txt` lines 55-68) shows exactly:

- 6 shared blocks `[0,512)...[2560,3072)` with `ref=2` (stored once, both sequences reference them) - the check's intent holds;
- 4 tail blocks with `ref=1`: `[3072,3584)`/`[3584,4096)` for A (`ab3fdf44...`, `0c23857f...`) and the same positions for B (`0d6e8334...`, `c0d1c2d3...`).

With `chunk = 512`, each 1024-token salted tail is 2 blocks, so the unique count is 6 + 2 + 2 = 10. The brief's `8` ("6+2") counts both tails as one block each. The analogous B-mini check (`1+4 = 5`, four 512-token tails) passes, which confirms the same arithmetic done correctly. The observed tree behavior (dedup of the shared trunk, distinct salted tails, all restores bit-exact) is exactly the designed content-addressed semantics, so this is a wrong expected constant, not a product defect.

Bytes: the check compares `bytes_ram + bytes_disk <= seq_bytes * 10 / 8` (1.25 full-range blobs). Measured total 103,443,240 =

- blocks: 10 x 6,303,912 = 63,039,120 (= 1.25 x the 4096-token KV, modulo blob headers)
- tip anchors: 2 x 20,202,060 = 40,404,120 (recurrent/partial state of the hybrid 2B, one per sequence; Task 2 report already measured 20,202,060 per anchor)

Even with the brief's assumed 8 blocks, 8 x 6,303,912 + 40,404,120 = 90,843,216 > 1.25 x seq_bytes (~63.04 MB), so no single constant fix can satisfy the check as written; the assertion must be reformulated to account only for attention-KV block payloads (e.g. subtract the two anchor sizes, or split per-kind byte stats), or be replaced by an explicit block-payload bound. Note the KV-block portion alone (63,039,120) is exactly at the 1.25 x threshold, i.e. the threshold constant itself matches 10 blocks, not 8 - further evidence the check was written for blocks-only accounting with an even more wrong block count.

Per the task policy ("if a counter, block count, or token sequence differs ... collect evidence, document it, and report BLOCKED") the assertions were left untouched.

## Finding 2: 3B control `ssd` cannot pass with the scenario's hardcoded budget

`scenario_ssd` hardcodes `ram_limit = 64 KiB`, `disk_limit = 64 MiB` (CLI `--disk-mib` is ignored by that scenario). For a plain attention memory there is no recurrent part, and `llama_kv_cache::state_write/state_write_range` ignore `PARTIAL_ONLY` (only hybrid/ISWA/SWA implementations gate on it), so `get_partial(PARTIAL_ONLY)` returns the **whole attention KV**. For the 3B:

- 3 blocks = 3 x 18,881,400 = 56,644,200 bytes
- tip anchor = ~56.64 MB (the refusal message shows 56,642,424 bytes left in RAM)
- total ~113.3 MB > 64 MiB disk budget -> `enforce_budget` cannot demote/evict (new payloads are pinned) -> `park` refused -> the whole ssd scenario cascades (including a vacuous "restore misses" PASS and a FAIL on "disk error counted").

tip/fork/sparsify all pass on the 3B, and the fork criterion in the current harness (`fork: A' restores at the deepest usable anchor` = 512, `heal` = 1024, via the checkpoint anchor) is met. The brief's wording "A' 无锚点 -> C=-1" does not match the harness code that was actually written in Tasks 1-6 (which expects `C=512` there), so that part of the Expected text is inconsistent with the delivered harness.

So the 3B expectation in the brief ("tip / sparsify / ssd PASS") does not hold: ssd fails for pure-attention models by construction. This is either (a) an expectation error (the ssd scenario assumes hybrid-size anchors), or (b) a design question (for pure attention, tip anchors duplicate the entire KV; the design targets attention+recurrent hybrids), which needs a controller ruling, not an executor edit.

## Why I did not edit assertions, update docs, or commit

- Task policy: do not loosen/adjust assertions on a differing counter; collect evidence and report BLOCKED. Both findings are exactly such cases (block count 10 vs 8; ssd control expectation).
- The channel-doc text (Step 3) asserts "结论: park/restore 逐 token 与基线一致 ... " and a green stage exit; writing it would contradict the archives, which contain FAILs. The archives themselves were written exactly as the brief commands specify and are truthful.
- Step 4's commit is predicated on Step 2 green. I left `tests/test-t32-tree.cpp` uncommitted so the controller can rule on the expectations first; the harness diff is ready to commit unchanged once the plan is patched. No `src/` changes, no push, no deploys.

## Proposed remedies (for the controller / plan patch)

1. `tests/test-t32-tree.cpp` (and plan lines 2855-2857):
   - block count: expect `10`, label `"accept: shared trunk stored once (6+2+2 blocks)"`.
   - bytes: compare attention-KV block payloads only, not `bytes_ram + bytes_disk` (which include the two tip anchors). Minimal option: compute the anchor size in the harness via `llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)` before parking, and assert `bytes_ram + bytes_disk - 2 * anchor_bytes <= seq_bytes * 10 / 8`. Cleaner option (stage 3 scope): per-kind byte counters in `kv_tree_stats`.
2. 3B control: choose one:
   - accept hybrid-only behavior and change the expected result for pure attention (ssd `park` refused with the explicit WRN is correct degradation), or
   - make `scenario_ssd`'s budget data-sized (e.g. `disk_limit = 4 * seq_bytes`) so the control can pass, or
   - module change: skip storing the tip anchor payload for non-recurrent memories (design change; the block chain fully determines a pure-attention state).
3. Re-run: logic, model, accept, 3B control; then archives can be overwritten by the re-run and Step 3/4 executed as written.

## Files changed

- `tests/test-t32-tree.cpp`: +75 lines (uncommitted; `git diff` shows only this file; no working-tree changes elsewhere; 0 bytes > 127 in the added text; only the brief's two comments added).
- New channel files (outside repo): `artifacts\t32-stage2-model.txt`, `artifacts\t32-stage2-accept.txt`.
- Channel docs (`TASKS\T32-agent-session-reuse.md`, `RESULTS.md`, `STATUS.md`, `artifacts\t32-tree-storage-design.md`): not modified (see above).
- Commit: none.

## Self-review findings

- Transcription: run_accept body and main wiring match the brief character-for-character (verified by re-reading the inserted block); no extra comments, no scope creep, ASCII only.
- Counts: logic 18 PASS; model 36 PASS; accept 40 PASS / 2 FAIL; 3B 30 PASS / 6 FAIL. PASS counts line up with the per-scenario check counts (tip 8, fork 13, sparsify 7, ssd 8).
- The two accept FAILs are the only mismatches and are both expectation defects, not data-integrity defects: all 12 tip restores were bit-exact against fresh-prefill baselines, `tokens_reused` matched exactly, and the B-mini dedup count (5) matched exactly.
- Archive integrity: UTF-8 without BOM (first bytes `t e s t`, no EF BB BF); SHA256 recorded above.

## Issues / concerns

- The task's acceptance gate (all three modes exit 0, no FAIL in archives, 3B ssd PASS) is not met, and cannot be met by transcription alone; the plan (not just the brief) carries these expectations.
- Both findings should be adjudicated before the stage-2 exit evidence is declared green; the harness code itself is complete and faithful to the brief.

---

# Fix round 1/5: controller rulings applied

Status: DONE

Both BLOCKED items were ruled plan arithmetic defects and the plan text was patched to match. The fixes were applied exactly as specified in the rulings.

## Changes

1. `run_accept`: after the A prefill, added `anchor_bytes = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)` next to `seq_bytes`. Block count assertion is now `10` with label `"accept: shared trunk stored once (6+2+2 blocks)"`. The bytes assertion is now `check(tree.stats().bytes_ram + tree.stats().bytes_disk < 2 * ((int64_t) seq_bytes + (int64_t) anchor_bytes), "accept: stored bytes below two full copies (dedup)")`.
2. `scenario_ssd`: `kv_tree_io_llama io(ctx, 0)` stays in place; the blocks `tokens` declaration, `seq_rm`, prefill, the two measurements (`seq_bytes = range(0, 1536)`, `anchor_bytes = PARTIAL_ONLY` after the prefill), `cfg.disk_limit = 4 * (seq_bytes + anchor_bytes)` and the `kv_tree tree(cfg);` construction now sit between temp-dir setup and the park check, in the ruling's order.

## Commands and results (all re-run; every mode exit 0)

- Build: `BUILD_EXIT=0` (only the pre-existing C4297 warning in `src/llama.cpp`).
- logic: 18/18 PASS, exit 0.
- model 2B: 36/36 PASS, 0 FAIL, exit 0 -> archive rewritten.
- accept 2B: **42/42 PASS**, 0 FAIL, exit 0 -> archive rewritten. Key lines:
  ```
  [t32-tree] accept: tokens reused                               PASS (got 49152, want 49152)
  [t32-tree] accept: shared trunk stored once (6+2+2 blocks)     PASS (got 10, want 10)
  [t32-tree] accept: stored bytes below two full copies (dedup)  PASS
  [kv-tree] blocks: 10 ram, 0 disk, 103443240 bytes ram, 0 bytes disk
  [kv-tree] anchors: 2 ram, 0 disk, 2 added, 0 skipped
  [kv-tree] restore: 12 calls, 12 hits, 0 miss, 49152 tokens reused
  ```
  (B-mini: 5 blocks (1+4), 4/4 restores at 1024, tokens match; all six A/B rounds bit-exact.)
- 3B control: **36/36 PASS**, 0 FAIL, exit 0 (tip 8/8, fork 13/13, sparsify 7/7, ssd 8/8), log at `<user>\AppData\Local\Temp\opencode\t32-stage2-3b-control.txt`. ssd dump: `blocks: 0 ram, 3 disk, 113286624 bytes disk` = 3 x 18,881,400 blocks + 56,642,424 full-KV anchor, with `disk_limit = 4 x 113,286,624` (= 4x(seq_bytes + anchor_bytes), measured after the prefill).

## Archives (rewritten with the final runs, UTF-8 no BOM, first bytes `t e s t`)

- `D:\LLM\Backend\v100-collab\artifacts\t32-stage2-model.txt` - 7314 B, SHA256 `DCC34B6FD7A43E8CC1B6B4259C0F6B5CD9FC2378E144DB3B569721443B4807EF`
- `D:\LLM\Backend\v100-collab\artifacts\t32-stage2-accept.txt` - 5915 B, SHA256 `7D831A2427E13B08D34A7E9B9E4FB73E2E69374876A93492A8C889458CF6BCF2`

## Channel docs updated

- `TASKS\T32-agent-session-reuse.md`: appended the brief's exact `## Result (stage 2)` block (now line 613+).
- `RESULTS.md`: appended `## T32 阶段 2: ...` section with the final numbers (logic 18/18, model 36/36, accept 42/42, 3B 36/36, 10-block/49152/103,443,240 breakdown, 5-block B-mini) plus the plan-errata line.
- `STATUS.md`: appended the stage-2 bullet (12 commits, not pushed/deployed, all four runs green, archive paths).
- `artifacts\t32-tree-storage-design.md`: status line replaced with `状态: 阶段 0-2 已完成 (阶段 2 树模块 + harness, 分支 \`t32-stage2\`, 未合并/未 push); 阶段 3-4 未开始`.
All channel writes used .NET `File.ReadAllText`/`WriteAllText` with `UTF8Encoding($false)` (except the design status line via the exact old/new edit); tails verified with the read tool (no mojibake, original content intact).

## Commit

`a9d5d9595` on `t32-stage2`: `tests : add kv tree acceptance harness` (1 file, +84/-1; body trailer `Assisted-by: opencode`). Working tree clean afterwards; channel docs are outside the repo and not committed. `git rev-list --count 52b7bf7de..HEAD` = 12 (11 stage-2 commits from Tasks 1-6 + this one).

## Self-review

- Diff vs rulings: both replacement regions match the controller's snippets; no other code touched; ASCII only (0 bytes > 127 in the added text); no `src/` edits; no push.
- Cosmetic only: the now-overwritten `cfg.disk_limit = 64 << 20;` line remains in `scenario_ssd` before the measured assignment (kept because the ruling's replacement region started at `kv_tree tree(cfg);`).
- Archives contain the final green runs only; the earlier BLOCKED-run evidence remains in this report (lines above).

## Concerns

- None new. All work used the small models (2B for model/accept, 3B for the control); the 27B was not run.
