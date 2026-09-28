# Final fix wave report: T32 stage 3 (branch `t32-stage3`)

Date: 2026-09-27
Base: `b936d687f` (final whole-branch review verdict "With fixes": 3 Important, 13 Minor; no Critical)
Fix-wave head: `3416ea622`
Scope: F1, F2, F3 (D12, docs only), selected minors, plus the acceptance-script heal/b3 corrections. No push, no deploy, production dir untouched.

## Commits

| commit | files | message |
|---|---|---|
| `2edf4a6df` | `tools/server/server-kv-tree.{h,cpp}` (43+/5-) | `server : fix kv tree heal capture and reporting` |
| `5d0aee637` | `tests/test-t32-tree.cpp` (59+) | `tests : add kv tree capture-inside-chunk tests` |
| `3416ea622` | `tools/server/server-context.cpp` (12+/6-) | `server : clean up kv tree integration logs and slot lifecycle` |

All three carry `Assisted-by: opencode`. Existing commits not amended.

## F1 (Important): heal capture must work off chunk boundaries

Change: `tools/server/server-kv-tree.cpp:318-352` (`kv_tree::capture_anchor`).

- When `containing_block(pos, &chain)` returns 0 (the request's truncated tail chunk hash cannot match a full stored chunk), fall back to the stored block that covers `pos`: scan the `blocks_at` bucket just below `pos` (same range rule as `containing_block`), then require `pos - b.pos0 <= b.tokens.size()` and `std::equal(b.tokens[0..pos-b.pos0), tokens[b.pos0..pos))` before using it as the anchor's `blk_hash`.
- If no covering/matching block exists, keep the old fprintf and refuse.

TDD (test written and watched to fail first):

- Test: `tests/test-t32-tree.cpp:223-285` (`run_logic_capture`, called from `run_logic()` at `:287`).
  - park 1536 tokens (3 fake blocks), negative first: `capture_anchor(io, nullptr, tokens, 1400)` with the tokens truncated to the capture point (mirrors `slot.prompt.tokens` at heal time) and `io.pos_max = 1399`.
  - a request forking right after 1400 must `restore` at `C = 1400`.
  - boundary negative: extended 1700-token sequence, capture at 1700 must return false (no stored block covers it) and `anchors_added` must not change.
- RED (first build, before the fallback):
  ```
  [t32-tree] capture: heal inside the tail block                      FAIL
  [t32-tree] capture: the anchor was stored                           FAIL (got 1, want 2)
  [t32-tree] capture: fork restores at the stored anchor              FAIL (got -1, want 1400)
  [t32-tree] capture: the anchor state was written back               FAIL (got 0, want 1)
  [t32-tree] capture: nothing stored on refusal                       FAIL (got 1, want 2)
  EXIT: 1
  [kv-tree] capture refused at 1400: no chain block contains this position
  [kv-tree] capture refused at 1700: no chain block contains this position
  ```
- GREEN (after the fallback, whole logic suite): `logic PASS=46 FAIL=0`, exit 0. New checks:
  ```
  [t32-tree] capture: heal inside the tail block                      PASS
  [t32-tree] capture: the anchor was stored                           PASS (got 2, want 2)
  [t32-tree] capture: fork restores at the stored anchor              PASS (got 1400, want 1400)
  [t32-tree] capture: no heal needed after the restore                PASS (got -1, want -1)
  [t32-tree] capture: the anchor state was written back               PASS (got 1, want 1)
  [t32-tree] capture: refused with no stored block                    PASS
  [t32-tree] capture: nothing stored on refusal                       PASS (got 2, want 2)
  ```
- Model regression (2B, device 0): 43/43 PASS, exit 0 (full log: `<user>\AppData\Local\Temp\opencode\t32-stage3-model2b-fixwave.txt`); `fork: heal capture at the fork point` and `heal: A' restores at the self-healed anchor` still pass.
- Logic log: `<user>\AppData\Local\Temp\opencode\t32-stage3-logic-fixwave.txt`.

## F2 (Important): "captured" must mean stored

Change:

- `tools/server/server-kv-tree.cpp:370-374`: anchor-spacing skip now logs `[kv-tree] capture skipped at %d: within anchor_step`, counts `anchors_skipped`, and returns **false**. Refusal paths keep their fprintf and false. An anchor that already exists at `(blk,pos)` still returns true (it is heated; the heal anchor is genuinely available).
- `tools/server/server-context.cpp:3594-3598`: `SLT_INF "kv tree: captured heal anchor at %d"` only on true; otherwise `SLT_TRC "kv tree: heal anchor at %d not stored"` (was a WRN on a path that is now benign by design).

Root cause of the skipped 1024 capture (investigated, comparison is correct):

- The acceptance script ran `--tree-anchor-step 4096` (hard-coded in `Start-Srv`), not 512. On the first heal park a MESSAGE anchor at 487 was adopted (checkpoint candidate); the next heal pos was 1024, so `prev = 487` and `1024 - 487 = 537 < 4096` -> spacing skip. The module's greedy rule (`pos - prev >= step`) matches the spec; no comparison bug. The review text assumed step 512, which did not match the script revision.
- The 1024 capture was a false success: the module returned true, so the server logged "captured heal anchor at 1024" while no `@1024` anchor existed in any dump, and a later request re-restored at C=487.

Acceptance changes (`D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1`, channel artifact, not git):

- `Start-Srv` gained an `$anchor_step` parameter (default 4096); heal mode now starts with step 512 so `537 >= 512` forces a real store.
- Heal assertions now require a genuinely stored anchor: parse the captured pos from the INF line and assert the `--tree-debug` dump contains `anchor <hash>@<pos> kind=2` (plus kept the capture-count and no-failed-capture checks).
- Re-run `-Mode heal` (tree pass first), archived to `artifacts\t32-stage3-heal.txt`:
  ```
  HEAL ANCHOR pos=1024 stored_in_dump=True
  HEAL METRICS captured=1 stored=True notstored=0 failed=0 missed=0 parked=7 restored=2
  PASS  heal: exactly one heal capture
  PASS  heal: the captured anchor is genuinely stored (dump shows @pos kind=2)
  PASS  heal: no failed captures
  PASS  heal: request 1..4 identical (tree vs full prefill)
  RESULT heal: 0 failure(s)
  ```
- Server evidence (`...\t32-stage3\srv-heal-err.txt`): `10 added, 3 skipped` after the capture and dump line
  `[kv-tree]   anchor 3209c939e15d941d@1024 kind=2 ref=2 heat=0 ram`.
- Observable improvement: heal request 4 (second `$h$tF`) now restores 2443 tokens (`heal = 2447`) instead of a full prefill; before the fix it re-restored at C=487.

## F3 (Important, controller ruling D12): checkpoint-table rebuild deferred, documented

No code behavior implemented (as ruled). Documentation only:

- `tools/server/server-kv-tree.h:146`: `anchors` marked `// reserved for the stage-4 checkpoint-table rebuild (D12)`.
- `artifacts/t32-tree-plan-stage3.md` (设计决定 section): new D12 bullet after D9 - `prompt.checkpoints` stays cleared, `res.anchors` reserved; correctness preserved by the full-reprocess fallback; rebuild is the stage-4 fast path for reasoning-rollback/SWA trims.
- `artifacts/t32-tree-storage-design.md` (section 9): D12 note next to the D2 gap - section 3.2 step 3 deferred to stage 4.
- `RESULTS.md` / `STATUS.md`: fix-wave line with the D12 ruling (UTF-8, no BOM, append-only).

## Selected minors

| item | change | location |
|---|---|---|
| park success log | INF `parked ...` only when `ok`; else `SLT_TRC "kv tree: park did not store (see module log)"` | `server-context.cpp:337-345` |
| `tree_heal` lifecycle | `tree_heal = -1;` added at the end of `prompt_clear()` | `server-context.cpp:417` |
| `write_disk` errors | check `create_directories` `ec` -> fprintf + `disk_errors++` + false; fopen failure -> fprintf + `disk_errors++` + false (was silent) | `server-kv-tree.cpp:472-491` |
| idle TRC wording | `"saving idle slot to prompt cache"` -> `"saving idle slot state"` (covers both tree and prompt-cache paths) | `server-context.cpp:2572` |
| (D10) comment | `// no point restoring when the request will not reuse the prompt` (behavior, no internal id) | `server-context.cpp:1764` |
| acceptance b3 | `$sys = $sys3` at the top of the b3 branch so `RunTurns` actually prepends the 256-token shared system prefix (`$sys3` was dead) | `t32-stage3-ab.ps1` b3 branch |

b3 re-run, archived to `artifacts\t32-stage3-b3.txt`:
```
B3 METRICS parked=11 restored=0 miss=6 failed_captures=0
PASS b3: tree still parks pure-attention content (D11: no fork reuse)
PASS b3: no failed captures
PASS equal: P/Q round 1..3 (tree vs full prefill)
RESULT b3: 0 failure(s)
```
Metrics are unchanged vs the old run because the stock VRAM LCP already reused the shared prefix; the scenario now matches its intent (real 256-token shared prefix, D11 structural miss remains).

## Verification runs

| run | command | result |
|---|---|---|
| logic | `build_test_t32.cmd test-t32-tree` then `test-t32-tree.exe --mode logic` | 46/46 PASS, exit 0 |
| model 2B | `$env:CUDA_VISIBLE_DEVICES='0'; test-t32-tree.exe -m Qwen3.5-2B-UD-Q4_K_XL.gguf --mode model` | 43/43 PASS, exit 0 |
| server build | delete `llama-server-impl.dll`, `build_server.cmd` | linked clean |
| heal | `t32-stage3-ab.ps1 -Mode heal` | `RESULT heal: 0 failure(s)`; stored=True @1024 kind=2 |
| b3 | `t32-stage3-ab.ps1 -Mode b3` | `RESULT b3: 0 failure(s)` |

Other acceptance modes (calib/ab/overlap/b/neg/ref) were NOT re-run in this wave (scope); their archived numbers belong to `b936d687f`. Note the F1 fallback does change calib/neg behavior (heal captures at non-boundary pos like 16912 now store an anchor) - that path is covered by the new harness test, not by a re-run of those modes.

## Concerns / residuals

1. Heal at prompt end is never captured (pre-existing, not a regression): in heal request 4 the restore point leaves only 4 tokens, so the prompt completes in the same update pass and the capture block never sees `n_tokens == hp`; `tree_heal` is dropped when the slot finishes. Cost is a few replayed tokens per fork; D6 assumed a following iteration. Candidate stage-4 follow-up.
2. `disk_errors` can double-count a write failure (`write_disk` increments, and `demote_block` / `demote_anchor` increment again after the false return). Diagnostic counter only; not asserted to an exact value in tests.
3. `artifacts\t32-stage3-accept.txt` (the 8-mode summary) was not regenerated; its heal wording/counts still describe `b936d687f`. `t32-stage3-heal.txt` / `t32-stage3-b3.txt` and RESULTS/STATUS carry the corrected fix-wave text.
4. F2 strictly: `capture_anchor` still returns true for an already-existing `(blk,pos)` anchor (heat bump, no new store). This is intentional: the anchor is present and usable, so the heal log is truthful; only spacing skip and refusal are non-store cases.

## Doc updates (all UTF-8, no BOM)

- `D:\LLM\Backend\v100-collab\artifacts\t32-tree-plan-stage3.md`: D12 bullet.
- `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md`: D12 note (section 9).
- `D:\LLM\Backend\v100-collab\RESULTS.md`: fix-wave correction + re-run results (heal claim corrected: old "captured@1024" was a false success).
- `D:\LLM\Backend\v100-collab\STATUS.md`: fix-wave summary line.
- `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1`, `t32-stage3-heal.txt`, `t32-stage3-b3.txt`: updated as above.

## Archive refresh at the fix head (follow-up pass)

All remaining acceptance modes were re-run at `3416ea622` (same binary as the fix-wave runs; no repo code touched). Safeguards unchanged: bounded /health polling, `-TimeoutSec` on every request, `Stop-Srv` in `finally` (own PID only).

| mode | command | result | key metrics |
|---|---|---|---|
| calib | `-Mode calib` | `RESULT calib: 0 failure(s)` | total 417,408,732 B (unchanged) |
| ab | `-Mode ab` with `T32_RAM_MIB=133` | `RESULT ab: 0 failure(s)` | parked=23 restored=12 miss=0 rammax=138,760,464 diskmax=497,257,704 (unchanged) |
| overlap | `-Mode overlap` | `RESULT overlap: 0 failure(s)` | parked=0 restored=0 miss=0 (unchanged) |
| b | `-Mode b` | `RESULT b: 0 failure(s)` | restored=12 ref4_lines=2 (unchanged) |
| neg | `-Mode neg` | `RESULT neg: 0 failure(s)` | 65 block files removed, strict=2 wide=2 (unchanged) |
| ref (x2) | `-Mode ref` -> `t32-stage3-ref.txt`, `t32-stage3-ab-ref2.txt` | `RESULT ref: 0 failure(s)` both | 30/30 hash lines identical |
| heal | kept from the fix wave | `RESULT heal: 0 failure(s)` | captured=1 stored=True @1024 kind=2 |
| b3 | kept from the fix wave | `RESULT b3: 0 failure(s)` | parked=11 restored=0 miss=6 |

Notes:

- ab must run with `T32_RAM_MIB=133` to match the previous archive; a first refresh run at the script default 96 MiB was discarded (script defaults to 96 when the env var is unset).
- Behavior-only change confirmed in the refreshed server logs: calib/neg used to log `capture refused at 16912: no chain block contains this position` (old head); at the fix head the same capture resolves the covering stored block via the F1 fallback and is then refused by the spacing rule: `[kv-tree] capture skipped at 16912: within anchor_step` (16912-16405 = 507 < 4096). No anchor is stored, so anchor sets and every metric are unchanged. A fallback store with a step that permits it is covered by the logic harness test (`capture@1400`, prev=-1) and by the heal acceptance (`@1024 kind=2`, step 512).
- Archives refreshed: `artifacts\t32-stage3-{calib,ab,overlap,b,neg,ref,ab-ref2}.txt` re-tee'd; `artifacts\t32-stage3-logs\srv-*.txt` copied from the fix-head run dir (including heal/b3); `artifacts\t32-stage3-accept.txt` regenerated (build `3416ea622`, all 8 modes, ref determinism 30/30).
- `RESULTS.md` / `STATUS.md`: append-only archive-refresh correction lines added (UTF-8, no BOM); the previous "not re-run" note in the fix-wave line is superseded by the refresh line.
- This also resolves report concern 3 (stale accept.txt).
