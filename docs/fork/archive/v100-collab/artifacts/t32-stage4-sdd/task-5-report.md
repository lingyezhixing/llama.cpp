# Task 5 Report: long soaks + full regression + archive + channel docs (bookkeeping half)

Status: DONE_WITH_CONCERNS

Summary: Steps 1-3 had already run in a previous invocation (30-min soak green, full regression
green); the 60-min run was cancelled by the user and skipped by ruling. This session verified all
evidence and the already-made archive, completed the four channel-doc updates, and wrote this
report. No soaks or regressions were re-run; no commits, no tidy/merge (Step 5 deferred by
controller ruling).

## Step 1: 30-min acceptance soak (already run; evidence verified here)

Command (from task-5-brief.md):

```
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode soak -Minutes 30 | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-soak-30.txt'
```

Result lines, verbatim from `t32-stage4-soak-30.txt`:

```
SOAK METRICS rounds=1703 parked=1194 restored=432 rebuilt=336 failed=0 evict_refused=96 diskmax=536460856 cmp_ok=340 cmp_bad=0 rss_mb=1902->1894 handles=245->271 wr_mb=87767 rd_mb=25003
SOAK RESTART files_before=79 cleared_lines=1 files_after=0
RESULT soak: 0 failure(s)
```

(14 PASS lines; `cmp_ok=340 cmp_bad=0` = 340 sampled requests were bitwise identical to a full
prefill, `failed=0` = no failures in the server log.)

Server-log cross-check performed in this session on the archived log:

- `t32-stage4-logs\srv-soak-30min-err.txt` (2103314 B, 17928 lines, runs 0.00.000 -> 30.03.615):
  parked=1194, restored=432, rebuilt=336, `failed to`=0 - matches the SOAK METRICS line exactly.
- `t32-stage4-logs\srv-soak-restart-30min-err.txt` contains `[kv-tree] cleared 79 stale files`,
  matching `files_before=79`.

## Step 2: full regression (already run; evidence verified here)

Commands (cuda1, from task-5-brief.md; `$exe` = `build\bin\Release\test-t32-tree.exe`):

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
& $exe --mode logic 2>&1 | Tee-Object "$art\t32-stage4-logic.txt" | Select-Object -Last 1
& $exe -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model  --ram-mib 4096 2>&1 | Tee-Object "$art\t32-stage4-model.txt" | Select-Object -Last 1
& $exe -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode accept --ram-mib 4096 2>&1 | Tee-Object "$art\t32-stage4-accept.txt" | Select-Object -Last 1
$env:T32_RAM_MIB='133'
foreach ($m in @('ab','overlap','b','b3','neg','heal','ref')) {
    & powershell -ExecutionPolicy Bypass -File "$art\t32-stage3-ab.ps1" -Mode $m 2>&1 | Tee-Object "$art\t32-stage4-$m.txt" | Select-String 'RESULT|FAIL'
}
$env:T32_RAM_MIB=''
```

Results (verbatim result lines; this session counted 0 FAIL via grep in each file):

```
t32-stage4-logic.txt   : 0 FAIL (per-check PASS lines; no aggregate RESULT line)
t32-stage4-model.txt   : 0 FAIL (per-check PASS lines; no aggregate RESULT line)
t32-stage4-accept.txt  : 0 FAIL (per-check PASS lines; no aggregate RESULT line)
t32-stage4-ab.txt      : RESULT ab: 0 failure(s)
t32-stage4-overlap.txt : RESULT overlap: 0 failure(s)
t32-stage4-b.txt       : RESULT b: 0 failure(s)
t32-stage4-b3.txt      : RESULT b3: 0 failure(s)      [B3 METRICS parked=11 restored=0 miss=6 failed_captures=0]
t32-stage4-neg.txt     : RESULT neg: 0 failure(s)     [SSD read failure visible; request still succeeded]
t32-stage4-heal.txt    : RESULT heal: 0 failure(s)    [HEAL METRICS captured=1 stored=True notstored=0 failed=0 missed=0 parked=7 restored=2]
t32-stage4-ref.txt     : RESULT ref: 0 failure(s)
```

## Step 3: 60-min final soak - SKIPPED by user ruling

- `Test-Path artifacts\t32-stage4-soak-60.txt` = False; no 60-min evidence exists.
- Ruling (SDD ledger `progress.md`, Task 5): same workload as the 30-min run; 5-min smoke +
  30-min soak + full regression are the acceptance evidence. Cost if wrong: shorter soak horizon.
- The cancelled attempt did start a server: temp `srv-soak-err.txt` was rewritten at 18:41:47 with
  a 404-byte startup fragment, and `slot-save\` was recreated in the temp dir
  (`<TEMP>\v100\t32-stage3\`).

## Archive (Step 4a): verified, no re-copy

The 30-min and regression logs were already copied by the previous invocation before its
cancellation; because the temp source `srv-soak-err.txt` is now overwritten (see anomaly 1), the
copy was skipped per the brief and the existing archive was kept and verified instead.

`D:\LLM\Backend\v100-collab\artifacts\t32-stage4-logs\`:

| file | bytes | content |
|---|---|---|
| srv-soak-30min-err.txt | 2103314 | 30-min soak server log (count-verified) |
| srv-soak-30min-out.txt | 0 | stdout (harness redirects stderr only) |
| srv-soak-restart-30min-err.txt | 1830 | 30-min restart, `cleared 79 stale files` |
| srv-soak-restart-30min-out.txt | 0 | |
| srv-soak-err.txt | 362903 | existing 5-min smoke log (kept, not deleted) |
| srv-soak-out.txt | 0 | |
| srv-soak-restart-err.txt | 1830 | existing 5-min smoke restart log |
| srv-soak-restart-out.txt | 0 | |
| srv-ab-err.txt | 204956 | regression `ab` |
| srv-overlap-err.txt | 10718 | regression `overlap` |
| srv-b-err.txt | 108343 | regression `b` |
| srv-b3-err.txt | 25357 | regression `b3` |
| srv-neg-err.txt | 40644 | regression `neg` |
| srv-heal-err.txt | 16807 | regression `heal` |
| srv-ref-err.txt | 24791 | regression `ref` |
| srv-{ab,overlap,b,b3,neg,heal,ref}-out.txt | 0 each | stdout |

Hash checks (this session): all 7 mode err logs MATCH their still-live temp sources (SHA-256);
`srv-soak-restart-30min-err.txt` MATCHes the still-live temp `srv-soak-restart-err.txt`.
The two 30-min copies were made before 18:41:47 (source mtimes 18:35:53 / 18:35:59).

## Channel docs (Step 4b), append-only, UTF-8 without BOM

1. `artifacts\t32-tree-storage-design.md`:
   - line 3 status -> `状态: 阶段 0-4 已完成 (阶段 4: soak 长跑 + D12/D13, 分支 t32-stage4 未合并/未 push); 磨损优化/持久化待后续`
   - §9 D12 bullet appended: stage-4 implemented; rebuilt checkpoints use tail semantics
     (`pos_min = pos_max = pos - 1`) when the context cannot partial seq_rm, PART keeps
     `pos_min = 0` (D18, see `artifacts/t32-tree-plan-stage4.md`).
2. `RESULTS.md` (147764 -> 149753 B): appended `## T32 阶段 4: 长跑 soak + D12/D13` with the
   30-min metrics line, the 5-min smoke metrics line, the regression list, D12/D13 (and
   D17/D18/D20/D22/D23) implemented, D14 wear deferral, D15 60-min skip, archive paths.
3. `STATUS.md` (39631 -> 40408 B): appended 3-line stage-4 completion note (same facts).
4. `TASKS\T32-agent-session-reuse.md` (55160 -> 56272 B): appended `## Result (阶段 4, ...)` with
   deliverables, evidence paths, and the D14-D23 reference.

Verification: each append increased the file by exactly the fragment byte length; BOM checks
remain noBOM; the edited/untouched neighborhoods were re-read and match the intended text.
Design doc `noBOM` 14063 B (was 13814 B; status line replaced + D12 sentence appended).

## Anomalies

1. Temp `srv-soak-err.txt` (the 30-min soak's main log source) was overwritten at 18:41:47 by the
   aborted 60-min attempt (404 bytes: startup lines only, plus a fresh `slot-save\` dir). The
   archived `-30min` copy has source mtime 18:35:53 (before the overwrite) and was re-verified by
   exact count match vs the metrics line, so it is trusted; it can no longer be hash-compared to
   its source.
2. All `*-out.txt` logs are 0 bytes by design (`Start-Srv` redirects stderr only).
3. The cancelled invocation left no task-5 report; this report is the complete one.

## Out of scope (by ruling)

- Step 5 (tidy merge + branch cleanup): deferred. Branch `t32-stage4` (head `c0619da14`) kept;
  master `daf4186d3` unchanged; origin/master `daf4186d3`; production still `ba41cccec`.
- No git commits were made in this session; `git status --short` is clean.

```
branch:       t32-stage4
HEAD:         c0619da14 server : use tail semantics for rebuilt checkpoints on recurrent contexts
master:       daf4186d3
origin/master: daf4186d3
status:       clean
```
