# Task 4 Report: fork 验收模式 + 全回归 + 短 soak + 文档 (T32 阶段 5)

日期: 2026-09-27 | 分支: `t32-stage5` @ `04954b468` | 设备: cuda1 | 状态: DONE

无 git 提交 (artifact-only); `git status --short` 为空 (src 未动, 分支保持 `04954b468`).

## 1. fork 模式 (加入 `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1`)

```powershell
        'fork' {
            $dir = "$OutDir\tree-fork"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            # no message delimiters -> no checkpoint guesses -> the first fork is a restore miss
            # ctx 32768: b1 is ~20500 tokens (the brief's 16384 rejected it); anchor_step stays 32768 so no regular anchor
            # sim 0: the stock slot-similarity shortcut would reuse the slot in-place and skip the tree (f_keep >= 0.5)
            $p = Start-Srv $true 512 $dir $true 32768 1 '' 2048 $false 32768 -sim 0
            Build-Sys
            $base = Filler 'shared' 8192
            $a1 = Filler 'branchA1' 8192
            $a2 = Filler 'branchA2' 4096
            $b1 = $base + $a1 + $a2                  # fork 1 at ~len(base)
            $b2 = $base + (Filler 'branchB' 2048)    # diverges at fork 1
            $b3 = $base + $a1 + (Filler 'branchC' 2048)  # diverges at ~fork 1 + 8192 (>= fork_step)

            $hashes = @{}
            foreach ($pass in @(@($true,'tree'), @($false,'full'))) {
                $i = 0
                foreach ($q in @($b1, $b2, $b2, $b3, $b3)) {
                    $i++
                    $r = Req $q $pass[0]
                    $hashes["$($pass[1])/fork/$i"] = (ContentHash $r.content)
                    Write-Output ("[fork/$($pass[1])] req ${i}: prompt_n=$($r.timings.prompt_n) cached=$($r.tokens_cached)")
                }
            }
            Stop-Srv

            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $captured = @($log | Select-String 'captured heal anchor at (\d+)' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value })
            $restored = @($log | Select-String 'kv tree: restored (\d+) tokens' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value })
            Write-Output "FORK METRICS captured=[$($captured -join ',')] restored=[$($restored -join ',')]"
            Assert ($captured.Count -ge 2) 'fork: two fork anchors captured (miss heal + second fork)'
            Assert ($restored -contains $captured[0]) 'fork: the miss-heal anchor is reused'
            Assert ($restored -contains $captured[1]) 'fork: the second fork anchor is reused'
            for ($i = 1; $i -le 5; $i++) {
                Assert ($hashes["tree/fork/$i"] -eq $hashes["full/fork/$i"]) "fork: request $i identical (tree vs full prefill)"
            }
        }
```

Supporting `Start-Srv` change (only for this mode; existing behavior preserved):

```powershell
function Start-Srv(..., $ctx = 24576, $log_tag = '', $fork_step = 8192, $sim = -1.0) {
    ...
    if ($sim -ge 0.0) { $sargs += @('--slot-prompt-similarity', "$sim") }
```

## 2. fork 结果

Final run (`t32-stage5-fork.txt`):

```
[fork/tree] req 1: prompt_n=20501 cached=20548
[fork/tree] req 2: prompt_n=10258 cached=10305
[fork/tree] req 3: prompt_n=2066 cached=10305
[fork/tree] req 4: prompt_n=10262 cached=18501
[fork/tree] req 5: prompt_n=2070 cached=18501
[fork/full] req 1..5: full prefill each (20501/10258/10258/18454/18454)
FORK METRICS captured=[8192,16384] restored=[8192,8192,16384]
PASS  fork: two fork anchors captured (miss heal + second fork)
PASS  fork: the miss-heal anchor is reused
PASS  fork: the second fork anchor is reused
PASS  fork: request 1..5 identical (tree vs full prefill)
RESULT fork: 0 failure(s)
```

Behavior: req2 (b2 after b1) is a restore miss -> full prefill -> heal capture @8192; req3 (b2 again) restores @8192 (prompt_n=2066 = 10258-8192, heal 10258 within fork_step -> skipped); req4 (b3) restores @8192 and replays to 16384, capturing the second fork anchor (16384-8192 = 8192 >= fork_step); req5 (b3 again) restores @16384 (prompt_n=2070).

## 3. Calibration fixes (scenario only; thresholds/assertions unchanged)

1. `-c 16384` (brief value) rejects b1: preflight run `t32-stage5-fork-preflight.txt` shows
   `request (20501 tokens) exceeds the available context size (16384 tokens)`, HTTP 400 `exceed_context_size`.
   Fix: ctx 32768 (anchor_step stays 32768 -> still no regular checkpoint anchors).
2. Default `--slot-prompt-similarity 0.1` made reqs 3-5 bypass the tree: `selected slot by LCP similarity ... f_keep = ...`
   with f_keep >= 0.5 reuses the slot in place, so no park/restore ran (`FORK METRICS captured=[8192] restored=[]`,
   first post-brief run). Fix: fork mode passes `--slot-prompt-similarity 0`, which forces the LRU/tree path for every request.
   With the tree path restored, the brief's assertions all pass unchanged.

## 4. Regression (cuda1)

Harness:
- `t32-stage5-logic.txt`: 74 PASS / 0 FAIL verdicts
- `t32-stage5-model.txt`: 64 PASS / 0 FAIL
- `t32-stage5-accept.txt`: 42 PASS / 0 FAIL

Script (`T32_RAM_MIB=133` for ab only):
- ab: `AB METRICS ram_mib=133 parked=23 restored=12 miss=0 rammax=138760464 diskmax=497257704` / `RESULT ab: 0 failure(s)`
- overlap: `OVL METRICS parked=0 restored=0 miss=0` / `RESULT overlap: 0 failure(s)`
- b: `B METRICS restored=12 ref4_lines=2` / `RESULT b: 0 failure(s)`
- b3: `B3 METRICS parked=11 restored=0 miss=6 failed_captures=0` / `RESULT b3: 0 failure(s)`
- neg: `NEG METRICS strict=2 wide=2` / `RESULT neg: 0 failure(s)`
- heal: `HEAL METRICS captured=1 stored=True notstored=0 failed=0 missed=0 parked=7 restored=2` / `RESULT heal: 0 failure(s)`
- ref: `RESULT ref: 0 failure(s)`
- fork: `FORK METRICS captured=[8192,16384] restored=[8192,8192,16384]` / `RESULT fork: 0 failure(s)`

All numbers match stage 4 records.

## 5. 5 min soak (`t32-stage5-soak.txt`)

```
SOAK METRICS rounds=302 parked=219 restored=254 rebuilt=143 failed=0 evict_refused=10 diskmax=536394020 cmp_ok=60 cmp_bad=0 rss_mb=1902->180 handles=244->254 wr_mb=10844 rd_mb=7786
SOAK RESTART files_before=61 cleared_lines=1 files_after=0
RESULT soak: 0 failure(s)
```

## 6. Archive

`artifacts\t32-stage5-logs\`: srv-{ab,overlap,b,b3,neg,heal,ref,fork,soak,soak-restart}-{err,out}.txt (copied from
`<TEMP>\v100\t32-stage3\`; out files are empty by design, stderr carries the logs).
Evidence txt: `t32-stage5-{logic,model,accept,ab,overlap,b,b3,neg,heal,ref,fork,soak}.txt` + `t32-stage5-fork-preflight.txt`.

## 7. Docs

- `artifacts\t32-tree-storage-design.md`: status line -> `阶段 0-5 已完成 (... 分支 t32-stage4/t32-stage5 未合并/未 push)`;
  §9 added D25 (双档间距), D26 (heal-on-miss), D27 (删除 `promote_prune` 死代码).
- `v100-collab\RESULTS.md`: appended `## T32 阶段 5: 检查点命名 + 分叉锚点 (D25-D30)` section.
- `v100-collab\STATUS.md`: appended stage 5 bullet.
- `TASKS\T32-agent-session-reuse.md`: appended `## Result (阶段 5, ...)`.
- All doc writes via `[System.IO.File]::WriteAllText` + `UTF8Encoding($false)`; verified: no BOM, CRLF preserved.

## 8. Anomalies / notes

- The harness has no final summary line; verdicts were counted by uppercase `PASS`/`FAIL` (74/64/42 match the expected
  logic 74/0, model 64/0, accept 0 FAIL). `2>&1` in PS 5.1 appends child stderr (module dumps, expected negative-path
  `failed to ...` lines) after stdout, so `Select-Object -Last 1` in the brief's command prints a dump line, not a verdict.
- b3 `miss=6` is the same as stage 4 (pure-attention D11 behavior), not a regression.
- Soak `evict_refused=10` (budget rollback) is within the `<= 10% of rounds` bound (302 rounds).
- No src/ edits, no commits, no merge/push; branch `t32-stage5` left for the user together with stage 4 (D30).
