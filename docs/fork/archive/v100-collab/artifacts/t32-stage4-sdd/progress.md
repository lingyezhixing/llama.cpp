# SDD ledger — plan: D:\LLM\Backend\v100-collab\artifacts\t32-tree-plan-stage4.md

Spec: D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md (reachable; §9 stage 4 + D12/D13/D2 notes)
Branch: t32-stage4, base daf4186d3 (master), created 2026-09-27.
Environment: CUDA_VISIBLE_DEVICES=1; 2B model; build_test_t32.cmd / build_server.cmd; no push/deploy.

## Preflight scan (before Task 1)

### Cross-task pairs (shared file / interface)
| pair | produces -> consumes | finding |
|---|---|---|
| T1 -> T3 | stats_line() -> server aggregate log (3d) | clean (name/fields match) |
| T1 -> T4 | spacing/disk_errors fixes -> soak behavior | clean |
| T2 -> T4 | ctor wipe log `[kv-tree] cleared %zu stale files` -> soak restart grep `cleared \d+ stale files` | clean |
| T3 -> T4 | `kv tree: rebuilt %zu context checkpoints` -> soak/heal grep `rebuilt \d+ context checkpoints` | clean |
| T3 -> call site | prompt_restore_tree(tree, tokens, n_ckpt_max) -> server-context.cpp:1765 | clean (only caller) |
| T3 -> harness | kv_tree_restore::anchors type change -> path_result (line 423) reads no fields | clean |
| T4 -> T5 | -Minutes param + soak mode -> 30/60 min runs | clean |
| T1 + T2 -> run_logic() registration | both append calls to run_logic() | sequential execution; T2 appends after T1 |
| T3 + T4 -> t32-stage3-ab.ps1 | T3 edits heal asserts; T4 edits param/Start-Srv/soak | sequential; different sections |
| T5 -> T1..T4 | tidy merge + tree hash check | clean |

### Self-consistency per task
| task | check | finding |
|---|---|---|
| T1 | disk_errors expected 2 after fix (2 failed write_disk calls x 1) / 4 before | derived from code; discriminates |
| T1 | park with ram_limit=0: demote fails, evict refuses (pinned), rollback | derived from code; park=false expected |
| T2 | wipe test: remove_all clears files | clean |
| T3 | model SSD variant disk_limit = 1 MB | DEFECT: real 2B KV for 3072 tok is ~150 MB -> park refused. Plan edited to 1 GB |
| T4 | D12 rebuild during soak | DEFECT: without message delimiters no mid-prefix anchors -> C = tip only -> rebuilt = 0. Plan edited: soak uses ReqDelim + "User:" markers |
| T5 | tidy tree hash verification | clean |

### Preflight rulings
- Ruling: T3 SSD test `cfg2.disk_limit` 1 MB -> 1 GB (real 2B KV ~150 MB; park would be refused). Cost if wrong: test only.
- Ruling: soak uses message_delimiters (ReqDelim) + "User:" markers so checkpoint anchors exist in the shared prefix; otherwise D12 rebuild never triggers. Cost if wrong: rebuilt assert fails; smoke catches.
- Ruling: Task 1 lands as ONE commit (fixes + stats_line + tests) instead of the plan's two-commit split (splitting two functions in the same two files on Windows needs git add -p; the final tidy merge regroups commits anyway). Cost if wrong: none.

## Task log

BASE for Task 1: daf4186d3
Task 1: complete (commits daf4186d3..98c613325, review clean)
Task 1: minor (deferred): tests/test-t32-tree.cpp:374 "same-chain spacing still refuses" passes via containment, not spacing (plan-mandated; real coverage exists at :93/:146/:190/:228)
Task 1: minor (deferred): tests/test-t32-tree.cpp:285 fixed temp name t32-tree-diskerr could collide with leftovers

BASE for Task 2: 98c613325
Task 2: complete (commits 98c613325..ce03a50f8, review clean)
Task 2: minor (deferred): ctor wipe count includes files outside blocks/anchors (log over-reports; e.g. srv.txt)
Task 2: minor (deferred): recursive_directory_iterator uses the throwing increment (ec only on construction)
Task 2: minor (deferred): two check(true, ...) no-op assertions in run_logic_wipe (plan-mandated); fixed temp path t32-tree-wipe

BASE for Task 3: ce03a50f8
Task 3: fix round 1/5 (2 addressed, 0 open — double load_payload + missing accounting assert; commits 2575f41f0..569f05669)
Task 3: complete (commits ce03a50f8..569f05669, review clean)
Task 3: minor (deferred): tree_ops counts every slot pick, not every park+restore (plan-mandated cadence)
Task 3: minor (deferred): stats-line runtime evidence pending; soak (>=64 ops) will exercise it in Task 4/5
Task 3: Ruling: fix the plan-mandated double-load_payload defect (spec authority = correctness; reviewer showed permanent bytes_ram leak can eventually refuse all parks). Cost if wrong: one extra early-return in load_payload.
Task 3: note: implementer corrected the brief's test bug (suffix prefill slice with from=2048 decoded nothing) to prefill(ctx, 0, tokens, 2048, 512) per harness semantics; reviewer verified.

BASE for Task 4: 569f05669
Task 3: fix round 2/5 (1 addressed, 0 open — D12 rebuilt checkpoints aborted hybrid contexts; tail semantics for FULL/RS; commits 569f05669..c0619da14)
Task 3: complete (commits ce03a50f8..c0619da14, review clean, 2 fix rounds)
Task 3: Ruling: rebuilt checkpoints use tail semantics (pos_min=pos_max=pos-1) when the context cannot partial seq_rm (FULL/RS); keep pos_min=0 for PART. Cost if wrong: on hybrid the reasoning fast path only picks anchors before the rollback target (safe, slightly less optimal).
Task 3: note: controller stock repro (no --kv-tree, same 2 requests) does not abort (do_reset path), confirming the abort came from the rebuilt checkpoint being selected.
Task 4: concern resolved: 5-min soak reaches the deadline after Task 3 fix round 2 (293 rounds, 0 failures, was abort ~137). 2-min smoke: rebuilt=43, cmp 24/24, D13 restart pass.
Task 4: Ruling: ratify the script adjustments (soak --tree-anchor-step 512 for D12 reachability; evict_refused -eq 0 -> -le 20 for designed disk-full park refusals). Cost if wrong: slightly weaker assertions.
Task 4: no repo commit (artifact script only).

BASE for Task 5: c0619da14
Task 4: fix round 1/5 (2 addressed, 0 open — rate-based evict_refused assert; green 5-min evidence archived; no commits)
Task 4: complete (artifact-only, no commits, review clean after 1 fix round)
Task 4: minor (deferred): $ioEnd null -> handle/RSS bounds pass vacuously on crash; crash mid-loop skips SOAK METRICS/RESULT; files_after counts files only; diskmax=0 would pass
Task 5: Ruling: T5 stops before tidy/merge; merge/push deferred to the user's finishing choice (same as stage 3). Cost if wrong: master stays unmerged until the user picks.

BASE for Task 5: c0619da14
Task 5: Ruling: 60-min final soak skipped (user: same workload as the 30-min run; 5-min + 30-min + full regression are the acceptance evidence). Cost if wrong: shorter soak horizon; 30-min ran 1703 rounds with stable RSS/handles and 340/340 bitwise compares.
Task 5: evidence: 30-min soak green (rounds=1703 parked=1194 restored=432 rebuilt=336 failed=0 evict_refused=96 diskmax=536460856 cmp_ok=340 cmp_bad=0 rss 1902->1894 MB handles 245->271 wr=87767 MB rd=25003 MB; restart 79->0 files, cleared); regression green: logic/model/accept (0 FAIL) + ab/overlap/b/b3/neg/heal/ref (0 failures).
Task 5: subagent cancelled by the user during the 60-min run; bookkeeping (report + channel docs) dispatched fresh.
Final review (daf4186d3..c0619da14): With fixes. 1 Important: rebuilt checkpoints have no byte cap (PART contexts copy full-KV payloads; hybrid anchors are per-anchor large -> up to 32 copies per slot, unbudgeted). Deferred minors triaged: all acceptable for merge.
Ruling: one fix wave = (a) byte cap 256 MiB on rebuilt checkpoints, drop shallowest first; (b) load_payload(kv_tree_block&) transient guard for symmetry; (c) ctor wipe count via remove_all returns (fixes over-report + throwing increment). Cost if wrong: (a) fewer rebuilt checkpoints on large-anchor models (deepest kept); (b) none; (c) wipe log counts only blocks/anchors.
Ruling: plan D18 amended (tail semantics for FULL/RS) + D24 (byte cap) recorded in the plan file.
