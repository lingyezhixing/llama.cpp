# SDD ledger — plan: D:\LLM\Backend\v100-collab\artifacts\t32-tree-plan-stage5.md

Spec: D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md (§9) reachable.
Branch: t32-stage5, base 42ee7a6f7 (t32-stage4 head, stage 4 not merged), created 2026-09-27.
Environment: CUDA_VISIBLE_DEVICES=1; 2B; build_test_t32.cmd / build_server.cmd; no push/merge/deploy.

## Preflight scan (before Task 1)

### Cross-task pairs (shared file / interface)
| pair | produces -> consumes | finding |
|---|---|---|
| T1 -> T2 | cfg.fork_step (field + mapping) -> capture_anchor spacing | clean |
| T1 -> T4 | renamed CLI + new option -> script (Start-Srv) and fork mode | clean (script rename in T1 Step 5; fork mode uses defaults) |
| T1 -> T3 | server-context mapping (fork_step) -> startup log | clean |
| T2 -> T3 | res.heal on miss -> prompt_restore_tree wiring | clean |
| T2 -> T4 | `captured heal anchor at %d` wording reused -> fork mode grep | clean |
| T2 -> T4 | `within fork_step` log wording -> no script greps it | clean |
| T2 -> T3 | stats anchors_skipped_step -> stats_line step_skips= (test asserts) | clean |
| T3 -> T4 | miss-heal behavior -> fork mode asserts restored contains captured[0] | clean |
| T4 -> T1 | Start-Srv positional order (tree,ram,diskdir,idle,anchor_step,np,slot_save,disk_mib,tree_debug,ctx) -> fork call | clean |
| T4 -> T1/T2 | fork mode relies on default fork_step 8192 with $a1 ~ 8192 | clean (Filler overshoot only widens the gap) |

### Self-consistency per task
| task | check | finding |
|---|---|---|
| T1 | rename touches arg.cpp/common.h/server-context/script; no behavior change; heal regression | clean |
| T1 | `kv_tree_config.anchor_step` kept (internal), `fork_step` added | clean |
| T2 | fork spacing test: prev=1024, 5120 refused (4096<8192), 9216 stored (8192>=8192) | clean |
| T2 | miss-heal engine test uses fake io with matching prefix (2048) | clean |
| T3 | model test: B shares 2048 of A; tip anchor on A tail block not on B path -> miss | clean |
| T3 | caller preserves tree_heal across prompt_clear only when cache_prompt | clean |
| T4 | fork mode: b2 miss-heal at L1; b3 second fork at L1+len($a1) >= fork_step | clean |
| T4 | assertions parse positions from logs (no hardcoded token counts) | clean |

### Preflight rulings
- Ruling: T4 fork mode's second fork distance set to `len($a1)` (~8192) instead of the earlier ~1K idea, so the default fork_step admits it (no new Start-Srv parameter). Cost if wrong: none.
- Ruling: `promote_prune` deletion (D27) is part of T2; proof recorded in the plan (spacing keeps prev <= pos - step, so the prune window is always empty). Cost if wrong: none (function was a no-op).

## Task log

BASE for Task 1: 42ee7a6f7
Task 1: complete (commits 42ee7a6f7..ad81d0efc, review clean)
Task 1: minor (deferred): none worth acting on (common.h alignment per brief)
BASE for Task 2: ad81d0efc
Task 2: complete (commits ad81d0efc..9e7784270, review clean; 74 logic checks)
Task 2: minor (deferred): stats field alignment; comment blank line; plan check-count arithmetic (71 vs 74, doc-only)
BASE for Task 3: 9e7784270
Task 3: implementer concerns + controller rulings:
  (a) scenario_fork needed an explicit cfg.fork_step = 512 (close-position heal mechanics; policy defaults tested elsewhere) -> ratified.
  (b) artifact script Start-Srv gained a 12th param $fork_step (default 8192) and heal mode passes -fork_step 512 -> ratified (required for heal to stay green; T4 fork mode keeps the default 8192).
  (c) Ruling: stage-5 model/accept runs use -c 8192 (at -c 4096 the new 4096-token scenario's generation equality is vacuous: both gens empty). Plan T4 commands updated. Cost if wrong: none.
  (d) Note: the same vacuous-generation caveat applies to earlier accept-mode runs at -c 4096 (pre-existing; the non-vacuous evidence is the server bitwise checks + engine C/token checks). Record in stage-5 docs.
BASE for Task 3 review: 9e7784270
Task 3: complete (commits 9e7784270..04954b468, review clean)
Task 3: minor (deferred): add `check(!base.empty(), ...)` to scenario_fork_miss (hardening; non-vacuous at -c 8192)
Task 3: minor (deferred): plan check-count arithmetic (63 vs 64, doc-only)
Task 3: carry to Task 4: preserve heal mode's `-fork_step 512` when editing the script; use -c 8192 for model/accept runs
BASE for Task 4: 04954b468
Task 4: complete (artifact-only, review clean; fork captured=[8192,16384] restored=[8192,8192,16384], 5/5 bitwise, regression + 5-min soak green)
Task 4: minor (deferred): param comment missing `fork`; default-sim bypass run not archived; note in RESULTS that fork scope is forced-LRU (overlap covers default)
Task 4: Ruling: ratify the two scenario calibrations (ctx 32768 for 20501-token prompts; --slot-prompt-similarity 0 to force the tree path instead of stock in-place LCP reuse). Thresholds/assertions unchanged; reviewer verified both against logs/code. Cost if wrong: none.
Final review (42ee7a6f7..04954b468): Ready to merge YES (no Critical/Important; 7 minors).
Final polish wave (04954b468..af5e4c934): all 6 findings ADDRESSED (comment rationale, baseline guard, ctor fork_step validation, park-side step-skip assert, design doc notes, script comment); scoped re-review clean; logic 75/0, model 65/0, heal 0 failures.
Stage 5 complete: branch t32-stage5 head af5e4c934 (4 commits), not merged/pushed.
