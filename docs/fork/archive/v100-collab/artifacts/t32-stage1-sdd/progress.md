# SDD ledger — plan: D:\LLM\Backend\v100-collab\artifacts\t32-tree-plan-stage0-1.md

Spec: D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md (readable, same machine)

## Setup rulings

- Ruling: work on branch `t32-stage1` (created from `ba41cccec`) instead of master — the skill forbids implementing on master without explicit consent; a git worktree would force a full rebuild (the existing `build/` is path-bound), so a branch in the same repo keeps incremental builds working. Cost if wrong: user must fast-forward master (or cherry-pick) at the end.
- Ruling (superseded by user authorization, same day): user authorized automatic commits for this plan on branch `t32-stage1`, including review-fix commits, each followed by a summary to them; push / deploy / merge to master still require separate explicit approval. Implementers now commit as part of their task (per the skill template). Cost if wrong: user revokes; commits are local and reversible on this branch.
- Ruling: this harness's Task tool has no model parameter (subagent types: explore/general only), so per-skill model selection is not controllable; all subagents run on the session model. Cost if wrong: higher cost/latency than optimal model routing; no correctness impact.

## Preflight scan (before Task 1 dispatch)

| pair | producer -> consumer | finding |
|---|---|---|
| T0 -> T1 | T0 `tests/test-t32-range.cpp` (`fill_context`, mode dispatch) -> T1 adds `run_correctness` + dispatch line | consistent |
| T0 -> T3 | T0 same file -> T3 adds `run_range_bench` + dispatch line | consistent |
| T1 -> T2 | T1 `size_full`/`size_part` in `run_correctness`, `state_write_range`, `get_data_range_ext` -> T2 read checks + append restore | consistent |
| T1 -> T3 | T1 APIs -> T3 bench | consistent |
| T2 -> T3 | T2 `set_data_range_ext` -> T3 bench read path | consistent |
| T1 self | Step 1 contained the full kv-cache/hybrid implementation while Step 2 expects a failing test | CONFLICT -> fixed: implementation blocks moved to Step 4; Step 1 = declarations + context plumbing + base default-throw |
| T2 self | same shape (Step 1 lands declarations + implementations, Step 2 expected FAIL) | CONFLICT -> ruling below |
| T0 self | default `--mode correctness` has no handler in Task 0 -> prints usage; every Task 0 run passes `--mode h2d` | consistent |
| T3 self | `run_range_bench` decl/def/dispatch consistent; regression targets exist in the build dir | consistent |

- Ruling: Task 2's red phase dropped (declarations and implementations land in one step) to avoid a ~300-line plan restructuring; TDD red/green is fully executed in Task 1, and Task 2 is verified by the same test set green plus the task review. Cost if wrong: Task 2's tests are not proven to fail without the implementation; mitigated by the reviewer checking that the new checks exercise the new code paths.

## Progress

- Task 0: implemented + measured. Commit `82674b9f5`. h2d 27B 32K: size 2.100 GiB (default V f16), D2H 2.60 GiB/s, H2D 2.37 GiB/s, fill 796 t/s.
  - Ruling: build tree had `LLAMA_BUILD_TESTS=OFF`; keep tests ON in the shared build tree for the rest of this plan; restore OFF before any production build/deploy (stage 3/4).
  - Ruling: harness ran with default V type (f16); production uses `-ctv q8_0` -> Task 3 bench runs add `-ctv q8_0`; spec per-token/chunk numbers corrected from Task 3 measurements (100K production ~4.8 GiB -> ~2 s H2D at measured bandwidth).
  - GPU window: user approved running Task 3's bench (4-6 min) when reached.
- Task 0: complete (commits ba41cccec..82674b9f5, review clean). Trailer `Assisted-by: opencode` verified.
  - Minor (deferred): harness `chunk` arg unused until Task 3; `--mode h2d` defaults need `-c` on the CLI; unused `<cstring>` include (all brief-mandated).
- Task 1: BASE = 82674b9f5. Implemented by subagent: commit `3d05ef570` (9 files, +167). TDD red observed (range=0, "not supported" log, exit 1 on 2B); green on 2B (size relation) and 3B (byte-equality). Build tree now `LLAMA_BUILD_TESTS=ON` (left ON).
- Task 1: task review -> Needs fixes (2 Important, both plan-mandated or scope):
  - Important 1: `state_write_impl` comment promises independent p0/p1 sentinels but the filter only runs when `p0 >= 0`. Ruling: fix the filter to honor each sentinel independently (`(p0 < 0 || pos >= p0) && (p1 < 0 || pos < p1)`); plan text updated to match. Cost if wrong: none (strictly more correct).
  - Important 2: `llama_memory_hybrid_idx` inherits the hybrid range forwarder, so a QSA model would get a silently incomplete payload. Ruling: extend the fix's file scope to `src/llama-memory-hybrid-idx.{h,cpp}` and add a throwing `state_write_range` override (loud failure, return 0) — consistent with iswa types. Same treatment for `state_read_range` will be carried into Task 2's dispatch. Cost if wrong: idx models lose the range API they were never going to get in this plan anyway.
  - Minor (deferred): rejection paths not tested (p1<=p0 / negative seq / flags!=0); range-content coverage comes with Task 2's chunked restore.
  - ⚠️ trailer `Assisted-by: opencode` verified by controller via `git log -1 --format=%B 3d05ef570`.
- Task 1: fix round 1/5 (2 addressed, 0 open — sentinel filter + idx throw; commit `df4bacd74`). Scoped re-review: all findings addressed, no new breakage.
- Task 1: complete (commits 82674b9f5..df4bacd74, review clean after 1 fix round).
- Task 2: implemented by subagent: commit `4ae887b1c` (12 files, +229/-8). All three runs green (2B hybrid 16/16, 3B pure-attn 14/14, 2B `-kvu` 16/16), exit 0.
  - Ruling: implementer's test deviation 1 accepted — the brief's final check (`gen1b` re-decoding at L) is impossible because `gen1` already decoded L..L+7 (batch position rule); replaced with a continuation comparison seq 1 vs seq 2 at L+8, same check name/intent, strictly stronger. Plan text updated to match. Cost if wrong: the intactness check compares two continuations instead of one re-run; verified green on all three configs.
  - Ruling: deviation 2 accepted — `-kvu` is gated to other example types by `common/arg.cpp`; the harness now intercepts `-kvu`/`--kv-unified` itself (arg.cpp out of scope). Cost if wrong: none (test-only plumbing).
  - Ruling carried into implementation: `llama_memory_hybrid_idx` got a throwing `state_read_range` override too (parallel to the write side).
- Task 2: task review -> Approved (0 Critical/Important). ⚠️ controller item: report's "16/16 / 14/14" phrasing was imprecise (harness prints 15 checks per run); not a plan requirement, all-PASS exit 0 is. Resolved.
  - Minor (deferred): post-allocation append-failure path and mirrored-layout append rejection untested; `-kvu` interception is harness-local; `seq_id_read` read-and-discarded could use a comment.
- Task 2: complete (commits df4bacd74..4ae887b1c, review clean).
- Task 3: implemented by subagent: commit `dfd8a7035` (tests only, +55). Bench (27B, q8_0 V, 32K): read 12.8/23.2/41.5 ms/chunk (1896/2063/2309 MiB/s), write 20.3/34.8/60.1 ms/chunk (1198/1378/1594 MiB/s); ms/chunk ~doubles -> `--tree-chunk` stays 512. Measured 50200 B/token -> 512-block 24.5 MiB, 100K 4.68 GiB -> H2D ~2.0s. Regressions both pass (exit 0). Spec 1.2/3.4/appendix updated; TASKS + bench archive + branch patch written.
- Task 3: task review -> Approved (0 Critical/Important). ⚠️ trailer verified by controller (`git log -1 --format=%B dfd8a7035` shows `Assisted-by: opencode`); GPU/regression exit codes verified from report + archive (reviewer confirmed archive matches).
  - Minor (deferred): write MiB/s is a conservative lower bound (includes per-chunk allocation); spec appendix 100K D2H ~1.8s vs range-write ~4.0s cross-reference nit; `--chunk 0` divide-by-zero in the dev tool.
- Task 3: complete (commits 4ae887b1c..dfd8a7035, review clean).
- All 4 tasks complete. Final whole-branch review -> **With fixes** (2 Important, 0 Critical):
  - Important 1: stage-1 equivalence evidence (spec 2.3 a-d) not archived in the channel (it lives only in the workspace reports). Ruling: fold into the fix wave — re-run the three correctness runs, archive to `artifacts/t32-stage1-correctness.txt`, note in TASKS. Cost if wrong: none (archival only).
  - Important 2: `llama_kv_cache::state_read_range` lacks the `other` (mirrored cache) guard its write twin has; `state_read_sinfo` would silently return, reporting success with nothing restored. Ruling: add the guard (loud throw -> public 0). Cost if wrong: none (unreachable today, contract hardening).
  - Folded cheap minors into the same wave: `!gen1b.empty()` guard; contract checks (p1<=p0, p0<0, flags!=0 -> 0); truncated-append-blob cleanup test (exercises `state_clear_append`, reviewer's "fix next"); llama.h get_data_range_ext failure note; spec §2.2.3 sentence fix; spec appendix range-throughput cross-reference; spec iswa limitation + blob-magic note.
  - Remaining deferred minors stay deferred (see triage: all "fine to defer").
- Fix wave complete: commit `c5a91c15e` (3 files, +43/-2). Correctness re-runs on small models only: 2B hybrid 23/23, 2B `-kvu` 23/23, 3B 22/22, all exit 0; archived to `artifacts/t32-stage1-correctness.txt` + TASKS line; spec §2.2.3/appendix/§8/§9 edited.
  - Ruling (user-approved): the out-of-list library fix (`state_clear_append` resets `head` to the smallest freed cell, mirroring `seq_rm`) is accepted — it makes the failed-append cleanup leave the cache as if the append never happened; without it the destination re-append lands in different cells and FA reduction order flips a near-tie token (3B). Only affects the append-failure cleanup path. Cost if wrong: revert that hunk; failure path only.
  - Standing constraint from user: do NOT run Qwen3.8-27B anymore (too slow, too noisy); verification runs use small models (2B / 3B) only.
- Fix wave: scoped re-review dispatched (dfd8a7035..c5a91c15e), read-only, no model runs.

