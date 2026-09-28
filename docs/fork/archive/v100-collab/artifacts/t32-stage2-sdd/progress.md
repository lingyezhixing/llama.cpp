# SDD ledger - plan: D:\LLM\Backend\v100-collab\artifacts\t32-tree-plan-stage2.md

Spec: D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md (v2, read)
Branch: t32-stage2 (from master 52b7bf7de, in the main repo - see Ruling 1)
Plan workspace: .superpowers/sdd/t32-tree-plan-stage2/

## Preflight rulings

- Ruling 1: work in the main repo on branch t32-stage2 instead of a git worktree - the build scripts and CMake build tree use absolute paths to this directory, and stage 0/1 established this pattern. Cost if wrong: none (a worktree can be added later).
- Ruling 2: the `task` tool of this harness has no model parameter, so per-role model selection (skill: Model Selection) cannot be enforced; subagents run on the session default. Cost if wrong: slower/less optimal model choice, no correctness impact.
- Ruling 3: the plan adds tests/test-t32-tree.cpp, which AGENTS.md gates on maintainer approval; this is a private fork and the user commissioned and approved this plan. Cost if wrong: file must be dropped before any upstream submission (already tracked as fork-only).

## Preflight scan (task pairs sharing files/interfaces)

| pair | produces -> consumes | finding |
|---|---|---|
| T1 -> T2 | header types/park/match/make_room -> restore/store_anchor/adapter | OK, T2 replaces park wholesale and inserts store_anchor after make_room |
| T2 -> T3 | store_anchor/restore/harness helpers -> adoption block before seq record; `touched` defined here and consumed by T5 park tail | OK, T5 runs after T3 |
| T3 -> T4 | scenario_fork A' block -> replaced wholesale by the heal version | OK |
| T4 -> T5 | capture_anchor ending -> (3c) replaces exactly the text T4 produced | OK, verbatim match checked |
| T1/T2 -> T5 | make_room (T1 def, T2 park pre-check) -> T5 deletes both and adds enforce_budget/park_rollback/remove_block | OK, ordering T5 after T2/T3 |
| T5 -> T6 | enforce_budget/remove_block/demote_* -> T6 replaces enforce_budget, adds demote_one/evict_*/remove_seq (uses remove_block) | OK |
| T6 -> T7 | eviction/stats -> run_accept with budgets that must not evict | OK |
| T1 self | asserts blocks 3/4/4/5 -> park stores A=3, B2 adds 1, re-park A no-op, C tail adds 1 | OK |
| T2 self | tip asserts anchors_added==1 and park refusal past L | OK |
| T3 self | sparsify asserts C=512/1536/2048 with step 1024 and checkpoints 512/1024/1536 | OK |
| T4 self | heal asserts C=512 then 1024 with step 512 | OK |
| T5 self | ssd asserts blocks_ram==0 after park with 64KiB RAM + disk tier | OK, (3a) allows demoting pinned payloads (eviction still skips pinned) |
| T6 self | evict: park B forces eviction; 4KiB budget refuses with no residual | OK, park pins its own new payloads so eviction only touches older data |
| T7 self | accept asserts 8 blocks (A/B) and 5 blocks (B-mini), tokens_reused 12*4096 | OK |
| cross | AGENTS.md tests/* gate vs plan creating tests/test-t32-tree.cpp | Ruling 3 |
| cross | harness needs LLAMA_BUILD_TESTS=ON (currently ON) | OK, restore OFF before deploy - out of stage 2 scope |

Known minors (for final review triage):
- remove_block increments st.evicted_blocks also when called from park_rollback (rollback deletions counted as evictions) - stats-only, no behavior impact.
- settle() promote/demote order over unordered_map is non-deterministic - placement only, no correctness impact.

## Progress

- Task 1: first dispatch aborted mid-flight by the harness (no commit, no report). Files present uncommitted and logic test green on controller inspection; fresh implementer dispatched to verify/complete/commit.
- Task 1: complete (commits 52b7bf7de..15588ab6c, review clean, Approved)
- Task 1: warnings resolved: commit trailer `Assisted-by: opencode` present; branch has no upstream (local only); rebuild-freshness warning is moot (reviewer verified sources verbatim-identical to the brief, so the committed content is what compiled; Task 2 rebuilds anyway).
- Task 1: minor (deferred): partial-tail block gets refcount++/heat++ though it is not in seqs[tip].chain (eviction-semantics decision, T6).
- Task 1: minor (deferred): match() uses blocks.at() - blocks/blocks_at must stay in sync once removal paths exist (T5/T6).
- Task 1: minor (deferred): no cfg.chunk > 0 validation (div-by-zero / infinite loop if a future parser allows 0).
- Task 1: minor (deferred): fake-io blobs are position-based, not prefix-based - cannot detect cross-sequence KV mixing; restore tests need better fake data (T2+).
- Task 1: minor (deferred): unused includes and an unused `args` vector in the harness (placeholders for later modes).
- Task 1: minor (deferred): no test for the park-refusal path in T1; T2 adds a past-L refusal check.
- Task 2: Ruling: plan defect - the Task 2 park replacement dropped Task 1's `nb.bytes = nb.data.size()`; spec 5.1 requires byte stats and T5/T6 accounting uses `bytes`. Decision: restore the assignment (align with Task 1) and patch the plan text. Cost if wrong: none (byte counters would otherwise under-count from T2 onward).
- Task 2: Ruling: the two plan-mandated Important findings (draft capture/restore failures ignored at server-kv-tree.cpp:226/381) violate the global "never silent" constraint even though io_dft is null in stage 2. Decision: fix now - draft capture failure refuses the park, draft restore failure fails the restore (both WRN + counter), plan text patched. Cost if wrong: none in stage 2 (dormant path); stage 3 gets correct behavior instead of silent draft loss.
- Task 2: minor (deferred): anchors.end() fallback at restore sets ok=false with no message (unreachable).
- Task 2: minor (deferred): seq_rm(C,-1) return ignored (false = "nothing to trim" is the normal case; a real failure would be invisible).
- Task 2: minor (deferred): capture_partial defined but unused until T3 (brief-mandated).
- Task 2: minor (deferred): run_tree_path still generates on a restore miss (contract loose; asserts fail first).
- Task 2: minor (deferred): store_anchor failure after the block commit is not atomic (unreachable).
- Task 2: fix round 1/5 (2 addressed, 0 open; commits 7b59bb340..a2ebafb65)
- Task 2: complete (commits 15588ab6c..a2ebafb65, review clean, Approved)
- Task 3: Ruling: plan defect - blocks_at declared as unordered_map but containing_block uses upper_bound (does not compile). Decision: type is std::map<llama_pos, std::vector<uint64_t>>; plan text patched. Cost if wrong: none (ordered map, slightly more memory per entry).
- Task 3: Ruling: plan defect - the harness prefill helper always runs to the end of the token vector, so the plan's fork/sparsify prefills captured checkpoint states at the wrong positions (e.g. capture at 1024 after prefilling to 2048). Decision: slice the token vector per segment (prefill up to the capture position, capture, continue); batch boundaries unchanged. Plan text patched. Cost if wrong: none.
- Task 3: Ruling: plan defect - scenarios chain in one context; scenario_tip leaves seq 0 occupied, so fork/sparsify hit the M-RoPE position-order check. Decision: llama_memory_seq_rm(mem, 0, -1, -1) at the start of fork/sparsify. Plan text patched. Cost if wrong: none.
- Task 3: Ruling: plan defect - containing_block used upper_bound on blocks_at (wrong for a checkpoint exactly on a chunk boundary) and returned the first block at a pos0 even when it belongs to another sequence's divergent variant. Decision: keep the implementer's lower_bound fix and add an optional chain filter (`containing_block(pos, chain)`); adoption passes the parked chain; Task 4's capture_anchor gains a tokens parameter and passes its chain (plan patched before Task 4). Cost if wrong: an anchor could attach to an off-chain variant and stay unusable (no corruption).
- Task 3: Ruling: visibility for candidate skips - counters satisfy the "never silent" constraint for expected filters (invalid candidate, spacing skip); WRN + counter for unexpected failures (no chain block, store failure). Fix now. Cost if wrong: none (visibility only).
- Task 3: fix round 1/5 (3 addressed, 0 open; commits 0ca557660..5b843e2f6)
- Task 3: complete (commits a2ebafb65..5b843e2f6, review clean, Approved)
- Task 3: minor (deferred): duplicate identical candidates with anchor_step <= 0 double-count refcount/touched (degenerate config).
- Task 3: minor (deferred): no direct assertions on anchors_skipped or refcount sync (functional outcomes are asserted instead).
- Task 3: important-class (deferred, out of fix-diff scope): park's early return on an already-stored tip silently discards passed checkpoints (no counter/WRN) - same silent-drop class; final review should triage.
- Task 4: Ruling: plan defects corrected by the implementer, accepted, plan text patched: (a) kv_tree_anchor needed `transient` already in Task 4 (remove_anchor uses it; Task 5 now only adds it to kv_tree_block); (b) capture_anchor precondition is `pos_max() == pos - 1` (positions are 0-based, pos is a covered-token count); (c) the harness A' block used r.res.C/r.res.heal on a kv_tree_restore (compile error) - corrected to r.C/r.heal; (d) the A' replay prefill must be sliced to [0,1024) so the capture happens at state 1024 and the following prefill [1024,1536) does not overlap; partial-capture failure now WRNs + counts. Cost if wrong: none (all four verified by the 28/28 green model run).
- Task 4: Ruling: folded a reviewer Minor into fix round 1 - guard the A' prefill with `r.C >= 0` so a restore miss fails cleanly instead of reading tokens[-1] (plan text patched). Cost if wrong: none (test-only robustness).
- Task 4: minor (deferred): anchor_bytes unused until T5 (MSVC silent, GCC -Wall would warn).
- Task 4: minor (deferred): `pos <= 0` guard in capture_anchor returns false silently (input guard).
- Task 4: minor (deferred): `last_promote = pos` unconditional - an out-of-order capture rewinds the cursor (benign under monotonic order).
- Task 4: minor (deferred): promote_prune's deletion branch and remove_anchor run in no test (coverage).
- Task 4: fix round 1/5 (2 addressed, 0 open; commits e4a50cb83..8d370c4ce)
- Task 4: complete (commits 5b843e2f6..8d370c4ce, review clean after fix)
- Task 4: minor (deferred, out-of-scope observation): capture_anchor's other refusal exits do not count (`pos <= 0` silent, `pos_max` mismatch WRN-only, `blk == 0` WRN-only); store_anchor failure silent.
- Task 5: Ruling: plan defect - scenario_ssd omitted `llama_memory_seq_rm` before its first prefill (previous scenario left seq 0 occupied -> decode failed). Decision: add the clear line, same as fork/sparsify; plan text patched. Cost if wrong: none.
- Task 5: minor (deferred): read_disk uses ftell/long (2 GiB cap on Windows) - payloads are <= ~162 MiB by design.
- Task 5: minor (deferred): capture_anchor does not roll back its stored anchor when enforce_budget fails; the anchor stays (consistent stats) while the caller sees a failed heal.
- Task 5: complete (commits 8d370c4ce..0c9d8ffd9, review clean, Approved; trailer verified)
- Task 5: minor (deferred): payload-mismatch WRN says "dropping it" but the entry stays and is retried on later restores (spec wording vs implementation).
- Task 5: minor (deferred): rollback does not restore heat/last_used; remove_block counts rollback removals as evicted_blocks.
- Task 5: minor (deferred): demote_* disk-limit refusals are silent in the capture path (park path WRNs).
- Task 5: minor (deferred): settle ignores filesystem::remove failure yet adjusts bytes_disk/blocks_disk (stale uncounted file).
- Task 5: minor (deferred): rename fallback removes the destination first (non-atomic only in that path).
- Task 5: minor (deferred): no test for the hash-mismatch branch or the budget-refusal+rollback path.
- Task 5: minor (deferred): directory_iterator removal while iterating in scenario_ssd (collect first would be more robust).
- Task 6: Ruling: plan defect (found and reproduced by the implementer) - the brief ran verbatim fails the tiny-budget refusal checks: `evict_seq_one` could evict the just-parked sequence (its blocks are pinned, but the seq record was not, and `remove_seq` does not check pinned), so enforce_budget succeeded by deleting the new data and park returned success instead of refusing. Decision: pin the new seq record (`s.pinned = true`) across `enforce_budget` in park, unpin after; plan text patched. Cost if wrong: none (the 18/18 logic run, including the refusal checks, is the evidence).
- Task 6: minor (deferred): `remove_seq` does not check `pinned` on the blocks it removes; unreachable after the ruling above (eviction never runs during a restore, and older sequences' shared blocks have refcount >= 2), but defense-in-depth would add the check.
- Task 6: Ruling: plan defect - spec 5.3 step 3 says "whole leaf sequences" but the plan's `evict_seq_one` selected any unpinned sequence (oldest first), so a parent whose chain is a prefix of a live child could be destroyed first. Decision: add the leaf guard (a sequence is a leaf when no other stored sequence's chain extends it) as a chain-prefix check; plan text patched. Cost if wrong: none for safety (refcounts already protect shared trunk); the guard preserves higher-reuse parents.
- Task 6: minor (deferred): refusal path inflates `evicted_blocks` via `park_rollback` -> `remove_block` (stats-only, known).
- Task 6: minor (deferred): whole-seq eviction counts each block in `evicted_blocks` and cascade-removed anchors are not counted in `evicted_anchors` (monitoring overlap).
- Task 6: minor (deferred): `remove_seq` anchor refcount decrement has no position guard (drift possible with repeated chunk content; anchors linger as unusable hints).
- Task 6: minor (deferred): demote score's leaf penalty can be outweighed by heat/last_used (heuristic inversion, not reachable at test sizes).
- Task 6: minor (deferred): no test isolates the block-eviction and seq-eviction steps (r.C == -1 is explained by the anchor eviction alone).
- Task 6: minor (deferred): park does not refcount/pin pre-existing blocks reused beyond a match gap (benign bookkeeping drift).
- Task 6: fix round 1/5 (1 addressed, 0 open; commits bf1e4562d..19acf3fb2)
- Task 6: complete (commits 0c9d8ffd9..19acf3fb2, review clean after fix)
- Task 7: Ruling: plan arithmetic defects found by the implementer (BLOCKED, harness left uncommitted): (a) A-mini expects 8 blocks but a 1024-token tail is 2 chunks, so the correct count is 10 (6 shared + 2 + 2); the byte assertion also ignored the two tip anchors and could never pass. Decision: assert 10 blocks and `stored < 2 * (seq_bytes + anchor_bytes)` as the dedup evidence; plan patched. (b) scenario_ssd's hardcoded 64 MiB disk limit cannot hold a pure-attention model (PARTIAL_ONLY = the whole KV, ~56.6 MB anchor at 1536 tokens for 3B) so park was refused. Decision: size the disk limit at runtime as 4 * (seq_bytes + anchor_bytes) after the prefill, constructing the tree after the measurement; plan patched. Cost if wrong: none (harness arithmetic only; all assertions stay strict).
- Task 7: fix round 1/5 (2 addressed, 0 open; commit 19acf3fb2..a9d5d9595); logic 18/18, 2B model 36/36, 2B accept 42/42, 3B control 36/36; archives + channel docs updated.
- Task 7: complete (commits 19acf3fb2..a9d5d9595, review clean, Approved)
- Task 7: minor (deferred): dead store `cfg.disk_limit = 64 << 20;` before the runtime sizing (cosmetic).
- Task 7: minor (deferred): archives start with a PowerShell NativeCommandError wrapper header from `2>&1 | Out-String` (content complete and truthful).
- Task 7: minor (deferred): run_accept ignores prefill() return codes (existing harness style).
- Task 7: minor (deferred): accept proves zero replay indirectly (C == len + tokens_reused), no explicit heal == -1 assert.
- Final review (52b7bf7de..a9d5d9595): verdict "With fixes". Triage: FIX NOW (one wave) - (1) park early-return silently drops non-empty checkpoints [WRN + counter]; (2) promote_prune prunes by global position range instead of the capture chain, so a heal on one branch can delete another branch's anchors [design decision 9 conformance]; (3) no real-IO coverage of the non-chunk-aligned restore path (partial-tail match + partial block load + trim), the production-common case [test-only].
- Final review deferred to stage 3/4 (accepted triage): bytes_load under-count; restore-miss INF logging; global sparsification prev-anchor (same chain-scoping idea as #2); "dropping it" wording vs retention; stats drift (rollback/cascade counters, heat/last_used); demote scoring inversion; cfg.chunk > 0 guard; dead cfg.debug; usage text missing accept; 32-bit/ftell caps; promote_prune deletion branch and payload-mismatch branch untested; harness hygiene (directory_iterator removal, ignored prefill returns).
- Final review stage-3 recommendations: bound restore transient RAM peak (~13 GB with defaults); add CLI validation for tree knobs; drop-sequence API; restore-miss INF logs.

