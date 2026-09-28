# SDD ledger - plan: D:\LLM\Backend\v100-collab\artifacts\t32-media-tree-plan.md

Plan: media reuse in the kv tree. Branch: t32-media. Base commit: d701c2c13.
Spec: D:\LLM\Backend\v100-collab\artifacts\t32-media-tree-spec.md (read by implementers per task).
Controller: opencode session (night 2026-09-28). User asleep; implementers must decide autonomously.

## Pre-flight scan (before Task 1)

| pair / item | what one produces vs the other consumes | finding | ruling |
|---|---|---|---|
| T1 -> T2 | T1: mapping helpers, chain_split/hash, park/match, signatures (media param, anchor tok, block tok0/media, blocks_by_pos0); T2: restore/capture/drop media paths | T1 must leave restore/capture/drop compiling; plan says text-only behavior intact there | consistent, no ruling needed |
| T1 -> T3 | T3 consumes kv_tree_media, park/restore/drop/capture signatures | T1 defines all of them | consistent |
| T2 -> T3 | T3 consumes kv_tree_restore_anchor.tok | T2 defines it | consistent |
| T3 -> T4 | T4 needs the media-wired server built | sequential | consistent |
| T1 internal | tests assert ranges from the fake io; plan's test 3 mentions pos_at which is cpp-internal | harness cannot call the static helper | Ruling: the harness computes expected positions locally from the spans it built; do not export pos_at |
| T1 internal | test 4 "clamp" as written builds an impossible request | real requests never end inside a chunk (chunks are whole) | Ruling: the clamp is defensive; test it with a deliberately malformed truncated input passed straight to match(), asserting n_part clamps to the chunk start |
| T2 internal | test 1 expects res.C == 2026 | verified by hand: starts=[0,2026], partial path reaches 2500, anchor at tok 2026 | consistent |
| plan-mandated defect check | none found (no test-that-asserts-nothing, no verbatim duplication mandated) | - | - |

## Tasks

Task 1: tree storage side (data model, mapping, park, match) - dispatched
Task 1: review (1 Important plan-mandated, 1 Minor opportunistic, 3 Minor deferred)
Task 1: fix round 1/5 (2 addressed, 0 open - interleaved-chain block lookup walk-back + stale header comments; commits 5722fac1e..2e341d9e9)
Task 1: minor (deferred): duplicated media logic in server-kv-tree.cpp (identity compare x3, inside-chunk test x2) - candidate for a small static helper at final review
Task 1: minor (deferred): test-t32-tree.cpp identity-mismatch checks use m.deep <= 1000 where == 1000 is provable; alignment check evaluates position space (exact ranges[1].second / ranges.size() assertions still catch the realistic regression)
Task 1: complete (commits d701c2c13..2e341d9e9, review clean)

Task 2: tree retrieval side (restore, anchors, capture, drop) - dispatched
Task 2: review clean (tests-only commit; named-risk check confirmed the retrieval media paths exist at base)
Task 2: minor (deferred): r.heal on a hit (C < m.deep) not asserted in the boundary scenario
Task 2: minor (deferred): capture refusal inside a media chunk is untested (read-verified)
Task 2: minor (deferred): brief wording tokens.size()==tok vs implementation tok<=tokens.size() - implementation is the more correct one; traceability only
Task 2: complete (commits 2e341d9e9..70157be75, review clean)

Task 3: server integration (spans, guards, restore rebuild, erase, heal) + text regressions - dispatched
Task 3: review (1 Critical plan-defect: get_tokens() assert aborts on mmproj servers; 1 Minor fixed: keep_first chunk boundary)
Task 3: fix round 1/5 (2 addressed, 0 open - get_tokens_raw accessor at 4 tree call sites + keep_first boundary guard; commits 7d5b11bb1..587272960)
Task 3: minor (deferred): FNV offset-basis sentinel for null/empty chunk id vs spec text "nullptr id -> 0" - equality-only use, harmless
Task 3: complete (commits 70157be75..587272960, review clean; mmproj smoke park/restore for text and image chats)

Task 4: E2E media verification on 0.8B + mmproj + evidence + docs - dispatched
Task 4: complete (controller-run E2E, no subagent commit; evidence artifacts\t32-media-e2e-20260928.txt; 5/5 outputs bit-identical to no-tree baseline; media park/restore verified incl. 1024-token image chunk and adjacent chunks; docs updated)
Ruling: Task 4 ran in the controller session instead of a subagent - the dispatched subagent was cancelled after a reported hang, and step-by-step controller execution was the safer verification path. Cost if wrong: no independent review of the E2E script (the script is an artifact, and the final branch review still covers the code).
Final whole-branch review: MERGE-READY (no Critical/Important; 4 Minor: FNV constant, checkpoint rebuild units, keep_first partial-chunk cut, unknown-id sentinel note)
Final fix wave: 587272960..90f2f2d69 (3 fixed, 1 noted-acceptable; harness logic 127 PASS, 2B model 65 PASS, server clean link)
Final scoped re-review: all findings addressed, no new breakage
Branch complete: t32-media = d701c2c13 + 6 commits, head 90f2f2d69, merge-ready, not merged, not pushed
Ruling: workspace archived to v100-collab\artifacts\t32-media-sdd\ then deleted from the repo (SDD finish step; the collab archive is the record).
