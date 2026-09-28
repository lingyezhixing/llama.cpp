# Task 5 report: A/B acceptance (calib + low-overlap long sessions + multi-short + 3B control + negative + heal + regression)

Status: **DONE** - all rulings applied and re-verified. Final build `b936d687f` (D10) with the
amended script: all 9 runs end with `RESULT <mode>: 0 failure(s)` (calib, ab, overlap, b, b3,
neg, heal, ref, ref2); ref determinism 30/30 identical. Round-1/2 findings are recorded in
sections 4 and 10; round-3 evidence is in section 11. No push/deploy; no 27B.

## 1. Run summary (mandatory order)

| mode | exit | RESULT | key metrics | assertions |
|---|---|---|---|---|
| calib | 0 | `RESULT calib: 0 failure(s)` | total = 417,408,732 B (398 MiB), disk = 0 | PASS tree stores data |
| ab | 1 | `RESULT ab: 1 failure(s)` | ram=133 MiB, parked=23, restored=22, miss=2, rammax=138,760,464 B, diskmax=585,310,392 B | 4/4 log asserts PASS; `equal: B round 2` FAIL |
| overlap | 0 | `RESULT overlap: 0 failure(s)` | parked=0, restored=0, miss=1 | 8/8 PASS |
| b | 0 | `RESULT b: 0 failure(s)` | restored=20, ref4_lines=4 | 14/14 PASS |
| b3 | 1 | `RESULT b3: 1 failure(s)` | parked=11, restored=0, miss=12, all parks `0 checkpoint candidates` | `b3: pure-attention model restores from the tree` FAIL; 6/6 hash equal |
| neg | 0 | `RESULT neg: 0 failure(s)` | 65 files removed, strict=2, wide=2 | 2/2 PASS |
| heal | 1 | `RESULT heal: 1 failure(s)` | captured=0, failed=0, missed=0, parked=4, restored=2, f_keep=0.517 | `exactly one fork anchor captured` FAIL; 4/4 hash equal |
| ref | 0 | `RESULT ref: 0 failure(s)` | 30 hash lines | 0 failures |
| ref2 | 0 | `RESULT ref: 0 failure(s)` | 30 hash lines, identical to ref | 0 failures |

Hash-equality outcomes: ab 15/16, b 12/12, b3 6/6, heal 4/4, overlap 6/6.
ref determinism: `t32-stage3-ref.txt` vs `t32-stage3-ab-ref2.txt` -> **identical, 30/30 lines**.

Commands (driver, per mode):
`powershell -NoProfile -ExecutionPolicy Bypass -File D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1 -Mode <m> 2>&1 | Tee-Object <t32-stage3-<m>.txt>`
with `T32_RAM_MIB=133` set for `ab` only.

## 2. Script adjustments vs the brief (all recorded)

1. **Prompt construction moved after `Start-Srv`** (`Build-Sys`/`Build-Long`/`Build-Ovl`/`Build-Sess`
   called inside each branch). The brief builds `$sys/$baseA/...` at script load through `Filler`,
   which calls `/tokenize` - that requires a live server, so the brief's script aborts immediately
   (`Invoke-RestMethod : Unable to connect`; reproduced on the first calib attempt). Tokenization
   logic is unchanged, so the prompts are the same strings.
2. **b3: `Filler` calls moved after `Start-Srv`** - same reason (the brief computes `$sys3/$b1/$b2`
   before the server exists).
3. `$ram` env parsing made safe: `[int]$env:T32_RAM_MIB` throws when the variable is unset;
   default 96 MiB behavior preserved.
4. Server lifecycle hardened: `Stop-Srv` helper + `try/finally`, `WaitForExit` after kill,
   early-exit detection in the health poll, `-TimeoutSec 3` on `/health`, `-TimeoutSec 300` on
   `/completion` (`Req`). No smoke server was left running.
5. **neg deletes recursively**: the tree writes to `<dir>\blocks\` and `<dir>\anchors\`
   (`server-kv-tree.cpp:430,436`); the brief's `Get-ChildItem $dir -File` would have removed 0 files.
   Used `-Recurse -File` (65 files removed). The assertion regex was widened to
   `failed to read block|failed to read anchor|restore failed` while also printing the strict brief
   count; both were 2, so the assert is effectively the brief's.
6. **overlap rerun with the system prefix**: adjustment 1 initially omitted `Build-Sys` in overlap,
   so its prompts were 512 tokens shorter than the brief's geometry. The first run passed, but it was
   rerun with `Build-Sys; Build-Ovl` to match the brief; the rerun also passes and now matches the
   `ref` hashes exactly (extra diagnostic value, see 4.1).
7. Output naming per dispatch instruction: `t32-stage3-<mode>.txt`, second ref run ->
   `t32-stage3-ab-ref2.txt` (not the brief's `t32-stage3-ab-<mode>.txt`).
8. Two PowerShell parsing fixes (`RESULT ${Mode}:`, `req ${i}:`) - `$Mode:`/`$i:` are parsed as
   scope-qualified variables.

`/tokenize` was verified present (`server-context.cpp:5260-5299`, returns `{"tokens":[...]}`), so no
`TokCount` fallback was needed.

## 3. Calibration

- `calib` (2 rounds x 2 sessions, tree pass): last parked line `ram = 417408732 B, disk = 0 B`
  (ram tier 4096 MiB, no spill). `CALIB TOTAL BYTES AFTER 2 ROUNDS: 417408732`.
- `T32_RAM_MIB = round(total/3/1MB) = 133`.
- No adjustment was needed: `ab` at 133 MiB hit `diskmax = 585,310,392 B > 0` and restored=22 >= 8;
  the "raise toward total/2" fallback was not required. `rammax = 138,760,464 B` (temporary
  transient loads can exceed the tier limit by design; `settle()`/`enforce_budget` keep it bounded).

## 4. Failure diagnoses

### 4.1 ab: `equal: B round 2` mismatch - heal-break leaks into `cache_prompt=false` requests

Evidence:
- ab full B2 = `C63150FC6F2AA3C7`; ab tree B2 = `699E5CA954D651E4`.
- `699E5CA954D651E4` is also produced by calib B2, neg B2, and **ref B2** (tree off, cache_prompt=false,
  pure full prefill). So the tree-restore output equals the no-tree baseline; the full arm is the outlier.
- overlap rerun (tree on, no restores, identical prompts) produces byte-identical hashes to ref,
  i.e. an enabled-but-unused tree does not perturb outputs.
- In the ab full arm the tree still engages: `get_available_slot` calls `prompt_park` +
  `prompt_restore_tree` without checking `task.params.cache_prompt` (`server-context.cpp:1750-1770`).
  B2's restore sets `tree_heal = 16912`; `cache_prompt=false` then discards the restored KV
  (`n_past = 0`, `server-context.cpp:3420-3423`) but the batch fill still breaks at the heal position
  (`server-context.cpp:3697-3700`). The resulting ubatch split differs from ref, and one greedy
  near-tie flipped: 14 x `capture refused at 16912: no chain block contains this position` +
  14 x `W ... kv tree: failed to capture heal anchor at 16912` in the run.

Diagnosis: the tree data path is faithful (matches ref); the full arm deviates because a
`cache_prompt=false` request is still routed through restore/heal. Candidate fix (needs ruling):
skip `prompt_restore_tree`/`tree_heal` when `!task.params.cache_prompt` (also removes the 14 WRN
lines per run). This is the only ab failure; the 4 log assertions pass.

### 4.2 b3: 0/12 restores - no anchors exist for plain-attention models

Evidence:
- All 11 parks log `0 checkpoint candidates`; blocks and tip anchors are stored (`park ok`), and
  6 anchors exist in the final dump, but every one of the 12 restores misses
  (`restore: 12 calls, 0 hits, 12 miss`); tree-pass `prompt_n` equals the full prompt length.
- Server checkpoints are created only when `ctx_tgt_seq_rm_type == FULL/RS` or `n_swa > 0`
  (`server-context.cpp:3642-3645`). Qwen2.5-Coder-3B is plain attention -> `COMMON_CONTEXT_SEQ_RM_TYPE_PART`
  (`common/common.h:1012`) -> `do_checkpoint = false` -> no checkpoints ever.
- The tree's only anchors are tip anchors at chain end (prompt + generated tokens). The next-round
  prompt does not repeat the round's `\nUser turn N...\nAssistant:` suffix (hist is inserted right
  after the base), so it diverges at ~4101 while the tip anchor is at 4158 -> no anchor `<= m.deep`
  -> miss. A re-tokenization probe (`t32-b3-probe.ps1/2/3`) shows the generated IDs do round-trip;
  the blocker is the anchor position, not tokenization.

Diagnosis: integration gap - for `RM_TYPE_PART` models the tree can never be fed an anchor at or
below the incoming prompt's matched depth in this scenario. Candidate fixes (needs ruling):
create server checkpoints when kv-tree is on regardless of rm type, or adopt park-time anchors at
message/prompt-end positions, or change the b3 scenario. The plan's self-review defers
"anchor strategy polish" to stage 4, but the b3 assertion requires it now.

### 4.3 heal: `captured=0` (expected exactly 1) - D1 geometry miss

Evidence:
- `selected slot by LCP similarity, f_sim_best = 0.525 (> 0.100 thold), f_keep = 0.517` (3x).
  Head h ~= 1103 tokens vs tails 1000 -> `f_keep = 0.517 >= 0.5`, so the first fork
  (h+tA -> h+tF) is served by the stock VRAM LCP path and the tree is never asked to park/restore.
- The later exact-repeat request restores `491 tokens (heal = 2107)`; `heal == task length`, so the
  capture window (`n_tokens == hp` while still PROCESSING_PROMPT) never opens
  (Task 4 root cause E). Result: captured=0, failed=0, missed=0.
- Task 4's ruled geometry (head ~1024, tail 936) had f_keep ~0.44 and captured at 1024.

Diagnosis: brief geometry misses the `f_keep < 0.5` requirement by ~0.02. Candidate scenario fix
(not applied - assertion semantics are fixed and geometry changes were not authorized): lengthen
the tails (e.g. 1000 -> 1400) or shorten the head so the first fork engages the tree.

## 5. What passed (confidence evidence)

- Long sessions: 22 restore hits / 2 misses, park on every switch, SSD tier used (558 MiB), output
  hash equality 15/16 (the single miss diagnosed in 4.1).
- High-overlap: tree stays completely out of the way (`parked=0`, `restored=0`), full == tree 6/6.
- Multi-short sessions: 20 restores; shared prefix stored once with refcount 4 on exactly two blocks
  (`block 8aa538c3... [0,512) ref=4`, `block 3e73c95c... [512,1024) ref=4`, both disk tier).
- Negative: after deleting 65 block files, `failed to read block cbb69eb3caf557d4 from disk` +
  `restore failed: cannot load block [512, 1024)` then `restore miss ... full prefill`; the request
  still succeeds.
- Determinism: two full `ref` runs are hash-identical (30/30); `overlap` full-pass hashes are
  identical to `ref` for the same prompts.

## 6. Integration findings (no code patched)

1. `prompt_restore_tree`/`tree_heal` engage on `cache_prompt=false` requests; the full-prefill arm
   is split at a non-block-boundary heal position and emits `failed to capture heal anchor` WRNs
   (14 in ab). Output-visible effect: 1 of 16 ab comparisons flipped. Candidate fix in 4.1.
2. Plain-attention (`RM_TYPE_PART`) models get no checkpoints, so the tree cannot restore in the b3
   scenario (0/12). Candidate fixes in 4.2.
3. Scenario geometry: heal mode's head/tail ratio gives `f_keep=0.517` (needs < 0.5). Candidate
   fixes in 4.3.

No crash, no data corruption, no wrong tree payload observed; the tree-restore outputs match the
no-tree baseline in every cross-checked case (ref/calib/neg/overlap).

## 7. Environment / cleanliness

- `CUDA_VISIBLE_DEVICES=0` (RTX 4060 Laptop; the V100 is ~99% used by the two foreign servers).
  Ports used: 8933 (all modes), 8940/8941 (b3 probes only). No listeners left on them.
- Pre-existing `llama-server` PIDs 16520/16688 (ports 10002/10006, parent python.exe) untouched.
- No repo file changed; `git status` clean; no commit, no push, no deploy, no 27B.
- Step 3 doc updates (RESULTS.md/STATUS.md, design status line) were deferred pending the ruling.

## 8. Artifacts

- Script: `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` (UTF-8 no BOM, syntax-checked)
- Summary: `...\t32-stage3-accept.txt`; per-mode: `t32-stage3-{calib,ab,overlap,b,b3,neg,heal,ref}.txt`,
  second ref: `t32-stage3-ab-ref2.txt`
- Server logs: `<TEMP>\v100\t32-stage3\srv-<mode>-err.txt` (all modes)
- Diagnostic probes: `<user>\AppData\Local\Temp\opencode\t32-b3-probe{,2,3}.ps1`

## 9. Concerns

- The `ComparePasses` bit-exact criterion is fragile under ubatch-shape changes: the same greedy
  near-tie class documented for llama.cpp `cache_prompt` can flip a token (here B2), even though
  this run showed run-to-run determinism (ref x2) and tree-vs-no-tree equivalence when batching is
  equal (overlap == ref). A controller may want to treat single near-tie flips as non-blocking once
  the heal-break leak (4.1) is fixed and re-measure.
- The 14 `failed to capture heal anchor` WRNs per ab run indicate the heal-capture position
  (`m.deep`, a partial-match end) usually is not a stored-chain block boundary; the capture can only
  succeed on block boundaries (as in heal mode). Worth a look in the fix round.

---

## 10. Ruling applied (D10) - rerun evidence

### 10.1 Code fix and build

- `tools/server/server-context.cpp`: tree branch of `get_available_slot` now skips the restore when
  the request will not reuse the prompt (D10); `park` stays unconditional. Diff: 2 insertions,
  1 deletion.
- Rebuilt (`llama-server-impl.dll` deleted, `build_server.cmd` exit 0, relinked 14:41:53).
- Commit `b936d687f server : skip kv tree restore when the prompt is not reused`
  (`Assisted-by: opencode`), branch `t32-stage3`, working tree clean.

### 10.2 Script updates (per amended plan)

- heal branch now uses the Task 4-proven literal construction:
  `$h = "User: " + ('filler ' * 480) + ...`; `$tA = 'alpha-tail ' * 450`;
  `$tF = 'gamma-fork ' * 450`; evictor `"Zeta: " + ('omega ' * 1500)`; request bodies keep
  `message_delimiters`; assertions unchanged (1 capture / 0 failed / per-request hash equality).
- b3 branch now uses the D11 assertions: structural note (`Assert $true`), `failed to capture == 0`,
  and `ComparePasses`; the reuse assertion was dropped.
- Everything else (calib/ab/overlap/b/neg/ref) unchanged; my earlier structural fixes kept
  (post-Start-Srv prompt building, recursive neg delete, b3 Filler ordering).

### 10.3 Rerun results (build b936d687f, `T32_RAM_MIB=133`; calib/ref kept from round 1)

| mode | exit | RESULT | metrics | failed assertions |
|---|---|---|---|---|
| ab | 1 | `RESULT ab: 1 failure(s)` | parked=23, restored=12, miss=0, failed_captures=0, diskmax=497,257,704 B | `ab: the first round of each session is a visible miss` (actual 0) |
| overlap | 0 | `RESULT overlap: 0 failure(s)` | parked=0, restored=0, miss=0 | none (8/8) |
| b | 0 | `RESULT b: 0 failure(s)` | restored=12, ref4_lines=2 | none (14/14) |
| b3 | 0 | `RESULT b3: 0 failure(s)` | parked=11, restored=0, miss=6, failed_captures=0 | none (D11 + 6/6 hashes) |
| neg | 0 | `RESULT neg: 0 failure(s)` | 65 files, strict=2, wide=2 | none |
| heal | 1 | `RESULT heal: 1 failure(s)` | parked=7, restored=3, miss=1, captured=0, failed=0 | `heal: exactly one fork anchor captured` (actual 0) |

Hash equality: ab 16/16 (including B round 2 = `699E5CA954D651E4` == external ref baseline),
b 12/12, b3 6/6, heal 4/4, overlap 6/6.

### 10.4 D10 verified effects

- ab full pass no longer restores: `restored` 22 -> 12 (tree pass only), `miss` 2 -> 0,
  `failed to capture heal anchor` 14 -> 0, and the B2 near-tie flip disappeared
  (full arm == tree arm == ref baseline).
- No SSD reads wasted on discarded restores; output neutrality of `--kv-tree` on
  `cache_prompt=false` requests is restored.

### 10.5 Remaining failures - both are D10-vs-expectation conflicts (no code fix applicable)

**ab `miss >= 2`.** The old misses came only from the `cache_prompt=false` full pass (A1: cold
tree; B1: no B data yet). D10 skips those restores entirely, and the tree pass is pre-populated by
the full pass, so all 12 restores hit (`miss=0`). The assertion is unsatisfiable under D10.
Proposal (needs ruling): assert `miss -eq 0` in ab and move the cold-tree miss evidence to `calib`
(first request on an empty tree still logs `restore miss`, cache_prompt=true), or accept `neg`/`b3`
miss counts as that evidence.

**heal `captured == 1`.** The amended geometry's first fork (F2 `h+tF` against F1's `h+tA` chain,
`m.deep = 1024`) occurs in the `cache_prompt=false` full pass; D10 skips that restore, so no
`tree_heal` is produced there. In the tree pass every restore is an exact repeat of a chain stored
by the full pass -> `m.deep == task length` -> `heal == task length` -> the capture window
(`n_tokens == hp` while PROCESSING_PROMPT) never opens (Task 4 root cause E). Log evidence:
`restored 487 tokens (heal = 2447)` x3, `captured = 0`, `failed = 0`.
Additionally X's tree-pass restore misses (`restore miss for 1504`): its near-end checkpoint at 988
is skipped during park adoption because the shared head anchor at 487 is within `anchor_step` 4096
(988 - 487 = 501); the final dump has only 4 anchors (`msg@487`, `tip@2478` x2, `tip@1535`), and
tip@1535 > request 1504.
Proposals (needs ruling): (a) run heal's tree pass before the full pass (same process, capture at
the first fork, hashes unchanged); or (b) give the tree pass a fresh tree/server; or (c) amend the
assertion to `captured -eq 0` with a D10 note. No script change was applied - per instruction,
diagnosed and reported rather than forced.

### 10.6 Artifacts updated

- Re-tee'd: `t32-stage3-{ab,overlap,b,b3,neg,heal}.txt`; regenerated `t32-stage3-accept.txt`
  (with a note that calib/ref come from the pre-D10 build; both are unaffected: calib uses
  `cache_prompt=true`, ref has the tree off).
- No new commit needed beyond `b936d687f`; no repo file changed by the rerun.

Observation: the pre-existing foreign `llama-server` PID 16520 (port 10002) disappeared at some
point during the session; this task never targeted it (`Stop-Srv` only kills the PID returned by
its own `Start-Process -PassThru`). PID 16688 (port 10006) is still listening; ports 8933/8940/8941
are free (only a client-side TIME_WAIT on 8933).

---

## 11. Round 3: ruling applied - final verification (all modes green)

### 11.1 Script changes per ruling

1. `calib`: added `Assert ((Select-String ... 'kv tree: restore miss').Count -ge 1)
   'calib: cold restore miss is visible'` after the existing assertion (cold-tree miss moved here
   from ab).
2. `ab`: `miss >= 2` -> `Assert ($miss -eq 0) 'ab: every tree-pass turn restored from the tree'`.
3. `heal`: pass order swapped to `@(@($true,'tree'), @($false,'full'))` with the comment
   `# D10: run the tree pass first on a fresh tree so the first fork restore happens with
   cache_prompt=true`. Assertions unchanged.

### 11.2 Final rerun (build b936d687f, `T32_RAM_MIB=133` for ab)

| mode | exit | RESULT | metrics |
|---|---|---|---|
| calib | 0 | `RESULT calib: 0 failure(s)` | total 417,408,732 B; cold miss >= 1 PASS |
| ab | 0 | `RESULT ab: 0 failure(s)` | parked=23, restored=12, miss=0, diskmax=497,257,704 B; 16/16 asserts, 12/12 hashes |
| heal | 0 | `RESULT heal: 0 failure(s)` | captured=1, failed=0, parked=7, restored=2; 4/4 hashes |
| overlap/b/b3/neg | 0 | `RESULT <m>: 0 failure(s)` | unchanged from round 2 (kept) |
| ref/ref2 | 0 | `RESULT ref: 0 failure(s)` | 30/30 identical (pre-D10, tree off, unaffected) |

heal log (tree pass first):
```
kv tree: restore miss for 2447 tokens, full prefill      (req 1, cold)
kv tree: restored 487 tokens (heal = 1024)               (req 2, fork vs h+tA chain)
kv tree: captured heal anchor at 1024                    (exactly 1)
kv tree: restore miss for 1504 tokens, full prefill      (req 3, evictor)
kv tree: restored 487 tokens (heal = 2447)               (req 4, exact repeat)
```
All three checks requested by the ruling hold: ab `miss=0` with `restored=12 (>=10)` and
`disk > 0`; calib cold miss visible; heal exactly 1 capture, 0 failed, 4/4 hash equality.

### 11.3 Artifacts

- Re-tee'd on the final script: `t32-stage3-{calib,ab,heal}.txt`; `t32-stage3-accept.txt`
  regenerated with the build/script note.
- Repo: `b936d687f` only; working tree clean; no listeners left on 8933/8940/8941.
- Remaining notes: D11 (pure-attention no fork reuse) is documented by the amended b3 assertions;
  the near-tie fragility noted in section 9 did not reappear under D10 (16/16 in ab).

### 11.4 Controller amendment: non-vacuous b3 assertion (re-run)

`Assert $true 'b3: pure-attention has no mid-prompt anchors -> no fork reuse (structural, D11)'`
was replaced with the ruled non-vacuous form
`Assert (($log | Select-String 'kv tree: parked').Count -ge 1) 'b3: tree still parks pure-attention content (D11: no fork reuse)'`;
re-ran `-Mode b3` on the final script: `RESULT b3: 0 failure(s)`, `B3 METRICS parked=11 restored=0
miss=6 failed_captures=0`, both asserts PASS, 6/6 hashes equal (`t32-stage3-b3.txt` re-tee'd,
`t32-stage3-accept.txt` regenerated).

---

## 12. Step 3 (archive + docs) - completed

- `RESULTS.md`: appended `## T32 阶段 3: server 集成 (--kv-tree) + A/B 验收` (4 bullets: branch/commit
  facts, acceptance numbers, D1/D10/D11, artifacts). Read back via `[IO.File]::ReadAllText` - tail
  matches; still UTF-8 no BOM (first bytes 23-20-52).
- `STATUS.md`: appended the `- **T32 阶段 3 完成 ...**` block (same content). Read back - tail matches;
  still UTF-8 no BOM (23-20-53).
- `t32-tree-storage-design.md` line 3 is now
  `状态: 阶段 0-3 已完成 (阶段 3: server 集成 + A/B 验收, 分支 t32-stage3 未合并/未 push); 阶段 4 未开始`
  (stale "阶段 2 分支未合并" dropped). Read back - line 3 matches; still UTF-8 no BOM (23-20-54).
- Raw server logs archived to `artifacts/t32-stage3-logs/`: all 16 `srv-*.txt` (total 462,107 B < 2 MB,
  so the full set was copied; the 8 `*-out.txt` files are 0 B). Read back: 16 files, 462,107 B.
- Write pattern used: temp file + `[IO.File]::WriteAllText` / `AppendAllText` with
  `New-Object System.Text.UTF8Encoding($false)`; verification via `[IO.File]::ReadAllText`.
- No code changes; no commits; repo still clean at `b936d687f`; no listeners on 8933/8940/8941.
- Re-review fix (D2 omission): appended `- **已知缺口 (D2, 记录)**: 树锚点载荷不含 MTP/spec 状态 ...`
  at EOF of `RESULTS.md` and `STATUS.md`, and into design doc section 9 (after the fallback bullet,
  before `## 附`). Read-back verified; all three files still UTF-8 no BOM (23-20-52 / 23-20-53 / 23-20-54).
