# Task 4 report: heal capture + idle park + SLOT_ERASE drop

Status: **BLOCKED** (no commit; working tree unchanged, implementation verified and built).
Reason: the brief's heal smoke cannot reach the new code - the fork sits inside the first
512-token chunk, so `restore` misses and `tree_heal` is never set. Detailed evidence below.
Per the dispatch instruction ("If the heal anchor is not captured ... STOP and report
BLOCKED ... do not tweak the scenario to force a pass"), no scenario tweaks were applied
and the remaining smokes (idle rerun, Task 3 regression) were not run.

## 1. Uncommitted diff review (nothing changed by me)

`tools/server/server-context.cpp`, 39 insertions / 0 deletions vs tip `376fa06ad`.
Five hunks, all matching the brief:

| hunk | current lines | brief step | verdict |
|---|---|---|---|
| idle gate `&& !params_base.kv_tree` | 1527 | Step 3 | exact |
| idle park branch `if (tree) prompt_park else prompt_save` | 2567-2572 | Step 3 | exact |
| SLOT_ERASE `drop_seq` with `has_mtmd` guard | 2806-2810 | Step 4 | exact, except `%s` literal wrapper (matches surrounding style, e.g. :2570) |
| heal capture block, kept after `} // end of SLOT_STATE_STARTED` | 3575-3598 | Step 1 | exact per controller ruling (placement kept, not moved) |
| batch fill break at heal position | 3697-3700 | Step 2 | exact |

Interfaces verified present: `tree_heal` (:301), `prompt_park` (:303),
`capture_anchor`/`drop_seq` (`server-kv-tree.h:162/165`), `n_tokens()` returns `int`,
`get_tokens()` returns `const llama_tokens &`, `has_mtmd` public bool. No file other than
`server-context.cpp` is modified.

## 2. Build

Deleted `build\bin\Release\llama-server-impl.dll`, ran
`<TEMP>\v100\build_server.cmd` -> exit 0
(UI asset download warnings only; cached assets used). DLL relinked at 13:22:08.

## 3. Smoke: heal scenario (brief command + bounded health poll / timeouts / finally-kill)

Server: 2B model, port 8932, `--cache-ram 0 --no-cache-idle-slots --kv-tree --tree-chunk 512
--tree-anchor-step 512 --tree-ram 256 --tree-disk ... --tree-disk-limit 256`.
Log: `<user>\AppData\Local\Temp\opencode\t32-task4\s3heal-err.txt`.

Request `timings.prompt_n`: A = 1110, A' first = 1110, short = 510, A' second = 604.

Tree log lines (all of them; exact):

```
kv tree: restore miss for 1110 tokens, full prefill      (A, first request)
kv tree: parked 1141 tokens, 2 checkpoint candidates, ram = 71368596 B, disk = 0 B
kv tree: restore miss for 1110 tokens, full prefill      (A' first)
kv tree: parked 1141 tokens, 2 checkpoint candidates, ram = 142737192 B, disk = 0 B
kv tree: restore miss for 510 tokens, full prefill       (short)
```

- `captured heal anchor` count = **0**; `failed to capture heal anchor` = 0;
  `heal position ... missed` = 0; `restored` = 0.
- A' second did no park/restore at all (slot reuse via stock LCP path;
  `f_keep` >= 0.5), hence prompt_n 604 - not a tree hit.

SLOT_ERASE: `POST /slots/0?action=erase` -> **HTTP 501**, standard error body
"`This server does not support slots action. Start it with --slot-save-path`"
(gate: `server-context.cpp:4949-4952`; the brief's server flags omit `--slot-save-path`).
`kv tree: dropped the stored sequence` count = 0 - the handler was never reached.

Independent reproduction: the cancelled attempt's log
`<TEMP>\v100\s3heal-err.txt` (13:04) shows the identical
restore misses (1110 / 1110 / 510) and no heal lines. This failure predates this session
and is not caused by the uncommitted code.

## 4. Root cause (scenario geometry, not code)

Exact tokenization via `/tokenize` on the same model:

- shared head `H = "User: " + "filler "*500 + "\nAssistant: ok\nUser: "` = **510 tokens**
- `H + "short"` = 510 tokens; `H + "alpha-tail "*200` = 1110 tokens

The A/A' fork is at token index ~510 (first tail token). Chunking is a fixed 512 grid
(`chain_hashes`, `server-kv-tree.cpp:16-26`). The stored first block `[0, 512)` of A
therefore contains the diverging tokens, `memcmp` fails at the first block, so:

- `m.n_full = 0`; partial matching only runs when `m.n_full == tokens.size()/chunk`
  (`server-kv-tree.cpp:220`), so it is skipped -> `m.deep = 0`
- `restore()` returns early with a miss (`server-kv-tree.cpp:1248`)
- `prompt_restore_tree` never sets `tree_heal` (`server-context.cpp:365` is the only
  setter, `-1` everywhere else), so the new heal block (:3575) cannot run.

The brief's own comment says `# ~1000+ tokens shared head`; the model tokenizes
`'filler '` to ~1 token/unit, so the head is 510 - two tokens short of one chunk.
The plan's acceptance-grade heal mode (Task 5, `t32-tree-plan-stage3.md:1052-1063`)
sizes the head with `Filler 'head' 1024` tokens and starts the server with
`--tree-anchor-step 4096`, i.e. the fork lands beyond the first chunk; the brief's
inline smoke is an under-sized derivative of that scenario.

## 5. Files changed / state

- No repository file changed by me. `tools/server/server-context.cpp` remains the only
  modified file (39 insertions, uncommitted, byte-identical to attempt 1 except review
  only). No commit created. No stray `llama-server` processes left.

## 6. Self-review findings

- Diff matches the brief and the binding controller ruling; no fixes were needed.
- No prompt/flag tweaking to force the expected log lines.
- Only cosmetic deviation: `%s` wrapper on the erase log literal (safer, matches style).
- Risk if committed as-is: none observed at build level; heat capture path itself is
  untested by this smoke (unreachable), so it should be exercised before any final
  acceptance claim.

## 7. Recommendations for the controller (no action taken)

1. Fix the inline heal smoke so the fork is beyond the first chunk, e.g. reuse the
   Task 5 sizing (`Filler 'head' 1024` + `--tree-anchor-step 4096`) or scale
   `'filler ' * 1200`; then rerun and expect `captured heal anchor at N` once and
   the second A' `restored N tokens` with the same N.
2. Add `--slot-save-path <dir>` to the erase smoke server; the endpoint is gated on it.
3. Idle-enabled rerun and the Task 3 regression smoke were not run (STOP rule);
   schedule them in the same fix round, then commit.

---

# Addendum: controller-ruled corrected smoke rerun (attempt 2)

Controller ruling: heal head changed to `'filler ' * 900`; erase server gains
`--slot-save-path`. All four smoke groups were run with bounded /health polling,
`-TimeoutSec 60` on completions, and server kill in `finally`.
Result: the mandated heal smoke STILL does not capture (different root cause:
`f_keep`), but targeted diagnostics prove all four code paths work. No commit
created; diff unchanged.

## 1. Corrected heal smoke (plan Step 5, exact flags + --slot-save-path)

Log: `<user>\AppData\Local\Temp\opencode\t32-task4\s3heal-err.txt`
(script `heal-smoke2.ps1`).

prompt_n: A = 1510, A' first = 1510, short = 910, A' second = 604.
Tree lines: exactly one, `kv tree: restore miss for 1510 tokens, full prefill`
(first request). captured = 0, failed = 0, missed = 0, restored = 0, parked = 0.

Slot selection explained the bypass (log):

```
selected slot by LCP similarity, f_sim_best = 0.602 (> 0.100 thold), f_keep = 0.590
```

`f_keep >= 0.5` (server-context.cpp:1712) keeps `update_cache = false`, so the
stock VRAM LCP path serves A' (and short/A' second); the tree is never asked to
park or restore again. Root cause A below.

Erase step: HTTP 200, `{"id_slot":0,"n_erased":1541}`; no `dropped` line (the
tree stored nothing in this run), i.e. the endpoint now works but has nothing
to drop.

## 2. Idle-enabled rerun (drop --no-cache-idle-slots, same other flags)

Log: `s3idle-err.txt` (script `idle-smoke.ps1`). Same request metrics as (1),
only the initial restore miss; `parked_count = 0`; no idle lines at all.
Root cause C below.

## 3. Task 3 four-request regression (same flags as Task 3 Step 3)

Log: `s3reg-err.txt` (script `t3-regression.ps1`). PASS, identical to Task 3:

```
prompt_n A1 = 407, A2 = 4
prompt_n B1 = 407, B2 = 4
content match A: True, B: True
kv tree: restore miss for 407 tokens, full prefill          (A first)
kv tree: parked 454 tokens, 1 checkpoint candidates, ram = 44686416 B, disk = 0 B
kv tree: restore miss for 407 tokens, full prefill          (B first)
kv tree: parked 454 tokens, 1 checkpoint candidates, ram = 60606180 B, disk = 28766652 B
kv tree: restored 403 tokens (heal = 407)                   (A second)
kv tree: parked 454 tokens, 1 checkpoint candidates, ram = 64888476 B, disk = 24484356 B
kv tree: restored 403 tokens (heal = 407)                   (B second)
```

All four expected Task 3 items hold: prompt_n drop, parked+restored present,
disk > 0, misses only the first request of each session.

## 4. Targeted diagnostics (not the mandated smoke; labeled as evidence)

### 4a. Idle park + erase drop with `-np 2`

Log: `s3idle2-err.txt` (script `idle-np2-diag.ps1`), two distinct sessions,
cache_idle_slots enabled by default, gate fix active:

```
kv tree: parked 1541 tokens, 2 checkpoint candidates, ...   (slot 1, idle, while rB runs)
kv tree: parked 1235 tokens, 2 checkpoint candidates, ...   (slot 0, idle, while rA2 runs)
erase -> 200 {"id_slot":0,"n_erased":1235}
kv tree: dropped the stored sequence
```

No `requires --cache-ram, disabling` warning (gate change works). Idle branch
and `drop_seq` verified.

### 4b. Heal capture with a geometry that satisfies the design constraints

Log: `s3heald-err.txt` (script `heal-diag.ps1`). Request prompt (4 requests:
A, F1, X, F2) and per-request `message_delimiters` = [{user, "User:"}]:

```
head = "User: " + 'filler '*480 + "\nAssistant: ok\nUser: " + 'filler '*600 + "\nAssistant: ok\nUser: "
A = head + 'alpha-tail '*450 ; F = head + 'gamma-fork '*450
X = "Zeta: " + 'omega '*1500        (non-sharing evictor session)
```

Observed:

```
rA=2447, rF1=1960, rX=1504, rF2=4 prompt_n
kv tree: restored 487 tokens (heal = 1024)      (F1 first: anchor at 487, fork at ~1112)
kv tree: captured heal anchor at 1024           (exactly 1; 0 failed, 0 missed)
kv tree: restored 2443 tokens (heal = 2447)     (F2: tree restore, 1960 -> 4 prefill)
erase -> 200 {"id_slot":0,"n_erased":2478}
kv tree: dropped the stored sequence
```

This proves the new code paths end to end: restore hit sets `tree_heal`, the
batch fill stops at 1024 (state stays PROCESSING_PROMPT because 1024 < prompt
length), the capture block runs and `capture_anchor` stores at 1024, and the
later request is served through the tree. Note the second restore returns
2443, not 1024 (see root cause D).

## Root causes (all in the scenario/expectations, not in the diff)

- A. `f_keep`: head H = 910 tokens, tail 600 -> `f_sim = 910/1510 = 0.602`,
  `f_keep = 0.602*1510/1541 = 0.590 >= 0.5`. Per D1, high overlap is served by
  stock VRAM LCP reuse and the tree is never engaged. For tree engagement the
  tail must exceed the head (`f_keep = H/(H+tail+31) < 0.5`).
- B. Anchor availability: with raw `/completion` and no `message_delimiters`,
  the only checkpoints are the two near-prompt-end offsets
  (server-context.cpp:3740-3753; matches the observed "1"/"2" candidate counts).
  For a fork inside a non-final block, `m.deep` = last fully matched block
  boundary because partial matching requires all full chunks to match
  (server-kv-tree.cpp:220), so a usable anchor must exist at or below that
  boundary. The diagnostic needed `message_delimiters` plus an early user
  message (anchor 487 <= deep 1024).
- C. Idle branch: it parks only non-processing slots at task dispatch
  (server-context.cpp:2562-2580); with `-np 1` the dispatched slot is always
  processing, so the plan's idle rerun can never log an idle park. Needs
  `np >= 2` (4a).
- D. `restored N == captured N` is not an invariant: after A' is itself parked,
  its own end-of-prompt checkpoints are deeper than the heal anchor and win the
  next restore (2443 vs 1024 in 4b). The design contract (design doc 4) is
  "first fork replays, later forks replay <= step"; assert `captured >= 1` and
  reuse growth, not equality.
- E. Benign: when `hp == task length` the slot flips to DONE_PROMPT in the
  same iteration the prompt completes (server-context.cpp:3767-3769), so the
  capture block cannot run. That is why the Task 3 regression shows
  `heal = 407` with no capture line; only a real mid-prompt fork
  (`hp < task length`) captures, which is the intended case.

## Proposed corrected acceptance (one-shot, based on 4b + 4a)

- Heal: use the 4b prompts/flags (or fix the plan smoke as: head with two user
  messages via `message_delimiters`, tails longer than the head, non-sharing
  evictor session). Assert: exactly 1 `captured heal anchor at N`; no
  `failed`/`missed`; a later A' is served through the tree (restore hit with
  `N >= captured N` or prompt_n collapse). Drop the strict equality.
- Erase: same server, `--slot-save-path` present; assert HTTP 200 and
  `dropped the stored sequence` (4a/4b).
- Idle: run with `-np 2`; assert idle `parked` lines (4a).
- Task 3 regression: unchanged, passes (3).

## State after this round

- No commit created. `tools/server/server-context.cpp` still the only modified
  file (39 insertions, byte-identical to the ruled brief). Build unchanged.
- No servers left by this session's runs (no listeners on 8931/8932/8935/8936).
  Two pre-existing `llama-server.exe` processes (PIDs 16520/16688, ports
  10002/10006, parent `python.exe` 12864) were observed; they are not from this
  task's runs and could not be terminated (access denied) - left untouched.

## Recommendation

The diff is complete, build-clean, and each of its four behaviors is verified by
the diagnostics above. On your ruling (accept the diagnostics + amended smoke,
or authorize a scenario fix), I can commit the same unchanged diff immediately
with the prescribed message.

---

# Addendum 2: amended Step 5 recorded evidence (final)

Ruling applied: diagnostics accepted, diff committed as-is, amended Step 5 run
on the committed tree.

## Commit

`69aeacd84 server : heal fork points, idle parks and slot erase in the kv tree`
with `Assisted-by: opencode`; 1 file changed, 39 insertions(+), 2 deletions(-);
branch `t32-stage3` (previous tip 376fa06ad). Working tree clean after.

## Deviations from the amended plan text (disclosed)

- Plan Step 5 line 817 writes `message_delimiters=@(@{user='User:'})`, which
  serializes to `{"user":"User:"}`. `common_chat_msg_delimiters_parse`
  (common/chat.cpp:139-140) reads the keys `role` and `delimiter`, so the literal
  form yields zero delimiters. The recorded run used the parser-correct
  equivalent `@(@{role='user'; delimiter='User:'})` (same construction as
  diagnostic 4b); serialized body: `{"delimiter":"User:","role":"user"}`.
- Plan Step 5 kills the server before the erase block; the erase call was
  executed on the same live server before the kill, as the ruling requires.
- Everything else (port, flags, prompts, request bodies) is verbatim Step 5.

## 1. Heal smoke (port 8932, script `heal-amended.ps1`, log `s3heal-err.txt`)

Request metrics: A = 2447, F1 = 1960, X = 1504, F2 = 4 (prompt_n);
content F1 == F2: True.

```
kv tree: restore miss for 2447 tokens, full prefill            (A)
kv tree: parked 2478 tokens, 4 checkpoint candidates, ram = 124383636 B, disk = 0 B
kv tree: restored 487 tokens (heal = 1024)                     (F1 first)
kv tree: captured heal anchor at 1024
kv tree: parked 2478 tokens, 4 checkpoint candidates, ram = 218906508 B, disk = 0 B
kv tree: restore miss for 1504 tokens, full prefill            (X)
kv tree: parked 1535 tokens, 2 checkpoint candidates, ram = 253587192 B, disk = 0 B
kv tree: restored 2443 tokens (heal = 2447)                    (F2)
```

Assertions: exactly 1 `captured heal anchor at N` (N = 1024) -> PASS;
0 `failed to capture` -> PASS; F2 served through the tree (prompt_n 1960 -> 4,
`restored 2443`) -> PASS; first-fork line `restored 487 tokens (heal = 1024)`
present -> PASS.

## 2. SLOT_ERASE (same server, `--slot-save-path` present)

`POST /slots/0?action=erase` -> HTTP 200 `{"id_slot":0,"n_erased":2478}` and

```
kv tree: dropped the stored sequence
```

`dropped` count = 1 -> PASS.

## 3. Idle smoke (port 8933, `-np 2`, script `idle-amended.ps1`,
## log `s3idle-err.txt`)

Request metrics: s1 = 403, s2 = 403, s1-more = 5 (prompt_n).

```
kv tree: parked 418 tokens, 1 checkpoint candidates, ram = 44346900 B, disk = 0 B   (slot 1 idle)
kv tree: parked 418 tokens, 1 checkpoint candidates, ram = 88693800 B, disk = 0 B   (slot 0 idle)
```

parked count = 2 (>= 1 required) -> PASS; `requires --cache-ram` warnings = 0
(gate fix) -> PASS.

## 4. Safeguards and cleanliness

Bounded `/health` polling (60 x 1 s, `-TimeoutSec 3`), completion/erase requests
`-TimeoutSec 60`, server killed in `finally` on every run. No listeners left on
8932/8933; the two pre-existing `llama-server.exe` PIDs 16520/16688 (ports
10002/10006, parent python.exe) were left untouched as instructed.

## Final status

All amended Step 5 assertions and the Task 3 regression pass on commit
`69aeacd84`. The four Task 4 behaviors (heal capture, batch break, idle park +
gate, SLOT_ERASE drop) are verified end to end.

---

# Fix round: heal log format (review finding)

Review finding (Important): `tools/server/server-context.cpp:3595` passed
`slot.prompt.n_tokens()` to `%zu`; `server_prompt::n_tokens()` returns `int`
(server-task.h:576), so the variadic format was a mismatch (UB).

## Change

One line, argument unchanged, plan updated to the same form:

```diff
-                            SLT_TRC(slot, "kv tree: heal position %d missed (now %zu)\n", hp, slot.prompt.n_tokens());
+                            SLT_TRC(slot, "kv tree: heal position %d missed (now %d)\n", hp, slot.prompt.n_tokens());
```

Diff: 1 insertion, 1 deletion, only `tools/server/server-context.cpp`.

## Covering check

1. Build: deleted `build\bin\Release\llama-server-impl.dll`, ran
   `build_server.cmd` -> exit 0 (`server-context.cpp.obj` rebuilt;
   `llama-server-impl.dll` / `llama-server.exe` relinked).
2. Amended heal smoke rerun (same script `heal-amended.ps1`, port 8932, log
   `s3heal-err.txt`; the fixed TRC line is not emitted at default verbosity):

```
rA = 2447, rF1 = 1960, rX = 1504, rF2 = 4 prompt_n; content F1 == F2: True
kv tree: restored 487 tokens (heal = 1024)
kv tree: captured heal anchor at 1024
kv tree: restored 2443 tokens (heal = 2447)
erase -> 200 {"id_slot":0,"n_erased":2478}
kv tree: dropped the stored sequence
counts: captured = 1, failed = 0, missed = 0, dropped = 1
```

The build is good and the capture/restore evidence is unchanged from Addendum 2.

## Commit

`c0254f32c server : fix kv tree heal log format`
(`Assisted-by: opencode`; new commit on top of 69aeacd84, not amended; working
tree clean). No listeners left on port 8932 after the rerun.
