# Task 3 report: server integration (media prompts into the kv tree)

Status: DONE
Commit: 7d5b11bb1 "server : wire media prompts into the kv tree" (branch t32-media)

## What I implemented

All five brief steps, verbatim where the brief gave exact code.

1. `server_tokens` helpers (`tools/server/server-common.{h,cpp}`):
   - `llama_pos pos_last() const`: -1 when empty; when the last token is a media placeholder, walks
     `map_idx_to_media` backwards to find the chunk that ends at `tokens.size()` and returns
     `pos_next(chunk.idx)` (media cells share the chunk start position under M-RoPE); otherwise
     `pos_next() - 1`.
   - `const std::map<size_t, mtmd::input_chunk_ptr> & media_map() const`.
   - `bool has_media() const { return !map_idx_to_media.empty(); }`.
   - No dependency from server-common.h on server-kv-tree.h; the accessor exposes only the existing
     mtmd map type.
2. `static uint64_t fnv1a64(const char *)` and `static std::vector<kv_tree_media> tree_media_spans(const
   server_tokens &)` in `server-context.cpp`, exactly as in the brief (FNV-1a 64 over
   `mtmd_input_chunk_get_id()`, nullptr/empty id hash to the same sentinel; `idx/id/n_tok/n_pos` from
   the chunk; map order makes the spans sorted by idx).
3. Four call sites in `server-context.cpp`:
   - `prompt_park`: dropped `prompt.tokens.has_mtmd` from the guard; guard is now
     `p_max != prompt.tokens.pos_last()`; passes `tree_media_spans(prompt.tokens)`; checkpoint
     candidates now set `in.tok = c.n_tokens`.
   - `prompt_restore_tree`: dropped `tokens.has_mtmd` from the guard; passes spans; on hit rebuilds
     the slot prompt with `tokens.clone()` + `keep_first(res.C)` (preserves media map and has_mtmd)
     and skips the checkpoint rebuild when `prompt.tokens.has_media()`.
   - `SLOT_ERASE`: dropped the `!has_mtmd` guard; `drop_seq(tokens, spans)`.
   - heal capture: `capture_anchor(..., spans, hp)`; `tree_heal` stays a token index.
   - Untouched upstream media gates: checkpoint creation (`do_checkpoint && !has_mtmd`),
     `n_cache_reuse`, stock prompt cache.
4. Build + tests + smoke (below).
5. Commit.

## Tests and exact results

### Server build
Deleted `build\bin\Release\llama-server-impl.dll`, ran
`<TEMP>\v100\build_server.cmd`: clean compile and clean link of
`llama-server-impl.dll` and `llama-server.exe` (only pre-existing warning C4297 in src/llama.cpp, and
the unrelated UI download timeout which falls back to "assets unchanged"). No warnings from the
changed files.

### Harness (rebuilt with build_test_t32.cmd test-t32-tree)
- `build\bin\Release\test-t32-tree.exe --mode logic` -> exit 0, every check PASS, including the whole
  media scenario block (park/restore with n_pos < n_tok, identity mismatch, tail clamp, chunk-end
  capture, drop_seq with media).
- `CUDA_VISIBLE_DEVICES=0 test-t32-tree.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99
  -fa on --mode model --ram-mib 4096 -c 8192` -> exit 0, 65 PASS. The only "failed" strings in the log
  are the intended `[kv-tree] failed to read block ...` lines of the ssd scenario, whose assertions
  (`ssd: restore misses when block files are gone`, `ssd: disk error counted`) PASS.

### Text-only server smoke (2B, no mmproj, cuda0, port 18081)
Command (stdout/stderr redirected to files):

```
llama-server.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf --host 127.0.0.1 --port 18081
  --kv-tree --tree-ram 1024 --tree-checkpoint-anchor-step 512 --tree-checkpoint-fork-step 512
  -np 1 -c 8192 -b 1024 -ub 512 -fa on -ngl 99 --temp 0 --slot-prompt-similarity 0
```

Two /completion requests, the second extends the first (prompt A, then A + generated content),
`cache_prompt=true`, n_predict=64, temp 0. Server log lines:

```
kv tree enabled: chunk = 512, anchor_step = 512, fork_step = 512, ram = 1024 MiB, disk = off
kv tree: restore miss for 681 tokens, full prefill          (request 1, tree empty)
kv tree: parked 744 tokens, 2 checkpoint candidates, ram = 69766644 B, disk = 0 B
kv tree: rebuilt 2 context checkpoints
kv tree: restored 744 tokens (heal = -1)                    (request 2)
```

Timings: request 1 prompt_n=681, request 2 prompt_n=1 (744 tokens reused). No WRN/ERR lines. The
server process I started was stopped afterwards; no llama-server process remains.

Note on `--slot-prompt-similarity 0`: with the default 0.1, an extending request is served by the
stock LCP-similarity slot shortcut (f_keep >= 0.5) and the tree park/restore is intentionally not
reached; the first smoke run confirmed exactly that (only the cold restore miss appeared). The repo's
own `t32-stage3-ab.ps1` `fork` mode documents this same shortcut and uses `-sim 0` to force the tree
path, so I added that one flag for the smoke only.

## Files changed

- `tools/server/server-common.h` (+8)
- `tools/server/server-common.cpp` (+27)
- `tools/server/server-context.cpp` (+46/-11, net)
- Commit 7d5b11bb1, 3 files, 70 insertions, 11 deletions.

## Self-review findings

- All brief steps implemented; diff reviewed line by line; `git diff --check` clean; no new non-ASCII
  (the one non-ASCII line in server-context.cpp, 584, is pre-existing).
- `res.C`/`tree_heal`/checkpoint `tok` are token indices; `pos_last()`/spans only feed positions into
  the tree, consistent with the spec invariants.
- For text-only prompts everything is behavior-identical (empty span list, pos_last() == size-1).
- The checkpoint rebuild loop still uses `a.pos`; this is safe because media prompts skip the loop
  and for text-only `tok == pos`. I left it unchanged as the brief did not ask for a change there.
- `prompt_restore_tree` now preserves `has_mtmd` from the request; on an mmproj server this means a
  restored text-only prompt keeps `has_mtmd == true` with an empty media map. All `server_tokens`
  paths with an empty map behave identically to `has_mtmd == false` (verified by reading pos_next,
  keep_first, get_common_prefix, size_up_to_pos).
- No harness test targets `pos_last()` directly; it is server-only code and was exercised by the
  smoke park (guard passed, park stored 744 tokens). The media branch of `pos_last()` (last token
  NULL) is not exercised by this task's smoke (no mmproj); it is covered by the controller's planned
  media E2E.

## Decisions made autonomously

1. FNV offset basis: used the brief's `1469598103934665603ull` verbatim. The spec's parenthetical
   "(nullptr id -> 0)" is about a deterministic value for an unknown id; the brief's loop maps both
   nullptr and the empty string to the same fixed sentinel (the offset basis), and ids are only ever
   compared for equality, so the literal value has no semantic effect. No code compares `id == 0`.
2. `--slot-prompt-similarity 0` added to the smoke invocation only (see above). The prescribed core
   flags are unchanged.
3. Commit trailer `Assisted-by: opencode`, matching the other commits on this branch and AGENTS.md.
4. Kept the one-line comment in `pos_last()` and two short header comments; the brief's code had
   none, but these explain non-obvious invariants (media cells share the chunk start position).

## Concerns

- Media server E2E (mmproj, image round trip through park/restore) was not run in this task; it needs
  the 0.8B-MTP + mmproj model and is planned for the controller's verification. This task's smoke was
  text-only per the brief.
- The `tree_heal` capture path for media prompts stops the batch exactly at `tree_heal` (a token
  index) and passes spans; correct by construction, but only the controller's media E2E will exercise
  it end to end.

---

# Fix report (review findings 1 and 2)

Status: DONE
Commit: 587272960 "server : fix kv tree media access and keep_first chunk boundary"

## What changed

Finding 1 (Critical): the four tree call sites passed `server_tokens::get_tokens()`, which asserts
`!has_mtmd`; on any mmproj server `slot.prompt.tokens.has_mtmd` is true (server-context.cpp:1445) and
chat tasks go through `process_mtmd_prompt` -> `server_tokens(chunks, true)` even with no media, so
park/restore/erase/heal would abort on every mmproj server.

- Added `const llama_tokens & get_tokens_raw() const { return tokens; }` to `server_tokens`
  (server-common.h) with a comment that media placeholders are included and only media-aware
  consumers may use it. `get_tokens()` and its assert are unchanged.
- Switched the four tree call sites to `get_tokens_raw()`: `prompt_park`, `prompt_restore_tree`,
  `SLOT_ERASE` `drop_seq`, heal `capture_anchor`.
- Left stock callers of `get_tokens()` untouched: the context-shift path at server-context.cpp:3177
  (has its own `GGML_ASSERT(!has_mtmd)`) and the tokenize path at server-context.cpp:5154.

Finding 2 (small): `server_tokens::keep_first` rejected a valid boundary between two adjacent media
chunks (both neighbors NULL even when a new chunk starts at `n`).

- The guard now also requires that no chunk starts at `n`:
  `tokens[n-1] == NULL && tokens[n] == NULL && map_idx_to_media.find(n) == map_idx_to_media.end()`.
  A boundary at the start of a new chunk passes; a cut inside a chunk still goes through
  `find_chunk(n - 1)` and throws.
- Comment updated to state that the boundary between two adjacent images is allowed.

## Covering tests (all after the fix, commit 587272960)

Server build: deleted `build\bin\Release\llama-server-impl.dll`, ran
`<TEMP>\v100\build_server.cmd` -> clean compile, clean link (only the
pre-existing src/llama.cpp C4297 warning and the unrelated UI download timeout).

Harness (rebuilt via `<TEMP>\v100\build_test_t32.cmd test-t32-tree`):
- `build\bin\Release\test-t32-tree.exe --mode logic` -> exit 0, 127 PASS; the only "FAIL" strings are
  the intended disk-error scenario messages, whose assertions PASS.
- `CUDA_VISIBLE_DEVICES=0 test-t32-tree.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99
  -fa on --mode model --ram-mib 4096 -c 8192` -> exit 0, 65 PASS; only the intended ssd scenario
  "failed to read block"/"restore failed" lines.

mmproj smoke (Finding 1 covering test), cuda0, port 18082, stderr to file:

```
llama-server.exe -m <models>\Qwen3.5-0.8B-MTP\Qwen3.5-0.8B-UD-Q4_K_XL.gguf
  --mmproj <models>\Qwen3.5-0.8B-MTP\mmproj-F16.gguf --image-min-tokens 1024
  --host 127.0.0.1 --port 18082 --kv-tree --tree-ram 1024
  --tree-checkpoint-anchor-step 512 --tree-checkpoint-fork-step 512
  -np 1 -c 8192 -b 1024 -ub 512 -fa on -ngl 99 --temp 0 --slot-prompt-similarity 0
```

Requests: (a) text-only chat twice, second extends the first (user, then user + assistant reply +
user); (b) same with a 96x96 PNG (blue square on red) attached via image_url data URI, generated with
System.Drawing. `chat_template_kwargs = { enable_thinking: false }`, max_tokens 32/16, temperature 0.

Results: server alive after all requests, no `GGML_ASSERT`/abort lines. Log:

```
kv tree: restore miss for 20 tokens, full prefill          (a1, tree empty)
kv tree: parked 38 tokens, 1 checkpoint candidates, ram = 40872144 B, disk = 0 B
kv tree: restored 16 tokens (heal = -1)                    (a2)
kv tree: parked 71 tokens, 3 checkpoint candidates, ram = 82150584 B, disk = 0 B
kv tree: restore miss for 1052 tokens, full prefill        (b1, tree empty for this conversation)
kv tree: parked 1053 tokens, 1 checkpoint candidates, ram = 135519576 B, disk = 0 B
kv tree: restored 1048 tokens (heal = -1)                  (b2, deep restore through the image chunk)
```

Chat answers: a1/a2 = "The ocean is a vast, endless, and ever-changing sea that def...", b1/b2 =
"blue". The text case restores 16 tokens because the Qwen template puts the empty think block only on
the generation prompt, so the parked sequence and the multi-turn rendering diverge at token 16; the
image case restores 1048 of 1052 tokens including the 1024-token image chunk, which is the real
media-path evidence. The server process started for the smoke was stopped afterwards; no
llama-server process remains.

## Decisions

1. `--slot-prompt-similarity 0` was again added only to force the tree path (the stock LCP shortcut
   would serve an extending request in place); the controller's prescribed flags are unchanged.
2. `keep_first`: implemented exactly the directed condition and kept `find_chunk(n - 1)` as the
   thrower. Note (pre-existing, out of scope): if `n - 1` is itself a chunk start and `n` is inside
   the same chunk, `find_chunk(n - 1)` succeeds, so keeping a single token of a chunk is still not
   rejected. No server path produces such an `n` (restores are chunk-aligned).
3. Follow-up commit on t32-media with the `Assisted-by: opencode` trailer, as before.

## Concerns

- The adjacent-chunk `keep_first` boundary is not exercised by any automated test; no request in the
  smoke produced two directly adjacent media chunks. The change is a pure condition relaxation and
  was reviewed against the map layout (chunks sorted, non-overlapping, whole).
- The text-only chat restore stays shallow (16 tokens) for template reasons described above; this is
  expected and unrelated to the fix.
