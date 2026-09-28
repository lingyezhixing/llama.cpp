# Final fix report: whole-branch review findings (t32-media)

Branch: t32-media. Base: d701c2c13. Head before fixes: 587272960.
Fix commit: 90f2f2d69 "server : fix kv tree review nits" (2 files, +7/-4).

Findings 1-3 fixed; finding 4 left unchanged and noted below.

## Finding 1: non-canonical FNV-1a basis (fixed)

tools/server/server-context.cpp:57: `1469598103934665603ull` -> `14695981039346656037ull`,
the canonical FNV-1a-64 offset basis. The hash is computed and compared only for equality at the
single call site that builds `kv_tree_media::id` (server-context.cpp:71), the tree state is wiped on
restart, and ids are computed consistently at runtime, so the change has no behavioral effect.

## Finding 2: checkpoint rebuild mixed token index and position (fixed)

tools/server/server-context.cpp prompt_restore_tree, `n_ckpt_max > 0` rebuild loop:
- filter: `a.pos <= 0 || a.pos >= res.C` -> `a.tok <= 0 || a.tok >= res.C` (res.C is a token index)
- `ck.update_pos(a.pos, ...)` -> `ck.update_pos(a.tok, ...)`

For text-only anchors tok == pos, so text-only behavior is identical. Media prompts still skip the
loop (`!prompt.tokens.has_media()`), so this is future-proofing for a relaxed guard; with it the
checkpoint token count and position range would be correct for media spans.

## Finding 3: keep_first could keep a single token of a chunk (fixed)

tools/server/server-common.cpp has_mtmd branch. `find_chunk(n - 1)` returns the chunk whose start is
n - 1; if that chunk extends past the cut (`n - 1 + n_tok > n`), throw
"Cannot resize in the middle of a media chunk". The existing throw for `n - 1` not begin-of-chunk
and the existing relaxation for a chunk starting exactly at `n` are unchanged. No tree path produces
such `n` (restore points are chunk-aligned); this is a defensive guard only.

## Finding 4: unknown/empty chunk id sentinel (left, noted)

Unknown/empty chunk ids hash to the FNV basis sentinel, so two unknown-id chunks at the same index
with equal n_tok/n_pos compare equal. Unreachable from server-tokenized requests: mtmd sets a
sha256 id, and placeholders preserve it. Behavior deliberately not changed.

## Commands and results

Harness build:
`cmd /c <TEMP>\v100\build_test_t32.cmd test-t32-tree`
- exit 0; linked bin\Release\test-t32-tree.exe; only the pre-existing warning C4297 in src/llama.cpp.

Logic mode:
`build\bin\Release\test-t32-tree.exe --mode logic`
- exit 0, 127 PASS. The only FAIL/failed strings are the 6 intended disk-error log lines of the
  diskerr scenario, whose assertions PASS.
- full log: <user>\AppData\Local\Temp\opencode\t32-logic-fix.txt

Model mode (2B, GPU 0):
`CUDA_VISIBLE_DEVICES=0 test-t32-tree.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on --mode model --ram-mib 4096 -c 8192`
- exit 0, 65 PASS. The only FAIL/failed lines are the intended ssd scenario messages.
- full log: <user>\AppData\Local\Temp\opencode\t32-model-fix.txt

Server:
deleted build\bin\Release\llama-server-impl.dll, then
`cmd /c <TEMP>\v100\build_server.cmd`
- exit 0, clean link of llama-server-impl.dll and llama-server.exe; no errors and no warnings from
  the changed files.
- full log: <user>\AppData\Local\Temp\opencode\server-build-fix.txt

All added lines are ASCII; `git diff --check` clean.

## Decisions

1. Finding 3 uses the literal range test `n - 1 + n_tok > n` to stay traceable to the review wording;
   the message follows the file style ("Cannot serialize media chunk ...") and the existing
   find_chunk throw path.
2. Finding 1 changed only the literal; the sentinel semantics of finding 4 were intentionally left
   as-is per the review.
3. Commit on t32-media only, no push, no PR. Message: one-line subject + `Assisted-by: opencode`
   trailer, matching the other branch commits.
4. No test changes: the harness does not exercise server_tokens, and all existing checks pass.
