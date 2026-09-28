# KV tree

Purpose: reuse stable KV prefixes across agent sessions, so a server slot that switches
between long conversations can restore a parked prefix from RAM or SSD instead of
re-prefilling it. Attention KV is stored in content-addressed blocks (deduplicated by
prefix), recurrent state in anchors; a restore loads the deepest anchor-covered prefix and
re-prefills only the remainder. Built for T32 (stages 0-5b, media follow-up); off by
default (`--kv-tree`).

## Data model

### Blocks (attention KV)

- A block is one token range of a sequence plus its serialized attention KV: `hash`,
  `tok0` (start token index), `pos0`/`pos1` (real positions), `media` spans, `refcount`,
  `heat`, `last_used`, `on_disk`, `pinned`, `transient`, `path`, `bytes`, `tokens`, `data`.
- Chunking: `--tree-chunk` tokens per full block (default 512, locked by the stage 1
  range bench), with one possible shorter trailing block. A block end is extended to the
  end of a media chunk, so chunks never straddle blocks.
- Identity: chained content hash `h_i = XXH64(block_i tokens, seed = h_{i-1})`, with the
  media identity of covered chunks mixed in. Same prefix -> same block -> radix tree; a
  fork is a hash-chain fork. A hash hit is always re-verified with a token `memcmp`; a
  collision is a miss.
- Metadata operations (split, dedup, refcount updates) never move KV payload, and each
  payload is stored exactly once, in RAM or on SSD, never in both.
- Draft (MTP) KV is not stored per block: only the draft context's partial recurrent
  state travels with an anchor (`data_dft`, often empty), following the checkpoint
  slim-down decision.

### Anchors (recurrent state) and sequences

- An anchor is a recurrent snapshot (`PARTIAL_ONLY` blob) at one captured position:
  `blk_hash` (hash of the containing block), `tok` (token index), `pos` (real position),
  `kind` (TIP | MESSAGE | ONDEMAND), refcount/heat/tier, `data_tgt`, optional `data_dft`.
- Hard constraint: an anchor can only be created where the state was actually captured
  (sequence tip, an existing slot checkpoint, or after a replay); it is never synthesized.
- TIP: sequence end, captured on every park. MESSAGE: adopted from slot context
  checkpoints, spaced by `--tree-checkpoint-anchor-step` (default 32768). ONDEMAND: fork
  anchor captured by heal, spaced by `--tree-checkpoint-fork-step` (default 8192);
  MESSAGE guesses never suppress an ONDEMAND fork anchor.
- Anchors are keyed by `(blk_hash, pos)` and validated through the already verified block
  chain, so they carry no token list of their own.

- Stored sequence: `{tip hash, chain, length, last_used, pin}` in `seqs`; `blocks_at`
  (`tok0`) and `blocks_by_pos0` (`pos0`) index blocks for tail matching and anchors.

### Media identities and the tok/pos mapping

- `kv_tree_media {idx, id, n_tok, n_pos}`: start token index, opaque identity hash
  (server: FNV-1a 64 over `mtmd_input_chunk_get_id` = sha256 hex of the raw media bytes),
  and token/position counts of one chunk; spans are sorted, whole and disjoint.
- Media identity is part of the block hash, so a request with a different image, audio or
  video cannot restore from that chain; an empty chunk id means unknown identity.
- Why the mapping is needed: for M-RoPE an image chunk of `n_tok` tokens advances only
  `n_pos` positions (`n_pos < n_tok`), and all cells of a chunk share its start position.
  The tree matches on token indices but does KV I/O, anchor keys and spacing in positions.
- Helpers (`pos_at`, `pos_last_cell`, `pos_anchor`, `block_end`, `chain_split`,
  `chain_hashes`, `in_media_chunk`) are only valid at token 0, at the sequence end, at
  chunk starts and at chunk-aligned block boundaries. `block_end` extends a block over a
  straddling chunk, so no boundary or restore point ever falls inside a chunk.
- The tree stores no media bytes; the restored slot prompt is rebuilt from the incoming
  request (`server_tokens::clone()` and `keep_first`), preserving the media map.

## Lifecycle

### Park (store a finished sequence)

1. Guard: `io.pos_max() == pos_last_cell(media, L)`; the sequence must end exactly on its
   last KV cell (the position mapping handles a media-ending prompt).
2. Compute the chained hashes; an already stored tip only refreshes `last_used`.
3. Serialize blocks missing after the deepest shared prefix with
   `get_range(pos_at(a), pos_at(b))`; new blocks start in RAM with refcount 1, shared path
   blocks get `refcount++`, `heat++` and a new `last_used`.
4. Capture the tip anchor with `get_partial(PARTIAL_ONLY)` for the target context and,
   when present, the draft context; this is mandatory.
5. Adopt checkpoint candidates (message boundaries): sort by `tok`, greedy spacing with
   `--tree-checkpoint-anchor-step`, keep the earlier one on conflict, skip candidates
   inside a chunk or without a containing block on the parked chain.
6. Book the sequence (tip, chain, length, last_used).
- Every refusal (end mismatch, range read failure, RAM budget, anchor capture failure)
  is explicit: WRN plus a `park_refused` counter, never silent.

### Restore (deepest anchor-covered prefix)

1. `match()` verifies the longest block chain of the request: hash lookup plus per-token
   `memcmp`, with a media clamp that never verifies into the middle of a chunk. `deep` is
   the deepest verified token index.
2. Collect anchors with `tok <= deep` on the matched path; `C` is the largest such `tok`.
   `C <= 0` is a miss.
3. Load blocks with `tok0 < C`, plus the partially matched block when it starts before C,
   with `append=true`; trim with `seq_rm(pos_at(C), -1)`; install the anchor state at C
   with `set_partial` for target and draft.
4. Rebuild `prompt.checkpoints` from the returned path anchors: keep `0 < tok <= C`, cap
   by `n_ctx_checkpoints` and a 256 MiB burst, drop the shallowest first. Recurrent
   (FULL/RS) contexts use tail semantics (`pos_min = pos_max = pos - 1`); PART contexts
   keep `pos_min = 0`.
5. Return `res.C` (token index) and `res.heal = deep > C ? deep : -1`, with `leave_one`
   capping C when the memory cannot roll back, so at least one token stays unprocessed
   for logits.
- Degrade paths: no anchor -> full prefill (INF + counter); SSD read failure -> the block
  counts as missing (shallower anchor or full prefill); pool failure -> full prefill.

### Heal anchors

- On a shallow restore (`C < deep`) or a miss, `res.heal` carries the divergence token
  index. The server keeps it across `prompt_clear` (only when `cache_prompt=true`), stops
  the prefill batch exactly at that index and captures an ONDEMAND anchor there.
- The engine re-verifies the position (`n_tokens == heal`, `pos_max == pos - 1`);
  `tree_heal` is cleared unconditionally when the slot is selected for the next task.
- A fork replayed once is free afterwards; a task ending exactly at the heal position
  skips the capture (benign, retried next time).

### drop_seq

- `drop_seq(tokens, media)` matches the stored sequence whose tip (or partial tail) is
  covered by the request tokens and releases it, walking block and anchor refcounts.
  Used by `POST /slots/{id}?action=erase`.

### Eviction and tiering (RAM / SSD)

- Limits: `--tree-ram` (default 8192 MiB), `--tree-disk` (directory; empty = no SSD
  tier) and `--tree-disk-limit` (default 65536 MiB).
- Placement: RAM prefers high-refcount trunk blocks, high heat and blocks near anchors;
  SSD prefers refcount 1 leaf blocks. New payloads start in RAM, a demotion moves the
  single authoritative copy, and an SSD hit is loaded back. A restore can stream a disk
  block through a scratch buffer when RAM is full; anchors stay resident (one at a time,
  after which `settle()` restores the budget).
- Eviction order when space is short: (1) anchors, coldest and lowest coverage first
  (they are rebuildable by replay); (2) leaf blocks with `refcount <= 1` and no
  successor; (3) whole leaf sequences, walking refcounts down to 0; (4) refuse the
  park/restore explicitly (WRN + counter). Shared trunk blocks (`refcount >= 2`) are
  touched last.
- Pinning: after a restore target is selected its whole path is pinned; eviction skips
  pinned payloads. If everything is pinned and space is still short, the park of the
  working sequence is refused (deferred cost) rather than dropping the target (immediate
  cost).
- Disk envelope: every SSD file has a header (magic, version, hash, position, payload
  sizes, payload hash); a failed check drops the block (`disk_errors`), never trusts it.

### Startup

- No persistence: the in-memory index dies with the process, and with `--tree-disk` the
  constructor wipes `blocks/` and `anchors/` (including `.tmp`) and logs the stale count.
  The directory must be dedicated to the tree; never point it at other data (the tree
  writes continuously while enabled: SSD wear and space).

## Invariants

1. Attention KV is a pure function of the prefix (blocks are shared); recurrent state is
   not, so it is reachable only through anchors.
2. An anchor exists only where a state was captured; park and restore never synthesize one.
3. Block identity is the chained hash of tokens plus media identity; every hit is
   re-verified token by token and a mismatch is a miss (no silent reuse).
4. A media chunk is atomic: no block boundary, restore point or anchor position falls
   inside a chunk, and a chunk always belongs entirely to one block.
5. Without media, every path is behavior-identical to the text-only tree (token index ==
   position, no mapping offsets).
6. Token indices (`C`, `heal`, candidates) and positions (KV I/O, anchor keys, spacing)
   are different units, related only through the mapping helpers; boundaries and restore
   points are chunk-aligned.
7. The range API never overwrites: appending positions that already exist fails and
   returns 0; on failure the engine cleans up only the cells it allocated.
8. Park stores only what the sequence has: the end must be exactly the last captured
   position, and only blocks missing after the deepest shared boundary are serialized.
9. Restore leaves at least one token unprocessed when the memory cannot roll back
   (`leave_one`), so the server can obtain logits.
10. Heal capture requires an exact match (`n_tokens == heal`, `pos_max == pos - 1`), so an
    anchor cannot land at the wrong position.
11. Fork spacing (`fork_step`) applies between ONDEMAND captures and ignores MESSAGE
    guesses; message spacing (`anchor_step`) applies to the guesses.
12. Eviction never drops pinned payloads, and every degradation is visible in the log and
    metrics; the tree never fails silently.
13. The tree stores no media bytes and no spec/MTP sampling state (gap D2); anything the
    caller did not capture cannot be restored.

## Options

(also settable via the `LLAMA_ARG_*` env vars; copied from FORK-NOTES.md)

| option | default | meaning |
|---|---|---|
| `--kv-tree` / `--no-kv-tree` | disabled | enable the tree |
| `--tree-chunk N` | 512 | block size in tokens |
| `--tree-checkpoint-anchor-step N` | 32768 | minimum spacing of checkpoint (message) anchors in tokens |
| `--tree-checkpoint-fork-step N` | 8192 | minimum spacing of fork anchors in tokens |
| `--tree-ram N` | 8192 | RAM tier limit in MiB |
| `--tree-disk PATH` | empty | SSD tier directory (empty = no SSD tier) |
| `--tree-disk-limit N` | 65536 | SSD tier limit in MiB |
| `--tree-debug` / `--no-tree-debug` | disabled | dump the tree after each park/restore |

Notes: spacing options are non-negative (0 = no minimum); MESSAGE spacing compares against
all anchors, ONDEMAND spacing only against other non-guess anchors. `--tree-debug` dumps
after each park/restore, and the server logs `stats_line()` every 64 park+restore
operations.

## Integration points

- `tools/server/server-kv-tree.{h,cpp}`: the module; pure logic sits behind the
  `kv_tree_io` seam (fake IO in tests, `kv_tree_io_llama` on the server) and the module
  does not reference llama_context directly.
- `tools/server/server-context.cpp`: `server_slot::prompt_park`,
  `prompt_restore_tree` (checkpoint rebuild, heal, `tree_restore_point`), heal capture in
  the prefill loop, idle-slot park, `SLOT_ERASE` drop, tree creation at model load
  (including the `--kv-unified` check) and the periodic `kv tree stats` line.
- `common/common.h`, `common/arg.cpp`: `common_params` fields and the CLI options.
- Range API dependency: `llama_state_seq_get_size_range_ext`,
  `llama_state_seq_get_data_range_ext` and `llama_state_seq_set_data_range_ext`.
  `llama_memory_hybrid` forwards range calls to its attention component only; recurrent
  state uses the existing `PARTIAL_ONLY` state API.
- Media: `server_tokens::pos_last()`, the media map accessor, `tree_media_spans()` and
  `mtmd_input_chunk_get_id/n_tokens/n_pos`.
- Tests: `tests/test-t32-tree.cpp` (logic with fake IO, model scenarios on a 2B hybrid,
  accept A/B) and `tests/test-t32-range.cpp` (range API correctness and throughput); the
  FORK-NOTES rebase checklist lists the harness and server smoke commands.

## Known limits

- Not supported with `--kv-unified`: the server logs a warning and disables the tree.
- Needs a memory type that implements the range state API: plain KV and the attention
  part of hybrid. iswa / dsv4 / dsa / msa / hybrid-idx and mirrored caches refuse it:
  error logged, park refused, server keeps running; only qwen35 was tested.
- No persistence: the disk tier is wiped at startup and the in-memory index dies with the
  process. Never point `--tree-disk` at data you want to keep.
- Speculative sampler bookkeeping is not carried by anchors (D2). A restore reloads the
  draft context's recurrent part (`data_dft`) and the draft KV is dropped from checkpoints,
  but the per-task speculative state is rebuilt from scratch; the 27B/MTP deployment
  accepted this.
- Media prompts keep the upstream gates for stock context checkpoints, `n_cache_reuse`
  and speculative state. The tree stores no media bytes, and a media chunk is atomic, so
  a very large video becomes one very large block (coarse budget granularity).
- Floating point: a full prefill and a restore into a fragmented (np > 1) cache can
  differ in reduction order, and greedy near-ties may flip wording; each path is
  deterministic by itself.
- v1 I/O is synchronous and the scheduler is single-threaded: park and restore block the
  server for the transfer duration; asynchronous I/O is out of scope.
- Multi-GPU is untested (developed on a single V100). Layer split is expected to work for
  the tree (same range state path as the full state save); row split was not validated.
- Context shift: the tree acceptance runs disable it; a park is skipped when the sequence
  does not end exactly at its last captured position (truncation, front removal), and the
  request degrades to a full prefill.
- Lora adapters skip park/restore, and requests with `cache_prompt=false` skip restore.
  The checkpoint rebuild is capped both by `n_ctx_checkpoints` and by a 256 MiB burst.
- SSD wear optimization (write-once, clean copies, cold eviction) is deferred; the soak
  only records IO counters as a baseline.

Source:
- archive/v100-collab/artifacts\t32-tree-storage-design.md
- archive/v100-collab/artifacts\t32-media-tree-spec.md
- archive/v100-collab/artifacts\t32-media-tree-plan.md
- archive/v100-collab/artifacts\t32-tree-plan-stage0-1.md
- archive/v100-collab/artifacts\t32-tree-plan-stage2.md
- archive/v100-collab/artifacts\t32-tree-plan-stage3.md
- archive/v100-collab/artifacts\t32-tree-plan-stage4.md
- archive/v100-collab/artifacts\t32-tree-plan-stage5.md
- archive/v100-collab/artifacts\t32-tree-plan-stage5b.md
- archive/v100-collab/TASKS\T32-agent-session-reuse.md
- archive/v100-collab/RESULTS.md
- FORK-NOTES.md
- tools\server\server-kv-tree.h
- tools\server\server-context.cpp
- common\common.h
- common\arg.cpp
