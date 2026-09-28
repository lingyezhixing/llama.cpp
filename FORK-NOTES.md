# Fork notes

This is a personal fork of llama.cpp. It carries private work on top of `upstream/master`.
It is not upstream-supported. Use at your own risk.

## What this fork adds

1. `ggml-cuda` V100 (sm70) tuning: K-quant vector dequant, FATTN split, sm70 MMA configs.
2. ReplaySSM: GDN recurrent state replayed from per-batch records instead of per-token snapshots
   (`ggml_gated_delta_net` gained `rec`/`fold`/`diag` arguments; opt-in via `GGML_CUDA_GDN_REPLAY=1`).
3. `server`: drop the draft KV from context checkpoints.
4. Range state API: `llama_state_seq_get_data_range_ext` / `llama_state_seq_set_data_range_ext`
   (save/load a position range of one sequence).
5. KV tree (`--kv-tree`): content-addressed RAM+SSD storage of stable KV prefixes, reused across
   tasks; includes media (image/audio/video) support via chunk identity and token/position mapping.
   - `tools/server/server-kv-tree.{h,cpp}` - the module
   - `tools/server/server-context.cpp` - park/restore/erase/heal integration
   - `common/arg.cpp`, `common/common.h` - options and `common_params`
   - `tests/test-t32-tree.cpp`, `tests/test-t32-range.cpp` - harnesses

KV tree options (also settable via the `LLAMA_ARG_*` env vars):

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

Tuning env vars of the V100 FATTN work (leave unset for production):

| variable | meaning |
|---|---|
| `GGML_CUDA_FATTN_STREAM_K=0/1` | disable / force the stream-K heuristic (default: heuristic) |
| `GGML_CUDA_FATTN_BLOCKS=N` | override the stream-K block count |
| `GGML_CUDA_FATTN_PB=N` | override the KV split per output tile (BLOCKS wins over PB) |

The BLOCKS/PB overrides are only consulted when stream-K is used.

## Known limitations

- A media chunk is atomic: one chunk belongs to one block; a very large video becomes one very
  large block (coarse budget granularity).
- Stock context checkpoints, `n_cache_reuse` and speculative state stay gated for media prompts
  (upstream behavior).
- On hybrid/recurrent models without rollback support (`llama_n_rs_seq(ctx) == 0`) a restore
  leaves at least one token unprocessed, so the server can evaluate it and obtain logits.
- A full prefill and a restore into a fragmented (np > 1) cache can differ in floating point
  reduction order; greedy near-ties may flip wording. Each path is deterministic by itself.
- `--tree-disk` is cleared at startup: never point it at a directory that holds other data.
  The tree writes to it continuously while enabled (SSD wear and space).
- The V100 tuning and the ReplaySSM path are hardware/fork specific; expect different behavior
  or performance on other GPUs.
- `--kv-tree` is not supported with `--kv-unified`; the server logs a warning and disables the tree.

## Portability notes

- State files are fork specific: `LLAMA_STATE_SEQ_VERSION` is 6 here (upstream is 3). Slot saves
  and state files are not interchangeable between this fork and upstream, in either direction.
- The V100 FATTN configs live in the Volta table only. The Q5_K/Q6_K vectorized dequant replaces
  the fp16 dequant dispatch for every CUDA arch; the math matches the scalar path (verified on
  V100), but it is untuned and untested on sm80+.
- `GGML_CUDA_GDN_REPLAY=1` requires all GDN layers on CUDA. Records are written by the CUDA kernel
  only; with GDN layers on CPU (CPU-only or partial offload) the host bookkeeping still accepts
  record rollbacks, so a rollback silently keeps the stale CPU state. Do not enable it without
  full offload.
- The KV tree needs a memory type that implements the range state API: plain KV and hybrid
  (attention part). iswa / dsv4 / dsa / msa / hybrid-idx and mirrored KV caches refuse the range
  calls: an error is logged, the park is refused and the server keeps running. Only the qwen35
  family has been tested.
- The tree file I/O uses std::filesystem and stdio with forward-slash paths, no platform-specific
  code, but it has only been built and tested on Windows/MSVC. Build on Linux before shipping.
- On Windows, `--tree-disk` with a non-ASCII path may fail (narrow fopen).
- Multi-GPU is untested (the fork was developed on a single V100). Layer split should work for the
  tree (the range state path is the same as the full state save) and for ReplaySSM (records are
  allocated per layer on the layer device). Row split breaks ReplaySSM: the kernel derives the
  record layout from the tensor strides, which a row split changes. Keep the replay env var to
  layer split with all GDN layers on CUDA.

## Standardized rebase workflow

Remotes:

- `origin`  - this personal fork
- `upstream` - `https://github.com/ggml-org/llama.cpp.git`

Private commits always sit directly on top of `upstream/master`. Rebase, never merge upstream in.

### 1. Preconditions

```sh
git status                     # clean tree, on master
git log --oneline -1           # note the current head
git branch backup-pre-rebase-$(date +%Y%m%d)   # safety net
```

### 2. Fetch and survey

```sh
git fetch upstream
git merge-base master upstream/master          # the base our commits sit on
git log --oneline --no-merges <base>..upstream/master | head -40
git diff --stat <base>..master                 # our private delta
git diff --stat <base> upstream/master         # upstream delta
```

### 3. Conflict and semantic-conflict checks

Textual overlap: compare the two name-only lists and inspect the shared files.

```sh
comm -12 <(git diff --name-only <base>..master | sort) \
         <(git diff --name-only <base> upstream/master | sort)
```

Semantic checks that a clean rebase does NOT catch:

- `ggml_gated_delta_net`: the fork adds `rec`, `fold`, `diag`. Grep every call site and make sure
  new upstream code passes the extra arguments.
- `llama_state_seq_*_range_ext`: fork-only API; check for new upstream state APIs that overlap.
- `common_params`: fork fields (`kv_tree`, `tree_*`); check upstream param parsing changes.
- `server-context.cpp`: the fork touches park/restore/erase/heal and the prompt loop; check
  upstream changes in the same areas (reuse logic, checkpoints, post_decode, spec).
- CUDA config tables: upstream FA config tuning must survive next to the sm70 additions.

### 4. Rebase

```sh
git rebase upstream/master
```

If a conflict appears, resolve it keeping both intents (never drop an upstream change), then
`git rebase --continue`. Re-run the semantic checks from step 3 on the result.

### 5. Verification checklist (all must pass before pushing)

```sh
# build everything (CUDA included)
cmake --build build --config Release -j        # or the per-target scripts

# harnesses (CPU logic and a small model on GPU)
build/bin/Release/test-t32-tree.exe --mode logic
build/bin/Release/test-t32-tree.exe -m <model.gguf> -ngl 99 -fa on --mode model --ram-mib 4096 -c 8192
build/bin/Release/test-t32-range.exe -m <model.gguf> -ngl 99 -fa on

# merged CUDA op tests (validates V100 + ReplaySSM + upstream changes together)
build/bin/Release/test-backend-ops.exe -b CUDA0 -o GATED_DELTA_NET
build/bin/Release/test-backend-ops.exe -b CUDA0 -o SSM_SCAN

# server smoke: text reuse and media reuse
#   1) text: two requests sharing a prefix -> prompt_n drops on the second
#   2) media: image request twice -> "kv tree: restored ..." with a small prompt_n
#   3) media identity: a different image at the same position -> restore miss
#   4) greedy outputs equal to a run without --kv-tree
```

Also check `--help` still lists the `--tree-*` options and the tree still starts with
`kv tree enabled: ...` in the log.

### 6. Push and deploy

```sh
# only with the owner's explicit approval
git push --force-with-lease origin master
```

Deploy (Windows, production directory):

1. Stop the running server (files are locked while it runs).
2. Copy the new binaries to the production directory, backing up the old ones first.
3. Verify SHA256 of every copied file.
4. Start the production binary once and check the tree log lines.
5. Keep the backup directory path for rollback (copy the files back and restart).

### 7. Record

Append what changed, the verification results and the deployed hashes to the development
record in `docs/fork/`.
