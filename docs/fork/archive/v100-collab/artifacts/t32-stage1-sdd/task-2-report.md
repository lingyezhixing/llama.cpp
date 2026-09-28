# Task 2 Report: range state read API (append semantics) + overlap rejection

## What was implemented

Followed the brief's steps 1-2. Library/API code matches the brief verbatim; two test-only
deviations were required (see "Deviations" below).

1. `src/llama-kv-cells.h`: added `seq_pos_has(seq_id, p)` after `seq_pos_max`.
2. `src/llama-memory.h`: added base `llama_memory_i::state_read_range(io, seq_id, append, flags)`
   (default throws) after `state_write_range`.
3. `src/llama-kv-cache.h`:
   - public `state_read_range` override next to `state_write_range`;
   - `state_read_sinfo(..., bool append = false)` with updated comment;
   - private `state_read_meta(..., bool append = false)`, `state_clear_append`, `state_zero_data`.
4. `src/llama-kv-cache.cpp`:
   - `state_read_range`: rejects `seq_id < 0` and `flags != 0`, forwards to `state_read_sinfo`;
   - `state_read_sinfo`: takes/forwards `append`, failure branch picks `state_clear_append` vs `state_clear`;
   - `state_read_meta`: `append && sinfo_in` rejected, `seq_rm` skipped on append, per-position overlap
     check (`cells.seq_pos_has(dest_seq_id, pos)`) inserted right after the `n_seq_id != 1` check;
   - `state_clear` split into `state_clear` + `state_clear_append` + `state_zero_data` (body moved verbatim).
5. `src/llama-memory-hybrid.h/.cpp`: `state_read_range` forwarding to `mem_attn` (parallel to the write side).
6. `src/llama-memory-hybrid-idx.h/.cpp`: throwing `state_read_range` override (controller ruling),
   exactly parallel to the existing `state_write_range` override.
7. `include/llama.h`: `llama_state_seq_set_data_range_ext` declaration.
8. `src/llama-context.h/.cpp`: `state_seq_set_data_range` (host io, try/catch -> LLAMA_LOG_ERROR + 0,
   `io.discard()`) and the public C wrapper (calls `synchronize()`); no exception can escape the C API.
9. `tests/test-t32-range.cpp`: `generate()` helper and the chunked-restore / append / overlap checks.

## Commands run and results

Build (all runs used the resulting binary):

```
<TEMP>\v100\build_test_t32.cmd test-t32-range
-> [1/2] Building tests\...\test-t32-range.cpp.obj; [2/2] Linking ... test-t32-range.exe (no errors)
```

### Run 1: 2B hybrid (non-unified)

```
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```

```
0.01.631.769 W llama_context: n_ctx is not divisible by n_seq_max - rounding down to 768
0.01.643.349 I cmn          init: llama threadpool init, n_threads = 8
[t32] hybrid=1 full=20990228 partial=20202092 range=788144
[t32] full state size > 0                                                      PASS
[t32] range state size > 0                                                     PASS
[t32] hybrid: range(0,L) size == full - partial + 8                            PASS
[t32] range(0,L/2) size is between header and full range                       PASS
[t32] read chunk 0                                                             PASS
[t32] read chunk 1                                                             PASS
[t32] read recurrent state                                                     PASS
[t32] restore recurrent state into seq 1                                       PASS
[t32] restore chunk 0 (append=false)                                           PASS
[t32] restore chunk 1 (append=true)                                            PASS
[t32] read full state                                                          PASS
[t32] restore full state into seq 2                                            PASS
[t32] chunked restore generates the same tokens as a full restore              PASS
0.01.958.975 E state_read_meta: position 0 is already present in seq 1
[t32] append with overlapping positions is rejected                            PASS
0.01.959.022 E state_seq_set_data_range: error loading range state: failed to restore kv cache
[t32] rejected append left the existing state intact                           PASS
EXIT=0
```

### Run 2: 3B pure attention

```
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen2.5-Coder-3B-IQ4_XS.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```

```
0.00.556.263 W load: control-looking token: 128247 '</s>' was not control-type; this is probably a bug in the model. its type will be overridden
0.01.496.260 W llama_context: n_ctx is not divisible by n_seq_max - rounding down to 768
0.01.505.097 I cmn          init: llama threadpool init, n_threads = 8
[t32] hybrid=0 full=2360960 partial=2360960 range=2360960
[t32] full state size > 0                                                      PASS
[t32] range state size > 0                                                     PASS
[t32] attn: range(0,L) size == full size                                       PASS
[t32] payload reads returned the expected sizes                                PASS
[t32] attn: range(0,L) payload == full payload                                 PASS
[t32] range(0,L/2) size is between header and full range                       PASS
[t32] read chunk 0                                                             PASS
[t32] read chunk 1                                                             PASS
[t32] restore chunk 0 (append=false)                                           PASS
[t32] restore chunk 1 (append=true)                                            PASS
[t32] read full state                                                          PASS
[t32] restore full state into seq 2                                            PASS
[t32] chunked restore generates the same tokens as a full restore              PASS
0.01.805.107 E state_read_meta: position 0 is already present in seq 1
[t32] append with overlapping positions is rejected                            PASS
0.01.805.156 E state_seq_set_data_range: error loading range state: failed to restore kv cache
[t32] rejected append left the existing state intact                           PASS
EXIT=0
```

(`read recurrent state` / `restore recurrent state into seq 1` are skipped for hybrid=0, as expected.)

### Run 3: 2B unified (`-kvu`)

```
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -np 3 -c 512 -kvu --mode correctness
```

```
0.01.639.342 I cmn          init: llama threadpool init, n_threads = 8
[t32] hybrid=1 full=20990220 partial=20202092 range=788136
[t32] full state size > 0                                                      PASS
[t32] range state size > 0                                                     PASS
[t32] hybrid: range(0,L) size == full - partial + 8                            PASS
[t32] range(0,L/2) size is between header and full range                       PASS
[t32] read chunk 0                                                             PASS
[t32] read chunk 1                                                             PASS
[t32] read recurrent state                                                     PASS
[t32] restore recurrent state into seq 1                                       PASS
[t32] restore chunk 0 (append=false)                                           PASS
[t32] restore chunk 1 (append=true)                                            PASS
[t32] read full state                                                          PASS
[t32] restore full state into seq 2                                            PASS
[t32] chunked restore generates the same tokens as a full restore              PASS
0.01.951.077 E state_read_meta: position 0 is already present in seq 1
[t32] append with overlapping positions is rejected                            PASS
0.01.951.132 E state_seq_set_data_range: error loading range state: failed to restore kv cache
[t32] rejected append left the existing state intact                           PASS
EXIT=0
```

All 14/16 checks PASS per run, exit code 0. The overlap attempt returns 0 (logged error
"position 0 is already present in seq 1"), and the following generation check proves seq 1
still generates exactly like the untouched seq 2.

Note: unified full/range sizes are 8 bytes smaller than non-unified (20990220 vs 20990228).
This is the context header accounting for the unified stream count and is benign; all size
relations and payload checks pass.

## Deviations from the brief (test code only, library/API code is verbatim)

**Deviation 1 - the brief's final check is impossible as written.**
The brief's last check was:

```cpp
    const auto gen1b = generate(ctx, 1, first, L, 8);
    check(gen1b == gen1, "rejected append left the existing state intact");
```

`gen1` already decoded 8 tokens at positions L..L+7 (64..71) into seq 1, so a second decode
starting at L is rejected by the batch position consistency rule before any KV work happens
(M-RoPE: `X < Y`, non-M-RoPE: `Y = X + 1`; `src/llama-batch.cpp:287-353`). Exact failure
observed with the verbatim brief test (2B hybrid):

```
[t32] append with overlapping positions is rejected                            PASS
0.01.904.454 E init: the tokens of sequence 1 in the input batch have inconsistent sequence positions:
 - the last position stored in the memory module of the context (i.e. the KV cache) for sequence 1 is X = 71
 - the tokens for sequence 1 in the input batch have a starting position of Y = 64
 for M-RoPE, it is required that the position satisfies: X < Y
0.01.904.456 E decode: failed to initialize batch
[t32] rejected append left the existing state intact                           FAIL
EXIT=1
```

The failure is independent of the rejected append (it is a consequence of `gen1` itself), so
no implementation could make the original line pass. The minimal replacement keeps the check
name, output position, and intent (prove seq 1 survived the rejected append) by comparing the
next generation chunk of seq 1 against untouched seq 2:

```cpp
    // gen1/gen2 already decoded past L, so continue both: seq 1 went through the rejected append, seq 2 did not
    const llama_token next = gen1.empty() ? first : gen1.back();

    const auto gen1b = generate(ctx, 1, next, L + 8, 8);
    const auto gen2b = generate(ctx, 2, next, L + 8, 8);

    check(gen1b == gen2b, "rejected append left the existing state intact");
```

Before this check, the previous check already proves `gen1 == gen2` (chunked restore == full
restore), so the two sequences are identical states except for the rejected append. Any
corruption by the failed append (K/V zeroed, cells dropped, positions moved) would diverge
the continuations. This satisfies the controller's stated intent: overlap rejection returns 0
and the following generation check proves the existing state survived.

**Deviation 2 - `-kvu` cannot reach the harness with `LLAMA_EXAMPLE_COMMON`.**
`common/arg.cpp:1727` gates `-kvu/--kv-unified` to
`{SERVER, PERPLEXITY, BATCHED, BENCH, PARALLEL}`, so the brief's unified command failed with
`error: invalid argument: -kvu` (exit 1) before any check ran. `common/arg.cpp` is not in the
allowed file list, so the harness now intercepts `-kvu`/`--kv-unified` in its own arg loop and
sets `params.kv_unified = true` (same approach as `tests/test-state-restore-fragmented.cpp:21`),
keeping the brief's exact command line working.

## Files changed, commit

Commit (created, not pushed):

```
4ae887b1ca7425b3357162ee01bde6e39bc0902b
4ae887b1c t32 : add range state read API with append mode
```

Trailer verified via `git log -1 --format=%B`: `Assisted-by: opencode`. Working tree clean
after commit.

Diffstat:

```
 include/llama.h                 | 12 +++++++
 src/llama-context.cpp           | 30 ++++++++++++++++
 src/llama-context.h             |  2 ++
 src/llama-kv-cache.cpp          | 61 +++++++++++++++++++++++++++----
 src/llama-kv-cache.h            | 11 ++++--
 src/llama-kv-cells.h            | 11 ++++++
 src/llama-memory-hybrid-idx.cpp |  9 +++++
 src/llama-memory-hybrid-idx.h   |  2 ++
 src/llama-memory-hybrid.cpp     |  4 +++
 src/llama-memory-hybrid.h       |  2 ++
 src/llama-memory.h              | 11 ++++++
 tests/test-t32-range.cpp        | 80 +++++++++++++++++++++++++++++++++++++++++
 12 files changed, 229 insertions(+), 8 deletions(-)
```

The brief's `git add` list omitted `src/llama-memory-hybrid-idx.{h,cpp}`; both were added per
the controller ruling. No other files touched.

## Self-review findings and concerns

Verified before commit:

- Library/API diff matches the brief verbatim (all snippets: `seq_pos_has`, base default-throw,
  kv-cache declarations/implementations, `state_clear` split, hybrid forwarder, hybrid-idx
  throwing override, `llama.h`, context plumbing). The only non-brief changes are the two
  test deviations above.
- No exception can escape the C API: `state_seq_set_data_range` catches everything, logs
  `LLAMA_LOG_ERROR`, discards the io and returns 0. `flags != 0` and `seq_id < 0` throw in
  `state_read_range` and are converted to 0. Verified in-run: the overlap attempt returns 0.
- Append failure paths: overlap is detected in the metadata loop before `find_slot`/`apply_ubatch`,
  so `sinfo` is empty and `state_clear_append` is a no-op (destination untouched - proven by the
  continuation check). If `state_read_data` fails after `apply_ubatch`, `state_clear_append`
  detaches exactly the cells in `sinfo` and zeroes their K/V data.
- `seq_pos_has` uses `lower_bound({p, 0})` on `std::set<std::pair<llama_pos, uint32_t>>`; correct
  when several cells share a position (only the first entry matters for presence).
- `llama_memory_hybrid_idx::state_read_range` throws even when `mem_idx == nullptr`, matching the
  class's existing `state_write_range` choice (loud failure instead of a silently incomplete
  payload). Not reachable from the two test models; compile-time coverage only.
- Both test deviations were exercised by all three final runs; all green, exit 0.

Concerns for the controller:

1. The `gen1b == gen2b` continuation check relies on greedy determinism, same as the brief's
   `gen1 == gen2`; all three runs are stable.
2. `gen1.empty()` guard uses `first` as the continuation token to avoid `.back()` on an empty
   vector; if generation were broken, the preceding `!gen1.empty()` check already fails the run.
3. `-kvu` interception lives in the test harness; if a later task wants `-kvu` in more tests,
   the same pattern is needed (or `common/arg.cpp` gating would have to change - out of scope here).
