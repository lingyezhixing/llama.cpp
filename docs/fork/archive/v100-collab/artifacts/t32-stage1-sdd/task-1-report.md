# Task 1 Report: range state write API (attention KV only)

## What was implemented

Followed the brief's steps 1-5 exactly:

1. **Step 1 - declarations + throwing stubs**
   - `include/llama.h`: declared `llama_state_seq_get_size_range_ext` and `llama_state_seq_get_data_range_ext` after `llama_state_seq_set_data_ext`.
   - `src/llama-context.h`: declared private `state_seq_get_size_range` / `state_seq_get_data_range` after `state_seq_set_data`.
   - `src/llama-memory.h`: added `#include <stdexcept>` and the base-class virtual `llama_memory_i::state_write_range` whose default body throws `std::runtime_error("state_write_range is not supported by this memory type")`.
   - `src/llama-context.cpp`: added `llama_context::state_seq_get_size_range` (dummy io) and `state_seq_get_data_range` (host io) with try/catch -> `LLAMA_LOG_ERROR` + return 0; added the two public C wrappers (data wrapper calls `ctx->synchronize()`).
   - No kv-cache / hybrid changes yet, so the base-class throw is what runs at this point.

2. **Step 2 - write-side test**
   - `tests/test-t32-range.cpp`: added `n_fail`, `check()` and `run_correctness()` (as specified in the brief) and wired `--mode correctness` in `main`.

3. **Step 3 - RED confirmed** (see evidence below).

4. **Step 4 - implementation**
   - `src/llama-kv-cache.h`: public override `state_write_range` after `state_read`; private `state_write_impl` after `state_write_data`.
   - `src/llama-kv-cache.cpp`: replaced the old `state_write` body with three functions: `state_write` -> `state_write_impl(io, seq_id, -1, -1, flags)`; `state_write_range` (validates mirrored cache, `seq_id < 0 || p0 < 0 || p1 <= p0`, `flags != 0`); `state_write_impl` (old body plus the `cells.pos_in(i, p0, p1)` range filter when `p0 >= 0`). Existing `// TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]` comments were preserved.
   - `src/llama-memory-hybrid.h/.cpp`: `state_write_range` override forwarding to `mem_attn->state_write_range`.
   - `tests/test-t32-range.cpp` (from Step 2) is unchanged in Step 4.

5. **Step 5 - committed** (pre-authorized), see commit below. Not pushed.

Also ran `cmake -S . -B build -DLLAMA_BUILD_TESTS=ON` once and left it ON, as instructed. Builds were done via `<TEMP>\v100\build_test_t32.cmd test-t32-range`.

## TDD Evidence

### RED

Command (after Step 2 build, before Step 4 implementation):

```powershell
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```

Output (hybrid 2B):

```
0.01.655.041 I cmn          init: llama threadpool init, n_threads = 8
[t32] hybrid=1 full=20990228 partial=20202092 range=0
[t32] full state size > 0                                                      PASS
[t32] range state size > 0                                                     FAIL
[t32] hybrid: range(0,L) size == full - partial + 8                            FAIL
0.01.779.975 E state_seq_get_size_range: error getting range state size: state_write_range is not supported by this mem
ory type
[t32] range(0,L/2) size is between header and full range                       FAIL
0.01.780.010 E state_seq_get_size_range: error getting range state size: state_write_range is not supported by this mem
ory type
EXIT: 1
```

Why expected: Step 1 only installed the base-class `llama_memory_i::state_write_range` which throws; neither `llama_kv_cache` nor `llama_memory_hybrid` overrode it yet, so `range` size is 0 and the error log matches the stub's message. The same run was repeated for Step 3 and produced the identical shape. (PowerShell prints a `NativeCommandError` wrapper for stderr writes; that is shell noise, not test output.)

### GREEN

Command 1 (hybrid 2B):

```powershell
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```

Output:

```
0.01.696.623 I cmn          init: llama threadpool init, n_threads = 8
[t32] hybrid=1 full=20990228 partial=20202092 range=788144
[t32] full state size > 0                                                      PASS
[t32] range state size > 0                                                     PASS
[t32] hybrid: range(0,L) size == full - partial + 8                            PASS
[t32] range(0,L/2) size is between header and full range                       PASS
EXIT: 0
```

(size relation holds: 20990228 - 20202092 + 8 = 788144)

Command 2 (pure-attention 3B):

```powershell
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen2.5-Coder-3B-IQ4_XS.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```

Output:

```
0.01.682.180 I cmn          init: llama threadpool init, n_threads = 8
[t32] hybrid=0 full=2360960 partial=2360960 range=2360960
[t32] full state size > 0                                                      PASS
[t32] range state size > 0                                                     PASS
[t32] attn: range(0,L) size == full size                                       PASS
[t32] payload reads returned the expected sizes                                PASS
[t32] attn: range(0,L) payload == full payload                                 PASS
[t32] range(0,L/2) size is between header and full range                       PASS
EXIT: 0
```

## Files changed and commit

Commit (created, not pushed):

```
3d05ef570594afcc69ba61b53809dab865b2cf94
3d05ef570 t32 : add range state write API for attention KV
```

Diffstat:

```
 include/llama.h             | 19 +++++++++++++++++
 src/llama-context.cpp       | 44 ++++++++++++++++++++++++++++++++++++++
 src/llama-context.h         |  4 ++++
 src/llama-kv-cache.cpp      | 26 +++++++++++++++++++++++
 src/llama-kv-cache.h        |  5 +++++
 src/llama-memory-hybrid.cpp |  4 ++++
 src/llama-memory-hybrid.h   |  2 ++
 src/llama-memory.h          | 12 +++++++++++
 tests/test-t32-range.cpp    | 51 +++++++++++++++++++++++++++++++++++++++++++++
 9 files changed, 167 insertions(+)
```

No other files were touched; `git status` is clean after the commit.

## Self-review findings and concerns

- Verified the committed diff matches the brief verbatim for all snippets (declarations, stubs, wrappers, test code, kv-cache refactor, hybrid forwarder). The kv-cache `state_write` body is preserved as `state_write_impl`; the only behavioral addition is the `p0 >= 0` -> `cells.pos_in(i, p0, p1)` filter.
- Exception safety: every new path is inside try/catch in `llama_context`; `state_write_range` throws are converted to a log + return 0. No exception escapes the C API. `flags != 0` and invalid ranges fail explicitly.
- The full-attention 3B run shows `partial == full` size (2360960); this is existing behavior of `PARTIAL_ONLY` for a pure attention memory, and the test only uses it in the hybrid branch, so it does not affect the assertions.
- Observation (out of scope, no action taken): `llama_kv_cache_iswa` and `llama_memory_hybrid_iswa` derive directly from `llama_memory_i` and do not override `state_write_range`, so SWA/iSWA models get the base-class "not supported" throw -> logged error and 0 return. This is a safe, loud failure rather than a silent partial state, and matches the brief's scope (attention KV, flags == 0). SWA-only range semantics are presumably a later task.
- `llama_memory_hybrid_idx` derives from `llama_memory_hybrid` and inherits the new range forwarder unchanged.
- The test's `range(0,L/2) < range(0,L)` check relies on the cache being exactly filled with positions 0..63; `fill_context` fills 64 tokens at positions 0..63, so the half range has 32 cells and passes on both models.

# Fix report (review findings 1-2)

## What changed

**Finding 1 - sentinel handling in `state_write_impl` (`src/llama-kv-cache.cpp`)**

The header comment documented `p0 < 0` as "no lower bound" and `p1 < 0` as "no upper bound", but the filter only ran when `p0 >= 0` and called `cells.pos_in(i, p0, p1)`, so `p0 >= 0, p1 < 0` selected zero cells and `p0 < 0, p1 >= 0` selected all cells. Replaced the block with the controller ruling:

```cpp
            // keep only the cells in [p0, p1), when a range is given
            if (add_cell && (p0 >= 0 || p1 >= 0)) {
                add_cell = (p0 < 0 || cells.pos_get(i) >= p0) && (p1 < 0 || cells.pos_get(i) < p1);
            }
```

The behavior of the existing callers is unchanged: `state_write` passes `(-1, -1)` and skips the filter, `state_write_range` passes `p0 >= 0, p1 > p0`, which the new expression evaluates identically to `pos_in(i, p0, p1)`.

**Finding 2 - indexed hybrid memory (`src/llama-memory-hybrid-idx.h/.cpp`)**

Confirmed `llama_memory_hybrid_idx : public llama_memory_hybrid` (llama-memory-hybrid-idx.h:15), so it inherited the forwarder that writes only `mem_attn` and silently omitted the indexer section. Added an override next to the class's other state overrides (declared after `state_read` in the header, defined after `state_write` in the .cpp, matching the base-class file's layout) that throws, so the C API logs and returns 0 like every other unsupported memory type. `<stdexcept>` was already included in the .cpp and `GGML_UNUSED` comes from llama-impl.h, so the snippet compiled as written.

## Covering tests

Build:

```
<TEMP>\v100\build_test_t32.cmd test-t32-range
```

Result: links `llama.dll`, `llama-common.dll`, `test-t32-range.exe`; a follow-up invocation reports `ninja: no work to do.`

Command 1 (hybrid 2B):

```powershell
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```

Output:

```
0.01.687.513 I cmn          init: llama threadpool init, n_threads = 8
[t32] hybrid=1 full=20990228 partial=20202092 range=788144
[t32] full state size > 0                                                      PASS
[t32] range state size > 0                                                     PASS
[t32] hybrid: range(0,L) size == full - partial + 8                            PASS
[t32] range(0,L/2) size is between header and full range                       PASS
EXIT: 0
```

Command 2 (pure-attention 3B):

```powershell
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen2.5-Coder-3B-IQ4_XS.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```

Output:

```
0.01.473.649 I cmn          init: llama threadpool init, n_threads = 8
[t32] hybrid=0 full=2360960 partial=2360960 range=2360960
[t32] full state size > 0                                                      PASS
[t32] range state size > 0                                                     PASS
[t32] attn: range(0,L) size == full size                                       PASS
[t32] payload reads returned the expected sizes                                PASS
[t32] attn: range(0,L) payload == full payload                                 PASS
[t32] range(0,L/2) size is between header and full range                       PASS
EXIT: 0
```

Both runs are green and exit 0, as before the fix. The two test models exercise `p0 = 0, p1 = L` and `p0 = 0, p1 = L/2` only, so the sentinel branches (`p0 < 0` or `p1 < 0`) are not covered by these runs; they are dead paths in the current callers, and the fix keeps the documented contract true for future callers. The idx override is not reachable from these models either (neither carries an indexer); its coverage is compile-time plus the fact that every other unsupported memory type takes the same logged-error path.

## Commit

```
df4bacd7409fdd04e68be213e9ded9542b8ae334
df4bacd74 t32 : fix range write sentinel handling, reject indexed hybrid
```

Diffstat:

```
 src/llama-kv-cache.cpp          |  4 ++--
 src/llama-memory-hybrid-idx.cpp | 10 ++++++++++
 src/llama-memory-hybrid-idx.h   |  2 ++
 3 files changed, 14 insertions(+), 2 deletions(-)
```

Not pushed. `git status` is clean after the commit.

## Concerns after fix

- The `p0 < 0, p1 >= 0` branch now selects `pos < p1`, which is a superset of the seq's cells; `state_write_range` still rejects `p0 < 0` up front, so no current caller reaches it. The implementation now matches the header comment if a future caller relaxes that validation.
- `llama_memory_hybrid_idx::state_write_range` throws even when `mem_idx == nullptr` (model without indexer). That is the conservative choice: the class still carries the hybrid layout, and a loud failure is safer than an implicit assumption that no indexer means the base forwarder is exact.
