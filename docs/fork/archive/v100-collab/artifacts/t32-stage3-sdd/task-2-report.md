# Task 2 report: CLI params + tree instance creation and startup validation

Status: DONE

## What was implemented

All of the brief, verbatim:

1. `common/common.h` - 7 new `common_params` fields after `cache_ram_mib`:
   `kv_tree` (bool, false), `tree_chunk` (int32_t, 512), `tree_anchor_step` (int32_t, 32768),
   `tree_ram_mib` (int32_t, 8192), `tree_disk_mib` (int32_t, 65536),
   `tree_disk_dir` (std::string, ""), `tree_debug` (bool, false).
2. `common/arg.cpp` - 7 new options after the `--cache-idle-slots` block:
   `--kv-tree`/`--no-kv-tree` (`LLAMA_ARG_KV_TREE`), `--tree-chunk` (`LLAMA_ARG_TREE_CHUNK`,
   rejects `value <= 0`), `--tree-anchor-step` (`LLAMA_ARG_TREE_ANCHOR_STEP`, rejects `value < 0`),
   `--tree-ram` (`LLAMA_ARG_TREE_RAM`, rejects `value < 0`), `--tree-disk` (`LLAMA_ARG_TREE_DISK`),
   `--tree-disk-limit` (`LLAMA_ARG_TREE_DISK_LIMIT`, rejects `value < 0`),
   `--tree-debug`/`--no-tree-debug` (`LLAMA_ARG_TREE_DEBUG`). All SERVER examples.
3. `tools/server/server-context.cpp`:
   - added `#include "server-kv-tree.h"` in the include block;
   - added member `std::unique_ptr<kv_tree> tree;` right after `prompt_cache`;
   - in `load_model()`, right after the prompt-cache creation block: skip with warning when
     `kv_unified`, build `kv_tree_config` from the params, create the disk dir with
     `std::filesystem::create_directories` (clears `disk_dir` + warns on error), construct
     `tree = std::make_unique<kv_tree>(cfg)`, and log the startup line.

No other files touched. Default off: no behavior change when `--kv-tree` is absent.

## What was tested and results

Build: deleted `build\bin\Release\llama-server-impl.dll`, ran `<TEMP>\v100\build_server.cmd` -> exit 0 (step [64/64] linked `llama-server.exe`).

Help output (`--help | findstr "tree"`) - 7 lines containing `tree`:

```
--kv-tree, --no-kv-tree                 store stable KV prefixes in a RAM+SSD tree and reuse them across tasks
--tree-chunk N                          kv tree block size in tokens (default: 512)
--tree-anchor-step N                    minimum spacing between kv tree anchors in tokens (default: 32768)
--tree-ram N                            kv tree RAM tier limit in MiB (default: 8192)
--tree-disk PATH                        kv tree SSD tier directory (default: empty = no SSD tier)
--tree-disk-limit N                     kv tree SSD tier limit in MiB (default: 65536)
--tree-debug, --no-tree-debug           dump the kv tree state after each park/restore (default: disabled)
```

Rejection of invalid value (`--kv-tree --tree-chunk 0 -m ... -ngl 99`):

```
error while handling argument "--tree-chunk": tree-chunk must be positive
EXIT=1
```

Smoke with `--kv-tree --tree-ram 8 --tree-disk %TEMP%\v100\t32-tree-smoke --tree-disk-limit 64`
(2B model, `CUDA_VISIBLE_DEVICES=0`, port 8931, /health polled):
- `/health` returned `status = ok` after 2 s;
- `/completion` (prompt "Hello world", n_predict 8) returned content `! I am a new AI assistant.`;
- process killed, log line:

```
0.01.614.657 I srv    load_model: kv tree enabled: chunk = 512, anchor_step = 32768, ram = 8 MiB, disk = <TEMP>\v100\t32-tree-smoke (64 MiB)
```

- `t32-tree-smoke` directory was created by the server (parent-dir creation path exercised).

Default-off check (port 8932, no `--kv-tree`): health ok, completion served, and the stderr log
contains 0 matches for `kv tree`.

## Files changed

- `common/common.h` (+9)
- `common/arg.cpp` (+63)
- `tools/server/server-context.cpp` (+35)

Commit: `df4225828 common : add kv tree server options` (message plus `Assisted-by: opencode`).

## Self-review findings

- All 7 fields, all 7 options, member, include, creation, validation, disk-dir handling and log
  line are present and match the brief byte-for-byte; help/rejection/log outputs confirm it.
- `--tree-anchor-step 0` and `--tree-ram 0` / `--tree-disk-limit 0` are accepted (negative-only
  checks, per brief); `--tree-chunk` requires positive.
- `tree` is intentionally created but not yet used by any code path in this task (Task 3 wires it);
  no unused-variable warning since it is a class member.
- No metrics, idle-slot, prompt-cache or other unrelated changes.

## Concerns

- None blocking. The tree object is not exercised beyond construction in this task, so no runtime
  tree behavior is expected yet (by design).
