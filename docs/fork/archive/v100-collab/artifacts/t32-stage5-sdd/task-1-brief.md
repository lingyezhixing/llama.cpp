### Task 1: 重命名 + 新旋钮 (D25/D29, 无行为变化)

**Files:**
- Modify: `common/common.h:638`, `common/arg.cpp:1755-1763`, `tools/server/server-context.cpp:1497-1518`, `tools/server/server-kv-tree.h:14-21`
- Modify (artifact): `artifacts\t32-stage3-ab.ps1` (1 处)

**Interfaces:**
- Produces: `params.tree_checkpoint_anchor_step` (默认 32768), `params.tree_checkpoint_fork_step` (默认 8192); `kv_tree_config.fork_step` (默认 8192); CLI/env 新名 (Task 2 用 `cfg.fork_step`, Task 4 脚本用新名).

- [ ] **Step 1: 改 common.h**

```cpp
    int32_t     tree_chunk      = 512;      // KV tree block size in tokens
    int32_t     tree_checkpoint_anchor_step = 32768;   // min spacing between kv tree checkpoint anchors
    int32_t     tree_checkpoint_fork_step   = 8192;    // min spacing between kv tree fork anchors
```

- [ ] **Step 2: 改 arg.cpp (两个选项)**

原 `--tree-anchor-step` 块替换为:

```cpp
        {
            {"--tree-checkpoint-anchor-step"}, "N",
            string_format("minimum spacing between kv tree checkpoint anchors in tokens (default: %d)", params.tree_checkpoint_anchor_step),
            [](common_params & params, int value) {
                if (value < 0) {
                    throw std::invalid_argument("tree-checkpoint-anchor-step must be non-negative");
                }
                params.tree_checkpoint_anchor_step = value;
            }
        ).set_env("LLAMA_ARG_TREE_CHECKPOINT_ANCHOR_STEP").set_examples({LLAMA_EXAMPLE_SERVER}),
        {
            {"--tree-checkpoint-fork-step"}, "N",
            string_format("minimum spacing between kv tree fork anchors in tokens (default: %d)", params.tree_checkpoint_fork_step),
            [](common_params & params, int value) {
                if (value < 0) {
                    throw std::invalid_argument("tree-checkpoint-fork-step must be non-negative");
                }
                params.tree_checkpoint_fork_step = value;
            }
        ).set_env("LLAMA_ARG_TREE_CHECKPOINT_FORK_STEP").set_examples({LLAMA_EXAMPLE_SERVER}),
```

- [ ] **Step 3: 改 server-context.cpp 映射与启动日志**

```cpp
            cfg.chunk       = params_base.tree_chunk;
            cfg.anchor_step = params_base.tree_checkpoint_anchor_step;
            cfg.fork_step   = params_base.tree_checkpoint_fork_step;
```
启动日志加 fork_step:

```cpp
            SRV_INF("kv tree enabled: chunk = %d, anchor_step = %d, fork_step = %d, ram = %d MiB, disk = %s (%d MiB)\n",
                    cfg.chunk, cfg.anchor_step, cfg.fork_step, params_base.tree_ram_mib,
                    cfg.disk_dir.empty() ? "off" : cfg.disk_dir.c_str(), params_base.tree_disk_mib);
```

- [ ] **Step 4: server-kv-tree.h 加 cfg 字段**

```cpp
struct kv_tree_config {
    int         chunk       = 512;
    int         anchor_step = 32768;
    int         fork_step   = 8192;
    ...
```

- [ ] **Step 5: 脚本改名**

`artifacts\t32-stage3-ab.ps1` 的 Start-Srv 里 `'--tree-anchor-step'` -> `'--tree-checkpoint-anchor-step'` (只此一处).

- [ ] **Step 6: 验证**

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$repo='D:\LLM\Backend\src\llama.cpp-my'
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
Remove-Item -Force "$repo\build\bin\Release\llama-server-impl.dll" -ErrorAction SilentlyContinue
& '<TEMP>\v100\build_server.cmd'
& "$repo\build\bin\Release\llama-server.exe" --help 2>&1 | Select-String 'tree-checkpoint'
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode heal | Select-String 'RESULT'
```

Expected: help 显示 `--tree-checkpoint-anchor-step` 与 `--tree-checkpoint-fork-step`, 不再有 `--tree-anchor-step`; heal `RESULT heal: 0 failure(s)`.

- [ ] **Step 7: 提交**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add common/common.h common/arg.cpp tools/server/server-context.cpp tools/server/server-kv-tree.h
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "common : rename kv tree checkpoint options and add the fork step" -m "Assisted-by: opencode"
```

---

