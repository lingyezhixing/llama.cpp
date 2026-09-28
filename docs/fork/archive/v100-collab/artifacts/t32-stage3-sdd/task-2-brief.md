### Task 2: CLI 参数 + 树实例创建与启动校验

**Files:**
- Modify: `common/common.h`（server params 块 :620-633）
- Modify: `common/arg.cpp`（`--cache-ram` 旁 :1712-1719）
- Modify: `tools/server/server-context.cpp`（`load_model()` 里 prompt_cache 创建处 :1351-1363，成员区 :915）

**Interfaces:**
- Consumes: Task 1 的 `kv_tree` 模块（构造函数 + 配置校验）。
- Produces: `common_params` 字段 `kv_tree` / `tree_chunk` / `tree_anchor_step` / `tree_ram_mib` / `tree_disk_mib` / `tree_disk_dir` / `tree_debug`；`server_context::tree`（`std::unique_ptr<kv_tree>`）。默认全关 = 行为不变。

- [ ] **Step 1: 加参数字段（common.h）**

`cache_ram_mib`（:633）之后：

```cpp
    // kv tree params (T32)
    bool        kv_tree         = false;    // use the kv tree for cross-session KV storage
    int32_t     tree_chunk      = 512;      // KV tree block size in tokens
    int32_t     tree_anchor_step = 32768;   // min spacing between kv tree anchors
    int32_t     tree_ram_mib    = 8192;     // kv tree RAM tier limit in MiB
    int32_t     tree_disk_mib   = 65536;    // kv tree SSD tier limit in MiB
    std::string tree_disk_dir   = "";       // kv tree SSD tier directory (empty = no SSD tier)
    bool        tree_debug      = false;    // dump the kv tree after each park/restore
```

- [ ] **Step 2: 加 CLI（arg.cpp）**

`--cache-idle-slots`（:1728-1735）之后：

```cpp
    add_opt(common_arg(
        {"--kv-tree"},
        {"--no-kv-tree"},
        "store stable KV prefixes in a RAM+SSD tree and reuse them across tasks (default: disabled)",
        [](common_params & params, bool value) {
            params.kv_tree = value;
        }
    ).set_env("LLAMA_ARG_KV_TREE").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--tree-chunk"}, "N",
        string_format("kv tree block size in tokens (default: %d)", params.tree_chunk),
        [](common_params & params, int value) {
            if (value <= 0) {
                throw std::invalid_argument("tree-chunk must be positive");
            }
            params.tree_chunk = value;
        }
    ).set_env("LLAMA_ARG_TREE_CHUNK").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--tree-anchor-step"}, "N",
        string_format("minimum spacing between kv tree anchors in tokens (default: %d)", params.tree_anchor_step),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("tree-anchor-step must be non-negative");
            }
            params.tree_anchor_step = value;
        }
    ).set_env("LLAMA_ARG_TREE_ANCHOR_STEP").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--tree-ram"}, "N",
        string_format("kv tree RAM tier limit in MiB (default: %d)", params.tree_ram_mib),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("tree-ram must be non-negative");
            }
            params.tree_ram_mib = value;
        }
    ).set_env("LLAMA_ARG_TREE_RAM").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--tree-disk"}, "PATH",
        "kv tree SSD tier directory (default: empty = no SSD tier)",
        [](common_params & params, const std::string & value) {
            params.tree_disk_dir = value;
        }
    ).set_env("LLAMA_ARG_TREE_DISK").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--tree-disk-limit"}, "N",
        string_format("kv tree SSD tier limit in MiB (default: %d)", params.tree_disk_mib),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("tree-disk-limit must be non-negative");
            }
            params.tree_disk_mib = value;
        }
    ).set_env("LLAMA_ARG_TREE_DISK_LIMIT").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--tree-debug"},
        {"--no-tree-debug"},
        "dump the kv tree state after each park/restore (default: disabled)",
        [](common_params & params, bool value) {
            params.tree_debug = value;
        }
    ).set_env("LLAMA_ARG_TREE_DEBUG").set_examples({LLAMA_EXAMPLE_SERVER}));
```

注意：上面 `--kv-tree` 已用一正一反两个 flag（与 `--cache-idle-slots` 同风格）。

- [ ] **Step 3: 创建树实例 + 校验（server-context.cpp）**

成员区（:915 `std::unique_ptr<server_prompt_cache> prompt_cache;` 之后）加：

```cpp
    std::unique_ptr<kv_tree> tree;
```

`load_model()` 里 `prompt_cache` 创建块（:1351-1363）之后加：

```cpp
        if (params_base.kv_tree) {
            if (params_base.kv_unified) {
                SRV_WRN("%s", "--kv-tree is not supported with --kv-unified, disabling kv tree\n");
                params_base.kv_tree = false;
            }
        }

        if (params_base.kv_tree) {
            kv_tree_config cfg;
            cfg.chunk       = params_base.tree_chunk;
            cfg.anchor_step = params_base.tree_anchor_step;
            cfg.ram_limit   = (size_t) params_base.tree_ram_mib << 20;
            cfg.disk_limit  = (size_t) params_base.tree_disk_mib << 20;
            cfg.disk_dir    = params_base.tree_disk_dir;
            cfg.debug       = params_base.tree_debug;

            if (!cfg.disk_dir.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(cfg.disk_dir, ec);
                if (ec) {
                    SRV_WRN("failed to create kv tree disk directory '%s': %s, disabling the SSD tier\n",
                            cfg.disk_dir.c_str(), ec.message().c_str());
                    cfg.disk_dir.clear();
                }
            }

            tree = std::make_unique<kv_tree>(cfg);

            SRV_INF("kv tree enabled: chunk = %d, anchor_step = %d, ram = %d MiB, disk = %s (%d MiB)\n",
                    cfg.chunk, cfg.anchor_step, params_base.tree_ram_mib,
                    cfg.disk_dir.empty() ? "off" : cfg.disk_dir.c_str(), params_base.tree_disk_mib);
        }
```

说明：`--cache-idle-slots` 与 `--cache-ram 0` 的兼容在 Task 4 处理（改 :1420 的门槛检查让 `--kv-tree` 生效），本任务不动 `cache_ram_mib` 语义。

- [ ] **Step 4: 构建 + 启动校验**

```powershell
$repo='D:\LLM\Backend\src\llama.cpp-my'
Remove-Item -Force "$repo\build\bin\Release\llama-server-impl.dll" -ErrorAction SilentlyContinue
& '<TEMP>\v100\build_server.cmd'
```

Run（帮助 + 非法值）：
```powershell
& "$repo\build\bin\Release\llama-server.exe" --help | findstr "tree"
& "$repo\build\bin\Release\llama-server.exe" --kv-tree --tree-chunk 0 -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99
```
Expected: 第一条列出 7 个 `--tree-*` 选项；第二条以非零码退出并打印 `tree-chunk must be positive`（common_params_parse 抛错）。

Run（默认关 = 行为不变 + 树开启可启动）：
```powershell
$tmp='<TEMP>\v100'
$env:CUDA_VISIBLE_DEVICES='0'
$p = Start-Process -FilePath "$repo\build\bin\Release\llama-server.exe" -ArgumentList @('-m','<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf','-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','8192','--port','8931','--kv-tree','--tree-ram','8','--tree-disk',"$tmp\t32-tree-smoke",'--tree-disk-limit','64') -NoNewWindow -PassThru -RedirectStandardOutput "$tmp\s3-out.txt" -RedirectStandardError "$tmp\s3-err.txt"
Start-Sleep -Seconds 20
Invoke-RestMethod -Uri 'http://127.0.0.1:8931/health' -Method Get
Invoke-RestMethod -Uri 'http://127.0.0.1:8931/completion' -Method Post -ContentType 'application/json' -Body '{"prompt":"Hello world","n_predict":8,"temperature":0}'
Stop-Process -Id $p.Id -Force
Select-String -Path "$tmp\s3-err.txt" -Pattern 'kv tree enabled|kv-tree' | Select-Object -First 5
```
Expected: `/health` 200；completion 返回内容；日志出现 `kv tree enabled: chunk = 512 ...`。

- [ ] **Step 5: 提交**

```powershell
git -C $repo add common/common.h common/arg.cpp tools/server/server-context.cpp
git -C $repo commit -m "common : add kv tree server options" -m "Assisted-by: opencode"
```

---

