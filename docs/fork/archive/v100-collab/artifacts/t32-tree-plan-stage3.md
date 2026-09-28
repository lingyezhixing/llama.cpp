# T32 阶段 3: server 集成 (`--kv-tree`) + A/B 验收 - 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 T32 树状 KV 存储接入 llama-server（`--kv-tree` 开关，默认关），并做小模型 A/B 验收（含强制 SSD 溢出）。

**Architecture:** 复用阶段 2 的 `kv_tree` 模块（`tools/server/server-kv-tree.{h,cpp}`），在 `server_context` 增加一个实例；在 `get_available_slot()` 的"保存旧 slot / 装载新任务"两处、idle slot 保存处、`SLOT_ERASE` 处挂接。`--kv-tree` 关闭时 stock 路径逐行不变（A/B 对照）。

**Tech Stack:** C++17, llama.cpp server (tools/server), CMake/Ninja (MSVC), CUDA (Tesla V100, sm70), PowerShell 验收脚本。

**Spec:** `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md`（阶段 3 = §3.4 + §7；本计划的决定 D1-D8 是对 spec 的实施细化/勘误，见下）与阶段 2 计划 `t32-tree-plan-stage2.md`（模块接口与风格）。

## Global Constraints

- 只允许改这些文件：`tools/server/server-kv-tree.h`、`tools/server/server-kv-tree.cpp`、`tests/test-t32-tree.cpp`、`tools/server/server-context.cpp`、`common/common.h`、`common/arg.cpp`。不改 `src/`，不加新文件（`tests/*` 不新增）。
- 注释：ASCII only、简洁、`//`；禁止 emdash/unicode 箭头等非 ASCII 字符（现有中文注释不在本计划新增范围内，新写的一律英文 ASCII）。
- 构建（共享 build 树，Ninja Multi-Config，MSVC）：
  - 模块/测试：`<TEMP>\v100\build_test_t32.cmd test-t32-tree`（内部为 `cmake --build build --config Release --target test-t32-tree`）。
  - server：先删 `D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\llama-server-impl.dll`，再 `<TEMP>\v100\build_server.cmd`（改 `src/`/`tools/` 后必须这样做）。
- 测试模型 + 设备（用户约束）：验证只用小模型，`CUDA_VISIBLE_DEVICES=0`：
  - 2B hybrid: `<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf`
  - 3B 纯 attention: `<models>\Qwen2.5-Coder-3B-IQ4_XS.gguf`
- 不跑 27B；不 push；不部署；生产目录 `D:\LLM\Backend\llama.cpp-my` 不动。
- 提交：分支 `t32-stage3`（从 master `4f2631088` 起）；消息前缀沿用仓库风格（`server : ...` / `common : ...` / `tests : ...`），附 `Assisted-by: opencode`。
- 后台 server 必须 `-RedirectStandardError`（否则 PowerShell 包装会卡/丢日志）；服务器日志重定向到文件后再 grep。
- PowerShell 5.1：无 heredoc；写文件用 `[System.IO.File]::WriteAllText` + `New-Object System.Text.UTF8Encoding($false)`；频道文件追加用临时文件 + UTF8 no-BOM 拼接。
- 本计划中所有"行号"是写计划时的定位（base `4f2631088`），实施时以内容匹配为准。

## 文件结构

| 文件 | 职责 | 本阶段改动 |
|---|---|---|
| `tools/server/server-kv-tree.h` | 树模块公共接口 | 加 `drop_seq()` 声明、`set_block_payload()` 私有声明 |
| `tools/server/server-kv-tree.cpp` | 树模块实现 | 加 `drop_seq`、流式恢复（`set_block_payload`）、`bytes_load` 计数修正、ctor 校验、`cfg.debug` dump |
| `tests/test-t32-tree.cpp` | 模块 + 模型 harness | 加 `run_logic_drop()`、`run_logic_stream()`；接入 `run_logic()` |
| `tools/server/server-context.cpp` | server 集成 | `server_slot::{prompt_park,prompt_restore_tree}` + slot 成员；`get_available_slot` 分支；heal 捕获；idle park；`SLOT_ERASE` drop；树实例创建与启动校验 |
| `common/common.h` | 参数结构 | 加 `kv_tree` 及 `--tree-*` 字段（server params 块，`cache_ram_mib` 附近 :620-633) |
| `common/arg.cpp` | CLI | 加 `--kv-tree`/`--tree-ram`/`--tree-disk`/`--tree-disk-limit`/`--tree-chunk`/`--tree-anchor-step`/`--tree-debug`（`--cache-ram` 旁 :1712-1719 风格） |
| `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` | A/B 验收脚本（频道产物，不入 git） | 新建（Task 5） |

## 设计决定（对 spec 的细化/勘误；实施中如发现新问题按 SDD 裁决并回写本文件）

- **D1（验收场景勘误，重要）:** spec §7.1 的 "2x100K 共享 90K" 场景在 np=1 下 **不会触发树**：`get_available_slot` 只在 `update_cache=true`（LCP 保留率 `f_keep < 0.5` 或 LRU 选中）时才 park/restore（server-context.cpp:1606/1632/1642）；高共享（f_keep ≥ 0.5）由 VRAM 内 LCP 复用直接处理。因此阶段 3 A 场景改为**低重叠长会话**（2x16K 仅共享系统前缀 ~512 token，f_keep ~3%），每次切换必 park + restore；另加一个高共享场景（共享 12K）验证"不改动 stock 复用 + 输出一致"。此勘误需在验收报告中显式说明。
- **D2（MTP/spec 载荷缺口）:** `kv_tree_anchor_in` 无 spec 字段，模块无 spec io；2B/3B 无 MTP，阶段 3 不涉及。恢复后不还原 spec 状态（`common_speculative_set_state` 跳过）；27B/MTP 部署阶段补（记录为已知缺口）。
- **D3（lora 保护）:** 树 park/restore 在 `slot.lora` 非空（有适配器）时跳过 + TRC（KV 与适配器状态耦合，v1 保守）。
- **D4（idle park 门槛）:** `--kv-tree` 打开时 `--cache-idle-slots` 不再要求 `--cache-ram`（server-context.cpp:1420-1423 的检查加 `&& !kv_tree`）。
- **D5（RAM 峰值界限）:** 恢复时**磁盘块**按 RAM 余额决定是否驻留（不足则流式装载到临时缓冲，不进 RAM）；**锚点**仍走驻留加载（单次 O(1 个锚点) 超界，随后 `settle()` 按预算回落；2B ~20MB / 27B ~170MB 上界，记录在案）。
- **D6（heal 精确位置）:** heal 捕获必须发生在**恰好** `m.deep` 位置（该位置被已存块覆盖，`capture_anchor` 才能挂靠；更深的批边界位置可能没有已存块）。做法：prefill 批次在 heal 位置截断（一次 1 token 粒度检查），解码完成后的下一次迭代开头捕获。
- **D7（drop_seq 语义）:** `drop_seq(tokens)` 只在"请求 token 覆盖到某个已存序列的 tip 块"时删除该序列；部分覆盖（尾部块内截断）也删；其余返回 false（no-op）。用于 `SLOT_ERASE`。
- **D8（锚点预算校验）:** 验收用 `--tree-anchor-step 4096`（16K 会话）+ 预算校准（见 Task 5）；不触发 spec 的 32K 默认值。
- **D10（cache_prompt=false 跳过 restore）:** 请求不复用 prompt 时 restore 的结果会被 `n_past=0` 丢弃，且其 heal 会在重算批次里截断批次（任务 5 实测：对照臂因 heal 截断产生近并列翻转 + 14 次失败捕获）。树分支改为：`park` 照常（保存旧内容），`!task.params.cache_prompt` 时不 restore，直接 `prompt_clear()`（与 stock 最终效果等价）。
- **D11（纯 attention 无 fork 复用，结构性限制）:** 纯 attention 模型（3B 对照）不产生中段检查点（create_checkpoint 门控在 FULL/RS/SWA），锚点只有序列末端 -> 分叉请求 `deep < tip` 时无锚点可用 -> 树恢复必然 miss。对照运行只验证"树开不破坏正确性"，不做复用断言；"无 recurrent 模型允许无锚点恢复（C=最深块边界）"记入阶段 4 候选。
- **D9（park 守卫勘误，实施中发现）:**" spec §3.3 的"位置不从 0 -> 跳过 park"不能用 `llama_memory_seq_pos_min` 判定：hybrid 模型的该 API 返回 recurrent 尾部（实测 454-token 序列返回 453），永远不为 0。实际不变量：hybrid 的 `ctx_shift` 在启动时被自动禁用（server-context.cpp:1257-1259）；纯 attention 的 shift 用 `seq_add` 保持 0 基并把 token 列表同步；任何不支持的前端移除都会走 checkpoint 回退的全量重算（:3402-3407）或 `prompt_clear`。因此 park 守卫只保留 `seq_pos_max == tokens.size()-1`（长度一致性），"位置不从 0"的反例在当前 server 版本不可构造（记录，不测）。
- **D12（spec §3.2 检查点表重建延后，裁决）:** restore 后按路径锚点重建 slot 最小检查点表阶段 3 未实现（`prompt.checkpoints` 仍清空，`kv_tree_restore::anchors` 保留为预留接口）；正确性由"从 C 重算剩余 prompt"的既有回退保证，重建只是 reasoning 回退/SWA 裁剪的快速路径 -> 阶段 4 实现。

---

### Task 1: 树模块增量（drop_seq / 流式恢复 + bytes_load / debug / ctor 校验）

**Files:**
- Modify: `tools/server/server-kv-tree.h`
- Modify: `tools/server/server-kv-tree.cpp`
- Test: `tests/test-t32-tree.cpp`

**Interfaces:**
- Consumes: 无（纯模块内改动）
- Produces:
  - `bool kv_tree::drop_seq(const std::vector<llama_token> & tokens)`（public）
  - `bool kv_tree::set_block_payload(kv_tree_io & io, kv_tree_block & b, bool append, std::vector<uint8_t> & scratch)`（private）
  - 不变：`park/restore/capture_anchor/stats/dump` 签名。

- [ ] **Step 1: 写失败测试（drop + 流式恢复）**

在 `tests/test-t32-tree.cpp` 的 `run_logic_evict()` 之后（:139 之后）插入两个函数：

```cpp
static void run_logic_drop() {
    fprintf(stderr, "[t32-tree] logic: drop_seq\n");

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 1 << 20;
    cfg.disk_dir    = "";

    kv_tree tree(cfg);

    const auto tok   = make_tokens(1536, 0);
    auto       tok_x = tok;
    for (int i = 0; i < 512; ++i) {
        tok_x.push_back((llama_token) (10 + (i * 7 + 99) % 1000));
    }
    auto tok_mid   = std::vector<llama_token>(tok.begin(), tok.begin() + 1400);
    auto tok_other = make_tokens(1536, 1);

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok, {}), "drop: park A");
    }

    check(!tree.drop_seq(tok_other),                    "drop: unrelated tokens are a no-op");
    check(tree.drop_seq(tok_mid),                       "drop: mid-tail tokens drop the sequence");
    check_eq(tree.stats().evicted_seqs, 1,              "drop: sequence was dropped");
    check_eq(tree.stats().blocks_ram, 0,                "drop: its blocks were released");
    check(!tree.drop_seq(tok),                          "drop: second drop is a no-op");

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok, {}), "drop: re-park A");
    }
    check(tree.drop_seq(tok_x),                         "drop: superstring drops the sequence");
    check_eq(tree.stats().evicted_seqs, 2,              "drop: sequence was dropped again");
}

static void run_logic_stream() {
    fprintf(stderr, "[t32-tree] logic: streaming restore\n");

    const auto dir = std::filesystem::temp_directory_path() / "t32-tree-logic-stream";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 4096;      // fits one fake block (4096 B), nothing more
    cfg.disk_limit  = 1 << 20;
    cfg.disk_dir    = dir.string();

    kv_tree tree(cfg);

    const auto tok = make_tokens(1536, 0);

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok, {}),                          "stream: park with spill");
        check_eq(tree.stats().blocks_disk, 3,                           "stream: all blocks spilled to disk");
        check((size_t) tree.stats().bytes_ram <= cfg.ram_limit,         "stream: park respects the ram budget");
    }

    {
        kv_tree_io_fake io;
        const auto r = tree.restore(io, nullptr, tok);
        check_eq(r.C, 1536,                                             "stream: restore hits the tip");
        check_eq(r.heal, -1,                                            "stream: no heal needed for an exact tip");
        check_eq(io.set_range_bytes, 3 * 4096,                          "stream: all blocks were written to the io");
        check_eq(io.set_partial_calls, 1,                               "stream: the anchor was written");
        check((size_t) tree.stats().bytes_ram <= cfg.ram_limit,         "stream: restore respects the ram budget");
        check_eq(tree.stats().bytes_load, 3 * 4096,                "stream: bytes_load counts payloads moved from disk");
        check_eq(tree.stats().blocks_disk, 3,                           "stream: streamed blocks stay on disk");
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}
```

说明：fake io 的载荷 = `(p1-p0)*8` 字节（:46-49），512 token 块 = 4096 B；`get_partial` = 64 B（:57-60）。预算 4096 B 下 park 时 3 个块必然全部降级到磁盘（降级顺序与哈希表遍历无关：两次降级后仍超预算），锚点 64 B 留 RAM。两个函数均用文件里已有的全局 `n_fail` 计数（:14），不需要额外封装。

- [ ] **Step 2: 接入 `run_logic()` 并跑失败**

`run_logic()`（:141-193）中 `run_logic_evict();`（:144）之后加：

```cpp
    run_logic_drop();
    run_logic_stream();
```

Run: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree`
Expected: 编译失败（`drop_seq` 不存在）或链接失败 - 这是预期的失败。

- [ ] **Step 3: 实现 drop_seq + 流式恢复 + 计数**

`tools/server/server-kv-tree.h`：public 区 `capture_anchor` 之后（:162 后）加：

```cpp
    // release the stored sequence that matches a prefix of tokens; no-op when nothing matches
    bool drop_seq(const std::vector<llama_token> & tokens);
```

private 区 `load_payload` 声明旁（:199-200）加：

```cpp
    bool set_block_payload(kv_tree_io & io, kv_tree_block & b, bool append, std::vector<uint8_t> & scratch);
```

`tools/server/server-kv-tree.cpp`：
1. 在 `remove_seq`（:665-693）之后加：

```cpp
bool kv_tree::drop_seq(const std::vector<llama_token> & tokens) {
    if (tokens.empty()) {
        return false;
    }

    const kv_tree_match m = match(tokens);

    uint64_t tip = 0;
    if (m.n_part > 0) {
        tip = m.part_hash;
    } else if (m.n_full > 0) {
        tip = m.path.back();
    } else {
        return false;
    }

    const auto it = seqs.find(tip);
    if (it == seqs.end()) {
        return false;
    }

    remove_seq(it);
    return true;
}
```

2. `load_payload` 计数（移自 restore 循环，避免重复计数；磁盘 -> RAM 的搬运才计数）：
   - `load_payload(kv_tree_block &)`（:550-576）的 `b.transient = true;` 之前加：
   ```cpp
   st.bytes_load += (int64_t) b.data.size();
   ```
   - `load_payload(kv_tree_anchor &)`（:578-606）的 `a.transient = true;` 之前加：
   ```cpp
   st.bytes_load += (int64_t) anchor_bytes(a);
   ```

3. 在 `load_payload(kv_tree_block &)` 之后加新 helper：

```cpp
// feed one block payload to the io; keep it resident only when it fits the ram budget,
// otherwise stream it from disk through a scratch buffer
bool kv_tree::set_block_payload(kv_tree_io & io, kv_tree_block & b, bool append, std::vector<uint8_t> & scratch) {
    if (b.on_disk && (size_t) st.bytes_ram + b.bytes > cfg.ram_limit) {
        std::vector<uint8_t> buf;
        if (!read_disk(b.path, buf)) {
            fprintf(stderr, "[kv-tree] failed to read block %016" PRIx64 " from disk\n", b.hash);
            st.disk_errors++;
            return false;
        }

        kv_env e;
        if (!envelope_parse(buf, e) || e.magic != T32_ENV_MAGIC_BLOCK || e.hash != b.hash || e.p0 != b.pos0 || e.p1 != b.pos1 || e.n_dft != 0 || e.n_spec != 0) {
            fprintf(stderr, "[kv-tree] block %016" PRIx64 " payload check failed, dropping it\n", b.hash);
            st.disk_errors++;
            return false;
        }

        scratch.assign(e.tgt, e.tgt + e.n_tgt);
        st.bytes_load += (int64_t) scratch.size();
        return io.set_range(scratch.data(), scratch.size(), append);
    }

    if (!load_payload(b)) {
        return false;
    }

    return io.set_range(b.data.data(), b.data.size(), append);
}
```

4. `restore()`（:1226-1245）替换两个块装载段：

```cpp
    std::vector<uint8_t> scratch;

    // blocks covering [0, C): the last one may overshoot and is trimmed below
    for (const uint64_t hash : m.path) {
        kv_tree_block & b = blocks[hash];
        if (b.pos0 >= C) {
            break;
        }
        if (!set_block_payload(io_tgt, b, b.pos0 != 0, scratch)) {
            fprintf(stderr, "[kv-tree] restore failed: cannot load block [%d, %d)\n", b.pos0, b.pos1);
            ok = false;
            break;
        }
    }

    if (ok && m.n_part > 0 && C > blocks[m.part_hash].pos0) {
        kv_tree_block & b = blocks[m.part_hash];
        if (!set_block_payload(io_tgt, b, true, scratch)) {
            fprintf(stderr, "[kv-tree] restore failed: cannot load the partial block [%d, %d)\n", b.pos0, b.pos1);
            ok = false;
        }
    }
```

（删除原 :1236 的 `st.bytes_load += (int64_t) b.data.size();`。）

5. `park()`（:931）与 `restore()`（:1167）函数体末尾（`return true;`/成功 `return res;` 之前）加：

```cpp
    if (cfg.debug) {
        dump();
    }
```

（park 里放在最后一个 `return true;` 之前；restore 里放在 `st.restore_hits++...` 之后、`return res;` 之前。）

6. 构造函数（文件内 `kv_tree::kv_tree(const kv_tree_config & cfg)`，实施时定位）加校验：

```cpp
    if (cfg.chunk <= 0 || cfg.anchor_step < 0) {
        fprintf(stderr, "[kv-tree] invalid config: chunk = %d, anchor_step = %d\n", cfg.chunk, cfg.anchor_step);
        GGML_ABORT("invalid kv tree config");
    }
```

7. `write_disk()`（:435-486）在打开文件前创建父目录（审查批准的 brief 外改动；测试与 server 流程只保证顶层目录存在）：

```cpp
    std::error_code ec_dir;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec_dir);
    if (ec_dir) {
        return false;
    }
```

- [ ] **Step 4: 跑测试通过**

Run: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree` 然后 `& "$repo\build\bin\Release\test-t32-tree.exe" --mode logic`
Expected: 全部 PASS（新 19 项，总 37 项），exit 0。注意 `run_logic_stream` 用磁盘临时目录（`%TEMP%\t32-tree-logic-stream`），结束后应已删除。

再跑模型模式回归（确认流式改动未破坏真机路径）：
Run: `CUDA_VISIBLE_DEVICES=0`（PowerShell: `$env:CUDA_VISIBLE_DEVICES='0'`）`test-t32-tree.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096`
Expected: 43/43 PASS（与阶段 2 最终 head 相同）。

- [ ] **Step 5: 提交**

```powershell
git -C $repo add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git -C $repo commit -m "server : stream disk payloads during kv tree restore" -m "Assisted-by: opencode"
```

---

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

### Task 3: park/restore 接入 `get_available_slot`

**Files:**
- Modify: `tools/server/server-context.cpp`（includes、server_slot 成员与方法 :239-340、成员区 :915、`get_available_slot` :1636-1657）
- Test: server 冒烟（本任务内联命令）

**Interfaces:**
- Consumes: Task 1 的 `drop_seq` 不需要；Task 2 的 `server_context::tree`。
- Produces:
  - `server_slot::tree_heal`（`llama_pos`，默认 -1）
  - `bool server_slot::prompt_park(kv_tree & tree) const`
  - `bool server_slot::prompt_restore_tree(kv_tree & tree, const server_tokens & tokens)`
- 日志契约定死（验收脚本 grep 依赖）：
  - `kv tree: parked N tokens, K checkpoint candidates, ram = X B, disk = Y B`
  - `kv tree: restore miss for N tokens, full prefill`
  - `kv tree: restored N tokens (heal = H)`
  - `kv tree: park skipped (seq end P, tokens N)`

- [ ] **Step 1: include + slot 成员 + 两个方法**

`tools/server/server-context.cpp` 顶部 include 区（实施时定位 `#include "server-task.h"` 附近）加：

```cpp
#include "server-kv-tree.h"
```

`server_slot` 中 `server_prompt prompt;`（:297）之后加（方法与成员）：

```cpp
    // T32: heal position produced by the last kv tree restore (-1 = none)
    llama_pos tree_heal = -1;

    bool prompt_park(kv_tree & tree) const {
        if (!lora.empty() || prompt.tokens.empty() || prompt.tokens.has_mtmd) {
            return false;
        }

        // note: llama_memory_seq_pos_min is not usable as a guard - for hybrid memory it
        // reports the recurrent tail, not the first cached position; the server clears and
        // reprocesses whenever a front removal is not supported (see D9)
        const llama_pos p_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), id);
        if (p_max != (llama_pos) prompt.tokens.size() - 1) {
            SLT_WRN(*this, "kv tree: park skipped (seq end %d, tokens %zu)\n", p_max, prompt.tokens.size());
            return false;
        }

        kv_tree_io_llama io_tgt(ctx_tgt, id);
        std::unique_ptr<kv_tree_io_llama> io_dft;
        kv_tree_io * io_dft_ptr = nullptr;
        if (ctx_dft != nullptr) {
            io_dft = std::make_unique<kv_tree_io_llama>(ctx_dft, id);
            io_dft_ptr = io_dft.get();
        }

        std::vector<kv_tree_anchor_in> cks;
        for (const auto & c : prompt.checkpoints) {
            if (c.data_tgt.empty()) {
                continue;
            }
            kv_tree_anchor_in in;
            in.pos      = (llama_pos) c.n_tokens;
            in.data_tgt = c.data_tgt;
            in.data_dft = c.data_dft;
            cks.push_back(std::move(in));
        }

        const bool ok = tree.park(io_tgt, io_dft_ptr, prompt.tokens.get_tokens(), cks);
        SLT_INF(*this, "kv tree: parked %d tokens, %zu checkpoint candidates, ram = %lld B, disk = %lld B\n",
                (int) prompt.tokens.size(), cks.size(),
                (long long) tree.stats().bytes_ram, (long long) tree.stats().bytes_disk);
        return ok;
    }

    bool prompt_restore_tree(kv_tree & tree, const server_tokens & tokens) {
        if (!lora.empty() || tokens.empty() || tokens.has_mtmd) {
            return false;
        }

        kv_tree_io_llama io_tgt(ctx_tgt, id);
        std::unique_ptr<kv_tree_io_llama> io_dft;
        kv_tree_io * io_dft_ptr = nullptr;
        if (ctx_dft != nullptr) {
            io_dft = std::make_unique<kv_tree_io_llama>(ctx_dft, id);
            io_dft_ptr = io_dft.get();
        }

        const kv_tree_restore res = tree.restore(io_tgt, io_dft_ptr, tokens.get_tokens());
        if (res.C <= 0) {
            SLT_INF(*this, "kv tree: restore miss for %zu tokens, full prefill\n", tokens.size());
            return false;
        }

        prompt.tokens = server_tokens(llama_tokens(tokens.get_tokens().begin(), tokens.get_tokens().begin() + res.C), false);
        prompt.checkpoints.clear();
        tree_heal = res.heal > res.C ? res.heal : -1;

        SLT_INF(*this, "kv tree: restored %d tokens (heal = %d)\n", (int) res.C, (int) tree_heal);
        return true;
    }
```

注意：`server_tokens` **不是** `std::vector<llama_token>` 子类（私有成员、禁拷贝，见 server-common.h:138-171），所有需要 token 向量的地方必须用 `prompt.tokens.get_tokens()`（返回 `const llama_tokens &`）；赋值整体替换用 `server_tokens(llama_tokens(...), false)` 的移动赋值。

- [ ] **Step 2: `get_available_slot` 分支**

:1636-1657 整段替换为：

```cpp
        if (ret) {
            ret->tree_heal = -1;

            update_cache = update_cache && (prompt_cache || tree);

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                if (tree) {
                    ret->prompt_park(*tree);

                    // a restore is pointless when the request will not reuse the prompt (D10)
                    if (!task.params.cache_prompt || !ret->prompt_restore_tree(*tree, task.tokens)) {
                        ret->prompt_clear();
                    }
                } else {
                    ret->prompt_save(*prompt_cache);

                    if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                        ret->prompt_clear();
                    }

                    prompt_cache->update();
                }

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }
```

- [ ] **Step 3: 构建 + 冒烟（低重叠双会话，预算逼溢出）**

```powershell
$repo='D:\LLM\Backend\src\llama.cpp-my'; $tmp='<TEMP>\v100'
Remove-Item -Force "$repo\build\bin\Release\llama-server-impl.dll" -ErrorAction SilentlyContinue
& '<TEMP>\v100\build_server.cmd'
```

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
$srv = "$repo\build\bin\Release\llama-server.exe"
$dir = "$tmp\t32-tree-smoke"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
$a = 'alpha ' * 400; $b = 'beta ' * 400
$p = Start-Process -FilePath $srv -ArgumentList @('-m','<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf','-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','8192','--port','8931','--cache-ram','0','--no-cache-idle-slots','--kv-tree','--tree-chunk','512','--tree-anchor-step','512','--tree-ram','64','--tree-disk',$dir,'--tree-disk-limit','64') -NoNewWindow -PassThru -RedirectStandardOutput "$tmp\s3smoke-out.txt" -RedirectStandardError "$tmp\s3smoke-err.txt"
Start-Sleep -Seconds 25
function Req($prompt) { (Invoke-RestMethod -Uri 'http://127.0.0.1:8931/completion' -Method Post -ContentType 'application/json' -Body (@{prompt=$prompt; n_predict=48; temperature=0; top_k=1; seed=42; cache_prompt=$true} | ConvertTo-Json) ) }
$r1 = Req "system: you are helpful. $a"; $r2 = Req "system: you are helpful. $b"
$r3 = Req "system: you are helpful. $a"; $r4 = Req "system: you are helpful. $b"
$p | Stop-Process -Force
Write-Output "A2 output: $($r3.content)"; Write-Output "prompt_n A1 = $($r1.timings.prompt_n), A2 = $($r3.timings.prompt_n)"
Select-String -Path "$tmp\s3smoke-err.txt" -Pattern 'kv tree: (parked|restored|restore miss)' | ForEach-Object { $_.Line }
```

Expected:
- 4 个请求都成功；`r3.timings.prompt_n` 明显小于 `r1.timings.prompt_n`（A 第二次命中树）；
- 日志出现 `kv tree: parked`、`kv tree: restored`，且 parked 行里 `disk =` > 0（预算 64 MiB：两段内容约 65 MB + 2 个 tip 锚点各 ~20 MB，B 存档时必然把叶块降级到 SSD）；
- `restore miss` 仅出现在 A/B 的首次（各一次）。

基线对照（同一脚本换 `--kv-tree` 关闭 + `cache_prompt=false` 跑 r1/r3）输出内容必须与上面 r3 完全一致（逐 token 断言放 Task 5 脚本，这里人工比对 `content` 即可）。

- [ ] **Step 4: 提交**

```powershell
git -C $repo add tools/server/server-context.cpp
git -C $repo commit -m "server : reuse kv tree across slot switches" -m "Assisted-by: opencode"
```

---

### Task 4: heal 捕获 + idle park + SLOT_ERASE drop

**Files:**
- Modify: `tools/server/server-context.cpp`（STARTED 顶部 :3422 后、批填充循环 :3547、idle 分支 :2446-2462、`cache_idle_slots` 门槛 :1420-1423、`SLOT_ERASE` :2685-2688）
- Test: server 冒烟（heal 场景 + erase + idle park）

**Interfaces:**
- Consumes: Task 3 的 `tree_heal` / `prompt_park`。
- Produces: 新日志契约：
  - `kv tree: captured heal anchor at N`
  - `kv tree: failed to capture heal anchor at N`
  - `kv tree: heal position N missed (now M)`
  - `kv tree: dropped the stored sequence`

- [ ] **Step 1: heal 捕获（STARTED 顶部，checkpoint 裁剪块之后）**

:3422（`}` 结束 checkpoint 裁剪作用域）与 :3425（`// [TAG_PROMPT_LOGITS]`）之间插入：

```cpp
                        if (tree && slot.tree_heal >= 0) {
                            const llama_pos hp = slot.tree_heal;

                            if ((llama_pos) slot.prompt.n_tokens() == hp) {
                                kv_tree_io_llama io_h_tgt(ctx_tgt, slot.id);
                                std::unique_ptr<kv_tree_io_llama> io_h_dft;
                                kv_tree_io * io_h_dft_ptr = nullptr;
                                if (ctx_dft != nullptr) {
                                    io_h_dft = std::make_unique<kv_tree_io_llama>(ctx_dft, slot.id);
                                    io_h_dft_ptr = io_h_dft.get();
                                }

                                if (tree->capture_anchor(io_h_tgt, io_h_dft_ptr, slot.prompt.tokens.get_tokens(), hp)) {
                                    SLT_INF(slot, "kv tree: captured heal anchor at %d\n", hp);
                                } else {
                                    SLT_WRN(slot, "kv tree: failed to capture heal anchor at %d\n", hp);
                                }

                                slot.tree_heal = -1;
                            } else if ((llama_pos) slot.prompt.n_tokens() > hp) {
                                SLT_TRC(slot, "kv tree: heal position %d missed (now %d)\n", hp, (int) slot.prompt.n_tokens());
                                slot.tree_heal = -1;
                            }
                        }
```

- [ ] **Step 2: 批填充在 heal 位置截断**

填充循环（:3547）开头、`llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];`（:3549）之前加：

```cpp
                        // stop exactly at the heal position so the state can be captured as an anchor
                        if (tree && slot.tree_heal >= 0 && (llama_pos) slot.prompt.n_tokens() >= slot.tree_heal) {
                            break;
                        }
```

- [ ] **Step 3: idle park 分支 + 门槛**

idle 块 :2446-2462 里 `if (slot.prompt_save(*prompt_cache)) {...}` 替换为：

```cpp
                                if (tree) {
                                    slot.prompt_park(*tree);
                                } else if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }
```

`init()` 里 :1420-1423 替换为：

```cpp
        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0 && !params_base.kv_tree) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
```

（其余 else 分支不动。若 `tree` 存在且 `prompt_cache` 为空，`--cache-ram 0` 时 idle 分支不会再碰到空指针，因为有 `if (tree)`。）

- [ ] **Step 4: SLOT_ERASE drop**

:2685-2688 替换为：

```cpp
                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    if (tree && !slot->prompt.tokens.has_mtmd) {
                        if (tree->drop_seq(slot->prompt.tokens.get_tokens())) {
                            SLT_INF(*slot, "kv tree: dropped the stored sequence\n");
                        }
                    }

                    slot->prompt_clear();
```

- [ ] **Step 5: 构建 + heal/erase/idle 冒烟（修正版；场景经诊断验证）**

构建同 Task 3 Step 3 第一步。关键约束（诊断得出，必须遵守）：
- 树要参与，`f_keep < 0.5`：分叉会话的独有尾必须**长于**共享头。
- 中段锚点靠 `message_delimiters`（raw /completion 默认只有近尾部检查点）：请求体加 `"message_delimiters":[{"user":"User:"}]`，且头里带一个较早的 user 消息（提供 <= deep 的锚点）。
- `restored N == captured N` 不是不变量（后续更深的末端检查点会赢）；断言改为：恰好 1 次捕获、无 failed/missed、后续请求经树命中（prompt_n 塌缩）。
- 当 `hp == 任务长度`（请求恰好在分叉点结束）时不捕获（DONE_PROMPT 同迭代翻转，E 已知良性，记录）。

heal 场景（4 请求 A/F1/X/F2；X 为不共享的驱逐会话）：

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
$dir = "$tmp\t32-tree-heal"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$tmp\t32-slots" | Out-Null
$srv = "$repo\build\bin\Release\llama-server.exe"
$head  = "User: " + ('filler ' * 480) + "`nAssistant: ok`nUser: " + ('filler ' * 600) + "`nAssistant: ok`nUser: "
$tailA = 'alpha-tail ' * 450
$tailF = 'gamma-fork ' * 450
$p = Start-Process -FilePath $srv -ArgumentList @('-m','<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf','-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','8192','--port','8932','--cache-ram','0','--no-cache-idle-slots','--kv-tree','--tree-chunk','512','--tree-anchor-step','512','--tree-ram','256','--tree-disk',$dir,'--tree-disk-limit','256','--slot-save-path',"$tmp\t32-slots") -NoNewWindow -PassThru -RedirectStandardOutput "$tmp\s3heal-out.txt" -RedirectStandardError "$tmp\s3heal-err.txt"
for ($i=0; $i -lt 120; $i++) { Start-Sleep -Seconds 1; try { Invoke-RestMethod -Uri 'http://127.0.0.1:8932/health' -TimeoutSec 3 | Out-Null; break } catch {} }
function Req($prompt) { (Invoke-RestMethod -Uri 'http://127.0.0.1:8932/completion' -Method Post -ContentType 'application/json' -TimeoutSec 60 -Body (@{prompt=$prompt; n_predict=32; temperature=0; top_k=1; seed=42; cache_prompt=$true; message_delimiters=@(@{role='user'; delimiter='User:'})} | ConvertTo-Json -Depth 4) ) }
try {
  $rA  = Req "$head$tailA"                  # A 存档（中段 user 锚点 ~487）
  $rF1 = Req "$head$tailF"                  # F1 首次：分叉 -> heal 捕获
  $rX  = Req ("Zeta: " + ('omega ' * 1500)) # 不共享的驱逐会话，逼 A/F 出槽
  $rF2 = Req "$head$tailF"                  # F1 二次：应经树命中
} finally { $p | Stop-Process -Force }
Select-String -Path "$tmp\s3heal-err.txt" -Pattern 'kv tree: (captured heal anchor|restored|heal position|parked|restore miss)' | ForEach-Object { $_.Line }
```

Expected（与诊断日志一致的量级）：
- `restored 487 tokens (heal = 1024)`（F1 首次：锚点 487，分叉 ~1112）；
- 恰有 **1** 条 `captured heal anchor at N`；0 条 `failed to capture`；
- F2 经树命中（`restored 2443 tokens (heal = 2447)` 或 prompt_n 从 ~1960 塌到个位数）——不断言 `restored == captured`。

SLOT_ERASE drop（同一 server；erase 端点需 `--slot-save-path`，已在上面启动参数）：

```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8932/slots/0?action=erase' -Method Post
Select-String -Path "$tmp\s3heal-err.txt" -Pattern 'kv tree: dropped' | ForEach-Object { $_.Line }
```

Expected: HTTP 200 + `kv tree: dropped the stored sequence`。

idle park（必须 `-np 2`；np=1 时分派的槽总在处理中，idle 分支不会触发）：

```powershell
$dir2 = "$tmp\t32-tree-idle"; Remove-Item -Recurse -Force $dir2 -ErrorAction SilentlyContinue
$p = Start-Process -FilePath $srv -ArgumentList @('-m','<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf','-np','2','-ngl','99','-fa','on','-ctv','q8_0','-c','8192','--port','8933','--cache-ram','0','--kv-tree','--tree-chunk','512','--tree-anchor-step','512','--tree-ram','256','--tree-disk',$dir2,'--tree-disk-limit','256') -NoNewWindow -PassThru -RedirectStandardOutput "$tmp\s3idle-out.txt" -RedirectStandardError "$tmp\s3idle-err.txt"
for ($i=0; $i -lt 120; $i++) { Start-Sleep -Seconds 1; try { Invoke-RestMethod -Uri 'http://127.0.0.1:8933/health' -TimeoutSec 3 | Out-Null; break } catch {} }
function Req2($prompt) { (Invoke-RestMethod -Uri 'http://127.0.0.1:8933/completion' -Method Post -ContentType 'application/json' -TimeoutSec 60 -Body (@{prompt=$prompt; n_predict=16; temperature=0; cache_prompt=$true} | ConvertTo-Json) ) }
try { $null = Req2 ("s1 " + ('one ' * 400)); $null = Req2 ("s2 " + ('two ' * 400)); $null = Req2 ("s1 " + ('one ' * 400) + " more") } finally { $p | Stop-Process -Force }
Select-String -Path "$tmp\s3idle-err.txt" -Pattern 'kv tree: parked|requires --cache-ram' | ForEach-Object { $_.Line }
```

Expected: 至少 1 条 idle 的 `kv tree: parked`（槽 0/1 之一在处理、另一个空闲时被 park）；无 `requires --cache-ram` 警告（门槛修复生效）。

- [ ] **Step 6: 提交**

```powershell
git -C $repo add tools/server/server-context.cpp
git -C $repo commit -m "server : heal fork points, idle parks and slot erase in the kv tree" -m "Assisted-by: opencode"
```

---

### Task 5: A/B 验收（校准 + 低重叠长会话 + 多短会话 + 3B 对照 + 反例 + 回归）

**Files:**
- Create: `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1`（频道产物）
- 产物：`t32-stage3-accept.txt`（各模式汇总）、`t32-stage3-calib.txt`、`t32-stage3-neg.txt`、`t32-stage3-3b.txt`（原始日志）

**Interfaces:**
- Consumes: Task 3/4 的日志契约 + `--tree-*` CLI。
- Produces: 验收结论（脚本内 PASS/FAIL 行）+ 原始日志归档。

- [ ] **Step 1: 写验收脚本（完整）**

用 `[System.IO.File]::WriteAllText(..., UTF8Encoding($false))` 写入 `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1`：

```powershell
param(
    [Parameter(Mandatory=$true)][string]$Mode,   # calib|ab|overlap|b|b3|neg|heal|ref
    [string]$Model = '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf',
    [int]$Port = 8933,
    [string]$OutDir = '<TEMP>\v100\t32-stage3'
)

$ErrorActionPreference = 'Stop'
$repo = 'D:\LLM\Backend\src\llama.cpp-my'
$srv  = "$repo\build\bin\Release\llama-server.exe"
New-Item -ItemType Directory -Force $OutDir | Out-Null
$env:CUDA_VISIBLE_DEVICES = '0'

$fails = 0
function Assert($cond, $what) {
    if ($cond) { Write-Output "PASS  $what" } else { Write-Output "FAIL  $what"; $script:fails++ }
}

# ---- prompt construction (target ~N tokens, measured via /tokenize) ----
function TokCount($text) {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType 'application/json' -Body (@{content=$text} | ConvertTo-Json -Compress)
    return $r.tokens.Count
}
function Filler($salt, $n_tok) {
    $unit = "$salt filler sentence for the kv tree acceptance test number "
    $ut = TokCount $unit
    $t = ''
    $cur = 0
    while ($cur -lt $n_tok) {
        $need = [int][math]::Ceiling(($n_tok - $cur) / $ut)
        $t += ($unit * $need)
        $cur = TokCount $t
    }
    return $t
}

function Start-Srv($tree, $ram, $diskdir, $idle = $true) {
    $args = @('-m',$Model,'-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','24576',
              '--port',"$Port",'--cache-ram','0',
              '--no-context-shift','--host','127.0.0.1')
    if ($idle) { $args += '--cache-idle-slots' } else { $args += '--no-cache-idle-slots' }
    if ($tree) {
        $args += @('--kv-tree','--tree-chunk','512','--tree-anchor-step','4096',
                   '--tree-ram',"$ram",'--tree-disk',$diskdir,'--tree-disk-limit','2048','--tree-debug')
    }
    $p = Start-Process -FilePath $srv -ArgumentList $args -NoNewWindow -PassThru `
         -RedirectStandardOutput "$OutDir\srv-$Mode-out.txt" -RedirectStandardError "$OutDir\srv-$Mode-err.txt"
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 1000
        try { Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -Method Get | Out-Null; return $p } catch {}
    }
    throw "server did not come up"
}

function Req($prompt, $cache) {
    $body = @{prompt=$prompt; n_predict=48; temperature=0; top_k=1; seed=42; cache_prompt=$cache; stream=$false}
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -Body ($body | ConvertTo-Json -Compress)
    return $r
}

function ReqDelim($prompt, $cache) {
    $body = @{prompt=$prompt; n_predict=32; temperature=0; top_k=1; seed=42; cache_prompt=$cache; stream=$false; message_delimiters=@(@{role='user'; delimiter='User:'})}
    return Invoke-RestMethod -Uri "http://127.0.0.1:$Port/completion" -Method Post -ContentType 'application/json' -TimeoutSec 120 -Body ($body | ConvertTo-Json -Compress -Depth 4)
}

# ---- session types ----
# long sessions: shared system prompt only (low overlap -> f_keep < 0.5 -> tree park/restore)
$sys  = Filler 'system' 512
$baseA = Filler 'alpha' 16384
$baseB = Filler 'beta'  16384
# overlap sessions: share 12K (stock VRAM reuse; tree must stay out of the way)
$ovl  = Filler 'shared' 12288
$baseC = "$ovl" + (Filler 'gamma' 4096)
$baseD = "$ovl" + (Filler 'delta' 4096)
# short sessions for mode b
$sess = @()
foreach ($s in @('one','two','three','four')) { $sess += ("$sys" + (Filler $s 1536)) }

$results = @{}
$hashes  = @{}
function ContentHash($s) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $h = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($s))
    return ([System.BitConverter]::ToString($h)).Replace('-','').Substring(0,16)
}
function RunTurns($names, $bases, $rounds, $cache, $tag) {
    $hist = @{}
    foreach ($n in $names) { $hist[$n] = '' }
    for ($r = 1; $r -le $rounds; $r++) {
        for ($i = 0; $i -lt $names.Count; $i++) {
            $n = $names[$i]
            $prompt = "$sys`n" + $bases[$i] + $hist[$n] + "`nUser turn ${r}: continue`nAssistant:"
            $resp = Req $prompt $cache
            $hist[$n] += $resp.content
            $h = ContentHash $resp.content
            $hashes["$tag/$n/$r"] = $h
            $results["$tag/$n/$r"] = $resp.content
            Write-Output ("[$Mode/$tag] $n round ${r}: hash=$h prompt_n=$($resp.timings.prompt_n) pred=$($resp.timings.predicted_n) cached=$($resp.tokens_cached)")
        }
    }
}
function ComparePasses($names, $rounds) {
    for ($r = 1; $r -le $rounds; $r++) {
        foreach ($n in $names) {
            Assert ($hashes["tree/$n/$r"] -eq $hashes["full/$n/$r"]) "equal: $n round $r (tree reuse vs full prefill)"
        }
    }
}

switch ($Mode) {
    'ref' {
        $p = Start-Srv $false 0 '' $true
        RunTurns @('A','B') @($baseA,$baseB) 6 $false 'full'
        RunTurns @('C','D') @($baseC,$baseD) 3 $false 'full'
        RunTurns @('one','two','three','four') $sess 3 $false 'full'
        $p | Stop-Process -Force
    }
    'calib' {
        $dir = "$OutDir\tree-calib"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 4096 $dir
        RunTurns @('A','B') @($baseA,$baseB) 2 $true 'tree'
        $p | Stop-Process -Force
        $line = (Select-String -Path "$OutDir\srv-$Mode-err.txt" -Pattern 'kv tree: parked' | Select-Object -Last 1).Line
        Write-Output "CALIB LAST PARK: $line"
        $total = 0
        if ($line -match 'ram = (\d+) B, disk = (\d+) B') { $total = [int64]$matches[1] + [int64]$matches[2] }
        Write-Output "CALIB TOTAL BYTES AFTER 2 ROUNDS: $total"
        Assert ($total -gt 0) 'calib: tree stores data'
        Assert ((Select-String -Path "$OutDir\srv-$Mode-err.txt" -Pattern 'kv tree: restore miss').Count -ge 1) 'calib: cold restore miss is visible'
    }
    'ab' {
        $dir = "$OutDir\tree-ab"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $ram = [int]$env:T32_RAM_MIB; if ($ram -le 0) { $ram = 96 }
        $p = Start-Srv $true $ram $dir
        RunTurns @('A','B') @($baseA,$baseB) 6 $false 'full'
        RunTurns @('A','B') @($baseA,$baseB) 6 $true  'tree'
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        $parked = ($log | Select-String 'kv tree: parked').Count
        $restored = ($log | Select-String 'kv tree: restored').Count
        $miss = ($log | Select-String 'kv tree: restore miss').Count
        $diskmax = 0
        foreach ($l in $log) { if ($l -match 'ram = (\d+) B, disk = (\d+) B') { $d = [int64]$matches[2]; if ($d -gt $diskmax) { $diskmax = $d } } }
        Assert ($parked -ge 6) 'ab: park ran on every switch'
        Assert ($restored -ge 8) 'ab: later turns restored from the tree'
        Assert ($miss -eq 0)   'ab: every tree-pass turn restored from the tree'
        Assert ($diskmax -gt 0) 'ab: SSD tier actually used (disk > 0)'
        ComparePasses @('A','B') 6
    }
    'overlap' {
        $dir = "$OutDir\tree-ovl"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 96 $dir $false
        RunTurns @('C','D') @($baseC,$baseD) 3 $false 'full'
        RunTurns @('C','D') @($baseC,$baseD) 3 $true  'tree'
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        $parked = ($log | Select-String 'kv tree: parked').Count
        $restored = ($log | Select-String 'kv tree: restored').Count
        Assert ($parked -eq 0)    'overlap: no park for high-overlap switches (idle park off)'
        Assert ($restored -eq 0)  'overlap: no restore for high-overlap switches'
        ComparePasses @('C','D') 3
    }
    'b' {
        $dir = "$OutDir\tree-b"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 128 $dir
        RunTurns @('one','two','three','four') $sess 3 $false 'full'
        RunTurns @('one','two','three','four') $sess 3 $true  'tree'
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert ((($log | Select-String 'kv tree: restored').Count) -ge 4) 'b: short sessions restore from the tree'
        Assert ((($log | Select-String 'ref=') | Select-String 'ref=4').Count -ge 1) 'b: shared prefix stored once (refcount = 4)'
        ComparePasses @('one','two','three','four') 3
    }
    'b3' {
        $Model = '<models>\Qwen2.5-Coder-3B-IQ4_XS.gguf'
        $sys3 = Filler 'system' 256
        $b1 = Filler 'aa' 4096; $b2 = Filler 'bb' 4096
        $dir = "$OutDir\tree-b3"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 512 $dir
        RunTurns @('P','Q') @($b1,$b2) 3 $false 'full'
        RunTurns @('P','Q') @($b1,$b2) 3 $true  'tree'
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert (($log | Select-String 'kv tree: parked').Count -ge 1) 'b3: tree still parks pure-attention content (D11: no fork reuse)'
        Assert ((($log | Select-String 'failed to capture heal anchor').Count) -eq 0) 'b3: no failed captures'
        ComparePasses @('P','Q') 3
    }
    'neg' {
        $dir = "$OutDir\tree-neg"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 96 $dir
        RunTurns @('A','B') @($baseA,$baseB) 2 $true 'tree'
        $files = Get-ChildItem $dir -File
        Write-Output "NEG removing $($files.Count) block files"
        $files | Remove-Item -Force
        RunTurns @('A') @($baseA) 1 $true 'tree-after-loss'
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert ((($log | Select-String 'failed to read block|restore failed').Count) -ge 1) 'neg: SSD read failure is visible'
        Assert ($results.ContainsKey('tree-after-loss/A/1')) 'neg: request still succeeded after the failure'
    }
    'heal' {
        $dir = "$OutDir\tree-heal"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 512 $dir
        $h  = "User: " + ('filler ' * 480) + "`nAssistant: ok`nUser: " + ('filler ' * 600) + "`nAssistant: ok`nUser: "
        $tA = 'alpha-tail ' * 450
        $tF = 'gamma-fork ' * 450
        $reqs = @("$h$tA", "$h$tF", ("Zeta: " + ('omega ' * 1500)), "$h$tF")
        # D10: run the tree pass first on a fresh tree so the first fork restore happens with cache_prompt=true
        foreach ($pass in @(@($true,'tree'), @($false,'full'))) {
            $i = 0
            foreach ($q in $reqs) {
                $i++
                $r = ReqDelim $q $pass[0]
                $hsh = ContentHash $r.content
                $hashes["$($pass[1])/heal/$i"] = $hsh
                Write-Output ("[heal/$($pass[1])] req $i: hash=$hsh prompt_n=$($r.timings.prompt_n)")
            }
        }
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert ((($log | Select-String 'captured heal anchor').Count) -eq 1) 'heal: exactly one fork anchor captured'
        Assert ((($log | Select-String 'failed to capture heal anchor').Count) -eq 0) 'heal: no failed captures'
        for ($i = 1; $i -le 4; $i++) { Assert ($hashes["tree/heal/$i"] -eq $hashes["full/heal/$i"]) "heal: request $i identical (tree vs full prefill)" }
    }
}

Write-Output "RESULT $Mode: $fails failure(s)"
exit $fails
```

说明与调参（实施时允许按实测调，但必须保证断言语义不变）：
- `$ram` 由校准决定：`calib` 输出 `CALIB TOTAL BYTES`，阶段 3 验收取 `$env:T32_RAM_MIB = [int]($total/3/1MB)`（"精准溢出"：约 1/3 驻留、2/3 溢写 SSD）；预期 2x16K 总量 ~0.6-1.0 GB -> ram ~200-330 MiB。`ab` 默认 96 MiB 只是保守值。
- 预算按锚点大小定（重要）：2B 的 PARTIAL 锚点 ~20 MB/个；3B 纯 attention 的锚点 = 全量 KV（4K 会话 ~150 MB/个）。`b`=128 MiB、`b3`=512 MiB、`heal`=512 MiB 即为此定。`ab` 若出现"restore 命中数不足但 disk>0 成立"，允许把 ram 从 total/3 提到 total/2 再跑（保持 disk>0；在报告里记录调整与理由）。
- `ref` 用 `cache_prompt=false` + 树关，逐请求全量 prefill，仅作运行间确定性守门（见 Step 2）；主对比在模式内 `full` vs `tree` 两遍完成。
- `b` 的 refcount 证据：`dump()` 只在 `--tree-debug` 打开时打印；Start-Srv 在 `$tree` 为真时已追加 `--tree-debug`，`b` 模式断言日志里出现 `ref=4`。若日志量影响性能，可只对 `b` 保留 `--tree-debug`（其它模式去掉）。

- [ ] **Step 2: 运行 + 断言**

```powershell
$ps = 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1'
& powershell -ExecutionPolicy Bypass -File $ps -Mode calib | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-calib.txt'
$total = (Select-String -Path 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-calib.txt' -Pattern 'CALIB TOTAL BYTES: (\d+)').Matches.Groups[1].Value
$env:T32_RAM_MIB = [string][int]([int64]$total / 3 / 1MB)
& powershell -ExecutionPolicy Bypass -File $ps -Mode ab   | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab-ab.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode overlap | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab-overlap.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode b    | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab-b.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode ref  | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab-ref.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode b3   | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-3b.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode neg  | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-neg.txt'
& powershell -ExecutionPolicy Bypass -File $ps -Mode heal | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-heal.txt'
```

输出一致性断言（脚本内为主）：
1. 每个模式（`ab`/`overlap`/`b`/`b3`/`heal`）在同一进程内先跑 `full` 轮（`cache_prompt=false`，纯全量 prefill）再跑 `tree` 轮（`cache_prompt=true`），`ComparePasses` 逐轮比较内容的 SHA256 前 16 位 —— 这是设计 §7.1(b)"逐位一致"判据在 server 层的等价物（token 级一致 => 同 hash）。
2. `ref` 模式跑两遍（第二遍输出到 `*-ref2.txt`），两遍的 `[ref/full] ... hash=` 行必须逐行相同（运行间确定性守门；若不稳定，先如实报告并记录，不作为其他断言的前提）。
3. 日志证据由脚本内 Assert 覆盖：`ab` 的 parked/restored/miss/disk 峰值；`overlap` 的 parked==0 且 restored==0；`b` 的 refcount=4；`heal` 的 captured >= 1；`neg` 的读失败可见且请求成功。

- [ ] **Step 3: 汇总归档 + 文档更新**

- 把 `t32-stage3-ab.ps1`、`t32-stage3-ab-*.txt`、`t32-stage3-*.txt`、各 `srv-*.txt` 关键日志复制到 `D:\LLM\Backend\v100-collab\artifacts\`（日志已直接写在那里则跳过）。
- RESULTS.md / STATUS.md 追加阶段 3 段（UTF-8 no-BOM 追加）：集成行为、验收数字（restore 次数、disk 峰值、ram 上界、逐 token 一致性结论）、D1 场景勘误、已知缺口（MTP spec）。
- 设计文档状态行改为"阶段 0-3 已完成；阶段 4 未开始"。

- [ ] **Step 4: 提交（若有代码修正）**

验收过程中发现的集成 bug 修复走 SDD fix round（本计划回写 + 修复提交，前缀 `server :`）；纯脚本/文档改动提交：

```powershell
git -C $repo add tools/server/server-context.cpp tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp common/common.h common/arg.cpp
git -C $repo commit -m "server : fix issues found by kv tree acceptance" -m "Assisted-by: opencode"
```

---

## Self-Review（写计划时的自查）

1. **Spec 覆盖**：
   - §3.4 接缝（launch save/load + idle）→ Task 3/4；`--kv-tree` 开关 + 默认关 → Task 2；旋钮（chunk/anchor-step/ram/disk/debug）→ Task 2；VRAM 零变化（不动 unified、不加 seq）→ 设计约束（代码未碰 KV 分配）✓。
   - §3.2/4 heal 自愈 → Task 4；§5 pin/淘汰复用阶段 2 + 流式恢复补强 → Task 1；§6 可见性（WRN/INF 日志契约）→ Task 3/4 日志表；§7 验收 → Task 5（D1 勘误已标注）。
   - §2.3 引擎回归：本阶段不改 `src/llama*`，无需重跑引擎测试（阶段 1 已覆盖）。
   - 终审延后项：`bytes_load` 少计 → Task 1；restore RAM 峰值 → Task 1（D5）；CLI 校验 → Task 2；restore miss INF → Task 3；drop-seq → Task 1+4；`cfg.debug` → Task 1。其余（锚点策略打磨、统计漂移、demote 评分、32-bit 上限、部分分支测试）留阶段 4 ✓。
2. **占位符扫描**：无 TBD/TODO；Task 1 中 `n_fail_guard()` 是误留，**已删除**（见勘误）；Task 2 中 `cache_ram_mib` 覆盖语句是误留，**已删除**。
3. **类型一致性**：`tree_heal`（llama_pos）、`prompt_park/prompt_restore_tree`、`drop_seq`、`set_block_payload`、`kv_tree_config` 字段名、日志契约字符串在 Task 1/2/3/4/5 中一致 ✓。
4. **已知风险**（实施时注意）：
   - `prompt_restore_tree` 后 `prompt.tokens` 被替换，`pos_next()` 与 KV 对齐依赖 `restore()` 内部的 `seq_rm(C,-1)`；Task 3 冒烟必须验证 A 第二次请求 `prompt_n` 下降。
   - heal 截断依赖"上一批 decode 已完成"（server 的批循环语义）；失败时 Task 4 冒烟会暴露。
   - `/tokenize` 端点若不存在，`TokCount` 改用 `Req` 加 `{n_predict:0}` 读 `timings.prompt_n`（实施时先 curl 验证）。
   - 脚本的 `ref` 模式需按 Step 2 说明扩展为 A/B+C/D+短会话三组。
