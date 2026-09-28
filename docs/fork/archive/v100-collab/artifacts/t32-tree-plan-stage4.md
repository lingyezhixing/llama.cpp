# T32 阶段 4 实施计划: 长跑 soak + 正确性打磨 (D12/D13 + 稳定性)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在阶段 3 的 server 集成之上, 补齐恢复路径的检查点表重建 (D12) 与启动清理 (D13), 修复两个已记录的小缺陷, 并用长跑 soak (反复建树/修剪/整树删除/RAM-SSD 调度) 证明长时间运行的稳定性与正确性.

**Architecture:** 引擎侧只做三处小改 (计数修复/间距作用域修复/restore 返回锚点载荷) + ctor 清理; server 侧在 `prompt_restore_tree` 里用树锚点重建 `prompt.checkpoints` 并加聚合统计日志; 验收侧给现有 PowerShell 脚本加 `soak` 模式, 在 np=2 + 小预算下强制 churn, 周期性做逐位一致性抽检, 监控 RSS/句柄/IO 计数/磁盘占用, 结束 kill -9 重启验证清理与恢复.

**Tech Stack:** C++ (llama.cpp server + 引擎模块), PowerShell 5.1 (验收), Ninja Multi-Config (shared build).

**Spec:** `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md` (§9 阶段 4 路线) + 本计划 §Decisions (D14-D24). 设计文档状态行与缺口段在 Task 5 更新.

---

## Global Constraints

- 分支: `t32-stage4`, base = master `daf4186d3`. 用户已授权本计划在 `t32-stage4` 上自动提交 (每任务一个或两个提交); **不 push, 不建 PR, 不部署**; 生产目录 `D:\LLM\Backend\llama.cpp-my` 不动.
- 测试设备: 一律 `CUDA_VISIBLE_DEVICES=1` (cuda0 归用户白天任务). 脚本 `t32-stage3-ab.ps1` 第 12 行从 `'0'` 改为 `'1'`.
- 模型: 2B `<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf` (主); 3B `<models>\Qwen2.5-Coder-3B-IQ4_XS.gguf` 仅回归 b3. **不跑 27B**.
- 构建: 测试目标 `<TEMP>\v100\build_test_t32.cmd test-t32-tree`; server: 先删 `build\bin\Release\llama-server-impl.dll` 再 `<TEMP>\v100\build_server.cmd`. 共享 build 树保持 `LLAMA_BUILD_TESTS=ON` (部署前才恢复 OFF, 本阶段不部署).
- 日志契约 (验收 grep 依赖, 不得改动词形): `kv tree: parked`, `kv tree: restored`, `kv tree: restore miss`, `kv tree: park skipped (seq end %d, tokens %zu)`, `kv tree: captured heal anchor at %d`, `kv tree: heal anchor at %d not stored`, `kv tree: dropped the stored sequence`; 本阶段新增 `kv tree: rebuilt %zu context checkpoints`, `[kv-tree] cleared %zu stale files`, `kv tree stats: ...` (新增行不得破坏旧 grep).
- 命令环境: PowerShell 5.1; 写文本文件用 `[System.IO.File]::WriteAllText` + `UTF8Encoding($false)`; 后台 server 必须 `-RedirectStandardError`; 频道文件含裸控制字符时用临时文件拼接.
- 代码风格: 注释极少且只写非显然信息; ASCII only; 复用现有模式 (`promote_prune` 的链作用域写法, `check_eq`/`check` 断言宏).
- 磨损优化 (write-once/clean-copy/冷淘汰) **本阶段不做** (用户裁决 D14); soak 只记录 IO 计数作为后续基线.
- 时长预算: smoke 2-5 min; 验收 30 min; 最终 60 min; 全部在 cuda1.

## Decisions (D14-D24, 记录裁决与判错代价)

- **D14 (用户裁决):** SSD 磨损优化延后到后续阶段; 本阶段只做正确性与长跑稳定性; soak 记录 `ReadTransferCount/WriteTransferCount` 作为基线. 判错代价: 无 (用户明确要求).
- **D15:** soak 时长 = 2-5 min smoke + 30 min 验收 + 60 min 最终. 判错代价: 覆盖窗口偏短 (由周期抽检与 60 min 补偿).
- **D16:** D12 (检查点表重建) 与 D13 (启动清理) 进本阶段; D11 (纯 attention 无锚点恢复) 不做. 判错代价: D11 场景继续无 fork 复用 (已记录).
- **D17:** D13 语义 = 构造 `kv_tree` 时清空 `disk_dir/blocks` 与 `disk_dir/anchors` (含 `.tmp`), 目录约定单实例独占; 启动日志报清理数量. 判错代价: 共享目录会互删 (文档写明独占).
- **D18:** D12 重建上限 = `n_ctx_checkpoints` (默认 32), 超限保留最深 N 个; 只重建 `0 < pos < res.C` 的路径锚点; `pos_min = 0, pos_max = pos - 1` (等价于 stock 检查点在该 n_tokens 处的覆盖语义, 且满足回退过滤 `pos_min == 0`); 载荷 = 树锚点 PARTIAL 状态; `data_spec` 为空 (D2 已知缺口). 阶段 4 soak 修复 (c0619da14): FULL/RS 上下文改用 tail 语义 `pos_min = pos_max = pos - 1` (与 stock 检查点一致, 避免回退搜索选中后对 recurrent 上下文做 partial seq_rm), PART 上下文仍 `pos_min = 0`. 判错代价: reasoning 回退位置比 stock 早 1 token (多算 1 token, 正确性不变).
- **D19:** 验收脚本硬编码设备改为 `'1'`; 所有回归/soak 在 cuda1. 判错代价: 与阶段 3 归档数字在不同设备上, 但同型号 V100, 逐位一致性判据不受影响.
- **D20:** 聚合统计日志每 64 次 park+restore 输出一行 `kv tree stats: ...` (模块 `stats_line()`); 不新增 HTTP 端点. 判错代价: 观测靠日志 (已够 soak 解析).
- **D21:** soak 口径 = np=2, `-c 16384` (每 slot 8192), `--cache-idle-slots`, `--slot-save-path` (erase 必需), `--tree-ram 64`, `--tree-disk-limit 512`, `--tree-anchor-step 4096`, 无 `--tree-debug`. 判错代价: 预算极小时更多 evict, 与生产口径不同 (压的是机制不是性能).
- **D22:** `disk_errors` 双计数修复 = `write_disk` 不再自增, 由调用方 (`demote_block`/`demote_anchor`) 单点自增. 判错代价: 无.
- **D23:** 锚点间距 `prev` 改为链作用域, 同时修 `capture_anchor` (363-368) 与 `park` 候选采纳 (1165-1170) 两处 (与 `promote_prune` 一致). 判错代价: 跨链不再抑制, 同链锚点更密 (间距仍由 step 保证).
- **D24:** D12 重建检查点除计数上限外新增字节上限 256 MiB, 超限从最浅 (最旧) 先丢, 保留最深; 理由: PART 模型锚点载荷近似全 KV, 重建拷贝不计入 `kv_tree_stats` 与 `--tree-ram`, 属突发拷贝, 需独立封顶; 不新增 CLI 选项 (YAGNI). 判错代价: 超长 prompt 下浅层 checkpoint 不可用, reasoning 回退多算 token (正确性不变).

## 文件结构

- Modify: `tools/server/server-kv-tree.h` — `kv_tree_restore_anchor`, `kv_tree_restore::anchors` 类型, `stats_line()`.
- Modify: `tools/server/server-kv-tree.cpp` — write_disk 计数, 两处间距作用域, ctor 清理, restore 收集锚点载荷, stats_line.
- Modify: `tools/server/server-context.cpp` — `prompt_restore_tree` 重建检查点 (+签名/调用点), 聚合统计日志, `tree_ops` 成员.
- Modify: `tests/test-t32-tree.cpp` — logic: `run_logic_fixes`, `run_logic_wipe`; model: `scenario_restore_anchors`.
- Modify (artifact, 非 git): `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` — 设备 1, `Start-Srv` 新参数, `soak` 模式.
- New (artifact, 非 git): `t32-stage4-*.txt` 证据 + `t32-stage4-logs\`; 本计划文件.

---

### Task 1: 模块小修 (D22/D23) + 统计行

**Files:**
- Modify: `tools/server/server-kv-tree.h:167` (stats 附近加声明)
- Modify: `tools/server/server-kv-tree.cpp:470-510` (write_disk), `:363-368` (capture_anchor prev), `:1165-1170` (park prev), 文件尾 (stats_line 实现)
- Test: `tests/test-t32-tree.cpp` (新 `run_logic_fixes`, 注册到 `run_logic`)

**Interfaces:**
- Consumes: 现有 `kv_tree_stats`, `kv_tree_io_fake`, `make_tokens`, `check`/`check_eq`.
- Produces: `std::string kv_tree::stats_line() const` (Task 4/5 日志解析依赖 `parks=` 等字段名); 修复后的计数语义 (Task 3 测试依赖 `disk_errors` 单计数).

- [ ] **Step 1: 写失败测试**

在 `tests/test-t32-tree.cpp` 的 `run_logic()` 之前插入 (放在 `run_logic_capture` 之后):

```cpp
static void run_logic_fixes() {
    fprintf(stderr, "[t32-tree] logic: fixes\n");

    // disk_errors must count one error per failed write, not two
    {
        const auto file = std::filesystem::temp_directory_path() / "t32-tree-diskerr";
        std::error_code ec;
        std::filesystem::remove(file, ec);
        {
            std::FILE * f = fopen(file.string().c_str(), "wb");
            if (f != nullptr) {
                fputc('x', f);
                fclose(f);
            }
        }

        kv_tree_config cfg;
        cfg.ram_limit  = 0;           // force demotion
        cfg.disk_limit = 1ull << 20;
        cfg.disk_dir   = file.string(); // a file, not a directory: every write fails

        kv_tree tree(cfg);

        kv_tree_io_fake io;
        io.max_pos = 1023;
        check(!tree.park(io, nullptr, make_tokens(1024, 0), {}), "fixes: park refused when the disk tier fails");
        check_eq(tree.stats().disk_errors, 2, "fixes: one disk error per failed write (block + anchor)");

        std::filesystem::remove(file, ec);
    }

    // anchor spacing must be scoped to the chain, not global
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 4096;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);

        const auto tok_a = make_tokens(512, 0);
        const auto tok_b = make_tokens(2048, 1);

        {
            kv_tree_io_fake io;
            io.max_pos = 511;
            check(tree.park(io, nullptr, tok_a, {}), "fixes: park chain A (tip anchor at 512)");
        }

        {
            kv_tree_anchor_in ck;
            ck.pos      = 1024;
            ck.data_tgt = kv_tree_io_fake::pattern(0, 64);

            kv_tree_io_fake io;
            io.max_pos = 2047;
            check(tree.park(io, nullptr, tok_b, { ck }), "fixes: park chain B with a candidate at 1024");
        }

        // chain B now has anchors at 1024 and 2048; chain A has one at 512
        check_eq(tree.stats().anchors_added, 3, "fixes: the cross-chain anchor does not suppress the candidate");
    }

    // capture spacing must be chain-scoped too, and same-chain spacing must still apply
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 4096;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);

        const auto tok_a = make_tokens(512, 0);
        const auto tok_b = make_tokens(2048, 1);

        {
            kv_tree_io_fake io;
            io.max_pos = 511;
            check(tree.park(io, nullptr, tok_a, {}), "fixes: park chain A");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 2047;
            check(tree.park(io, nullptr, tok_b, {}), "fixes: park chain B");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 1535;
            check(tree.capture_anchor(io, nullptr, tok_b, 1536), "fixes: cross-chain anchor does not block the capture");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 2559;
            check(!tree.capture_anchor(io, nullptr, tok_b, 2560), "fixes: same-chain spacing still refuses");
        }
    }

    // stats summary line
    {
        kv_tree_config cfg;
        cfg.ram_limit = 1ull << 20;

        kv_tree tree(cfg);

        kv_tree_io_fake io;
        io.max_pos = 1023;
        check(tree.park(io, nullptr, make_tokens(1024, 0), {}), "fixes: park for the stats line");

        const std::string s = tree.stats_line();
        check(s.find("parks=1") != std::string::npos, "fixes: stats line has parks");
        check(s.find("anchors=1") != std::string::npos, "fixes: stats line has anchors");
        check(s.find("disk_err=0") != std::string::npos, "fixes: stats line has disk_err");
    }
}
```

并在 `run_logic()` (284-287) 里追加调用:

```cpp
    run_logic_capture();
    run_logic_fixes();
```

- [ ] **Step 2: 运行确认失败**

Run: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree; if ($?) { & 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic 2>&1 | Select-String 'FAIL|fixes' }`
Expected: 编译失败 (`stats_line` 不存在) 或运行 FAIL: `fixes: one disk error per failed write` (got 4), `fixes: the cross-chain anchor does not suppress the candidate` (got 2), `fixes: cross-chain anchor does not block the capture` (FAIL).

- [ ] **Step 3: 实现修复与 stats_line**

`server-kv-tree.h` 在 `const kv_tree_stats & stats() const { return st; }` 之后加:

```cpp
    std::string stats_line() const;
```

`server-kv-tree.cpp` 三处:

(1) `write_disk` (470-510) 删掉两个自增 (调用方已计数):

```cpp
bool kv_tree::write_disk(const std::string & path, const std::vector<uint8_t> & buf) {
    const std::string tmp = path + ".tmp";

    {
        std::error_code ec;   // the tier subdirectories are created on demand
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
        if (ec) {
            fprintf(stderr, "[kv-tree] failed to create the directory for %s: %s\n", tmp.c_str(), ec.message().c_str());
            return false;
        }
    }

    std::FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        fprintf(stderr, "[kv-tree] failed to open %s for writing\n", tmp.c_str());
        return false;
    }
```

(2) `capture_anchor` 的 prev (363-368) 改为链作用域:

```cpp
    llama_pos prev = -1;
    for (const auto & kv : anchors) {
        if (kv.second.pos >= pos || kv.second.pos <= prev) {
            continue;
        }
        if (std::find(chain.begin(), chain.end(), kv.first.first) != chain.end()) {
            prev = kv.second.pos;
        }
    }
```

(3) `park` 候选采纳的 prev (1165-1170) 改为链作用域:

```cpp
        llama_pos prev = -1;
        for (const auto & kv : anchors) {
            if (kv.second.pos >= c.pos || kv.second.pos <= prev) {
                continue;
            }
            if (std::find(h.begin(), h.end(), kv.first.first) != h.end()) {
                prev = kv.second.pos;
            }
        }
```

(4) 文件尾 (dump 之前) 加实现:

```cpp
std::string kv_tree::stats_line() const {
    char buf[512];

    snprintf(buf, sizeof(buf),
             "parks=%" PRId64 " ok=%" PRId64 " refused=%" PRId64
             " restore=%" PRId64 " hits=%" PRId64 " miss=%" PRId64
             " anchors=%" PRId64 " skipped=%" PRId64 " reuse_tok=%" PRId64
             " store=%" PRId64 " load=%" PRId64
             " evicted=%" PRId64 "/%" PRId64 "/%" PRId64 " evict_refused=%" PRId64
             " disk_err=%" PRId64 " ram=%" PRId64 " disk=%" PRId64,
             st.park_calls, st.park_ok, st.park_refused,
             st.restore_calls, st.restore_hits, st.restore_miss,
             st.anchors_added, st.anchors_skipped, st.tokens_reused,
             st.bytes_store, st.bytes_load,
             st.evicted_anchors, st.evicted_blocks, st.evicted_seqs, st.evict_refused,
             st.disk_errors, st.bytes_ram, st.bytes_disk);

    return buf;
}
```

- [ ] **Step 4: 运行确认通过**

Run: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree; if ($?) { & 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' --mode logic 2>&1 | Select-Object -Last 3 }`
Expected: `logic: N PASS, 0 FAIL` (N = 46 + 12 = 58).

- [ ] **Step 5: 提交**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : fix kv tree disk error counting and anchor spacing" -m "Assisted-by: opencode"
```

`stats_line` 与测试同属本任务, 但按仓库惯例拆第二个提交 (只含实现与断言行):

```powershell
# 注: 若与上一条同文件, 用 git add -p 不可行 -> 实际执行时把 stats_line 声明/实现/断言拆到本提交
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : add kv tree stats summary line" -m "Assisted-by: opencode"
```

(执行者注: 为让第二个提交干净, 可在 Step 3 时先只做修复, 提交后再加 `stats_line` + 其断言并提交. 两个提交都留在 `t32-stage4` 上.)

---

### Task 2: D13 启动清理

**Files:**
- Modify: `tools/server/server-kv-tree.cpp:182-187` (ctor)
- Test: `tests/test-t32-tree.cpp` (新 `run_logic_wipe`, 注册到 `run_logic`)

**Interfaces:**
- Consumes: `kv_tree_config.disk_dir`, `check`.
- Produces: ctor 清理语义 + 日志 `[kv-tree] cleared %zu stale files from %s` (Task 4 restart 断言 grep `cleared \d+ stale files`).

- [ ] **Step 1: 写失败测试**

`run_logic_fixes` 之后插入:

```cpp
static void run_logic_wipe() {
    fprintf(stderr, "[t32-tree] logic: startup wipe\n");

    const auto dir = std::filesystem::temp_directory_path() / "t32-tree-wipe";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir / "blocks", ec);
    std::filesystem::create_directories(dir / "anchors", ec);

    auto touch = [](const std::filesystem::path & p) {
        std::FILE * f = fopen(p.string().c_str(), "wb");
        if (f != nullptr) {
            fputc('x', f);
            fclose(f);
        }
    };

    touch(dir / "blocks" / "aa.bin");
    touch(dir / "blocks" / "bb.bin.tmp");
    touch(dir / "anchors" / "cc.bin");

    {
        kv_tree_config cfg;
        cfg.disk_dir = dir.string();

        kv_tree tree(cfg);
        check(true, "wipe: constructor with a stale disk dir");
    }

    check(!std::filesystem::exists(dir / "blocks" / "aa.bin"), "wipe: block file removed");
    check(!std::filesystem::exists(dir / "blocks" / "bb.bin.tmp"), "wipe: tmp file removed");
    check(!std::filesystem::exists(dir / "anchors" / "cc.bin"), "wipe: anchor file removed");

    {
        kv_tree_config cfg;
        kv_tree tree(cfg);
        check(true, "wipe: constructor without a disk dir");
    }

    std::filesystem::remove_all(dir, ec);
}
```

`run_logic()` 里追加 `run_logic_wipe();`.

- [ ] **Step 2: 运行确认失败**

Run: 同上 build + `--mode logic`
Expected: FAIL `wipe: block file removed` 等 (文件仍在).

- [ ] **Step 3: 实现 ctor 清理**

`server-kv-tree.cpp:182-187` 替换为:

```cpp
kv_tree::kv_tree(const kv_tree_config & cfg) : cfg(cfg) {
    if (cfg.chunk <= 0 || cfg.anchor_step < 0) {
        fprintf(stderr, "[kv-tree] invalid config: chunk = %d, anchor_step = %d\n", cfg.chunk, cfg.anchor_step);
        GGML_ABORT("invalid kv tree config");
    }

    if (!cfg.disk_dir.empty()) {
        // the tree is not persistent: drop the leftovers of a previous run
        size_t n_stale = 0;

        std::error_code ec;
        for (const auto & entry : std::filesystem::recursive_directory_iterator(cfg.disk_dir, ec)) {
            if (entry.is_regular_file(ec)) {
                n_stale++;
            }
        }

        std::filesystem::remove_all(cfg.disk_dir + "/blocks", ec);
        std::filesystem::remove_all(cfg.disk_dir + "/anchors", ec);

        fprintf(stderr, "[kv-tree] cleared %zu stale files from %s\n", n_stale, cfg.disk_dir.c_str());
    }
}
```

- [ ] **Step 4: 运行确认通过 (logic + server 冒烟)**

Run (logic): build + `--mode logic` → 0 FAIL (61 checks).

Run (server 冒烟, cuda1):

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$repo='D:\LLM\Backend\src\llama.cpp-my'
Remove-Item -Force "$repo\build\bin\Release\llama-server-impl.dll" -ErrorAction SilentlyContinue
& '<TEMP>\v100\build_server.cmd'
$d='<TEMP>\v100\t32-stage4-wipe'; Remove-Item -Recurse -Force $d -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$d\blocks","$d\anchors" | Out-Null
Set-Content -Path "$d\blocks\aa.bin" -Value 'x'; Set-Content -Path "$d\anchors\cc.bin" -Value 'x'
$p = Start-Process -FilePath "$repo\build\bin\Release\llama-server.exe" -ArgumentList @('-m','<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf','-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','4096','--port','8937','--host','127.0.0.1','--kv-tree','--tree-disk',$d,'--tree-ram','64') -NoNewWindow -PassThru -RedirectStandardError "$d\srv.txt"
for ($i=0; $i -lt 120; $i++) { Start-Sleep -Milliseconds 1000; try { Invoke-RestMethod 'http://127.0.0.1:8937/health' -TimeoutSec 3 | Out-Null; break } catch {} }
Stop-Process -Id $p.Id -Force; $p.WaitForExit(10000) | Out-Null
$cleared = (Select-String -Path "$d\srv.txt" -Pattern 'cleared \d+ stale files').Count
$left = (Get-ChildItem $d -Recurse -File -Exclude 'srv.txt').Count
Write-Output "cleared_lines=$cleared files_left=$left"
```

Expected: `cleared_lines=1 files_left=0`.

- [ ] **Step 5: 提交**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : clear stale kv tree files on startup" -m "Assisted-by: opencode"
```

---

### Task 3: D12 锚点载荷返回 + 检查点表重建

**Files:**
- Modify: `tools/server/server-kv-tree.h:143-147` (struct), `:167` 附近 (无新方法)
- Modify: `tools/server/server-kv-tree.cpp:1274-1411` (restore)
- Modify: `tools/server/server-context.cpp:348-373` (prompt_restore_tree), `:1765` (调用点), `:990-1000` 附近 (`tree_ops` 成员 + 聚合日志)
- Modify: `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` (heal 断言加 `rebuilt`)
- Test: `tests/test-t32-tree.cpp` (新 `scenario_restore_anchors`, model 模式注册)

**Interfaces:**
- Consumes: 现有 `restore`, `load_payload`, `kv_tree_anchor`, `capture_partial`, `prefill`, `check`.
- Produces: `struct kv_tree_restore_anchor { llama_pos pos; std::vector<uint8_t> data_tgt; std::vector<uint8_t> data_dft; };`; `kv_tree_restore::anchors` 变为 `std::vector<kv_tree_restore_anchor>` (升序, 只含 ≤ C 的路径锚点); server 日志 `kv tree: rebuilt %zu context checkpoints` (Task 4/5 grep).

- [ ] **Step 1: 写失败测试 (engine, model 模式)**

`tests/test-t32-tree.cpp` 里 `scenario_unaligned` 之后插入:

```cpp
static int scenario_restore_anchors(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] scenario: restore anchors\n");

    const auto tokens = make_tokens(3072, 7);

    // park with a checkpoint candidate at 2048
    {
        kv_tree tree(cfg);
        kv_tree_io_llama io(ctx, 0);

        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        if (prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 2048), 0, 512) != 0) {
            return 1;
        }

        const kv_tree_anchor_in ck = capture_partial(ctx, 0, 2048);
        check(!ck.data_tgt.empty(), "restore-anchors: capture the partial state at 2048");

        if (prefill(ctx, 0, std::vector<llama_token>(tokens.begin() + 2048, tokens.end()), 2048, 512) != 0) {
            return 1;
        }

        check(tree.park(io, nullptr, tokens, { ck }), "restore-anchors: park with the checkpoint");

        const kv_tree_restore r = tree.restore(io, nullptr, tokens);
        check_eq(r.C, 3072, "restore-anchors: restore at the tip");
        check_eq((long long) r.anchors.size(), 2, "restore-anchors: two anchors on the path");

        bool found = false;
        bool sorted = true;
        llama_pos prev = -1;
        for (const auto & a : r.anchors) {
            if (a.pos <= prev) {
                sorted = false;
            }
            prev = a.pos;
            if (a.pos == 2048) {
                found = true;
                check(a.data_tgt == ck.data_tgt, "restore-anchors: the payload matches the parked checkpoint");
            }
        }
        check(found, "restore-anchors: the checkpoint anchor is returned");
        check(sorted, "restore-anchors: anchors are sorted by position");
    }

    // same with the payloads on disk
    {
        kv_tree_config cfg2 = cfg;
        cfg2.disk_dir   = (std::filesystem::temp_directory_path() / "t32-tree-anchors").string();
        cfg2.ram_limit  = 8 * 1024;    // force demotion
        cfg2.disk_limit = 1ull << 30;  // the real 2B KV for 3072 tokens is ~150 MB

        std::error_code ec;
        std::filesystem::remove_all(cfg2.disk_dir, ec);

        kv_tree tree(cfg2);
        kv_tree_io_llama io(ctx, 0);

        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        if (prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 2048), 0, 512) != 0) {
            return 1;
        }

        const kv_tree_anchor_in ck = capture_partial(ctx, 0, 2048);

        if (prefill(ctx, 0, std::vector<llama_token>(tokens.begin() + 2048, tokens.end()), 2048, 512) != 0) {
            return 1;
        }

        check(tree.park(io, nullptr, tokens, { ck }), "restore-anchors: park with a small ram budget");
        check(tree.stats().anchors_disk + tree.stats().blocks_disk > 0, "restore-anchors: data went to disk");

        const kv_tree_restore r = tree.restore(io, nullptr, tokens);
        check_eq(r.C, 3072, "restore-anchors: restore at the tip (ssd)");

        bool found = false;
        for (const auto & a : r.anchors) {
            if (a.pos == 2048) {
                found = true;
                check(a.data_tgt == ck.data_tgt, "restore-anchors: the disk payload matches the parked checkpoint");
            }
        }
        check(found, "restore-anchors: the disk anchor is returned");

        std::filesystem::remove_all(cfg2.disk_dir, ec);
    }

    return 0;
}
```

`main` 的 model 分派 (862-869) 加一行:

```cpp
        ret |= scenario_ssd(ctx, cfg);
        ret |= scenario_restore_anchors(ctx, cfg);
        ret |= scenario_unaligned(ctx, cfg);
```

- [ ] **Step 2: 运行确认失败**

Run: `& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree; if ($?) { $env:CUDA_VISIBLE_DEVICES='1'; & 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096 2>&1 | Select-String 'FAIL|restore-anchors' }`
Expected: 编译失败 (`r.anchors` 是 `std::vector<llama_pos>`, 无 `.pos/.data_tgt`).

- [ ] **Step 3: 实现 (engine + server + heal 断言)**

(3a) `server-kv-tree.h:143-147` 替换:

```cpp
struct kv_tree_restore_anchor {
    llama_pos pos = 0;
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
};

struct kv_tree_restore {
    llama_pos C = -1;              // restore point, -1 = caller must do a full prefill
    llama_pos heal = -1;           // capture an anchor when the prefill crosses this pos
    std::vector<kv_tree_restore_anchor> anchors; // path anchors up to C, ascending (D12)
};
```

(3b) `server-kv-tree.cpp` restore 的锚点收集 (1296-1311) 替换:

```cpp
    llama_pos C = -1;
    uint64_t  c_hash = 0;
    std::vector<std::pair<uint64_t, llama_pos>> path_anchors;

    for (const uint64_t hash : cand) {
        for (auto it = anchors.lower_bound(std::make_pair(hash, 0));
             it != anchors.end() && it->first.first == hash; ++it) {
            if (it->second.pos <= m.deep) {
                path_anchors.emplace_back(hash, it->second.pos);
                if (it->second.pos > C) {
                    C = it->second.pos;
                    c_hash = hash;
                }
            }
        }
    }
```

在 `if (ok) { ... }` 块之后、解 pin 之前 (1373 之后) 插入载荷收集:

```cpp
    std::vector<kv_tree_restore_anchor> out_anchors;
    if (ok) {
        for (const auto & key : path_anchors) {
            kv_tree_anchor & a = anchors.at(key);
            if (!load_payload(a)) {
                continue;   // payload unavailable: fewer rebuilt checkpoints
            }

            kv_tree_restore_anchor ra;
            ra.pos      = a.pos;
            ra.data_tgt = a.data_tgt;
            ra.data_dft = a.data_dft;
            out_anchors.push_back(std::move(ra));
        }
    }
```

尾部 (1402-1404) 替换:

```cpp
    res.C       = C;
    res.heal    = m.deep > C ? m.deep : -1;
    res.anchors = std::move(out_anchors);
```

(3c) `server-context.cpp:348-373` 替换 `prompt_restore_tree` 头部与重建逻辑:

```cpp
    bool prompt_restore_tree(kv_tree & tree, const server_tokens & tokens, int n_ckpt_max) {
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

        kv_tree_restore res = tree.restore(io_tgt, io_dft_ptr, tokens.get_tokens());
        if (res.C <= 0) {
            SLT_INF(*this, "kv tree: restore miss for %zu tokens, full prefill\n", tokens.size());
            return false;
        }

        prompt.tokens = server_tokens(llama_tokens(tokens.get_tokens().begin(), tokens.get_tokens().begin() + res.C), false);
        prompt.checkpoints.clear();

        if (n_ckpt_max > 0) {
            std::list<common_prompt_checkpoint> cks;

            for (auto & a : res.anchors) {
                if (a.pos <= 0 || a.pos >= res.C) {
                    continue;
                }

                common_prompt_checkpoint ck;
                ck.id_task = -1;
                ck.update_pos(a.pos, 0, a.pos - 1);
                ck.data_tgt = std::move(a.data_tgt);
                ck.data_dft = std::move(a.data_dft);
                cks.push_back(std::move(ck));
            }

            while ((int) cks.size() > n_ckpt_max) {
                cks.pop_front();
            }

            prompt.checkpoints = std::move(cks);

            if (!prompt.checkpoints.empty()) {
                SLT_INF(*this, "kv tree: rebuilt %zu context checkpoints\n", prompt.checkpoints.size());
            }
        }

        tree_heal = res.heal > res.C ? res.heal : -1;

        SLT_INF(*this, "kv tree: restored %d tokens (heal = %d)\n", (int) res.C, (int) tree_heal);
        return true;
    }
```

调用点 `server-context.cpp:1765`:

```cpp
                    if (!task.params.cache_prompt || !ret->prompt_restore_tree(*tree, task.tokens, params_base.n_ctx_checkpoints)) {
```

(3d) 聚合统计日志: 在 `server_context` 的 `std::unique_ptr<kv_tree> tree;` (994) 旁加成员:

```cpp
    int64_t tree_ops = 0;
```

在 `get_available_slot` 的 `if (update_cache) { ... }` 块之后 (1779 之后) 插入:

```cpp
            if (tree && ++tree_ops % 64 == 0) {
                SRV_INF("kv tree stats: %s\n", tree->stats_line().c_str());
            }
```

(3e) heal 验收断言 (`t32-stage3-ab.ps1` heal 段, 在 `HEAL METRICS` 行附近):

```powershell
            $rebuilt = ($log | Select-String 'rebuilt \d+ context checkpoints').Count
            Write-Output "HEAL REBUILT lines=$rebuilt"
            Assert ($rebuilt -ge 1) 'heal: context checkpoints rebuilt after the fork restore (D12)'
```

- [ ] **Step 4: 运行确认通过**

Run (harness):

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096 2>&1 | Select-String 'FAIL|scenario: restore anchors'
```

Expected: 0 FAIL; `restore-anchors` 全部 PASS.

Run (server + heal 验收):

```powershell
$repo='D:\LLM\Backend\src\llama.cpp-my'
Remove-Item -Force "$repo\build\bin\Release\llama-server-impl.dll" -ErrorAction SilentlyContinue
& '<TEMP>\v100\build_server.cmd'
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode heal | Select-String 'REBUILT|RESULT|FAIL'
```

Expected: `RESULT heal: 0 failure(s)` 且 `REBUILT lines>=1`.

- [ ] **Step 5: 提交 (两个)**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : return kv tree anchor payloads on restore" -m "Assisted-by: opencode"

git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-context.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : rebuild context checkpoints after a tree restore" -m "Assisted-by: opencode"
```

(脚本改动在频道 artifacts, 不入 git.)

---

### Task 4: soak 模式 + smoke

**Files:**
- Modify: `D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1` (设备 1, `param`, `Start-Srv`, 新 helpers, `soak` 模式)
- 产物: `artifacts\t32-stage4-soak-smoke.txt` + `t32-stage4-logs\srv-soak-*.txt` (从 `$OutDir` 复制)

**Interfaces:**
- Consumes: 现有 `Assert/Stop-Srv/TokCount/Filler/Req/ContentHash` 与 Task 2 的 `cleared` 日志、Task 3 的 `rebuilt` 日志.
- Produces: `-Mode soak [-Minutes N]` 运行; 结束输出 `SOAK METRICS ...` 与 `RESULT soak: N failure(s)`.

- [ ] **Step 1: 脚本改动**

(1) 第 12 行: `$env:CUDA_VISIBLE_DEVICES = '0'` -> `'1'`.

(2) `param` 块加: `[int]$Minutes = 5`.

(3) `Start-Srv` 整体替换 (向后兼容旧调用, 新参数都有默认值):

```powershell
function Start-Srv($tree, $ram, $diskdir, $idle = $true, $anchor_step = 4096, $np = 1, $slot_save = '', $disk_mib = 2048, $tree_debug = $true, $ctx = 24576, $log_tag = '') {
    $tag = $Mode; if ($log_tag -ne '') { $tag = $log_tag }
    $sargs = @('-m',$Model,'-np',"$np",'-ngl','99','-fa','on','-ctv','q8_0','-c',"$ctx",
              '--port',"$Port",'--cache-ram','0',
              '--no-context-shift','--host','127.0.0.1')
    if ($idle) { $sargs += '--cache-idle-slots' } else { $sargs += '--no-cache-idle-slots' }
    if ($slot_save -ne '') { $sargs += @('--slot-save-path',$slot_save) }
    if ($tree) {
        $sargs += @('--kv-tree','--tree-chunk','512','--tree-anchor-step',"$anchor_step",
                   '--tree-ram',"$ram",'--tree-disk',$diskdir,'--tree-disk-limit',"$disk_mib")
        if ($tree_debug) { $sargs += '--tree-debug' }
    }
    $p = Start-Process -FilePath $srv -ArgumentList $sargs -NoNewWindow -PassThru `
         -RedirectStandardOutput "$OutDir\srv-$tag-out.txt" -RedirectStandardError "$OutDir\srv-$tag-err.txt"
    $script:srv_proc = $p
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 1000
        if ($p.HasExited) { Stop-Srv; throw "server exited early (code $($p.ExitCode))" }
        try { Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -Method Get -TimeoutSec 3 | Out-Null; return $p } catch {}
    }
    Stop-Srv
    throw "server did not come up"
}
```

(4) `switch ($Mode)` 里加 `soak` 分支:

```powershell
        'soak' {
            $dir  = "$OutDir\tree-soak"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
            $save = "$OutDir\slot-save"; Remove-Item -Recurse -Force $save -ErrorAction SilentlyContinue
            New-Item -ItemType Directory -Force $save | Out-Null

            $np = 2
            $p = Start-Srv $true 64 $dir $true 4096 $np $save 512 $false 16384
            $soak_pid = $p.Id

            Build-Sys

            $sessions = @()
            foreach ($s in @('alpha','beta','gamma','delta')) {
                $sessions += @{ body = (Filler $s 2048); hist = '' }
            }
            $fork = Filler 'shared' 1536
            $sessions += @{ body = ($fork + (Filler 'forkA' 1024)); hist = '' }
            $sessions += @{ body = ($fork + (Filler 'forkB' 1024)); hist = '' }

            $deadline = (Get-Date).AddMinutes($Minutes)
            $round = 0
            $cmp_ok = 0; $cmp_bad = 0
            $io0 = $null; $rss0 = 0; $h0 = 0
            $last_prompt = ''

            while ((Get-Date) -lt $deadline) {
                $round++
                $si = $round % $sessions.Count
                $s  = $sessions[$si]

                # the "User:" delimiter (message_delimiters in ReqDelim) makes the server create a
                # checkpoint anchor in the shared prefix; that is what D12 rebuilds from
                $prompt = "$sys`nUser: " + $s.body + $s.hist + "`nUser turn ${round}: continue`nAssistant:"
                $last_prompt = $prompt

                $r = ReqDelim $prompt $true
                $s.hist += $r.content
                if ($s.hist.Length -gt 4000) { $s.hist = $s.hist.Substring($s.hist.Length - 2000) }

                if ($round % 5 -eq 0) {
                    $rf = ReqDelim $prompt $false
                    if ((ContentHash $r.content) -eq (ContentHash $rf.content)) { $cmp_ok++ } else { $cmp_bad++ }
                }

                if ($round % 11 -eq 0) {
                    $es = [int]([math]::Floor($round / 11)) % $np
                    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/slots/$es`?action=erase" -TimeoutSec 60 | Out-Null
                }

                $io = Get-CimInstance Win32_Process -Filter "ProcessId=$soak_pid"
                if ($null -eq $io0) { $io0 = $io; $rss0 = $io.WorkingSetSize; $h0 = $io.HandleCount }

                $fs = Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
                Write-Output ("[soak] round=$round sess=$si prompt_n=$($r.timings.prompt_n) pred=$($r.timings.predicted_n) cached=$($r.tokens_cached) rss_mb=$([int]($io.WorkingSetSize/1MB)) handles=$($io.HandleCount) wr_mb=$([int]($io.WriteTransferCount/1MB)) rd_mb=$([int]($io.ReadTransferCount/1MB)) files=$($fs.Count) tree_mb=$([int]($fs.Sum/1MB))")
            }

            $alive = -not $p.HasExited
            $ioEnd = Get-CimInstance Win32_Process -Filter "ProcessId=$soak_pid"
            Stop-Srv

            $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
            $parked   = ($log | Select-String 'kv tree: parked').Count
            $restored = ($log | Select-String 'kv tree: restored').Count
            $rebuilt  = ($log | Select-String 'rebuilt \d+ context checkpoints').Count
            $failed   = ($log | Select-String 'failed to').Count
            $evref    = ($log | Select-String 'eviction could not free enough ram').Count
            $diskmax  = 0
            foreach ($l in $log) {
                if ($l -match 'ram = (\d+) B, disk = (\d+) B') {
                    $d = [int64]$matches[2]; if ($d -gt $diskmax) { $diskmax = $d }
                }
            }
            Write-Output "SOAK METRICS rounds=$round parked=$parked restored=$restored rebuilt=$rebuilt failed=$failed evict_refused=$evref diskmax=$diskmax cmp_ok=$cmp_ok cmp_bad=$cmp_bad rss_mb=$([int]($rss0/1MB))->$([int]($ioEnd.WorkingSetSize/1MB)) handles=$h0->$($ioEnd.HandleCount) wr_mb=$([int]($ioEnd.WriteTransferCount/1MB)) rd_mb=$([int]($ioEnd.ReadTransferCount/1MB))"
            Assert ($alive) 'soak: server alive at the end of the run'
            Assert ($cmp_bad -eq 0) 'soak: sampled tree vs full prefill outputs are identical'
            Assert ($cmp_ok -ge 3) 'soak: at least three comparisons ran'
            Assert ($parked -ge 5) 'soak: parks ran repeatedly'
            Assert ($restored -ge 5) 'soak: restores ran repeatedly'
            Assert ($rebuilt -ge 1) 'soak: context checkpoints were rebuilt after a tree restore (D12)'
            Assert ($failed -eq 0) 'soak: no failures in the server log'
            Assert ($evref -eq 0) 'soak: the budget always freed enough ram'
            Assert ($diskmax -le 512*1024*1024) 'soak: the disk tier stayed within the limit'
            Assert ($ioEnd.HandleCount -le $h0 + 100) 'soak: handle count stable'
            Assert ($ioEnd.WorkingSetSize -le $rss0 + 400MB) 'soak: RSS growth bounded'

            # crash-restart: stale files must be cleared and the server must work again
            $files_before = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue).Count
            $p2 = Start-Srv $true 64 $dir $true 4096 $np $save 512 $false 16384 'soak-restart'
            $log2 = Get-Content "$OutDir\srv-soak-restart-err.txt" -Encoding UTF8
            $cleared = ($log2 | Select-String 'cleared \d+ stale files').Count
            $files_after = (Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue).Count
            Write-Output "SOAK RESTART files_before=$files_before cleared_lines=$cleared files_after=$files_after"
            Assert ($cleared -ge 1) 'soak: stale tree files cleared on restart (D13)'
            Assert ($files_after -eq 0) 'soak: tree disk is empty after restart'
            $r2 = ReqDelim $last_prompt $true
            Assert ($null -ne $r2.content -and $r2.content.Length -gt 0) 'soak: request succeeds after restart'
            Stop-Srv
        }
```

- [ ] **Step 2: 2 min smoke**

```powershell
$env:T32_RAM_MIB=''
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode soak -Minutes 2 | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-soak-smoke.txt'
```

Expected: `RESULT soak: 0 failure(s)`; 输出含 `SOAK METRICS` 与 `SOAK RESTART` 行.

- [ ] **Step 3: 修正脚本问题并复跑 5 min**

把 smoke 暴露的问题 (参数拼写/断言口径/日志路径) 修好, 复跑 `-Minutes 5`, 覆盖 `t32-stage4-soak-smoke.txt`.

- [ ] **Step 4: 归档日志**

```powershell
New-Item -ItemType Directory -Force 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-logs' | Out-Null
Copy-Item '<TEMP>\v100\t32-stage3\srv-soak-*.txt' 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-logs\'
```

(无 git 提交: 脚本与证据都在频道 artifacts.)

---

### Task 5: 长跑 (30/60 min) + 全回归 + 归档 + 整理合并

**Files:**
- 产物: `artifacts\t32-stage4-soak-30.txt`, `t32-stage4-soak-60.txt`, `t32-stage4-{logic,model,accept,ab,overlap,b,b3,neg,heal,ref}.txt`, `t32-stage4-logs\`
- Modify: `artifacts\t32-tree-storage-design.md` (状态行 + §9 D12/D13 已实现), `D:\LLM\Backend\v100-collab\RESULTS.md`, `STATUS.md`, `TASKS\T32-agent-session-reuse.md`

**Interfaces:**
- Consumes: Task 1-4 全部.
- Produces: 阶段 4 终态 + 合并后的 master (未 push).

- [ ] **Step 1: 30 min 验收 soak**

```powershell
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode soak -Minutes 30 | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-soak-30.txt'
```

Expected: `RESULT soak: 0 failure(s)`; 若失败 -> 定位修复 (新提交) 后重跑.

- [ ] **Step 2: 全回归 (cuda1)**

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
$repo='D:\LLM\Backend\src\llama.cpp-my'
$art='D:\LLM\Backend\v100-collab\artifacts'
$exe="$repo\build\bin\Release\test-t32-tree.exe"
& $exe --mode logic 2>&1 | Tee-Object "$art\t32-stage4-logic.txt" | Select-Object -Last 1
& $exe -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096 2>&1 | Tee-Object "$art\t32-stage4-model.txt" | Select-Object -Last 1
& $exe -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode accept --ram-mib 4096 2>&1 | Tee-Object "$art\t32-stage4-accept.txt" | Select-Object -Last 1
$env:T32_RAM_MIB='133'
foreach ($m in @('ab','overlap','b','b3','neg','heal','ref')) {
    & powershell -ExecutionPolicy Bypass -File "$art\t32-stage3-ab.ps1" -Mode $m 2>&1 | Tee-Object "$art\t32-stage4-$m.txt" | Select-String 'RESULT|FAIL'
}
$env:T32_RAM_MIB=''
```

Expected: 每个 `RESULT <mode>: 0 failure(s)`; logic/model/accept 0 FAIL.

- [ ] **Step 3: 60 min 最终 soak**

```powershell
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode soak -Minutes 60 | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage4-soak-60.txt'
```

Expected: `RESULT soak: 0 failure(s)`; 记录 `SOAK METRICS` 全量数字 (IO 基线).

- [ ] **Step 4: 归档 + 频道文档**

- 复制本轮 srv 日志到 `artifacts\t32-stage4-logs\`.
- `t32-tree-storage-design.md` 状态行改为: `阶段 0-4 已完成 (阶段 4: soak 长跑 + D12/D13, 分支 t32-stage4 未 push); 磨损优化/持久化待后续`.
- `RESULTS.md`/`STATUS.md` 追加阶段 4 段 (数字 + D14 磨损延后 + 已知缺口), `TASKS\T32-agent-session-reuse.md` 加阶段 4 Result.

- [ ] **Step 5: 整理提交 + 合并 (沿用阶段 3 惯例)**

```powershell
$repo='D:\LLM\Backend\src\llama.cpp-my'
$want = git -C $repo rev-parse 't32-stage4^{tree}'
git -C $repo checkout -b t32-stage4-tidy master
git -C $repo checkout t32-stage4 -- tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp
git -C $repo commit -m "server : fix kv tree counters, spacing scope and startup cleanup" -m "Assisted-by: opencode"
git -C $repo checkout t32-stage4 -- tests/test-t32-tree.cpp
git -C $repo commit -m "tests : add kv tree fix, wipe and anchor payload tests" -m "Assisted-by: opencode"
git -C $repo checkout t32-stage4 -- tools/server/server-context.cpp
git -C $repo commit -m "server : rebuild context checkpoints after a tree restore" -m "Assisted-by: opencode"
$t = git -C $repo rev-parse 'HEAD^{tree}'
if ($t -ne $want) { throw "tree mismatch: $t != $want" }
git -C $repo checkout master
git -C $repo merge --ff-only t32-stage4-tidy
git -C $repo branch -d t32-stage4-tidy; git -C $repo branch -D t32-stage4
```

(合并后复跑: logic + model + 5 min soak, 全绿; 分支删除前记录 head 到 STATUS.)

---

## Self-Review

- **Spec 覆盖:** D12 (Task 3), D13 (Task 2), D22/D23 (Task 1), 聚合指标 (Task 3d), soak/压力/正确性 (Task 4-5), 磨损延后 (D14, 无代码), D11 不做 (D16). 用户清单: 反复建树 (soak 轮换+分叉), 修剪 (小预算 churn), 整树删除 (erase + restart), RAM/SSD 调度 (demote/restore/diskmax 断言), 检查点更新 (D12 + rebuilt 断言) - 全部有任务对应.
- **Placeholder 扫描:** 无 TBD/TODO; 每步含实际代码/命令/期望输出.
- **类型一致性:** `kv_tree_restore_anchor.pos/data_tgt/data_dft` 在 h/cpp/测试/server 一致; `prompt_restore_tree(tree, tokens, n_ckpt_max)` 签名与调用点一致; `stats_line()` 字段名与 Task 4 日志解析不耦合 (soak 只 grep 既有日志).
- **风险点:** (1) heal 模式 `rebuilt >= 1` 依赖请求 4 恢复在 C=1024 且 487 锚点仍在 - 阶段 3 证据支持; 若实测为 0, 在 Task 3 Step 4 记录并调整断言口径 (改在 soak 断言, 已双保险). (2) soak RSS/句柄阈值是启发式, 首轮 smoke 后可按实测微调并记录. (3) `--slot-save-path` 目录与 tree 目录分离, 互不影响.
