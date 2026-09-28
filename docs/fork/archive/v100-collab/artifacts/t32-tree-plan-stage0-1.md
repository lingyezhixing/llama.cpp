# T32 树状 KV 存储: 阶段 0 + 阶段 1 实施计划

状态: 已执行完成并 ff 合并到本地 master (2026-09-27; 2 提交: `2b526ac74` API + `52b7bf7de` tests; 全分支审查通过; 未 push/未部署); 阶段 2-4 待另出计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给引擎加"按位置区间序列化/恢复 attention KV"的 C API (写 / 读 / append), 用引擎测试证明与全量序列化等价, 并实测 H2D 带宽与区间 API 吞吐, 为树状存储锁定 `--tree-chunk` / `--tree-anchor-step` 默认值.

**Architecture:** `llama_memory_i` 增加 `state_write_range` / `state_read_range` (基类默认抛异常 = 不支持); `llama_kv_cache` 实现: 写侧把 `state_write` 重构为带 `[p0,p1)` 过滤的 `state_write_impl`, 读侧给 `state_read_meta` 加 `append` 模式 (跳过 `seq_rm`, 拒绝位置重叠, 失败只清理本次分配的 cell); `llama_memory_hybrid` 只转发 attention 组件; 三个公开 C API 失败一律返回 0, 异常不穿过 C 边界. 不使用 range API 时现有行为不变.

**Tech Stack:** C++17 (llama.cpp fork), CMake Ninja Multi-Config, MSVC 2022, CUDA (Tesla V100 sm70, 设备 1).

**Spec:** `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md` (第 2 节 = 引擎接口, 第 3 节 = 存取路径; 本计划只做阶段 0/1)

## Global Constraints

- 仓库: `D:\LLM\Backend\src\llama.cpp-my`, HEAD `ba41cccec`, 工作树干净. 直接在 master 上改 + 增量构建, 不新建 worktree/build dir.
- 构建脚本: `<TEMP>\v100\build_test_t32.cmd` (Task 0 创建, 接受 target 列表作为参数).
- `build/` 已开 `LLAMA_BUILD_TESTS=ON`. 若 target 找不到, 先 `cmake -S . -B build -DLLAMA_BUILD_TESTS=ON`.
- 产物: `build\bin\Release\test-t32-range.exe` (Ninja Multi-Config 的 Release 子目录).
- GPU 口径: `$env:CUDA_VISIBLE_DEVICES=1`; 一律 `-ngl 99 -fa on`.
- 模型: 27B `<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf`; hybrid 2B `<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf`; 纯 attention 3B `<models>\Qwen2.5-Coder-3B-IQ4_XS.gguf`.
- 公开 API 合同: 新函数失败一律返回 0 并 `LLAMA_LOG_ERROR`; 异常不得穿过 C API; `flags != 0` 视为不支持 (返回 0).
- 回归: 不使用 range API 的现有路径行为不变; `test-state-restore-fragmented` 与 `test-save-load-state` 必须全绿.
- 代码风格: 只用 ASCII; 注释 1-2 行且只在必要处; 不改与任务无关的文件; 保持周围代码风格.
- 提交: 每个 Task 末尾的 Commit 步骤必须先把 `git diff --stat` + 摘要给用户, 获明确同意后才提交. 提交信息 `t32 : <desc>` + 空行 + `Assisted-by: opencode`. 禁止 push.
- 不碰生产目录 `D:\LLM\Backend\llama.cpp-my`; 本计划不改 server / 不加 CLI 旋钮.

## 文件结构

| 文件 | 责任 | 动作 |
|---|---|---|
| `tests/test-t32-range.cpp` | H2D 基准 + 正确性检查 + 区间吞吐基准 harness | 新建 |
| `tests/CMakeLists.txt` | 注册 `test-t32-range` (仅构建, 不注册 ctest) | 修改 (line 320 后加一行) |
| `include/llama.h` | 3 个公开 API 声明 | 修改 (line 950 后) |
| `src/llama-context.h` | 3 个私有方法声明 | 修改 (line 162 后) |
| `src/llama-context.cpp` | 3 个私有实现 + 3 个公开包装 | 修改 (line 3221 后 / line 4301 后) |
| `src/llama-memory.h` | 2 个默认虚函数 (抛异常 = 不支持) | 修改 (line 126 后) |
| `src/llama-kv-cells.h` | `seq_pos_has` 帮助函数 | 修改 (line 387 后) |
| `src/llama-kv-cache.h` | range 方法声明 + `append` 参数 + 清理帮助声明 | 修改 |
| `src/llama-kv-cache.cpp` | 写侧区间过滤 / append 读 / 重叠拒绝 / 失败清理 | 修改 |
| `src/llama-memory-hybrid.h` `.cpp` | range 调用只转发 attention | 修改 |

---

### Task 0: H2D 基准 harness + 实测

**Files:**
- Create: `tests/test-t32-range.cpp`
- Modify: `tests/CMakeLists.txt` (line 320 后)
- Create: `<TEMP>\v100\build_test_t32.cmd`

**Interfaces:**
- Consumes: 现有 `llama_state_seq_get_size_ext` / `llama_state_seq_get_data_ext` / `llama_state_seq_set_data_ext`, `llama_memory_seq_rm`, `common_params_parse`, `common_batch_add`, `common_init_from_params`.
- Produces: `test-t32-range.exe`; `fill_context(llama_context *, int n_tokens, int n_ubatch)` 帮助函数; `--mode h2d` 的实测数字 (后续 Task 在同一文件上加 mode).

- [ ] **Step 1: 写 harness 骨架 (只含 h2d mode)**

创建 `tests/test-t32-range.cpp`:

```cpp
// Benchmarks and correctness checks for the range state API (T32 tree storage)
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "usage: %s -m model.gguf [common args] --mode <h2d|correctness|range-bench> [--n N] [--chunk C]\n", prog);
}

static int fill_context(llama_context * ctx, int n_tokens, int n_ubatch = 512) {
    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);

    for (int i = 0; i < n_tokens; ) {
        batch.n_tokens = 0;

        const int n = std::min(n_ubatch, n_tokens - i);

        for (int j = 0; j < n; ++j) {
            common_batch_add(batch, 10 + (i + j) % 1000, i + j, { 0 }, j == n - 1);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at token %d\n", i);
            llama_batch_free(batch);
            return 1;
        }

        i += n;
    }

    llama_batch_free(batch);
    return 0;
}

static int run_h2d(llama_context * ctx, int n_tokens) {
    const int64_t t_fill0 = ggml_time_us();

    if (fill_context(ctx, n_tokens) != 0) {
        return 1;
    }

    const double t_fill = (ggml_time_us() - t_fill0) / 1e6;

    const size_t size = llama_state_seq_get_size_ext(ctx, 0, 0);
    if (size == 0) {
        fprintf(stderr, "failed to get state size\n");
        return 1;
    }

    std::vector<uint8_t> data(size);

    const int64_t t_d2h0 = ggml_time_us();
    const size_t n_d2h = llama_state_seq_get_data_ext(ctx, data.data(), data.size(), 0, 0);
    const double t_d2h = (ggml_time_us() - t_d2h0) / 1e6;

    llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);

    const int64_t t_h2d0 = ggml_time_us();
    const size_t n_h2d = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), 1, 0);
    const double t_h2d = (ggml_time_us() - t_h2d0) / 1e6;

    const double mib = size / 1024.0 / 1024.0;

    fprintf(stderr, "[t32-h2d] fill: %d tokens in %.1f s (%.1f t/s)\n", n_tokens, t_fill, n_tokens / t_fill);
    fprintf(stderr, "[t32-h2d] size: %.1f MiB\n", mib);
    fprintf(stderr, "[t32-h2d] D2H:  %.2f s (%.2f GiB/s) [%zu bytes]\n", t_d2h, mib / 1024.0 / t_d2h, n_d2h);
    fprintf(stderr, "[t32-h2d] H2D:  %.2f s (%.2f GiB/s) [%zu bytes]\n", t_h2d, mib / 1024.0 / t_h2d, n_h2d);

    return (n_d2h == size && n_h2d == size) ? 0 : 1;
}

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 1234;
    params.n_parallel = 3;
    params.n_ctx = 512;

    std::string mode = "correctness";
    int n_tokens = 32000;
    int chunk    = 512;

    // extract our own options before handing the rest to the common arg parser
    std::vector<char *> args;
    args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            continue;
        }
        if (arg == "--n" && i + 1 < argc) {
            n_tokens = atoi(argv[++i]);
            continue;
        }
        if (arg == "--chunk" && i + 1 < argc) {
            chunk = atoi(argv[++i]);
            continue;
        }

        args.push_back(argv[i]);
    }

    common_init();

    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    if (mode == "h2d") {
        return run_h2d(ctx, n_tokens);
    }

    print_usage(argv[0]);
    return 1;
}
```

- [ ] **Step 2: 注册构建目标**

在 `tests/CMakeLists.txt` line 320 之后 (即 `set_tests_properties(test-state-restore-fragmented ...)` 之后) 加一行:

```cmake
llama_build(test-t32-range.cpp)
```

注意: 用 `llama_build` (只构建, 不注册 ctest) - 该 harness 需要本机大模型, 不适合放进 ctest.

- [ ] **Step 3: 创建构建脚本**

创建 `<TEMP>\v100\build_test_t32.cmd`:

```bat
@echo off
call "E:\Program files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d D:\LLM\Backend\src\llama.cpp-my
cmake --build build --config Release -j %NUMBER_OF_PROCESSORS% --target %* 2>&1
```

- [ ] **Step 4: 构建**

Run: `<TEMP>\v100\build_test_t32.cmd test-t32-range`

Expected: 编译通过, 产出 `build\bin\Release\test-t32-range.exe`.

- [ ] **Step 5: 运行 H2D 基准 (27B, 需要 GPU 时段, 约 1-2 分钟)**

Run (PowerShell):
```powershell
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf -ngl 99 -fa on -np 2 -c 65536 -b 512 -ub 512 --mode h2d --n 32000
```

Expected (形状):
```
[t32-h2d] fill: 32000 tokens in ~45 s (~710 t/s)
[t32-h2d] size: ~1860 MiB
[t32-h2d] D2H:  x.xx s (x.xx GiB/s) [1950000000 bytes]
[t32-h2d] H2D:  x.xx s (x.xx GiB/s) [1950000000 bytes]
```

判读: D2H/H2D 的 GiB/s 就是树恢复的搬运带宽; 100K 的搬运时间 = 5.0 GiB / 带宽. 把这 4 个数写进 `artifacts/t32-stage0-h2d.txt`.

- [ ] **Step 6: 归档 + 记录**

- 把 Step 5 的完整输出写到 `D:\LLM\Backend\v100-collab\artifacts\t32-stage0-h2d.txt` (含完整命令行).
- 在 `TASKS\T32-agent-session-reuse.md` 末尾追加 `## Result (stage 0)` 段: 命令行 + 4 个数 + 100K 外推结论 (更新 spec 附录的 H2D 行, 把"~2s/GB 保守口径"替换为实测).

- [ ] **Step 7: Commit (需用户批准)**

先给用户看 `git diff --stat` (预期: 新增 `tests/test-t32-range.cpp`, 修改 `tests/CMakeLists.txt` 一行). 获同意后:

```bash
git add tests/test-t32-range.cpp tests/CMakeLists.txt
git commit -m "t32 : add range state API bench harness

Assisted-by: opencode"
```

---

### Task 1: 区间写 API (`get_size_range_ext` / `get_data_range_ext`)

**Files:**
- Modify: `include/llama.h` (line 950 后)
- Modify: `src/llama-context.h` (line 162 后), `src/llama-context.cpp` (line 3221 后, line 4301 后)
- Modify: `src/llama-memory.h` (line 126 后)
- Modify: `src/llama-kv-cache.h` (line 152 后公开, line 345 前私有), `src/llama-kv-cache.cpp` (line 2055-2123 重构)
- Modify: `src/llama-memory-hybrid.h` (line 77 后), `src/llama-memory-hybrid.cpp` (line 195 后)
- Modify: `tests/test-t32-range.cpp` (加 correctness mode 的写侧检查)

**Interfaces:**
- Consumes: `llama_kv_cache::state_write_impl` 的新签名 (本 Task 定义), `llama_io_write_host` / `llama_io_write_dummy`.
- Produces (后续 Task 依赖):
  - C API: `size_t llama_state_seq_get_size_range_ext(llama_context *, llama_seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags)`
  - C API: `size_t llama_state_seq_get_data_range_ext(llama_context *, uint8_t * dst, size_t size, llama_seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags)`
  - 虚函数: `llama_memory_i::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const` (基类默认抛 `std::runtime_error`)
  - 私有: `llama_kv_cache::state_write_impl(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const` (p0 < 0 表示无下界)

- [ ] **Step 1: 加 API 声明 + 抛异常的桩**

`include/llama.h` line 950 (`llama_state_seq_set_data_ext` 声明结束) 之后插入:

```c
    // Save a position range [p0, p1) of the attention KV of a single sequence.
    // Recurrent state is not included - use the PARTIAL_ONLY flags for that.
    // Returns 0 on failure (unsupported memory, invalid range).
    LLAMA_API size_t llama_state_seq_get_size_range_ext(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
                      llama_pos    p0,
                      llama_pos    p1,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_get_data_range_ext(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id,
                      llama_pos    p0,
                      llama_pos    p1,
           llama_state_seq_flags   flags);
```

`src/llama-context.h` line 162 (`state_seq_set_data` 声明) 之后插入:

```cpp
    size_t state_seq_get_size_range(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags);

    size_t state_seq_get_data_range(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_pos p0, llama_pos p1, llama_state_seq_flags flags);
```

`src/llama-memory.h` line 126 (`state_read` 虚函数) 之后插入 (并在文件头 `#include <functional>` 后加 `#include <stdexcept>`):

```cpp
    // serialize a position range [p0, p1) of the attention KV of one sequence
    virtual void state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags = 0) const {
        GGML_UNUSED(io);
        GGML_UNUSED(seq_id);
        GGML_UNUSED(p0);
        GGML_UNUSED(p1);
        GGML_UNUSED(flags);

        throw std::runtime_error("state_write_range is not supported by this memory type");
    }
```

`src/llama-kv-cache.h` / `src/llama-memory-hybrid.h` / `src/llama-kv-cache.cpp` / `src/llama-memory-hybrid.cpp` 的改动**全部放 Step 4** (override 的声明与定义必须同时出现, 否则链接失败).

`src/llama-context.cpp` line 3221 (`state_seq_set_data` 定义结束) 之后加私有实现:

```cpp
size_t llama_context::state_seq_get_size_range(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) {
    llama_io_write_dummy io(false);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        if (memory) {
            memory->state_write_range(io, seq_id, p0, p1, flags);
        }

        return io.n_bytes();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting range state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data_range(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) {
    llama_io_write_host io(dst, size);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        if (memory) {
            memory->state_write_range(io, seq_id, p0, p1, flags);
        }

        return io.n_bytes();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving range state: %s\n", __func__, err.what());
        return 0;
    }
}
```

`src/llama-context.cpp` line 4301 (`llama_state_seq_set_data_ext` 定义结束) 之后加公开包装:

```cpp
size_t llama_state_seq_get_size_range_ext(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size_range(seq_id, p0, p1, flags);
}

size_t llama_state_seq_get_data_range_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data_range(seq_id, dst, size, p0, p1, flags);
}
```

- [ ] **Step 2: 在 harness 里加写侧测试 (先失败)**

在 `tests/test-t32-range.cpp` 的 `run_h2d` 之后插入:

```cpp
static int n_fail = 0;

static void check(bool cond, const char * what) {
    fprintf(stderr, "[t32] %-72s %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        n_fail++;
    }
}

static int run_correctness(llama_model * model, llama_context * ctx) {
    const int L = 64;

    if (fill_context(ctx, L, L) != 0) {
        return 1;
    }

    const bool hybrid = llama_model_is_hybrid(model);

    const size_t size_full  = llama_state_seq_get_size_ext(ctx, 0, 0);
    const size_t size_part  = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const size_t size_range = llama_state_seq_get_size_range_ext(ctx, 0, 0, L, 0);

    fprintf(stderr, "[t32] hybrid=%d full=%zu partial=%zu range=%zu\n", hybrid, size_full, size_part, size_range);

    check(size_full > 0, "full state size > 0");
    check(size_range > 0, "range state size > 0");

    if (hybrid) {
        check(size_range == size_full - size_part + 8, "hybrid: range(0,L) size == full - partial + 8");
    } else {
        check(size_range == size_full, "attn: range(0,L) size == full size");

        std::vector<uint8_t> data_full(size_full);
        std::vector<uint8_t> data_range(size_range);

        const size_t n_full  = llama_state_seq_get_data_ext(ctx, data_full.data(), data_full.size(), 0, 0);
        const size_t n_range = llama_state_seq_get_data_range_ext(ctx, data_range.data(), data_range.size(), 0, 0, L, 0);

        check(n_full == size_full && n_range == size_range, "payload reads returned the expected sizes");
        check(n_range == n_full && memcmp(data_full.data(), data_range.data(), size_full) == 0, "attn: range(0,L) payload == full payload");
    }

    const size_t size_half = llama_state_seq_get_size_range_ext(ctx, 0, 0, L/2, 0);
    check(size_half > 8 && size_half < size_range, "range(0,L/2) size is between header and full range");

    return n_fail == 0 ? 0 : 1;
}
```

并把 `main` 里的分发改成:

```cpp
    if (mode == "h2d") {
        return run_h2d(ctx, n_tokens);
    }
    if (mode == "correctness") {
        return run_correctness(model, ctx);
    }
```

Run: `<TEMP>\v100\build_test_t32.cmd test-t32-range` 然后
```powershell
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```
Expected: FAIL (桩抛异常 -> `size_range == 0`), 进程退出码 1.

- [ ] **Step 3: 确认失败形态**

Run: 同 Step 2.

Expected:
```
[t32] hybrid=1 full=... partial=... range=0
[t32] full state size > 0                                                     PASS
[t32] range state size > 0                                                    FAIL
...
```
且 stderr 有 `error getting range state size: state_write_range is not supported by this memory type`.

- [ ] **Step 4: 落地实现 + 跑测试确认通过**

`src/llama-kv-cache.h`: line 152 (`state_read` 声明) 之后加公开声明:

```cpp
    void state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags = 0) const override;
```

line 346 (`state_write_data` 声明) 之后加私有声明:

```cpp
    // p0 < 0 means no lower bound, p1 < 0 means no upper bound
    void state_write_impl(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const;
```

`src/llama-memory-hybrid.h` line 77 (`state_read` 声明) 之后加:

```cpp
    void state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags = 0) const override;
```

`src/llama-kv-cache.cpp`: 用下面内容**整体替换**现有 `state_write` (line 2055-2123):

```cpp
void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    state_write_impl(io, seq_id, -1, -1, flags);
}

void llama_kv_cache::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        throw std::runtime_error("range state write is not supported on a mirrored kv cache");
    }

    if (seq_id < 0 || p0 < 0 || p1 <= p0) {
        throw std::runtime_error("invalid sequence id or position range");
    }

    if (flags != 0) {
        throw std::runtime_error("range state write supports flags == 0 only");
    }

    state_write_impl(io, seq_id, p0, p1, flags);
}

void llama_kv_cache::state_write_impl(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // keep only the cells in [p0, p1), when a range is given
            if (add_cell && (p0 >= 0 || p1 >= 0)) {
                add_cell = (p0 < 0 || cells.pos_get(i) >= p0) && (p1 < 0 || cells.pos_get(i) < p1);
            }

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}
```

`src/llama-memory-hybrid.cpp` line 195 (`state_write` 定义结束) 之后加:

```cpp
void llama_memory_hybrid::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_state_seq_flags flags) const {
    mem_attn->state_write_range(io, seq_id, p0, p1, flags);
}
```

Run: 同 Step 2 (先重新构建).

Expected:
```
[t32] hybrid=1 full=... partial=... range=...
[t32] hybrid: range(0,L) size == full - partial + 8                           PASS
[t32] range(0,L/2) size is between header and full range                      PASS
```

再跑纯 attention 模型 (验证逐字节相等):
```powershell
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen2.5-Coder-3B-IQ4_XS.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```
Expected:
```
[t32] hybrid=0 ...
[t32] attn: range(0,L) size == full size                                     PASS
[t32] attn: range(0,L) payload == full payload                                PASS
[t32] range(0,L/2) size is between header and full range                      PASS
```

- [ ] **Step 5: Commit (需用户批准)**

给用户看 `git diff --stat` 与两个模型的 PASS 输出. 获同意后:

```bash
git add include/llama.h src/llama-context.h src/llama-context.cpp src/llama-memory.h src/llama-kv-cache.h src/llama-kv-cache.cpp src/llama-memory-hybrid.h src/llama-memory-hybrid.cpp tests/test-t32-range.cpp
git commit -m "t32 : add range state write API for attention KV

Assisted-by: opencode"
```

---

### Task 2: 区间读 API (append 语义) + 重叠拒绝

**Files:**
- Modify: `src/llama-kv-cells.h` (line 387 后)
- Modify: `src/llama-kv-cache.h` (公开 + 私有声明), `src/llama-kv-cache.cpp` (line 2129 起 `state_read_sinfo`, line 2333 起 `state_read_meta`, line 2696 起 `state_clear`)
- Modify: `src/llama-memory.h`, `src/llama-memory-hybrid.h` `.cpp`
- Modify: `src/llama-context.h`, `src/llama-context.cpp`, `include/llama.h`
- Modify: `tests/test-t32-range.cpp` (读侧检查)

**Interfaces:**
- Consumes: Task 1 的 `state_write_range` 与测试数据; `llama_kv_cells::seq_pos_has` (本 Task 新增).
- Produces:
  - C API: `size_t llama_state_seq_set_data_range_ext(llama_context *, const uint8_t * src, size_t size, llama_seq_id, bool append, llama_state_seq_flags)`
  - 虚函数: `llama_memory_i::state_read_range(llama_io_read_i & io, llama_seq_id seq_id, bool append, llama_state_seq_flags flags)` (基类默认抛异常)
  - 私有: `llama_kv_cache::state_clear_append(llama_seq_id, uint32_t strm, const slot_info &)`, `llama_kv_cache::state_zero_data(uint32_t strm, const slot_info &)`
  - `state_read_meta(..., bool append = false)` / `state_read_sinfo(..., bool append = false)`

- [ ] **Step 1: 加 `seq_pos_has` + 声明 + 全部读侧实现**

`src/llama-kv-cells.h` line 387 (`seq_pos_max` 结束) 之后加:

```cpp
    // true if sequence seq_id has a cell at position p
    bool seq_pos_has(llama_seq_id seq_id, llama_pos p) const {
        assert(seq_id >= 0);
        assert(seq_id < LLAMA_MAX_SEQ);

        const auto & sp = seq_pos[seq_id];

        auto it = sp.lower_bound({ p, 0 });
        return it != sp.end() && it->first == p;
    }
```

`src/llama-memory.h`: 在 Task 1 加的 `state_write_range` 之后加:

```cpp
    // restore a position range of the attention KV of one sequence
    // append == true keeps the cells that the sequence already has and fails on position overlap
    virtual void state_read_range(llama_io_read_i & io, llama_seq_id seq_id, bool append, llama_state_seq_flags flags = 0) {
        GGML_UNUSED(io);
        GGML_UNUSED(seq_id);
        GGML_UNUSED(append);
        GGML_UNUSED(flags);

        throw std::runtime_error("state_read_range is not supported by this memory type");
    }
```

`src/llama-kv-cache.h`: line 152 区 (Task 1 的 `state_write_range` 之后) 加:

```cpp
    void state_read_range (llama_io_read_i  & io, llama_seq_id seq_id, bool append, llama_state_seq_flags flags = 0) override;
```

把 `state_read_sinfo` 声明 (line 175-180) 改成:

```cpp
    // state_read, plus the cells the restored tokens were placed in
    // a cache that mirrors another one (the qwen4exp indexer) must not search for its own cells: two searches agree only by luck
    //   sinfos_out: if set, filled with the layout used; a stream with no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching. one entry per stream, cell count must match the blob
    //   append    : if set, keep the cells that dest seq already has; fail on position overlap
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
          slot_info_vec_t *   sinfos_out,
    const slot_info_vec_t *   sinfos_in,
                     bool   append = false);
```

把 `state_read_meta` 声明 (line 349) 改成:

```cpp
    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    // append, when set, keeps the cells of dest_seq_id and fails on position overlap
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr, bool append = false);
```

line 352 的私有区改成:

```cpp
    void state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo);
    void state_clear_append(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo);
    void state_zero_data(uint32_t strm, const slot_info & sinfo);
```

`src/llama-kv-cache.cpp`: 把 `state_read` (line 2125-2127) 之后加:

```cpp
void llama_kv_cache::state_read_range(llama_io_read_i & io, llama_seq_id seq_id, bool append, llama_state_seq_flags flags) {
    if (seq_id < 0) {
        throw std::runtime_error("range state read requires a sequence id");
    }

    if (flags != 0) {
        throw std::runtime_error("range state read supports flags == 0 only");
    }

    state_read_sinfo(io, seq_id, flags, nullptr, nullptr, append);
}
```

把 `state_read_sinfo` (line 2129-2199) 的签名与失败分支改成:

```cpp
void llama_kv_cache::state_read_sinfo(
        llama_io_read_i & io,
           llama_seq_id   seq_id,
  llama_state_seq_flags   flags,
      slot_info_vec_t *   sinfos_out,
const slot_info_vec_t *   sinfos_in,
                 bool   append) {
```

签名后函数体不变, 只改两处:

1) line 2182 的调用加 `append`:
```cpp
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id, sinfos_in ? &(*sinfos_in)[s] : nullptr, append);
```

2) line 2190 的失败分支:
```cpp
        if (!res) {
            if (append) {
                state_clear_append(seq_id, strm, sinfo);
            } else {
                state_clear(seq_id, strm, sinfo);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
```

`state_read_meta` (line 2333 起): 定义签名改成 (函数体其余不变):

```cpp
bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id, const slot_info * sinfo_in, bool append) {
```

单序列分支开头改成:

```cpp
    if (dest_seq_id != -1) {
        // single sequence
        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        if (append && sinfo_in) {
            LLAMA_LOG_ERROR("%s: append restore does not support mirrored slot layouts\n", __func__);
            return false;
        }

        if (!append) {
            seq_rm(dest_seq_id, -1, -1);
        }
```

并在 meta 读循环里, 紧跟 `n_seq_id != 1` 检查之后插入:

```cpp
            // append must not overwrite the cells that the sequence already has
            if (append && cells.seq_pos_has(dest_seq_id, pos)) {
                LLAMA_LOG_ERROR("%s: position %d is already present in seq %d\n", __func__, pos, dest_seq_id);
                return false;
            }
```

`state_clear` (line 2695-2771): 拆成三个函数 (函数体其余部分原样搬进 `state_zero_data`):

```cpp
// the cleared ranges mirror the write pattern of state_read_data() - keep both in sync
void llama_kv_cache::state_clear(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    seq_rm(seq_id, -1, -1);

    state_zero_data(strm, sinfo);
}

// detach only the cells that a failed append restore has allocated
void llama_kv_cache::state_clear_append(llama_seq_id seq_id, uint32_t strm, const slot_info & sinfo) {
    if (sinfo.empty() || sinfo.size() == 0) {
        return;
    }

    auto & cells = v_cells[strm];

    for (uint32_t idx : sinfo.idxs[0]) {
        cells.seq_rm(idx, seq_id);
    }

    state_zero_data(strm, sinfo);
}

// zero the K/V data of the failed restore attempt - the attention can still read the data of free cells
void llama_kv_cache::state_zero_data(uint32_t strm, const slot_info & sinfo) {
    if (sinfo.empty() || sinfo.size() == 0) {
        return;
    }

    const auto & cells = v_cells[strm];

    const uint32_t cell_count = sinfo.size();

    const bool is_contiguous = sinfo.is_contiguous();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);

        if (is_contiguous) {
            llama_clear_tensor_data(k, sinfo.head() * k_size_row, cell_count * k_size_row);
        } else {
            for (uint32_t i = 0; i < cell_count; ++i) {
                llama_clear_tensor_data(k, sinfo.idxs[0][i] * k_size_row, k_size_row);
            }
        }
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

        auto * v = layer.v_stream[strm];
        if (!v) {
            continue;
        }

        if (!v_trans) {
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);

            if (is_contiguous) {
                llama_clear_tensor_data(v, sinfo.head() * v_size_row, cell_count * v_size_row);
            } else {
                for (uint32_t i = 0; i < cell_count; ++i) {
                    llama_clear_tensor_data(v, sinfo.idxs[0][i] * v_size_row, v_size_row);
                }
            }
        } else {
            const size_t v_size_el = ggml_type_size(v->type);

            if (is_contiguous) {
                const uint32_t h = sinfo.head();

                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    llama_clear_tensor_data(v, (h + j * cells.size()) * v_size_el, cell_count * v_size_el);
                }
            } else {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        llama_clear_tensor_data(v, (sinfo.idxs[0][i] + j * cells.size()) * v_size_el, v_size_el);
                    }
                }
            }
        }
    }
}
```

`src/llama-memory-hybrid.h`: Task 1 的 `state_write_range` 之后加:

```cpp
    void state_read_range (llama_io_read_i  & io, llama_seq_id seq_id, bool append, llama_state_seq_flags flags = 0) override;
```

`src/llama-memory-hybrid.cpp`: Task 1 的 `state_write_range` 之后加:

```cpp
void llama_memory_hybrid::state_read_range(llama_io_read_i & io, llama_seq_id seq_id, bool append, llama_state_seq_flags flags) {
    mem_attn->state_read_range(io, seq_id, append, flags);
}
```

`include/llama.h`: Task 1 的两个声明之后加:

```c
    // Restore a blob saved by llama_state_seq_get_data_range_ext.
    // append == false replaces the destination sequence, append == true requires
    // that none of the positions in the blob are already present in it.
    // Returns 0 on failure (unsupported memory, invalid blob, position overlap).
    LLAMA_API size_t llama_state_seq_set_data_range_ext(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   seq_id,
                            bool   append,
           llama_state_seq_flags   flags);
```

`src/llama-context.h`: Task 1 的声明之后加:

```cpp
    size_t state_seq_set_data_range(llama_seq_id seq_id, const uint8_t * src, size_t size, bool append, llama_state_seq_flags flags);
```

`src/llama-context.cpp`: Task 1 的 `state_seq_get_data_range` 之后加:

```cpp
size_t llama_context::state_seq_set_data_range(llama_seq_id seq_id, const uint8_t * src, size_t size, bool append, llama_state_seq_flags flags) {
    llama_io_read_host io(src, size);
    try {
        uint32_t magic_read;
        io.read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io.read(&seq_id_read, sizeof(seq_id_read));

        if (memory) {
            memory->state_read_range(io, seq_id, append, flags);
        }

        return io.n_bytes();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading range state: %s\n", __func__, err.what());
        io.discard();
        return 0;
    }
}
```

line 4301 区 (Task 1 的公开包装之后) 加:

```cpp
size_t llama_state_seq_set_data_range_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, bool append, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data_range(seq_id, src, size, append, flags);
}
```

- [ ] **Step 2: 在 harness 里加读侧测试 (红阶段省略: 声明与实现同块落地; TDD 红绿已在 Task 1 完整执行)**

在 `tests/test-t32-range.cpp` 的 `run_correctness` 之前加 `generate` 帮助函数:

```cpp
static std::vector<llama_token> generate(llama_context * ctx, llama_seq_id seq_id, llama_token first, llama_pos pos0, int n) {
    std::vector<llama_token> res;

    llama_sampler * smpl = llama_sampler_init_greedy();
    llama_batch batch = llama_batch_init(1, 0, 1);

    llama_token cur = first;
    llama_pos   pos = pos0;

    for (int i = 0; i < n; ++i) {
        batch.n_tokens = 0;
        common_batch_add(batch, cur, pos++, { seq_id }, true);

        if (llama_decode(ctx, batch) != 0) {
            res.clear();
            break;
        }

        cur = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, cur);
        res.push_back(cur);
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    return res;
}
```

在 `run_correctness` 的 `return n_fail == 0 ? 0 : 1;` 之前插入:

```cpp
    // chunked restore must match a full restore
    const int mid = L / 2;

    const size_t size_c0 = llama_state_seq_get_size_range_ext(ctx, 0, 0,   mid, 0);
    const size_t size_c1 = llama_state_seq_get_size_range_ext(ctx, 0, mid, L,   0);

    std::vector<uint8_t> data_c0(size_c0);
    std::vector<uint8_t> data_c1(size_c1);

    check(llama_state_seq_get_data_range_ext(ctx, data_c0.data(), size_c0, 0, 0,   mid, 0) == size_c0, "read chunk 0");
    check(llama_state_seq_get_data_range_ext(ctx, data_c1.data(), size_c1, 0, mid, L,   0) == size_c1, "read chunk 1");

    std::vector<uint8_t> data_part;
    if (hybrid) {
        data_part.resize(size_part);
        check(llama_state_seq_get_data_ext(ctx, data_part.data(), data_part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size_part, "read recurrent state");
        check(llama_state_seq_set_data_ext(ctx, data_part.data(), data_part.size(), 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size_part, "restore recurrent state into seq 1");
    }

    check(llama_state_seq_set_data_range_ext(ctx, data_c0.data(), size_c0, 1, false, 0) == size_c0, "restore chunk 0 (append=false)");
    check(llama_state_seq_set_data_range_ext(ctx, data_c1.data(), size_c1, 1, true,  0) == size_c1, "restore chunk 1 (append=true)");

    std::vector<uint8_t> data_full(size_full);
    check(llama_state_seq_get_data_ext(ctx, data_full.data(), data_full.size(), 0, 0) == size_full, "read full state");
    check(llama_state_seq_set_data_ext(ctx, data_full.data(), data_full.size(), 2, 0) == size_full, "restore full state into seq 2");

    const llama_token first = 42;
    const auto gen1 = generate(ctx, 1, first, L, 8);
    const auto gen2 = generate(ctx, 2, first, L, 8);

    check(!gen1.empty() && gen1 == gen2, "chunked restore generates the same tokens as a full restore");

    const size_t n_overlap = llama_state_seq_set_data_range_ext(ctx, data_c0.data(), size_c0, 1, true, 0);
    check(n_overlap == 0, "append with overlapping positions is rejected");

    // gen1/gen2 already decoded past L, so continue both: seq 1 went through the rejected append, seq 2 did not
    const llama_token next = gen1.empty() ? first : gen1.back();

    const auto gen1b = generate(ctx, 1, next, L + 8, 8);
    const auto gen2b = generate(ctx, 2, next, L + 8, 8);

    check(gen1b == gen2b, "rejected append left the existing state intact");
```

Run: `<TEMP>\v100\build_test_t32.cmd test-t32-range` 然后
```powershell
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```
Expected (全部 PASS, 退出码 0):
```
[t32] hybrid=1 ...
[t32] full state size > 0                                                     PASS
[t32] range state size > 0                                                    PASS
[t32] hybrid: range(0,L) size == full - partial + 8                           PASS
[t32] range(0,L/2) size is between header and full range                      PASS
[t32] read chunk 0                                                            PASS
[t32] read chunk 1                                                            PASS
[t32] read recurrent state                                                    PASS
[t32] restore recurrent state into seq 1                                      PASS
[t32] restore chunk 0 (append=false)                                          PASS
[t32] restore chunk 1 (append=true)                                           PASS
[t32] read full state                                                         PASS
[t32] restore full state into seq 2                                           PASS
[t32] chunked restore generates the same tokens as a full restore             PASS
[t32] append with overlapping positions is rejected                           PASS
[t32] rejected append left the existing state intact                          PASS
```

再跑 3B (纯 attention):
```powershell
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen2.5-Coder-3B-IQ4_XS.gguf -ngl 99 -fa on -np 3 -c 512 --mode correctness
```
Expected: 全部 PASS (其中 `read recurrent state` / `restore recurrent state` 因 hybrid=0 跳过不打印).

再跑 unified 路径 (2B, 同一份检查, 共享 stream):
```powershell
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on -np 3 -c 512 -kvu --mode correctness
```
Expected: 全部 PASS (overlap 检查在 unified 下同样生效).

注: SWA 模型不在本机模型集内. range 写路径的 SWA 掩码与现有全量路径共用同一段代码 (只多一个 pos 过滤), 由 Task 3 的 `test-save-load-state` 回归覆盖; 不为 SWA 单测.

- [ ] **Step 3: Commit (需用户批准)**

给用户看 `git diff --stat` 与三个模型的 PASS 输出. 获同意后:

```bash
git add include/llama.h src/llama-context.h src/llama-context.cpp src/llama-memory.h src/llama-kv-cache.h src/llama-kv-cache.cpp src/llama-kv-cells.h src/llama-memory-hybrid.h src/llama-memory-hybrid.cpp tests/test-t32-range.cpp
git commit -m "t32 : add range state read API with append mode

Assisted-by: opencode"
```

---

### Task 3: 区间吞吐基准 + 回归 + 默认值锁定

**Files:**
- Modify: `tests/test-t32-range.cpp` (加 `run_range_bench` + 分发)
- Create: `D:\LLM\Backend\v100-collab\artifacts\t32-stage1-bench.txt`
- Modify: `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md` (附录 + 3.4 旋钮默认值)
- Modify: `D:\LLM\Backend\v100-collab\TASKS\T32-agent-session-reuse.md` (`## Result (stage 1)`)

**Interfaces:**
- Consumes: Task 1/2 的全部 API; 现有 `test-state-restore-fragmented` / `test-save-load-state`.
- Produces: `--tree-chunk` 默认值结论; range API 每 chunk 固定开销; 是否需要批量装载路径的结论.

- [ ] **Step 1: 加区间吞吐基准 mode**

在 `tests/test-t32-range.cpp` 的 `run_correctness` 之后插入:

```cpp
static int run_range_bench(llama_context * ctx, int n_tokens, int chunk) {
    const int64_t t_fill0 = ggml_time_us();

    if (fill_context(ctx, n_tokens) != 0) {
        return 1;
    }

    const double t_fill = (ggml_time_us() - t_fill0) / 1e6;

    const int n_chunks = (n_tokens + chunk - 1) / chunk;

    std::vector<std::vector<uint8_t>> data(n_chunks);

    size_t total = 0;

    const int64_t t_write0 = ggml_time_us();
    for (int c = 0; c < n_chunks; ++c) {
        const int p0 = c * chunk;
        const int p1 = std::min(n_tokens, p0 + chunk);

        const size_t size = llama_state_seq_get_size_range_ext(ctx, 0, p0, p1, 0);
        data[c].resize(size);

        if (llama_state_seq_get_data_range_ext(ctx, data[c].data(), size, 0, p0, p1, 0) != size) {
            fprintf(stderr, "chunk %d write failed\n", c);
            return 1;
        }

        total += size;
    }
    const double t_write = (ggml_time_us() - t_write0) / 1e6;

    llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);

    const int64_t t_read0 = ggml_time_us();
    for (int c = 0; c < n_chunks; ++c) {
        if (llama_state_seq_set_data_range_ext(ctx, data[c].data(), data[c].size(), 1, c > 0, 0) != data[c].size()) {
            fprintf(stderr, "chunk %d read failed\n", c);
            return 1;
        }
    }
    const double t_read = (ggml_time_us() - t_read0) / 1e6;

    const double mib = total / 1024.0 / 1024.0;

    fprintf(stderr, "[t32-bench] n=%d chunk=%d chunks=%d fill=%.1fs total=%.1f MiB\n", n_tokens, chunk, n_chunks, t_fill, mib);
    fprintf(stderr, "[t32-bench] write: %.2f s (%.1f MiB/s, %.2f ms/chunk)\n", t_write, mib / t_write, 1e3 * t_write / n_chunks);
    fprintf(stderr, "[t32-bench] read:  %.2f s (%.1f MiB/s, %.2f ms/chunk)\n", t_read,  mib / t_read,  1e3 * t_read  / n_chunks);

    return 0;
}
```

并把 `main` 的分发改成:

```cpp
    if (mode == "h2d") {
        return run_h2d(ctx, n_tokens);
    }
    if (mode == "correctness") {
        return run_correctness(model, ctx);
    }
    if (mode == "range-bench") {
        return run_range_bench(ctx, n_tokens, chunk);
    }
```

- [ ] **Step 2: 构建**

Run: `<TEMP>\v100\build_test_t32.cmd test-t32-range`

Expected: 编译通过.

- [ ] **Step 3: 跑区间吞吐基准 (27B, GPU 时段, 约 4-6 分钟)**

Run (三次, 每次自己 prefill 32K):
```powershell
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf -ngl 99 -fa on -ctv q8_0 -np 2 -c 65536 -b 512 -ub 512 --mode range-bench --n 32000 --chunk 512
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf -ngl 99 -fa on -ctv q8_0 -np 2 -c 65536 -b 512 -ub 512 --mode range-bench --n 32000 --chunk 1024
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf -ngl 99 -fa on -ctv q8_0 -np 2 -c 65536 -b 512 -ub 512 --mode range-bench --n 32000 --chunk 2048
```

Expected (形状):
```
[t32-bench] n=32000 chunk=512 chunks=63 fill=~45s total=~1860 MiB
[t32-bench] write: x.xx s (xxx MiB/s, xx.xx ms/chunk)
[t32-bench] read:  x.xx s (xxx MiB/s, xx.xx ms/chunk)
```

判读:
- `ms/chunk` 随 chunk 变大而近似翻倍 -> 固定开销可忽略, 用 512 (复用粒度最好).
- `ms/chunk` 在 512/1024/2048 之间基本持平 -> 固定开销大, 默认上调到 1024/2048, 并把"批量装载路径"记入 spec 阶段 2 风险.
- read MiB/s 应接近 Step 5 (Task 0) 的 H2D 带宽; 明显更低则说明逐 cell 分配是瓶颈.

- [ ] **Step 4: 回归测试**

Run:
```powershell
<TEMP>\v100\build_test_t32.cmd test-state-restore-fragmented test-save-load-state
```
然后:
```powershell
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-state-restore-fragmented.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-save-load-state.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on
```

Expected:
- `test-state-restore-fragmented`: `SUCCESS - state restore works with fragmented KV cache`, 退出码 0.
- `test-save-load-state`: 末尾汇总无 FAIL (若有 SKIP 记录原因), 退出码 0.

- [ ] **Step 5: 归档 + 锁定默认值**

- 三次 bench 的完整输出 (含命令行) 写入 `D:\LLM\Backend\v100-collab\artifacts\t32-stage1-bench.txt`.
- 更新 spec `t32-tree-storage-design.md`:
  - 附录: 用实测 H2D 数字替换 "~2s/GB 保守口径"; 加 range API 的 `ms/chunk`.
  - 1.2 / 3.4: 把 `--tree-chunk` 的 "阶段 1 出口定" 换成实测结论 (默认 512 或上调).
  - 4: 若 H2D 实测远慢于 2s/GB, 按成本表重估 `--tree-anchor-step` (16K vs 32K).
- 在 `TASKS\T32-agent-session-reuse.md` 追加 `## Result (stage 1)`: bench 表 + 回归结论 + 默认值决定 + 与 spec 的对应改动.
- 生成补丁归档: `git diff ba41cccec..HEAD > D:\LLM\Backend\v100-collab\artifacts\t32-stage1-worktree.patch` (UTF-8, 全分支补丁).

- [ ] **Step 6: Commit (需用户批准)**

给用户看 `git diff --stat` (预期: `tests/test-t32-range.cpp` 追加 bench mode) 与 bench/回归输出. 获同意后:

```bash
git add tests/test-t32-range.cpp
git commit -m "t32 : add range state API throughput bench

Assisted-by: opencode"
```

---

## 阶段 2-4 路线图 (本计划不含, 阶段 1 通过后另出计划)

- **阶段 2**: 树模块 (`tools/server/server-kv-tree.{h,cpp}`): 块链/链式哈希/锚点表/refcount/heat/RAM+SSD 放置/淘汰/pin; 独立 harness 用短序列验证 树正确性 (park->restore 逐位一致, 引用计数, 淘汰次序), 不接 server.
- **阶段 3**: server 集成 (`--kv-tree` 开关 + 旋钮): `launch_slot_with_task` 的存/取两处 + idle slot 保存; A/B 验收 (spec 第 7 节), 含 replay=0 逐位一致、H2D 时间、RAM/SSD 预算.
- **阶段 4**: 淘汰/分层打磨 + 指标 (命中率/复用 token/搬运字节与耗时/锚点数/降级次数) + 反例矩阵 (无锚点 / 位置不从 0 / SSD 读失败 / 池满).
