# T32 树状 KV 存储: 阶段 2 (树模块 + 独立 harness) 实施计划

状态: 待用户审阅 (2026-09-27)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `tools/server/server-kv-tree.{h,cpp}` 树模块 (块链/锚点/放置/淘汰) 与独立 harness `tests/test-t32-tree.cpp`, 用 2B 小模型证明 park/restore/分叉/锚点/SSD/淘汰在数据级与逐 token 级全部正确。

**Architecture:** 模块分两层: 纯逻辑层 (哈希链/匹配/锚点选择/淘汰, 无 llama_context 依赖) + `kv_tree_io` 虚拟 I/O 接口 (测试用 fake, 真机用 `kv_tree_io_llama`). 块 = 内容寻址的 KV 区间 blob; 锚点 = 位置处 recurrent (PARTIAL_ONLY) 快照; 恢复点 C = 已验证前缀内最深的可用锚点; 数据单份权威 (RAM 或 SSD 文件).

**Tech Stack:** C++17, llama.cpp 公开 C API (阶段 1 的 range API + PARTIAL_ONLY), 内置 xxhash (`vendor/hash/xxhash`), std::filesystem.

**Spec:** `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md` (v2; 本计划实现 §1-§6, 不接 server)

## Global Constraints

- 引擎路线 R1: 不开 `-kvu`, 不加 seq, VRAM 零增长; 树只存 blob, 不碰 VRAM 布局
- 生产口径 (最终验收): `-np 1`, `-ngl 99 -fa on`; 本阶段 harness 一律 `-np 1`
- 验证模型: `<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf` (hybrid: GDN recurrent + attention, 正确性主用); 运行前 `$env:CUDA_VISIBLE_DEVICES='0'`
- 阶段 2 范围: 只做模块 + harness; 不接 server (`--kv-tree` 属阶段 3); 不改 `src/` 引擎代码
- 降级全部可见: 拒绝/失败必须 fprintf WRN + 计数; 绝不静默
- 代码与注释 ASCII only; 注释极少 (只写非显然的不变量); commit 前缀 `server :` / `tests :`
- 构建: `<TEMP>\v100\build_test_t32.cmd <target>` (Ninja Multi-Config, 自动 reconfigure); 服务器侧编译校验用 `build_server.cmd` (会先删 `build\bin\Release\llama-server-impl.dll`, 见其脚本)
- 分支 `t32-stage2` 从 master (`52b7bf7de`) 起; 沿用阶段 0/1 授权: 本计划执行期间可在该分支自动提交; push / 部署 / 合并不在授权内
- 阈值: chunk 512; anchor_step 默认 32768 (harness 用 512/1024 小值); RAM 默认 8192 MiB; SSD 上限默认 65536 MiB
- 大文件不写盘: harness 的 SSD 测试用 `%TEMP%\t32-tree-<pid>` 临时目录, 结束清理
- 每晚静音规则照旧: 长构建/GPU 重负载前先要时段

## 文件结构

| 文件 | 职责 |
|---|---|
| 新增 `tools/server/server-kv-tree.h` | 模块公开类型 + `kv_tree` 类 + `kv_tree_io` 接口 + llama adapter 声明 |
| 新增 `tools/server/server-kv-tree.cpp` | 哈希链/匹配/锚点/放置/淘汰/SSD 文件 I/O |
| 修改 `tools/server/CMakeLists.txt` | `server-context` 加入两个新文件 + `vendor` include |
| 新增 `tests/test-t32-tree.cpp` | harness: `--mode logic` (fake IO 单测) / `model` (真机场景) / `accept` (mini A/B) |
| 修改 `tests/CMakeLists.txt` | `llama_build(test-t32-tree.cpp <模块源>)` + include dirs |

## 设计决定 (审阅重点)

1. **模块位置** `tools/server/` 并加入 `server-context` 静态库 (阶段 3 的 server 直接链接); harness 直接编译模块源 (测试目标独立, 不链接 mtmd/httplib).
2. **I/O 接缝** `kv_tree_io` 虚接口 (6 个方法): 纯逻辑用 fake IO 秒级单测; 真机 `kv_tree_io_llama` 薄适配器. 模块不直接引用 `llama_context`.
3. **块模型** 块 = 完整 chunk (`[k*512, (k+1)*512)`) + 每序列至多一个尾部块 (`< 512 token`); 身份 = 链式 XXH64 `h_i = XXH64(chunk_i_tokens, seed=h_{i-1})`; 命中后逐 token 校验防碰撞.
4. **锚点键** `(含锚点的块哈希, pos)` 而非全前缀哈希: 借已验证的块链做校验, 不额外存 token; pos 必须 > 0 且落在已存块内.
5. **尾部半匹配** 恢复匹配允许最后一块内逐 token 部分匹配; 但 park 只从最后一个完整块边界开始补块 (块内重复 <= 512 token, spec 已接受).
6. **信封只用于 SSD** RAM 里 payload 保持分段 vector (`data_tgt/data_dft/data_spec`); 写盘时包一层信封 (`magic/version/hash/pos/n_tgt/n_dft/n_spec/payload_hash`), 读盘校验 payload_hash 失败即丢块. 这解决 spec §9 "range payload 无独立 magic" 风险, 且不改引擎.
7. **单份权威** payload 只在 RAM 或 SSD 之一; SSD 命中先读入 RAM (`loaded` 状态), 使用后 `settle()` 决定留在 RAM (删文件) 或落回 SSD (丢 RAM).
8. **park 前置条件** 序列末端必须恰为 L (`io.pos_max() == L-1`): recurrent 不可回卷, harness 保证; 阶段 3 由 server 流程保证 (任务结束/检查点时刻).
9. **分叉提升/自愈** 只在已捕获过的位置建锚点; `capture_anchor` 距前一锚点 < anchor_step 时跳过 (与稀疏化一致); 建点后剪掉 `(前一锚点, 新点)` 间 refcount<2 的非末端锚点.
10. **阶段 4 留白** 指标只在本模块内计数 + harness 打印; server 日志/接口属阶段 4.

---

### Task 1: 模块骨架 + 哈希链 + 匹配 + park (纯逻辑, fake IO)

**Files:**
- Create: `tools/server/server-kv-tree.h`
- Create: `tools/server/server-kv-tree.cpp`
- Create: `tests/test-t32-tree.cpp`
- Modify: `tools/server/CMakeLists.txt:7-26` (源列表加 `server-kv-tree.cpp` `server-kv-tree.h`; `target_include_directories` 加 `${PROJECT_SOURCE_DIR}/vendor`)
- Modify: `tests/CMakeLists.txt:322` (在 `llama_build(test-t32-range.cpp)` 之后加新目标)

**Interfaces:**
- Consumes: 无 (Task 1 只用 fake IO)
- Produces: `kv_tree_config`, `kv_tree_anchor_kind`, `kv_tree_anchor_in`, `kv_tree_stats`, `kv_tree_io`, `kv_tree_block`, `kv_tree_anchor`, `kv_tree_seq`, `kv_tree_match`, `kv_tree::park`, `kv_tree::stats`, `kv_tree::dump`

- [ ] **Step 1: 写头文件 `tools/server/server-kv-tree.h`**

```cpp
// T32 stage 2: content-addressed tree storage for attention KV and recurrent state anchors.
// Pure logic is separated from llama_context I/O via kv_tree_io.

#pragma once

#include "llama.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct kv_tree_config {
    int         chunk       = 512;
    int         anchor_step = 32768;
    size_t      ram_limit   = 8ull  << 30;
    size_t      disk_limit  = 64ull << 30;
    std::string disk_dir;
    bool        debug       = false;
};

enum kv_tree_anchor_kind {
    KV_TREE_ANCHOR_TIP      = 0,
    KV_TREE_ANCHOR_MESSAGE  = 1,
    KV_TREE_ANCHOR_ONDEMAND = 2,
};

// anchor candidate captured by the caller (e.g. a server prompt checkpoint)
struct kv_tree_anchor_in {
    llama_pos pos = 0;
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
};

struct kv_tree_stats {
    int64_t blocks_ram      = 0;
    int64_t blocks_disk     = 0;
    int64_t anchors_ram     = 0;
    int64_t anchors_disk    = 0;
    int64_t bytes_ram       = 0;
    int64_t bytes_disk      = 0;
    int64_t park_calls      = 0;
    int64_t park_ok         = 0;
    int64_t park_refused    = 0;
    int64_t restore_calls   = 0;
    int64_t restore_hits    = 0;
    int64_t restore_miss    = 0;
    int64_t anchors_added   = 0;
    int64_t anchors_skipped = 0;
    int64_t tokens_reused   = 0;
    int64_t bytes_store     = 0;
    int64_t bytes_load      = 0;
    int64_t evicted_anchors = 0;
    int64_t evicted_blocks  = 0;
    int64_t evicted_seqs    = 0;
    int64_t evict_refused   = 0;
    int64_t disk_errors     = 0;
};

// I/O seam: the harness uses a fake, the server a llama_context adapter
struct kv_tree_io {
    virtual ~kv_tree_io() = default;

    // attention KV of [p0, p1) of the sequence, as a self-contained blob
    virtual bool get_range(llama_pos p0, llama_pos p1, std::vector<uint8_t> & out) = 0;
    // load a blob produced by get_range; append = keep the existing cells
    virtual bool set_range(const uint8_t * data, size_t size, bool append) = 0;
    // recurrent (PARTIAL_ONLY) state of the sequence at its current end
    virtual bool get_partial(std::vector<uint8_t> & out) = 0;
    virtual bool set_partial(const uint8_t * data, size_t size) = 0;
    // drop attention KV in [p0, p1); p1 = -1 means "to the end", p0 = -1 means "everything"
    virtual bool seq_rm(llama_pos p0, llama_pos p1) = 0;
    // largest position currently in the sequence, -1 when empty
    virtual llama_pos pos_max() = 0;
};

struct kv_tree_io_llama : public kv_tree_io {
    kv_tree_io_llama(llama_context * ctx, llama_seq_id seq_id);

    bool get_range(llama_pos p0, llama_pos p1, std::vector<uint8_t> & out) override;
    bool set_range(const uint8_t * data, size_t size, bool append) override;
    bool get_partial(std::vector<uint8_t> & out) override;
    bool set_partial(const uint8_t * data, size_t size) override;
    bool seq_rm(llama_pos p0, llama_pos p1) override;
    llama_pos pos_max() override;

    llama_context * ctx;
    llama_seq_id    seq_id;
};

// one chunk of attention KV, content-addressed
struct kv_tree_block {
    uint64_t    hash     = 0;   // chained content hash
    llama_pos   pos0     = 0;   // [pos0, pos1) covered by this block
    llama_pos   pos1     = 0;
    int64_t     refcount = 0;   // stored sequences referencing this block
    int64_t     heat     = 0;
    int64_t     last_used = 0;
    bool        on_disk  = false;
    bool        pinned   = false;
    std::string path;
    size_t      bytes    = 0;   // payload size, valid in both tiers
    std::vector<llama_token> tokens;
    std::vector<uint8_t>     data;
};

// recurrent state (PARTIAL_ONLY) captured at pos
struct kv_tree_anchor {
    uint64_t    blk_hash = 0;   // hash of the block containing pos
    llama_pos   pos      = 0;
    int         kind     = KV_TREE_ANCHOR_MESSAGE;
    int64_t     refcount = 0;
    int64_t     heat     = 0;
    int64_t     last_used = 0;
    bool        on_disk  = false;
    bool        pinned   = false;
    std::string path;
    size_t      bytes    = 0;   // payload size, valid in both tiers
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
    std::vector<uint8_t> data_spec;
};

struct kv_tree_seq {
    std::vector<uint64_t> chain;   // block hashes in order
    llama_pos   len       = 0;
    int64_t     last_used = 0;
    bool        pinned    = false;
};

// verified prefix of a request against the stored blocks
struct kv_tree_match {
    std::vector<uint64_t> path;    // hashes of fully matched blocks
    size_t      n_full    = 0;     // number of fully matched blocks
    size_t      n_part    = 0;     // tokens verified inside the last partial block
    uint64_t    part_hash = 0;     // hash of the partially matched block
    llama_pos   deep      = 0;     // deepest verified position
};

struct kv_tree_restore {
    llama_pos C = -1;              // restore point, -1 = caller must do a full prefill
    llama_pos heal = -1;           // capture an anchor when the prefill crosses this pos
    std::vector<llama_pos> anchors; // path anchors usable by the caller
};

class kv_tree {
public:
    explicit kv_tree(const kv_tree_config & cfg);

    // store the sequence tokens[0, L); the io state must end exactly at L
    bool park(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens,
              const std::vector<kv_tree_anchor_in> & checkpoints);

    const kv_tree_stats & stats() const { return st; }

    void dump() const;

private:
    kv_tree_match match(const std::vector<llama_token> & tokens) const;

    bool make_room(size_t need_ram);

    kv_tree_config cfg;
    kv_tree_stats  st;

    std::unordered_map<uint64_t, kv_tree_block> blocks;
    std::map<llama_pos, std::vector<uint64_t>> blocks_at;  // pos0 -> hashes (ordered: containing_block uses upper_bound)
    std::map<std::pair<uint64_t, llama_pos>, kv_tree_anchor> anchors;
    std::unordered_map<uint64_t, kv_tree_seq> seqs;                  // tip hash -> sequence

    int64_t now = 0;
};
```

- [ ] **Step 2: 写 `tools/server/server-kv-tree.cpp` (Task 1 部分)**

```cpp
#include "server-kv-tree.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>

#define XXH_INLINE_ALL
#include "hash/xxhash/xxhash.h"

static uint64_t chunk_hash(const llama_token * tok, size_t n, uint64_t seed) {
    return XXH64(tok, n * sizeof(llama_token), seed);
}

static void chain_hashes(const std::vector<llama_token> & tokens, int chunk, std::vector<uint64_t> & h) {
    h.clear();

    uint64_t cur = 0;

    for (size_t a = 0; a < tokens.size(); a += (size_t) chunk) {
        const size_t b = std::min(tokens.size(), a + (size_t) chunk);
        cur = chunk_hash(tokens.data() + a, b - a, cur);
        h.push_back(cur);
    }
}

kv_tree::kv_tree(const kv_tree_config & cfg) : cfg(cfg) {
}

kv_tree_match kv_tree::match(const std::vector<llama_token> & tokens) const {
    kv_tree_match m;

    std::vector<uint64_t> h;
    chain_hashes(tokens, cfg.chunk, h);

    for (size_t i = 0; i < h.size(); ++i) {
        const auto it = blocks.find(h[i]);
        if (it == blocks.end()) {
            break;
        }

        const kv_tree_block & b = it->second;
        const size_t a = i * (size_t) cfg.chunk;

        if (a + b.tokens.size() > tokens.size()) {
            break;
        }
        if (memcmp(tokens.data() + a, b.tokens.data(), b.tokens.size() * sizeof(llama_token)) != 0) {
            break;
        }

        m.path.push_back(h[i]);
        m.n_full++;
        m.deep = (llama_pos) (a + b.tokens.size());
    }

    // partial match inside the last chunk only when the full chain is verified
    const size_t n_full_req = tokens.size() / (size_t) cfg.chunk;
    const size_t tail_a = n_full_req * (size_t) cfg.chunk;

    if (m.n_full == n_full_req && tail_a < tokens.size()) {
        const auto it = blocks_at.find((llama_pos) tail_a);
        if (it != blocks_at.end()) {
            for (const uint64_t hash : it->second) {
                const kv_tree_block & b = blocks.at(hash);

                const size_t n = std::min(b.tokens.size(), tokens.size() - tail_a);
                size_t k = 0;
                while (k < n && b.tokens[k] == tokens[tail_a + k]) {
                    ++k;
                }

                if (k > m.n_part) {
                    m.n_part = k;
                    m.part_hash = hash;
                }
            }
        }

        m.deep = std::max(m.deep, (llama_pos) (tail_a + m.n_part));
    }

    return m;
}

bool kv_tree::make_room(size_t need_ram) {
    return (size_t) st.bytes_ram + need_ram <= cfg.ram_limit;
}

bool kv_tree::park(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens,
                   const std::vector<kv_tree_anchor_in> & checkpoints) {
    (void) io_dft;
    (void) checkpoints;

    st.park_calls++;

    const llama_pos L = (llama_pos) tokens.size();

    if (L <= 0 || io_tgt.pos_max() != L - 1) {
        fprintf(stderr, "[kv-tree] park refused: sequence end %d, expected %d\n", io_tgt.pos_max(), L - 1);
        st.park_refused++;
        return false;
    }

    std::vector<uint64_t> h;
    chain_hashes(tokens, cfg.chunk, h);

    const uint64_t tip = h.back();

    if (seqs.count(tip) != 0) {
        seqs[tip].last_used = ++now;
        if (!checkpoints.empty()) {
            fprintf(stderr, "[kv-tree] park: sequence already stored, %zu checkpoint candidates not adopted\n", checkpoints.size());
            st.anchors_skipped += (int64_t) checkpoints.size();
        }
        st.park_ok++;
        return true;
    }

    const kv_tree_match m = match(tokens);

    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> blobs;
    std::vector<size_t> idxs;

    for (size_t i = m.n_full; i < h.size(); ++i) {
        if (blocks.count(h[i]) != 0) {
            continue;
        }

        const size_t a = i * (size_t) cfg.chunk;
        const size_t b = std::min(tokens.size(), a + (size_t) cfg.chunk);

        std::vector<uint8_t> blob;
        if (!io_tgt.get_range((llama_pos) a, (llama_pos) b, blob)) {
            fprintf(stderr, "[kv-tree] park refused: failed to read range [%zu, %zu)\n", a, b);
            st.park_refused++;
            return false;
        }

        blobs.emplace_back(h[i], std::move(blob));
        idxs.push_back(i);
    }

    size_t need = 0;
    for (const auto & p : blobs) {
        need += p.second.size();
    }

    if (!make_room(need)) {
        fprintf(stderr, "[kv-tree] park refused: no room for %zu bytes\n", need);
        st.park_refused++;
        return false;
    }

    for (size_t k = 0; k < blobs.size(); ++k) {
        const size_t i = idxs[k];
        const size_t a = i * (size_t) cfg.chunk;
        const size_t b = std::min(tokens.size(), a + (size_t) cfg.chunk);

        kv_tree_block & nb = blocks[blobs[k].first];

        nb.hash      = blobs[k].first;
        nb.pos0      = (llama_pos) a;
        nb.pos1      = (llama_pos) b;
        nb.refcount  = 1;
        nb.last_used = ++now;
        nb.tokens.assign(tokens.begin() + a, tokens.begin() + b);
        nb.data      = std::move(blobs[k].second);
        nb.bytes     = nb.data.size();

        blocks_at[nb.pos0].push_back(nb.hash);
        st.blocks_ram++;
        st.bytes_ram += (int64_t) nb.data.size();
    }

    for (const uint64_t hash : m.path) {
        kv_tree_block & b = blocks[hash];
        b.refcount++;
        b.heat++;
        b.last_used = ++now;
    }
    if (m.n_part > 0) {
        kv_tree_block & b = blocks[m.part_hash];
        b.refcount++;
        b.heat++;
        b.last_used = ++now;
    }

    kv_tree_seq & s = seqs[tip];
    s.chain     = h;
    s.len       = L;
    s.last_used = ++now;

    st.park_ok++;
    return true;
}

void kv_tree::dump() const {
    fprintf(stderr, "[kv-tree] blocks: %" PRId64 " ram, %" PRId64 " disk, %" PRId64 " bytes ram, %" PRId64 " bytes disk\n",
            st.blocks_ram, st.blocks_disk, st.bytes_ram, st.bytes_disk);
    fprintf(stderr, "[kv-tree] anchors: %" PRId64 " ram, %" PRId64 " disk, %" PRId64 " added, %" PRId64 " skipped\n",
            st.anchors_ram, st.anchors_disk, st.anchors_added, st.anchors_skipped);
    fprintf(stderr, "[kv-tree] park: %" PRId64 " calls, %" PRId64 " ok, %" PRId64 " refused\n",
            st.park_calls, st.park_ok, st.park_refused);
    fprintf(stderr, "[kv-tree] restore: %" PRId64 " calls, %" PRId64 " hits, %" PRId64 " miss, %" PRId64 " tokens reused\n",
            st.restore_calls, st.restore_hits, st.restore_miss, st.tokens_reused);

    for (const auto & p : blocks) {
        const kv_tree_block & b = p.second;
        fprintf(stderr, "[kv-tree]   block %016" PRIx64 " [%d, %d) ref=%" PRId64 " heat=%" PRId64 " %s %zu bytes\n",
                b.hash, b.pos0, b.pos1, b.refcount, b.heat, b.on_disk ? "disk" : "ram", b.bytes);
    }
    for (const auto & p : anchors) {
        const kv_tree_anchor & a = p.second;
        fprintf(stderr, "[kv-tree]   anchor %016" PRIx64 "@%d kind=%d ref=%" PRId64 " heat=%" PRId64 " %s\n",
                a.blk_hash, a.pos, a.kind, a.refcount, a.heat, a.on_disk ? "disk" : "ram");
    }
    for (const auto & p : seqs) {
        const kv_tree_seq & s = p.second;
        fprintf(stderr, "[kv-tree]   seq tip=%016" PRIx64 " len=%d blocks=%zu\n", p.first, s.len, s.chain.size());
    }
}
```

- [ ] **Step 3: 写 harness 骨架 `tests/test-t32-tree.cpp` (fake IO + Task 1 测试)**

```cpp
// Harness for the T32 KV tree module (stage 2)
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "server-kv-tree.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool cond, const char * what) {
    fprintf(stderr, "[t32-tree] %-72s %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        n_fail++;
    }
}

static void check_eq(long long got, long long want, const char * what) {
    fprintf(stderr, "[t32-tree] %-72s %s (got %lld, want %lld)\n", what, got == want ? "PASS" : "FAIL", got, want);
    if (got != want) {
        n_fail++;
    }
}

// fake io: payload bytes are a deterministic function of the position, so that
// identical prefixes produce identical blobs (models "attention KV is a function of the prefix")
struct kv_tree_io_fake : public kv_tree_io {
    llama_pos max_pos = -1;
    int64_t set_range_bytes = 0;
    int64_t set_range_calls = 0;
    int64_t set_partial_calls = 0;

    static std::vector<uint8_t> pattern(llama_pos p0, size_t n) {
        std::vector<uint8_t> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = (uint8_t) (p0 * 31 + i * 7);
        }
        return v;
    }

    bool get_range(llama_pos p0, llama_pos p1, std::vector<uint8_t> & out) override {
        out = pattern(p0, (size_t) (p1 - p0) * 8);
        return true;
    }
    bool set_range(const uint8_t * data, size_t size, bool append) override {
        (void) data;
        (void) append;
        set_range_calls++;
        set_range_bytes += (int64_t) size;
        return true;
    }
    bool get_partial(std::vector<uint8_t> & out) override {
        out = pattern(0, 64);
        return true;
    }
    bool set_partial(const uint8_t * data, size_t size) override {
        (void) data;
        (void) size;
        set_partial_calls++;
        return true;
    }
    bool seq_rm(llama_pos p0, llama_pos p1) override {
        if (p0 == -1) {
            max_pos = -1;
        } else if (p1 == -1) {
            max_pos = p0 - 1;
        }
        return true;
    }
    llama_pos pos_max() override {
        return max_pos;
    }
};

static std::vector<llama_token> make_tokens(int n, int salt) {
    std::vector<llama_token> t(n);
    for (int i = 0; i < n; ++i) {
        t[i] = 10 + (i * 7 + salt) % 1000;
    }
    return t;
}

static int run_logic() {
    fprintf(stderr, "[t32-tree] mode logic\n");

    kv_tree_config cfg;
    cfg.chunk = 512;
    cfg.ram_limit = 64ull << 20;

    kv_tree tree(cfg);

    const auto tok_a = make_tokens(1536, 0);   // 3 full chunks
    const auto tok_b = make_tokens(1024, 0);   // shares the first 2 chunks with tok_a, then diverges
    auto tok_b2 = tok_b;
    tok_b2.resize(1536);
    for (int i = 1024; i < 1536; ++i) {
        tok_b2[i] = 500 + i % 100;             // same as tok_a in [0, 1024), different tail
    }
    auto tok_c = make_tokens(1024, 0);         // shared prefix only, short tail
    tok_c.resize(1124);
    for (int i = 1024; i < 1124; ++i) {
        tok_c[i] = 700 + i % 50;
    }

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_a, {}), "park A (3 chunks)");
        check_eq(tree.stats().blocks_ram, 3, "blocks after A");
    }
    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_b2, {}), "park B2 (shares 2 chunks)");
        check_eq(tree.stats().blocks_ram, 4, "blocks after B2 (dedup)");
    }
    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_a, {}), "park A again (identical)");
        check_eq(tree.stats().blocks_ram, 4, "blocks after A re-park (no duplicate)");
    }
    {
        kv_tree_io_fake io;
        io.max_pos = 1123;
        check(tree.park(io, nullptr, tok_c, {}), "park C (partial tail block)");
        check_eq(tree.stats().blocks_ram, 5, "blocks after C (tail is a new block)");
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::string mode = "logic";

    std::vector<char *> args;
    args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            continue;
        }

        args.push_back(argv[i]);
    }

    if (mode == "logic") {
        return run_logic();
    }

    fprintf(stderr, "mode %s not implemented yet\n", mode.c_str());
    return 1;
}
```

- [ ] **Step 4: 改 CMake, 构建并跑 logic 测试**

`tools/server/CMakeLists.txt`: 在 `add_library(${TARGET} STATIC ...)` 源列表按字母序插入 `server-kv-tree.cpp` 与 `server-kv-tree.h`; 在 `target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})` 之后加一行:

```cmake
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/vendor)
```

`tests/CMakeLists.txt`: 在 `llama_build(test-t32-range.cpp)` 之后加:

```cmake
llama_build(test-t32-tree.cpp ${PROJECT_SOURCE_DIR}/tools/server/server-kv-tree.cpp)
target_include_directories(test-t32-tree PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/vendor)
```

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
```
Expected: `mode logic`, 全部 PASS, exit 0. 另跑 `& '...\build_server.cmd'` 确认模块在 server-context 里也能编译 (Task 1 出口检查, 不部署).

- [ ] **Step 5: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tools/server/CMakeLists.txt tests/test-t32-tree.cpp tests/CMakeLists.txt
git commit -m "server : add kv tree module skeleton and block matching" -m "Assisted-by: opencode"
```

---

### Task 2: 信封缺省 + tip 锚点 + restore + llama adapter + 真机 tip 场景

**Files:**
- Modify: `tools/server/server-kv-tree.h` (加 `restore` 声明与 `store_anchor` 私有方法)
- Modify: `tools/server/server-kv-tree.cpp` (加 adapter / `store_anchor` / 改 `park` / 加 `restore`)
- Modify: `tests/test-t32-tree.cpp` (加 model 模式 + `scenario_tip`)

**Interfaces:**
- Consumes: Task 1 的全部类型与 `park`/`match`
- Produces: `kv_tree::restore`, `kv_tree::store_anchor`, `kv_tree_io_llama` 全部实现, harness 助手 `prefill`/`generate`/`run_baseline`/`run_tree_path`/`capture_partial`

- [ ] **Step 1: 头文件加声明**

在 `class kv_tree` 的 public 段 `park` 之后加:

```cpp
    // load the deepest anchor-covered prefix of tokens into the (cleared) sequence
    kv_tree_restore restore(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens);
```

在 private 段 `match` 之后加:

```cpp
    bool store_anchor(uint64_t blk_hash, llama_pos pos, int kind,
                      std::vector<uint8_t> && data_tgt, std::vector<uint8_t> && data_dft);
```

- [ ] **Step 2: `server-kv-tree.cpp` 加 adapter 与 `store_anchor`**

在 `kv_tree::kv_tree` 之前插入:

```cpp
kv_tree_io_llama::kv_tree_io_llama(llama_context * ctx, llama_seq_id seq_id) : ctx(ctx), seq_id(seq_id) {
}

bool kv_tree_io_llama::get_range(llama_pos p0, llama_pos p1, std::vector<uint8_t> & out) {
    const size_t size = llama_state_seq_get_size_range_ext(ctx, seq_id, p0, p1, 0);
    if (size == 0) {
        return false;
    }

    out.resize(size);

    return llama_state_seq_get_data_range_ext(ctx, out.data(), out.size(), seq_id, p0, p1, 0) == size;
}

bool kv_tree_io_llama::set_range(const uint8_t * data, size_t size, bool append) {
    return llama_state_seq_set_data_range_ext(ctx, data, size, seq_id, append, 0) == size;
}

bool kv_tree_io_llama::get_partial(std::vector<uint8_t> & out) {
    const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (size == 0) {
        return false;
    }

    out.resize(size);

    return llama_state_seq_get_data_ext(ctx, out.data(), out.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size;
}

bool kv_tree_io_llama::set_partial(const uint8_t * data, size_t size) {
    return llama_state_seq_set_data_ext(ctx, data, size, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size;
}

bool kv_tree_io_llama::seq_rm(llama_pos p0, llama_pos p1) {
    return llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1);
}

llama_pos kv_tree_io_llama::pos_max() {
    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
}
```

在 `make_room` 之后插入:

```cpp
bool kv_tree::store_anchor(uint64_t blk_hash, llama_pos pos, int kind,
                           std::vector<uint8_t> && data_tgt, std::vector<uint8_t> && data_dft) {
    if (pos <= 0 || blk_hash == 0) {
        return false;
    }

    const auto key = std::make_pair(blk_hash, pos);

    const auto it = anchors.find(key);
    if (it != anchors.end()) {
        it->second.refcount++;
        it->second.heat++;
        it->second.last_used = ++now;
        return true;
    }

    const size_t size = data_tgt.size() + data_dft.size();

    kv_tree_anchor & a = anchors[key];

    a.blk_hash  = blk_hash;
    a.pos       = pos;
    a.kind      = kind;
    a.refcount  = 1;
    a.last_used = ++now;
    a.bytes     = size;
    a.data_tgt  = std::move(data_tgt);
    a.data_dft  = std::move(data_dft);

    st.anchors_ram++;
    st.anchors_added++;
    st.bytes_ram += (int64_t) size;

    return true;
}
```

- [ ] **Step 3: 改 `park` (tip 锚点 + 预算一次算清)**

用下面完整函数替换 Task 1 的 `park`:

```cpp
bool kv_tree::park(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens,
                   const std::vector<kv_tree_anchor_in> & checkpoints) {
    (void) checkpoints;

    st.park_calls++;

    const llama_pos L = (llama_pos) tokens.size();

    if (L <= 0 || io_tgt.pos_max() != L - 1) {
        fprintf(stderr, "[kv-tree] park refused: sequence end %d, expected %d\n", io_tgt.pos_max(), L - 1);
        st.park_refused++;
        return false;
    }

    std::vector<uint64_t> h;
    chain_hashes(tokens, cfg.chunk, h);

    const uint64_t tip = h.back();

    if (seqs.count(tip) != 0) {
        seqs[tip].last_used = ++now;
        if (!checkpoints.empty()) {
            fprintf(stderr, "[kv-tree] park: sequence already stored, %zu checkpoint candidates not adopted\n", checkpoints.size());
            st.anchors_skipped += (int64_t) checkpoints.size();
        }
        st.park_ok++;
        return true;
    }

    const kv_tree_match m = match(tokens);

    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> blobs;
    std::vector<size_t> idxs;

    for (size_t i = m.n_full; i < h.size(); ++i) {
        if (blocks.count(h[i]) != 0) {
            continue;
        }

        const size_t a = i * (size_t) cfg.chunk;
        const size_t b = std::min(tokens.size(), a + (size_t) cfg.chunk);

        std::vector<uint8_t> blob;
        if (!io_tgt.get_range((llama_pos) a, (llama_pos) b, blob)) {
            fprintf(stderr, "[kv-tree] park refused: failed to read range [%zu, %zu)\n", a, b);
            st.park_refused++;
            return false;
        }

        blobs.emplace_back(h[i], std::move(blob));
        idxs.push_back(i);
    }

    // tip anchor: the io state is exactly at L
    std::vector<uint8_t> part_tgt;
    std::vector<uint8_t> part_dft;

    if (!io_tgt.get_partial(part_tgt)) {
        fprintf(stderr, "[kv-tree] park refused: failed to capture the tip state\n");
        st.park_refused++;
        return false;
    }
    if (io_dft != nullptr && !io_dft->get_partial(part_dft)) {
        fprintf(stderr, "[kv-tree] park refused: failed to capture the tip draft state\n");
        st.park_refused++;
        return false;
    }

    size_t need = part_tgt.size() + part_dft.size();
    for (const auto & p : blobs) {
        need += p.second.size();
    }

    if (!make_room(need)) {
        fprintf(stderr, "[kv-tree] park refused: no room for %zu bytes\n", need);
        st.park_refused++;
        return false;
    }

    for (size_t k = 0; k < blobs.size(); ++k) {
        const size_t i = idxs[k];
        const size_t a = i * (size_t) cfg.chunk;
        const size_t b = std::min(tokens.size(), a + (size_t) cfg.chunk);

        kv_tree_block & nb = blocks[blobs[k].first];

        nb.hash      = blobs[k].first;
        nb.pos0      = (llama_pos) a;
        nb.pos1      = (llama_pos) b;
        nb.refcount  = 1;
        nb.last_used = ++now;
        nb.tokens.assign(tokens.begin() + a, tokens.begin() + b);
        nb.data      = std::move(blobs[k].second);
        nb.bytes     = nb.data.size();

        blocks_at[nb.pos0].push_back(nb.hash);
        st.blocks_ram++;
        st.bytes_ram += (int64_t) nb.data.size();
    }

    for (const uint64_t hash : m.path) {
        kv_tree_block & b = blocks[hash];
        b.refcount++;
        b.heat++;
        b.last_used = ++now;
    }
    if (m.n_part > 0) {
        kv_tree_block & b = blocks[m.part_hash];
        b.refcount++;
        b.heat++;
        b.last_used = ++now;
    }

    if (!store_anchor(tip, L, KV_TREE_ANCHOR_TIP, std::move(part_tgt), std::move(part_dft))) {
        fprintf(stderr, "[kv-tree] park refused: cannot store the tip anchor at %d\n", L);
        st.park_refused++;
        return false;
    }

    kv_tree_seq & s = seqs[tip];
    s.chain     = h;
    s.len       = L;
    s.last_used = ++now;

    st.park_ok++;
    return true;
}
```

- [ ] **Step 4: 加 `restore`**

在 `park` 之后插入:

```cpp
kv_tree_restore kv_tree::restore(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens) {
    st.restore_calls++;

    io_tgt.seq_rm(-1, -1);
    if (io_dft != nullptr) {
        io_dft->seq_rm(-1, -1);
    }

    kv_tree_restore res;

    const kv_tree_match m = match(tokens);

    if (m.deep <= 0) {
        st.restore_miss++;
        return res;
    }

    std::vector<uint64_t> cand = m.path;
    if (m.n_part > 0) {
        cand.push_back(m.part_hash);
    }

    llama_pos C = -1;
    uint64_t  c_hash = 0;
    std::vector<llama_pos> path_anchors;

    for (const uint64_t hash : cand) {
        for (auto it = anchors.lower_bound(std::make_pair(hash, 0));
             it != anchors.end() && it->first.first == hash; ++it) {
            if (it->second.pos <= m.deep) {
                path_anchors.push_back(it->second.pos);
                if (it->second.pos > C) {
                    C = it->second.pos;
                    c_hash = hash;
                }
            }
        }
    }

    if (C < 0) {
        st.restore_miss++;
        return res;
    }

    for (const uint64_t hash : cand) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = true;
        }
    }
    for (auto & kv : anchors) {
        if (kv.second.pos <= C && std::find(cand.begin(), cand.end(), kv.first.first) != cand.end()) {
            kv.second.pinned = true;
        }
    }

    bool ok = true;

    // blocks covering [0, C): the last one may overshoot and is trimmed below
    for (const uint64_t hash : m.path) {
        kv_tree_block & b = blocks[hash];
        if (b.pos0 >= C) {
            break;
        }
        if (!io_tgt.set_range(b.data.data(), b.data.size(), b.pos0 != 0)) {
            fprintf(stderr, "[kv-tree] restore failed: cannot load block [%d, %d)\n", b.pos0, b.pos1);
            ok = false;
            break;
        }
        st.bytes_load += (int64_t) b.data.size();
    }

    if (ok && m.n_part > 0 && C > blocks[m.part_hash].pos0) {
        kv_tree_block & b = blocks[m.part_hash];
        if (!io_tgt.set_range(b.data.data(), b.data.size(), true)) {
            fprintf(stderr, "[kv-tree] restore failed: cannot load the partial block [%d, %d)\n", b.pos0, b.pos1);
            ok = false;
        }
    }

    if (ok) {
        io_tgt.seq_rm(C, -1);

        const auto it = anchors.find(std::make_pair(c_hash, C));
        if (it == anchors.end()) {
            ok = false;
        } else {
            const kv_tree_anchor & a = it->second;
            if (!io_tgt.set_partial(a.data_tgt.data(), a.data_tgt.size())) {
                fprintf(stderr, "[kv-tree] restore failed: cannot load the state at %d\n", C);
                ok = false;
            } else if (io_dft != nullptr && !a.data_dft.empty()) {
                if (!io_dft->set_partial(a.data_dft.data(), a.data_dft.size())) {
                    fprintf(stderr, "[kv-tree] restore failed: cannot load the draft state at %d\n", C);
                    ok = false;
                }
            }
        }
    }

    for (const uint64_t hash : cand) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = false;
        }
    }
    for (auto & kv : anchors) {
        if (kv.second.pos <= C && std::find(cand.begin(), cand.end(), kv.first.first) != cand.end()) {
            kv.second.pinned = false;
        }
    }

    if (!ok) {
        io_tgt.seq_rm(-1, -1);
        st.restore_miss++;
        return res;
    }

    kv_tree_anchor & ca = anchors.at(std::make_pair(c_hash, C));
    ca.heat++;
    ca.last_used = ++now;

    st.restore_hits++;
    st.tokens_reused += C;

    res.C      = C;
    res.heal   = m.deep > C ? m.deep : -1;
    res.anchors = path_anchors;

    return res;
}
```

- [ ] **Step 5: harness 加 model 模式与 `scenario_tip`**

在 `run_logic` 之后插入 (helper 与场景), 并改 `main` 支持 model 模式:

```cpp
static int prefill(llama_context * ctx, llama_seq_id seq, const std::vector<llama_token> & tokens, int from, int ubatch) {
    llama_batch batch = llama_batch_init(ubatch, 0, 1);

    for (int i = from; i < (int) tokens.size(); ) {
        batch.n_tokens = 0;

        const int n = std::min(ubatch, (int) tokens.size() - i);

        for (int j = 0; j < n; ++j) {
            common_batch_add(batch, tokens[i + j], i + j, { seq }, j == n - 1);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[t32-tree] decode failed at token %d\n", i);
            llama_batch_free(batch);
            return 1;
        }

        i += n;
    }

    llama_batch_free(batch);
    return 0;
}

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

static kv_tree_anchor_in capture_partial(llama_context * ctx, llama_seq_id seq, llama_pos pos) {
    kv_tree_anchor_in c;
    c.pos = pos;

    const size_t size = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (size == 0) {
        return c;
    }

    c.data_tgt.resize(size);

    const size_t n = llama_state_seq_get_data_ext(ctx, c.data_tgt.data(), c.data_tgt.size(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (n != size) {
        c.data_tgt.clear();
    }

    return c;
}

static std::vector<llama_token> run_baseline(llama_context * ctx, const std::vector<llama_token> & tokens, llama_token first, int n_gen) {
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    if (prefill(ctx, 0, tokens, 0, 512) != 0) {
        return {};
    }

    return generate(ctx, 0, first, (llama_pos) tokens.size(), n_gen);
}

struct path_result {
    std::vector<llama_token> gen;
    kv_tree_restore res;
};

static path_result run_tree_path(llama_context * ctx, kv_tree & tree, kv_tree_io & io,
                                 const std::vector<llama_token> & tokens, llama_token first, int n_gen) {
    path_result r;

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    r.res = tree.restore(io, nullptr, tokens);

    if (r.res.C >= 0 && prefill(ctx, 0, tokens, (int) r.res.C, 512) != 0) {
        return r;
    }

    r.gen = generate(ctx, 0, first, (llama_pos) tokens.size(), n_gen);
    return r;
}

static int scenario_tip(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] scenario: tip\n");

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tokens = make_tokens(1536, 0);

    if (prefill(ctx, 0, tokens, 0, 512) != 0) {
        return 1;
    }

    check(tree.park(io, nullptr, tokens, {}), "tip: park");
    check_eq(tree.stats().anchors_added, 1, "tip: one tip anchor");

    const auto base = run_baseline(ctx, tokens, 42, 8);
    check(!base.empty(), "tip: baseline generation");

    const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
    check_eq(r.res.C, 1536, "tip: restore point is the tip");
    check_eq(r.res.heal, -1, "tip: no heal needed");
    check(r.gen == base, "tip: tokens match the baseline");

    // the sequence now ends at L + 8, so parking it again must be refused
    check(!tree.park(io, nullptr, tokens, {}), "tip: park refused when the sequence ends past L");
    check_eq(tree.stats().park_refused, 1, "tip: refusal counted");

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}
```

`main` 改为 (完整替换):

```cpp
int main(int argc, char ** argv) {
    std::string mode = "logic";

    int    chunk       = 512;
    int    anchor_step = 32768;
    int    ram_mib     = 8192;
    int    disk_mib    = 65536;
    std::string disk;

    std::vector<char *> args;
    args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            continue;
        }
        if (arg == "--chunk" && i + 1 < argc) {
            chunk = atoi(argv[++i]);
            continue;
        }
        if (arg == "--anchor-step" && i + 1 < argc) {
            anchor_step = atoi(argv[++i]);
            continue;
        }
        if (arg == "--ram-mib" && i + 1 < argc) {
            ram_mib = atoi(argv[++i]);
            continue;
        }
        if (arg == "--disk" && i + 1 < argc) {
            disk = argv[++i];
            continue;
        }
        if (arg == "--disk-mib" && i + 1 < argc) {
            disk_mib = atoi(argv[++i]);
            continue;
        }

        args.push_back(argv[i]);
    }

    if (mode == "logic") {
        return run_logic();
    }

    common_params params;
    params.sampling.seed = 1234;
    params.n_parallel = 1;

    common_init();

    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        fprintf(stderr, "usage: %s -m model.gguf [common args] --mode <logic|model> [--chunk N] [--anchor-step N] [--ram-mib N] [--disk DIR] [--disk-mib N]\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();

    if (ctx == nullptr) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    kv_tree_config cfg;
    cfg.chunk       = chunk;
    cfg.anchor_step = anchor_step;
    cfg.ram_limit   = (size_t) ram_mib << 20;
    cfg.disk_limit  = (size_t) disk_mib << 20;
    cfg.disk_dir    = disk;

    if (mode == "model") {
        int ret = 0;
        ret |= scenario_tip(ctx, cfg);
        return ret;
    }

    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
```

- [ ] **Step 6: 构建并跑 tip 场景**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```
Expected: `scenario: tip`, `tip: park PASS`, `tip: one tip anchor PASS`, `tip: restore point is the tip PASS`, `tip: no heal needed PASS`, `tip: tokens match the baseline PASS`, exit 0. 另跑 `--mode logic` 确认无回归.

- [ ] **Step 7: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : add kv tree tip anchor and prefix restore" -m "Assisted-by: opencode"
```

---

### Task 3: 检查点收编 + 稀疏化 + 分叉场景

**Files:**
- Modify: `tools/server/server-kv-tree.h` (加 `containing_block`)
- Modify: `tools/server/server-kv-tree.cpp` (`containing_block` + `park` 收编逻辑)
- Modify: `tests/test-t32-tree.cpp` (`scenario_fork` / `scenario_sparsify` + model 模式串联)

**Interfaces:**
- Consumes: Task 2 的 `store_anchor`/`restore`/harness 助手
- Produces: `kv_tree::containing_block`; park 的收编语义 (排序/`anchor_step`/冲突留靠前/锚点 refcount 覆盖)

- [ ] **Step 1: 头文件加 `containing_block`**

private 段 `store_anchor` 之后加:

```cpp
    uint64_t containing_block(llama_pos pos, const std::vector<uint64_t> * chain) const;
```

- [ ] **Step 2: `server-kv-tree.cpp` 加 `containing_block` 与收编逻辑**

在 `store_anchor` 之后插入:

```cpp
uint64_t kv_tree::containing_block(llama_pos pos, const std::vector<uint64_t> * chain) const {
    if (pos <= 0) {
        return 0;
    }

    auto it = blocks_at.lower_bound(pos);
    if (it == blocks_at.begin()) {
        return 0;
    }
    --it;

    for (const uint64_t hash : it->second) {
        const auto b = blocks.find(hash);
        if (b == blocks.end() || b->second.pos0 >= pos || pos > b->second.pos1) {
            continue;
        }
        if (chain != nullptr && std::find(chain->begin(), chain->end(), hash) == chain->end()) {
            continue;
        }
        return hash;
    }

    return 0;
}
```

`park` 中, 在 `if (!store_anchor(tip, L, KV_TREE_ANCHOR_TIP, ...)) { ... }` 之后、`kv_tree_seq & s = seqs[tip];` 之前插入:

```cpp
    // adopt checkpoint candidates: sort by pos, greedy with anchor_step, conflict keeps the earlier one
    std::vector<kv_tree_anchor_in> cand = checkpoints;
    std::sort(cand.begin(), cand.end(), [](const kv_tree_anchor_in & a, const kv_tree_anchor_in & b) {
        return a.pos < b.pos;
    });

    std::vector<std::pair<uint64_t, llama_pos>> touched;
    touched.emplace_back(tip, L);

    llama_pos last_kept = -1;

    for (const kv_tree_anchor_in & c : cand) {
        if (c.pos <= 0 || c.pos > L || c.data_tgt.empty()) {
            st.anchors_skipped++;
            continue;
        }

        const uint64_t blk = containing_block(c.pos, &h);
        if (blk == 0) {
            fprintf(stderr, "[kv-tree] candidate at %d skipped: no containing block on the parked chain\n", c.pos);
            st.anchors_skipped++;
            continue;
        }

        llama_pos prev = -1;
        for (const auto & kv : anchors) {
            if (kv.second.pos < c.pos && kv.second.pos > prev) {
                prev = kv.second.pos;
            }
        }

        const llama_pos prev_kept = prev > last_kept ? prev : last_kept;

        if (prev_kept >= 0 && c.pos - prev_kept < cfg.anchor_step) {
            st.anchors_skipped++;
            continue;
        }

        if (!store_anchor(blk, c.pos, KV_TREE_ANCHOR_MESSAGE, std::vector<uint8_t>(c.data_tgt), std::vector<uint8_t>(c.data_dft))) {
            fprintf(stderr, "[kv-tree] candidate at %d skipped: anchor store failed\n", c.pos);
            st.anchors_skipped++;
            continue;
        }

        touched.emplace_back(blk, c.pos);
        last_kept = c.pos;
    }

    // anchors that already cover this sequence keep their refcount in sync
    for (auto & kv : anchors) {
        if (kv.second.pos > L) {
            continue;
        }

        bool in_chain = false;
        for (const uint64_t hash : h) {
            if (hash == kv.first.first) {
                in_chain = true;
                break;
            }
        }
        if (!in_chain) {
            continue;
        }

        bool was_touched = false;
        for (const auto & t : touched) {
            if (t.first == kv.first.first && t.second == kv.second.pos) {
                was_touched = true;
                break;
            }
        }
        if (!was_touched) {
            kv.second.refcount++;
            touched.emplace_back(kv.first.first, kv.second.pos);
        }
    }
```

注意: `park` 的签名里 `(void) checkpoints;` 要删掉.

- [ ] **Step 3: harness 加 `scenario_fork` 与 `scenario_sparsify`**

在 `scenario_tip` 之后插入:

```cpp
static std::vector<llama_token> make_fork_tokens(int n_shared, int n_tail, int salt) {
    std::vector<llama_token> t = make_tokens(n_shared, 0);
    for (int i = 0; i < n_tail; ++i) {
        t.push_back(10 + (i * 13 + salt) % 1000);
    }
    return t;
}

static int scenario_fork(llama_context * ctx, const kv_tree_config & cfg_in) {
    fprintf(stderr, "[t32-tree] scenario: fork\n");

    kv_tree_config cfg = cfg_in;
    cfg.anchor_step = 512;

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tok_a  = make_fork_tokens(1024, 1024, 0);
    const auto tok_b  = make_fork_tokens(1024, 1024, 1);
    const auto tok_a2 = make_fork_tokens(1024, 512,  2);

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, std::vector<llama_token>(tok_a.begin(), tok_a.begin() + 512), 0, 512);

    const kv_tree_anchor_in ck = capture_partial(ctx, 0, 512);

    prefill(ctx, 0, tok_a, 512, 512);
    check(tree.park(io, nullptr, tok_a, { ck }), "fork: park A");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, tok_b, 0, 512);
    check(tree.park(io, nullptr, tok_b, {}), "fork: park B");

    {
        const auto base = run_baseline(ctx, tok_a, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tok_a, 42, 8);
        check_eq(r.res.C, 2048, "fork: A restores at the tip");
        check(r.gen == base, "fork: A tokens match the baseline");
    }
    {
        const auto base = run_baseline(ctx, tok_b, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tok_b, 42, 8);
        check_eq(r.res.C, 2048, "fork: B restores at the tip");
        check(r.gen == base, "fork: B tokens match the baseline");
    }
    {
        const auto base = run_baseline(ctx, tok_a2, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tok_a2, 42, 8);
        check_eq(r.res.C, 512, "fork: A' restores at the deepest usable anchor");
        check_eq(r.res.heal, 1024, "fork: A' asks for a heal at the fork point");
        check(r.gen == base, "fork: A' tokens match the baseline");
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

static int scenario_sparsify(llama_context * ctx, const kv_tree_config & cfg_in) {
    fprintf(stderr, "[t32-tree] scenario: sparsify\n");

    kv_tree_config cfg = cfg_in;
    cfg.anchor_step = 1024;

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tokens = make_tokens(2048, 0);

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 512), 0, 512);

    const kv_tree_anchor_in ck1 = capture_partial(ctx, 0, 512);

    prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 1024), 512, 512);

    const kv_tree_anchor_in ck2 = capture_partial(ctx, 0, 1024);

    prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 1536), 1024, 512);

    const kv_tree_anchor_in ck3 = capture_partial(ctx, 0, 1536);

    prefill(ctx, 0, tokens, 1536, 512);

    check(tree.park(io, nullptr, tokens, { ck1, ck2, ck3 }), "sparsify: park");

    {
        const std::vector<llama_token> short1(tokens.begin(), tokens.begin() + 1024);
        const auto base = run_baseline(ctx, short1, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, short1, 42, 8);
        check_eq(r.res.C, 512, "sparsify: [0, 1024) restores at 512");
        check(r.gen == base, "sparsify: [0, 1024) tokens match");
    }
    {
        const std::vector<llama_token> short2(tokens.begin(), tokens.begin() + 1536);
        const auto base = run_baseline(ctx, short2, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, short2, 42, 8);
        check_eq(r.res.C, 1536, "sparsify: [0, 1536) restores at 1536");
        check(r.gen == base, "sparsify: [0, 1536) tokens match");
    }
    {
        const auto base = run_baseline(ctx, tokens, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
        check_eq(r.res.C, 2048, "sparsify: full length restores at the tip");
        check(r.gen == base, "sparsify: full length tokens match");
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}
```

`main` 的 model 分支改为:

```cpp
    if (mode == "model") {
        int ret = 0;
        ret |= scenario_tip(ctx, cfg);
        ret |= scenario_fork(ctx, cfg);
        ret |= scenario_sparsify(ctx, cfg);
        return ret;
    }
```

- [ ] **Step 4: 构建并跑 model 模式**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```
Expected: tip / fork / sparsify 三场景全 PASS, exit 0. 若 fork 的 A' 出现 token 不匹配: 先看 logits 是否近并列 (打印 tokens 差异), 近并列属已知 T24 现象, 记录后继续; 否则是 bug.

- [ ] **Step 5: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : adopt checkpoint anchors in the kv tree" -m "Assisted-by: opencode"
```

---

### Task 4: 自愈捕获 + 分叉提升/剪枝

**Files:**
- Modify: `tools/server/server-kv-tree.h` (加 `capture_anchor` / `promote_prune` / `remove_anchor` / `last_promote` 成员)
- Modify: `tools/server/server-kv-tree.cpp` (`capture_anchor` / `promote_prune` / `remove_anchor` + `anchor_bytes` 助手)
- Modify: `tests/test-t32-tree.cpp` (`scenario_fork` 的 A' 块替换为带自愈的版本)

**Interfaces:**
- Consumes: Task 2/3 的 `containing_block`/`store_anchor`/`restore`
- Produces: `kv_tree::capture_anchor`; 自愈后同一分叉点第二次恢复免费

**设计注 (审阅重点):** spec §4 的 "删除 (前一锚点, P) 之间无分叉价值的中间锚点" 在本计划里实现为**保守版**: 维护 `last_promote` (上次提升位置), 提升 P 时删除 `(last_promote, P)` 间 `refcount<2 && kind != TIP && !pinned` 的锚点; 首次提升 (last_promote < 0) 只记录不删, 避免一次提升就清掉所有单序列检查点.

- [ ] **Step 1: 头文件加声明与成员**

public 段 `restore` 之后加:

```cpp
    // capture the state at pos (the io state must be exactly at pos) as a fork anchor;
    // tokens is the sequence content, used to attach the anchor to the right chain block
    bool capture_anchor(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens, llama_pos pos);
```

private 段加:

```cpp
    void promote_prune(const std::vector<uint64_t> & chain, llama_pos pos);

    void remove_anchor(std::map<std::pair<uint64_t, llama_pos>, kv_tree_anchor>::iterator it);
```

并在 `kv_tree_anchor` 结构里加 `bool transient = false;` (`remove_anchor` 用到; Task 5 只给 `kv_tree_block` 加). 不引入 `last_promote` 成员: 前一锚点按捕获链推导 (最终审查修复).

- [ ] **Step 2: `server-kv-tree.cpp` 加实现**

文件头 include 区加 `#include <filesystem>`; 在 `store_anchor` 之后插入:

```cpp
static size_t anchor_bytes(const kv_tree_anchor & a) {
    return a.bytes;
}

void kv_tree::remove_anchor(std::map<std::pair<uint64_t, llama_pos>, kv_tree_anchor>::iterator it) {
    kv_tree_anchor & a = it->second;

    if (a.transient) {
        st.bytes_ram -= (int64_t) a.bytes;
        st.anchors_ram--;
    }

    if (a.on_disk) {
        std::error_code ec;
        std::filesystem::remove(a.path, ec);
        st.bytes_disk -= (int64_t) a.bytes;
        st.anchors_disk--;
    } else {
        st.bytes_ram -= (int64_t) a.bytes;
        st.anchors_ram--;
    }

    anchors.erase(it);
}

bool kv_tree::capture_anchor(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens, llama_pos pos) {
    if (pos <= 0) {
        return false;
    }
    if (io_tgt.pos_max() != pos - 1) {
        fprintf(stderr, "[kv-tree] capture refused at %d: sequence end %d\n", pos, io_tgt.pos_max());
        return false;
    }

    std::vector<uint64_t> chain;
    chain_hashes(tokens, cfg.chunk, chain);

    const uint64_t blk = containing_block(pos, &chain);
    if (blk == 0) {
        fprintf(stderr, "[kv-tree] capture refused at %d: no chain block contains this position\n", pos);
        return false;
    }

    const auto key = std::make_pair(blk, pos);

    const auto it = anchors.find(key);
    if (it != anchors.end()) {
        it->second.heat++;
        it->second.last_used = ++now;
        return true;
    }

    llama_pos prev = -1;
    for (const auto & kv : anchors) {
        if (kv.second.pos < pos && kv.second.pos > prev) {
            prev = kv.second.pos;
        }
    }

    if (prev >= 0 && pos - prev < cfg.anchor_step) {
        st.anchors_skipped++;
        return true;
    }

    std::vector<uint8_t> tgt;
    std::vector<uint8_t> dft;

    if (!io_tgt.get_partial(tgt)) {
        fprintf(stderr, "[kv-tree] capture refused at %d: failed to capture the state\n", pos);
        st.anchors_skipped++;
        return false;
    }
    if (io_dft != nullptr) {
        io_dft->get_partial(dft);
    }

    if (!store_anchor(blk, pos, KV_TREE_ANCHOR_ONDEMAND, std::move(tgt), std::move(dft))) {
        return false;
    }

    promote_prune(pos);

    return true;
}

void kv_tree::promote_prune(const std::vector<uint64_t> & chain, llama_pos pos) {
    // previous anchor on this chain
    llama_pos prev = -1;
    for (const auto & kv : anchors) {
        const kv_tree_anchor & a = kv.second;
        if (a.pos >= pos || a.pos <= prev) {
            continue;
        }
        if (std::find(chain.begin(), chain.end(), a.blk_hash) != chain.end()) {
            prev = a.pos;
        }
    }

    if (prev < 0) {
        return;
    }

    for (auto it = anchors.begin(); it != anchors.end(); ) {
        kv_tree_anchor & a = it->second;

        if (a.pos > prev && a.pos < pos && a.kind != KV_TREE_ANCHOR_TIP && a.refcount < 2 && !a.pinned &&
            std::find(chain.begin(), chain.end(), a.blk_hash) != chain.end()) {
            remove_anchor(it++);
        } else {
            ++it;
        }
    }
}
```

- [ ] **Step 3: 替换 `scenario_fork` 的 A' 块**

用下面替换 Task 3 里 `// A' forks at 1024` 的那个 `{ ... }` 块:

```cpp
    {
        const auto base = run_baseline(ctx, tok_a2, 42, 8);

        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

        const auto r = tree.restore(io, nullptr, tok_a2);
        check_eq(r.C, 512, "fork: A' restores at the deepest usable anchor");
        check_eq(r.heal, 1024, "fork: A' asks for a heal at the fork point");

        if (r.C >= 0) {
            prefill(ctx, 0, std::vector<llama_token>(tok_a2.begin(), tok_a2.begin() + 1024), (int) r.C, 512);
        }

        check(tree.capture_anchor(io, nullptr, tok_a2, 1024), "fork: heal capture at the fork point");

        prefill(ctx, 0, tok_a2, 1024, 512);

        const auto gen = generate(ctx, 0, 42, (llama_pos) tok_a2.size(), 8);
        check(gen == base, "fork: A' tokens match the baseline");

        const auto r2 = run_tree_path(ctx, tree, io, tok_a2, 42, 8);
        check_eq(r2.res.C, 1024, "heal: A' restores at the self-healed anchor");
        check_eq(r2.res.heal, -1, "heal: no second heal needed");
        check(r2.gen == base, "heal: tokens still match the baseline");
    }
```

- [ ] **Step 4: 构建并跑**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```
Expected: tip / fork / sparsify 全 PASS, 其中 `heal:` 两条必须 PASS (自愈锚点生效, 第二次恢复无重放).

- [ ] **Step 5: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : self-heal kv tree anchors at fork points" -m "Assisted-by: opencode"
```

---

### Task 5: SSD 层 (信封/文件/移动语义) + 预算降级

**Files:**
- Modify: `tools/server/server-kv-tree.h` (加 payload 存取/落盘/结算声明; block/anchor 加 `transient`)
- Modify: `tools/server/server-kv-tree.cpp` (信封 + 文件 I/O + `load_payload`/`demote_*`/`settle` + `enforce_budget` 降级 + `restore` 接入)
- Modify: `tests/test-t32-tree.cpp` (`scenario_ssd`)

**Interfaces:**
- Consumes: Task 2-4 全部
- Produces: SSD 单份权威移动语义; `st.bytes_disk`/`blocks_disk`/`anchors_disk`/`disk_errors`/`bytes_store` 计数

- [ ] **Step 1: 头文件**

`kv_tree_block` 加一个字段 (放在 `pinned` 之后; `kv_tree_anchor` 已在 Task 4 加过):

```cpp
    bool        transient = false;
```

private 段: 把 `bool make_room(size_t need_ram);` 换成

```cpp
    bool enforce_budget();

    void park_rollback(const std::vector<std::pair<uint64_t, std::vector<uint8_t>>> & blobs,
                       const kv_tree_match & m,
                       const std::vector<std::pair<uint64_t, llama_pos>> & touched,
                       uint64_t tip);

    void remove_block(std::unordered_map<uint64_t, kv_tree_block>::iterator it);
```

再加:

```cpp
    bool load_payload(kv_tree_block & b);
    bool load_payload(kv_tree_anchor & a);
    bool demote_block(kv_tree_block & b);
    bool demote_anchor(kv_tree_anchor & a);
    void settle();

    bool write_disk(const std::string & path, const std::vector<uint8_t> & buf);
    bool read_disk(const std::string & path, std::vector<uint8_t> & out);
    std::string block_path(uint64_t hash) const;
    std::string anchor_path(uint64_t blk_hash, llama_pos pos) const;
```

- [ ] **Step 2: `server-kv-tree.cpp` 加信封与文件 I/O**

文件头 include 区加 `#include <filesystem>`; 在 `chain_hashes` 之后插入:

```cpp
static const uint32_t T32_ENV_MAGIC_BLOCK  = 0x42323354u;  // "T32B"
static const uint32_t T32_ENV_MAGIC_ANCHOR = 0x41323354u;  // "T32A"
static const uint32_t T32_ENV_VERSION      = 1;

static void put_u32(std::vector<uint8_t> & v, uint32_t x) {
    uint8_t b[4];
    memcpy(b, &x, 4);
    v.insert(v.end(), b, b + 4);
}

static void put_i32(std::vector<uint8_t> & v, int32_t x) {
    put_u32(v, (uint32_t) x);
}

static void put_u64(std::vector<uint8_t> & v, uint64_t x) {
    uint8_t b[8];
    memcpy(b, &x, 8);
    v.insert(v.end(), b, b + 8);
}

struct kv_env {
    uint32_t magic   = 0;
    uint64_t hash    = 0;
    llama_pos p0     = 0;
    llama_pos p1     = 0;
    uint32_t n_tgt   = 0;
    uint32_t n_dft   = 0;
    uint32_t n_spec  = 0;
    const uint8_t * tgt  = nullptr;
    const uint8_t * dft  = nullptr;
    const uint8_t * spec = nullptr;
};

static std::vector<uint8_t> envelope_make(uint32_t magic, uint64_t hash, llama_pos p0, llama_pos p1,
                                          const std::vector<uint8_t> & tgt,
                                          const std::vector<uint8_t> & dft,
                                          const std::vector<uint8_t> & spec) {
    uint64_t ph = 0;
    if (!tgt.empty()) {
        ph = XXH64(tgt.data(), tgt.size(), 0);
    }
    if (!dft.empty()) {
        ph = XXH64(dft.data(), dft.size(), ph);
    }
    if (!spec.empty()) {
        ph = XXH64(spec.data(), spec.size(), ph);
    }

    std::vector<uint8_t> v;
    v.reserve(44 + tgt.size() + dft.size() + spec.size());

    put_u32(v, magic);
    put_u32(v, T32_ENV_VERSION);
    put_u64(v, hash);
    put_i32(v, (int32_t) p0);
    put_i32(v, (int32_t) p1);
    put_u32(v, (uint32_t) tgt.size());
    put_u32(v, (uint32_t) dft.size());
    put_u32(v, (uint32_t) spec.size());
    put_u64(v, ph);
    v.insert(v.end(), tgt.begin(), tgt.end());
    v.insert(v.end(), dft.begin(), dft.end());
    v.insert(v.end(), spec.begin(), spec.end());

    return v;
}

static bool envelope_parse(const std::vector<uint8_t> & buf, kv_env & e) {
    if (buf.size() < 44) {
        return false;
    }

    const uint8_t * p = buf.data();
    size_t off = 0;

    uint32_t version = 0;
    uint64_t ph = 0;

    memcpy(&e.magic,   p + off, 4); off += 4;
    memcpy(&version,   p + off, 4); off += 4;
    memcpy(&e.hash,    p + off, 8); off += 8;
    memcpy(&e.p0,      p + off, 4); off += 4;
    memcpy(&e.p1,      p + off, 4); off += 4;
    memcpy(&e.n_tgt,   p + off, 4); off += 4;
    memcpy(&e.n_dft,   p + off, 4); off += 4;
    memcpy(&e.n_spec,  p + off, 4); off += 4;
    memcpy(&ph,        p + off, 8); off += 8;

    if (version != T32_ENV_VERSION) {
        return false;
    }
    if (off + e.n_tgt + e.n_dft + e.n_spec != buf.size()) {
        return false;
    }

    e.tgt  = p + off; off += e.n_tgt;
    e.dft  = p + off; off += e.n_dft;
    e.spec = p + off;

    uint64_t check = 0;
    if (e.n_tgt > 0) {
        check = XXH64(e.tgt, e.n_tgt, 0);
    }
    if (e.n_dft > 0) {
        check = XXH64(e.dft, e.n_dft, check);
    }
    if (e.n_spec > 0) {
        check = XXH64(e.spec, e.n_spec, check);
    }

    return check == ph;
}
```

在 `containing_block` 之后插入:

```cpp
std::string kv_tree::block_path(uint64_t hash) const {
    char buf[32];
    snprintf(buf, sizeof(buf), "%016" PRIx64 ".bin", hash);
    return cfg.disk_dir + "/blocks/" + buf;
}

std::string kv_tree::anchor_path(uint64_t blk_hash, llama_pos pos) const {
    char buf[48];
    snprintf(buf, sizeof(buf), "%016" PRIx64 "_%d.bin", blk_hash, (int) pos);
    return cfg.disk_dir + "/anchors/" + buf;
}

bool kv_tree::write_disk(const std::string & path, const std::vector<uint8_t> & buf) {
    const std::string tmp = path + ".tmp";

    std::FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }

    const size_t n = fwrite(buf.data(), 1, buf.size(), f);
    const bool ok = fclose(f) == 0 && n == buf.size();

    if (!ok) {
        remove(tmp.c_str());
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path, ec);
    }
    if (ec) {
        remove(tmp.c_str());
        return false;
    }

    return true;
}

bool kv_tree::read_disk(const std::string & path, std::vector<uint8_t> & out) {
    std::FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }

    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return false;
    }

    out.resize((size_t) size);

    const size_t n = fread(out.data(), 1, out.size(), f);
    fclose(f);

    return n == out.size();
}

bool kv_tree::demote_block(kv_tree_block & b) {
    if (cfg.disk_dir.empty() || b.on_disk || b.data.empty()) {
        return false;
    }
    if (st.bytes_disk + (int64_t) b.data.size() > (int64_t) cfg.disk_limit) {
        return false;
    }

    const std::string path = block_path(b.hash);

    if (!write_disk(path, envelope_make(T32_ENV_MAGIC_BLOCK, b.hash, b.pos0, b.pos1, b.data, {}, {}))) {
        fprintf(stderr, "[kv-tree] failed to write block %016" PRIx64 " to disk\n", b.hash);
        st.disk_errors++;
        return false;
    }

    st.bytes_ram  -= (int64_t) b.data.size();
    st.bytes_disk += (int64_t) b.data.size();
    st.blocks_ram--;
    st.blocks_disk++;
    st.bytes_store += (int64_t) b.data.size();

    b.on_disk = true;
    b.path    = path;
    std::vector<uint8_t>().swap(b.data);

    return true;
}

bool kv_tree::demote_anchor(kv_tree_anchor & a) {
    if (cfg.disk_dir.empty() || a.on_disk || a.data_tgt.empty()) {
        return false;
    }

    const size_t size = anchor_bytes(a);
    if (st.bytes_disk + (int64_t) size > (int64_t) cfg.disk_limit) {
        return false;
    }

    const std::string path = anchor_path(a.blk_hash, a.pos);

    if (!write_disk(path, envelope_make(T32_ENV_MAGIC_ANCHOR, a.blk_hash, a.pos, a.pos, a.data_tgt, a.data_dft, a.data_spec))) {
        fprintf(stderr, "[kv-tree] failed to write anchor %016" PRIx64 "@%d to disk\n", a.blk_hash, a.pos);
        st.disk_errors++;
        return false;
    }

    st.bytes_ram  -= (int64_t) size;
    st.bytes_disk += (int64_t) size;
    st.anchors_ram--;
    st.anchors_disk++;
    st.bytes_store += (int64_t) size;

    a.on_disk = true;
    a.path    = path;
    std::vector<uint8_t>().swap(a.data_tgt);
    std::vector<uint8_t>().swap(a.data_dft);
    std::vector<uint8_t>().swap(a.data_spec);

    return true;
}

bool kv_tree::load_payload(kv_tree_block & b) {
    if (!b.on_disk) {
        return true;
    }

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

    b.data.assign(e.tgt, e.tgt + e.n_tgt);
    b.transient = true;

    st.bytes_ram += (int64_t) b.data.size();
    st.blocks_ram++;

    return true;
}

bool kv_tree::load_payload(kv_tree_anchor & a) {
    if (!a.on_disk) {
        return true;
    }

    std::vector<uint8_t> buf;
    if (!read_disk(a.path, buf)) {
        fprintf(stderr, "[kv-tree] failed to read anchor %016" PRIx64 "@%d from disk\n", a.blk_hash, a.pos);
        st.disk_errors++;
        return false;
    }

    kv_env e;
    if (!envelope_parse(buf, e) || e.magic != T32_ENV_MAGIC_ANCHOR || e.hash != a.blk_hash || e.p0 != a.pos || e.p1 != a.pos) {
        fprintf(stderr, "[kv-tree] anchor %016" PRIx64 "@%d payload check failed, dropping it\n", a.blk_hash, a.pos);
        st.disk_errors++;
        return false;
    }

    a.data_tgt.assign(e.tgt, e.tgt + e.n_tgt);
    a.data_dft.assign(e.dft, e.dft + e.n_dft);
    a.data_spec.assign(e.spec, e.spec + e.n_spec);
    a.transient = true;

    st.bytes_ram += (int64_t) anchor_bytes(a);
    st.anchors_ram++;

    return true;
}

void kv_tree::settle() {
    for (auto & p : blocks) {
        kv_tree_block & b = p.second;
        if (!b.transient) {
            continue;
        }

        b.transient = false;

        if ((size_t) st.bytes_ram <= cfg.ram_limit) {
            std::error_code ec;
            std::filesystem::remove(b.path, ec);

            st.bytes_disk -= (int64_t) b.data.size();
            st.blocks_disk--;
            b.on_disk = false;
            b.path.clear();
        } else {
            st.bytes_ram -= (int64_t) b.data.size();
            st.blocks_ram--;
            std::vector<uint8_t>().swap(b.data);
        }
    }

    for (auto & p : anchors) {
        kv_tree_anchor & a = p.second;
        if (!a.transient) {
            continue;
        }

        a.transient = false;

        const size_t size = anchor_bytes(a);

        if ((size_t) st.bytes_ram <= cfg.ram_limit) {
            std::error_code ec;
            std::filesystem::remove(a.path, ec);

            st.bytes_disk -= (int64_t) size;
            st.anchors_disk--;
            a.on_disk = false;
            a.path.clear();
        } else {
            st.bytes_ram -= (int64_t) size;
            st.anchors_ram--;
            std::vector<uint8_t>().swap(a.data_tgt);
            std::vector<uint8_t>().swap(a.data_dft);
            std::vector<uint8_t>().swap(a.data_spec);
        }
    }
}
```

- [ ] **Step 3: 预算模型改为 commit-then-enforce (pin + 回滚), 接入 restore**

**(3a)** cpp: 删掉 `make_room` 的定义, 在 `settle` 之后插入 (`remove_block` 在 Task 6 复用):

```cpp
bool kv_tree::enforce_budget() {
    for (auto & p : blocks) {
        if ((size_t) st.bytes_ram <= cfg.ram_limit) {
            break;
        }

        kv_tree_block & b = p.second;
        if (b.on_disk || b.data.empty()) {
            continue;
        }

        demote_block(b);
    }

    for (auto & p : anchors) {
        if ((size_t) st.bytes_ram <= cfg.ram_limit) {
            break;
        }

        kv_tree_anchor & a = p.second;
        if (a.on_disk || a.data_tgt.empty()) {
            continue;
        }

        demote_anchor(a);
    }

    return (size_t) st.bytes_ram <= cfg.ram_limit;
}

void kv_tree::remove_block(std::unordered_map<uint64_t, kv_tree_block>::iterator it) {
    kv_tree_block & b = it->second;

    // anchors whose containing block is gone can never be used again
    for (auto a = anchors.begin(); a != anchors.end(); ) {
        if (a->first.first == b.hash) {
            remove_anchor(a++);
        } else {
            ++a;
        }
    }

    if (b.transient) {
        st.bytes_ram -= (int64_t) b.bytes;
        st.blocks_ram--;
    }

    if (b.on_disk) {
        std::error_code ec;
        std::filesystem::remove(b.path, ec);
        st.bytes_disk -= (int64_t) b.bytes;
        st.blocks_disk--;
    } else {
        st.bytes_ram -= (int64_t) b.bytes;
        st.blocks_ram--;
    }

    auto & v = blocks_at[b.pos0];
    v.erase(std::remove(v.begin(), v.end(), b.hash), v.end());
    if (v.empty()) {
        blocks_at.erase(b.pos0);
    }

    blocks.erase(it);
    st.evicted_blocks++;
}

void kv_tree::park_rollback(const std::vector<std::pair<uint64_t, std::vector<uint8_t>>> & blobs,
                            const kv_tree_match & m,
                            const std::vector<std::pair<uint64_t, llama_pos>> & touched,
                            uint64_t tip) {
    for (const auto & p : blobs) {
        auto it = blocks.find(p.first);
        if (it != blocks.end()) {
            remove_block(it);
        }
    }

    for (const auto & key : touched) {
        auto it = anchors.find(key);
        if (it == anchors.end()) {
            continue;
        }
        if (it->second.refcount <= 1) {
            remove_anchor(it);
        } else {
            it->second.refcount--;
        }
    }

    for (const uint64_t hash : m.path) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.refcount--;
        }
    }
    if (m.n_part > 0) {
        auto it = blocks.find(m.part_hash);
        if (it != blocks.end()) {
            it->second.refcount--;
        }
    }

    seqs.erase(tip);
}
```

**(3b)** `park`: 删掉预检查块

```cpp
    size_t need = part_tgt.size() + part_dft.size();
    for (const auto & p : blobs) {
        need += p.second.size();
    }

    if (!make_room(need)) {
        fprintf(stderr, "[kv-tree] park refused: no room for %zu bytes\n", need);
        st.park_refused++;
        return false;
    }
```

并把结尾 (从 `st.park_ok++;` 到最后) 换成:

```cpp
    // pin the new payloads and the new sequence while the budget is enforced, then roll back if it cannot be met
    std::vector<uint64_t> pinned_blocks;
    for (const auto & p : blobs) {
        pinned_blocks.push_back(p.first);
    }
    for (const uint64_t hash : pinned_blocks) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = true;
        }
    }
    for (const auto & key : touched) {
        auto it = anchors.find(key);
        if (it != anchors.end()) {
            it->second.pinned = true;
        }
    }

    s.pinned = true;

    const bool fits = enforce_budget();

    for (const uint64_t hash : pinned_blocks) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = false;
        }
    }
    for (const auto & key : touched) {
        auto it = anchors.find(key);
        if (it != anchors.end()) {
            it->second.pinned = false;
        }
    }

    s.pinned = false;

    if (!fits) {
        fprintf(stderr, "[kv-tree] park refused: the budget cannot hold the sequence\n");
        park_rollback(blobs, m, touched, tip);
        st.park_refused++;
        return false;
    }

    st.park_ok++;
    return true;
```

**(3c)** `capture_anchor`: 把结尾

```cpp
    if (!store_anchor(blk, pos, KV_TREE_ANCHOR_ONDEMAND, std::move(tgt), std::move(dft))) {
        return false;
    }

    promote_prune(pos);

    return true;
```

换成

```cpp
    if (!store_anchor(blk, pos, KV_TREE_ANCHOR_ONDEMAND, std::move(tgt), std::move(dft))) {
        return false;
    }

    if (!enforce_budget() || anchors.count(key) == 0) {
        st.anchors_skipped++;
        return false;
    }

    promote_prune(chain, pos);

    return true;
```

**(3d)** `restore` 里三处替换:

1. 整块装载: `if (!io_tgt.set_range(b.data.data(), b.data.size(), b.pos0 != 0)) {` 换成 `if (!load_payload(b) || !io_tgt.set_range(b.data.data(), b.data.size(), b.pos0 != 0)) {`
2. 部分块装载: `if (!io_tgt.set_range(b.data.data(), b.data.size(), true)) {` 换成 `if (!load_payload(b) || !io_tgt.set_range(b.data.data(), b.data.size(), true)) {`
3. 锚点装载: `const kv_tree_anchor & a = it->second;` 换成 `kv_tree_anchor & a = it->second;`, 且 `if (!io_tgt.set_partial(a.data_tgt.data(), a.data_tgt.size())) {` 换成 `if (!load_payload(a) || !io_tgt.set_partial(a.data_tgt.data(), a.data_tgt.size())) {`
4. 在两组 unpin 循环之后、`if (!ok) {` 之前插入一行:

```cpp
    settle();
```


- [ ] **Step 4: harness 加 `scenario_ssd`**

文件头 include 区加 `#include <filesystem>`; 在 `scenario_sparsify` 之后插入:

```cpp
static int scenario_ssd(llama_context * ctx, const kv_tree_config & cfg_in) {
    fprintf(stderr, "[t32-tree] scenario: ssd\n");

    kv_tree_config cfg = cfg_in;
    cfg.ram_limit  = 64 << 10;    // force everything to disk
    cfg.disk_limit = 64 << 20;
    cfg.disk_dir   = (std::filesystem::temp_directory_path() / "t32-tree-test").string();

    std::error_code ec;
    std::filesystem::remove_all(cfg.disk_dir, ec);
    std::filesystem::create_directories(cfg.disk_dir + "/blocks", ec);
    std::filesystem::create_directories(cfg.disk_dir + "/anchors", ec);

    kv_tree_io_llama io(ctx, 0);

    const auto tokens = make_tokens(1536, 0);

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, tokens, 0, 512);

    // size the disk budget to the model: pure-attention PARTIAL_ONLY is the whole KV
    const size_t seq_bytes    = llama_state_seq_get_size_range_ext(ctx, 0, 0, 1536, 0);
    const size_t anchor_bytes = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    cfg.disk_limit = 4 * (seq_bytes + anchor_bytes);

    kv_tree tree(cfg);

    check(tree.park(io, nullptr, tokens, {}), "ssd: park");
    check(tree.stats().blocks_disk > 0, "ssd: blocks demoted to disk");
    check(tree.stats().anchors_disk > 0, "ssd: tip anchor demoted to disk");
    check_eq(tree.stats().blocks_ram, 0, "ssd: no blocks left in ram");

    {
        const auto base = run_baseline(ctx, tokens, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
        check_eq(r.res.C, 1536, "ssd: restore point is the tip");
        check(r.gen == base, "ssd: tokens match the baseline");
    }

    for (const auto & e : std::filesystem::directory_iterator(cfg.disk_dir + "/blocks")) {
        std::filesystem::remove(e.path());
    }

    const int64_t errors_before = tree.stats().disk_errors;

    {
        const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
        check_eq(r.res.C, -1, "ssd: restore misses when block files are gone");
        check(tree.stats().disk_errors > errors_before, "ssd: disk error counted");
    }

    std::filesystem::remove_all(cfg.disk_dir, ec);

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}
```

`main` 的 model 分支加 `ret |= scenario_ssd(ctx, cfg);`.

- [ ] **Step 5: 构建并跑**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```
Expected: tip / fork / sparsify / ssd 全 PASS, 其中 `ssd: restore point is the tip` 与 `ssd: tokens match the baseline` 必须 PASS (SSD 往返逐位一致).

- [ ] **Step 6: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : add ssd tier and single-copy payload moves to the kv tree" -m "Assisted-by: opencode"
```

---

### Task 6: 淘汰次序 + pin + 拒绝 + 降级优先

**Files:**
- Modify: `tools/server/server-kv-tree.h` (加淘汰/删除方法声明)
- Modify: `tools/server/server-kv-tree.cpp` (`has_successor` / `remove_block` / `remove_seq` / `demote_one` / `evict_*` + 完整 `enforce_budget`)
- Modify: `tests/test-t32-tree.cpp` (`run_logic_evict`)

**Interfaces:**
- Consumes: Task 5 的 `enforce_budget`/`demote_*`/`remove_anchor`
- Produces: 淘汰次序 (锚点 -> 叶块 -> 整条叶序列 -> 拒绝), pin 纪律, `evicted_*`/`evict_refused` 计数

- [ ] **Step 1: 头文件加声明**

private 段加:

```cpp
    bool has_successor(const kv_tree_block & b) const;

    bool demote_one();

    bool evict_anchor_one();
    bool evict_block_one();
    bool evict_seq_one();

    void remove_seq(std::unordered_map<uint64_t, kv_tree_seq>::iterator it);
```

- [ ] **Step 2: `server-kv-tree.cpp` 加实现**

在 `settle` 之后插入:

```cpp
bool kv_tree::has_successor(const kv_tree_block & b) const {
    const auto it = blocks_at.find(b.pos1);
    return it != blocks_at.end() && !it->second.empty();
}

void kv_tree::remove_seq(std::unordered_map<uint64_t, kv_tree_seq>::iterator it) {
    kv_tree_seq & s = it->second;

    for (auto & kv : anchors) {
        bool in_chain = false;
        for (const uint64_t hash : s.chain) {
            if (hash == kv.first.first) {
                in_chain = true;
                break;
            }
        }
        if (in_chain && kv.second.refcount > 0) {
            kv.second.refcount--;
        }
    }

    for (auto rit = s.chain.rbegin(); rit != s.chain.rend(); ++rit) {
        const auto b = blocks.find(*rit);
        if (b == blocks.end()) {
            continue;
        }
        if (--b->second.refcount <= 0) {
            remove_block(b);
        }
    }

    seqs.erase(it);
    st.evicted_seqs++;
}

bool kv_tree::demote_one() {
    auto best = blocks.end();
    int64_t best_score = 0;

    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        const kv_tree_block & b = it->second;
        if (b.on_disk || b.data.empty()) {
            continue;
        }

        int64_t score = b.refcount * 1000000 + b.heat * 1000 + b.last_used;
        if (!has_successor(b)) {
            score -= 100000000;   // leaves go to disk first
        }

        if (best == blocks.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best != blocks.end() && demote_block(best->second)) {
        return true;
    }

    auto best_a = anchors.end();
    int64_t best_a_score = 0;

    for (auto it = anchors.begin(); it != anchors.end(); ++it) {
        const kv_tree_anchor & a = it->second;
        if (a.on_disk || a.data_tgt.empty()) {
            continue;
        }

        const int64_t score = a.refcount * 1000000 + a.heat * 1000 + a.last_used;

        if (best_a == anchors.end() || score < best_a_score) {
            best_a = it;
            best_a_score = score;
        }
    }

    return best_a != anchors.end() && demote_anchor(best_a->second);
}

bool kv_tree::evict_anchor_one() {
    auto best = anchors.end();
    int64_t best_score = 0;

    for (auto it = anchors.begin(); it != anchors.end(); ++it) {
        const kv_tree_anchor & a = it->second;
        if (a.pinned) {
            continue;
        }

        const int64_t score = a.refcount * 1000000 + a.heat * 1000 + a.last_used;

        if (best == anchors.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best == anchors.end()) {
        return false;
    }

    remove_anchor(best);
    st.evicted_anchors++;
    return true;
}

bool kv_tree::evict_block_one() {
    auto best = blocks.end();
    int64_t best_score = 0;

    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        const kv_tree_block & b = it->second;
        if (b.pinned || b.refcount > 1 || has_successor(b)) {
            continue;
        }

        const int64_t score = b.refcount * 1000000 + b.heat * 1000 + b.last_used;

        if (best == blocks.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best == blocks.end()) {
        return false;
    }

    remove_block(best);
    return true;
}

bool kv_tree::evict_seq_one() {
    auto best = seqs.end();
    int64_t best_score = 0;

    for (auto it = seqs.begin(); it != seqs.end(); ++it) {
        const kv_tree_seq & s = it->second;
        if (s.pinned) {
            continue;
        }

        // only leaf sequences: no other stored sequence extends this one
        bool leaf = true;
        for (const auto & other : seqs) {
            const kv_tree_seq & o = other.second;
            if (o.chain.size() <= s.chain.size()) {
                continue;
            }
            if (std::equal(s.chain.begin(), s.chain.end(), o.chain.begin())) {
                leaf = false;
                break;
            }
        }
        if (!leaf) {
            continue;
        }

        const int64_t score = s.last_used;

        if (best == seqs.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best == seqs.end()) {
        return false;
    }

    remove_seq(best);
    return true;
}
```

用下面替换 Task 5 的 `enforce_budget`:

```cpp
bool kv_tree::enforce_budget() {
    while ((size_t) st.bytes_ram > cfg.ram_limit && demote_one()) {
    }

    // still short: evict in the order anchors -> leaf blocks -> whole leaf sequences
    while ((size_t) st.bytes_ram > cfg.ram_limit) {
        if (!evict_anchor_one() && !evict_block_one() && !evict_seq_one()) {
            break;
        }
    }

    if ((size_t) st.bytes_ram > cfg.ram_limit) {
        fprintf(stderr, "[kv-tree] eviction could not free enough ram (%" PRId64 " > %zu)\n", st.bytes_ram, cfg.ram_limit);
        st.evict_refused++;
        return false;
    }

    return true;
}
```

- [ ] **Step 3: harness 加 `run_logic_evict`**

在 `run_logic` 之前插入:

```cpp
static int run_logic_evict() {
    fprintf(stderr, "[t32-tree] logic: eviction\n");

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 20 * 1024;   // 5 chunks worth of fake payloads
    cfg.disk_dir    = "";          // no disk tier: eviction only

    kv_tree tree(cfg);

    const auto tok_a = make_tokens(1536, 0);
    const auto tok_b = make_tokens(1536, 1);

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_a, {}), "evict: park A");
        check_eq(tree.stats().blocks_ram, 3, "evict: A has 3 blocks");
    }

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_b, {}), "evict: park B forces eviction");
        check(tree.stats().evicted_blocks > 0, "evict: blocks were evicted");
        check_eq(tree.stats().park_refused, 0, "evict: park still succeeded");
    }

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        const auto r = tree.restore(io, nullptr, tok_a);
        check_eq(r.C, -1, "evict: A degrades to a full prefill (visible)");
    }

    {
        kv_tree_config cfg2 = cfg;
        cfg2.ram_limit = 4 * 1024;

        kv_tree tree2(cfg2);
        kv_tree_io_fake io;
        io.max_pos = 1535;

        check(!tree2.park(io, nullptr, tok_a, {}), "evict: park refused with a tiny budget");
        check_eq(tree2.stats().park_refused, 1, "evict: refusal counted");
        check_eq(tree2.stats().evict_refused, 1, "evict: refusal is visible");
        check_eq(tree2.stats().blocks_ram, 0, "evict: nothing was left behind");
    }

    return n_fail == 0 ? 0 : 1;
}
```

`run_logic` 的第一行改为:

```cpp
    fprintf(stderr, "[t32-tree] mode logic\n");

    run_logic_evict();
```

- [ ] **Step 4: 构建并跑 logic**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
```
Expected: 全部 PASS; `evict: park B forces eviction` 与 `evict: park refused with a tiny budget` 必须 PASS; `blocks_ram == 0` 表示拒绝后无残留.

- [ ] **Step 5: Commit**

```
git add tools/server/server-kv-tree.h tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git commit -m "server : add kv tree eviction order and pin discipline" -m "Assisted-by: opencode"
```

---

### Task 7: 验收 harness (mini A/B) + 归档 + 文档

**Files:**
- Modify: `tests/test-t32-tree.cpp` (`run_accept` + main 接线)
- Modify: `D:\LLM\Backend\v100-collab\TASKS\T32-agent-session-reuse.md` (追加 `## Result (stage 2)`)
- Modify: `D:\LLM\Backend\v100-collab\RESULTS.md` / `STATUS.md` (追加阶段 2 段)
- Modify: `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md` (状态行)
- Create: `D:\LLM\Backend\v100-collab\artifacts\t32-stage2-model.txt` / `t32-stage2-accept.txt` (运行归档)

**Interfaces:**
- Consumes: Task 1-6 全部
- Produces: 阶段 2 出口证据 (mini A/B 全绿 + 归档)

- [ ] **Step 1: harness 加 `run_accept`**

在 `scenario_ssd` 之后插入:

```cpp
static int run_accept(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] mode accept\n");

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    // A-mini: two 4096-token sequences sharing 3072 tokens, six alternating restores
    const auto tok_a = make_fork_tokens(3072, 1024, 0);
    const auto tok_b = make_fork_tokens(3072, 1024, 1);

    prefill(ctx, 0, tok_a, 0, 512);

    const size_t seq_bytes    = llama_state_seq_get_size_range_ext(ctx, 0, 0, 4096, 0);
    const size_t anchor_bytes = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    check(tree.park(io, nullptr, tok_a, {}), "accept: park A (4096)");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, tok_b, 0, 512);
    check(tree.park(io, nullptr, tok_b, {}), "accept: park B (4096)");

    const auto base_a = run_baseline(ctx, tok_a, 42, 8);
    const auto base_b = run_baseline(ctx, tok_b, 42, 8);

    for (int round = 0; round < 6; ++round) {
        const auto ra = run_tree_path(ctx, tree, io, tok_a, 42, 8);
        check_eq(ra.res.C, 4096, "accept: A restores at the tip");
        check(ra.gen == base_a, "accept: A tokens match the baseline");

        const auto rb = run_tree_path(ctx, tree, io, tok_b, 42, 8);
        check_eq(rb.res.C, 4096, "accept: B restores at the tip");
        check(rb.gen == base_b, "accept: B tokens match the baseline");
    }

    check_eq(tree.stats().tokens_reused, 12 * 4096, "accept: tokens reused");
    check_eq(tree.stats().blocks_ram + tree.stats().blocks_disk, 10, "accept: shared trunk stored once (6+2+2 blocks)");
    check(tree.stats().bytes_ram + tree.stats().bytes_disk < 2 * ((int64_t) seq_bytes + (int64_t) anchor_bytes), "accept: stored bytes below two full copies (dedup)");

    // B-mini: four 1024-token sessions sharing a 512-token prefix
    {
        kv_tree_config cfg2 = cfg;
        cfg2.ram_limit = 4096ull << 20;

        kv_tree tree2(cfg2);
        kv_tree_io_llama io2(ctx, 0);

        std::vector<std::vector<llama_token>> toks;
        for (int i = 0; i < 4; ++i) {
            toks.push_back(make_fork_tokens(512, 512, 10 + i));
        }

        for (int i = 0; i < 4; ++i) {
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
            prefill(ctx, 0, toks[i], 0, 512);
            check(tree2.park(io2, nullptr, toks[i], {}), "accept: park short session");
        }

        check_eq(tree2.stats().blocks_ram + tree2.stats().blocks_disk, 5, "accept: shared prefix stored once (1+4 blocks)");

        for (int i = 0; i < 4; ++i) {
            const auto base = run_baseline(ctx, toks[i], 42, 8);
            const auto r = run_tree_path(ctx, tree2, io2, toks[i], 42, 8);
            check_eq(r.res.C, 1024, "accept: short session restores at the tip");
            check(r.gen == base, "accept: short session tokens match the baseline");
        }
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}
```

`main` 的 model 分支之后加:

```cpp
    if (mode == "accept") {
        return run_accept(ctx, cfg);
    }
```

- [ ] **Step 2: 全量跑 + 归档**

Run:
```powershell
& '<TEMP>\v100\build_test_t32.cmd' test-t32-tree
$env:CUDA_VISIBLE_DEVICES='0'
& '.\build\bin\Release\test-t32-tree.exe' --mode logic
$out = & '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096 2>&1 | Out-String
[System.IO.File]::WriteAllText('D:\LLM\Backend\v100-collab\artifacts\t32-stage2-model.txt', $out, (New-Object System.Text.UTF8Encoding($false)))
$out = & '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on -c 8192 --mode accept --ram-mib 4096 2>&1 | Out-String
[System.IO.File]::WriteAllText('D:\LLM\Backend\v100-collab\artifacts\t32-stage2-accept.txt', $out, (New-Object System.Text.UTF8Encoding($false)))
```
Expected: 三模式 exit 0; 归档里无 FAIL. 另外用 3B 纯 attention 模型做一次对照:
```powershell
& '.\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen2.5-Coder-3B-IQ4_XS.gguf' -ngl 99 -fa on -c 4096 --mode model --ram-mib 4096
```
Expected: tip / sparsify / ssd PASS; fork 场景里 "A' 无锚点 -> C=-1" 的判据对纯 attention 同样成立 (锚点仍由检查点提供, 行为一致).

- [ ] **Step 3: 频道文档**

`TASKS\T32-agent-session-reuse.md` 追加:

```
## Result (stage 2)

- 树模块 `tools/server/server-kv-tree.{h,cpp}` + harness `tests/test-t32-tree.cpp` 完成 (分支 `t32-stage2`)
- 模式: logic (fake IO 单测: 哈希链/去重/稀疏化/淘汰/拒绝) / model (2B 真机: tip/fork/自愈/sparsify/ssd) / accept (mini A/B)
- 结论: park/restore 逐 token 与基线一致 (np=1); 分叉按最深可用锚点恢复, 自愈后第二次免费;
  SSD 单份权威往返逐位一致; 淘汰次序 (锚点 -> 叶块 -> 叶序列 -> 拒绝) 与 pin 纪律生效, 拒绝可见
- 归档: artifacts/t32-stage2-model.txt, artifacts/t32-stage2-accept.txt
- 未接 server (`--kv-tree` 属阶段 3); 生产未动
```

`RESULTS.md` / `STATUS.md` 各追加一段 (格式同阶段 1): 完成内容 + 关键数字 (块数/复用 token/淘汰计数来自归档) + 未 push/未部署. spec 状态行改为:

```
状态: 阶段 0-2 已完成 (阶段 2 树模块 + harness, 分支 `t32-stage2`, 未合并/未 push); 阶段 3-4 未开始
```

- [ ] **Step 4: Commit**

```
git add tests/test-t32-tree.cpp
git commit -m "tests : add kv tree acceptance harness" -m "Assisted-by: opencode"
```

频道文档在 `D:\LLM\Backend\v100-collab` (非 git 仓库), 不进 commit.

---

## 终审修复 (final whole-branch review, 2026-09-27)

三条 Important, 已随 fix wave 落地 (代码见 `tools/server/server-kv-tree.cpp` / `tests/test-t32-tree.cpp`):

1. **park 早退静默丢弃检查点** (违反"绝不静默"): 序列已存时, 非空 `checkpoints` 现在 WRN + `st.anchors_skipped += checkpoints.size()` 再返回.
2. **`promote_prune` 改为按捕获链作用域** (设计决定 9): 签名 `promote_prune(const std::vector<uint64_t> & chain, llama_pos pos)`, "前一锚点"按链上锚点推导, 删除仅限链上 `refcount<2 && kind != TIP && !pinned` 的中间锚点; 去掉全局 `last_promote` 成员. 防止 B 分支的自愈剪掉 A 分支的锚点.
3. **非对齐恢复路径真机覆盖** (`scenario_unaligned`, model 模式): park 1300 token (2 满块 + 276 尾块) 并在非对齐位置 1150 收编锚点; 恢复 1200 token (尾部部分匹配 176) 断言 `C=1150`, `heal=1200`, 与同批次的匹配基线逐 token 一致; 恢复全 1300 断言 `C=1300`, `heal=-1`, 逐 token 一致. 这覆盖生产常见的"部分块装载 + `seq_rm(C,-1)` 裁剪 + 非对齐锚点"路径.

延期 (阶段 3/4, 终审 triage): `bytes_load` 少计部分块/锚点; restore miss 无 INF 日志; 稀疏化"前一锚点"仍为全局 (同 #2 的链化思路可在阶段 3 一并处理); "dropping it" 措辞与实际保留; 统计漂移 (rollback/级联计数); demote 评分可被 heat/last_used 反转; `cfg.chunk > 0` 校验; `cfg.debug` 未用; usage 文本缺 accept; 32-bit/ftell 上限; promote_prune 删除分支与 payload 校验分支无测试; harness 卫生项. 阶段 3 建议: 限制 restore 瞬态 RAM 峰值 (默认参数下可达 ~13 GB); CLI 校验; drop-sequence API; restore miss 的 INF 日志.

---

## Self-Review

**Spec 覆盖检查 (§1-§6, 阶段 2 范围):**

| spec 条目 | 任务 |
|---|---|
| §1.2 块链/内容哈希/逐 token 校验/元数据 | T1 |
| §1.3 锚点 (tip/message/ondemand, 硬约束不凭空生成) | T2/T3/T4 |
| §2 引擎 API 使用 (range + PARTIAL_ONLY) | T2 (adapter) |
| §3.1 park (缺失段/末端锚点/收编/簿记) | T1/T2/T3 |
| §3.2 restore (C=最深锚点/装载/裁剪/状态/路径锚点) | T2/T3 |
| §3.2 自愈 (重放后捕获) | T4 |
| §3.3 降级 (位置不从 0 / SSD 读失败 / 预算不够) | T2 (前置检查) / T5 / T6 |
| §4 稀疏化 + 冲突留靠前 + 分叉提升/剪枝 | T3 / T4 |
| §5.1 统计 | T1-T6 (stats) |
| §5.2 RAM/SSD 放置 (单份权威/移动) | T5/T6 |
| §5.3 淘汰次序 + 拒绝 | T6 |
| §5.4 pin 纪律 | T5/T6 |
| §6 错误处理与可观测 (WRN + 计数) | T2-T6 |
| §7.3 反例 (无锚点/位置不从 0/SSD 读失败) | T3 (C=-1) / T2 (park 拒绝) / T5 |
| 独立 harness (短序列正确性) | T1-T7 |

未覆盖 (属阶段 3/4, 见 spec §9): server 集成/`--kv-tree`/A/B 真实场景/指标上报/持久化.

**Placeholder 扫描:** 无 TBD/TODO; 每个代码步骤都是完整函数或明确 old/new 替换.

**类型一致性:** 全计划统一使用 `kv_tree_block{hash,pos0,pos1,refcount,heat,last_used,on_disk,pinned,transient,bytes,path,tokens,data}`、`kv_tree_anchor{blk_hash,pos,kind,refcount,heat,last_used,on_disk,pinned,transient,bytes,path,data_tgt,data_dft,data_spec}`、`kv_tree_io` 六方法; 计数一律用 `bytes` 字段 (与 data vector 无关, SSD 上 vector 为空); 锚点键一律 `(blk_hash, pos)`; 恢复点一律 `res.C` / `res.heal`.

## Execution Handoff

Plan complete and saved to `D:\LLM\Backend\v100-collab\artifacts\t32-tree-plan-stage2.md`. 两种执行方式:

1. **Subagent-Driven (推荐)** - 每任务派新 subagent, 任务间双阶段审查 (与阶段 0/1 相同流程)
2. **Inline Execution** - 本会话内按 executing-plans 批量执行 + 检查点

执行前提: 分支 `t32-stage2` 从 master `52b7bf7de` 起; 提交按 Global Constraints 自动进行; push/部署/合并需另行批准.
