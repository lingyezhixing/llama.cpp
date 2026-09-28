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

    const size_t seq_bytes = llama_state_seq_get_size_range_ext(ctx, 0, 0, 4096, 0);

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
    check_eq(tree.stats().blocks_ram + tree.stats().blocks_disk, 8, "accept: shared trunk stored once (6+2 blocks)");
    check(tree.stats().bytes_ram + tree.stats().bytes_disk <= (int64_t) seq_bytes * 2 * 10 / 16, "accept: stored bytes are far below two full copies");

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
