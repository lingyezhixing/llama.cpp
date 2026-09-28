### Task 3: server 接线 (miss-heal) + 模型测试

**Files:**
- Modify: `tools/server/server-context.cpp:361-372` (prompt_restore_tree), `:1795-1801` (调用方)
- Test: `tests/test-t32-tree.cpp` (`scenario_fork_miss`, 注册)

**Interfaces:**
- Consumes: Task 2 的 `res.heal` on miss.
- Produces: server 在 miss 时也设 `tree_heal` (Task 4 fork 模式验收依赖日志 `captured heal anchor at <deep>` 出现在 miss 之后).

- [ ] **Step 1: 写失败测试 (model)**

`scenario_restore_anchors` 之后插入:

```cpp
static int scenario_fork_miss(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] scenario: fork miss\n");

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tok_a = make_tokens(4096, 3);
    auto tok_b = make_tokens(4096, 4);
    for (int i = 0; i < 2048; ++i) {
        tok_b[i] = tok_a[i];
    }

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    if (prefill(ctx, 0, tok_a, 0, 512) != 0) {
        return 1;
    }
    check(tree.park(io, nullptr, tok_a, {}), "fork-miss: park A");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    const kv_tree_restore r1 = tree.restore(io, nullptr, tok_b);
    check_eq(r1.C, -1, "fork-miss: the first visit misses");
    check_eq(r1.heal, 2048, "fork-miss: the miss reports the divergence point");

    // the server prefills to the heal position and captures there
    if (prefill(ctx, 0, std::vector<llama_token>(tok_b.begin(), tok_b.begin() + 2048), 0, 512) != 0) {
        return 1;
    }
    check(tree.capture_anchor(io, nullptr, tok_b, 2048), "fork-miss: capture at the divergence point");
    if (prefill(ctx, 0, tok_b, 2048, 512) != 0) {
        return 1;
    }

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    const kv_tree_restore r2 = tree.restore(io, nullptr, tok_b);
    check_eq(r2.C, 2048, "fork-miss: the second visit restores at the fork");
    check_eq(r2.heal, -1, "fork-miss: no heal on a full match");

    const auto base = run_baseline(ctx, tok_b, 42, 8);
    const auto r = run_tree_path(ctx, tree, io, tok_b, 42, 8);
    check_eq(r.res.C, 2048, "fork-miss: the tree path restores at the fork");
    check(r.gen == base, "fork-miss: tokens match the baseline");

    tree.dump();

    return 0;
}
```

`main` 的 model 分派加 `ret |= scenario_fork_miss(ctx, cfg);`.

- [ ] **Step 2: 运行确认失败**

Run: build + model mode
Expected: FAIL `fork-miss: the first visit misses` 之后的 heal 断言 (engine 已支持, 但 server 未接线不影响 harness; 此步实际为 GREEN 验证引擎路径) —— 若引擎 Task 2 已完成, 本测试应直接 PASS; 失败点应在 server 端行为 (下一步的验收模式). 记录实际结果.

- [ ] **Step 3: server 接线**

(3a) `prompt_restore_tree` miss 分支:

```cpp
        kv_tree_restore res = tree.restore(io_tgt, io_dft_ptr, tokens.get_tokens());
        if (res.C <= 0) {
            SLT_INF(*this, "kv tree: restore miss for %zu tokens, full prefill\n", tokens.size());
            tree_heal = res.heal;   // seed a fork anchor while the prompt is re-prefilled
            return false;
        }
```

(3b) 调用方 (get_available_slot) 保留 heal 过 prompt_clear:

```cpp
                if (tree) {
                    ret->prompt_park(*tree);

                    // no point restoring when the request will not reuse the prompt
                    if (!task.params.cache_prompt || !ret->prompt_restore_tree(*tree, task.tokens, params_base.n_ctx_checkpoints,
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL || ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS)) {
                        const llama_pos heal = task.params.cache_prompt ? ret->tree_heal : -1;
                        ret->prompt_clear();
                        ret->tree_heal = heal;
                    }
                }
```

- [ ] **Step 4: 运行确认通过**

Run: build test + model mode -> 0 FAIL (56 + 7 = 63 checks); rebuild server; heal 模式 `RESULT heal: 0 failure(s)`.

- [ ] **Step 5: 提交**

```powershell
git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-context.cpp tests/test-t32-tree.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : seed fork anchors when a tree restore misses" -m "Assisted-by: opencode"
```

---

