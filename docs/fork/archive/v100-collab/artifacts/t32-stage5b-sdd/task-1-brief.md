### Task 1: 分叉间距忽略猜测 + 测试 + 验收

**Files:**
- Modify: `tools/server/server-kv-tree.cpp` (`capture_anchor` 的 prev 扫描)
- Modify: `tests/test-t32-tree.cpp` (`run_logic_fork` 增断言)
- Modify (artifact): `artifacts\t32-stage3-ab.ps1` (heal 模式去掉 `-fork_step 512`, 恢复默认)

**Interfaces:**
- Consumes: `KV_TREE_ANCHOR_MESSAGE` 常量 (已有), `cfg.fork_step`.
- Produces: 分叉捕获不受 MESSAGE 锚点影响 (heal 验收默认 fork_step 下仍捕获).

- [ ] **Step 1: 写失败测试**

`run_logic_fork` 第一个块 (fork spacing) 之后插入:

```cpp
    // guesses never suppress a fork anchor
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 32768;
        cfg.fork_step   = 8192;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);
        const auto tok = make_tokens(4096, 0);

        kv_tree_anchor_in guess;
        guess.pos      = 1024;
        guess.data_tgt = kv_tree_io_fake::pattern(0, 64);

        {
            kv_tree_io_fake io;
            io.max_pos = 4095;
            check(tree.park(io, nullptr, tok, { guess }), "fork: park with a guess at 1024");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 5119;
            check(tree.capture_anchor(io, nullptr, tok, 5120), "fork: the guess does not suppress the fork");
        }
        check_eq(tree.stats().anchors_skipped_step, 0, "fork: no spacing skip against the guess");
        {
            kv_tree_io_fake io;
            io.max_pos = 13311;
            check(!tree.capture_anchor(io, nullptr, tok, 13312), "fork: another fork anchor still suppresses (8192)");
        }
        check_eq(tree.stats().anchors_skipped_step, 1, "fork: the fork-vs-fork skip is counted");
    }
```

- [ ] **Step 2: 运行确认失败**

Run: build + `--mode logic` → FAIL (`fork: the guess does not suppress the fork`).

- [ ] **Step 3: 实现**

`capture_anchor` 的 prev 扫描 (阶段 5 后的 363-371 附近) 替换为:

```cpp
    llama_pos prev = -1;
    for (const auto & kv : anchors) {
        if (kv.second.pos >= pos || kv.second.pos <= prev) {
            continue;
        }
        if (kv.second.kind == KV_TREE_ANCHOR_MESSAGE) {
            continue;   // guesses never suppress a fork anchor
        }
        if (std::find(chain.begin(), chain.end(), kv.first.first) != chain.end()) {
            prev = kv.second.pos;
        }
    }
```

- [ ] **Step 4: 运行确认通过**

Run: build + `--mode logic` → 0 FAIL (75 + 3 = 78 checks).

- [ ] **Step 5: 验收 (heal 恢复默认 fork_step)**

`t32-stage3-ab.ps1` heal 模式: `Start-Srv $true 512 $dir $true 512 -fork_step 512` -> `Start-Srv $true 512 $dir $true 512` (去掉 fork_step 覆盖, 用默认 8192); 注释同步更新 (487 猜测不再压制 1024 分叉).

```powershell
$env:CUDA_VISIBLE_DEVICES='1'
Remove-Item -Force 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\llama-server-impl.dll' -ErrorAction SilentlyContinue
& '<TEMP>\v100\build_server.cmd'
& powershell -ExecutionPolicy Bypass -File 'D:\LLM\Backend\v100-collab\artifacts\t32-stage3-ab.ps1' -Mode heal | Tee-Object 'D:\LLM\Backend\v100-collab\artifacts\t32-stage5b-heal.txt' | Select-String 'REBUILT|RESULT'
```

Expected: `RESULT heal: 0 failure(s)`, `REBUILT lines=1` (1024 分叉锚点在默认 fork_step 下仍被捕获).

- [ ] **Step 6: 回归 + 提交**

```powershell
$art='D:\LLM\Backend\v100-collab\artifacts'
& 'D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-tree.exe' -m '<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf' -ngl 99 -fa on --mode model --ram-mib 4096 -c 8192 2>&1 | Tee-Object "$art\t32-stage5b-model.txt" | Select-Object -Last 1
& powershell -ExecutionPolicy Bypass -File "$art\t32-stage3-ab.ps1" -Mode fork | Tee-Object "$art\t32-stage5b-fork.txt" | Select-String 'RESULT'
git -C D:\LLM\Backend\src\llama.cpp-my add tools/server/server-kv-tree.cpp tests/test-t32-tree.cpp
git -C D:\LLM\Backend\src\llama.cpp-my commit -m "server : do not let checkpoint guesses suppress fork anchors" -m "Assisted-by: opencode"
```

Expected: model 0 FAIL; fork `RESULT fork: 0 failure(s)` (5/5 逐位一致).
