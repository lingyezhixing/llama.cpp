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

                    if (!ret->prompt_restore_tree(*tree, task.tokens)) {
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

