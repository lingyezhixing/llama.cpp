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
                                SLT_TRC(slot, "kv tree: heal position %d missed (now %zu)\n", hp, slot.prompt.n_tokens());
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

- [ ] **Step 5: 构建 + heal/erase 冒烟**

构建同 Task 3 Step 3 第一步。heal 场景（消息边界制造锚点 + 分叉）：

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
$dir = "$tmp\t32-tree-heal"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
$env:CUDA_VISIBLE_DEVICES='0'
$srv = "$repo\build\bin\Release\llama-server.exe"
$common = 'filler ' * 500                       # ~1000+ tokens shared head
$tail_a = 'alpha-tail ' * 200                   # A 的 msg2
$tail_f = 'gamma-fork ' * 200                  # A'' 的 msg2（与 A 在 msg2 内分叉）
$p = Start-Process -FilePath $srv -ArgumentList @('-m','<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf','-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','8192','--port','8932','--cache-ram','0','--no-cache-idle-slots','--kv-tree','--tree-chunk','512','--tree-anchor-step','512','--tree-ram','256','--tree-disk',$dir,'--tree-disk-limit','256') -NoNewWindow -PassThru -RedirectStandardOutput "$tmp\s3heal-out.txt" -RedirectStandardError "$tmp\s3heal-err.txt"
Start-Sleep -Seconds 25
function Req($prompt) { (Invoke-RestMethod -Uri 'http://127.0.0.1:8932/completion' -Method Post -ContentType 'application/json' -Body (@{prompt=$prompt; n_predict=32; temperature=0; top_k=1; seed=42; cache_prompt=$true} | ConvertTo-Json) ) }
$rA  = Req "User: $common`nAssistant: ok`nUser: $tail_a"     # A 存档（含 msg 边界检查点）
$rF1 = Req "User: $common`nAssistant: ok`nUser: $tail_f"     # A' 首次：分叉 -> heal 捕获
$rS  = Req "User: $common`nAssistant: ok`nUser: short"       # 短会话：逼 A/A' 出槽
$rF2 = Req "User: $common`nAssistant: ok`nUser: $tail_f"     # A' 二次：应命中 heal 锚点
$p | Stop-Process -Force
Select-String -Path "$tmp\s3heal-err.txt" -Pattern 'kv tree: (captured heal anchor|restored|heal position|parked)' | ForEach-Object { $_.Line }
Invoke-RestMethod -Uri 'http://127.0.0.1:8932/health' -Method Get -ErrorAction SilentlyContinue | Out-Null
```

Expected:
- 日志恰有 **1** 条 `captured heal anchor at N`（A' 首次）；
- A' 二次的 `restored N tokens` 中 N 等于上面捕获的 N（命中自愈锚点，不再重放）；
- 没有 `failed to capture heal anchor`。

SLOT_ERASE drop（同一 server 或新起）：
```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:8932/slots/0?action=erase' -Method Post
Select-String -Path "$tmp\s3heal-err.txt" -Pattern 'kv tree: dropped' | ForEach-Object { $_.Line }
```
Expected: 出现 `kv tree: dropped the stored sequence`（槽内内容覆盖到已存序列 tip 时）。

idle park：把 `--no-cache-idle-slots` 换成默认（启用）重跑上面 3 个请求，日志应出现 idle 的 `kv tree: parked`（同一内容重复 park 时 `seqs.count(tip)` 会命中，日志照打但无新数据；这是幂等设计）。

- [ ] **Step 6: 提交**

```powershell
git -C $repo add tools/server/server-context.cpp
git -C $repo commit -m "server : heal fork points, idle parks and slot erase in the kv tree" -m "Assisted-by: opencode"
```

---

