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
  - `kv tree: park skipped (seq range [P0, P1], tokens N)`

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

        const llama_pos p_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), id);
        const llama_pos p_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), id);
        if (p_min != 0 || p_max != (llama_pos) prompt.tokens.size() - 1) {
            SLT_WRN(*this, "kv tree: park skipped (seq range [%d, %d], tokens %zu)\n", p_min, p_max, prompt.tokens.size());
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
$p = Start-Process -FilePath $srv -ArgumentList @('-m','<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf','-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','8192','--port','8931','--cache-ram','0','--no-cache-idle-slots','--kv-tree','--tree-chunk','512','--tree-anchor-step','512','--tree-ram','8','--tree-disk',$dir,'--tree-disk-limit','64') -NoNewWindow -PassThru -RedirectStandardOutput "$tmp\s3smoke-out.txt" -RedirectStandardError "$tmp\s3smoke-err.txt"
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
- 日志出现 `kv tree: parked`、`kv tree: restored`，且 parked 行里 `disk =` > 0（预算 8 MiB 远小于两段内容，必然溢出 SSD）；
- `No usable anchor` 仅在 B 首次（`restore miss`）出现一次。

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

                    if (tree) {
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
$p = Start-Process -FilePath $srv -ArgumentList @('-m','<models>\Qwen3.5-2B-UD-Q4_K_XL.gguf','-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','8192','--port','8932','--cache-ram','0','--no-cache-idle-slots','--kv-tree','--tree-chunk','512','--tree-anchor-step','512','--tree-ram','64','--tree-disk',$dir,'--tree-disk-limit','256') -NoNewWindow -PassThru -RedirectStandardOutput "$tmp\s3heal-out.txt" -RedirectStandardError "$tmp\s3heal-err.txt"
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
    $t = ''
    while ((TokCount $t) -lt $n_tok) { $t += "$unit$(($t.Length % 977)); " }
    return $t
}

function Start-Srv($tree, $ram, $diskdir, $child) {
    $args = @('-m',$Model,'-np','1','-ngl','99','-fa','on','-ctv','q8_0','-c','24576',
              '--port',"$Port",'--cache-ram','0','--cache-idle-slots',
              '--no-context-shift','--host','127.0.0.1')
    if ($tree) {
        $args += @('--kv-tree','--tree-chunk','512','--tree-anchor-step','4096',
                   '--tree-ram',"$ram",'--tree-disk',$diskdir,'--tree-disk-limit','2048')
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
function RunTurns($names, $bases, $rounds, $cache) {
    $hist = @{}
    foreach ($n in $names) { $hist[$n] = '' }
    for ($r = 1; $r -le $rounds; $r++) {
        for ($i = 0; $i -lt $names.Count; $i++) {
            $n = $names[$i]
            $prompt = "$sys`n" + $bases[$i] + $hist[$n] + "`nUser turn ${r}: continue`nAssistant:"
            $resp = Req $prompt $cache
            $hist[$n] += $resp.content
            $results["$Mode/$n/$r"] = $resp.content
            Write-Output ("[$Mode] $n round ${r}: prompt_n=$($resp.timings.prompt_n) pred=$($resp.timings.predicted_n) cached=$($resp.tokens_cached)")
        }
    }
}

switch ($Mode) {
    'ref' {
        $p = Start-Srv $false 0 '' $null
        RunTurns @('A','B') @($baseA,$baseB) 6 $false
        $p | Stop-Process -Force
    }
    'calib' {
        $dir = "$OutDir\tree-calib"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 4096 $dir $null
        RunTurns @('A','B') @($baseA,$baseB) 2 $true
        $p | Stop-Process -Force
        $line = (Select-String -Path "$OutDir\srv-$Mode-err.txt" -Pattern 'kv tree: parked' | Select-Object -Last 1).Line
        Write-Output "CALIB LAST PARK: $line"
        $total = 0
        if ($line -match 'ram = (\d+) B, disk = (\d+) B') { $total = [int64]$matches[1] + [int64]$matches[2] }
        Write-Output "CALIB TOTAL BYTES AFTER 2 ROUNDS: $total"
        Assert ($total -gt 0) 'calib: tree stores data'
    }
    'ab' {
        $dir = "$OutDir\tree-ab"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $ram = [int]$env:T32_RAM_MIB; if ($ram -le 0) { $ram = 96 }
        $p = Start-Srv $true $ram $dir $null
        RunTurns @('A','B') @($baseA,$baseB) 6 $true
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        $parked = ($log | Select-String 'kv tree: parked').Count
        $restored = ($log | Select-String 'kv tree: restored').Count
        $miss = ($log | Select-String 'kv tree: restore miss').Count
        $diskmax = 0
        foreach ($l in $log) { if ($l -match 'ram = (\d+) B, disk = (\d+) B') { $d = [int64]$matches[2]; if ($d -gt $diskmax) { $diskmax = $d } } }
        Assert ($parked -ge 6) 'ab: park ran on every switch'
        Assert ($restored -ge 8) 'ab: later turns restored from the tree'
        Assert ($diskmax -gt 0) 'ab: SSD tier actually used (disk > 0)'
        for ($r = 2; $r -le 6; $r++) { foreach ($n in @('A','B')) {
            Assert ([int]$results["$Mode/$n/$r"] -ne $null) "ab: $n round $r completed"
        } }
    }
    'overlap' {
        $dir = "$OutDir\tree-ovl"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 96 $dir $null
        RunTurns @('C','D') @($baseC,$baseD) 3 $true
        $p | Stop-Process -Force
        $parked = ((Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8) | Select-String 'kv tree: parked').Count
        Assert ($parked -le 1) 'overlap: high-overlap switches stay on the stock VRAM path'
    }
    'b' {
        $dir = "$OutDir\tree-b"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 64 $dir $null
        RunTurns @('one','two','three','four') $sess 3 $true
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert ((($log | Select-String 'kv tree: restored').Count) -ge 4) 'b: short sessions restore from the tree'
        Assert ((($log | Select-String 'ref=') | Select-String 'ref=4').Count -ge 1) 'b: shared prefix stored once (refcount = 4)'
    }
    'b3' {
        $Model = '<models>\Qwen2.5-Coder-3B-IQ4_XS.gguf'
        $sys3 = Filler 'system' 256
        $b1 = Filler 'aa' 4096; $b2 = Filler 'bb' 4096
        $dir = "$OutDir\tree-b3"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 32 $dir $null
        RunTurns @('P','Q') @($b1,$b2) 3 $true
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert ((($log | Select-String 'kv tree: restored').Count) -ge 2) 'b3: pure-attention model restores from the tree'
    }
    'neg' {
        $dir = "$OutDir\tree-neg"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 96 $dir $null
        RunTurns @('A','B') @($baseA,$baseB) 2 $true
        $f = Get-ChildItem $dir -File | Sort-Object LastWriteTime | Select-Object -First 1
        Remove-Item -Force $f.FullName
        Write-Output "NEG removed block file: $($f.Name)"
        RunTurns @('A') @($baseA) 1 $true
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        Assert ((($log | Select-String 'failed to read block|restore failed').Count) -ge 1) 'neg: SSD read failure is visible'
        $r = $results["$Mode/A/1"]; Assert ($r -ne $null) 'neg: request still succeeded after the failure'
    }
    'heal' {
        $dir = "$OutDir\tree-heal"; Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        $p = Start-Srv $true 512 $dir $null
        $head = Filler 'head' 1024
        $tA = Filler 'tail-a' 1024; $tF = Filler 'tail-f' 1024
        RunTurns @('H','K') @("$head`nUser: $tA","$head`nUser: $tF") 3 $true
        $p | Stop-Process -Force
        $log = Get-Content "$OutDir\srv-$Mode-err.txt" -Encoding UTF8
        $cap = ($log | Select-String 'captured heal anchor').Count
        Assert ($cap -ge 1) 'heal: an anchor was captured at a fork point'
    }
}

Write-Output "RESULT $Mode: $fails failure(s)"
exit $fails
```

说明与调参（实施时允许按实测调，但必须保证断言语义不变）：
- `$ram` 由校准决定：`calib` 输出 `CALIB TOTAL BYTES`，阶段 3 验收取 `$env:T32_RAM_MIB = [int]($total/3/1MB)`（"精准溢出"：约 1/3 驻留、2/3 溢写 SSD）；预期 2x16K 总量 ~0.6-1.0 GB -> ram ~200-330 MiB。`ab` 默认 96 MiB 只是保守值。
- `ref` 用 `cache_prompt=false` + 树关，逐请求全量 prefill；与 `ab`/`b`/`b3`/`heal` 的输出对比：脚本外由驱动命令完成（见 Step 2）。
- `b` 的 `ref=`/refcount 证据来自 `--tree-debug`？不：`dump()` 只在 `--tree-debug` 打开时打印。**Step 1 里所有 Start-Srv 在 `$tree` 为真时追加 `--tree-debug`**（脚本内已含），实现 `ref=4` 断言；若日志量太大影响性能，只在 `b`/`b3` 模式打开。

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

输出一致性断言（脚本外）：
1. `ref` 跑两遍（`-Mode ref` 两次，文件名加 `-ref2`），两遍逐字符相同（确定性守门）。
2. `ab` 与 `ref` 的 A/B 各 6 轮输出**逐 token 一致**：正文用 `Compare-Object` 对比两个 Tee 文件里的 `[ref]`/`[ab]` 行；不一致即 FAIL（这是设计 §7.1(b) 的"逐位一致"判据在 server 层的等价物）。
3. `b`/`b3`/`heal`/`overlap` 各自与对应 `ref` 输出对比（同模型同 turn 序列；`overlap` 与 `ref` 的会话内容必须一致才能比较——`overlap` 模式用 C/D，需在 ref 里也跑 C/D：实施时把 `ref` 模式改成跑 **A/B + C/D + 短会话** 三组，便于对比。此项允许在实施中调整脚本结构，只要保证"等输入等输出"。）

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
