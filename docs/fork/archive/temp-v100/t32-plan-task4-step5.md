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
function Req($prompt) { (Invoke-RestMethod -Uri 'http://127.0.0.1:8932/completion' -Method Post -ContentType 'application/json' -TimeoutSec 60 -Body (@{prompt=$prompt; n_predict=32; temperature=0; top_k=1; seed=42; cache_prompt=$true; message_delimiters=@(@{user='User:'})} | ConvertTo-Json -Depth 4) ) }
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

