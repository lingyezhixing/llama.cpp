
---

## T20: MTP 轨迹一致性独立排查 (2026-09-23, implementer)

> 用户要求: 不盲信旧报告 (`D:\LLM\Backend\MTP-轨迹一致性分析与修复报告.md`), 独立复现/定位/修复。
> 本节的每条结论都有独立证据; 与旧报告不一致处已注明。

### 0. 结论摘要

- **H1 (输出 token 来源) 确认**: 提交进上下文的 token 永远是 target 自己的采样, 不是 draft。
  代码级: `common/sampling.cpp:678-706` `common_sampler_sample_and_accept_n` 对 target logits 逐位置
  `common_sampler_sample` (greedy=argmax) 并 `accept`; `draft[i] != id` 只决定提前结束验证。draft 只影响速度。
  -> **不存在"降智"机制** (与旧报告一致, 已独立复核)
- **分叉源共 3 个** (旧报告只列了 FA 一个, 且漏掉本 fork 引入的 GDN 那个):
  | # | 源 | 性质 | 首个分叉证据 | 修复 |
  |---|---|---|---|---|
  | S1 | FA VEC (n_q=1) vs TILE (n_q>=2), Volta | 上游 kernel 选择 | 层 3 输出 maxabs 2.5e-3 | 小批量强制 VEC (旧报告 env) |
  | S2 | **GDN vec4 布局分支** `vec4 = n_tokens > 1` | **本 fork T03/A2 引入** | 层 38/45/57 输出 1e-5..4e-3 (数据相关) | 布局统一 (vec4 for all 或连续标量) |
  | S3 | **FA VEC split-K 的 padding 边界** | 上游 split-K 设计 | 层 15 输出 2.9e-6 (n_kv 跨 256 时) | 小批量禁用 KV 切分 (PB=1, 代价大) |
- **验收 (S1+S2+S3 全开)**: n-max 1/2/3 与无 spec **逐 token 完全一致**:
  - 短上下文 (prompt 14 tok): **900/900 token 全等** (n-max1/2/3 各自独立运行)
  - 32k (prompt 32041 tok): **150/150 全等**
  - 128k (prompt 128270 tok): **100/100 全等**
- **性能代价** (llama-bench 同 session 交替 A/B):
  - S2 (GDN 布局统一, 用 `GGML_CUDA_GDN_VEC4=1` 模拟): tg128 d0 **-0.6%** / d32768 **-1.3%** (交错复测)
  - S1 (小批量 VEC): 短点 nmax1/2/3 = 36.9/39.1/37.2 vs 36.9/39.5/36.9 t/s (噪声内)
  - **S3 (`GGML_CUDA_FATTN_PB_FORCE=1`): tg128 d32768 -31% (23.10 -> 15.83)** -> 不能默认开
  - PPL (3 修复全开) = **4.3562** (与交付一致; PPL 走批量路径, 不受 decode 修复影响)
- **生产判定**: 无降智机制; 残余分叉 (仅 S1+S2 修复时) 全部发生在 **极端近并列** (top1-top2 = 0.003..0.015 nats),
  两条续写均合法连贯。要 100% 逐 token 一致必须开 S3, 代价 -31% @32k -> **由用户裁决 (见 QUESTIONS)**

### 1. 工具与基线 (Phase 0)

- apply 旧 patch (`mtp-work.patch` 转 UTF-8/LF): `fattn.cu` 的 `GGML_CUDA_FA_SMALL_BATCH_VEC` + `GGML_CUDA_FA_DEBUG`;
  恢复 `examples/batch-invariance/` 并 `-DLLAMA_BUILD_EXAMPLES=ON` 重建 (工具可编译, API 均存在)
- **当前部署 (T19 交付) 不含旧修复** -> 基线 = 未修复状态, 已复现旧报告的"修复前"分叉
- 本次新增调试开关 (仅实验, 默认全关):
  `GGML_CUDA_GDN_VEC4` (0/1 覆盖 GDN 布局), `GGML_CUDA_FA_FORCE` (vec/tile/mma),
  `GGML_CUDA_FATTN_PB_FORCE` (非 stream-K 内核强制 parallel_blocks), `GGML_DBG_LOGITS` (采样点 top-8 logits dump)
- 差分工具增强: `--seq-a` (A 逐 token 解码), `--probe/--rollback` (模拟 server 部分接受+回滚), 全行 (不只 row 0) 对比

### 2. 独立复现 (server greedy A/B, temp0/top-k1/seed42)

- 命令: `llama-server -m ... -np 1 -ngl 99 -fa on -c 16384 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max N`
  + `/completion` (prompt 14 tok, n_predict 900, temperature 0, top_k 1, seed 42, return_tokens)
- 未修复基线: **n-max 1/2/3 全部在 token 207 分叉** (三者同点, 同续写) -> 分叉与 n-max 无关
- 确定性: 同配置独立重启两次, 900 token 逐 token 一致 (含 draft 统计)
- 分叉点性质: `top1-top2 = 0.0154 nats` (" Egyptians" -0.6894 vs " Babylon" -0.7048, 概率比 1.015)

### 3. 根因定位 (算子级差分 + logits dump)

- **S1 (FA VEC/TILE)** (旧报告 H2, 确认): A=1(VEC) vs B=2/4(TILE), 首分叉层 4 输入 = 层 3 (首个 FA) 输出,
  maxabs 2.5e-3; A=2 vs B=4 (都 TILE) 全等; `GGML_CUDA_FA_DEBUG` 确认 n_q=1->VEC(100), n_q=2..4->TILE(200)
- **S2 (GDN vec4)** (**新发现**, 旧报告"GDN batch-invariant"在新 base 上不成立):
  - 开 `FA_SMALL_BATCH_VEC=1` (两臂都 VEC) 后仍有残余分叉 (层 58/46/39 起, 前缀相关)
  - A=2 vs B=3/4 (都 vec4) 全等; A=1 vs B>=2 (标量 vs vec4) 分叉 -> 定位到 `n_tokens > 1` 分支
  - 机理: vec4 路径 lane 行映射 = `4*lane+r` (连续), 标量 = `r*warp_size+lane` (跨步);
    `warp_reduce_sum` 归约顺序不同 -> kv/attn 的 ulp 级差 (数据相关, 大部分层恰好舍入相同)
  - **决定性验证**: `GDN_VEC4=1` (两臂都 vec4) 或 `=0` (都标量) -> A=1 vs B=2 **全 64 层 x 2 行逐位一致**
- **S3 (FA split-K padding)** (**新发现**, 上游设计属性):
  - 工具在 `prefill 255` (A n_kv=256, B n_kv=257->padded 512) 时分叉 (48/64 层, 首差 2.9e-6 @层 15);
    `prefill 254` (n_kv 255 vs 256, 同 padding) 全等 -> 与 padded n_kv 相关
  - `GGML_CUDA_FATTN_PB_FORCE=1` (两臂 pb=1) -> `prefill 255` **逐位一致** -> 确认 = KV 切分
  - 机理: `launch_fattn` 非 stream-K 路径 `ntiles_KV = ceil(n_kv/256)` -> parallel_blocks -> 内核 KV 交错切分
    (`k_VKQ_0 += gridDim.y*nthreads`) -> online-softmax 累加顺序不同
  - server 侧: `GGML_DBG_LOGITS` dump 显示 **首个 logits 差在采样点 242 (position 255, 即首个 256 边界)**,
    此前 242 个采样点 top-8 logits 逐位一致 (如 28.1647377); 差异 ~0.02, 之后持续存在
  - 256 边界每次跨越都会产生一次 1e-6 级扰动; 是否翻转 token 取决于是否遇到近并列

### 4. 修复与验收

- 修复方式 (实验用 env, 生产需固化为代码):
  - S1: `GGML_CUDA_FA_SMALL_BATCH_VEC=1` (Volta 小批量 n_q*gqa_eff<=16 走 VEC; 旧报告 patch)
  - S2: `GGML_CUDA_GDN_VEC4=1` (布局统一; 推荐固化方向 = 删除 `n_tokens > 1` 分支, 一律 vec4, 代价 ~1%)
  - S3: `GGML_CUDA_FATTN_PB_FORCE=1` (非 stream-K 内核 pb=1; 代价 -31% @d32768, 只能可选)
- 验收 (greedy, temp0/top-k1/seed42, `return_tokens` 逐 token 对比, 含 `ignore_eos`):

| 场景 | prompt | 生成 | none | n-max1 | n-max2 | n-max3 |
|---|---|---|---|---|---|---|
| 短 | 14 | 900 | 26.11/25.89 | 36.94/36.93 | 39.46/39.20 | 36.93/37.38 |
| 32k | 32041 | 150 | 15.85 | 22.06 | 24.18 | 24.26 |
| 128k | 128270 | 100 | 7.23 | 9.60 | 10.00 | 10.58 |

  (短/32k/128k 各列均与 none **逐 token 一致**; 短上下文为两组独立复测)
- 确定性: 同配置重启逐 token 一致 (none/nmax1 各复测一次)
- PPL (3 修复全开) = 4.3562; 与 T19 交付一致
- 性能 (llama-bench tg128, 同 session 交错): base 26.61/26.22/23.10 (d0/d4096/d32768);
  GDN 修复 26.45/25.93/21.64; +PB 修复 26.43/24.45/15.77 (d32768 -31.7%)
  - 交错复测 (base,gdn / gdn,base): d32768 23.23/23.02 与 22.70/23.08 -> GDN 修复实际 ~-1.3% (首次 -6.3% 是热漂移)

### 5. 生产 MTP3 风险判定

- **无降智机制** (H1); 全部 token 来自 target 采样
- 仅修 S1+S2 时: 分叉 = 极端近并列 (<=0.02 nats) 的数值翻转, 幅度 ~1e-6 级扰动累积; 900 token 内 1-3 次
  (n-max1@558 / n-max2@587 / n-max3@286, 均经 logits 证据确认是近并列)
- 两条续写均连贯合法 ("Egyptians" vs "Babylon" / "rooms" vs "racks"), 不是错误答案
- 要 100% 一致: 必须开 S3 (代价 -31% @32k, 且 128k 未测代价) -> **请用户裁决** (采纳/放弃/只保留 S1+S2)

### 6. 交付物与状态

- 调试工具: `examples/batch-invariance/batch-invariance.cpp` (增强版, 未提交)
- 实验 patch: `artifacts/t20-work.patch` (工作区 diff); 备份 `D:\LLM\Backend\mtp-backup\` 原 patch 仍在
- 构建: T20 调试版 ggml-cuda.dll SHA `DE3F5E50...`; llama-common.dll (含 logits dump) 备份于 `%TEMP%\v100\llama-common-T19.dll`
- 原始数据: `%TEMP%\v100\t20_*` (server JSON/dump/log/工具输出/性能日志)
- **未提交、未 push**; 部署已回滚为 T19 交付 (7F1B9B24) — 见 ENVIRONMENT

### 7. 与旧报告的差异 (独立复核结论)

| 旧报告结论 | 本次复核 |
|---|---|
| H1 输出 token 永远来自 target | **成立** (代码+行为复核) |
| H2 唯一分叉源 = FA VEC/TILE | **不成立**: 还有 S2 (GDN, 本 fork 引入) 与 S3 (FA split-K padding) |
| GDN/FFN/norm 全部 batch-invariant | **不成立** (S2; 旧 base 上恰好未触发, 数据相关) |
| 修复后 n-max1/2 逐字节一致, n-max3 残余 = rs=3 快照 | **不成立**: 快照/回滚经工具验证是精确的 (rollback 1/2/3 + probe 读回全等); n-max3 残余来自 S3 |
| n-max3 残余与接受率无关, 与回滚深度相关 | **不成立**: 三者分叉点均由近并列 + S3 扰动决定, 与回滚深度无关 |
