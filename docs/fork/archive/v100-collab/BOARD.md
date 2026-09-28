# 任务板 (BOARD) - 唯一入口

最后整理: 2026-09-27 (analyst; T32 S1 实现+冒烟后)

> 状态: **T19-L DONE; T24 DONE + 已部署 (单提交 `3eae5cdae` 已推 fork origin; 生产 `llama.cpp-my` = `299DAFC7` T24 构建 + `GGML_CUDA_GDN_REPLAY=1`; 备份 `deploy-backup\llama.cpp-my-t19l-20260926` = T19-L `054BFFD6`; T24 待办仅剩"纳入交付基线")**; **T31 PARKED (用户 2026-09-27: Phase A + 复核完成; 收益上限有限 = B2 ~+5% / 内核无已证路径 -> 转 T32; B2 延后)**; **T32 RUNNING (S1 改动已按用户指示放弃回滚 [快照 artifacts/t32-s1-worktree.patch]; S2a [VRAM fork] 已停; 需求三条: ①剥思考后重算 ②RAM 淘汰 ③树状 RAM+SSD 存储, 待重写方案)**; **T30 REJECTED (verify VEC 慢 3.1x; 128K 接受率 0.388 -> T31-B)**; **T25 REJECTED**;
> T21/T22 用户明确否 (2026-09-26); 投机组合 (first-wins 回退链) 不采纳; T29 并入 T31-C; T14 延后 (先集中 T31); T20 已结 (不采纳, 已回滚/封存); d131072 已降级; 其余候选 T23/T26-T28 (`OPTIONS` §H) 待判断。
> T20 裁决依据: "开 MTP 逐位透明" 与 "no-MTP 与原版逐位一致" 在 FA 切分数 (pb) 上互斥, 保留加速则必然与原版不一致;
> 实验 patch 与全部材料 (含四臂群测结果) 封存 `D:\LLM\Backend\MTP封存-2026-09-23`, 工作区已回滚 (当时 = T19 交付 `afbab1748`; 现演进为 T24 `3eae5cdae` 部署)。
> 要点: 3 个分叉源 (S1 FA VEC/TILE + **S2 GDN vec4, 本 fork A2 引入** + **S3 FA split-K padding, 上游**);
> 三源全修后 短900/32k150/128k100 token 与无 spec 逐 token 一致; 详见 RESULTS "T20" 与 QUESTIONS 末条 analyst note。

## 恢复指引 (上下文压缩后先读这里 -> STATUS.md -> 对应 TASKS 文件)

1. **融合与 cuBLAS 调用侧全部关闭 (用户 2026-09-22 最终定死, ARCHIVE Q8)**: T02/T09 (dequant 融合) + T08 (cublasLt 真实方向 **-1.11%**)。
   不要再试 algo hint / workspace / math mode (均已证伪, 见不要重试清单与 ARCHIVE 第 9 节)
2. **外部参考**: REFERENCE-1cat-vllm.md (1Cat-vLLM, 4xV100 跑 Qwen3.8-27B-FP8): 独立验证融合否决;
   启示 = 大 M/chunked prefill 是第一杠杆 (支持 T04) + 长上下文 attention 已侦察并关闭 (T10/T11)
3. **T10 + 近路 (T11) 全部关闭**: 侦察结论 = attention "慢" 的真因是 ub512 grid/波次
   (grid 恒 192 / 80 SM / 3 波), **不是 kernel 效率** (ub2048 同 kernel 39.9 TF/s >= 1Cat 29-38)
   -> **不移植 split-D/N32**。近路 KV-split 按 Q6 严格验收 = **FAIL 并已回退**:
   nsys 机制成立 (grid 192->80, attention -5.6%, 29.5->31.3 TF/s) 但端到端 depth32k **+2.0%** /
   pp32768 **+1.1%** (门槛 3%/2%); 波次效率模型高估 (每 block 串行 ~2.4 tile + 接缝归并抵消)。
   详见 RESULTS "T10 近路", patch 在 artifacts
4. **T07 全关闭**: silu 已入库; rms_norm 向量化 REJECTED (kernel 合计 +5.7% 更慢, 端到端噪声地板,
   理论上限 0.1-0.2%); 未入库, patch 仅在 artifacts
5. **待办顺序**: ~~T19~~ **DONE** -> ~~T20~~ **DONE -> 未采纳 (2026-09-23)** -> ~~T19-L 瘦身~~ **DONE (2026-09-24, `d24474edd`)** ->
   **T24 (ReplaySSM) DONE** - 单提交 **`3eae5cdae`** (四提交 squash), **已推 fork origin/master** (analyst 核实); 判据达成: d0 1000token 老==新; MTP3 ON==OFF==base; PPL 4.3567 两边一致; 自检 0 mismatch; 双模型矩阵 (Qwen3.8 MTP n-max 1/2/3 x np 1..4 + Qwen3.6 DFlash n-max 6 x np 1..4, 含生产采样/并发/长程/save-restore) 全逐 token 一致; VRAM np=1 -420MiB / np=4 -1.37GiB; 代价 tg -1.9% (权衡, 开关默认关 = T19-L 零风险); 两个根因 + 3 处扩面发现 (per-seq 记账/回滚边界/自检假阳性) 全修;    最终构建 = 生产部署 (2026-09-26, `299DAFC748DE5D91` + `GGML_CUDA_GDN_REPLAY=1`; 备份 t19l 完好); 未覆盖: EAGLE3/DSpark/KDA/np>=5; **T24 仅剩纳入交付基线**; 当前 = **T32 (S1 已验证待提交 + S2a 冒烟门通过; 待决 S1.5/S2a/模板/T24 复查)**; **T31 PARKED**; T30 REJECTED (verify VEC 慢 3.1x); **T14 延后**; 之后 = 纳入交付基线 -> T04 (用户口径) -> 最终统一回顾; 回顾之前不重开 T11/双 stream/silu 等已否决项
6. 工作区 / 补丁 / 构建: 见 **ENVIRONMENT.md** (交付提交 `d24474edd` (squash 后单条) = A1 + A4 + sm70 调参;
   2 patch 在 artifacts; 部署 DLL `D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll` SHA256-16 `299DAFC7...` (T24 + `GGML_CUDA_GDN_REPLAY=1`))
7. 不要重试清单在本文件末尾; **当前主线 = T32 (S1 改动已放弃回滚; 需求三条待重写方案: ①剥思考后重算 ②RAM 淘汰 ③树状 RAM+SSD 存储; T24 replay 复查待立案)**; T31 PARKED (B2 延后, 内核不立); ub (T14) 延后; 验收口径固定 **ub512**

## 已落地优化 (工作区已改 + patch 已归档 + DLL 已部署)

| # | 优化 | 效果 (实测) | 数值影响 | 位置 |
|---|---|---|---|---|
| 1 | Q6_K/Q5_K 向量化反量化 | 反量化 471 -> 707-825 GB/s; pp512 788 -> 950 (该 patch 的历史贡献) | 逐位相同 | `convert.cu` + `dequantize.cuh`; `artifacts/v100-dequant-vec.patch` |
| 2 | ~~GDN vec4 行布局 (按 n_tokens 分支) + KDA 修复~~ | kernel 879.9 -> 826.3 us/层 (-6.1%); pp512 +1.0% | PPL 4.3572 -> 4.3569 (1 ulp 重结合, **唯一来源已溯源**) | `gated_delta_net.cu`; **已退役 (T19-L 2026-09-24; patch 在 `artifacts/retired/`)** |
| 3 | ~~silu float4 向量化~~ | kernel 11.42 -> 10.84ms (-5.1%); 端到端噪声内 | 逐位相同 | `unary.cu`; **已退役 (T19-L 2026-09-24; patch 在 `artifacts/retired/`)** |
| 4 | FATTN 长文加速: stream-K 启发式 + KV 切分 PB=2 (T12+T16) | **pp8192@depth128k +11.28%** / depth32k +4.77% / pp32768 +2.18%; 短点无回退 | PPL 4.3568 -> 4.3562 (1 ulp 级, 良性) | `fattn-common.cuh`; `artifacts/v100-t12t16-fattn-split.patch` |

当前验收 (T12+T16 入库后, 同 session A/B 的 B 值; 本机热漂移大, 绝对值仅同 session 可比):
pp512 **950.8** / pp4096 932.5 / pp8192 912.1 / pp32768 **791.1** / depth32k **682.2** / **pp8192@depth128k 375.5** / tg128 26.68 / PPL **4.3562**

## 任务表

| ID | 任务 | 状态 | 结果/结论 |
|---|---|---|---|
| T01 | cuBLAS GEMM 效率核查 | **VERIFIED (矛盾已解决)** | 墙 84-107 TF 成立; "调用侧无空间"成立 (T08 Step 1 复核); T02 的 +9% = 转置方向伪影 |
| T02 | dequant + fp16 MMA 融合 GEMM | **REJECTED (Gate A)** | 骨架 57.5 TF (<80); **mma 路径单独 1.472ms 已 > baseline 1.404ms** -> 融合无理论空间; 详见 RESULTS |
| T03 | gated_delta_net 优化 | **CLOSED (NO PATH)** | vec4 已入库; chunked V1-V3 全线实测否决 (FLOP 1.4-2.4x + 效率天花板); 真实规模 (H=48) 排查后判定 GDN 近实际下限 (~0.8ms); `__expf` (-10% kernel / e2e +0.7%) 低于门槛记档 |
| T04 | ub/服务端配置固化 | PROPOSED (调参类, 放最后) | ub2048 = 1152-1157 t/s (+21%); 1Cat 架构佐证 (大 M chunked prefill) |
| T05 | decode profile | **CLOSED (剩余项无实质收益)** | profile VERIFIED 后, 剩余项实测结案: 稳态 MMVQ 29.84ms/86.9% (逐矩阵 685-845 GB/s 已饱和); **关融合对照 tg128 仅 -2.6% (= 边际 0.9-1.0us/kernel)**; 剩余上限 +2.7% (tg128 27.3-27.5); 28.5-30 需核心级 kernel 合并; rms_norm 配置实验无效已回退 |
| T06 | MTP 投机解码 | **SKIPPED (用户)** | 生产 draft-mtp n-max 3: 25.3 -> 42.7 t/s |
| T07 | elementwise 向量化 | **DONE (全部关闭)** | silu kernel -5.1% 已入库; **rms_norm 向量化 REJECTED** (kernel 级 +5.7% 更慢, 理论上限 0.1-0.2% < 噪声) |
| T08 | cublasLt 集成 (per-shape) | **Step 1 DONE -> REJECTED (VERIFIED)** | 真实方向逐形状 ABAB (8 形状, 5 轮交错): 加权 **-1.11%** of GEMM; gate/up -2.9%; llama.cpp 0 行改动 |
| T09 | 融合 2.0 验证 (量化 operand 版) | **REJECTED / CLOSED (VERIFIED)** | 三条独立否决: 去解包 1.569ms > baseline 1.404ms; 解包 18-30% 不可重叠; BM=512 物理不可能 |
| T10 | 长上下文 attention 侦察 (只测量) | **DONE -> VERIFIED (analyst 复核)** | 占比: d0 3.3% / pp32768 18.5% / depth32k 32.0%; ub512 29.5 TF/s, **ub2048 39.9 TF/s** (grid 192->768) -> 不移植 split-D/N32 |
| T11 | ub512 FA KV-split (T10 近路) | **REJECTED -> Q10 采纳 (T12)** | 原裁决: 端到端 depth32k +2.0% / pp32768 +1.1% < 门槛 3%/2% (attention -5.6% 真实); **用户 Q10 按"正提升+显存小"采纳, 由 T12 重放集成** |
| T12 | 采纳 T11: KV-split 重放集成 | **DONE -> VERIFIED (analyst 2026-09-23)** | 合并验收全过: **pp8192@depth128k +11.28%** / depth32k +4.77% / pp32768 +2.18% / 短点 +0.03~0.56% / tg128 +0.06% / ub2048 -0.1% / PPL 4.3562 / 生成 OK; patch `v100-t12t16-fattn-split.patch` |
| T13 | `__expf` (GDN) 集成 | **CANCELLED (Q12)** | 用户砍掉 (保 100% 零漂移; 代价 +0.7%); 规格保留 TASKS/T13 |
| T14 | ub2048 采纳判定 | **延后 (用户 2026-09-26: 先集中 T31)** | 曾 Q14 不采纳; 0.5h 测量 + 服务器 `-ub 2048` 可启用 (预期 +21% / 128K +25%; 验收口径仍 ub512) |
| T15 | 双 stream 正式 A/B | **DROPPED (Q13)** | 带宽饱和锁死 (dequant 707 + GEMM 150 ≈ 95% 峰值), 实测 +0.5%; 不做 A/B |
| T16 | ub512 attention parallel_blocks | **DONE -> VERIFIED (analyst 2026-09-23)** | mma 内核无 parallel_blocks 路径 (纯 stream-K) -> 等价实现 = 覆盖 blocks_num.x; **PB=2 -14.7% vs T11-off / -6.7% vs T12 默认**; PB=4 过切, 占用/更多 warp 全否证; 验收 (合并) 全过: **128k +11.28%** 等; 与 T12 同一 patch 入库 |
| T17 | Q8_0 权重反量化补测 | **DONE -> CLOSED (不投入)** | 当前内核 767 GB/s (微基准) / ~877 GB/s (实模型折算), pp512 占比 1.2% -> 空间 <=0.3% < 门槛 +0.4%; 向量化候选逐位相同但更慢; `artifacts/t17_q8_bench.cu` |
| T18 | Volta FA mma 内核效率重写 (长文) | **CLOSED (用户 Q16)** | 生产实测: 128k 点 **+0.11%** (门槛 +2% 未达) / depth32k +1.50% / pp32768 +1.17%; nsys 逐 launch 证明长 l 下 ncols=32 反慢 0.3-0.6% (harness 长 l 保真缺口); 已回退, patch `artifacts/t18-ncols32-REJECTED.patch` |
| T19 | 交付定稿 + 全曲线对照 + 质量分析 | **DONE (2026-09-23)** | NEWBASE `e6ab7c1a4` 两边同 base; 交付提交 `afbab1748` (rebase 后; 后并入 `d24474edd` 随 T24 推送); pp **+8.7/+9.7/+11.5/+19.6/+35.6%** (512..131072); tg d0-d8192 持平, d32768 复测**持平 (+0.2%, 矩阵值=异常)**; d131072 **存疑 (-3~-10%, 未定论)**; PPL 4.3562 vs 4.3572; 事实型 prompt 200 token 逐字节一致; 图/CSV/脚本在 artifacts; 详见 RESULTS "T19" |
| T21 | MTP K 扫描 + greedy drafting 检查 | **降级 (用户 2026-09-23, 暂不做)** | T19 交付口径; none/n-max1/2/3(/4) x d0/d32k/d128k 速度+接受率+显存; drafter 采样模式检查; 给生产 K 建议; 详见 `TASKS/T21-mtp-k-sweep.md` |
| T22 | 内置 ngram 查表投机实测 | **降级 (用户 2026-09-23, 暂不做)** | 零代码 `--spec-type ngram-map-k4v`; 复述/代码/普通 prose x 短/长文; 与 MTP 组合可行性; greedy 无损核对; 详见 `TASKS/T22-ngram-lookup.md` |
| T19-L | 交付瘦身 (丢 A2/A3, 保 A1/A4) | **DONE (2026-09-24)** | 已入库 `d24474edd` (squash 后; 随 T24 推送 = `3eae5cdae` 祖先); 验收: pp512 **-1.5%** (948.1 ≈ 记录 948.5) / pp32768 -0.9% / pp131072 -0.5% / tg 短点持平 / **PPL 4.3567** / MTP3 无回退; patch 2 个 + 退役 2 个; 详见 RESULTS "T19-L" |
| T25 | GDN recurrent state fp16 | **REJECTED (用户 2026-09-23: 降低精度不采纳)** | state fp32->fp16 为有损存储 (违反质量红线); 存档 `TASKS/T25-gdn-state-fp16.md` |
| T24 | ReplaySSM (GDN 投机状态 raw-input 重放) | **DONE (单提交 `3eae5cdae`, 已推 fork origin/master)**: 两根因 + per-seq 记账 + 状态存取 (SEQ_VERSION 5); 双模型矩阵全过 (MTP n-max 1/2/3 x np 1..4; DFlash n-max 6 x np 1..4 含生产采样; 并发/长程/save-restore; PPL 4.3567 两边一致; 自检 0 mismatch) | 权衡: VRAM np=1 -420MiB / np=4 -1.37GiB, 代价 tg -1.9% (71.6->70.2); **已部署 (2026-09-26: `299DAFC7` + `GGML_CUDA_GDN_REPLAY=1`; 备份 `deploy-backup\llama.cpp-my-t19l-20260926`)**; 待办仅剩: 纳入交付基线; 未覆盖: EAGLE3/DSpark/KDA/np>=5 |
| T30 | MTP 长文提速 (验证路径 TILE->VEC 为主候选) | **REJECTED (2026-09-26, Phase A 实测, 未进 Phase B/C)** | VEC n_q=4/l128K **慢 3.1x** (4.68 vs 1.50 ms/层, nsys 直测); V 物化仅占 ~10% (去掉也不划算); PB/NBATCH 13->160 慢 15%; PPL 控制位逐位不变; 128K 接受率 0.388 (vs d0 0.808 = 另一半原因, 属 T31-B); 备选未做: TILE 直读 q8_0 V (上限 ~10%, 需新内核); 实验已回滚 |
| T31 | MTP 轮内开销 (128K) | **PARKED (用户 2026-09-27: ROI 不足, 转 T32; B2 延后)** | Phase A + 复核完成: 轮 ~112-115ms = verify+host **90ms (78%)** + draft 17.9ms + 采样 6.8ms + accept ~0; 原 "未解释 ~50ms" = verify 本体 (H2/H3 否决); **口径修正: 生产采样 (temp0.6) = 27.0-27.6 tps (acc 0.694), greedy 19-20; 无 spec = 12.9 tps -> MTP3 2.1x**; K=3 最优 (MTP6 prod 20.2); 剩余 = B2 ~+5% (+1.4 tps) / V 直读 ~+7% / FA 无已证路径 -> **全包 ~+12% (31 tps) 封顶**; B2 留作廉价待办; 复核见 `RESULTS "T31 Phase A - analyst 复核"` |
| T32 | agent/长会话复用专题 | **RUNNING (S1 改动已放弃回滚; S2a 已停; 待重写方案)** | **2026-09-27 用户指示放弃全部未提交修改 (已回滚; 快照 `artifacts/t32-s1-worktree.patch` + temp `t32-worktree-dropped-20260927.patch`); 以下为历史记录**: **S1** (patch `t32-s1-worktree.patch` 39.7KB @3eae5cdae, 未提交): A pin 锚点 / B max-LCP+去 0.25 / D park 保护 + blob 检查点裁剪; 实测: T32-1 恢复 D-516 -> D-4 (3274->2762 tok); **T32-2 轮换 5.4-5.7K tok/6.7s -> 71 tok/0.6s (~12x TTFT)**; A22+B8 请求逐位一致; blob 1.9G -> 1.02G。**S2a 冒烟门通过** (共享前缀/refcount/分支重建逐位一致; 27B 每活分支 +220MiB; ~190MiB = GDN 状态不可共享)。**新发现: T24 replay 回滚运行间不确定性** (ref-vs-ref 1/6 value-diff; replay=0 干净; token 翻转仅近并列) -> 待立案。np=2 更正: unified KV 不自动共享前缀 (B 全量 prefill); `-np 2`+`id_slot` 常驻零代码可用。T32-1 源头 = 用户模板 (待用户修)。**S2a (VRAM 活分支 fork) 停止**: 用户 2026-09-27 澄清需求不含 VRAM 内复用 (analyst 误读), 产物仅归档 (含 COW/`--kv-unified`/数值不一致发现); **子问题 3 正主 = 树状 RAM+SSD 存储** (整序列 blob -> 节点级: 枝干优先 RAM / 溢出先叶 / 从槽位移除先上树-不能则新建树 / 新序列从树上复用), 待重写设计。待决: 提交 / S1.5 瘦身 / 树状存储设计 / T24 复查; spec `TASKS/T32-agent-session-reuse.md` |

## 待测队列 (2026-09-23, 最终接受清单 Q14)

| 序 | 项 | 预期 | 成本 | 状态 |
|---|---|---|---|---|
| 1 | ~~T12+T16 合并验收~~ | 实测 **128k +11.28% / depth32k +4.77% / pp32768 +2.18%** (全过) | 0.5 天 | **DONE -> VERIFIED** |
| 2 | ~~T17 Q8_0 反量化补测~~ | 实测无可利用空间 (767 GB/s 已同档) | 0.5h | **DONE (CLOSED 不投入)** |
| 3 | ~~T16 parallel_blocks~~ | PB=2 采纳, 验收并入序 1 (全过) | 1 天 | **DONE -> VERIFIED** |
| - | T14 ub2048 | +21% / +25% | 0.5h | **延后 (2026-09-26: 先集中 T31); 曾 Q14 不采纳** |
| - | ~~T15 / T13~~ | - | - | 放弃 |
| 4 | ~~T18 Volta FA mma 重写~~ | 实测 128k +0.11% (harness 长 l 保真缺口) | - | **CLOSED (用户 Q16: 1% 级不值得)** |
| 5 | ~~T19 全曲线对照 (终期)~~ | 实测 pp +8.7..+35.6%; tg d0-d32768 持平 (**d131072 存疑未定论**) | - | **DONE (ub2048 按用户指示跳过)** |
| 6 | ~~T20 MTP 轨迹一致性独立排查~~ | 实测: 3 源定位 + 全修后逐 token 一致 (短/32k/128k) | - | **DONE -> 未采纳 (用户 2026-09-23; 已回滚)** |
| 7 | 外研候选清单 (T21-T31) | **T24 DONE + 已部署; T31 PARKED (2026-09-27 ROI 不足); T30 REJECTED; T25 REJECTED; T21/T22 用户明确否; T29 并入 T31-C (随 T31 PARKED)**; T23/T26-T28 待判断; **T32 S1 待提交 / S2a 冒烟门通过 / S1.5 + T24 replay 复查候选** | - | **执行中** |
| 10 | ~~**T19-L 交付瘦身** (用户批准)~~ | 实测 pp512 -1.5% / 长点 -0.5~0.9% / PPL 4.3567 / MTP3 无回退 | ~1.5h | **DONE (2026-09-24)** |
| 8 | T19 遗留: tg d131072 -3~-10% 定论 | 4-6 轮交替 A/B (~30min) | 0.5h | **待判断 (可选)** |
| 9 | T04 (ub 配置) / 最终统一回顾 (Q7) | - | - | **待判断** |

## 不要重试 (已用数据证伪)

| 实验 | 结果 |
|---|---|
| 融合 dequant + fp16 MMA (T02) | Gate A 否决: mma 路径单独就慢于 baseline; dequant 指令还要吃 25% issue 预算 |
| 融合 Q6_K operand (T09-A) | 实测 37-41 TF; 去解包也仅 58 TF > baseline 1.404ms -> 彻底关闭 |
| 双 stream 反量化重叠 | +0.5% (带宽饱和) |
| FORCE_MMQ (dp4a) | pp512 -41% |
| fp16 累加 (compute=16F) | gate/up 63.7 vs 84.2 TF |
| cuBLAS workspace 4MB->256MB / 显式 algo 112 | 无增益 / -51% |
| 全局 algo hint (ALGO8 套进 llama.cpp) | pp512 -68% (形状相关) |
| 显式 algo 逐形状 (ALGO0..15, 真实方向) | 全部慢于 DEFAULT (gate/up 最好 62.6 vs 82.1 TF) |
| cublasLt 替换默认 (真实方向, 8 形状) | 加权 -1.11% (gate/up -2.9%); "+9%" 是转置方向伪影 |
| MMVQ nwarps=2/8, rows_per_block=2/4 | -2.6% / -9.6% / +0.3%(噪声) / -4.0% |
| rms_norm 向量化 (T07 剩余) | kernel 合计 174.81 -> 184.85ms (+5.7% 更慢); 1024 变体已 930GB/s DRAM 极限 |
| FA split-D/N32 移植 (T10) | 不移植: ub2048 同 kernel 已 39.9 TF/s (超 1Cat 29-38); 真问题是 ub512 grid 调度 |
| ub512 FA KV-split (T11) - 已移出 | 原 <门槛 回退; **用户 Q10 采纳** (depth32k +2.0% 真收益) -> T12 重放集成 |
| GDN chunked (T03, V1-V3 + 真实规模全线排查) | 算法正确 (1e-16/5e-7) 但 4.60/3.29/1.644/1.844ms 均 >> 现状 0.826; FLOP 多 1.4-2.4x; 非访存/非占用瓶颈 | H=48 已近实际下限 -> **CLOSED (NO PATH)** |
| rms_norm block 配置 (1024->256, T05) | 无效果 (tg128 两侧一致) | 已回退 |
| decode 侧继续融合/合并 (T05 剩余) | 关融合对照: 533 融合头总共只值 0.70ms (边际 0.9-1.0us/kernel); MMVQ 逐矩阵 685-845 GB/s (lm_head 94-99%) | 剩余上限 1.0-1.1ms (+2.7-3.0%) -> CLOSED; 28.5-30 需核心级合并 |
| verify FA TILE->VEC (T30) | VEC n_q=4/128K 慢 **3.1x** (4.68 vs 1.50 ms/层; TILE 共享 K/V tile, VEC cols_per_block=2/13 warp); V 物化仅占 10% -> 换核不划算 |
| verify FA KV 切分加大 (T30: PB/NBATCH 13->160) | -15% (setup/归并/尾波 > 并行收益); VEC+PB 组合 -36% |
| draft K 节流 (T31 Phase A, 零代码探针) | MTP1 greedy 16.3-19.0 / MTP6 prod 20.2 均 < MTP3 (greedy 19-20 / prod 27.0-27.6) -> **保持 K=3** (固定开销按轮摊薄; MTP6 draft 35ms + verify n_q=7 涨 31ms) |
| draft 窗口化 (B1, 分析否决) | draft 17.9ms 中可动仅 FA ~2-5ms -> 上限 +2~4%; 且改训练分布, 有接受率风险 |

## 关键判断 (当前口径)

- 当前口径 (T19-L @ NEWBASE `e6ab7c1a4`, OURS, 2026-09-24): pp512 **948.1** / pp32768 **790.5** / pp131072 **538.0** / tg128 26.66 / PPL **4.3567**
  (同 session T19: 962.8 / 797.5 / 540.5 / 26.66 / 4.3562 -> T19-L 代价 = pp512 -1.5%, 长点 -0.5~0.9%; 绝对值 948.1 ≈ 记录 948.5)
  (vs STOCK: 长文 +35.6% 保持, 短点 ~+19% 级; 旧 base 数字 (948.5/793.5/531.5) 仅历史参考; **禁止跨 base/session 混比**)
  (本机热漂移大: 连续负载 tg128 26.6->24.4; 长点必须轮换 A/B 顺序)
- prefill 预算 (532ms): GEMM ~305-349 + dequant 88.5 + GDN 40 + elementwise 35 + 杂项 20
  (GEMM 两项口径: cuBLAS log 一次前向合计 305.4ms vs 差值法 349ms; 差在真实负载时钟 1447MHz)
- **长文验收点 (Q14): pp8192@depth128k = 375.5** (T12+T16 后; +11.28% vs BASE) - 后续长文改动必须给这一项
- ub512 现状 (T19 @ NEWBASE): 短 ~948 / pp32768 793 / pp131072 531; **T20 已结**: MTP 分叉 3 源定位, 三源全修后
  短/32k/128k 逐 token 一致; 代价 S2 ~1% / S3 -31% @d32768 -> **裁决 = 不采纳, 已回滚** (材料封存); FA 内核路线已关闭 (T18)
  - GDN 接近实际下限 (~0.8ms, 占预算 7.5%); `__expf` 的 +0.7% 低于门槛, 可在最终回顾复议
  - 长上下文基线 (T12/T16 执行前): depth32k 649-653 / pp32768 775-780; 执行后更新
  - 融合 (T02/T09) 与 cublasLt (T08) 全部关闭; GEMM 侧已无调用空间
  - 1500+ 不可达: 1500 t/s = 341ms/ubatch = 79 TF, 而 GEMM 单独已需 305-349ms (STATUS 有定量证明)
  - ub2048 全面优化上限 ~1250-1300 (当前实测 1152; attention 在 ub2048 已 39.9 TF/s)
- decode: 纯 kernel 天花板 **27.3-27.5** (T05 结案, evidence-based; 现状 26.6); 有效加速由 MTP3 承担 (**新基线 @replay=1: d0 50.1 / d32768 30.8 / d131072 19.5-20.1 t/s; 接受率 0.808@d0 -> 0.388@128K**); T30 已否决 (verify VEC 慢 3.1x); T31 Phase A 完成 (复核通过): 生产采样 27.6 tps (口径修正), B1/B2 ≤ +10%, 内核路径待裁决
