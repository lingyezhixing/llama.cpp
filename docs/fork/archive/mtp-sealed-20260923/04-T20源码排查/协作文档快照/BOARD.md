# 任务板 (BOARD) - 唯一入口

最后整理: 2026-09-22 (analyst; T03 V2b 归档后)

> 状态: **T20 DONE; 掉速分析 RUNNING** (用户指派: S3 -31% @d32768 + T19 遗留 tg d131072 -3~-10%);
> 裁决 (A: S1+S2 固化 + S3 可选 / B: 全固化 -31% / C: 不采纳) **待掉速数据后由用户定**。
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
5. **待办顺序**: ~~T19~~ **DONE** -> ~~T20 (MTP 轨迹一致性独立排查)~~ **DONE (暂停等用户裁决: QUESTIONS 2026-09-23)** ->
   外研候选清单 (OPTIONS 第 H 节, 待判断) -> T04 (用户口径) -> 最终统一回顾; 回顾之前不重开 T11/双 stream/silu 等已否决项
6. 工作区 / 补丁 / 构建: 见 **ENVIRONMENT.md** (5 文件未提交, 4 patch;
   部署 DLL `D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll` SHA `453E2911...`, T12+T16 版)
7. 不要重试清单在本文件末尾; MTP 已跳过; ub 归调参类放最后; 验收口径固定 **ub512**

## 已落地优化 (工作区已改 + patch 已归档 + DLL 已部署)

| # | 优化 | 效果 (实测) | 数值影响 | 位置 |
|---|---|---|---|---|
| 1 | Q6_K/Q5_K 向量化反量化 | 反量化 471 -> 707-825 GB/s; pp512 788 -> 950 (该 patch 的历史贡献) | 逐位相同 | `convert.cu` + `dequantize.cuh`; `patches/v100-dequant-vec.patch` |
| 2 | GDN vec4 行布局 (按 n_tokens 分支) + KDA 修复 | kernel 879.9 -> 826.3 us/层 (-6.1%); pp512 +1.0% | PPL 4.3572 -> 4.3569 (1 ulp 重结合, **唯一来源已溯源**) | `gated_delta_net.cu`; `patches/v100-gdn-vec4.patch` |
| 3 | silu float4 向量化 | kernel 11.42 -> 10.84ms (-5.1%); 端到端噪声内 | 逐位相同 | `unary.cu`; `patches/v100-t07-silu-vec4.patch` |
| 4 | FATTN 长文加速: stream-K 启发式 + KV 切分 PB=2 (T12+T16) | **pp8192@depth128k +11.28%** / depth32k +4.77% / pp32768 +2.18%; 短点无回退 | PPL 4.3568 -> 4.3562 (1 ulp 级, 良性) | `fattn-common.cuh`; `patches/v100-t12t16-fattn-split.patch` |

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
| T14 | ub2048 采纳判定 | **DECLINED (Q14)** | 用户不采纳 ub (封存备查); 需要时 0.5h 测量 + 服务器 `-ub 2048` 可启用 |
| T15 | 双 stream 正式 A/B | **DROPPED (Q13)** | 带宽饱和锁死 (dequant 707 + GEMM 150 ≈ 95% 峰值), 实测 +0.5%; 不做 A/B |
| T16 | ub512 attention parallel_blocks | **DONE -> VERIFIED (analyst 2026-09-23)** | mma 内核无 parallel_blocks 路径 (纯 stream-K) -> 等价实现 = 覆盖 blocks_num.x; **PB=2 -14.7% vs T11-off / -6.7% vs T12 默认**; PB=4 过切, 占用/更多 warp 全否证; 验收 (合并) 全过: **128k +11.28%** 等; 与 T12 同一 patch 入库 |
| T17 | Q8_0 权重反量化补测 | **DONE -> CLOSED (不投入)** | 当前内核 767 GB/s (微基准) / ~877 GB/s (实模型折算), pp512 占比 1.2% -> 空间 <=0.3% < 门槛 +0.4%; 向量化候选逐位相同但更慢; `artifacts/t17_q8_bench.cu` |
| T18 | Volta FA mma 内核效率重写 (长文) | **CLOSED (用户 Q16)** | 生产实测: 128k 点 **+0.11%** (门槛 +2% 未达) / depth32k +1.50% / pp32768 +1.17%; nsys 逐 launch 证明长 l 下 ncols=32 反慢 0.3-0.6% (harness 长 l 保真缺口); 已回退, patch `artifacts/t18-ncols32-REJECTED.patch` |
| T19 | 交付定稿 + 全曲线对照 + 质量分析 | **DONE (2026-09-23)** | NEWBASE `e6ab7c1a4` 两边同 base; 交付提交 `afbab1748` (rebase 后, 未 push); pp **+8.7/+9.7/+11.5/+19.6/+35.6%** (512..131072); tg d0-d8192 持平, d32768 复测**持平 (+0.2%, 矩阵值=异常)**; d131072 **存疑 (-3~-10%, 未定论)**; PPL 4.3562 vs 4.3572; 事实型 prompt 200 token 逐字节一致; 图/CSV/脚本在 artifacts; 详见 RESULTS "T19" |
| T20 | MTP 轨迹一致性独立排查 | **DONE (2026-09-23, 暂停等裁决)** | **不盲信旧报告** (独立复现+定位): 3 个分叉源 - S1 FA VEC/TILE (旧报告 H2, 确认); **S2 GDN vec4 布局分支 (本 fork T03/A2 引入, 旧报告漏项)**; **S3 FA VEC split-K padding 边界 (上游设计属性)**; 三源全修后 n-max 1/2/3 与无 spec **逐 token 一致** (短 900 / 32k 150 token); S3 低代价修法 (小批量固定切分) 已实现, 总代价 **-0.5~-0.8%**; PPL 4.3562; 三臂对照: **T20 默认与上游 d0/d8192 均全等** (无更多不一致); 旧报告 5 条主张 4 条被推翻; 详见 RESULTS "T20" 1-9 节 |

## 待测队列 (2026-09-23, 最终接受清单 Q14)

| 序 | 项 | 预期 | 成本 | 状态 |
|---|---|---|---|---|
| 1 | ~~T12+T16 合并验收~~ | 实测 **128k +11.28% / depth32k +4.77% / pp32768 +2.18%** (全过) | 0.5 天 | **DONE -> VERIFIED** |
| 2 | ~~T17 Q8_0 反量化补测~~ | 实测无可利用空间 (767 GB/s 已同档) | 0.5h | **DONE (CLOSED 不投入)** |
| 3 | ~~T16 parallel_blocks~~ | PB=2 采纳, 验收并入序 1 (全过) | 1 天 | **DONE -> VERIFIED** |
| - | ~~T14 ub2048~~ | ~~+21% / +25%~~ | - | **不采纳 (Q14, 封存)** |
| - | ~~T15 / T13~~ | - | - | 放弃 |
| 4 | ~~T18 Volta FA mma 重写~~ | 实测 128k +0.11% (harness 长 l 保真缺口) | - | **CLOSED (用户 Q16: 1% 级不值得)** |
| 5 | ~~T19 全曲线对照 (终期)~~ | 实测 pp +8.7..+35.6%; tg d0-d32768 持平 (**d131072 存疑未定论**) | - | **DONE (ub2048 按用户指示跳过)** |
| 6 | ~~T20 MTP 轨迹一致性独立排查~~ | 实测: 3 源定位 + 全修后逐 token 一致 (短/32k/128k) | - | **DONE (暂停等用户裁决: S3 是否采纳)** |
| 7 | 外研候选清单 (T21-T29) | 见 `OPTIONS` 第 H 节 | - | **待判断 (T20 后统一)** |

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

## 关键判断 (当前口径)

- 当前口径 (T19 @ NEWBASE `e6ab7c1a4`, OURS): pp512 **948.5** / pp4096 929.9 / pp8192 907.5 / pp32768 **793.5** / pp131072 **531.5** / tg128 26.61 / PPL **4.3562**
  (vs STOCK: pp +8.7..+35.6%; 旧 base 的 depth32k 682.2 / 128K 点 375.5 / tg128 26.68 仍为长文参考; **禁止跨 base 混比**)
  (本机热漂移大: 连续负载 tg128 26.6->24.4; 长点必须轮换 A/B 顺序; 禁止跨 session 混比)
- prefill 预算 (532ms): GEMM ~305-349 + dequant 88.5 + GDN 40 + elementwise 35 + 杂项 20
  (GEMM 两项口径: cuBLAS log 一次前向合计 305.4ms vs 差值法 349ms; 差在真实负载时钟 1447MHz)
- **长文验收点 (Q14): pp8192@depth128k = 375.5** (T12+T16 后; +11.28% vs BASE) - 后续长文改动必须给这一项
- ub512 现状 (T19 @ NEWBASE): 短 ~948 / pp32768 793 / pp131072 531; **T20 已结**: MTP 分叉 3 源定位, 三源全修后
  短/32k/128k 逐 token 一致; 代价 S2 ~1% / S3 -31% @d32768 -> 待用户裁决 (是否固化 S1+S2 + S3 可选 env); FA 内核路线已关闭 (T18)
  - GDN 接近实际下限 (~0.8ms, 占预算 7.5%); `__expf` 的 +0.7% 低于门槛, 可在最终回顾复议
  - 长上下文基线 (T12/T16 执行前): depth32k 649-653 / pp32768 775-780; 执行后更新
  - 融合 (T02/T09) 与 cublasLt (T08) 全部关闭; GEMM 侧已无调用空间
  - 1500+ 不可达: 1500 t/s = 341ms/ubatch = 79 TF, 而 GEMM 单独已需 305-349ms (STATUS 有定量证明)
  - ub2048 全面优化上限 ~1250-1300 (当前实测 1152; attention 在 ub2048 已 39.9 TF/s)
- decode: 纯 kernel 天花板 **27.3-27.5** (T05 结案, evidence-based; 现状 26.6); 有效翻倍由用户 MTP3 承担 (42.7)
