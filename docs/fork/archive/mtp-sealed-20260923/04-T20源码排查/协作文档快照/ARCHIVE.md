# ARCHIVE: 决策记录、预测修订链、已证伪实验

最后整理: 2026-09-22 (analyst)。当前有效状态见 STATUS.md, 原始日志见 RESULTS.md。

## 1. 用户决策记录

| # | 日期 | 决策 |
|---|---|---|
| Q1 | 2026-09-22 | 验收口径固定 **ub512**; 增大 ub 归为调参收益, 放最后做 (T04 降为 P2) |
| Q2 | 2026-09-22 | 允许微小不可避免的数值漂移, 门槛 `|PPL - 4.3572| <= 0.013` + 生成质量检查; 不接受降智 |
| Q3 | 2026-09-22 | V100 独占, 无资源竞争 |
| Q4 | 2026-09-22 | **MTP (T06) 跳过**: 生产已用 `--spec-type draft-mtp --spec-draft-n-max 3` (25.3 -> 42.7 t/s), 属调参类, 不再测试 |
| Q5 | 2026-09-22 | 目标"发挥硬件 80%"; 允许以调参兜底; 协作交给我方与实现方直接进行 |
| Q7 | 2026-09-22 | **最终统一回顾制**: 等所有任务 (T03 chunked / 被批准的 T05) 结束后, 统一回顾排查, 届时再决定各方向保留/拒绝 (含 T11 KV-split 是否重开、双 stream 是否做正式 A/B、silu float4 是否保留); **在此之前不重开任何已否决项** |
| Q8 | 2026-09-22 | **融合路线彻底定死不可用** (用户最终确认): 含 MMQ 式 / 自定义 fp16 mma / 量化 operand / CUTLASS mainloop 全部分支, 不再讨论 |
| Q9 | 2026-09-22 | **T03 追加 0.5 天** (用户): 批准结构性重写 (寄存器分块 + 分块三角求解), 到点无条件归档 |
| Q10 | 2026-09-23 | **采纳所有正提升且显存变化小的方案** (用户): A1-A3 保留; B2 KV-split -> T12 重放集成; B3 `__expf` -> T13 集成; B1 ub2048 -> T14 条件采纳 (显存 <=1GB 自动采纳); B4 双 stream -> T15 先正式 A/B; C2 (MTP 词表切片) 待立项; D1 (CUTLASS) 随 B1 之后再评估 |
| Q11 | 2026-09-23 | **ub2048 先不测** (用户): T14 暂缓; B1/T04 保持未采纳; D1 (CUTLASS) 随之一并延后 |
| Q12 | 2026-09-23 | **`__expf` 砍掉** (用户): T13 取消, 保 100% 零数值漂移; 代价仅 +0.7% |
| Q13 | 2026-09-23 | **用户常用上下文 128-150K** -> 据此改判 (analyst): T12 接受 (128K ~+2.7%); **T14/ub 重开并强烈建议** (128K +25% 级, 配置级; 先测 150K 显存); 新增 T16 (parallel_blocks, ub512 补偿, 待 T14 显存结论); T15 抛弃 (带宽饱和 +0.5%); T13/Q8_0 放弃 |
| Q14 | 2026-09-23 | **最终接受清单 (用户)**: A1 + A2 GDN vec4 + A3 silu float4 + **T12 KV-split** + **T17 Q8_0 反量化补测** + **T16 parallel_blocks**; **ub2048 (T14) 不采纳** (封存备查); T15/T13 放弃; **长文验收点 = pp8192@depth128k (128K 深度 8k pp, 不做完整 pp128k)** |
| Q15 | 2026-09-23 | **T18 放宽** (用户): 多给时间继续探索; 无法突破则集成目前发现的最好变体; **只要正收益 >2% 就合并** (analyst 解释为主判据 pp8192@depth128k e2e > +2%) |
| Q16 | 2026-09-23 | **T18 撤回归档** (用户): 生产长文点仅 +0.1% (1% 级), 不值得牵动 attention 改动 -> 撤销两处源码, 交付 DLL 保持 `453E2911` |

## 2. 预测修订链 (prefill, ub512)

| 阶段 | 预测 | 被推翻的乐观假设 |
|---|---|---|
| 初始 (无实测) | 1920 (80% MFU) | 假设 cuBLAS 80-85% 效率; 假设 dequant 占 25-30% 且可全免 |
| T01 后 | 全优化 1250-1400 | cuBLAS 实测加权 ~90 TF (72%), 调用侧无空间 |
| T02 v1 + T05 后 | **1100-1200** | dequant 已向量化到 92% 实际带宽 (88.5ms = 16.6%, 原本 127-213ms), 可回收部分变小 |
| T08/T09 后 (终版) | 成功 ~1000 (ub512) / 1100+ 需 ub | 融合与 cuBLAS 调用侧全部关闭; 1500 不可达 (定量证明见 STATUS) |
| 最终口径 | 见 STATUS | 80% MFU 需 GEMM 151 TF = 硬件的 121%, 数学上不可能 |

## 3. 已证伪实验 (详情)

| 实验 | 数据 | 结论 |
|---|---|---|
| cuBLAS "75.6 TFLOPS (60%)" | 预热后 ABAB 实测 84.2-107 TF | 旧值是 ncu 锁频 1246MHz + 首轮冷状态伪影 |
| workspace 4MB -> 256MB | down +1.1%, gate/up +0.1% | 无增益; fork 原本就有 4MB |
| cublasLt heuristic / 显式 algo | 同值 / algo 112 仅 41 TF | 无增益 / 更差 |
| fp16 累加 (compute=16F) | gate/up 63.7 vs 84.2 | 更慢, 数值路径不需要动 |
| 双 stream 反量化重叠 | +0.5% (微基准重叠率 2.05x) | 带宽饱和 (dequant 707 + GEMM ~150 ≈ 95% 峰值) |
| FORCE_MMQ (dp4a) | pp512 557 (-41%) | V100 无 int8 TC, dp4a 路线排除 |
| MMVQ nwarps / rows_per_block | 2: -2.6%, 8: -9.6%, rows2: +0.3%, rows4: -4.0% | 现行 (nwarps=4, rows=1) 已最优, 参数空间穷尽 |
| T02 融合 kernel v1 | 9.9 TF, occupancy 6.25%, 三阶段串行 | 设计不可行; 教训: BM>=128, BN>=64-128, smem<=48KB, BKP=BK+8 padding, A 走寄存器 fragment |
| T07 前提 "elementwise 58-70% BW" | silu 实为 ~1.05 TB/s (L2 掩护), convert 已 vec4 | 收益下调到 +0.5-1.0% |
| rms_norm 向量化 (T07 剩余) | kernel 合计 174.81 -> 184.85ms (**+5.7% 更慢**); 端到端噪声地板 | 1024 变体已 930GB/s DRAM 极限, 256 变体延迟受限 (57GB/s); 上限 0.1-0.2% -> REJECTED |
| FA split-D/N32 移植 (T10) | ub2048 同 kernel 已 39.9 TF/s (超 1Cat 29-38); ub512 29.5 是 grid 192/80SM/3 波打尾 | 不移植; 真问题是 ub512 调度并行度, 近路 = KV-split |
| ub512 FA KV-split (T11, stream-K 启发式) | attention -5.6% (1899.4->1792.7ms, 29.5->31.3 TF/s), grid 192->80 + fixup 出现; 端到端 depth32k +2.0% / pp32768 +1.1% | 波次效率模型高估 (每 block 串行 ~2.4 tile + 接缝归并抵消); 不达门槛 3%/2% -> 回退; ub2048 优势 = 总并行度 + 更少 launch |
| GDN chunked 重写 (T03) | 算法 1e-16 / V1-V3 正确性全过; 速度 4.60 -> 3.29 -> 1.644 -> 1.844ms (均 >> 现状 0.826); FLOP 多 1.4-2.4x | 真实规模 (H=48) 全线实测: 非访存延迟 (prefetch/staging 更慢)、非占用 (MB=12 spill)、warp 总数抵消; 过门槛需 ~30x -> **CLOSED (NO PATH)**; `__expf` (-10% kernel) 低于门槛记档 |
| decode 侧继续融合/合并 (T05 剩余) | 关融合对照: 533 个融合头总共只值 0.70ms (边际 0.9-1.0us/kernel); MMVQ 逐矩阵 685-845 GB/s (lm_head 94-99%) 已饱和 | 剩余上限 1.0-1.1ms (+2.7-3.0%) -> **CLOSED**; decode kernel 天花板下调为 27.3-27.5 |
| Q8_0 反量化补测 (T17) | 当前内核 767.2 GB/s (与 Q6_K/Q5_K 参考线同档), 向量化候选 669.8 GB/s 更慢; 实模型折算 ~877 GB/s (已到带宽) | 空间 <=0.3% < 门槛 +0.4% -> **CLOSED** (记录数字) |
| T18 Volta FA mma 重写 | harness 1.12x **长文不转移**: 生产 A/B 128k 点 **+0.11%** (门槛 +2% 未达) / depth32k +1.50% / pp32768 +1.17%; nsys 逐 launch: l~135k ncols=32 反慢 0.3-0.6% (grid 384 vs 768, 40.28 vs 40.38ms); 根因 = harness 长 l 保真缺口 (只验过 l=35k), 长 l 瓶颈 = K/V 流式非 tile 配置 | **CLOSED (Q16, 用户: 1% 级 + 动 attention 划不来)**; 两处源码已撤, patch `t18-ncols32-REJECTED.patch` |

## 4. RESULTS.md 时间线索引

| 章节 | 一句话 |
|---|---|
| 基线 | pp512 950 / tg 26.64 / PPL 4.3572; 跳过 dequant 1242 |
| T01 | cuBLAS 墙 84-107 TF; 三个否证; ncu 画像 (L1 受限, occ 12.5%) |
| T03 第 1 步 + 收尾 | vec4 行布局; pp512 962.6 -> 959.7; tg 回归是漂移, 分支后噪声内; KDA 修复 |
| T02 Stage 1 v1 | 9.9 TF, 正确; 诊断与 Stage 1b 规格 |
| T05 | decode 分解; MMVQ 83-90% 实际带宽; 30 t/s 是临界 |
| 用户侧 MTP | draft-mtp n-max 3: 25.3 -> 42.7 t/s, acceptance 0.617 |
| T07 第 1 项 | silu kernel -5.1%, 端到端噪声内 |

## 5. 已花时间但未继续的方向

- dequant 的 FP32 路径向量化 (只影响非 cuBLAS 路径, 收益为 0)
- GDN chunked 重写 (kernel -60% 理论, 整机 +3-4%, 1-2 天高风险; 保留为最低优先级)
- T05 的 MMVQ vec_dot 数据通路改造 (研究性, 收益 <= 10% decode)

## 6. T02 融合 GEMM 的否决详情 (2026-09-22, Gate A)

数字 (ffn gate/up, K=5120, 91.27 GFLOP, V100 @1530MHz):

| 实现 | 时间/TFLOPS |
|---|---|
| 纯 mma 原语峰值 (寄存器常驻) | 97-99 TF (上限) |
| 完整流水线 v4 (fp16 权重, BM=BN=128) | 1.586ms / 57.5 TF |
| mma 路径单独 (mode 32) | 1.46-1.47ms / 62-78.7 TF |
| staging 单独 (k-blocked 布局) | 0.837ms / 1.42GB -> 1.7 TB/s |
| cuBLAS gemmEx 默认 | 1.179ms / 77.4 TF |
| cublasLt heuristic#0 (algo21/tile20/splitK2) | 1.083ms / **84.3 TF** |
| baseline dequant + cuBLAS 默认 | **1.404ms** (盈亏平衡 66 TF 等效) |

否决理由 (双重):

1. mma 路径单独 (1.472ms) 已慢于 baseline (1.404ms) -> 融合即使零成本也赢不了
2. dequant 指令预算: 每 SM issue 预算 10.07M 条 (1.645ms), 现用 2.92M; 每 block 一份 Q6_K->fp16
   约 2 指令/权重 -> 4.4M 指令/SM, 会吃满 issue 与 HMMA 争抢 -> 融合上限 50-67 TF, 卡在盈亏平衡线

可复用发现:

- **k-blocked 布局**: 权重/激活按 [kb][row][BK] 重排, staging 从 574GB/s 提到 1.7TB/s
  (原始行主序跨步 10240B -> DRAM 随机访问)
- grid.x = token 维 (相邻 block 共享权重面板): +6.5% (L2 复用)
- Volta `tile<32,4,half2>` (m8n8k4) 与 `nvcuda::wmma` 16x16x16 完整版同性能 (34.6/34.7 TF)
- 正确性坑: tile 的 get_i/get_j 用 threadIdx.x 当 lane, 必须二维 block (32, nwarps)

产物: `v100-collab/artifacts/` (t02_gateA_v4.cu 最终版, v6/v7 诊断, t02_gateA_peak.cu, t02_tile_dbg.cu, ncu 日志)

## 7. T01 vs T02 的 cuBLAS 数字矛盾 (已由 T08 解决)

| 来源 | gemmEx 默认 | cublasLt |
|---|---:|---:|
| T01 ABAB (预热交错) | 84.2 | 84.1 |
| T02 Gate A harness | 77.4 | 84.3 (algo21/tile20/splitK2) |
| 模型侧反推 | ~77 | - |

结论: **T01 的"无差"是对的**; T02 的 "+9%" = 转置方向 + math mode + 预热/时钟伪影。
llama.cpp 真实方向 (m=out_dim) 下 cublasLt 加权 -1.11% -> T08 REJECTED。详见第 9 节。

## 8. T09-A (量化 operand 融合) 否决详情 (2026-09-22) - 融合永久关闭

实现: 真实 Q6_K 权重按 half-super-block [40][17408][112] 重排 (0.88 B/元素), v4 骨架 + 块内解包
(LOP3/移位/half2), Python 全量验证布局, NMSE 2.48e-12。

| 配置 | 完整 | 去解包 | 备注 |
|---|---:|---:|---|
| v9 BM=64, BN=128 | 2.464ms / 37.0 TF | 2.007ms (45.5 TF) | 解包 0.46ms (18%) |
| v9 BM=128, BN=128 | 2.230ms / **40.9 TF** | 1.569ms (58.2 TF) | 解包 0.66ms (30%) |
| v4 fp16 operand (对照) | 1.586ms / 57.5 TF | - | 同骨架 |
| cuBLAS 默认 | 1.187ms / 76.9 TF | - | |
| baseline dequant+cuBLAS | **1.404ms / 65 TF** | - | 盈亏平衡 66 TF |

三条独立否决:

1. 去解包 (BM=128) 1.569ms > baseline 1.404ms -> 即使解包零成本也赢不了
2. 解包 18-30% 且不可重叠 (Volta 无 cp.async, issue 与 mma 争抢)
3. BM=512 (权重只解包一次) 物理不可能 (C tile 256 regs/thread + smem >96KB)

根因: kernel 不是 DRAM 受限而是 staging 延迟/发射受限; 量化版 staged 流量 1.07GB vs fp16 1.42GB
但时间几乎相同 (1.57 vs 1.59ms); 激活重读 (M/BM 次) 抵消权重侧节省。
T09-B (CUTLASS) 同步跳过: 76.9 TF staging + 18-30% 解包 = 53-63 TF < 66 TF, 上限为负。

## 9. T08 (cublasLt) 否决详情 (2026-09-22) - cuBLAS 调用侧关闭

真实方向 (llama.cpp: `GemmEx(OP_T,OP_N)`, m=out_dim, n=tokens) 逐形状 ABAB, 5 轮交错, 中位数:

| shape | llama.cpp 现状 | cublasLt best | 增益 |
|---|---:|---:|---:|
| ffn gate/up | 1.112ms / 82.1 TF | 1.145ms / 79.7 TF | **-2.9%** |
| ffn down | 0.934 / 97.7 | 0.930 / 98.2 | +0.5% |
| ssm in-proj | 0.583 / 92.1 | 0.583 / 92.1 | 0.0% |
| ssm out / o | 0.343 / 94.0 | 0.340 / 94.6 | +0.6% |
| ssm qkv | 0.456 / 70.7 | 0.447 / 72.1 | +2.0% |
| attn qkv | 0.797 / 80.8 | 0.796 / 80.9 | +0.1% |
| attn k/v | 0.100 / 53.8 | 0.100 / 53.9 | +0.3% |
| ssm_ba | 0.033 | 0.033 | -0.1% |

**加权 -1.11% of GEMM -> -0.67% 整机 prefill -> REJECTED**, Step 2 取消, llama.cpp 保持 0 行改动。

三个因素 (全部实测):

1. **方向**: T02 harness 是转置方向 (m=tokens), 那里 cublasLt 确实 +6.7%; 真实方向无优势
2. **math mode**: llama.cpp 现状 (`DEFAULT_TENSOR_OP + TF32 math`) 已是三者最优
   (gate/up +2.8%, down +8.6%, k/v +61%)
3. **时钟**: 真实混合负载 SM 中位 **1447MHz** vs 纯 GEMM 1530MHz (-5.4%, 背靠背预热 20 次 1.20ms ->
   1500 次 1.076ms), 属功耗管理, 不可控

附带否证: 显式 algo 0..15 在真实方向全部慢于 DEFAULT (gate/up 最好 62.6 vs 82.1);
全局 ALGO8 -> pp512 -68%。**所有 algo hint 路线彻底封死。**
盈亏平衡修订: T02 的 1.404ms baseline 是转置方向; 真实方向约 1.36-1.40ms -> 线 66-70 TF,
T02/T09 的否决结论不变。

产物: `artifacts/t08_arb.cu` (+ one/interf 诊断 + 原始 log + `cublas_calls_pp512.log`)。
