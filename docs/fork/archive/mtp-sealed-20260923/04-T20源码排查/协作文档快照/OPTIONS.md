# OPTIONS: 有提升的方案总表 (最终回顾 → 采纳执行)

最后整理: 2026-09-23 (analyst)。口径: ub512; 数字全部来自实测; 比较必须同 session A/B。
**用户裁决 Q10 (2026-09-23): 所有有正提升且显存变化小的方案都采纳** -> 状态列已更新为采纳/任务映射。

## A. 已落地 (交付版) - 保留

| # | 方案 | 实测收益 | 显存 | 状态 |
|---|---|---|---|---|
| A1 | Q6_K/Q5_K 向量化反量化 | pp512 788 -> 950 (**+20.6%**) | 0 变化 | 已入库 |
| A2 | GDN vec4 行布局 + KDA 修复 | kernel -6.1%; pp512 +1.0% | 0 变化 | 已入库 |
| A3 | silu float4 | kernel -5.1%; e2e +0.1% 级 | 0 变化 | 已入库 (Q10 确认保留) |
| A4 | FATTN 长文加速 (stream-K + PB=2, T12+T16) | **pp8192@depth128k +11.28%** / depth32k +4.77% / pp32768 +2.18% | PPL -0.0006 (1 ulp 级, 良性) | **已入库** (patch `v100-t12t16-fattn-split`) |

## B. 采纳执行 (用户 Q10)

| # | 方案 | 实测收益 | 显存 | 任务/状态 |
|---|---|---|---|---|
| B1 | ub -> 2048 | pp512 +21%; 128K ~+25% | - | **不采纳 (Q14, 封存备查)** |
| B2 | ub512 FA KV-split (T11) | depth32k **+2.0%** / pp32768 **+1.1%** | fixup partials, MB 级 | **T12 重放集成 (APPROVED)** |
| B3 | `__expf` (GDN) | kernel **-10%**; e2e +0.7% | 0 变化 | **砍掉 (用户 Q12: 保零漂移)**; 规格留 TASKS/T13 |
| B4 | 双 stream 反量化重叠 | +0.5% (带宽饱和锁死) | +~178MB | **DROPPED (Q13)** |
| B5 | T16: parallel_blocks KV 切分 | **实测: 长文点 +11.28% / depth32k +4.77% / pp32768 +2.18%** | 12.8MB fixup (pool 复用) | **DONE -> VERIFIED (入库)** |
| B6 | T17: Q8_0 权重反量化补测 | 无空间 (已 767-877 GB/s, 向量化候选更慢) | - | **CLOSED (T17)** |

## C. 工程外 / 用户侧

| # | 方案 | 收益 | 状态 |
|---|---|---|---|
| C1 | MTP (draft-mtp n-max 3) | **25.3 -> 42.7 t/s (+69%)** | 用户已在生产使用 |
| C2 | MTP draft 词表切片 (1Cat) | draft 开销 -25% 级 (其栈) | **待立项** (产品级改动, 多日; 需用户确认范围) |

## D. 外部未验证 (1Cat; 随 B1 之后再评估)

| # | 方案 | 外部收益 | 状态 |
|---|---|---|---|
| D1 | CUTLASS f16 s884 128x256x32 | M=8000 比 cuBLAS +3.5-5.7% (bitwise) | **暂缓 (Q11, 随 T14)**: 需 vendor CUTLASS; 只在 ub>=2048 有意义 |

## E. 已评估、不推荐 (Q10 之后仍不采纳)

| # | 方案 | 原因 |
|---|---|---|
| E1 | 跨 ubatch 缓存 fp16 权重 | ~8GB 显存, 与 B1 冲突 |
| E2 | decode 核心级 kernel 合并 | 天花板 27.3-27.5, 多日高风险 |
| E3 | decode FP8 KV + XQA | llama.cpp 无 fp8 KV, 移植量巨大 |
| E4 | 融合 / 自研 GEMM / cublasLt / algo hint | 实测 <= 0; 融合已定死 |

## F. 排序与叠加

- B1 (T14) 涵盖 B2 的场景 (ub2048 下 attention 已 39.9 TF/s, KV-split 不触发); 但 B2 对 ub512 口径仍有效
- B1 落地后 D1 才有意义; E1 与 B1 显存冲突 (B1 优先)
- 执行顺序 (Q14): **T12 (0.5d) -> T17 (0.5h 测量) -> T16 (1d, 有门槛)** -> 组合复测 (含 pp8192@depth128k)
- T12/T13 都改 kernel, 各自验收; 组合后做一次总验收 (pp512/4096/8192/32768/depth32k + tg128 + PPL)

## G. 裁决记录

- Q7 (回顾制) -> Q10 (采纳): 2026-09-23, 用户裁定 "所有有正提升且显存变化不大的都采纳"
- Q12 (2026-09-23): `__expf` 砍掉 (保零漂移) -> T13 取消, 代价 +0.7%
- Q13 (2026-09-23): 用户常用上下文 128-150K -> T12 接受, T14 重开 (强烈建议), T16 新增, T15 抛弃
- Q14 (2026-09-23): 最终接受 = A1/A2/A3 + T12 + T17 + T16; ub 不采 (封存); T15/T13 放弃; 长文点 = pp8192@depth128k
- 映射 (Q14 后): B1 不采, B2->T12, B3 砍, B4 抛, B5->T16 (APPROVED), B6->T17 (APPROVED), C2 待立项, D1 延后, A1-A3 保留, E 不采纳

## H. 外部调研候选 (2026-09-23, **待用户判断; 未批准, 不得自动执行**)

来源: NInfer (geoffwatts/Encapsulate/upstream, 已克隆)、v100-skinny、HyperQwen (qwen38-27b-rtx3090)。
调研总结论: **无可直接回移的内核级项** (NInfer vendored FA 比我们旧; sm86 内核不可用; chunked GDN 在 sm80+ 才启用)。

| # | 候选 | 来源 | 预期 | 显存 | 工作量 | 状态 |
|---|---|---|---|---|---|---|
| T21 | 128K 投机经济学实测 (MTP K 扫描 K=1/2/3 + greedy drafting 检查) | skinny/NInfer | 长文 decode 与显存双降 | 降 | 0.5-2 天 | 待判断 |
| T22 | 内置 ngram 查表投机实测 (`--spec-type ngram-map-k4v`, 零代码) | HyperQwen 灵感 | 复述/代码类 +20-40%, 普通 +2-5% | 中性 | 0.5 天 | 待判断 |
| T23 | draft-head 词表切片 (131072 行 Q4 + id 映射) | NInfer | MTP ~2-3% | **+350MB (边缘, 需实测)** | 2-4 天 | 待判断 (用户: 降级选项, 不排除) |
| T24 | ReplaySSM (GDN 投机状态 raw-input 重放) | NInfer/sglang | 快照 146.8MiB -> 1.71MiB, 省 ~430MB + 少写 | **降** | 高 (fold 须逐 bit 同路径) | 待判断 |
| T25 | GDN recurrent state fp16 | HyperQwen/vLLM | 144 -> 72MiB/份, 读写减半 | **降** | 小 | 待判断 |
| T26 | MMVQ 短 K 形状微调 | skinny | 1-3% | 中性 | 1-2 天 | 待判断 |
| T27 | 评测升级 (Needle 64K/128K/260K + 1M PPL 语料, 复用 NInfer eval/) | NInfer | 长文质量验证 | 中性 | 小 | 待判断 |
| T28 | BPE merge 扁平查表 | NInfer geoffwatts | 长文 TTFT -50~150ms CPU | 中性 | 小 | 待判断 |
| T29 | 设备端采样/接受 (条件: 生产用随机采样) | NInfer | MTP 轮延迟 | 中性 | 中 | 待判断 (条件性) |

- **已并入 T20**: 小批量量化 V 路由 VEC (旧报告 env 修复) — T20 决定是否转默认。
- **已排除**: `-ctk q8_0` (用户质量红线); v100-skinny QPN/内核; NInfer chunked GDN (sm80+ 禁用);
  NInfer Q6 rowsplit; sm86 内核 (Marlin/FP8/Triton); KVarN/DFlash2; 跨 ubatch fp16 权重 (~8GB)。
- **中期搁置**: Hadamard + 低比特 KV (质量路线; 用户已否决 K 量化)。
- **T19 遗留**: tg d131072 **-3~-10% 存疑未定论** (4-6 轮交替 A/B 可定论, ~30min) -> 并入 T20 期间顺带或 T20 后。
