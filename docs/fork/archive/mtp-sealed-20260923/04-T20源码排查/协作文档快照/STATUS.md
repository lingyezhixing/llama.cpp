# STATUS: 当前有效事实

最后整理: 2026-09-22 (analyst)

## 当前最快配置

```
CUDA_VISIBLE_DEVICES=1 llama-bench -m <models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf \
  -ngl 99 -fa on -ctv q8_0 -ub 512 -p 512,4096,8192 -n 128 -r 3
```

## 已验证数字 (含已入库的 4 项改动: dequant-vec / GDN vec4 / silu vec4 / t12t16-fattn-split)

| 指标 | 值 | 说明 |
|---|---:|---|
| pp512 | **950.8** | T12+T16 A/B 同 session B 值 (本机热漂移大, 绝对值仅同 session 可比) |
| pp4096 | 932.5 | 同上 |
| pp8192 | 912.1 | 同上 |
| pp32768 | **791.1** | **+2.18%** (T12+T16) |
| pp4096 @depth32k | **682.2** | **+4.77%** (T12+T16) |
| **pp8192 @depth128k** | **375.5** | **+11.28%** (T12+T16; 长文代表点, 用户 Q14) |
| tg128 | 26.68 | 热漂移: 连续负载 26.6 -> 24.4, 用冷却后复核值 |
| PPL (512ctx/8chunks/seed42) | **4.3562** | 基线 4.3572; 漂移 = GDN vec4 (-0.0003) + PB=2 softmax 归并 (-0.0006), 均 1 ulp 级良性 |
| ub2048 (调参口径, 未采纳) | 1151-1157 | +21%, 用户 Q14 不采纳 (封存) |

## 硬件

- Tesla V100-SXM2-32GB (sm70): FP16 TC 125 TFLOPS @1.53GHz, HBM2 900 GB/s, 300W, smem 96KB/SM
- 无 int8 TC (dp4a ~62.8 TOPS), 无 ldmatrix, mma 只有 m8n8k4 / m16n8k8
- 实际可用带宽实测 825-850 GB/s (dequant kernel 达 92%)
- 另一张 RTX 4060 Laptop 只跑 mmproj, 不参与

## 模型

- qwen35 混合: 65 block = 48 SSM (gated delta net) + 17 full attention (每 4 层, GQA 24/4, head_dim 256)
- hidden 5120, FFN 17408, vocab 248320; 21.97GB, 27.32B 参数 (26.05B 参与 GEMM)
- 量化: Q6_K 63% + Q5_K 22% + Q8_0 13% (含 output.weight)

## 关键常数

| 量 | 值 |
|---|---|
| GEMM 计算量 | 52.1-52.6 GFLOP/token |
| 理论 prefill 上限 (125 TF) | 2399 t/s (仅理论) |
| 理论 decode 上限 (900 GB/s) | 40.9 t/s (实际可用带宽推 28-29) |
| dequant pass | 20.9GB 读 + 52.1GB 写 = 73GB/ubatch; 微基准 825 GB/s |
| dequant 每 token 摊销 | ub512 0.173ms / ub1024 0.086ms / ub2048 0.043ms |
| KV | 64KB per ctx-token (16 层 GQA4) |

## prefill 时间预算 (ub512, 532ms 实测)

| 项 | 时间 | 可压缩性 |
|---|---:|---|
| cuBLAS GEMM | 349ms | 只能靠融合/自研 kernel 超过 cuBLAS (T02) |
| dequant | 88.5ms | 融合删除 (T02), 上限 16.6% |
| gated_delta_net | 40ms | 已优化 -6%; chunked 还能 -50% (高风险, 延后) |
| elementwise | 35ms | T07 现实只能省一半左右 (+0.5-1%) |
| 杂项 (FA/norm/embd/gap) | 20ms | 小 |

## decode profile (T05 结案, `--cuda-graph-trace=node` 稳态; 单 token ~37.5ms = 26.6 t/s)

| 类别 | 时间 | 占比 | 备注 |
|---|---:|---:|---|
| mul_mat_vec_q | 29.84ms | 86.9% | 逐矩阵 685-845 GB/s (lm_head 845 = 94-99% 可用), 已饱和 |
| quantize_q8_1 | 0.84 | 2.4% | 461 次 |
| rms_norm | 1.10 | 3.2% | |
| elementwise | 1.03 | 3.0% | |
| get_rows | 0.47 | 1.4% | |
| gated_delta_net | 0.34 | 1.0% | |
| flash_attn | 0.32 | 0.9% | |
| **合计** | **34.33ms** | - | 2024 kernels + host 间隙 3.4ms |
| 判读 | - | - | **关融合对照: 533 融合头总共只值 0.70ms**; 剩余上限 1.0-1.1ms (+2.7-3.0%) -> 天花板 27.3-27.5; CLOSED |

## 上限结论 (ub512)

| 场景 | 结论 |
|---|---|
| 融合路线 (T02 + T09-A) | **永久关闭**: Gate A 骨架 57.5 TF; T09-A 量化 operand 37-41 TF, 去解包 1.569ms 仍 > baseline 1.404ms; 解包 18-30% 不可重叠; BM=512 物理不可能 |
| cuBLAS 调用侧 (T08) | **关闭**: cublasLt 真实方向加权 -1.11%; 当前 `GemmEx(OP_T,OP_N)+TF32_TENSOR_OP_MATH` 已是三者最优 |
| prefill, 当前可达上限 | **短 ~948 (OURS@NEWBASE `e6ab7c1a4`); pp131072 531.5 (+35.6% vs STOCK); 长文 FA 内核路线已关闭 (T18: 长 l 瓶颈 = K/V 流式, 非 tile 配置)** |
| 长上下文 (T10/T11/T12/T16 实测) | attention 占比: d0 3.3% / pp32768 18.5% / depth32k 32.0%; **128K 点构成 (T18 nsys 实测): FA 41.7% / GEMM 35.2% / dequant 9.8% / GDN 4.6% / 其他 8.7%**; T12+T16 后: depth32k 682 (+4.77%) / pp32768 791 (+2.18%) / pp8192@depth128k 375.5 (+11.28%); **T19@NEWBASE: pp131072 531.5 (+35.6% vs STOCK)**; 不移植 split-D/N32 |
| 1500 | 不可达 (定量证明见下) |
| decode, 纯 kernel | 天花板 **27.3-27.5** (T05 结案, evidence-based; 现状 26.6); 有效吞吐由用户 MTP3 承担 (42.7) |

- 采纳执行 (Q14/Q16): T12+T16 已入库; T17 CLOSED; T18 CLOSED (Q16)
- **T19 DONE (2026-09-23)**: 整理入库 (`afbab1748`, rebase 到 NEWBASE `e6ab7c1a4`) + 双重建部署;
  OURS vs STOCK: pp **+8.7/+9.7/+11.5/+19.6/+35.6%** (512..131072); tg d0-d32768 持平, **d131072 存疑 (-3~-10%)**;
  PPL 4.3562 vs 4.3572; 事实型 prompt 200 token 逐字节一致; 图/CSV 在 artifacts; 详见 RESULTS "T19"
- **T20 DONE (2026-09-23, 暂停等用户裁决)**: MTP 轨迹一致性独立排查 (**不盲信旧报告**; 旧报告 5 条主张 4 条被推翻):
  - **3 个分叉源**: S1 FA VEC(n_q=1)/TILE(n_q>=2) (旧报告 H2, 确认); **S2 GDN vec4 布局分支**
    (本 fork T03/A2 引入: `n_tokens>1` 换 lane 行映射 -> warp 归约顺序变 -> ulp 级数据相关差; 旧报告"GDN batch-invariant"不成立);
    **S3 FA VEC split-K padding 边界** (上游设计: padded n_kv 跨 256 -> parallel_blocks/交错切分变 -> 1e-6 级差)
  - **验收 (三源全修)**: n-max 1/2/3 与无 spec **逐 token 一致** - 短 900 token / 32k 150 / 128k 100;
    回滚+快照路径工具级逐位验证精确 (rollback 1/2/3, rs=0..3, 全行); PPL **4.3562** (同交付); 确定性复测通过
  - **代价**: S1 ~0; S2 tg128 d0 -0.6% / d32768 -1.3%; S3 低代价修法 (小批量固定切分, 已实现) 总代价 **-0.5~-0.8%**
    (旧 PB=1 方案 -31% 已废弃); 128k 一致性按用户指示未测
  - **三臂对照 (no spec, 250 token)**: stock vs **T20(默认)** d0/d8192 **均全等**; stock vs T19 d0@207 分叉;
    T20fix(S1+S2) d8192@147 分叉 (S2 的 1-ulp 级改动) -> **T20 未引入更多不一致**
  - 生产判定: **无降智机制** (输出 token 恒为 target 采样, 代码+行为复核); 修复已备好, 待用户裁决采纳
  - 详见 RESULTS "T20" 1-9 节 + `TASKS/T20-mtp-trajectory.md` Result

## 长上下文 attention (T10 实测, nsys, ub512)

| 场景 | attention ms | 有用 FLOP | TF/s | 占比 |
|---|---:|---:|---:|---:|
| pp4096 d0 | 139.1 | 3.30 TF | 23.7 | 3.3% |
| pp4096 @depth32k | 1899.4 | 56.07 TF | 29.5 | 32.0% |
| pp32768 | 7368.2 | 211.1 TF | 28.7 | 18.5% |
| pp4096 @depth32k **ub2048** | 1404.2 | 56.07 TF | **39.9** | 34.6% |

- 唯一命中变体 `flash_attn_ext_f16<256,256,32,2,0,0,0>` (mma), **grid 恒 192x1x1** (1 CTA/SM, 3 波打尾, 效率 80%)
- 机制: `launch_fattn` 的 `stream_k` 分支判据 (fattn-common.cuh:1150)`效率 >= 75% -> 不拆`, 绕过 parallel_blocks
- 结论: **不移植 1Cat split-D/N32** (ub2048 已 39.9 TF/s >= 其 29-38 上限)
- 近路 KV-split (T11) 已试**并否决**: nsys 机制成立 (grid 192->80, attention 1899->1793ms = -5.6%,
  29.5->31.3 TF/s) 但端到端 depth32k +2.0% / pp32768 +1.1% (门槛 3%/2%) -> 已回退, patch 在 artifacts

## 认知修正 (实测)

- ub512 的 prefill 瓶颈**不是 DRAM 带宽** (848GB/s), 而是 smem staging 的延迟/发射 (v4 mode 2 单独 0.837ms 却无法与 mma 完全重叠)
- 融合"少读 2.6x"无法转化为时间: staged 流量 1.07GB (量化) vs 1.42GB (fp16), 时间几乎相同 (1.57 vs 1.59ms)
- 权重解包重复次数 = N/BN = 4 (ub512), 无法消除 (BM=512 需要 256 regs/thread 的 C tile)
- attention 在 ub512 的"慢"是**调度并行度**问题 (grid 192) 而非 kernel 效率 (ub2048 同 kernel 39.9 TF/s)
- rms_norm 无空间: `<1024>` 变体已 930GB/s = DRAM 极限; `<256>` 变体延迟受限 (57GB/s); 理论上限 0.1-0.2% < 噪声
- **长文 FA (l>=100k) 对 tile 配置不敏感 (T18 结案)**: ncols=32 在 l<=35k 快 5-10%, l~135k 反慢 0.3-0.6%
  -> 瓶颈 = K/V 流式/预取, 非 tile 配置; 后续 FA 类 harness 必须先在**目标 l** (如 128k) 做生产保真对照

## 1500 t/s 不可达的定量证明 (2026-09-22 终版)

1500 t/s = 512/1500 = 341.3ms/ubatch = model-level 26.93 TFLOP / 0.3413s = **78.9 TF (63% MFU)**

- cuBLAS in-model 实测 ≈ 339-349ms (79 TF) -> **GEMM 单独就把 341ms 预算吃光**, 没给 dequant/GDN/其它留任何时间
- 而 dequant 是强制的 (融合已死), ub512 下 88.5ms -> 已经超出预算
- 要 1500 需要 GEMM in-model > 100 TF = 全形状 80% 峰值; cuBLAS 自己只有 63% (加权 72% standalone)
- 即使 T08 (cublasLt) 证实 +9%: GEMM 321ms + dequant 88.5 + GDN 20 + elementwise 35 + 杂项 20 = 484ms -> **1058 t/s**

最终可实现目标:

| 口径 | 目标 |
|---|---|
| ub512, 全面优化 | **~1000-1080** |
| ub2048, 全面优化 (当前实测 1152) | ~1250-1300 |
| 1500+ | 本 GPU + 本模型不可达 (换 A100 级才谈得上) |

## 模型真实 GEMM 形状 (T08, cuBLAS log, pp512 一次前向)

| shape (m,n,k) | 调用/前向 | 每次 ms | 小计 ms | TFLOPS |
|---|---:|---:|---:|---:|
| ffn gate/up (17408,512,5120) x2 | 128 | 1.165 | 148.99 | 78.3 |
| ffn down (5120,512,17408) | 64 | 0.990 | 63.40 | 92.2 |
| ssm in-proj (10240,512,5120) | 48 | 0.609 | 29.23 | 88.0 |
| ssm out / o_proj (5120,512,6144) | 64 | 0.359 | 23.00 | 89.4 |
| ssm qkv (6144,512,5120) | 48 | 0.473 | 22.71 | 68.2 |
| attn qkv (12288,512,5120) | 16 | 0.817 | 13.07 | 78.8 |
| attn k/v (1024,512,5120) | 32 | 0.094 | 3.01 | 57.7 |
| ssm_ba (48,512,5120) | 96 | 0.021 | 1.97 | 7.7 |
| **合计** | 992 | - | **305.4** | - |

- 全部调用: `cublasGemmEx(OP_T, OP_N, m=out_dim, n=tokens, A=权重 f16 lda=k, B=激活 f16 ldb=k, C=fp32 ldc=m,
  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`; handle math = `CUBLAS_TF32_TENSOR_OP_MATH` + 4MB workspace
- 弱形状 (无解): ssm_ba 7.7 TF (极小), attn k/v 57.7, ssm qkv 68.2 (M/N 太小)
- 注: cuBLAS log 合计 305.4ms 与差值法 349ms 的差 = 真实负载时钟 (见下)

## T08 结案: T01/T02 矛盾的三个因素 (全部实测)

1. **方向 (主因)**: T02 harness 是转置方向 (m=tokens), 那里 cublasLt 确实 +6.7%;
   llama.cpp 真实方向 (m=out_dim) 无优势 -> T01 的"无差"是对的
2. **math mode**: llama.cpp 的 `DEFAULT_TENSOR_OP + TF32 math` 比 T02/T01 用的两种都更快
   (gate/up +2.8%, down +8.6%, k/v +61%) -> 当前配置已最优
3. **时钟**: 真实混合负载 SM 中位 **1447MHz** (纯 GEMM 1530MHz; 背靠背预热 20 次 1.20ms -> 1500 次 1.076ms)
   -> 约 -5.4%, 属功耗管理, 不可控
- 附带: 显式 algo 0..15 在真实方向全部慢于 DEFAULT (gate/up 最好 62.6 vs 82.1) -> 全局/逐形状 algo hint 彻底封死


## T19 完成 (2026-09-23, implementer)

- 整理入库: 5 文件注释清理 (零功能改动) -> 提交 `9403d528e`; fetch + ff-only (vanilla) + rebase (fork) 到 **NEWBASE `e6ab7c1a4`** 干净通过 -> 交付提交 `afbab1748` (未 push); 备份分支 `backup-t19-pre-rebase`
- 双边全量重建 + 部署 (SHA 核对): OURS `7F1B9B2403438803` / STOCK `976E2CABF9EADC7D`; 4 patch 基于 NEWBASE 重生成 + 双向校验通过
- 全曲线 (ub512): pp **+8.7% (512) / +9.7% (4k) / +11.5% (8k) / +19.6% (32k) / +35.6% (131k)**;
  tg128 d0/d4096/d8192 持平 (+1.3% 内), **d32768 -10.6% / d131072 -9.4% (待查, 遗留)**
- 质量: PPL 4.3562 vs 4.3572 (-0.0010); 事实型 prompt 200 token 逐字节一致; 创作型 prompt 首 token 因 greedy 近并列分叉 (解释已给)
- 产物: `artifacts/t19_{pp,tg,delta}.png` + `t19_data.csv` + `t19_plot.py`; 原始输出 `%TEMP%/v100/t19_matrix_ub512.txt`
- 遗留: 长文 decode -10% 待查 (下一步: nsys tg@d32768 逐 kernel 对比); ub2048 按用户指示跳过


## T19 遗留项排查 (2026-09-23, implementer)

- **d32768 decode: 排除** - nsys 逐 launch 两臂一致 (GPU busy 5443.7 vs 5446.3ms; flash_attn_ext_vec 761.78 vs 761.81);
  plain 复测 +0.2% (23.02/22.82 vs 22.87/22.87); 矩阵的 -10.6% = 测量状态异常
- **d131072 decode: 存疑** - plain 复测两轮同向 (OURS -3.4% / -10.3%), 幅度受热状态影响;
  STOCK 侧 nsys profile 两次截断 (缺 decode 段) -> 未能逐内核定论; 按"疑似小回归 (<=3%)"记档
- 详见 RESULTS "T19 遗留项排查"
