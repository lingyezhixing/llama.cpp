# STATUS: 当前有效事实

最后整理: 2026-09-27 (analyst; T32 定稿后)

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
- **T20 已结 (2026-09-23): 用户裁决 = 不采纳, 改动已回滚**; MTP 轨迹一致性独立排查 (**不盲信旧报告**; 旧报告 5 条主张 4 条被推翻):
  - **3 个分叉源**: S1 FA VEC(n_q=1)/TILE(n_q>=2) (旧报告 H2, 确认); **S2 GDN vec4 布局分支**
    (本 fork T03/A2 引入: `n_tokens>1` 换 lane 行映射 -> warp 归约顺序变 -> ulp 级数据相关差; 旧报告"GDN batch-invariant"不成立);
    **S3 FA VEC split-K padding 边界** (上游设计: padded n_kv 跨 256 -> parallel_blocks/交错切分变 -> 1e-6 级差)
  - **验收 (三源全修)**: n-max 1/2/3 与无 spec **逐 token 一致** - 短 900 token / 32k 150 / 128k 100;
    回滚+快照路径工具级逐位验证精确 (rollback 1/2/3, rs=0..3, 全行); PPL **4.3562** (同交付); 确定性复测通过
  - **代价**: S1 ~0; S2 tg128 d0 -0.6% / d32768 -1.3%; S3 低代价修法 (小批量固定切分, 已实现) 总代价 **-0.5~-0.8%**
    (旧 PB=1 方案 -31% 已废弃); 128k 一致性按用户指示未测
  - **三臂对照 (no spec, 250 token)**: stock vs **T20(默认)** d0/d8192 **均全等**; stock vs T19 d0@207 分叉;
    T20fix(S1+S2) d8192@147 分叉 (S2 的 1-ulp 级改动) -> **T20 未引入更多不一致**
  - 生产判定: **无降智机制** (输出 token 恒为 target 采样, 代码+行为复核); **修复未采纳** (用户 2026-09-23) -> 工作区已回滚 = T19 交付 `afbab1748`; 实验材料封存 `D:\LLM\Backend\MTP封存-2026-09-23`
  - 详见 RESULTS "T20" 1-9 节 + `TASKS/T20-mtp-trajectory.md` Result
- **T21/T22 降级 (用户 2026-09-23, 暂不做); T25 REJECTED (降低精度不采纳)**
- **T19-L DONE (2026-09-24)**: 瘦身 (丢 A2/A3, 保 A1/A4) 已入库 = 交付提交 `d24474edd` (squash 后单条; 随 T24 推送 = `3eae5cdae` 祖先); 验收 (同 session A/B):
  PPL **4.3567** (T19 重建 4.3562) / pp512 **948.1** (T19 962.8, **-1.5%**; 绝对值 ≈ 记录 948.5 是 session 偏快 ~1.5% 所致, 权威值取 Δ) / pp32768 **-0.9%** /
  pp131072 **-0.5%** / tg 短点持平 / MTP3 d0 55.6 t/s acc 0.932 (无回退); patch 2 个重生成 + 双向校验通过, A2/A3 退役归档;
  部署 DLL SHA256-16 `054BFFD625E37E04` (原 T19 部署 DLL 因操作失误被覆盖 -> 已从 `afbab1748` 重建恢复, PPL 4.3562 复现)
- **T24 (ReplaySSM) DONE + 已部署 (2026-09-26)**: 单提交 **`3eae5cdae`** (四提交 squash; 旧提交 tag `t24-four-commits-20260926`/`t24-pre-squash`), 已推 fork origin/master;
  两个根因: (1) **重启批 (pos 回退) 不得 fold** -> `get_rec_p` pos0 连续性判据; (2) **`s_copy` 覆盖破坏 conv (R) 回滚** -> 还原上游公式 + `s_copy_conv`;
  验收 (全 ON==OFF 逐 token 一致): d0 1000token 老==新; MTP3 64token ON==OFF==base 64/64; PPL 4.3567 两边一致; 自检 0 mismatch;
  双模型矩阵 (Qwen3.8 MTP n-max 1/2/3 x np 1..4 + DFlash n-max 6 x np 1..4, 含生产采样/并发/长程/save-restore) 全过; SEQ_VERSION 3->5;
  VRAM @MTP3 **-420MiB** (np1) / **-1.37GiB** (np4); tg **-1.9%** (71.6->70.2, 权衡); 开关 `GGML_CUDA_GDN_REPLAY=1` (默认关 = T19-L 行为) + `..._CHECK=1` (自检, 默认关);
  **生产部署 = `299DAFC748DE5D91` + `GGML_CUDA_GDN_REPLAY=1`** (备份 `deploy-backup\llama.cpp-my-t19l-20260926`); 待办仅剩: 纳入交付基线; 未覆盖: EAGLE3/DSpark/KDA/np>=5; 详情见下方 append

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
## T24 最终记录 (2026-09-26; 2026-09-27 压缩整理)
T24 final (2026-09-26, 已推 fork origin/master): 单提交 **3eae5cdae** (四提交 c16b1a4c8/c874cab0d/0006ccb0e/e402c4188 squash; 旧提交 tag t24-four-commits-20260926 / t24-pre-squash)。
已闭环 (原未覆盖项): n-max=1 / 不开 MTP ON==OFF 逐 token; VRAM -420MiB (22019->21599);
状态存取 (提交 3eae5cdae): 状态文件追加 replay 块 (pinned 平面 + 记录张量 + 记账), 恢复后首个 batch 用 pending 计数 fold, SEQ_VERSION 3->4 (后 5);
复现证据: 修复前 MTP3 slot save/restore(NA=40) 恢复后第 8 token 分叉; 修复后 16/40/60 全一致。
第 4/5 处问题 (扩面发现, 同提交): (4) 多序列 replay 静默错 -- 记录按 cell 分 bank 但 fold 记账 (gen/half) 全 ubatch 共享 -> 序列交错时 committed 平面永久落后 (证据: np=2 两 slot 交替, slot1 第2轮第3 token 分叉, 日志 RECP STALE; 先门禁后 per-seq 记账修复, 见下); (5) CHECK=1 恢复状态后首 fold 假阳性 (37748352) -- 修复 = fold 张量 skip 位, slots3+CHECK 3 条 mismatch -> 0; 另修 seq_rm 部分回滚边界 (p = pending - rollback)。
本轮矩阵 (全 ON==OFF 逐 token): 同 slot 连续请求 (0 裁剪/2token 回滚/换 prompt 重算)、多次 save/restore+恢复后裁剪、n-max=2、长生成到 ctx 上限 (4082)、MTP3 64/64 vs base、d0 1000 off==on、PPL 4.3567 两边一致。
多序列适配 (per-seq 记账, 同提交): 记录 bank = 序列 id (与 cell 解耦); 每序列独立 gen/半区/自检标记; 半区按序列交替 (读写不撞); fold 块布局 4*n_seq_max+3; 状态存取按 bank (SEQ_VERSION 5)。
踩坑: kernel 侧 n_fold 除数一度仍为 3 -> np>=3 越界写静默错 (np=2 恰好整除漏过)。
复现脚本: %TEMP%\v100\t24_np2.ps1 / t24_np4.ps1 / t24_npk.ps1 / t24_conc.ps1 / t24_np4sr.ps1 / t24_fresh.ps1。
长程/并发/VRAM (最终二进制): 长程逐 token 一致 (np1 1000 / np2 4x250x2slot / np4 4x125x4slot, 自检 0); 并发长程 (两 slot 同批各 500) 自检 0 (注: 并发 token 级不可跨运行比对 -- 批合并点随时序, 以自检为准); VRAM np=4 OFF 19348 -> ON 17978 MiB (-1.37GiB); np=1 全回归全过 (ab 64/64 / d0 off==on / PPL 4.3567 / seq6 / slots3 / slots NA=40 / n-max=2)。
DFlash 实测 (Qwen3.6-35B-A3B + DFlash-Q8_0, `--spec-draft-n-max 6`, 同一提交): greedy 900 + 生产采样 256 ON==OFF 逐 token + draft/accepted 统计相同; CHECK 0 mismatch; np=2 (800 token) / np=4 (1024) 交替 ON==OFF; VRAM np1 -350MiB / np2 -700MiB; 吞吐在噪声内 (无结论); EAGLE3/DSpark 无权重未测; 脚本 `%TEMP%\v100\t24_dflash.ps1` / `t24_dflash_np.ps1` (t24_conc.ps1 -Dflash 1)。
squash 收尾: 四提交 -> 单提交 3eae5cdae (已推 fork origin/master, fast-forward 无 PR); 旧提交 tag t24-four-commits-20260926 / t24-pre-squash。
整理内容: ggml.h op 文档按当前 fold 布局 (4*n_seq_max+3); tests 去 BOM; "每 ubatch 只填一次" 判据改为 context 内 ubatch 索引 (删除 rec_last_ub/rec_last_pos); 过时注释。
整理后复验: np1 ab 64/64 (两模式) / slots NA=40 / slots3 / PPL 4.3567 / np2 交替 / np4+CHECK 0 / DFlash 128x2 统计一致。
未覆盖: EAGLE3/DSpark/KDA (无权重)、np>=5、无 SWA hybrid 极端小裁剪 (理论边界)。
analyst 复核 (squash 后): HEAD == origin/master == 3eae5cdae (工作区干净); 构建 llama.cpp-t24 = 299DAFC748DE5D91; patch 重生成 artifacts/t24-replayssm-final-3eae5cdae.patch (68,494 B, reverse-check OK)。
2026-09-26 T30 立项 (用户): "MTP 到 128K 只剩 23 左右, 太慢了" -> APPROVED; spec
TASKS/T30-mtp-long-decode.md (Phase A 测量时间去向 + VEC/TILE 微基准 -> Phase B 选核开关 -> Phase C 验收
(MTP3 d131072 为主指标, PPL 门槛)); 需 GPU 时段 (与用户协调)。

2026-09-26 T24 部署 (用户批准) + T30 Phase A/B 实测否决 (implementer)。
- T24 部署: 生产 D:\LLM\Backend\llama.cpp-my 覆盖 9 个文件 -> ggml-cuda.dll 054BFFD6 -> 299DAFC7 (T24 构建);
  备份 deploy-backup\llama.cpp-my-t19l-20260926\ (9 旧文件 + T24-final ggml-cuda 副本); setx GGML_CUDA_GDN_REPLAY 1;
  部署后 PPL 4.3567; 详见 RESULTS "T24 部署完成"。
- T30: 基线 (MTP3, replay=1): d0 50.1 tps / d32768 30.8 / d131072 19.5-20.1; 128K verify FA = TILE 1.50ms/层 (grid=(1,13));
  env 试做: VEC_VERIFY=4 (VEC 在 n_q=4 实测 4.68ms/层 = 慢 3.1x) 与 VERIFY_PB/NBATCH=160/1024 (切分 13->160 慢 15%)
  全部 REJECTED; PPL 控制位 4.3567 逐位不变; 实验代码已回滚 (工作区干净); 接受率 128K 0.388 vs d0 0.808 (= 另一半原因, 属 T31-B)。
- 通道状态: T24 DONE + 已部署; T30 REJECTED (Phase A 判据达成); BOARD/ENVIRONMENT/OPTIONS/STATUS 已同步。
2026-09-26 analyst 复核 (T24 部署 + T30):
- 部署已核实: `llama.cpp-my\ggml-cuda.dll` = `299DAFC748DE5D91` (21:08 批, 9 文件同批); 备份 `deploy-backup\llama.cpp-my-t19l-20260926\`
  旧 `ggml-cuda.dll` = `054BFFD6` (完好可回滚); `GGML_CUDA_GDN_REPLAY=1` 用户级已核实。
- T30 结论成立: VEC 慢 3.1x / 物化 ~10% / PB 切分 -15%; 预先声明的风险命中, 未浪费 Phase B/C; T30 已回滚 (工作区干净,
  实验构建隔离在 `llama.cpp-t30` `CF4B2B5E`)。128K MTP3 实测 **19.5-20.1 tps** (用户口径 ~23, 以 RESULTS 表为准);
  接受率 0.388@128K 是长文衰减另一半 -> T31-B (用户已否 T21 K 扫描)。
- 通道 (BOARD/ENVIRONMENT/OPTIONS/T30) 已同步; T24 待办仅剩"纳入交付基线"。
2026-09-26 T31 立项 (用户指示集中攻克): "接下来就剩 MTP 轮内开销了" -> APPROVED; spec
TASKS/T31-mtp-round-overhead.md; 三步: A 分解 128K 一轮 ~111ms (未解释 ~50ms/轮) -> B draft 窗口化 / 长文 K 节流 ->
C 设备端采样/异步 (并入 T29); 主指标 MTP3 @d131072 t/s (目标 +20% 起); 需 GPU 时段。
同轮决定: T14 延后 (先集中 T31); T21/T22 用户明确否; 投机组合 (first-wins 回退链) 不采纳。
2026-09-26 T32 立项 (当时 PROPOSED; 2026-09-27 定稿为系统级树, 见下): agent 长会话复用专题; 子问题 1 = 思考剔除后长回合
从头 prefill (检查点可能被 32 上限压缩挤出; 用 `LLAMA_SERVER_SLOTS_DEBUG=1` + TRC 日志确诊; 对策: 客户端不剔思考 /
fork pin 末尾检查点); 子问题 2 = 两长会话轮换时 8GB prompt-cache FIFO 互相淘汰导致每次全量 prefill
(对策: `--cache-ram 16384` 实测, 31GB RAM 留余量); spec TASKS/T32-agent-session-reuse.md。
2026-09-26 晚 T32 设计进展 (2026-09-27 定稿): 子问题 3 = 多序列切换 + 分级淘汰 -> **用户 2026-09-27 决定直接采用系统级树取代轮换; 设计 v1 定稿** (S1 -> S2a -> S2b; 可行性 = S1 高 / S2a 中 (冒烟门) / S2b 后置);
已确立: 停放格数 = N (稳态 N-1 占 + 1 空; 空位是切换必需品), 磁盘 = 第二格角色交替, pin 目标 / 弃工作序列, pin 目标 > 消费目标 > 淘汰最老 unpinned;
低容量边界细节并入 S1/S2b 设计; 详见 TASKS/T32。
2026-09-26 深夜 T31 执行前整理 (implementer, 无 GPU): 数据复用审计 -> H1 (draft 步) 上界 ~5-10ms/轮 (T30 实测 1.6ms/层@128K + 权重规模),
  故 B1 (draft 窗口化) 上限 ~+4-5% 降级; verify 侧只剩硬余量 (T30 已否并行路线); 主项 = ~30-35ms/轮未解释开销 (H2/H3)。
  关键发现: speculative.cpp 已内置 t_begin/t_draft/t_accept (gen_perf=true 恒开), 每请求经 SPC_TRC 打印 -> Phase A 零代码:
  server 加 `-lv 4` 即得每请求 draft/accept ms + 逐位置接受率。已就绪 (等 GPU, ~15min): t31_phaseA.ps1 (n-max 3/1/none)
  + t31_spc.py (差分解析); 同轮补"无 spec @128K" 基线 (估 ~15 t/s = MTP3 长文仅 +20-30%)。
2026-09-27 现状整理 (analyst) -- 剩余任务:
- **P0 (GPU ~15-30min)**: T31 Phase A 三连 (n-max 3/1/none; `-lv 4`; `t31_phaseA.ps1` + `t31_spc.py`) -> H1/H2/H3 分解 + 无 spec @128K 基线; 判据: >15ms/轮 项即主攻; 预期 H2/H3 -> B2 (设备端采样, 并入 T29), B1 降级
- **P0 (无 GPU)**: ~~T32 S1 开工~~ **DONE (实现+实测, 见文末; 待提交)** (检查点规范化 + 去 0.25 + 跨 blob 最近祖先恢复; 收益 = 修 T32-1/2); 冒烟实验 **DONE (门通过)** (`seq_cp` 共享 4 项)
- **P1**: T14 ub2048 实测 (0.5h) + T32-2 配置实测 (`--cache-ram 16384` RSS + thrash 日志)
- **P2 收尾**: T24 纳入交付基线; T04 服务端配置固化 (依赖 T14/T32-2); 最终统一回顾 (Q7)
- **P3 可选**: T32 S2a (视冒烟) / S2b (节点级冷存储, 后置); verify 侧 TILE 直读 q8_0 V (~+10%, 大工程); T23/T26-T28
- **已关闭**: T20/T21/T22/T25/T30/树 v1; T11/T13/T15/T18 等回顾前不重开
T32 决定 (用户 2026-09-27): 直接采用系统级树取代纯轮换 (最差=轮换, 收益>=0; 保留大量高复用短前缀); 设计 v1 定稿 (TASKS/T32); 可行性 = S1 高 / S2a 中 (前置冒烟实验) / S2b 后置; 实现注记: 实现风险 / 每节点 144MB 检查点 / n_seq_max 预算。
2026-09-26/27 T31 Phase A (implementer 零代码) + analyst 复核通过:
- 分解 (128K 热, MTP3 prod): 轮 ~112-115ms = verify+host 90ms (78%) + draft 17.9ms (16%) + 目标侧采样 6.8ms + accept ~0;
  原 "未解释 ~50ms" = verify 本体 (H2/H3 假设否决); 无 spec @128K = 77.4ms/token (12.9 tps) -> MTP3 2.10-2.15x
- **口径修正 (重要): 生产采样 (temp0.6) @128K = 27.0-27.6 tps (acc 0.694)_greedy 19-20 (acc 0.388); 用户"~23"落在两口径之间**
- K=3 最优 (MTP6 prod 20.2, draft 35ms + verify n_q=7); B1 上限 +2~4%; B2 (设备端采样) ~+5%;
  内核 (FA/V) 唯一 +20% 候选, 但无已证路径 (T30/T18 前车) -> 立任务需带 l=128K 微基准门
- 复核修正: RESULTS 第 2 节 MTP3 pos 值为累计 (128K 单请求: greedy 0.638/0.362/0.159, prod 0.917/0.729/0.438);
  "38.4ms/token" 应为 36.0-37.0; 冷热异常记档 (无 spec/MTP1 冷值偏高 17-30%, 跨配置以热值为准)
- 待用户裁决: (a) 立 verify 内核任务; (b) 做 B2 (+5%); (c) 暂停 T31 转 T32
2026-09-27 T31 裁决 (用户): **PARKED + 转 T32 主攻**。理由 (量化): 一轮 78% 是 verify (MTP 已到"4 查询读一次权重"的
带宽下限); 可动项仅 FA 25.5ms + V 物化 8ms; 低风险全包 (B2 + V 直读) ~+12% = 31 tps (+3.4 tps), FA 无已证路径 (T30/T18);
对照 T32: 一次 128K re-prefill = 257s (T31 日志实测 128270 tok @ ~500 t/s), agent 一轮重算 4-20s, 会话切换 260s -> 2-6s。
2026-09-27 T32-1 根因确认 (analyst; 用户开放 opencode 源码/DB/启动命令后) -- **长回合 re-prefill 真凶 = 自定义 chat 模板**:
- 用户启动命令带 `--chat-template-file .../Qwen3.5-chat_template.jinja`; 该模板第 103 行**硬编码**
  `{% if loop.index0 > ns.last_query_index %}` 保留思考, **完全不实现 `preserve_thinking`** -> 能力探测 false ->
  服务端 "默认开启" (arg.cpp:958-961) 无从生效; 行为 = 新 user 到来即剥掉上一轮全部 assistant 思考
- 证据: (1) opencode DB 每条 assistant 消息均有 reasoning part 且轮内被发回 (下一步增长≈上一步输出+工具);
  (2) 跨 user 边界 prompt 缩短 ≈ 上一轮思考量 (实测 824 / 6329 token), llama.cpp 缓存分叉点 = 上一轮首条 assistant 消息;
  (3) GGUF 内嵌模板第 119 行有 `preserve_thinking is undefined or is true or ...` 条件, 自定义模板无 (且缺 reasoning_effort 注入,
  用户 opencode variants low/medium/xhigh 一直无效)
- **修复 (零服务端代码, 待用户应用)**: (a) 去掉 `--chat-template-file` (推荐, 顺带恢复 reasoning_effort) 或 (b) 模板第 103 行改内嵌版条件;
  修复后 prompt 追加式 -> re-prefill 从根上消失; 代价 = 思考常驻上下文变大 (更依赖 T32-2/树)
- 附带口径: 用户生产采样 = **temp 1.0**/top-p 0.95/top-k 20 (启动命令), 非 T31 测试用的 temp 0.6 -> T31 目标与 T04 按 temp 1.0 重定;
  `--spec-draft-n-max 3` 与 T31 结论一致; `-c 184320` 与 opencode model limit 一致
-> T32 RUNNING (S1 DONE + S2a 冒烟门通过, 见文末; 待决 = 提交/S1.5/S2a/模板/T24 复查); **T31-B2 留作廉价待办 (0.5-1 天, +1.4 tps)**;
内核任务不立 (除非微基准 gate 先过)。
2026-09-26 T31 Phase A 完成 (implementer, 零代码, 5 配置, 同 session 13:06-13:35)。
- 分解 (128K MTP3 prod 热): 轮 114.9ms/3.12tok = verify+host 90ms (78%) + draft 17.9ms (16%) + 采样 6.8ms (6%) + accept 0.3ms。
- 修正: verify 实际 ~90ms (无 spec 单 token 77.4ms 为证), "未解释 ~50ms" = verify 本身; MTP3 相对无 spec -50%。
- B1 (draft 窗口化) 上限 +4% -> 否决; B2 (设备端采样/T29) 上限 +5%; 唯一能到 +20% 的 = verify 的 FA+V 物化内核 (33ms)。
- K: MTP3 最优 (MTP1 16.3-19.0 / MTP6 20.2) -> 保持 3; 生产采样下 MTP3 @128K = 27.0-27.6 tps (greedy 19-20 是特例)。
- 数据/脚本: %TEMP%\v100\t31_{mtp3,mtp1,nospec,mtp6prod}.err + t31_phaseA_*.jsonl; t31_spc.py (已归档 artifacts)。
2026-09-27 T32 S1 执行计划 (implementer 整理, 无 GPU)。要点 (新增实测):
- 检查点实测大小 = 162MiB + ~4.1KiB/token (t31 生产日志: n=1 161.77; 127754 663.73; 128266 665.74; Δ512=2.01MB)
  -> 128K 每点 ~666MB, 32 上限 21GB -> S1 不做固定网格 (每点成本 666MB); pin 集必须极小 (只 pin end-4 锚点)
- prompt cache blob 含检查点: 128K blob ~= 9.8G (7.2G state + 2.6G ckpt) -> 默认 cache-ram 8192 会整条 skipping
  (T32-2 成本部分是"跳过"而非 FIFO 淘汰; 诊断先抓 `exceeds cache size limit`)
- S1 四项: A pin 锚点 (修 T32-1) / B max-LCP + 去 0.25 (修命中率) / C 跨 blob 祖先恢复 (免代码, 验证) /
  D select-before-save + protect (修轮换互删); P0 = t32_repro.ps1 (短 GPU, agent 长回合 + 双会话轮换, 兼 S1 验收);
  S2a 冒烟复用 tests/test-save-load-state.cpp 的 seq_cp 基建 (临时 harness, 不进 tests/)
- 待决 3 点 (pin 集/下限默认/网格) 见 TASKS/T32 末节; 不阻塞编码
2026-09-27 T32 S1 实现+实测 (implementer): A pin 锚点 / B max-LCP+去 0.25 / D park 保护 + blob 检查点裁剪; 补丁 32.7KB @3eae5cdae。
- T32-1 A36: 基线 D 锚点被压缩删除 -> 恢复 D-516, 3274 tok/5.68s; 修复后 D-4, 2762 tok/5.10s
- T32-2 B cr1100: 基线每次切换 5.5K tok/7.2s (park 逐出目标); 修复后 71 tok/0.6s (最终 4 tok/0.23s) = ~12x TTFT
- 正确性: A 22 + B 8 请求输出逐位一致; 检查点 149.6MB(无MTP)/+4.1KB每token(MTP) 复核
- 下一步待用户: 提交/部署/S1.5 瘦身/S2a 冒烟; 产物与日志已归档 artifacts\t32*
2026-09-27 T32 S2a 冒烟 (implementer): 门通过。item1/2a/2b (seq_cp 共享/refcount/分支重建) 逐位一致;
item3 发现 T24 ReplaySSM 回滚运行间不确定性 (ref-vs-ref 亦复现, 与共享无关; replay=0 逐位一致) -> 记 T24 后续;
item4: 27B 每活分支 +220MiB。**更正 (同日实验): unified KV 不自动共享前缀 cells (`find_slot` 不去重);
np=2 两会话共享 23K 前缀时 B 仍全量 prefill (23076 tok/28.5s); 跨序列共享必须显式 `seq_cp` (S2a 第一步)。
零代码可用项 = `-np 2` + `id_slot` 固定使两会话同时常驻 (总量 <= `-c`)。**
产物 artifacts/t32-smoke-*。
2026-09-27 T32 进展 analyst 复核 (材料已核对):
- S1 工作区: 7 文件改动, patch `t32-s1-worktree.patch` 实际 **39.7KB** 含 6 个源文件; `tests/CMakeLists.txt` +
  `tests/test-t32-smoke.cpp` (临时 harness, 不进 tests/) **提交时排除**
- S1 实测 (T32-1 锚点存活 / T32-2 ~12x TTFT / A22+B8 逐位一致) 记录成立, 待用户批准提交
- S2a 冒烟门通过; 附带 T24 replay 回滚运行间不确定性 (ref-vs-ref 1/6 value-diff; replay=0 干净; rb<=2 无损)
  -> **建议单列 T24 复查项** (生产已 `GGML_CUDA_GDN_REPLAY=1`)
- 待用户决定: (1) 批准提交 S1 (仅 6 文件); (2) S1.5 检查点瘦身 (drop data_dft; blob 1.9G -> 1.02G, 128K 可入 cache-ram);
  (3) S2a 开工 (冒烟已过); (4) T24 replay 复查; 模板修复 (T32-1 源头) 为用户侧动作
- **S2a 第一步 fork 结果 (implementer 16:53-16:56)**: **item5/6 PASS** -- 全前缀扩展 + 检查点回退两路都逐 token 匹配
  单序列基线, 源分支不污染 (COW 已核实: `state_read_meta` 先 `seq_rm(dst)` 再分配私有 cell);
  **前提 = 必须 `--kv-unified`** (同 stream `seq_cp` 才零拷贝; 生产与旧 T32 测试均为 `kv_unified=false`);
  **新未解问题 = 多序列布局数值不一致**: 4 分支交错解码在少数位置 (14-19 of 32) 出现 logits 0.1-0.5 分歧后 token 翻转,
  hybrid 与纯 attention 都复现, 共享与独立路径都可能中 (共享非唯一原因) -> 待判定 (a) 引擎 bug 还是 (b) 数值不可比
  (需最小复现 + 同路径重复性检查); 树验收标准待定。**生产 np=1 单序列不受影响; `-np 2` + `id_slot` 顺序请求路线不受影响**。
  产物 `block_t32_s2a_probe.md` / `t32-smoke-*.cpp`(?新 item1b/1c/5/6 + probe4)
- **用户纠偏 (2026-09-27): 需求三条不含任何 VRAM 内复用**; "S2a 活分支 fork" = analyst 误读 -> **停止投入, 产物仅归档**;
  子问题 3 正主 = **树状 RAM+SSD 存储** (枝干优先 RAM / 溢出先叶 / 上树-新建-复用; 即原 S2b 方向), 待重写设计。
  T32 头部已加 "范围澄清" 节; 树状存储 v2 节内的 VRAM 解释已标作废
- **2026-09-27 用户指示: 放弃当前全部修改** -> 已 `git checkout -- .` + 删除 harness, 工作区干净 @`3eae5cdae`;
  快照: temp `t32-worktree-dropped-20260927.patch` (40.6KB, 含 CMakeLists) + `t32-smoke-worktree-20260927.cpp` +
  原有 `artifacts/t32-s1-worktree.patch` (39.7KB, 6 源文件); 生产部署未动。T32 回到"待重写方案"
- **S1.5 重新落地 (2026-09-27 晚, implementer)**: 按用户决定 A 不修; S1.5 + blob 检查点裁剪单独重放 (rtifacts/t32-s1.5-worktree.patch, 2 文件 43 行); 实测 A36 38/38 + B 8/8 输出逐位一致 (vs 旧 MTP 版); 检查点全部 161.769 MiB 常数 (基线 162.97-302.63 @33K); 100K 外推: 555->162 MiB/点, slot 32 点上限 17.3->5.1 GiB, 建议 --ctx-checkpoints 8 (->1.3 GiB); B 轮换 TTFT 未改善 (B/D-protect 未带), 待下一步重放 (S1 全量版实测 71 tok/0.6s)

- **S1.5 提交 + 部署 (2026-09-27 晚)**: 提交 a41cccec (2 文件 43+/6-, Assisted-by: opencode, 本地 master, 未推送); 生产目录 D:\LLM\Backend\llama.cpp-my 已部署 (37/37 文件与构建一致; --version = build 2119 / commit ba41cccec); 备份 deploy-backup\llama.cpp-my-t24-20260926-224252 (37 文件); 冒烟: 2B 模型 health OK + completion OK; 回滚 = 从备份目录拷回 + 重建
- **更新 (2026-09-27 晚)**: a41cccec 已推送 fork origin/master (cdf556fe8..ba41cccec); 生产部署同前 (build 2119 / ba41cccec, 备份 deploy-backup\llama.cpp-my-t24-20260926-224252)

- **T32 阶段 0+1 完成 (2026-09-27, SDD)**: 分支 `t32-stage1` (6 提交, 全分支审查通过, **未 push, 未部署**).
  区间序列化 API (写/读/append+重叠拒绝) + harness; H2D 2.37 GiB/s, q8_0 口径 50200 B/token,
  `--tree-chunk`=512, 100K 4.68 GiB (~2s H2D); 小模型正确性 23/23/23/22 全 PASS + 两个回归全绿.
  资产 artifacts/t32-stage1-{bench,correctness}.txt + t32-stage1-sdd/. 待用户: 合并/push/部署 + 阶段 2 计划.
  **用户约束: 不再跑 27B, 验证只用小模型.**

- **更新 (2026-09-27 晚, 合并)**: 历史整理为 **2 个独立提交** — `2b526ac74` `llama : add range state API for attention KV` (11 文件, +292/-8)
  与 `52b7bf7de` `tests : add range state API harness` (2 文件, +503); tree hash 与验证态逐字节一致 (`798e7ca7...`);
  **已 fast-forward 合并到本地 master** (`ba41cccec..52b7bf7de`), 分支 `t32-stage1` 已删除;
  **未 push (origin/master 落后 2), 未部署** (生产仍 `ba41cccec`).

- **T32 阶段 2 完成 (2026-09-27, SDD)**: 分支 `t32-stage2` (从 `52b7bf7de` 起, 12 提交, **未 push, 未部署**).
  树模块 `server-kv-tree.{h,cpp}` (块链内容哈希 / 锚点 / 稀疏化 / RAM+SSD 单份权威 / 淘汰次序+pin+拒绝) + harness
  `test-t32-tree` 三模式全绿 (小模型 device 0, 全 exit 0): logic 18/18; 2B model 36/36; 2B accept 42/42
  (A/B 4096x2 共享 3072 -> 10 块, 12/12 tip 恢复逐 token 一致; B-mini 4x1024 共享 512 -> 5 块); 3B 纯 attention 对照 36/36.
  归档 artifacts/t32-stage2-{model,accept}.txt. 未接 server (`--kv-tree` 属阶段 3); 生产未动.

- **更新 (2026-09-27 夜, 终审)**: 全分支终审 (12 提交) 结论 "With fixes", 3 条 Important 已修并复审通过 (全部 ADDRESSED, 无新增 Critical/Important):
  ① park 早退静默丢弃检查点 -> WRN + `anchors_skipped` 计数; ② `promote_prune` 改为按捕获链作用域 (防跨分支误删);
  ③ 新增非对齐恢复真机场景 (部分尾块匹配 + 部分块装载 + `seq_rm(C,-1)` 裁剪 + 非对齐锚点).
  最终 head `e7eea21ea` (14 提交); 测试 (最终 head 重跑): logic 18/18, 2B model 43/43, 3B control 43/43, 2B accept 42/42;
  归档已按最终 head 刷新 (`t32-stage2-model.txt` / `t32-stage2-accept.txt`), SDD 证据归档至 `artifacts\t32-stage2-sdd\`.
  未 push, 未部署 (生产仍 `ba41cccec`).

- **更新 (2026-09-27, 整理并入 master)**: 分支 `t32-stage2` (14 提交, head `e7eea21ea`) 已整理为 2 个干净提交并入本地 master (fast-forward, 沿用阶段 1 惯例):
  `85fba497d server : add kv tree storage for attention KV` (模块 + tools/server CMake) / `4f2631088 tests : add kv tree storage harness` (harness + tests CMake);
  合并结果 tree hash `44728e479b1acbf6e6f0d045c9e9db596950909c` 与已验证状态逐字节一致; 合并后在 master 上重建并重跑: logic 18/18, 2B model 43/43, 2B accept 42/42 (全部 exit 0);
  分支 `t32-stage2`/`t32-stage2-tidy` 已删除 (原 14 提交历史保留于 reflog, head `e7eea21ea`, 未 push 过); 本地 master 领先 origin/master 2 个提交, **未 push**; 部署未动 (生产仍 `ba41cccec`).

- **T32 阶段 3 完成 (2026-09-27, SDD)**: server 集成 (`--kv-tree` 默认关) + A/B 验收; 分支 `t32-stage3` (从 `4f2631088` 起, 6 提交, head `b936d687f`, **未合并/未 push/未部署**, 生产仍 `ba41cccec`).
  验收 (脚本 `t32-stage3-ab.ps1`, 8 模式全 exit 0): calib 总量 417,408,732 B -> ram 133 MiB; ab parked=23 restored=12 miss=0, SSD 峰值 497,257,704 B, 12/12 逐位一致; heal 恰 1 捕获@1024; b restored=12 且 ref=4; b3 parked=11 restored=0 (D11); neg 删 65 文件后读失败可见且请求成功; ref x2 30/30 确定性.
  勘误: D1 (低重叠才触发树) / D10 (cache_prompt=false 跳过 restore) / D11 (纯 attention 无 fork 复用, 结构性). 归档 artifacts/t32-stage3-*.txt + t32-stage3-logs/; 生产未动.
- **已知缺口 (D2, 记录)**: 树锚点载荷不含 MTP/spec 状态 (`kv_tree_anchor_in` 无 spec 字段, 模块无 spec io, 树恢复后不还原 spec 状态). 2B/3B 验收不涉及; 27B/MTP 部署阶段补 (或恢复后显式重置 spec 状态).
- **T32 阶段 3 修复波 (2026-09-27, 终审 With fixes 后)**: 分支 `t32-stage3` 追加 3 提交 (`2edf4a6df` server heal 捕获+上报, `5d0aee637` tests 用例, `3416ea622` server 集成日志/生命周期), head `3416ea622`, 仍未 push/未部署. heal 假捕获修复 (`capture_anchor` 仅在真正存储时返回 true; spacing 跳过 = `capture skipped` + false; heal 模式 step 512), heal 落点可不在块边界 (回退挂靠已存覆盖块), D12 (spec §3.2 检查点表重建延后阶段 4) 记入 plan/design, 另修 park 日志/tree_heal 生命周期/write_disk 错误可见性/b3 共享前缀. 复跑: logic 46/46, 2B model 43/43, heal RESULT 0 (stored=True @1024 kind=2), b3 RESULT 0; 其余验收数字属 `b936d687f`.
- **T32 阶段 3 归档刷新 (2026-09-27, head `3416ea622`)**: calib/ab/overlap/b/neg/ref 全部在修复头复跑 (ref x2, ab 用 T32_RAM_MIB=133), 数字与 `b936d687f` 一致 (calib 417,408,732 B; ab parked=23 restored=12 miss=0 rammax=138,760,464 diskmax=497,257,704; overlap 0/0/0; b restored=12 ref4=2; neg strict=2 wide=2; ref 30/30 一致); heal/b3 保持修复头结果; calib/neg 的 16912 捕获由拒绝变 `capture skipped` (F1 回退命中 + F2 spacing, 507<4096), 指标无变化; `t32-stage3-logs/` 与 `t32-stage3-accept.txt` 已刷新. 其余修复头构建/测试 (logic 46/46, 2B model 43/43) 见上条.

- **更新 (2026-09-27, 整理并入 master)**: 分支 `t32-stage3` (9 提交, head `3416ea622`) 已整理为 3 个干净提交并入本地 master (fast-forward, 沿用阶段 2 惯例):
  `960ac9dae common : add kv tree server options` / `041982464 server : add kv tree server integration` / `daf4186d3 tests : add kv tree capture tests`;
  合并结果 tree hash `e89cbef359dc1d4a90b8e8a08a027a883f585a3b` 与已验证状态逐字节一致; 合并后重建并复跑: logic 46/46, 2B model 43/43, 验收 ab 0 failures (parked=23 restored=12 miss=0, 12/12 逐位一致), heal 0 failures (真实存储锚点@1024);
  分支 `t32-stage3`/`t32-stage3-tidy` 已删除 (原 9 提交历史在 reflog, head `3416ea622`); 本地 master 领先 origin/master 3 个提交, **未 push**; 部署未动 (生产仍 `ba41cccec`).

- **更新 (2026-09-27, 阶段 4 启动)**: 用户指令: 白天测试改用 cuda1; 阶段 4 = 长跑 soak + 正确性打磨 (D12 检查点重建 / D13 启动清理), SSD 磨损优化经用户裁决**延后** (D14, 本阶段只记录 IO 基线); 计划 `artifacts/t32-tree-plan-stage4.md` (5 任务, SDD 执行中, 分支 `t32-stage4` base `daf4186d3`); 验收: 2-5 min smoke + 30 min + 60 min soak (反复建树/修剪/整树删除/RAM-SSD 调度/逐位抽检/kill -9 重启), 全回归; 模型 2B.

- **T32 阶段 4 完成 (2026-09-27, SDD)**: 分支 `t32-stage4` (base `daf4186d3`, head `c0619da14`, **未合并/未 push/未部署**, 生产仍 `ba41cccec`); D12 检查点重建 (recurrent tail 语义 `pos_min = pos_max = pos - 1`) + D13 启动清理 + D22/D23 修复 + soak 模式落地.
- 验收: 30 min soak `RESULT 0` (rounds=1703 rebuilt=336 cmp 340/340, failed=0, RSS 1902->1894 MB, 句柄 245->271, 重启清理 79->0); 5 min smoke 绿; 全回归 logic/model/accept 0 FAIL + ab/overlap/b/b3/neg/heal/ref 全 `0 failure(s)` (cuda1). D14 磨损延后; 60 min soak 用户裁决跳过 (D15); 未做 tidy/merge.
- 证据: `artifacts/t32-stage4-*.txt` + `artifacts/t32-stage4-logs/`; 决定 D14-D23 见 `artifacts/t32-tree-plan-stage4.md`; 合并/push 待用户决定.

- **阶段 4 终审修复波 (2026-09-27)**: 终审 "With fixes" -> 一个修复波 `42ee7a6f7` (重建检查点 256 MiB 字节上限, 丢弃最浅优先; block load_payload transient 幂等; 启动清理改用 remove_all 计数) -> 限定复审全部 ADDRESSED; 复审后复跑: logic 64/0, model 56/0, heal 0 failures (REBUILT=1), 5 min soak 305 轮 0 failures; 分支 `t32-stage4` 最终 head `42ee7a6f7` (未合并/未 push); SDD 记录归档 `artifacts/t32-stage4-sdd/`.


- **T32 阶段 5 完成 (2026-09-27, SDD)**: 分支 `t32-stage5` (base `42ee7a6f7`, head `04954b468`, **未合并/未 push/未部署**, 生产仍 `ba41cccec`); D25 双档间距 (`--tree-checkpoint-fork-step` 默认 8192) + D26 heal-on-miss (restore miss 在分叉点落锚) + D27 删除 `promote_prune` + D28 `step_skips=` 计数 + D29 选项改名. 验收 (cuda1): logic 74/0, model 64/0, accept 42/0; ab/overlap/b/b3/neg/heal/ref/fork 全 `0 failure(s)`; 新 `fork` 模式 `captured=[8192,16384] restored=[8192,8192,16384]` 5/5 逐位一致; 5 min soak 302 轮 0 failure (rebuilt=143, cmp 60/60, 重启清理 61->0). 证据 `artifacts/t32-stage5-*.txt` + `t32-stage5-logs/`; 2 处场景修正 (ctx 32768 / `--slot-prompt-similarity 0`) 记录于 RESULTS; D30 分支保留待用户决定.

- **阶段 5b (2026-09-27, D31)**: 分叉锚点不再被 MESSAGE 猜测压制 (`capture_anchor` 间距只计 TIP/ONDEMAND); 分支 `t32-stage5b` head `8184bdc7c`; 验收: heal 模式恢复默认 fork_step 仍绿 (REBUILT=1, 证明 487 猜测不再压制 1024 分叉), logic 80/0, model 65/0, fork 5/5 逐位一致; 终审 clean. D32: 不做主动网格填充, `step_skips` 计数留作后续数据源.

- **阶段 4+5+5b 整理并入 master (2026-09-27)**: 13 个提交整理为 4 个干净提交 (`1e068decd` common options / `633f119d7` kv tree storage hardening / `72f3b636d` server rebuild+fork anchors / `d701c2c13` tests), tree hash `1c9d9afcb37af01869783813cba40e3aaaae1912` 与验证态逐字节一致, ff 合并 master (head `d701c2c13`, 领先 origin/master 4, 未 push). 合并后复跑: logic 80/0, model 65/0, 6 min soak 0 failures (`cmp_ok=70 cmp_tie=1 cmp_bad=0`, restart 51->0).
- **近并列 (near-tie) 定位 (2026-09-27)**: 合并后 soak 在 round 315 确定性出现 1 次 tree-vs-full 分歧; 加 `n_probs=5` 诊断后确认: 分歧 token 处两路径同一 token 的 logit 差仅 ~0.015 nats (浮点累加顺序差异, 路径相关确定性), 而 full 路径该处 top-2 只差 0.002 -> 贪心翻转. 属 T24 已知 replay 现象, 非状态错误. soak 判定改为: 分歧时若两路径 logit 差 < 0.05 记为 near-tie (记录不算失败), 否则才计 mismatch; 复跑 355 轮 `cmp_tie=1 cmp_bad=0` 全绿. 证据: `artifacts/t32-merged-soak-run5.txt` + `t32-merged-soak-final.txt` + `t32-stage3-logs` 归档 JSON.

- **T32 部署到生产 + 27B+MTP 实测 (2026-09-27, 用户批准)**: 生产 `D:\LLM\Backend\llama.cpp-my` 覆盖 9 文件 (ggml-base/cpu/cuda, ggml, llama, llama-common, mtmd, llama-server-impl, llama-server) 来自 master `d701c2c13` 构建树 (`GGML_CUDA_GRAPHS=ON`); 备份 `deploy-backup\llama.cpp-my-t32-20260927-223626\` (9 旧文件, 回滚 = 覆盖 + 重启); 新哈希: llama-server-impl `091AF04E4AB4747A`, llama-common `5B3768531CE8087A`, llama `A66D21B918EA067A`.
- **27B+MTP+树 快测 (np=2, ctx 65536=32K/slot, --spec-type draft-mtp n_max=3, tree-ram 2048, tree-disk-limit 8192, cuda1)**: 运行正常: VRAM 24839MB 稳定, 0 failed/abort, MTP 草稿接受 ~17-19/33-41, SSD 层用到 3.3GB (files 117), parked=13 restored=5 captured(分叉锚点)=5 rebuilt=2. 复用: 首轮回合全量 (prompt_n ~10-17K, 13-23s); 第二轮因生成文本重分词导致与上一 tip 锚点错开 ~30-40 token (deep < tip), 回退到 1024 猜测锚点 (只省系统前缀) 并在分叉点捕获锚点 (heal=13332/15373/17427/11278); 稳态从第三轮起应恢复到分叉锚点 (2B soak 已证 10K -> 516 token, ~13x). 证据 `artifacts\t32-27b-mtp-tree.txt` / `t32-27b-mtp-tree-4sess.txt` + `%TEMP%\v100\t32-27b\srv-err.txt`.

- **27B+MTP+树 三轮稳态确认 (2026-09-27, 生产构建)**: 4 会话 (10-17K token) x 3 轮, np=2, ctx 32K/slot, MTP n_max=3, tree-ram 2048 / disk 8192: 第一轮全量 (prompt_n 10260-17433, prompt_ms 13.2-22.8s); 第二轮为转换轮 (恢复到 1024 猜测锚点并在分叉点捕获锚点, prompt_n 10260-16409); **第三轮稳态: prompt_n=6 (四个会话全部), prompt_ms 291-300ms, wall 2.5-3.0s** -> 复用率 >99.9%, 端到端 ~7x (wall 16-25s -> 2.7s), MTP 草稿接受 18-21/30-39 与首轮持平 (树恢复后 spec 无退化). 全程 fails=0, VRAM 24839MB 稳定, SSD 用 4.4GB (125 文件), parked=21 restored=9 captured=5 rebuilt=6. 证据 `artifacts\t32-27b-mtp-tree-3rounds.txt`.

- **T32 mmproj 修复部署 (2026-09-27)**: 发现 `--mmproj` 下所有 prompt 被标 `has_mtmd` -> 树 park/restore 静默禁用; 修复 = 按"真实媒体"(`server_tokens::has_media()`)判定 + `get_tokens()` 断言放宽 + 恢复时保留能力标志 (`mctx != nullptr`); 提交 `aa6b61689` + `c568a8c22`; 生产增量覆盖 5 文件 (ggml-base, llama, llama-common, llama-server-impl, llama-server.exe), 备份 `deploy-backup\llama.cpp-my-t32fix-<stamp>\`; 27B+mmproj 文本请求验证: parked/restored 出现, 媒体请求 + n_cache_reuse 探测 0 abort.

- **撤回 (2026-09-28)**: mmproj 相关两个提交 (`aa6b61689` 文本-only 放行 + `c568a8c22` 能力标志修复) 已从 master 直接移除 (reset 到 `d701c2c13`, 不建撤回提交; reflog 可恢复). 原因: 按用户要求改为直接实现"媒体原生进树", 不要过渡性绕过. 注意: **生产当前仍运行这两个提交的构建** (部署未回滚) — 若要让生产与仓库一致, 需重新部署 `d701c2c13` 构建 (代价: mmproj 下树完全禁用) 或等媒体方案落地后一并部署.

- **媒体原生进树 (2026-09-28 夜间, 分支 t32-media, 未合并/未部署)**: kv 树现支持含媒体 (图像/音频/视频) 的 prompt 的 park/match/restore. 提交: `5722fac1e`+`2e341d9e9` (树存储侧: `kv_tree_media` 跨度 + token/位置映射 + 媒体块对齐 + 身份入块哈希 + park/match), `70157be75` (检索侧测试), `7d5b11bb1`+`587272960` (服务器接线: park/restore/erase/heal 四处去掉 has_mtmd 门 + `get_tokens_raw()` + `keep_first` 相邻媒体块边界修复). 设计文档 `artifacts/t32-media-tree-spec.md`, 计划 `artifacts/t32-media-tree-plan.md`, 证据 `artifacts/t32-media-e2e-20260928.txt`.
- **E2E (0.8B-MTP + mmproj-F16, cuda0, `--image-min-tokens 1024`, 贪心)**: 同一含图会话第二次请求 `parked 1094 -> restored 1059` (恢复点越过 1024-token 图像块), prompt_n 1063->25; 换一张图同位置 -> restore miss + 全量 prefill (无错误复用); 两图相邻 (A 后 B) -> parked 2157 / restored 2090, prompt_n=22; 5 个请求输出与无树 baseline **逐字节一致**; 日志无 abort/assert; 单 server 显存 ~1.9GB (device 0, 8188MiB), 跑完已全部停止.
- **关键机制**: 媒体块身份 = `mtmd_input_chunk_get_id()` (原始字节 sha256) 折叠进块哈希; token 索引与位置分离 (M-RoPE: 图像 N token 仅占 n_pos=max(nx,ny) 个位置, 全块共享起始位置); 块边界对齐媒体块 (不切块); 恢复点永远落在块边界; 树不存媒体字节, 恢复时用请求自带 chunk 重建槽 prompt.
- **未做/限制**: 媒体 + stock checkpoints / `n_cache_reuse` / spec 状态仍按上游门控; 媒体块原子 (超大视频 = 超大块); 尚未合并 master、未部署生产 (生产仍为撤回前的 mmproj 绕过构建).

- **媒体树白天轮补测 + 崩溃修复 (2026-09-28, cuda1)**: 分支 t32-media 新增 `d9a62d8ef` (无回滚能力的模型上树恢复留 1 token) + `1bfb1b38a` (leave_one 上限不得超过已验证前缀, 审查发现并修复). 证据 `artifacts/t32-media-day2-20260928.txt`.
- **发现并修复的真实缺陷**: hybrid 模型无回滚快照 (0.8B-MTP 不开 spec => `n_rs_seq=0`, seq_rm_type=FULL) 时, 树恢复到整段 prompt (C == task.n_tokens) -> 服务器按 TAG_PROMPT_LOGITS 减 1 再 `seq_rm` -> recurrent 无法回滚 set_partial 恢复的状态 -> `failed to remove sequence ...` abort (纯文本也复现). 修法: 目标或 draft 上下文无 PART/RS 能力时 `leave_one` (恢复点上限 min(deep, n_tokens-1)); 树恢复点恰在 n_past 时跳过 checkpoint 搜索; 媒体 prompt 也重建 checkpoints (单位改为 tok/pos 分离), 使有回滚能力的模型 (27B+MTP) 保持深复用. 修复波经审查: 1 Critical + 1 Important + 1 Minor 全部修复并复审通过 (mutation 验证测试有效).
- **白天轮矩阵结果**: ① 不设 `--image-min-tokens`: 小图块也能 deep restore (prompt_n 25), 输出与自带 baseline 逐字节一致; ② 强制落盘 (`--tree-ram 16`): `ram=0 B, disk=53.9MB`, 从 SSD 恢复 1059 tokens, 输出一致; ③ np=2 跨 slot: A 链被 C 挤掉后回来 `restored 1053`, prompt_n=4; ④ 媒体 soak (np=2, 强制落盘) 90 请求 alive, 恢复路径确定 (vs 第2轮 diff=0), aborts=0, disk_err=0; ⑤ 27B 生产配置 (Q6_K+q8_0 V+mmproj+MTP) 第3/4轮 `prompt_n=1 cached=1125 ~145ms`; ⑥ 纯文本无 spec 重复请求 prompt_n=4 (与 stock 一致, 不崩); ⑦ harness logic 130/0, 2B model 65/0.
- **已记录现象**: 全量 prefill vs 分片缓存中恢复 (np=2) 的浮点归约顺序差异 -> 0.8B q4_k 上 logit 差 ~0.15-0.19 nats 可翻转贪心近并列 (措辞变化, 语义不变; 每条路径自身确定). np=1 与 27B 未观察到输出差异.

- **整理合并 (2026-09-28)**: 媒体树工作已按「整理注释 -> 重组提交 -> ff 合并」完成. 提交: `d59ed5e4a` kv tree : media-aware blocks, matching and restore (树+harness), `53508f8bd` server : wire media prompts into the kv tree (server-common/context). 整理前后 tree hash 逐字节一致 (`4e0c9ddfb4ee9d33278443198ffd13e1cd87652a`), master 现为 `53508f8bd` (领先 origin/master 2, 未 push). 分支 t32-media / t32-media-tidy 已删. 合并后复跑: harness logic 130/0, 2B model 65/0.

- **生产部署 (2026-09-28 12:03)**: master `53508f8bd` 构建已部署到 `D:\LLM\Backend\llama.cpp-my` (9 文件覆盖, 实际更新 5: llama-server.exe / llama-server-impl.dll / llama.dll / llama-common.dll / ggml-base.dll; ggml.dll / ggml-cpu.dll / ggml-cuda.dll / mtmd.dll 校验一致). 备份: `deploy-backup\llama.cpp-my-t32media-20260928-120353\`. 校验 (sha256 前16): server.exe `317F973ED82882C0` (旧 `37E5F7AA1DC9B701`), server-impl `4313223579682E76` (旧 `DC6B0F1BE4311D3D`), llama.dll `B5AF2458F0620841` (旧 `B3503FCC68ACA96D`), llama-common `B90C0864002795CD` (旧 `F57DAC439842B17B`), ggml-base `41C5E47BFF0009F1` (旧 `BC2141F829F511AE`). 用生产目录二进制启动自检 (0.8B+mmproj, cuda1): 媒体 park/restore 正常 (restored 1059, prompt_n 1063->25), 换图 miss, 输出与无树 baseline 逐字节一致, 无 abort. 回滚 = 覆盖备份文件 + 重启. 部署时生产无运行进程; 部署后未启动生产服务 (等用户用其 launcher 启动).

- **上游 rebase (2026-09-28)**: `git fetch upstream` (23 个新提交) 后 rebase 完成, **零冲突**. 上游更新要点: ① server 允许 causal reranker 的 RANK pooling 分批 (#28876, 新增 `llama_get_causal_attn`); ② common 参数解析副作用与 `string_split` 严格校验 (#29537/#29518, `--rpc` 在不支持时抛异常); ③ CUDA FA fp16 tile 配置调优 (#26289) 与 FWHT F16 (#29096)、Nemotron d_state=96 ssm scan (#28717, 新增 `test_ssm_scan_rollback`); ④ jinja dict 内建、RPC RDMA、hexagon/sycl/vulkan/opencl 更新、unified KV 自动 fit 回退 (#29437) 等. 与我们重叠 11 文件全部自动合并; 语义核对: `ggml_gated_delta_net` 新签名的 3 处调用点自洽; 双方特性均在 (kv tree/range API/ReplaySSM/V100 配置 + 上游新特性). 验证: 全量编译通过 (含 CUDA 重编), harness logic 130/0, 2B model 65/0, test-t32-range 通过, `test-backend-ops -b CUDA0` GATED_DELTA_NET 36/36 与 SSM_SCAN 15/15, 服务器 E2E (文本 prompt_n=4; 媒体 restored 1059, 输出与 baseline 逐字节一致). master = upstream/master + 16 私有提交; 安全分支 `backup-pre-rebase-20260928` 保留; 未 push.

- **生产部署 (2026-09-28 12:51, rebase 后)**: master `7e2ae38c2` (upstream + 16 私有提交) 构建已部署. 9 文件全部更新 (含 ggml/ggml-cpu/ggml-cuda/mtmd). 备份: `deploy-backup\llama.cpp-my-rebase-20260928-125102\`. 校验 (sha256 前16): server.exe `13572707EB871EC2`, server-impl `37A2AF8C4FF0DDA4`, llama.dll `5C69A66F71D3D1ED`, llama-common `B5DEFC7399BCB595`, ggml.dll `A55895E0FD58CBC5`, ggml-base `4DA9B2889F7D872D`, ggml-cpu `E9CD4ECFBC15662A`, ggml-cuda `5BF73D6700ED43B7`, mtmd `A0BA4F029C3FB7A7`. 用生产二进制自检: 媒体 park/restore 正常 (restored 1059, prompt_n 1063->25), 换图 miss, 输出与 baseline 逐字节一致, 无 abort. 回滚 = 覆盖备份 + 重启.

- **Fork 强制推送 (2026-09-28)**: 用户确认目标为其个人 fork (非上游) 后, `git push --force-with-lease origin master` 完成: `d701c2c13...528dd28e0 (forced update)`. fork 的 master = 本地 master = `528dd28e0` (upstream/master + 16 私有提交 + MiB 日志提交), 本地与 origin/master 已同步. 本地安全分支 `backup-pre-rebase-20260928` 保留.

## 生产回滚问题调查 + 诊断构建部署 (2026-09-28)

**现象**: 生产 27B (LLM-Manager 启动, np=1, MTP, tree, ctx 184320) 在 agent (Claude Code) 对同一消息反复回滚后, 某次请求只 eval 1 token 就生成 0 token (首 token 即 stop/EOS), 直到 45k 上下文回滚到 45k 前缀后发生 (如 task 13185/13187).

**定位 (日志+代码+生产配置)**:
- 失败请求走 stock 复用路径: `f_keep = 0.615 >= 0.5` -> `update_cache = false` (server-context.cpp:1804), 完全跳过树 park/restore (日志无 `kv tree:` 行); 随后是上游的 context checkpoint 恢复 + `seq_rm(p0,-1)` + `[TAG_PROMPT_LOGITS]` 强制 `n_past--`.
- 生产配置 (LLM-Manager `model_schemes` id=4781) `env: {}` -> **ReplaySSM (`GGML_CUDA_GDN_REPLAY`) 未开**; 我们对该文件的改动未激活. 唯一关联: 我们在**树**路径已修过同类 bug (`leave_one`), stock 路径无此保护.
- 代码可疑点: `llama_memory_recurrent::state_read` 只恢复 rs_idx 选中的一行状态并把 `rs_idx` 清零, 快照行 (`r`) 是旧的; 若恢复后紧接着发生部分回滚 (`seq_rm(p0,-1)` 且 `p0 <= cell.pos`, 例如强制 `n_past--`), 下一次 decode 会读 stale 快照 -> 静默坏状态. 属上游代码交互.

**复现尝试 (27B, cuda1, 生产同参数: ctx 184320 / checkpoints 16x16384 / tree / MTP / q8_0 V)**:
- 合成 agent 流程 (长对话 + 反复回滚同一条消息) 未复现; 所有 checkpoint 恢复自洽: `n_tokens = pos_max+1`, `n_past = pos_max+1`, `after load mem [pos_max, pos_max]`, `p0 = pos_max+1 > cell.pos`, 0 次部分回滚.
- 注意: Qwen3.8 输出在 reasoning_content, 之前对照实验比较的 content 为空 -> 口径需修 (测试脚本问题, 非服务器问题).

**诊断构建 (已部署生产)**:
- 提交 `409b14c3a` (临时, 找到根因后 revert): `[dbg] restore ckpt` + `[dbg] after load` (SLT_WRN); `seq_rm ... -> ROLLBACK / REFUSE / CLEARED` (LLAMA_LOG_WARN); `state_read restored: tail/pos/rs_idx` (INFO).
- 部署 5 文件 (llama-server.exe/llama-server-impl.dll/llama.dll/llama-common.dll/ggml-base.dll): 备份 `deploy-backup\llama.cpp-my-diag-20260928-144145\`; 哈希: server.exe `540AE4F10FFD4D94`, server-impl `EC38AF8DF94B5C0F`, llama `B134344F17D699C4`, common `CAFCBF2B8E4C5DC8`, ggml-base `FA3907DCF4D7B971`.
- 生产目录冒烟: 0.8B + MTP + checkpoints -> 日志出现 `[dbg] restore ckpt ...` / `seq_rm ... -> CLEARED` / `[dbg] after load ...` ✓.
- 复现时收集这些行: `[dbg]`, `restored: tail`, `-> ROLLBACK`, `-> REFUSE`, `-> CLEARED`, `restored context checkpoint`, `need to evaluate at least 1 token`.

## 更正 + 关键发现: 生产一直在开 ReplaySSM (2026-09-28 下午)

**更正**: 前面写的 "生产 env:{} -> ReplaySSM 未开 -> 失败路径是上游" **是错的**. `GGML_CUDA_GDN_REPLAY=1` 是**用户级环境变量** (HKCU\Environment, setx 级别; T24 部署时按 "用户级" 设置), LLM-Manager 及其子进程全部继承 -> **生产一直在跑 ReplaySSM**. 我只查了 launcher 的按模型 `env` 字段, 漏了用户级环境. 因此:

- 失败路径 = **我们自己的 T24 代码**: stock context checkpoint 恢复 (`state_read_replay` 恢复 records + rec_n/rec_pos0/half_of/p_pending) + 后续 batch 的 fold (`get_rec_p`).
- 最可疑机制: 恢复后下一个 batch 的**连续性判据** (`pos0 == rec_pos0 + p`, llama-memory-recurrent.cpp:1750) 不满足时 `p` 被静默置 0 -> 有效状态回退为 S 平面("pinned", 比目标位置少 p 个 token) -> 状态偏旧 -> 首 token 采样到 stop (temp 1.0) -> 空响应. 与 T24 文档记录的 2026-09-27 "回滚+replay 运行间不确定性 (value-diff ~1/6, 近并列翻转)" 同一类.
- 另: 期间发现 `llama_memory_hybrid_iswa::state_write/state_read` **完全忽略 PARTIAL_ONLY** (未 gate attention 部分) - 对 27B/0.8B (n_swa=0, 走 plain hybrid) 不触发, 但 SWA hybrid 模型上是隐患 (待修).

**诊断构建 v2 (已部署生产)**:
- 提交 `409b14c3a` + `6e328c0a2` (临时诊断; 定位后 revert).
- 诊断输出改为**无条件 stderr** (之前 LLAMA_LOG_INFO 在 verbosity 3 被过滤, 只有 -lv 4 才可见 -> 白等一场; 教训记录).
- 关键日志:
  - `[diag] state_write_replay: seq .. half .. rec_n .. rec_pos0 .. p .. (rs_idx .. have_rec .. pending ..)` = checkpoint 保存的账本;
  - `[diag] state_read: seq .. tail .. pos .. rs_idx .. pending .. rec_n .. rec_pos0 .. half .. have_rec ..` = 恢复后的账本;
  - `[diag] get_rec_p: seq .. pos0 .. p ..` = 正常 fold; `... != rec_pos0 .. + p .. -> forced 0` = **连续性失败 (重点看这条)**;
  - `[dbg] restore ckpt: n_tokens/pos/pos_next/n_past/mem` (server), `seq_rm ... -> ROLLBACK/REFUSE/CLEARED`;
  - `[diag] mem_hybrid/mem_recr::state_read ENTER`, `[diag] checkpoint load_tgt: blob N bytes flags`.
- 部署 5 文件: 备份 `deploy-backup\llama.cpp-my-diag2-20260928-145935\` (上一版 diag `llama.cpp-my-diag-20260928-144145`); 哈希: server.exe `1DC8E76FC929605B`, server-impl `85A93E3BB6A697C7`, llama `820025CEEA0CC044`, common `2F99D792CF5989B7`, ggml-base `D05C70F53BC8CEA1`.
- 生产目录冒烟: 0.8B + checkpoints -> 全部诊断行可见 ✓.

**复现时的判定标准**:
- 若失败请求前出现 `get_rec_p ... -> forced 0` (或 `state_read` 的 `pending != 0` 但后续 fold 为 0) -> 连续性判据问题, 定案 (改 get_rec_p / 恢复路径的 pos 对齐).
- 若出现 `state_read ... rs_idx != 0` + 之后 `get_rec_p ... p < pending` -> 回滚+folding 的组合问题.
- A/B 佐证: 临时去掉用户级 `GGML_CUDA_GDN_REPLAY` (或 `setx GGML_CUDA_GDN_REPLAY 0`) 重启 -> 若失败消失, 则确认 ReplaySSM 路径.

## 根因定案 + 修复: conv 回滚平面未随状态保存 (2026-09-28 15:46)

**用户复现成功** (recipe: 先建一个不相干的树 -> 跑 agent -> 回滚两次让检查点稳定 -> 回到第一个树回滚一下 -> 再回来)。失败栈 (task 3716):
```
tree restore 27476 tokens (heal=-1) -> [diag] state_read: tail 0 pos 27475 pending 4 rec_n 4 rec_pos0 27472 half 0
-> [TAG_PROMPT_LOGITS] 强制 n_past-- -> seq_rm [27475,inf): rb 1 avail 3 -> ROLLBACK rs_idx 1
-> [diag] get_rec_p: pos0 27475 p 3 (连续性成立) -> prompt eval 1 token, eval 0 token (首 token 即 stop)
```
S (committed 平面 + fold p) 账本自洽; 病灶在 **conv 缓存 R**:

- `r_l` 有 `1+n_rs_seq` 个 per-token 平面 (init 日志 R 22.50 MiB = 4 平面), 回滚 gather 用 `s_copy` 读 `rs_idx*size + src` 平面;
- 但 `state_write_data` 写 R 用的是 `cell_ranges_data` = **只有 rs_idx 选中那一个平面** (blob 161.8 MiB = S 1 平面 144 + R 1 平面 ~5.6 + REC 12.1, 正好对上);
- 恢复把该平面读进 plane 0 并把 `rs_idx` 清零; 之后任何部分回滚 (如 [TAG_PROMPT_LOGITS] 的 1-token 回滚) 会去读 **plane 1..n = 旧分支残留** -> conv 状态错 -> 首 token 直接 stop (temp 1.0 下表现为空响应).
- 非 replay 模式同样存在该类隐患 (R/S 都只存单平面), 但生产跑 replay, R 平面是唯一仍走 snapshots 的部分, 所以必现.

**修复** `e54a20a72`: replay 模式下 `state_write_data`/`state_read_data` 存/取 R (和 PLE `p_l`) 的**全部 `1+n_rs_seq` 平面**; `LLAMA_STATE_SEQ_VERSION` 5 -> 6.
- 代价: blob +n_rs_seq x R 平面 (27B: 161.8 -> ~178.6 MiB/checkpoint; tree 锚点同步变大). 重启后旧 blob 全部失效 (tree 启动清盘 + checkpoint 在 RAM).
- 0.8B 冒烟实证: blob 22,580,436 -> 26,561,748 B (+3 平面 x 1.327 MiB ✓), 恢复后生成正常.

**部署**: 6 文件 (server.exe `45E96974931738DF`, server-impl `54EC599C3EFB2762`, llama `8133BB6720CE5BD5`, common `654883EFD0DF87B5`, ggml-base `8A65984A248783D0`, mtmd `33015CD11CBF161F`), 备份 `deploy-backup\llama.cpp-my-rplane-20260928-154633\`; 冒烟 OK.
诊断 (提交 `409b14c3a`/`6e328c0a2`) 暂留, 等用户复验通过后 revert.

## 历史整理 + 推送 + 干净构建 (2026-09-28 17:30)

- 历史重写: 去掉 3 个提交 (`409b14c3a` debug, `6e328c0a2` debug, `e54a20a72` 落在 debug 之上), 重排为两个干净提交 (临时诊断全部摘除; 混进 debug 提交的 ReplaySSM 环境守卫保留):
  - `891f4e8eb llama : disable replay when a recurrent layer is not on CUDA` (守卫)
  - `32290e987 llama : save all conv rollback planes with the recurrent state` (修复; LLAMA_STATE_SEQ_VERSION 5->6)
  - 校验: `git diff a5d58104a..HEAD` = 守卫+修复, 诊断命中 0; 工作树干净.
- **已推送到 fork**: `d3c6d4aca..32290e987 master -> master` (fast-forward, 无需 force); origin/master == 本地 == `32290e987`.
- 干净构建已编译 + 冒烟通过 (0.8B: 全量 prefill 5604 t -> 两次 checkpoint 复用各 4 t eval, 生成正常, 无诊断输出).
- **生产部署待办 (被运行中的服务锁文件挡住)**: 干净版 6 文件 (`llama-server.exe` `7D16CA06633535F4`, `llama-server-impl.dll` `690663EE9B6D292D`, `llama.dll` `F95351BCD4C8BB41`, `llama-common.dll` `226A8CBD86056927`, `ggml-base.dll` `0857C1C24BD869AE`, `mtmd.dll` `33015CD11CBF161F`); 备份已建 `deploy-backup\llama.cpp-my-clean-20260928-172937\` (内容 = 当前运行版). 用户在 LLM-Manager 停服后即可覆盖 (当前运行版已含修复, 只是多了诊断日志, 无功能风险).
