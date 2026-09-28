# REFERENCE: 1Cat-vLLM (V100/SM70 vLLM fork) 分析结论

来源: `D:\LLM\Backend\src\1Cat-vLLM` (v1.5.0-706, 2026-09-22), analyst 分析于 2026-09-22。
他们的配置: 4xV100 (72-SM PG503-216 16GB), Qwen3.8-27B-FP8 TP4, chunked prefill M=8000,
FA-V100 + XQA, MTP4 / DFlash2, 大量 SM70 专用 kernel (TurboMind 884 / Marlin / QPN8 / CUTLASS)。

## 0. 对我们结论的独立验证 (重要)

- 他们的整体架构 = **大 M 预反量化到 fp16 workspace + cuBLAS/CUTLASS**; 融合 kernel 只留给
  decode / small-M (<=32) 与 tail 形状。与我们的 T02/T09 否决方向完全一致。
- 融合 vs 反量化+cuBLAS 实测: FP8 M=3920 输 1.33x, M=7840 输 1.46x; AWQ M=4096 输 1.47-1.54x。
- 融合 kernel 的 NCU 根因: 248 regs / 65.5KB smem / 1 CTA/SM / 12.5% occupancy /
  62% 无 eligible warp / DRAM 仅 9.4% -> 与我们的"staging/发射受限而非 DRAM"判断一致。
- 他们的反量化约 660 GB/s, 我们 825 GB/s (我们更快)。
- 注意: 他们的 crossover 在 M~3920 (M<=32 融合仍是默认, 那是 decode 带宽受限场景); 未测 M=512。
  所以 ub512 下"融合输"仍是我们自己的结论。

## 1. 可借鉴 (按价值排序)

### 1.1 大 M / chunked prefill 是第一杠杆 (支持 T04)

- 他们 chunked prefill M=8000, dequant 只占 2.0% 时间; 我们 ub512 占 **16.6%** (88.5/532ms)。
- 他们每卡 prefill ~1290 t/s (还是 72-SM 弱卡); 我们 ub2048 实测 1152
  -> **差距主体是 batch 大小, 不是 kernel**。
- 这也解释我们的长上下文数字: pp4096@depth32k 649 / pp32768 775 (dequant 每 token 摊销大)。

### 1.2 长上下文 attention 是新的候选目标 (先侦察)

- 他们的 FA-V100 (D256/GQA6 = 我们的形状): prefill 29-38 causal TF/s; 长 prefill 专用核 77 logical TF/s;
  相对 generic FA2 的 split-D/N32 改进 1.23-1.6x; full-model 64K prefill -15.8%, 128K -22.4%。
- 我们的线索: pp4096@depth32k 比 pp4096 慢 -31% (差 1935ms), 粗反推 attention ~14.5 TF/s。
  只要 nsys 侦察确认, 有 ~2x 空间 -> 长上下文潜在 +5-15%。
- 可移植的技术点: BLOCK_M=64 / BLOCK_N=32 split-D (切成 4 个 D64), 8 warps 四组;
  **N32 online softmax 顺序 (质量要求, 他们 N64 合并 softmax 因换 token 被拒)**;
  Volta TT PV 映射 (V 当 K-by-D 直接消费, 省 128 LDS.U16 + 64 PRMT);
  双 PV register fragment; 长 chunk 的 gather-to-dense workspace。

### 1.3 cuBLASLt pinned 配置 (已被 T08 覆盖, 记录以免重复)

- 他们大形状用: 列主序物理权重 + `algo id=21 / tile=24 / split=1 / workspace=0`;
  但 M=16 的 sweep 说默认已最优。
- 我们 T08 Step 1 真实方向 ABAB: cublasLt 加权 **-1.11%** -> 已否决。
  当前 llama.cpp 的 `GemmEx(OP_T,OP_N) + TF32_TENSOR_OP_MATH` 已是三者最优。**不要重复尝试。**

### 1.4 CUTLASS 128x256x32 fp16 s884 (备用, 当前不投入)

- 配置: CTA 128x256x32, warp 64x64x32, mma 8x8x4, 2 stages, no split-K, swizzle 8, align 8, 220 regs。
- 他们 M=8000 实测比 cuBLAS 快 3.5-5.7% (bitwise 相同); 只放行 M∈[8000,8192]。
- 我们: 需 vendor CUTLASS + ub 提升到 2048+ 才有意义, 收益 3-5% -> 记录备用, 不投入。

### 1.5 MTP draft 词表切片 (非 kernel, 用户侧产品改动)

- 他们把 draft LM head 从全词表切到 131K 静态 / 98K 动态 (由 prompt top-k 构造):
  80.1 -> 100.6 tok/s (**+25%**); 原因: draft 开销 75% 在 LM head。
- 用户生产 MTP3 = 42.7 t/s; 属产品级改动, 不在本工程范围, 仅记录。

## 2. 不要借鉴 (他们已证伪)

| 项 | 他们的结果 |
|---|---|
| TurboMind "884" s884 融合 kernel | prefill 全面输 cuBLAS: AWQ M=4096 输 1.47-1.54x; FP16 M=4096 输 1.38-1.4x |
| SM70 Marlin | 比 TurboMind 还慢 (decode 也输), 已被他们弃用 |
| 侧流/双 workspace 反量化重叠 | regressed (与我们 +0.5% 一致) |
| fp16 累加 | 快 10.6% 但质量不过关 (仅 12.5% 精确元素) -> 拒绝; 与我们的 PPL 门槛同思路 |
| flashinfer-sm70 | 只是 WMMA 探针, 不是 attention 后端 |

## 3. 硬件校准

| 项 | 他们 (PG503-216 16GB) | 我们 (SXM2-32GB) |
|---|---|---|
| SM 数 | 72 | 80 |
| 无节流峰值 | 112.8 TF @1530 | 125 TF @1530 |
| 实测天花板 | 93.4 TF (持续时钟 1260-1275MHz) | 同功耗机制 (真实负载 1447MHz, T08 实测) |
| cuBLAS 最好 | 83.45 TF (N=8192 方阵) | 84-107 TF standalone |
| in-model GEMM | 86-90 TF @M=8000 | 77-79 TF @M=512 |

结论: 我们的卡确实更强; 双方 in-model GEMM 都已被 cuBLAS 吃干 (他们的 CUTLASS 也只多 3.5-5.7%)。

## 4. 方法论 (与我们一致, 值得保持)

- 每个优化要求 bitwise 或 token-hash 相同; "更快但数值不同" 一律拒绝 -> 同我们的 PPL 门槛
- 每个 kernel 有 rollback env gate
- 按形状特化 + 缓存 (exact shape dispatch); 大 M 静态 heuristic, 只有小 M 才 autotune
- 质量事故记录: MTP + GDN 混合模型的 recurrent state 边界重复 bug
  (acceptance=1.000 是 bug 信号不是通过) -> 与我们的 MTP3 长输出检查相关
