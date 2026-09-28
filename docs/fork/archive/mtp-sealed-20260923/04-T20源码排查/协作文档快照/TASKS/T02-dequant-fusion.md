# T02: dequant + fp16 MMA 融合 GEMM (sm70)

状态: APPROVED / 下一个执行 (analyst 2026-09-22 整理)
优先级: P0 (prefill 唯一超车机会)
预期收益: 整机 pp512 +5-10% (盈亏平衡 ~66 TF; 收益上限 +16.6%)

## 目标

把 prefill 的 "反量化 -> cuBLAS" 两趟路径换成 fused dequant + sm70 fp16 MMA 单 kernel:
删掉 dequant pass (73GB/ubatch, 88.5ms) 并省掉 GEMM 的 fp16 重读 (52.1GB)。

## 事实基础

- cuBLAS 墙 (T01): gate/up 84.2 TF (L1 受限, occ 12.5%), down 100-107, qkv 99.5, lm_head 95.1, ssm_gate 68.3, kv 53.0
- 盈亏平衡 (dequant @825GB/s): gate/up ~66 TF, down ~75, qkv ~80, lm_head ~77
- ncu: gate/up Tensor 72.2% / L1 67.4% / DRAM 29% -> 读量化权重 L1 流量少 2.6x 是唯一结构性优势
- 推论 (Gate B 的目标锚点): 若 L1 瓶颈解除, gate/up 的 tensor 利用率有望从 72% 升到 85-90%,
  即该形状 ~90-100 TF (对应整机 prefill 约 +10-15%); 这是"反超 cuBLAS"的量化依据

## v1 失败教训 (已花过时间, 必须避免)

- occupancy 6.25%: 每块 50.7KB smem -> 1 block/SM; dequant/mma/A-load 三阶段被 __syncthreads 串行
- padding: 行距取 8 half 的倍数 (BKP = BK+8), 否则 wmma fragment 加载全 bank conflict (修好后 mma 段 3.4x)
- BM=64 -> B tile 被每个 M-tile 重反量化 -> 量化权重读 8 次 (585MB > baseline 429MB)
- 物理约束: BM=128 + BN=128 + BK=256 双缓冲 = 256KB > 96KB smem, 放不下
- 正确性口径参考: v1 sum_rel 2.0e-7, max_rel 0.034 (需解释 max 出现在哪些元素)

## Stage 1b (timebox 1.5 天)

- **Gate A (0.5 天)**: 用 `mmf.cuh` 骨架 + **预反量化好的 fp16 权重** (不含反量化), gate/up (512, 17408, 5120) 形状
  - 先证明 sm70 上骨架能达到 **>= 80 TF**; 达不到 -> REJECTED (骨架到不了 cuBLAS 水平, 融合无望)
- **Gate B (1 天)**: 加 Q6_K 反量化 B tile loader
  - >= 84 TF: 优秀, 集成; 70-84: 条件集成 (按形状开关); 66-70: 报告数字交 analyst; < 66: REJECTED
- 允许简化: 只做 FFN 三个形状 (65% FLOPs); 不做 stream-k; 先把 occupancy / tile / 双缓冲三点做对
- 正确性: 与 (dequant + cuBLAS) 对比给 NMSE 和 "|ref| > 1 的 max_rel"

## 验收

- 整机 pp512 >= 1150 (基准 959.7), PPL 在门槛内, flag 关闭零回归
- 报告必须含: 融合 kernel TFLOPS、pp512/4096/8192、PPL、ncu Tensor/L1/occupancy 对照

## 风险

- 融合效率低于 cuBLAS 时, ub 越大越亏 -> 集成按形状/ub 分档启用
- Volta 无 ldmatrix, 手工 permute 的指令开销会吃掉收益 -> Gate A 就要暴露
- 若 Gate A 失败, prefill 上限退回 ~1000 (仅 T07 小项)

---

## Result (implementer, 2026-09-22): **Gate A FAILED -> REJECTED**

- fp16 骨架 (BM=BN=128, BK=32, 2 级 smem, wmma 16x16x16, k-blocked 布局): **57.5 TF**
  (C[17408,512]) / 54.3 TF (C[512,17408]); 正确 (NMSE 2.6e-12); 门槛 80 TF 未达。
- mma 路径单独 (mode 32) = 1.472ms, **已慢于 baseline dequant+cuBLAS = 1.404ms**
  -> 即使 staging 全隐藏、dequant 零成本也赢不了; 不存在推进 Gate B 的理由。
- 同形状 cuBLAS: 默认 77.4 TF, ALGO8-15 83.0 TF, cublasLt heuristic#0 84.3 TF。
- 详细数字/分析/可复用结论 (k-blocked 布局把 staging 从 574GB/s 提到 1.7TB/s 等) 见 RESULTS.md;
  产物在 `v100-collab/artifacts/` (4 个 .cu + ncu log)。
- 结论: T02 从队列移除。**t01 的"调用侧无空间"结论需修正** (见 RESULTS 第 4 节), 建议新开 T08 (cublasLt 集成)。

---

## Analyst 复核 (2026-09-22): REJECTED 接受, 结案

- 否决逻辑成立且优雅: `mma 路径单独 1.472ms > baseline (dequant+cuBLAS) 1.404ms`,
  即"即使 staging 全隐藏、dequant 零成本也赢不了", 不需要再做 Gate B
- 补充证据 (RESULTS 第 3.4 节): dequant 的整数指令要吃掉每 SM 约 44% 的 issue 预算 (4.4M / 10.07M),
  会把融合版压在 50-67 TF, 恰好卡在 66 TF 盈亏平衡线附近或之下 -> 双重否决
- Gate 设计成功: 0.5 天杀掉 1.5 天的赌注, 这是本任务最有价值的产出
- 可复用发现 (已归档): k-blocked 布局 (staging 574GB/s -> 1.7TB/s), grid.x = token 维 (+6.5%),
  wmma 16x16x16 与 tile m8n8k4 无差, 纯 mma 原语峰值 97-99 TF
- 状态: **REJECTED (analyst 确认)**; 不再重启, 除非硬件/量化格式改变