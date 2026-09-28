# T01: cuBLAS GEMM 效率核查 (make-or-break)

状态: VERIFIED / DONE (analyst)
优先级: P0
预期收益: 决定 prefill 能否翻倍 (950 -> 1900); 若成立, 无需任何新 kernel 也能 +20-25%

## 背景

报告 (V100-dequant-vec-优化实施报告.md 2.1) 实测: pp512 里 cutlass fp16 GEMM 占 428ms/800ms (ncu, 锁频 1246MHz),
折算 75.6 TFLOPS = V100 fp16 峰值 (125 TFLOPS) 的 60%。
这异常低: 同样的卡跑大 GEMM 应该能到 85-95%。当前无法判断是:
(a) cuBLAS/CUDA 12.x 在 sm70 上就这水平 (墙), 还是
(b) llama.cpp 的调用方式浪费了性能 (输出 F32、CUBLAS_COMPUTE_32F、单流、无 workspace、启发式选核差)

## 方法

### 1. standalone 微基准 (必须做)

用 cuBLAS 和 cuBLASLt 各写一个最小 harness, 只测下面形状 (A=fp16 权重, B=fp16 激活, C=fp32, compute=32F),
同时测一组 **C=fp16 / compute=16F / fp16 累加** 作为候选 (用户已批准可接受微小漂移, 门槛见 PROTOCOL.md):

| 形状 (M, N, K) | 对应 op | 说明 |
|---|---|---|
| 17408, 512, 5120 | ffn_gate/up (ub512) | 报告测的"FFN 形状" |
| 17408, 2048, 5120 | ffn_gate/up (ub2048) | 验证 N 的影响 |
| 5120, 512, 17408 | ffn_down | K=17408 |
| 10240, 512, 5120 | attn_qkv / SSM in-proj | |
| 248320, 512, 5120 | lm_head | N 小 M 极大 |
| 4096, 4096, 4096 | 对照 | V100 参考上限 |

输出: 每个形状的 TFLOPS、选中的 kernel 名 (cublasLt 可打印 heuristic)、以及 cuBLASGetVersion。

### 2. ncu 抓 llama.cpp (必须做)

对 pp512 的 cutlass GEMM kernel 抓 1-2 个 launch:
- `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` (TC 利用率)
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__throughput` / achieved occupancy
目的: 区分"TC 利用率上不去"还是"根本没有好 kernel"。

### 3. 快速对照组 (可选, 时间够再做)

`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16/f32` 环境变量对比; 以及 CUTLASS sm70 fp16 官方 example 跑同形状。

## 决策规则

| standalone 结果 | 结论 | 后续 |
|---|---|---|
| >= 95 TFLOPS | llama.cpp 调用侧问题 | analyst 给修复方向; 预期 prefill +20-25%, T02 降级 |
| 70-80 TFLOPS | cuBLAS 在 sm70 是墙 | T02 (融合) 价值最大; 也可试 CUTLASS 参考实现 |
| CUTLASS/手写 > 95 TFLOPS 且 cuBLAS ~75 | 换 kernel 路线 | T02 目标提高: 融合 + 自己写 mma |
| 16F (fp16 累加) 明显更快 (>= 90 TFLOPS) 且 PPL 在容差内 | 直接可用的调参收益 | analyst 开 T07 集成 (改 llama.cpp compute type / 输出类型) |

## Analyst 注记 (2026-09-22, 待 T08 仲裁)

- T02 Gate A 的 harness 测出同形状 (gate/up): `GemmEx 默认 77.4` vs `cublasLt heuristic#0 84.3` (+9%),
  与本任务 ABAB 的 `默认 84.2 vs Lt 84.1` (无差) **矛盾**。两者必有一错 (layout 差异或冷热状态)
- 在 T08 Step 1 用权威 ABAB (同 harness、同 layout、预热交错) 仲裁之前, 本任务"调用侧无空间"的结论
  仅对 workspace 与显式 algo 112 成立, **对 cublasLt 部分暂挂**
- 注意: 模型侧实测 GEMM 速率约 77 TF, 与"默认 77.4"一致, 与"默认 84.2"不一致 -> 需要解释 (可能来自调用 layout)

## 交付物

RESULTS.md 追加一条, 含: 形状表 + TFLOPS + ncu 指标 + cuBLAS 版本 + 结论与建议。

## 验收

- 所有数字可复现 (附 harness 源码路径与编译命令)
- ncu 指标必须写全名, 不允许只写"TC 利用率"
- 不允许修改模型数值路径
