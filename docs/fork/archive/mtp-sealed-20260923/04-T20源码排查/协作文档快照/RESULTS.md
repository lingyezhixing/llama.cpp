# RESULTS: 结果日志 (追加式)

格式要求 (每条): 日期 | 任务 | git hash | patch | 命令 | 数字 | 时钟/功耗 | 结论

> **本文件只作原始时间顺序日志 (追加式)。当前状态请看 BOARD.md / STATUS.md; 环境细节看 ENVIRONMENT.md。**

---

## 2026-09-22 | 基线 (来自 V100-dequant-vec-优化实施报告.md, analyst 转录)

- git: 4046f5c8c + `D:\LLM\Backend\patches\v100-dequant-vec.patch`
- 命令: `llama-bench -ub 512 -fa on -ctv q8_0, CUDA_VISIBLE_DEVICES=1`
- 数字: pp512 950 / pp4096 935 / pp8192 908 / pp32768 775; tg128 26.64; PPL 4.3572
- 时钟: prefill 1522MHz, 277W/300W, SM util 100%
- 关键边界: 跳过 dequant pp512 1242; 双 stream +0.5%; FORCE_MMQ -41%

---

## 2026-09-22 | T01 cuBLAS GEMM 效率核查 | DONE (待 analyst 复核)

- git: 4046f5c8c + 工作区 `v100-dequant-vec.patch` (llama.cpp 侧未改动, T01 只测量)
- 环境: cuBLAS 120901 (CUDA 12.9), V100 sm70, 驱动 581.80
- 时钟/功耗采样: **SM 1530MHz 满频, MEM 877MHz, util 99-100%, 150-278W** -> 无降频
- 源码/命令:
  - 形状表 + algo 扫描: `%TEMP%\v100\t01_bench.cu` -> `nvcc -O3 -arch=sm_70 -o t01_bench.exe t01_bench.cu -lcublas -lcublasLt`
  - 预热 + ABAB 对照: `%TEMP%\v100\t01_ab.cu` -> `t01_ab.exe`
  - ncu: `ncu --clock-control none --cache-control all -k regex:Kernel2 --launch-count 8 --section SpeedOfLight --section ComputeWorkloadAnalysis --section Occupancy --section LaunchStats llama-bench -m <model> -p 512 -n 0 -ub 512 -fa on -ctv q8_0 -r 1`

### 1) standalone cuBLAS (预热后 ABAB 5 轮 x 30 iter; TFLOPS / 占 125 TF 峰值)

| 形状 (tokens, out_dim, K) | cuBLAS 实测 | 占峰值 | cublasLt 最佳 | GemmEx + 256MB ws |
|---|---:|---:|---:|---:|
| ffn gate/up (512, 17408, 5120) | **84.2** | 67% | 84.1 | 84.3 |
| ffn gate/up (2048, 17408, 5120) | 96.9 | 78% | 97.0 | 97.0 |
| ffn down (512, 5120, 17408) | **100.3-107.0** | 80-86% | 106.8 | 106.9 |
| attn_qkv / ssm in-proj (512, 10240, 5120) | 99.5 | 80% | 99.7 | 99.5 |
| lm_head (512, 248320, 5120) | 95.1 | 76% | 95.0 | 95.0 |
| 参考 4096^3 | 96.5 | 77% | 96.7 | 96.8 |
| attn_q / ssm_gate (512, 6144, 5120) | 68.3 | 55% | 70.0 | 70.0 |
| attn_o / ssm_out (512, 5120, 6144) | 101.9 | 82% | 101.8 | 102.0 |
| attn_kv (512, 1024, 5120) | 53.0 | 42% | 50.9 | 50.0 |

### 2) 三个否证 (旧报告 "75.6 TFLOPS = 60%" 是假象)

1. **workspace 无增益**: 4MB vs 256MB, ABAB 5 轮: down +1.1%, gate/up +0.1%, attn_o -0.0%。
   且本 fork **本来就在 handle 上设了 4MB workspace** (common.cuh:1549)。
   之前"256MB 更好"是每形状首轮冷状态伪影 (首轮 92.3 TF -> 稳定 100-107 TF)。
2. **cublasLt 无增益**: heuristic 最佳 algo 与 GemmEx DEFAULT_TENSOR_OP 预热后同值
   (gate/up 84.1 vs 84.2; down 106.8 vs 106.9; qkv 99.7 vs 99.5)。
   之前"Lt 84.2 vs 默认 77.3"是冷/热对比的假象。
3. **显式 algo 更差**: algo 112 (及 108-115) 在 gate/up 上 41.0 TF (-51%)。
   旧 75.6 的来源: ncu 锁频 1246MHz + 冷状态。

### 3) ncu 真实 kernel (`Kernel2<cutlass_70_tensorop_s884gemm_f16_128x128_tn_align8>`)

| grid | CTAs | 对应 op | Duration | Compute(SM) | **Tensor(FP)** | L1/TEX | DRAM | L2 | Waves | 理论 occ |
|---|---|---|---|---|---|---|---|---|---|---|
| (32,17,2) | 1088 | ffn gate+up | - | 70.5% | **72.2%** | 67.4% | 29.0% | 48.2% | 6.8 | 12.5% |
| (32,10,1) | 320 | attn_qkv/ssm | 661us | 75.5% | 76.8% | 69.4% | 22.3% | 49.6% | 2 | 12.5% |
| (32,6,3) | 576 | attn_gate 类 | 467us | 61.3% | 64.6% | 62.1% | 31.8% | 43.4% | 3.6 | 12.5% |
| (32,5,1) | 160 | down/ssm_out/o | 373us | 87.1% | 87.6% | 77.5% | 22.2% | 57.0% | 1 | 12.5% |

- 指标名: "Tensor (FP)" = `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active`;
  "DRAM Throughput" = `dram__throughput.avg.pct_of_peak_sustained_elapsed`;
  "Compute (SM) Throughput" = `sm__throughput.avg.pct_of_peak_sustained_elapsed`;
  occupancy = Achieved/Theoretical Occupancy (236 regs/thread, 128 thr/block, 寄存器限制 2 blocks/SM)。
- 判读: 大 GEMM 是 **L1/TEX (62-77%) 与 Tensor (65-88%) 双高**, occupancy 仅 12.5%, DRAM 只有 22-32%。

### 4) 结论 (按任务表 decision rule)

- "standalone >= 95 -> 调用侧问题": **不成立**。只有 gate/up(84)、attn_q/ssm_gate(68)、attn_kv(53) 低于 95,
  且三者用 cublasLt / 显式 algo / 更大 workspace 都拿不到增量 (已 ABAB 排除冷热假象)。
- "70-80 -> cuBLAS 是墙": **成立, 但墙高是 84-107 TFLOPS (67-86%)**, 不是 75.6。
- 调用侧已无空间, 不建议改 cuBLAS 调用 (可省下这项工作量)。
- 对 T02 的输入 (盈亏平衡 = (GEMM+dequant) 合计时间为分母, dequant 按实测 707GB/s):
  **gate/up 56.3 TFLOPS / down 65.6 / attn_qkv 70.7 / lm_head 67.8**。
  即: 融合 kernel 带反量化若打到 84 TFLOPS (= cuBLAS 单独水平) -> gate/up 段整机 prefill +12%;
  打到 70 TF -> +7%; 打到 60 TF -> +1.8%; 低于 56 TF -> 负收益。
- 风险提示: cuBLAS 的弱形状 (gate/up 67%) 是被 L1 限制、occupancy 只有 12.5%。融合 kernel 若读量化权重
  (Q6_K 比 F16 少 2.4x L1 流量) 有可能同时降低 L1 压力, 这是 T02 唯一的"结构性"胜机。
---

## 2026-09-22 | T03 (第 1 步: 向量化行布局) | PARTIAL | implementer

- git: 4046f5c8c + `v100-dequant-vec.patch` + 工作区改动 `ggml/src/ggml-cuda/gated_delta_net.cu` (未提交)
- 命令: `llama-bench -m <model> -ngl 99 -fa on -ctv q8_0 -ub 512 -p 512,4096,8192 -n 128 -r 3`
- 时钟: prefill 1530MHz / 277W (满频, 无降频)

### 改动内容

行->lane 映射从 `i = r*warp_size + lane` (stride 32) 改为 `i = lane*rows_per_lane + r` (连续 4 行),
于是 state/k/q 的加载与写回从 4 条标量访存变成 1 条 16 字节 float4 访存 (S_v=128 时)。
只对 `rows_per_lane == 4 && warp_size == 32` 生效 (本模型 S_v=128), 其他形状保留原路径。

### 数字

| 指标 | 基线 | 本步 | 变化 |
|---|---:|---:|---:|
| gated_delta_net kernel (nsys, 单层) | 879.9 us | **826.3 us** | **-6.1%** |
| pp512 | 950 | 961.0 | +1.2% |
| pp4096 | 935 | 936.7 | +0.2% |
| pp8192 | 908 | 910.7 | +0.3% |
| tg128 | 26.64 | 26.47 | -0.6% (疑似噪声, 未复测) |
| PPL | 4.3572 | **4.3569** | 差 0.0003 (fp32 求和重结合, 非精度降级) |

### 结论 / 为什么不继续走微调

- MIO 访存指令 -29% 只换来 kernel -6.1%, 与 ncu 一致: 该 kernel 是**总指令吞吐受限**
  (SM throughput 66%, issue 0.64/scheduler, 40 warps/SM), 不是访存受限。
  每 token 每 warp 约 45 条指令中, **两次 warp 归约占 20 条** (10 shfl + 10 add), 是最大单项。
- 归约指令数无法再降: 128 行 / 32 lane 的对数归约最少 5 步, 两个归约 (kv, attn) 共 10 步。
  用 kq 恒等式 (attn = g*Sq + delta*(k.q)) 把两个归约合并, 指令数不变 (float2 shuffle 仍是 2 shfl+2 add/步),
  只能改善延迟链, 不能减指令。
- 结论: **T03 的 -40% 验收必须走 chunked 重写** (token 维分块 + 块内 matmul, 参考 FLA chunk_delta_rule)。
  这会把每 token 的 45 条标量指令换成 mma (每条 mma = 256 个 MAC), 归约与访存指令基本消失。
  代价: 1-2 天 + 需要处理 decay 因子/三角求解/快照槽 (keep_rs_t) 三个复杂点; 高风险。
- 本步改动建议保留 (kernel 严格变快, PPL 差异 0.0003), 但未达 T03 验收 (kernel -40%, pp512 +2%)。

### 下一步选项 (请 analyst 定)

A. 继续 T03: 实现 chunked kernel (chunk=64, mma 版), 目标 kernel -60% 以上 -> pp512 +3-4%
B. 暂停 T03, 转 T02 (融合 GEMM, 主线, 预期 +12-20%)
C. 先做 T05 (decode profile) 之类的小任务
---

## 2026-09-22 | T02 Stage 1 (第一版融合 kernel) | INCONCLUSIVE (需 analyst 决定) | implementer

- 源码: `%TEMP%\v100\t02_fused.cu` (standalone 微基准, 未进仓库)
- 编译: `nvcc -O3 -arch=sm_70 -o t02_fused.exe t02_fused.cu -lcublas`
- 形状: ffn gate/up (tokens=512, out=17408, K=5120), 真实 Q6_K 权重, X=f16, C=f32
- ncu: `--clock-control none --cache-control none -k regex:fused_q6k --launch-count 2 --section SpeedOfLight --section SchedulerStats --section WarpStateStats --section Occupancy --section LaunchStats`

### 数字 (mode 15 = 完整 kernel)

| 项 | 时间 | TFLOPS | 备注 |
|---|---:|---:|---|
| dequant kernel (向量化, 现路径) | 0.202 ms | - | 73MB 读 + 178MB 写 |
| cuBLAS (F16 权重) | 1.199 ms | 76.1 | 与 T01 一致 |
| **baseline 合计** | **1.402 ms** | **65.1** | 当前 llama.cpp 路径 |
| **融合 kernel v1** | **9.206 ms** | **9.9 (8% of 125)** | 正确性 OK: sum_rel=2.0e-7, max_rel=0.034 |

### 诊断 (ncu + mode 拆分, 这是本次的主要产出)

1. **occupancy 只有 6.25%**: `Block Limit Shared Mem = 1` —— 每块 50.7KB smem (A 33.8 + B 16.9, 已做 padding)
   -> 1 block/SM = 4 warps/SM, `No Eligible = 85.7%`, `Issued Warp Per Scheduler = 0.14`。
   直接原因: smem 预算 (96KB/SM) 与 "BN 要宽" 冲突。
2. mode 拆分 (单位 ms, 同一次运行): dequant-only 6.3, mma-only 2.6, A-load-only 2.6, 全量 9.2 -> **三个阶段几乎不重叠**
   (无 double buffering, 被 __syncthreads 串行化)。
3. **padding 修改有效**: smem 行距 = BK*2 = 512 字节是 128 的倍数 -> wmma fragment 加载全 bank conflict;
   改成 264 halfs (528 字节) 后 mma-only 从 8.8ms -> 2.6ms (3.4x)。
4. **结构性约束 (最关键)**: 融合 kernel 的 B tile 会被每个 M-tile 重新反量化一次。
   BM=64 -> 8 个 M-tile -> 量化权重被读 8 次 (585MB vs baseline 的 429MB 总流量, 反而不省)。
   要净省流量必须 BM>=128 (4 次) 或更大, 但 BM=128 时 A tile 就占 67KB smem。
   另一个约束: A 操作数的 L2 流量 = (N/BN) x (M*K*2)。BN=32 时 2.85GB (1.14ms @2.5TB/s, 比计算还贵),
   BN=128 时 712MB。cuBLAS 用的正是 128x128 tile。
5. 目标设计 (v2, 未实现完): BM=128 / BN=128 / BK=256 + 双缓冲 + swizzle 或 padding,
   但 (BM+BN)*BK*2*2 = 256KB > 96KB smem, **物理上放不下**。可行组合需要 BK=64/128 (非整 Q6_K 块,
   反量化要按半块/四分之一块处理) 或 A/B 之一走 global->register fragment 直读 (L2 流量翻倍)。

### 结论与建议

- 第一版证明: **融合 kernel 在 sm70 上可正确实现**, 但性能工程成本高: 要同时满足
  (a) occupancy >= 2 blocks/SM (smem <= 48KB/block), (b) BN >= 128 (A 流量), (c) 双缓冲 (阶段重叠),
  (d) BM >= 128 (量化权重重读次数), (e) 无 bank conflict —— 这 5 条互相冲突, 需要精细的 tile + swizzle 设计。
- 按 T02 门槛 (>=84 TF 继续): 当前 9.9 TF 远未达标, 但这是 v1 (无流水线, occupancy 6%);
  按 (d) 的流量分析, 理论天花板仍在 ~125 TF, 盈亏平衡 56 TF 在合理设计下可达。
- 建议 analyst 决策:
  A. 继续 T02 Stage 1b: 用 "A 走 global fragment 直读 + B 双缓冲 + BM=128/BN=64" 或改造 `mmf.cuh`
     (它已有 padding/流水线/mma 抽象, 只需加一个反量化 B tile loader) —— 估计还要 1-2 天;
  B. 暂停 T02, 先做低风险的 elementwise 向量化 (silu/rms_norm/concat/add 实测只跑 68-70% DRAM 带宽,
     共 35ms/548ms = 6.4%, 预计 +2-3% 整机, 工作量小时级);
  C. 转 T05/T06 (decode 侧)。
- 我个人建议 B 先做 (确定性收益, 小时级), 再做 A (高风险, 天级)。
### T03 第 1 步收尾: KDA 缺陷修复 + tg128 A/B + patch 归档

- **必修缺陷已修** (analyst 复核 2 指出的 KDA g 索引问题): `use_vec4 = rows_per_lane == 4 && warp_size == 32 && !KDA`
  —— KDA 分支保持旧的 strided 映射, 本模型 (kda=false) 数值不变。
- patch 归档: `D:\LLM\Backend\patches\v100-gdn-vec4.patch` (防 update bat 重置丢失)
- 复测 (r5): PPL **4.3569** (不变); pp512 **962.6** (+1.3% vs 950)
- **tg128 A/B 结论: 回归真实, 幅度 -0.5%** (同一台机, 换 DLL 交替):
  | 构建 | tg128 (r5) |
  |---|---:|
  | 旧 (仅 dequant-vec patch) | 26.62 |
  | 新 (+GDN vec4) | 26.48 |
  - 机制推测: decode 时 GDN 是 n_tokens=1 的小 kernel (约 7.4us/层), vec4 路径增加了寄存器/序言开销,
    在延迟主导的场景下略亏 (总影响 -0.5%, 对应 GDN kernel 约 +0.19ms/token)。
  - 净账: prefill +1.3%, decode -0.5%。T03 验收要求 "tg128 不退化", 因此本步**未达标**, 请 analyst 判定:
    (i) 接受 (decode 无感知, 且 T06 MTP 是 decode 主线) / (ii) 按 n_tokens 分支只在大 batch 用 vec4 /
    (iii) 回退本步 (回到 950/26.62)。
---

## 2026-09-22 | T05 decode 分解 (profile 部分完成, 修复项已试 1 项) | PARTIAL | implementer

- 命令: `nsys profile -o prof_dec1 --trace=cuda --cuda-event-trace=false llama-bench -m <model> -p 0 -n 1 -fa on -ctv q8_0 -r 1`
  (注意: `-n 128` 会因为 trace buffer 截断, 只能拿到约 1 个 token 的完整数据, 已用 `-n 1` 取得完整单 token)
- 时间口径: llama-bench 实测 **37.7ms/token (26.5 t/s)**; 单 token kernel 总和 **35.15ms**; 真空间隙 **2.5ms (6.6%)**

### 单 token kernel 分解 (2120 个 kernel)

| 类别 | 时间 | 占比 | 次数 | 备注 |
|---|---:|---:|---:|---|
| **mul_mat_vec_q 合计** | **29.50 ms** | **83.9%** | 454 | 权重读取主体 |
| - Q6_K (type 14) | 18.74 | 53.3% | 208 | 13.89GB -> **741 GB/s** |
| - Q5_K (type 13) | 6.84 | 19.5% | 95 | 4.93GB -> **721 GB/s** |
| - Q8_0 (type 8) | 3.93 | 11.2% | 151 | lm_head(1.35GB) + 小矩阵 -> 725 GB/s |
| quantize_q8_1 | 0.86 | 2.4% | 461 | 几乎 1:1 跟随 MMVQ (激活量化) |
| rms_norm (3 变体) | 1.12 | 3.2% | 305 | |
| scale_f32 | 0.64 | 1.8% | 192 | 64 层 x 3 |
| gated_delta_net | 0.63 | 1.8% | 48 | decode 侧每层仅 13.2us |
| flash_attn_ext_vec | 0.34 | 1.0% | 16 | |
| k_get_rows (embd) | 0.26 | 0.7% | 48 | |
| add / cpy / silu / concat | 0.73 | 2.1% | 284 | |
| rope/fwht/其他 | ~0.3 | 0.9% | | |
| **非 MMVQ 小计** | **5.65 ms** | **16.1%** | ~1666 | |
| **间隙 (2.5ms)** | 2.50 | - | - | 图启动 + 边界 |

### MMVQ 带宽判读

- 每 token 权重流量 = 21.97GB 文件 - token_embd(+1.04GB, 只查表) - 未读的 = 实际约 **20.2GB**
- MMVQ 合计 29.5ms -> 有效 **685-741 GB/s = 理论 900 的 76-82%**
- 参照: 本机 dequant kernel 实测 825 GB/s (92%), 即实际可用带宽上限约 825-850 GB/s
  -> **MMVQ 已到实际带宽的 83-90%, 剩余空间 <= 2-3ms/token**
- 微调实验 (T05 候选 1: nwarps):
  | nwarps (ncols_dst=1) | tg128 |
  |---|---:|
  | 4 (现行, GENERIC 表) | **26.62** (A/B 旧构建) |
  | 2 | 25.93 (-2.6%) |
  | 8 | 24.07 (-9.6%) |
  -> **现行配置已最优, 已回退**, 不要重复尝试

### 达到 30 t/s 的可行性

- 目标 30 t/s = 33.3ms/token, 需要砍掉 4.4ms:
  - MMVQ 从 78% -> 90% 实际带宽: -2.0ms (难度高, 需要改 vec_dot 的访存模式)
  - quantize_q8_1 融进 MMVQ (省 461 次 launch): -0.6~0.8ms (中等难度)
  - 小 kernel 合并 (rms_norm/scale/add/cpy/silu/concat 共 3.4ms, 每个 kernel 数据量只有几 KB, 基本是 launch 开销): -1.0ms (工作量中等)
  - 间隙: -0.5ms
- 乐观合计: -4.1ms -> 33.6ms = **29.8 t/s**;  即 **30 是临界值, 33 不现实**
- **decode 翻倍的唯一路线仍是 T06 (MTP)**: 需要实测有效 tg (接受率 x 步进), 见下一条
### T05 补充: MMVQ 调参实验全部完成 (负面结果, 已回退)

| 实验 | tg128 (r5) | 结论 |
|---|---:|---|
| 基线 (nwarps=4, rows_per_block=1) | 26.62 | 现行最优 |
| nwarps=2 | 25.93 | -2.6% |
| nwarps=8 | 24.07 | -9.6% |
| rows_per_block=2 | 26.70 | +0.3% (噪声内) |
| rows_per_block=4 | 25.57 | -4.0% |

- 所有实验已 `git checkout` 回退, 工作区只剩 3 个已归档改动 (convert.cu/dequantize.cuh/gated_delta_net.cu)
- **判定: MMVQ 的参数空间已穷尽** (nwarps x rows_per_block 共 6 组), 现行配置 (GENERIC 表: nwarps=4, rows=1) 就是 Volta 的最优。
- MMVQ 84% 的 decode 时间压在 76-82% 理论带宽上; 要再快只能改 vec_dot 的数据通路 (例如把 Q6_K 的 6-bit 解包从 dp4a 换成别的形式), 属于研究性工作, 收益 <= 10%。
- **无 MTP 的 decode 现实结论: 26.6 -> 28-29 可能需要小型 kernel 融合 (quantize_q8_1/rms_norm/scale 共 2.6ms, 2120 个 kernel 的 launch 开销 2.5ms);
  30 t/s 是理论边缘, 33 不现实。** 请 analyst 决定是否值得投入 (预计 1 天, 收益 +8-10%)。

### 用户侧信息 (重要): 含 MTP 的服务端 decode 实测 (供 T06 校准)

- 用户生产配置实测 (llama-server, 短 prompt, n_predict=128, 温度 0):
  | 配置 | decode |
  |---|---:|
  | 无 spec | **25.3 t/s** |
  | `--spec-type draft-mtp --spec-draft-n-max 3` | **42.2-42.8 t/s** |
- MTP 侧日志: draft acceptance = 0.617, mean accepted len = **2.82**
- 即 T06 的目标 (26.6 -> 40+) **在用户当前生产配置下已经达成**; 用户明确表示 n-max=3 是他们测过的最优,
  不需要再扫。T06 剩余空间: 提高接受率 (draft 质量) 或降低验证步成本 (MMVQ 在 ncols_dst=4 时的效率)。
### T03 收尾完成: vec4 按 n_tokens 分支

- 实现: `gated_delta_net_cuda<S_v, KDA, keep_rs_t, VEC4>`, launcher 里 `vec4 = n_tokens > 1`
  (decode 单 token 回退到原标量路径); S_v=16/32/64 实例化 VEC4=true (rows_per_lane != 4, 自动折叠)
- 复测: PPL **4.3569**; pp512 **959.7 (+1.0%)**; pp4096 936.0; pp8192 909.7
- **tg128 严格 A/B (2 轮交替, r5)**: OLD(仅 dequant-patch) 26.68/26.64 vs NEW 26.59/26.62
  -> **之前报告的 -0.5% 主要是构建/测量漂移, 分支后已回到噪声内 (-0.1%)**
- patch 已更新: `D:\LLM\Backend\patches\v100-gdn-vec4.patch`
- T03 状态: 可以 DONE (第 1 步 + 收尾); chunked 重写按 analyst 决定仍在最低优先级队列
---

## 2026-09-22 | T07 elementwise 向量化 (第 1 项: silu) | PARTIAL | implementer

- 改动: `ggml/src/ggml-cuda/unary.cu` 新增 `unary_gated_op_kernel_f32_vec4` (float4, 每线程 4 元素),
  条件 `k%4==0 && n%4==0 && o0%4==0 && o1%4==0` 时启用, 否则回退原标量 kernel。patch: `patches/v100-t07-silu-vec4.patch`
- kernel 级 (nsys, ub512 单 batch): `unary_gated_op (silu)` **11.42ms -> 10.84ms (-5.1%)** (112 次调用)
- 端到端: pp512 958.0 (前 959.7, 基线 950), pp4096 932.4 (前 936.0), tg128 26.68; PPL **4.3569** 不变
  -> **端到端在噪声内 (+/-0.5%), 只有 kernel 级 -5% 是确定收益**

### T07 的关键修正 (重要, 供 analyst 调整预期)

实测各 elementwise kernel 的有效带宽 (ub512, nsys 数据):
| kernel | 时间 | 数据量估算 | 有效带宽 | 判读 |
|---|---:|---:|---:|---|
| unary_gated (silu) | 11.4ms -> 10.8 | 112 x 107MB = 12GB | ~1.05 TB/s | 已超 DRAM 峰值 (L2 命中) -> 向量化只 -5% |
| convert_unary_cont_vec4 | 14.3ms | 496 x 15.7MB = 7.8GB | 549 GB/s | **已是 vec4**; 小 kernel (29us) 的发射/尾巴开销占比大 |
| rms_norm (3 变体) | 10.9ms | 305 x 21MB = 6.4GB | 590 GB/s | 标量实现, 但受两趟结构 + 归约限制 |
| k_bin_bcast (add) | 5.6ms | 176 x ~20MB = 3.5GB | 625 GB/s | |
| concat_non_cont | 5.9ms | 48 x ~85MB = 4GB | 680 GB/s | |

- **结论修正**: 这些 kernel 大多已接近内存墙 (或已被 L2 掩护), "只跑 58-70% BW"的推算高估了收益。
  T07 全做完的现实收益是 **+0.5-1.0% 整机**, 不是 +2-3%。
- 建议: silu 这项保留 (kernel 严格变快, 无副作用); rms_norm 向量化 (预估 590->750GB/s, +0.4%) 可做可不做;
  convert_unary 的 14.3ms 里大头是"每 call 只有 29us"的固定开销, 除非能减少调用次数 (融合进 GEMM 前置),
  否则优化空间有限。
---

> **注 (analyst 整理)**: 以下 SESSION RESUME 的内容已迁移到 `ENVIRONMENT.md` 与 `BOARD.md`, 这里保留为历史快照。

# ===== SESSION RESUME (历史快照, 已被 ENVIRONMENT.md 取代) =====

## 工作区状态 (D:\LLM\Backend\src\llama.cpp-my, 未提交, 4 个文件)

| 文件 | 内容 | patch 归档 |
|---|---|---|
| `ggml/src/ggml-cuda/dequantize.cuh` | Q6_K/Q5_K 向量化反量化设备函数 | `patches\v100-dequant-vec.patch` |
| `ggml/src/ggml-cuda/convert.cu` | 向量化反量化 kernel + launcher + 16B 对齐回退 | 同上 |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | vec4 行布局 (按 n_tokens 分支) + KDA 修复 | `patches\v100-gdn-vec4.patch` |
| `ggml/src/ggml-cuda/unary.cu` | silu float4 向量化 (带标量回退) | `patches\v100-t07-silu-vec4.patch` |

- 构建: `cmd /c %TEMP%\v100\build_ggml_cuda.cmd` (vcvars64 + `cmake --build build --config Release --target ggml-cuda`)
- 部署: `Copy-Item D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\ggml-cuda.dll D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll -Force`
- **已部署 DLL 就是当前最新构建** (含上面 4 项改动); 备份: `%TEMP%\v100\ggml-cuda-fast.dll` (仅 dequant patch), `ggml-cuda-old.dll` (原始)

## 当前已验证数字 (CUDA_VISIBLE_DEVICES=1, `-ub 512 -fa on -ctv q8_0`)

| 指标 | 值 | 对比基线 |
|---|---:|---:|
| pp512 | **958-960** | 950 (+1.0%) |
| pp4096 | 932-936 | 935 (+0.1%) |
| pp8192 | 909.7 | 908 (+0.2%) |
| tg128 (r5, A/B) | 26.59-26.68 | 26.64-26.68 (噪声内) |
| PPL (512ctx/8chunks/seed42) | **4.3569** | 4.3572 (重结合, 在门槛内) |
| ub2048 (调参口径, 用户已排除) | pp2048 1157 | - |

## 验证门槛 (每次改完必须过)

1. `llama-perplexity -m <model> -f %TEMP%\v100\ppl_text.txt -c 512 --chunks 8 -ngl 99 -fa on -ctv q8_0 --seed 42` -> PPL 必须 4.3569 或 4.3572 (|diff| <= 0.013)
2. `llama-bench -m <model> -p 512,4096,8192 -n 128 -ub 512 -fa on -ctv q8_0 -r 3`
3. 有疑义时用 A/B 交替换 DLL 复测 (单次测量在 +/-0.5% 噪声内, 不可作为判据)

## 队列 (analyst 第三次决议)

1. ~~T03 收尾~~ **已完成** (vec4 按 n_tokens 分支, pp512 959.7, tg 噪声内)
2. **T07** elementwise 向量化: 第 1 项 silu 已完成 (kernel -5%, 端到端噪声内);
   剩余 rms_norm/convert/add/concat 实测多已近内存墙或已是 vec4 -> **预期下调到 +0.5-1%**
3. **T02 Stage 1b** (timebox 1.5 天, Gate A: 骨架 >= 80 TF 才继续): 融合 kernel 重写
   - 已完成设计要点 (v1 微基准 `%TEMP%\v100\t02_fused.cu` 的教训):
     BM>=128 (量化权重重读次数), BN>=64-128 (A 操作数 L2 流量), smem <= 48KB/block (occupancy >= 2),
     B tile 必须 padding (BKP=BK+8) 否则 wmma fragment 加载全 bank conflict,
     A 走 global->register fragment 直读可省 smem, B 双缓冲 (但 BM=128+BN=128+BK=256 双缓冲 = 256KB > 96KB, 物理放不下)
   - 盈亏平衡线 (analyst 修订): 约 66 TF; 整机预期 +5-10%
4. T05 剩余修复项 (q8_1 融合 ~0.8ms, 小 kernel 合并 ~1.0ms, 间隙 ~0.5ms)
5. T04 ub 兜底 (用户已排除调参类) / T03 chunked (最低优先级)

## 关键工具与脚本 (%TEMP%\v100\)

- `build_ggml_cuda.cmd` 构建; `server_test.ps1` 服务端 pp 测试 (`-Tag -Port -ExtraArgs`);
  `server_decode_test.ps1` 服务端 decode 测试 (含/不含 MTP)
- `t01_bench.cu/.exe` cuBLAS/cuBLASLt 形状扫描; `t01_ab.cu/.exe` 预热+A/B 对照;
  `t02_fused.cu/.exe` 融合 kernel 微基准 (mode 位控制: 1=dequant, 2=mma, 8=A load)
- `gemm_sweep.exe` / `gemm_conc.exe` (并发性) / `dequant_bench.exe` (反量化带宽)
- ncu 提权脚本: `ncu_t01b.ps1`, `ncu_gdn.ps1`, `ncu_t02.ps1` (Start-Process -Verb RunAs, 需要 UAC)
- nsys: 无需提权, `--trace=cuda --cuda-event-trace=false`; `-n>2` 会丢数据, 用 `-n 1/2`
---

# T02 Gate A: 结果 = FAILED (建议 REJECT T02)

implementer, 2026-09-22. 产物: `v100-collab/artifacts/` (t02_gateA_v4.cu = 最终版, v6/v7 = 诊断变体,
t02_gateA_peak.cu = 原语峰值测试, t02_tile_dbg.cu = 布局单元测试, ncu_gateA_v1_v2.log)

## 1. 数字总表 (ffn gate/up: K=5120, 91.27 GFLOP, V100-SXM2 @1530MHz)

| 实现 | 朝向 C[17408,512] | 朝向 C[512,17408] | 说明 |
|---|---:|---:|---|
| 纯 mma 原语峰值 (寄存器常驻, 无 smem) | **97-99 TF** | - | 上限; 说明 TC/寄存器不是瓶颈 |
| 我的完整流水线 (v4, fp16 权重) | **57.5 TF** (1.586ms) | **54.3 TF** (1.681ms) | 正确 (NMSE 2.6e-12) |
| 我的 mma 路径单独 (mode 32, 无 global 预取) | 62.0-78.7 TF (1.46-1.47ms) | 62.7 TF | |
| staging 单独 (mode 2, 无 mma) | 0.837ms (1.42GB -> 1.7 TB/s) | 1.174ms | L2 带宽上限 |
| 循环+STS+barrier 单独 (mode 34) | 0.314ms | - | |
| cuBLAS fp16 gemmEx 默认 (llama.cpp 现用) | 1.179ms / **77.4 TF** | 1.178ms / 77.5 TF | |
| cuBLAS fp16 显式 algo (ALGO8-15_TENSOR_OP) | 1.100ms / 83.0 TF | - | |
| cuBLAS fp16 cublasLt heuristic #0 (algo21/tile20/splitK2) | **1.083ms / 84.3 TF** | - | 最好 |
| baseline "dequant + cuBLAS默认" | 1.404ms (65 TF 等效) | - | 盈亏平衡 66 TF |

## 2. 结论

**Gate A 未达标 (57.5 < 80 TF) -> T02 REJECTED。** 更重要的否决理由:

- mma 路径单独就 1.472ms, 已经慢于 baseline (dequant+cuBLAS) 1.404ms。
  即: **即使把 staging 完全隐藏、且 dequant 零成本, 也赢不了 baseline**。
- staging 与 mma 已经基本重叠 (完整 1.586ms = staging 0.837 + mma路径 1.472 - 重叠部分), 不是靠管线深度能救的
  (v5 加了第 3 级 smem + 2 级寄存器预取: 56.4 TF, 无改善; 寄存器数组还进了栈)。
- 我们的 mma 路径 (78.7 TF) 做不到 cuBLAS 的水平 (84.3 TF), 差 7%; 加上 smem staging 的暴露后差 31%。

## 3. 过程发现 (可复用, 已归档)

1. **k-blocked 布局是 staging 提速的关键**: 权重/激活按 [kb][row][BK] 重排后,
   staging 从 **574GB/s -> 1.7TB/s** (54 TF -> 57.5 TF 的完整版, staging 单独 2.474 -> 0.837ms)。
   原因: 原始行主序下每行每 k-block 只有 64B, 跨步 10240B -> DRAM 随机访问; 重排后顺序。
   (对任何"面板搬运"型 kernel 都适用; 融合路线若重启必须用它。)
2. **grid 顺序**: 让相邻 block 共享同一个权重面板 (grid.x = token 维) -> +6.5% (54.0 -> 57.5 TF), L2 复用。
3. Volta 的 `tile<32,4,half2>` 原语 (m8n8k4, k 跨度 8) 与 `nvcuda::wmma` 16x16x16 (k 跨度 16):
   两者完整版性能相同 (34.6/34.7 TF) -> fragment LDS 指令量不是瓶颈 (瓶颈是 staging, 见上)。
4. **dequant 成本推算 (融合致命项)**: 每 SM issue 预算 10.07M 指令 (1.645ms), 现用 2.92M (29%);
   每 block 做 1 份 Q6_K->fp16 反量化 (~2 指令/权重) 需要 4.4M 指令/SM -> 会把 issue 吃满,
   与 HMMA 争抢 -> 融合版上限约 50-67 TF, 在 66 TF 盈亏平衡线附近或之下。
   (注: 依赖链 FMA 注入实验显示延迟型 ALU 可以藏进气泡, 但吞吐型指令不行。)
5. 正确性调试: tile 的 `get_i/get_j` 内部用 `threadIdx.x` 当 lane,
   必须用二维 block (32, nwarps) 或自己按 lane 计算, 否则多 warp 的 C 存储全错。

## 4. T01 结论修正 (重要, 请 analyst 复核)

t01_bench 重跑 (同一微基准, 同形状 ffn_gate/up ub512):
- gemmEx 默认 = 1.179ms / **77.4 TF** (之前 T01 记的 84.2 是显式 algo 的数字)
- gemmEx ALGO8..15_TENSOR_OP = 1.100ms / **82.9-83.0 TF** (+7%)
- cublasLt heuristic #0 = algo21/tile20/**splitK=2** = 1.083ms / **84.3 TF** (+9%)
=> **cuBLAS 调用侧确实有 7-9% 空间, T01 的"调用侧无空间"结论应修正** (当时只测了 workspace 和 algo 112)。

但注意: **在 llama.cpp 里直接把 2D GEMM 的 algo 换成 ALGO8 会让 pp512 从 958 掉到 310 t/s (-68%)**
(已实测, 已回退) —— 因为 llama.cpp 的 GEMM 调用形状多 (attn_qkv/o、ssm、lm_head...),
algo hint 是形状相关的。所以这条路必须走 **cublasLt + 逐形状 heuristic 缓存 + 工作区**,
并且要逐形状验证, 属于有风险的工程改动 (估计 0.5-1 天 + 回归测试)。

## 5. 建议给 analyst 的下一步选项

- A. **T02: REJECTED** (已用数据否决), 从队列移除。
- B. **新任务 (建议): T08 cublasLt 集成** - 目标 pp512 +4-5% (GEMM 349ms 的 7-9%);
  风险: 形状相关 (见上), 需要 per-shape 缓存 + 回退开关 + 全形状回归。
- C. 回到 T07 剩余 (rms_norm 向量化, +0.4%) 与 T05 剩余 (小 kernel 合并, decode +5-8%), 两者都是低风险。
---

# T09-A: 量化 operand 微基准 (融合 2.0) | 结果 = **FAILED -> 建议永久关闭融合路线**

implementer, 2026-09-22. 产物: `artifacts/t09_q6k_v9.cu` (BM=64 版; BM=128 版为同一文件改 3 处常量),
`pack112.py` / `verify_formula.py` (布局推导的 Python 全量验证), `sass_v9.txt`

## 实现 (与 T09-A 规格一致)

- 真实 Q6_K 权重: 从模型 `blk.2.ffn_gate.weight` (Q6_K, [5120,17408]) 提取 73,113,600 字节,
  重排为 half-super-block 布局 [40][17408][112] (64B ql + 32B qh + 8B scales + 2B d + pad, 0.88 B/元素)
- kernel: v4 骨架 (wmma 16x16x16, 双缓冲, k-blocked 激活) + 权重按 half-super-block staged 到 smem,
  **块内解包** (手写 LOP3/移位/half2) 到 fp16 面板, 再 `load_matrix_sync` 喂 mma
- 布局/公式用 Python 对 raw 文件真值 **全量验证** (8 half x 全部 slice, 0 错误); GPU 侧也逐 slice dump 验证过
- 正确性: 对 (独立反量化 + cuBLAS) 参考, **NMSE = 2.48e-12** (与 fp16 路径同级)

## 结果 (ffn gate/up: M=17408, N=512, K=5120, 91.27 GFLOP)

| 配置 | 完整 | 去解包 (mode 8) | 去 mma (mode 4) | 备注 |
|---|---:|---:|---:|---|
| **v9 BM=64, BN=128, 2 blk/SM** | 2.464ms / **37.0 TF** | 2.007ms (45.5 TF) | 1.435ms | 解包成本 0.46ms (18%) |
| **v9 BM=128, BN=128, 1 blk/SM** | 2.230ms / **40.9 TF** | 1.569ms (58.2 TF) | 1.396ms | 解包成本 0.66ms (30%) |
| 参考: v4 fp16 权重 BM=128 | 1.586ms / 57.5 TF | - | - | 同骨架, 仅 operand 是 fp16 |
| 参考: cuBLAS (默认 algo) | 1.187ms / 76.9 TF | - | - | |
| 参考: baseline = dequant + cuBLAS | 1.404ms / 65 TF | - | - | 盈亏平衡 66 TF |

## 结论: 三条独立否决 (按 analyst 的 go/no-go: <66 TF = 永久关闭)

1. **即使解包零成本也赢不了**: BM=128 的 mode 8 (去解包) = 1.569ms, 已经慢于 baseline 1.404ms。
   即"读量化权重少 2.6x global 流量"这个正面机制, 在本 tile 空间内**落不到时间上**。
2. **解包成本 0.46-0.66ms (18-30%) 且不可重叠**: Volta 无 cp.async, 每 SM 4 个 scheduler,
   解包是 issue/latency 型开销, 只能串在 mma 之间。
3. **config 2 (BM=512, 权重读 1x) 物理不可能**: C 寄存器 = 512x128/256 线程 = 256 floats/thread ✗;
   smem 也需要 >96KB/block ✗。(config 1 的 BM=128 就是本次实测的上限。)

## 根因 (为什么"少 2.6x 流量"没有收益, 实测解释)

- BM=128 时 staged 流量 = Q6_K 权重 0.36GB + 激活 0.71GB = 1.07GB (fp16 对照 = 1.42GB),
  但时间几乎相同 (1.57 vs 1.59ms) -> **kernel 不是 DRAM 带宽受限**, 而是 staging 延迟/发射受限;
  流量优势无法转化为时间。
- 激活重读 (每个 (BM,·) tile 把激活面板读 17408/BM 次) 是主导项: BM=64 时激活流量 1.42GB,
  完全抵消权重侧的节省 -> 这解释了 BM=64 只有 37 TF。
- BM=128 + 1 block/SM 的 8 warps 又让解包无法被掩盖 (解包成本从 18% 涨到 30%)。

## 建议

- **T09-A: FAILED, 按 go/no-go 规则永久关闭融合路线** (T02/T09 都不再投入)。
- T09-B (CUTLASS 调研) 是否还要做: 从本次数据看收益上限不足 (即使 CUTLASS 的 staging 做到 cuBLAS 水平
  = 76 TF, 加 30% 解包 -> ~53 TF < 66 TF), 建议**跳过 T09-B**, 除非 analyst 有别的理由。
- 附带修正一条既有认知: ub512 的 prefill 瓶颈不是 DRAM 带宽 (848GB/s) 而是 smem staging 的
  延迟/发射 (v4 的 mode 2 单独 = 0.837ms 却无法与 mma 完全重叠)。
---

# T08 Step 1: 逐形状 ABAB 仲裁 (gemmEx 默认 vs cublasLt) | 结果 = **REJECTED** (建议从队列移除)

implementer, 2026-09-22. 改动 llama.cpp 代码: **0 行** (纯测量)。
产物: `artifacts/t08_arb.cu` (主仲裁 harness), `t08_one.cu` / `t08_interf.cu` (诊断),
`t08_arb_run.log` (原始输出), `cublas_calls_pp512.log` (模型真实 cuBLAS 调用参数, CUBLAS_LOGINFO_DBG=1)

## 1. 模型真实 GEMM 形状与时间 (from cuBLAS log + nsys pp512, 1 次前向)

| shape (m,n,k) | 调用/前向 | 每次 ms | 小计 ms | TFLOPS (@1530) |
|---|---:|---:|---:|---:|
| ffn gate / up (17408,512,5120) x2 | 128 | 1.165 | 148.99 | 78.3 |
| ffn down (5120,512,17408) | 64 | 0.990 | 63.40 | 92.2 |
| ssm in-proj (10240,512,5120) | 48 | 0.609 | 29.23 | 88.0 |
| ssm out / o_proj (5120,512,6144) | 64 | 0.359 | 23.00 | 89.4 |
| ssm qkv (6144,512,5120) | 48 | 0.473 | 22.71 | 68.2 |
| attn qkv (12288,512,5120) | 16 | 0.817 | 13.07 | 78.8 |
| attn k / v (1024,512,5120) | 32 | 0.094 | 3.01 | 57.7 |
| ssm_ba (48,512,5120) | 96 | 0.021 | 1.97 | 7.7 |
| **合计** | 992 | - | **305.4** | (kernel 总计 507.8ms) |

全部走 `cublasGemmEx(OP_T, OP_N, m=out_dim, n=tokens, A=权重 f16 lda=k, B=激活 f16 ldb=k, C=fp32 ldc=m,
CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`; handle: `math=CUBLAS_TF32_TENSOR_OP_MATH` + 4MB workspace
(两者都由 common/ggml 设置, 已从 cuBLAS 日志确认)。形状分组与 nsys kernel grid 一一对应。

## 2. 仲裁结果 (同 harness, 同 layout, 5 轮交错, 中位数)

llama.cpp 真实方向 (m=out_dim, n=tokens):

| shape | def_tf32 (llama.cpp 现状) | cublasLt best | lt 增益 |
|---|---:|---:|---:|
| ffn gate/up | 1.112 ms / 82.1 TF | 1.145 ms / 79.7 TF | **-2.9%** |
| ffn down | 0.934 / 97.7 | 0.930 / 98.2 | +0.5% |
| ssm in-proj | 0.583 / 92.1 | 0.583 / 92.1 | 0.0% |
| ssm out/o | 0.343 / 94.0 | 0.340 / 94.6 | +0.6% |
| ssm qkv | 0.456 / 70.7 | 0.447 / 72.1 | +2.0% |
| attn qkv | 0.797 / 80.8 | 0.796 / 80.9 | +0.1% |
| attn k/v | 0.100 / 53.8 | 0.100 / 53.9 | +0.3% |
| ssm_ba | 0.033 | 0.033 | -0.1% |

**加权 (按模型时间): cublasLt = -1.11% of GEMM -> -0.67% of prefill kernel time。**
判定线是 >= +3% -> **REJECTED, 不进 Step 2, llama.cpp 调用侧不动**。

## 3. T01 / T02 矛盾的最终解释 (三个因素, 都已实测)

1. **方向不同 (主因)**: T02 的 harness 调用是 `GemmEx(m=N tokens, n=M out, A=激活, B=权重)` = 转置方向。
   在**那个方向**上确实: gemmEx 默认 78.0 TF vs cublasLt 83.2 TF (+6.7%) -> T02 的 "+9%" 由此而来。
   但 llama.cpp 用的是**反方向** (m=out_dim), 在那个方向 cublasLt 没有优势, DEFAULT 已是最优。
2. **math mode**: llama.cpp 设 `CUBLAS_TF32_TENSOR_OP_MATH`; 我的 ABAB 里
   `DEFAULT_TENSOR_OP + TF32 math` 比 `+ TENSOR_OP math`(T02 harness) 和 `+ DEFAULT math`(T01 bench) 快
   (gate/up +2.8~2.9%, down +8.6%, k/v +61%)。**llama.cpp 当前配置已是三者最优**。
3. **预热/时钟**: 同一调用 gate/up 背靠背跑, 前 20 次 1.20ms -> 1500 次后 1.076ms (84.8 TF);
   而模型实际每次 1.163ms -> 主要原因是**真实混合负载下 SM 时钟只有中位 1447MHz** (纯 GEMM 循环是 1530MHz,
   nvidia-smi 在负载确认后 10s 采样 64 点: min 217 / p10 960 / med 1447 / max 1515)。-5.4% 与 1.082->1.141ms 吻合。
   **不是可用优化空间** (锁频不在范围内)。
4. 附带否证: 显式 algo (ALGO0..15_TENSOR_OP) 在真实方向全部比 DEFAULT 慢 (gate/up 最好 62.6 vs 82.1)
   -> 与之前"全局 ALGO8 让 pp512 -68%"完全一致, 那条路彻底封死。

## 4. 对既有结论的修正 (T02/T09 的盈亏平衡线)

T02 的 baseline "dequant + cuBLAS = 1.404ms" 是在**转置方向**测的; 真实方向 + 真实时钟下
baseline 约 1.36-1.40ms, 盈亏平衡线约 66-70 TF。**T02 (57.5 TF) 与 T09-A (40.9 TF) 的否决结论不变**。

## 5. 遗留可选线索 (未进一步投入)

- gate/up 在"背靠背连续 GEMM"下会退化到 1.15ms, 而"每次 GEMM 前写 178MB" (模拟反量化) 反而稳定 1.082ms
  (t08_interf: mode0 1.152 vs mode1/2/3 1.081)。机制未完全解释 (疑似 L2 dirty-line 周转),
  但它不是模型侧 78 TF 的主因 (主因是时钟), 且模型本来就是 mode1 模式 -> 判为无可利用空间。
---

# PPL 偏移溯源 (4.3572 -> 4.3569) | implementer, 2026-09-22

问题: 当前交付的 PPL = 4.3569, 基线 4.3572, 偏移 0.0003 来自哪一项改动?
方法: 用 3 个 DLL 做端到端 A/B (同 seed 42, 512ctx/8chunks) + 一次隔离构建。

| DLL | 内容 | PPL |
|---|---|---|
| `ggml-cuda-old.dll` | 原始 (pristine) | **4.3572** |
| `ggml-cuda-fast.dll` | + 向量化 dequant (Q6_K/Q5_K) | **4.3572** (与原始**逐位相同**) |
| 隔离构建 (unary.cu 回退后重建) | dequant + GDN vec4 | **4.3569** |
| 当前交付 DLL | dequant + GDN vec4 + silu vec4 | **4.3569** |

结论: **偏移来自 GDN vec4 (gated_delta_net.cu), 与 dequant 和 silu 无关**
- 向量化 dequant: 运算与索引和标量版完全一致 (同样的 `(d*sc)*q` 顺序 + 同样的 `__float2half` 取整) -> 逐位相同
- silu vec4: 纯 elementwise, 每个元素表达式不变 -> 逐位相同 (加回后 PPL 不再变化)
- GDN vec4: 改变的是**每个 lane 拥有的状态元素** (标量: 元素 r*32+lane; vec4: 元素 4*lane+r),
  因此 warp 内点积 `sum_r s[r]*q[r]` 的**求和分组/顺序**改变 -> fp32 加法重结合 -> 约 1 ulp
- 量级: 0.0003 (相对 0.007%), 远小于验收门槛 ±0.013, 也远小于 PPL 自身的 ±0.24 误差棒; 属良性
- 记录: 之前的 "fp32 重结合" 归因正确, 但责任文件是 **gated_delta_net.cu** (此前误记为 dequant)

附带复核: 同源码重新构建的 DLL 与上次交付版仅差 4 字节 (PE header/.rdata 构建元数据),
`.text` 逐字节相同, PPL 一致 -> 构建功能等价 (Windows/CUDA 构建嵌入 PDB GUID/时间戳, 非逐字节可复现)。

---

# T10: 长上下文 attention 侦察 (只测量) | implementer, 2026-09-22

方法: nsys (`--trace=cuda --cuda-event-trace=false`) 抓 3 个场景 + 1 个 ub2048 诊断。
按时间轴切分相位 (ub512 每 16 次 FA launch = 1 个 ubatch), 按 kernel 类别聚合。
原始 kernel 汇总: `artifacts/t10_{pp4096_d0,pp4096_d32k,pp32768,d32k_ub2048}_kern_sum.csv`

## 目标准则 (口径)

attention = 16 层 full-attn (GQA 24/4, head_dim 256)。FLOP = 4*256*24*(n_q*d_prev + n_q*(n_q+1)/2)
(causal, QK+PV 都算), 即"有用 FLOP"。

## 1. 按相位分解 (GPU kernel 时间, 实测)

| 场景 | 窗口 | 总计 | GEMM | attention | dequant | GDN | other |
|---|---|---:|---:|---:|---:|---:|---:|
| pp4096 d=0 (1 forward) | 4181.5ms | - | 2458.3 (58.8%) | **139.1 (3.3%)** | 790.5 (18.9%) | 314.9 (7.5%) | 478.7 (11.4%) |
| pp4096 @depth32k (measured eval = 4096 tok) | 5934.3ms | - | 2454.9 (41.4%) | **1899.4 (32.0%)** | 796.5 (13.4%) | 313.4 (5.3%) | 470.2 (7.9%) |
| pp32768 (1 forward = 32768 tok) | 39876.2ms | - | 19826.7 (49.7%) | **7368.2 (18.5%)** | 6335.6 (15.9%) | 2546.4 (6.4%) | 3799.2 (9.5%) |

- pp4096@depth32k = fill 32768 (64 ubatch) + warmup eval + measured eval; 上表是 measured eval 窗口
  (FA timeline 显示 eval 从 47021ms 起, 到 53337ms, 与 llama-bench 649 t/s -> 6311ms/forward 吻合)
- pp32768 = 2 个 forward (无 fill), 上表是第 2 个 forward

## 2. attention FLOP / TFLOPS (有用 causal FLOP)

| 场景 | attention 时间 | 有用 FLOP | TF/s | 占该场景 |
|---|---:|---:|---:|---:|
| pp4096 d=0 | 139.1 ms | 3.30 TFLOP | 23.7 | 3.3% |
| pp4096 @depth32k | 1899.4 ms | 56.07 TFLOP | 29.5 | 32.0% |
| pp32768 | 7368.2 ms | 211.1 TFLOP | 28.7 | 18.5% |
| pp4096 @depth32k, **ub2048** (诊断) | 1404.2 ms | 56.07 TFLOP | **39.9** | 34.6% |

**逐 ubatch 效率曲线 (pp32768, 每 512 token 一个点): 25.9 / 27.5 / 28.3 / 28.9 / 29.0 / 29.4 / 29.4 / 29.7 TF/s
(depth 4k -> 32k), 深度 32k 的 eval 段 33.1-33.7 TF/s -> 效率不随深度衰减** (不是长上下文特有的缺陷)。

## 3. 命中的 kernel 变体 (sm70)

- 唯一变体: `flash_attn_ext_f16<(int)256, (int)256, (int)32, (int)2, (bool)0, (bool)0, (bool)0>`
  = mma (tensor core, HMMA.884) 路径, DKQ=DV=256, ncols1=32, ncols2=2 (=64 列/tile), 无 softcap / 无 precise-softmax / 无 sparse
- 启动配置: block=(32,4,1)=128 线程 (4 warps), **grid=(192,1,1)** — 所有深度/所有场景都一样
- 没有命中 vec / stream-k fixup / sparse 变体
- (ub2048 时同一 kernel 的 grid = (768,1,1), 另有 `flash_attn_mask_to_KV_V_max` 预处理核 18us x 320 = 5.8ms, 可忽略)

## 4. 关键诊断: 不是 kernel 慢, 是 ub512 让 grid 太小

| | ub512 | ub2048 | 变化 |
|---|---|---|---|
| grid | 192 CTA (n_q/64 x 24 head x gqa/2) | 768 CTA | x4 |
| attention (depth32k eval) | 1899.4 ms | 1404.2 ms | **-26%** |
| attention TF/s | 29.5 | **39.9** | **+35%** |
| 同一 eval 的 dequant | 796.5 ms | 160.4 ms | -80% (另一话题) |

- 机制: `launch_fattn` (fattn-common.cuh:1139) 在 `stream_k=true` 分支里, 用
  `should_use_stream_k` 判断 (第 1150 行: `tiles_efficiency_percent < 75`)。
  ub512 时 ntiles_dst=192, occupancy 限制 max_blocks_per_sm=1 -> max_blocks=80 ->
  waves=3 -> 效率 192/240 = **80% >= 75% => 不拆分**, `blocks_num.x = ntiles_dst = 192`。
  即 512 token 的 prompt 在 80 SM 上只有 192 个 128 线程的 CTA (1 CTA/SM, 4/64 warps) + 3 个波次
  的打尾 (第 3 波只有 32 CTA) -> 吞吐被并行度和打尾同时限制。
- ub2048 时 ntiles_dst=768 -> 效率 768/800=96% -> 波次饱和平滑 -> 39.9 TF/s。

## 5. 结论 (T10 验收要求: 值得动 / 不值得动 + 工程量 + 预期)

**移植 1Cat 的 split-D/N32: 不值得。**
- 1Cat FA-V100 同形状 29-38 causal TF/s; 我们 ub2048 已 **39.9 TF/s** (在其区间之上),
  ub512 的 29.5 是**调度/并行度**问题, 不是 kernel 计算效率问题
- 他们的 "split-D/N32 相对 generic FA2 1.23-1.6x" 是相对更差的基线; 对我们没有已证明的空间
- 若移植: 3-6 天 (新 kernel + 数值/PPL 验证), 预期 <= 1.0x (按 ub2048 已达其上限)

**ub512 口径下唯一可做的近路: 让 KV-split (parallel_blocks) 在 ub512 生效**
- 代码里已有该机制 (fattn-common.cuh:1188+ `parallel_blocks_test` 循环, grid = ntiles_dst x PB + fixup),
  但 `stream_k=true` 分支绕过了它; 且 stream-K 分支即使启用也只会给 max_blocks=80 个 block (比 192 更少)
- 做法: stream_k 分支判定不启用时, 回退到 parallel_blocks 循环 (或对该配置强制 PB=2/4)。
  预期 grid 192 -> 384/768, 波次效率 80% -> 96%, 参考 ub2048 实测 -> attention -20~26%
- 预期收益 (ub512): pp4096@depth32k **+5-8%** / pp32768 **+3-5%** / pp8192 **+1.2%** / pp4096 **+0.7%** / pp512 **+0.2%**
  (按 attention 占比 x 20-26%; 长上下文才有意义)
- 成本: 0.5-1 天 (启发式改动 + fixup 正确性 + PPL 门槛), 中低风险
- 数值: KV 分块会改 softmax 归并顺序 -> 约 1 ulp 级重结合 (同 GDN vec4 先例, PPL 门槛内)

**其它 (更大杠杆, 已单独在板上)**: ub 提高是全局最大单项 (T04: ub2048 实测 +21%, 且 attention/dequant
都随 ub 改善); 1Cat 的 chunked prefill 架构与之同源。

---

# T07 剩余项: rms_norm 向量化 | implementer, 2026-09-22 | **REJECTED (kernel 级负收益)**

实现: `rms_norm_f32_vec4<block_size, do_multiply, do_add>` (float4 两趟 + 对齐/整除条件 + 标量回退),
覆盖 plain / mul / mul+add 三条路径。patch 已归档 `artifacts/t07_rms_norm_vec4_REJECTED.patch`, **未入库 (已回退)**。

## Kernel 级 (nsys, pp4096, 同命令同口径, 2 个 forward 合计)

| 变体 (调用/ubatch) | before | after (vec4) | Δ |
|---|---:|---:|---:|
| `rms_norm_f32<256,true>` (mul, 80) | 54.86us x 1280 = 70.23ms | 58.46us x 1280 = 74.83ms | **+6.6%** |
| `rms_norm_f32<1024,true>` (mul, 129) | 33.89us x 2064 = 69.94ms | 37.44us x 2064 = 77.28ms | **+10.5%** |
| `rms_norm_f32<256,false>` (32) | 22.55us x 1536 = 34.64ms | 21.32us x 1536 = 32.74ms | -5.5% |
| **合计** | **174.81 ms** | **184.85 ms** | **+5.7% (更慢)** |

## 端到端 A/B (交替 4 轮, `-r 3`, ub512, 无 tg)

| 轮 | DLL | pp512 | pp4096 | pp8192 |
|---|---|---:|---:|---:|
| 1 | A (vec4) | 965.22 | 941.05 | 915.39 |
| 2 | B (base) | 961.20 | 935.82 | 913.10 |
| 3 | B (base) | 955.22 | 932.39 | 909.10 |
| 4 | A (vec4) | 958.49 | 933.07 | 908.60 |
| 均值 A | | 961.86 | 937.06 | 911.99 |
| 均值 B | | 958.21 | 934.11 | 911.10 |
| Δ | | +0.38% | +0.32% | +0.10% |

- 轮间漂移 (同 DLL 两轮) 达 0.4-0.75% -> 上表 Δ **在噪声地板内**, 与 kernel 级 (+5.7% 更慢 = 端到端 -0.12%) 方向矛盾
- 按 PROTOCOL "单次测量 +/-0.5% 不能作判据" + T07 自身验收条件 (kernel 级为准) -> **拒绝**

## PPL (门槛 4.3572 +/- 0.013)

- A (vec4): **4.3563**; B (4 文件基线): 4.3569 -> Δ 0.0006, 门槛内 -> 纯性能否决, 无质量问题

## 根因 (为什么没有收益)

1. `rms_norm_f32<1024,true>` (ncols=5120, 130 次/ubatch) **已达 DRAM 带宽极限**: 单次 512 行 x 5120 列 x 3 遍
   (2 读 1 写) = 31.5MB / 33.89us = **930 GB/s** -> 无空间, 向量化只带来负载不均 (ncols4=1280 在 block=1024
   下 256 线程要做第 2 轮)
2. `rms_norm_f32<256,*>` (ncols=128/256, 176 次/ubatch) 是**延迟受限** (约 57 GB/s): 单次 22-58us 中大头是
   固定开销/归约/DRAM 延迟, 不是指令条数 -> float4 减少指令无用
3. 上限估算: 即使做完美也只有 4-9ms/forward = **0.1-0.2% 端到端**, 低于测量噪声地板 -> 不可验证

## 结论

- rms_norm 不再投入 (T07 全部关闭: silu 已入库 +5%/kernel, rms_norm 拒绝)
- 小 kernel 的延迟受限问题属于 **T05 剩余项 (减少调用次数/融合)** 的范畴, 向量化解决不了
- 交付状态已回退到 4 文件, DLL 重新构建部署, 体检: PPL 4.3569 / pp512 962.1 / pp4096 937.4 /
  pp8192 912.7 / tg128 26.68

---

# T10 近路: ub512 下启用 FA KV-split (stream-K 启发式) | implementer, 2026-09-22 | **REJECTED (BLOCKED)**

## 改动 (最小化, 已回退)

`fattn-common.cuh` `should_use_stream_k` +13 行:
- 新增 env 开关 `GGML_CUDA_FATTN_STREAM_K` (0=禁用/1=强制/未设=自动), 便于即时回退
- 新增条件: `NVIDIA && ntiles_dst > max_blocks && tiles_efficiency_percent < 96` -> 启用 stream-K
  (ub512 + depth32k: ntiles_dst=192, max_blocks=80, 效率 80% -> 触发; ub2048 的 96% 不受影响)
- 机制: grid 192 CTA/3 波 -> 80 CTA/1 波 + `flash_attn_stream_k_fixup_general<256,32,2>` 归并

## nsys 验证 (pp4096@depth32k, 与 T10 同口径)

| 项 | before | after |
|---|---:|---:|
| FA grid | (192,1,1) | **(80,1,1)** |
| fixup kernel | 无 | `stream_k_fixup_general<256,32,2>` 1280 x 23.55us = 30.1ms |
| eval attention | 1899.4 ms | **1792.7 ms (-5.6%)** |
| eval attention TF/s | 29.5 | **31.3** |

- 波次效率模型 (80% -> 100% 应 -20%) **高估**: 实测只 -5.6%。
  原因: stream-K 下每 block 串行做 ~2.4 个 tile + tile 接缝的 needs_fixup 归并, 单 block 效率下降,
  抵消了大部分打尾收益

## 端到端 A/B (交替换 DLL 2 轮, gate #1)

| 指标 | A 均值 (R1/R4) | B 均值 (R2/R3) | Δ |
|---|---:|---:|---:|
| pp4096 @d32768 | 666.35 (665.89/666.81) | 653.33 (654.28/652.38) | **+2.0%** |
| pp32768 | 788.67 (788.47/788.86) | 780.10 (781.82/778.38) | **+1.1%** |
| pp512 | 946.29 | 951.47 | -0.55% |
| pp4096 | 930.96 | 928.54 | +0.26% |
| pp8192 | 908.54 | 904.47 | +0.45% |
| tg128 | 25.15 | 24.89 | +1.0% (噪声 ±0.9) |
| ub2048 pp4096 (gate #5) | 1152.46 | 1153.39 | -0.1% (无回退) |

## 验收判定 (analyst 门槛)

| gate | 要求 | 实测 | 判定 |
|---|---|---|---|
| #1 depth32k | >= +3% | +2.0% | **FAIL** |
| #1 pp32768 | >= +2% | +1.1% | **FAIL** |
| #2 pp512/4096/8192 | 不回退 >0.5% | -0.55%/+0.26%/+0.45% | 噪声内 (无收益) |
| #2 tg128 | 不回退 | +1.0% (与机器漂移同量级) | PASS |
| #3 PPL | 4.3572 +/- 0.013 | A 4.3568 / base 4.3569 | PASS (数值机制安全) |
| #4 nsys TF/s 前后 | 必须给 | 29.5 -> 31.3, grid 192->80, fixup 出现 | PASS |
| #5 ub2048 | 不回退 | -0.1% | PASS |
| #6 最小改动/env 开关/patch | - | +13 行, `GGML_CUDA_FATTN_STREAM_K`, 已归档 | PASS |

## 结论 (按 Q6 规则)

- **A 失败**: 效果真实但只达门槛的 ~1/3~2/3 (attention -5.6%, depth32k +2.0%, pp32768 +1.1%)
  -> 不硬凑, 已回退 (工作区 0 行改动), patch 归档 `artifacts/t11_fattn_kvsplit_REJECTED.patch`
- 保留的事实: **ub512 的长上下文 attention 确实受 grid/波次限制**, 但 stream-K 这一条路只能回收 ~1/4;
  ub2048 的 39.9 TF/s 主要来自"每 launch 4x 并行度 + 4x 更少 launch 次数"的整体效应, 不是单纯打尾
- 机器状态注记: 本次 A/B 期间全指标比历史基线低 ~1% (pp512 ~951 vs 958-963; tg128 25.1-25.7 vs 26.6),
  故所有比较都用同 session 的交替 A/B, 绝对值不能与历史数字混用
- 下一步: 按 Q6 = **转 T03 chunked** (prefill 全局 +3-4%, timebox 1.5 天)

---

# T03 (chunked GDN 重写): 0.75 天 checkpoint 报告 | implementer, 2026-09-22

状态: **数学原型 + 三个 checkpoint 全部通过 (机器精度)**; CUDA 实现未开始 (下一 block)。

## Checkpoint 结果 (analyst 要求的三项)

| # | checkpoint | 方法 | 结果 |
|---|---|---|---|
| 1 | decay 递推与 scan 参考一致 | numpy 对拍 (随机输入, L=1..64, D=8..128) | **通过, 相对误差 ~1e-16** |
| 2 | 三角求解正确性 | WY 恒等式数值验证 + 全流程对拍 | **通过**: WY 恒等式 4.4e-16; 全流程 1e-16 |
| 3 | keep_rs_t / chunk 边界状态传递 | 多 chunk 状态交接 (T=256, L=64) 与串行版对拍 | **通过, 1.1e-15** |

验证脚本归档: `artifacts/t03_chunked_reference.py` (可直接当 CUDA 移植的 ground truth)。

## 推导出的算法 (per chunk, per (seq, head))

```
Ac      = exp(cumsum(g))                                  # 块内累计 decay (checkpoint 1)
Bt      = beta / Ac
KKT     = K K^T
A_beta  = inv(I + tril(diag(beta) KKT, -1)) @ diag(beta)  # L x L 三角求解 (checkpoint 2)
W       = A_beta @ K            # L x D
U       = A_beta @ (diag(1/Ac) V)
Sn_new  = Sn_in + K^T (U - W Sn_in)                       # 归一化状态 (checkpoint 3)
S_out   = Ac[L-1] * Sn_new                                # 反归一化交给下一 chunk
QK      = Q K^T
Qtil    = Q - tril(QK, 0) @ W
C       = QK - tril(QK, 0) @ A_beta @ tril(KKT, -1)
O       = scale * Ac * ( Qtil @ Sn_in + tril(C) @ (Bt * V) )
```
- 关键点: **decay 是标量 (非 KDA)**, 所以把状态按块内累计 decay 归一化后, 块内递推无 decay,
  只需每块一次 cumsum+exp 和一次标量反归一化 -> checkpoint 1 极简
- 实现时不需要显式求逆: 用前代 (forward substitution) 三次, 同一个 (I+M) 矩阵, L 个右端项:
  `W_t = beta_t (k_t - sum_{s<t} (k_t.k_s) W_s)`, U 同理, 第三组解出 `R = A_beta @ tril(KKT,-1)` (给 C 用)
- 成本估算 (per chunk per (seq,head), L=64, D=128): KKT 0.52M + 三角求解 ~0.4M + 状态 2.1M + 输出 1.6M
  ≈ 4.6M MAC = 9.2 MFLOP (对比串行 kernel 每块 3.1M MAC) -> FLOP 多 1.5x 但全是规整 matmul,
  现状 kernel 只有 ~0.5 TFLOPS (指令吞吐受限), 目标 2 TFLOPS 级即可满足 -50%

## 设计 (待实现)

1. **K1 (chunk 并行)**: per (seq, head, chunk) 计算 KKT / 三次前代 / W / U / QK / Qtil / C / 块内输出项
   `tril(C)@(Bt*V)`; 存 W, U, Qtil, C (或 C 的块内输出) 到 scratch
2. **K2 (chunk 串行, (seq,head) 并行)**: 状态扫描: 逐 chunk `Sn += K^T(U - W Sn)` + 状态项输出
   `Qtil@Sn_in` (需 K, W, U, Qtil; L=64/D=128 时每 chunk 约 2.1M+1.05M MAC)
3. 合计 2 个 kernel + scratch (per chunk: W,U 各 L*D*4B + C L*L*4B ≈ 80KB @L=64,D=128)
4. 限制: 仅当 `n_tokens > 1 && !keep_rs_t && !KDA && S_v == 128` 走新路径; 其余保持现有 kernel
   (decode 单 token 路径完全不动)
5. 预期: GDN 40ms -> ~20ms (整机 pp512 +3-4%); 长上下文 (pp32768/depth32k) 同比例受益

## 下一步 (未完成, 不硬凑)

- CUDA 实现 K1/K2 + 板载三角求解 (warp/block 级前代)
- 验收: kernel 级 (目标 -50%) + 端到端 pp512/4096/8192/32768/depth32k + tg128 不回退
  + PPL |x-4.3572| <= 0.013 + 200 token 生成检查; 同 session A/B 交替 >=2 轮

---

# T03 CUDA 原型 V1: chunked GDN kernel 正确性达成, 性能未达标 | implementer, 2026-09-22

产物: `artifacts/t03_chunked_v1.cu` (harness: 参考实现 + chunked V1 + 中间量对拍),
`artifacts/t03_check_state.py` (numpy 逐阶段校验)。工作区未改 (llama.cpp 0 行)。

## 正确性 (harness, D=128, H=32, L=32, COLS=32)

与"逐 token 串行参考"对比 (纯 fp32):

| tokens | chunks | out rel err | state rel err |
|---:|---:|---:|---:|
| 32 | 1 | 5.7e-7 | 1.4e-7 |
| 64 | 2 | 4.9e-7 | 1.5e-7 |
| 128 | 4 | 4.8e-7 | 1.8e-7 |
| 256 | 8 | 5.0e-7 | 1.8e-7 |
| 512 | 16 | **5.3e-7** | **1.7e-7** |

- 16 个连续 chunk 后仍只有 ~1e-7 相对偏差 -> **算法与实现正确, 三个 checkpoint 在实际 CUDA 代码上通过**
- 逐阶段 numpy 对拍 (W/U/R/QK/T1/OS/C, Ac/Bt) 全部到 fp32 精度; 过程中修掉两个真 bug:
  1. C 的内层求和缺 `m <= t` 掩码 (tril(QK)) -> 12% 输出误差
  2. **U 的右端项写成 `beta_t*Bt_t*v` (多乘一个 beta)**: 正确形式是 `U = A_tri @ (diag(1/Ac) V)`,
     即前代右端项 `v_t/Ac_t` (不是 `Bt_t*v_t`); 该错误使状态 32% 偏差

## 性能 (per layer-ubatch, H=32, T=512)

| 实现 | 时间 | 说明 |
|---|---:|---|
| harness 串行参考 (1 线程/列, 1 block/head) | 1.86 ms | 弱并行, 仅作正确性基线 |
| **chunked V1** | **4.60 ms** | 0.40x vs 上面那个参考 |
| **真 kernel (现状, 实测)** | **0.826 ms** | llama.cpp 现有实现 (grid H x n_seqs x 32) |

- V1 比真 kernel **慢 5.6x** -> **性能不达标, 未集成**
- 已定位的瓶颈 (下一版必修):
  1. **smem bank conflict**: `sM[jj*D + m]` / `sK[s*D + i]` 在连续线程上 stride=128 -> 32-way 冲突
     (T1/QK 两个主循环) -> 加 pad (D+1) 即可
  2. **occupancy**: 73KB smem -> 1 block/SM (8 warps = 12.5%); L=16 + pad 后约 43KB -> 2 blocks/SM
  3. 串行前代 + C 阶段只用 32/256 线程 -> 需并行化 (或 warp 级)
  4. 每 chunk 8 次 __syncthreads x 16 chunks; 可合并阶段
- 参考成本: 每 chunk 每 block ~0.56M MAC (COLS=32 时分块间还会重复 KKT/solve/QK), 理论上限远高于现状

## 结论与下一步

- 算法侧完成 (checkpoint 全过); 工程侧 V1 正确但慢, 需要一次性能重写 (换 2D tiling / 寄存器分块 /
  pad / 更高 occupancy), 预计还需要一个 ~半天到一天的 block
- timebox 记账: 数学原型 ~0.4 天 + CUDA V1 ~0.35 天; 性能 pass 未开始 -> 是否继续由 analyst 裁决

---

# T03 V2 进展报告 (0.5 天点): 资源门槛全过, 速度门槛未达 | implementer, 2026-09-22

产物: `artifacts/t03_chunked_v2_L16.cu` (V2b, L=16)。工作区仍 0 行改动。

## 迭代路径 (harness, D=128, H=32, T=512, per layer-ubatch)

| 版本 | 改动 | 时间 | smem | occupancy | 正确性 (out/state rel) |
|---|---|---:|---:|---|---|
| V1 | 初版 (L=32, COLS=32) | 4.60 ms | 73 KB | 1 blk/SM | 5.3e-7 / 1.7e-7 ✓ |
| V2a | sK/sM 加 pad (消 32-way bank conflict) + C 阶段全并行 | 3.29 ms | 77 KB | 1 blk/SM | 5.3e-7 / 1.7e-7 ✓ |
| **V2b** | **L=16** | **1.644 ms** | **43.0 KB ✓** | **2 blk/SM ✓** | **5.7e-7 / 1.7e-7 ✓** |
| (现状真 kernel) | - | 0.826 ms | - | - | - |

- 门槛检查: smem <=48KB ✓, >=2 block/SM ✓, 正确性 ✓ -> **但速度 1.644ms > 0.60ms 门槛 -> 按 gate #1 不集成**
- 相对 V1 已快 2.8x; 相对现状 kernel 仍慢 2.0x

## 仍慢的根因 (实测推算)

- L=16 时每 chunk 每 block 约 276K MAC / 256 线程 = ~1080 FMA/线程, 但实测每 chunk ~51us = ~71K cycles
  -> **~66 cycles/FMA**, 即算力只用了 1.5%
- 结构性原因: 每 chunk 8 个阶段各带 __syncthreads (L=16 时 32 chunk x 8 = 256 次 sync/block),
  每阶段每线程实际工作只有几十~几百条指令 -> 大量延迟暴露 (smem ~30 cycle + barrier),
  occupancy 只有 25% (16 warps/64) 不足以掩盖
- 结论: 不是单点 bug, 是 V1/V2 的**阶段式 + 块级同步**结构本身低效

## 下一版要改的 (若继续)

1. 减少阶段与同步: 把 KKT/QK/T1/A 合并为一次 2D 分块扫描 (寄存器分块, 每线程算一个 2x2/4x4 tile)
2. 前代求解改 **分块三角求解** (对角小块串行 + 块间 matmul 更新), 并让 W/U/R 三系统一起做
3. 状态更新/输出合并, 减少 smem 往返
4. 或改成 K1(chunk 并行, 大 matmul)/K2(chunk 串行, 轻量扫描) 两 kernel 结构, 用全局 scratch 换并行度
- 预期: 每次只做上述 1-2 项即可过 0.60ms; 但都需要一次结构性重写 (~0.3-0.5 天)

## timebox 记账

- 数学原型 ~0.4 天 + V1 ~0.35 天 + V2 ~0.15 天 = ~0.9 天 (硬上限 1 天)
- 按 analyst 规则: 这个 0.5 天点报告后, 若继续需要再投入 ~0.3-0.5 天做结构性重写;
  若不继续则 BLOCKED 归档 (算法已验证可行, 工程成本超预期)

---

# T03 V3 (合并阶段) 结果: barrier 假设被否证, 提前停 (analyst 0.25 天规则) | implementer, 2026-09-22

产物: `artifacts/t03_chunked_v3_merged.cu`。工作区仍 0 行改动。

## 改动与结果

V3 = 把 V2b 的 ~10 个块级同步合并到 **4 个** (phase1: Ac/Bt+KKT+sK 加载; phase2: 三次前代+QK;
phase3: T1/A+C; phase4: 输出(inline OS)+状态更新(折叠反归一化)), 另把 OS 改成输出时内联重算, 省一个 buffer。

| 版本 | barrier/chunk | 时间/layer-ubatch | us/chunk | cycles/FMA |
|---|---:|---:|---:|---:|
| V2b (L=16) | ~10 | 1.644 ms | 51.4 | ~66 |
| **V3 (L=16, 合并)** | **4** | **1.844 ms** | **57.6** | **~80** |

- **cycles/FMA 没有改善 (反而更差)** -> 按 analyst 过程规则 4 (0.25 天点无 ~2x 改善即提前停) **停止**
- V3 还带一个 sK 加载 bug (把 strided 循环误写成单次条件加载), 修掉后只会再慢一点, 不影响结论
- smem 41.0KB, 2 block/SM (16 warps) 均已达标

## 诊断结论 (这轮最有价值的产出)

barrier 假设**被否证**: 同步次数减半以上, 每 chunk 时间反而 +12%。真瓶颈是**每线程串行点积循环里
暴露的访存延迟**:

- KKT/QK/T1/A 四个阶段都是 "一个线程对一对 (t,s) 或 (t,jj) 做长度 128 的串行点积":
  `for m < 128: acc += W[m][t]*M[jj][m]` — 每迭代 2-3 次 smem/L2 访问, 循环上界是运行时值
  (`len`) 导致编译器不展开 -> 每次迭代的 L2/smem 延迟 (~30-200 cycles) 直接暴露
  (而 barrier 只是把这些延迟串起来, 减少 barrier 不解决延迟本身)
- 实测算力利用率 ~1.3% (1080 FMA/线程 vs 57.6us/chunk = 86K cycles @1.5GHz)
- 现状 kernel 用的正是 "warp 协作 + 归约" 的方式, 所以它 0.826ms 反而更快

## 修炼方案 (明确, 但需要一次重写, 未做)

1. 四个点积循环改成 **warp 协作 + float4**: 一个 warp 负责一对 (32 lane x 4 floats = 128) -> 5 次
   shuffle 归约; 预计这些阶段 ~6x (13K -> 2K cycles/chunk)
2. 前代求解循环展开 (按 L 固定长度模板化) + 每线程 2-4 个累加器
3. 预计合并后: 57.6us -> ~10-15us/chunk -> 0.32-0.48ms/layer-ubatch (**过 0.60 门槛**) 
   -> 需 ~0.3-0.4 天; 但已超本轮 0.5 天时段的可用余量 (且需重跑正确性/门槛全套)

---

# T03 结案: GDN kernel 加速路线全部否证, 无可行路径 (implements, 2026-09-22) | NO PATH FOUND

工作区已回退到交付态 (4 文件, 无本任务净改动), PPL 4.3569 复核通过。
产物: `artifacts/t03_existing.cu` (现状 kernel 复刻 + 分段计时), `t03_variant.cu` (C 列/warp),
`t03_sweep.cu` (occupancy 扫描 + expf 消融), `t03_stage.cu` (smem 双缓冲), `t03_chunked_v1/v2_L16/v3_merged.cu`。

## 全部尝试与实测 (harness H=48 真实 head 数; 现状 kernel 实测 0.814 ms/layer-ubatch -> harness 复刻 0.922)

| 尝试 | 机制假设 | 实测结果 | 判定 |
|---|---|---|---|
| 寄存器预取 (t+1 加载提前) | 全局加载延迟暴露 | ~1.0x (0.88/0.83) | ✗ 假设否证 |
| **C=2 列/warp** | 归约与加载延迟按列摊薄 | harness 0.85-0.90; **真实模型集成后 867us vs 814us = +6%** | ✗ 无常驻收益 (warp 总数减半抵消) |
| C=2 + MB=12 (48 warp/SM) | 占用不足 | 1.01ms (寄存器压到 40, spill) | ✗ 假设否证 |
| block 级 smem 双缓冲 staging | 用 smem 替代寄存器预取 | 1.15ms+ (且实现有 bug) | ✗ 更慢 |
| 分段 clock64 计时 | 定位瓶颈 | loads 4%, **kv_local+reduce1+expf 69%**, 其余 26% | 诊断有效 |
| warp 归约微基准 | 归约延迟 vs 吞吐 | 单次 144.5 cycles 延迟, **ILP 可完全掩盖** (C=8 -> 18 cycles/次) | 机制成立但需 ILP |
| **`__expf` 替换 `expf`** | 指令数 (expf ~20 条/次) | **0.922 -> 0.838 (-10%)**, state_rel 7.5e-8 -> 1.5e-7 | ✓ 真实但小 |
| 无 expf 消融 (数值故意错) | expf 的总代价上限 | 0.792 (-14%) | expf 全部价值只有 14% |

## 为什么没有 2x (被实测限定的结论)

1. **不是内存延迟瓶颈**: 预取和 smem staging 两条独立路径都失败 -> 40 warp/SM 的驻留已经掩盖了访存延迟
2. **不是占用瓶颈**: 强行上 48 warp/SM (MB=12) 反而 -10% (寄存器 spill)
3. **不是归约延迟瓶颈**: C=2 把每列归约减半, 但 warp 总数也减半, 真实模型实测净 -6%
4. **指令数上限**: expf 只占 14%, 去掉后仍有 0.79ms -> 剩余指令 (FMA/归约/寻址/存储) 的下限就在 0.75ms 级
5. 现状 kernel 在 H=48 下已接近"指令+延迟平衡"的实际下限 (~0.8ms); **analyst 的 -50% (0.41ms) 不可达**

## chunked 路线 (V1/V2b/V3) 的量化否决

- 算法正确 (numpy 1e-16 / CUDA 5e-7), 但 FLOP 比串行多 1.4-2.4x (列切分重复 + KKT/QK/T1/A 开销)
- 最好 V2b = 1.644ms (harness) = 现状 2x 慢; 要过 0.60 需把效率从 ~1.5% 提到 >25% 峰值 (30x) -> 不现实
- 结论: chunked 只是把"每 token 的串行点积"换成"每 chunk 的串行点积", 没有碰到真瓶颈

## 保留价值

- `__expf` 是一个**真实可行但低于集成门槛**的选项 (kernel -10%, 端到端约 +0.7%, state_rel 1.5e-7 远在 PPL 门槛内);
  按 analyst gate "0.60-0.83 不集成" 未集成, 若未来放宽门槛可一行启用 (记录在此)
- 三份 harness (现状 kernel 复刻 / C 列变体 / occupancy 扫描) 可复用, 分段 clock64 计时法可复用于其他 kernel 诊断

---

# T05 剩余 (quantize 融合 + 小 kernel 合并) 结案: 无可接受风险的实质收益 | CLOSED (evidence-based)

工作区 0 净改动 (rms_norm 实验已回退), PPL 4.3569 复核通过, 4 文件交付态。

## 方法修正 (重要, 影响后续所有 decode profile)

- nsys 默认 `--cuda-graph-trace=graph`: **图重放的 token 的 kernel 根本不进 kernel 表**。
  llama-bench tg128 里每个 token 走 cudaGraphLaunch (实测 7 次 graphLaunch / 8 token, 每个 ~1.5-2.4ms),
  默认 trace 只留下图之前 1-2 个 token 的 kernel 事件 -> analyst 原 T05 profile 是"首 token / 图前"数据。
  **正确做法: `nsys profile --trace=cuda --cuda-graph-trace=node`** (本次已用)。
- 首 token 与稳态差异已被量化 (见下): `scale_f32` 0.64 -> 0.15ms (96 个 ne=786432 的 state 清零 SCALE 只在首 token/rs_z>=0 时出现)。

## 稳态单 token 分解 (graph-replay 窗口, 2024 kernels, busy 34.33ms; wall 37.7ms)

| 类别 | calls | ms | %busy |
|---|---:|---:|---:|
| mul_mat_vec_q (全部) | 461 | 29.84 | 86.9% |
| quantize_q8_1 | 461 | 0.84 | 2.4% |
| rms_norm (3 变体) | 305 | 1.10 | 3.2% |
| elementwise 合计 (add/cpy/silu/sigmoid/softplus/concat/rope/fwht) | 460 | 1.03 | 3.0% |
| k_get_rows | 97 | 0.47 | 1.4% |
| gated_delta_net | 48 | 0.34 | 1.0% |
| flash_attn_ext_vec | 16 | 0.32 | 0.9% |
| scale_f32 (predelta 2/层) | 96 | 0.15 | 0.4% |
| set_rows (KV q8_0) | 32 | 0.11 | 0.3% |
| **kernel 合计** | **2024** | **34.33** | 100% |
| host/graph-submit 间隙 | - | ~3.4 | - |

## MMVQ 已饱和 (逐矩阵实测带宽)

| 矩阵 (Q6_K/Q5_K/Q8_0) | grid | calls | avg us | 实测 GB/s |
|---|---:|---:|---:|---:|
| lm_head Q8_0 (248320 行) | 248320 | 1 | 1598 | **845** (=94-99% 可用) |
| ffn down / attn o_proj Q6_K | 5120 | 55 | 98 | 746 |
| ffn gate 或 up Q6_K | 17408 | 25 | 102 | 713 |
| gate+up 融合 GLU Q6_K | 17408 | 26 | 213 | 685 |
| ffn Q5_K 部分 | 17408 | 26 | 82 | 752 |

- MMVQ 平均 677 GB/s (可用 825-850), 分矩阵 685-845 -> 剩余空间约 5-8% 且分布在各小矩阵,
  改 vec_dot/访存模式属研究级改动; 与 analyst "纯 kernel 路线到顶" 结论一致。

## 两个对照实验 (均为可逆, 已回退/未入库)

1. **rms_norm block 配置** (env GGML_CUDA_RMS_NORM_BS, 把 5120 宽 norm 从 block=1024 换成 256; 129 个单行 norm 占 0.73ms, 5.62us/次):
   - BS=1024: tg128 26.69/26.60, pp512 968.6/964.0
   - BS=4096 (即 <256>): tg128 26.63/26.68, pp512 963.2/962.5
   - **结论: 无效果 (噪声内)** -> 单行 norm 的 5.6us 是"跟在带宽饱和 kernel 之后的延迟+节点开销", 不是 block 配置问题。未入库。
2. **GGML_CUDA_DISABLE_FUSION=1** (量化"融合机制"的边际价值):
   - 关闭: tg128 25.93/25.97, pp512 950.4/945.2
   - 开启: tg128 26.63 (-2.6% when disabled), pp512 963.6 (-1.7%)
   - **结论: 现有全部融合 (GLU/norm/softplus/ssm_conv/GDN, 每 token 533 个融合头) 总共只值 0.70ms**;
     折算 **每消除 1 个 kernel 的边际价值约 0.9-1.0us**。

## 结案判定 (定量)

- 剩余可压缩项上限 = 461 (quantize) + ~590 (小 kernel) 个 kernel, 按 0.9-1.0us/kernel 折算 = **1.0-1.1ms = +2.7-3.0%** -> tg128 **27.3-27.5**。
- analyst 原估 "quantize 0.8 + 小 kernel 1.0 + 间隙 0.5 = -4.1ms / 28.5-30 t/s" 中的 0.8ms 与 1.0ms 两项,
  按实测边际价被高估约 2-4 倍; 间隙 3.4ms 是 host 侧 (logits 取回 + graph submit), 与 kernel 数几乎无关。
- 要到 28.5-30 必须把 ~1500 个小 kernel 合并进大 kernel (核心级重写, 多日, 高风险), 或 MMVQ 再快 10% (已饱和)。
- **建议: T05 剩余 CLOSED (无实质收益/风险比不成立); decode 侧 kernel 天花板约 27.3-27.5;**
  decode 的实际翻倍已由 MTP (42.7 t/s) 提供, 不建议再投 kernel 侧。

## 产物

- `artifacts/t05r_dec4gr.sqlite` (graph-node 级 trace, 稳态 token), `t05r_dec8.sqlite` (默认 trace, 对照)
- `artifacts/q_dec4gr*.py`, `q_sig.py`, `analyze_nodes*.py`, `nodes_raw.txt` (node->kernel 对应与分解脚本)
- `artifacts/trace_ops.patch` = GGML_CUDA_TRACE_OPS 诊断补丁 (打印图内 node 名/op/尺寸; 仅诊断, 未入库)

---

# T12/T16 (2026-09-23, implementer): T12 重放 + T16 实现 (静默时段: 仅代码与轻量准备)

## T12 (KV-split 重放集成, spec TASKS/T12)

- patch 重放成功 (fattn-common.cuh +13 行, env `GGML_CUDA_FATTN_STREAM_K` 保留); 工作区 5 文件
- PPL **4.3568** (门槛 4.3572+/-0.013) PASS
- nsys 机制验证 (pp4096@depth32k, `--cuda-graph-trace=node`): FA grid **192 -> 80** (+1280 CTA),
  `flash_attn_stream_k_fixup_general<256,32,2>` 1280 x 23.6us = **30.2ms** -> 与 T11 实测完全一致 (可复现)
- **完整 A/B 验收 (含 pp8192@depth128k 首次基线) 因用户夜间静音要求推迟到允许时段执行**

## T16 设计与实现 (spec 机制修正)

关键事实 (读码 + 实测资源):
- mma FA 内核 (`flash_attn_ext_f16<256,256,32,2,..>`) **只用 blockIdx.x** 做 work-item 分解 (stream-K),
  spec 设想的 `parallel_blocks` (grid = ntiles_x x PB x z) 路径 **在 mma 内核不存在** (那是 vec/tile 路径);
  强制 stream_k=false 会让 blockIdx.y 被忽略 -> 结果错误。因此 T16 的正确等价物 = 调 `blocks_num.x`。
- fixup 机制支持任意切分: `fixup_uniform` (grid = k*ntiles_dst, 逐 block 累积) 与 `fixup_general`
  (回溯 loop) 都能合并 >2 个 partial -> PB=2/4 数值上安全。
- 实测该内核资源: dynSM **67584 B** (= nwarps*cols_per_warp*(nbatch_combine+4)*4 = 4*32*132*4, Volta
  cols_per_warp=32), regs 254 -> **1 CTA/SM (4 warps)**; 理论 issue 极限 (mma ~1K cycles/KV chunk,
  LDS ~1-4K) 与实测 ~20.7K cycles/KV chunk 差 5-10x -> 内核是**延迟/停顿受限**, 不是吞吐受限。
  这解释了 24-32% MFU, 也解释了为什么 ub2048 只快 1.35x (同内核, 纯波次效应)。

实现 (env 化, 1 个 DLL 可测全部):
1. `GGML_CUDA_FATTN_BLOCKS=N` / `GGML_CUDA_FATTN_PB=N`: 覆盖 stream-K 的 blocks_num.x
   (`PB=1` -> grid=192 = T11-off 对照; `PB=2/4` -> 384/768 = 每 tile 的 KV 一分为 PB)
2. Volta (256,256,64) 配置变体 (直接改表, 每次重建一个 DLL):
   - cfgA: nbatch_combine 128->64 + Q_in_reg true -> smem 34816 B -> **2 CTA/SM** (regs 255, stack 480 = 有 spill)
   - cfgB: nbatch_combine 128->64 (对照, smem 53248 -> 仍 1 CTA/SM)
   - cfgC: cfgA + nbatch_fa 32->64 (smem 36352 -> 2 CTA/SM, stack 624)
   (cfgB 与 cfgA 之差 = 占用 1->2 的净效果)

后续: 轻量 nsys 扫测 (depth8k, 9 配置, ~3 min GPU) -> 取最优 -> 128k 深度验收 (需用户许可时段)。

### T16 轻量扫测结果 (2026-09-23, depth8k: fill 8192 + eval 4096, 512 FA launch, nsys `--cuda-graph-trace=node`)

| 配置 (blocks_num.x) | FA 总时间 | us/launch | vs blocks192 | fixup |
|---|---:|---:|---:|---:|
| blocks192 (= T11-off 对照) | 1216.2 ms | 2375 | - | 0 |
| blocks96 | 1604.3 ms | 3133 | +31.9% (更差) | 0 |
| blocks160 | 1128.8 ms | 2205 | **-7.2%** | 23.0 ms |
| base-grid80 (T12 默认) | 1112.2 ms | 2172 | **-8.5%** | 12.1 ms |
| **pb2-grid384** | **1037.9 ms** | **2027** | **-14.7%** | 34.2 ms |

结论 (数据否决了我事前的"PB 无益"推断, 支持 analyst 的原始假设方向):
1. **PB=2 (每 tile 的 KV 切 2 段, grid = 2*ntiles_dst) 最优**: 比 T12 默认 (grid=80) 再快 6.7%,
   比 T11-off 快 14.7%; 换算 attention TF/s ~ 29.5*1.17 = **~34.5 (depth32k 口径, 门槛 36 仍差一步)**
2. **细粒度更优**: 192 -> 384 是收益方向; grid=96 (每 CTA 串行 2 整 tile) 是灾难 (+32%),
   验证 "每 CTA 串行多个整 tile" 的代价 (块数 < SM 数时更明显)
3. grid=80 的 2.4 tile/CTA 相比理想 2.4*t 有 ~14% 开销 (2172 vs 1900us 外推); 切半后仅 ~4%
   (PB=2: 4.8 波 x (t/2 + ~9us/CTA 固定开销)); 机制 = **更短的串行段 + 更细的尾部填充**
4. fixup 代价: PB=2 = 34.2ms/512 launch = FA 的 3.3% (已含在上面净收益内); PB=4 预计 ~2x
5. 未跑 (用户因夜间噪音中止): PB=4 (grid=768)、占用变体 cfgA/B/C

### T16 完整轻量扫测 (2026-09-23 暂停点, 全部为 depth8k: `-p 4096 -n 0 -d 8192 -r 1`, nsys, 512 FA launch)

| 配置 | 说明 | FA 总时间 | us/launch | vs T11-off(192) | fixup |
|---|---|---:|---:|---:|---:|
| blocks96 | 2 整 tile/CTA | 1604.3 ms | 3133 | +31.9% | 0 |
| blocks192 | 1 tile/CTA (T11-off 对照) | 1216.2 ms | 2375 | - | 0 |
| blocks160 | 1.2 tile/CTA | 1128.8 ms | 2205 | -7.2% | 23.0 |
| base-grid80 | T12 默认 (min(max_blocks=80)) | 1112.2 ms | 2172 | -8.5% | 12.1 |
| cfgB-combine64 | combine=64 对照 (1 CTA/SM) | 1115.4 ms | 2179 | -8.3% | 12.1 |
| pb4-grid768 | 每 tile KV 切 4 | 1073.5 ms | 2097 | -11.7% | 58.8 |
| **pb2-grid384** | **每 tile KV 切 2** | **1037.9 ms** | **2027** | **-14.7%** | 34.2 |
| cfgA-Qreg | combine=64 + Q_in_reg (spill 480B) | 2126.7 ms | 4154 | +74.9% | 23.1 |
| cfgE-w8 | 256 线程/Q_in_reg/combine16 (spill 1040B) | 8967.0 ms | 17514 | +637% | 22.9 |
| cfgD-w16 | 512 线程 | **编译失败** (96 errors, ncols1_16-ncols2_4 实例) | - | - | - |

结论 (T16 定论):
1. **PB=2 是唯一赢家**: 比 T12 默认再快 **6.7%**, 比 T11-off 快 **14.7%** (attention ~34.5 TF/s 折算);
   fixup 代价 3.3% (已净算). 机制 = 更短的串行段 + 更细的尾部填充 (grid=96 的 +32% 反证了 "串行多整 tile" 的代价)
2. PB=4 过切 (fixup 5.5%, 不如 PB=2); 细粒度收益在 PB=2 饱和
3. **占用/更多 warp 路线全部否证**: Q_in_reg 必然 spill (配置行同时服务 ncols=64 的所有 (ncols1,ncols2) 实例,
   改 nthreads/Q_in_reg 会破坏该族) -> cfgA +75%, cfgE +637%, cfgD 编译失败; cfgB 证明 combine=64 本身中性
4. FA 仍是延迟受限 (4 warps/CTA, 1 CTA/SM); 现有 mma 内核结构下无法安全提占用 -> 不作为

代码落地 (工作区, 已构建但未验收):
- `fattn-common.cuh`: T12 (stream-K 启发式 + env) + T16 (`blocks_num.x = max(nblocks_stream_k,
  min(ntiles_KV*ntiles_dst, 2*ntiles_dst))` = PB=2, 且用 max 保底不减少块数 -> 保护 decode 的 KV-split) + env 覆盖
- `fattn-mma-f16.cuh`: 已 `git checkout` 回 HEAD (实验配置全部回退, 0 行改动)

风险注记 (重要): 中止构建会删除 `build\bin\Release\ggml-cuda.dll` (ninja 先删后链);
本次会话曾观察到部署路径 `D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll` 被自动更新为新构建 (疑似硬链接),
已恢复为交付版 BASE (SHA 102BF84488F2FD43)。**恢复后必须每次构建/部署后核对部署 SHA**。

## 2026-09-23 暂停点状态 (恢复指引)

| 项 | 值 |
|---|---|
| 部署 DLL | **BASE 交付版** `102BF84488F2FD43` (已恢复, 用户可直接用) |
| 构建输出 | `build\bin\Release\ggml-cuda.dll` **不存在** (被中止的 ninja 删除) -> 下次直接 `cmake --build` 重建 |
| 工作区 | 5 文件: 交付 4 文件 + `fattn-common.cuh` (T12+T16); `fattn-mma-f16.cuh` 干净 |
| 待测 DLL | `%TEMP%\v100\ggml-cuda-T16-final.dll` **未生成** (重建即可); 其余实验 DLL 9 个已归档 |
| nsys 报告 | `%TEMP%\v100\t16_*.nsys-rep` 9 个 (base-grid80/blocks96/160/192/pb2/pb4/cfgA/cfgB/cfgE) |
| 下一步 | ① 重建 -> ② 短上下文 PB=2 检查 (`-p 512,4096,8192 -n 128` + ub2048) -> ③ 完整验收 A/B (depth32k/pp32768/**pp8192@depth128k**/短点/tg128) -> ④ PPL+生成 -> ⑤ 入库+patch+文档 |

---

# T12+T16 合并验收 (2026-09-23, implementer): **全部门槛 PASS, 已采纳入库**

构建: `453E29111E5E29C9...` (T12 stream-K 启发式 + T16 PB=2); 工作区 5 文件; patch `patches/v100-t12t16-fattn-split.patch`
A/B: A = BASE 交付版 (`102BF844`), B = T12+T16; 同 session, 长点按轮次**轮换 A/B 顺序** (消除热漂移系统性偏差)

| 门槛 | 要求 | 实测 (A -> B) | 判定 |
|---|---|---:|---|
| **pp8192@depth128k (主判据)** | >= +4% | 337.39 -> **375.45** = **+11.28%** | **PASS** (超 2.8x) |
| depth32k (`-p 4096 -d 32768`) | >= +1.5% | 651.10 -> 682.16 = **+4.77%** | **PASS** |
| pp32768 | >= +0.8% | 774.19 -> 791.07 = **+2.18%** | **PASS** |
| pp512 | |Δ| <= 0.3% | 945.5 -> 950.8 = +0.56% (B 更快) | PASS |
| pp4096 | |Δ| <= 0.3% | 932.2 -> 932.5 = +0.03% | PASS |
| pp8192 | |Δ| <= 0.3% | 909.2 -> 912.1 = +0.32% (B 更快) | PASS |
| tg128 | 噪声内 | 26.667 -> 26.682 = +0.06% | PASS |
| ub2048 (抽查) | 不回退 | 1151.4 -> 1149.9 = -0.1% | PASS |
| PPL | 4.3572 ± 0.013 | **4.3562** | PASS |
| 200 token 生成 | 无乱码/重复 | 连贯 (Flash Attention 技术总结), 无重复行 | PASS |
| 显存增量 | 记录 | fixup partials = `nblocks*64*130*4B ≈ 12.8MB`/launch (CUDA pool 复用); 128k 实跑无 OOM | PASS |

细节:
- 128k 点: 单轮 (用户确认前两阶段波动 <0.5% 后停测, 省 20+ min 重负载): A 337.39 / B 375.45
- 长点稳定性: depth32k A 650.0-652.8 / B 680.9-682.9; pp32768 A 771.8-777.8 / B 787.8-795.1 -> 轮间 <0.5%
- 短点 3 轮: pp512 A 929.4/953.1/954.2 (r1 低离群), B 954.9/952.0/945.6
- **热漂移**: 本机为笔记本机箱, 连续负载下 tg128 从 26.6 逐步掉到 24.4 (A/B 同步) -> 长点必须轮换顺序; tg128 采用冷却后复核值
- **数值**: PPL 4.3568 (T12 单独) -> 4.3562 (PB=2 后) = -0.0006 漂移, 来源 = KV 切 2 段后 softmax 归并顺序变化 (与 GDN vec4 同类, 良性; |diff| 远小于门槛 0.013); 生成质量检查通过
- 采纳后当前最快: pp512 950.8 / pp4096 932.5 / pp8192 912.1 / pp32768 791.1 / depth32k 682.2 / **pp8192@depth128k 375.5** / tg128 26.68 / PPL 4.3562

---

# T17: Q8_0 权重反量化补测 (2026-09-23, implementer): **记录数字, 关闭 (不投入)**

方法: 新增微基准 `artifacts/t17_q8_bench.cu` (当前上游内核 verbatim vs 向量化候选) +
nsys pp512 (`-p 512 -n 0 -r 2`, 部署版 453E2911)。

## 1. 微基准 (512M 元素, 570MB 读 + 1074MB 写)

| 内核 | 时间 | 带宽 |
|---|---:|---:|
| 当前上游 (`dequantize_block_q8_0_f16`, warp/2048, smem 中转) | 2.143 ms | **767.2 GB/s** |
| 向量化候选 (warp/256, 每 lane 8 连续 int8 -> uint4) | 2.455 ms | 669.8 GB/s (更慢) |

- 正确性: 候选与当前内核**逐位相同** (0 / 536870912 mismatch) -> 数值无障碍, 但**无收益**
- 原因: Q8_0 块 = 34 字节 (2 字节对齐), 错位布局下 "smem 聚合读 + 按 char2 展开" 比直接向量化读更优;
  当前内核已在 767 GB/s = Q6_K/Q5_K 参考线 (707-825) 同档

## 2. nsys pp512 占比 (2 reps, kernel 合计 1551.7 ms)

| kernel | 时间 | 占比 |
|---|---:|---:|
| `dequantize_block_q8_0_f16<(bool)0>` | 18.84 ms (501 次) | **1.2% of kernel time** |
| (对照) dequant q6_K vec / q5_K vec | 182.3 / 83.4 ms | 11.8% / 5.4% |
| Q8_0 在 dequant 内部占比 | 18.84/284.6 = 6.6% | 低于其权重占比 13% (即相对更快) |

- 反推实模型有效带宽: Q8_0 权重 2.86GB 读 + 5.38GB 写 = 8.24GB/ubatch, 单 forward ~9.4ms -> **~877 GB/s**
  (>= 实测可用 825-850) -> **已无空间**
- 上限估算: 即使 Q8_0 dequant 完全免费也只有 +1.2% (单 forward); 现实收益 ~0.2-0.3% -> 低于 +0.4% 门槛

## 3. 判定 (spec 第 3 条)

- Q8_0 路径**不偏慢** (767 GB/s >= 参考线; 实模型折算 877 GB/s), 向量化候选更慢 ->
  **不实施, 记录数字关闭**; 不新增 patch, 工作区/交付态不变 (仍 5 文件 + 4 patch, DLL 453E2911)

---

# T18 Stage 0: FA mma 独立 harness (2026-09-23, implementer): **DONE**

产物: `artifacts/t18_fa_harness.cu` (+exe)。做法: **直接 include `fattn-mma-f16.cuh`** (不是 verbatim 拷贝) ->
Stage 1 修改源文件时 harness 自动跟随, 迭代最快。编译: `build_harness.cmd <exe> <cu> --extended-lambda`
(+ `ggml_abort` stub)。

## 形状与方法

- 真实形状: DKQ=DV=256, ncols1/ncols2=32/2, n_q=512 (ub512), GQA 24/4, head_dim 256, mask [n_kv, n_q] causal
- 两个配置: **grid=192 (1 tile/CTA, 无 fixup, = T11-off 对照)** 与 **grid=384 (PB=2 + uniform fixup = 生产配置)**
- 配置/共享内存/步幅全部从源文件读取 (nthreads=128, nbatch_fa=32, combine=128, Q_in_reg=0, smem=67584 ✓ 与生产一致)
- 正确性: grid384+fixup vs grid192 -> max_abs 2.6e-4 (mean 1.0e-5) ✓

## 时间 (20 iters, 单 launch 均值)

| n_kv | grid=192 | grid=384+fixup | ratio | 生产对照 (grid=192) |
|---:|---:|---:|---:|---|
| 12288 | 6.308 ms | 4.926 ms | 0.781 | 5.29 ms (nsys per-instance, depth8k 末 ubatch) |
| 24576 | 10.704 ms | 8.930 ms | 0.834 | - |
| **35072** | **15.218 ms** | **12.718 ms** | **0.836** | **14.84 ms** (depth32k avg, T11-off) |

- 保真度: l=35072 绝对时间差 +2.5% (时钟/干扰), 相对增益 -16.4% vs 生产 ~-15% ✓ -> **可直接用于 Stage 1 门槛**
- **Stage 1 基线 (生产配置) = 12.718 ms/launch @ l=35072; 门槛 >=1.20x -> 目标 <= 10.60 ms**
- 附带发现: 每 launch 固定开销 ~70us (生产分布拟合), 每 KV token ~0.43us (l>=12k 基本线性)
- 生产 per-launch 分布 (depth8k, 512 launch): l=512 -> 0.29ms ... l=12288 -> 5.29ms (16 层 x 每 l 16 次)

---

# T18 Stage 1: 占用 vs L1 的双瓶颈实测 (2026-09-23, implementer) -> **1.12x (门槛 1.20x 未达)**

基线 (harness, 生产配置): ncols=64 + PB=2/grid=384, l=35072 -> **12.660 ms/launch**

## 尝试矩阵 (全部 harness 实测, 同一形状/时钟)

| 变体 | smem | 占用 | 时间 (l=35072, 最优 grid) | 相对基线 |
|---|---|---:|---:|---:|
| ncols=64, grid=384 (基线) | 67584 | 1 CTA / 4 warps | 12.660 | 1.000 |
| ncols=64, nthreads=256 | - | - | 编译失败 (np=8 不受支持) | - |
| **ncols=32, grid=768 (PB=2)** | **35072** | **2 CTA / 8 warps** | **11.297** | **1.120x** |
| ncols=32, grid=1152 / 1536 | 35072 | 2 CTA | 11.821 / 11.424 | 1.071 / 1.109 |
| ncols=32 + Q_in_reg=true | 34816 | 2 CTA | 45.751 (REG 255 + STACK 472B spill) | 0.277 |
| ncols=16 | - | - | NO_DEVICE_CODE (Volta 内核限制 ncols>=32) | - |

## 随 KV 长度的收益趋势 (同一 harness)

| l (n_kv) | ncols=64 | ncols=32 | speedup |
|---:|---:|---:|---:|
| 12288 | 4.802 | 3.939 | **1.219x** |
| 35072 | 12.660 | 11.297 | **1.120x** |
| 70000 | 25.512 | 23.093 | **1.105x** |
| 100000 | 36.780 | 33.390 | **1.102x** |

-> 收敛于 **~1.10x**; 长文 (128k 点, attention 占比 ~65%) 折算 e2e **~+6.3%**;
depth32k (32%) ~+3.2%; pp32768 (18.5%) ~+1.85% -> **Stage 2 三道门槛 (+10%/+4%/+2%) 都会差一点**

## ncu 诊断链 (关键)

| | ncols=64 (1 CTA) | ncols=32 (2 CTA) |
|---|---|---|
| Occupancy (theoretical=achieved) | 6.25% (4 warps) | **12.5% (8 warps)** |
| Compute (SM) Throughput | 26.5% | 40.0% |
| **L1/TEX Cache Throughput** | 47.6% | **74.5% (新瓶颈)** |
| Mem Busy / DRAM / L2 | - / 2.9% / - | 71.3% / 8.5% / 96% |
| No Eligible / Issued per scheduler | 78.5% / 0.22 | 70.1% / 0.30 |
| ncu "Est. Speedup" (scheduler) | 61.3% | 28.7% |

- 结论: ncols=32 把内核从**延迟受限**推到**L1/shared 数据通路受限** (74.5%, 且 DRAM/L2 远未饱和)
- 下一步本应减少 LDS 操作数流量, 但两条路都被硬件挡死:
  1. **Q_in_reg** (把 Q 操作数常驻寄存器, 消掉每 KV chunk 重读 smem 的主项): ncols=32 下仍需 255 regs + 472B/thread spill
     (寄存器墙 255/thread), 实测 4x 变慢
  2. **降 np** (nwarps*cols_per_warp/ncols; 现在 np=4, 每 KV chunk 有跨 warp 归并): np=1 需要 ncols = nwarps*32
     -> ncols=128 -> Q staging 67.5KB (smem 爆) 或 nwarps=1 (单 warp/CTA 无并行) -> 不可行
- Volta 的 mma.m8n8k4 操作数流量与 tile 布局 (cols_per_warp=32 固定) 决定了这个上限; 突破需要重写 tile/warp 布局
  (T_C_KQ/T_C_VKQ 族) = 多日 + 高风险, 超出 T18 3 天 timebox 的合理范围

## 判定 (按 spec 门槛)

- Stage 1 gate: **1.20x -> 未达 (1.12x)**; 1.5 天 checkpoint (1.10x) 刚过
- Stage 2 预估: 三道门槛都差一点 (+6.3% vs +10%; +3.2% vs +4%; +1.85% vs +2%)
- **建议: T18 归档 (不集成)**; 1.12x 变体保留在 harness/artifacts 供未来参考;
  若 analyst/user 认为"均匀 +1.10~1.22x"值得入库, 剩余工作 = dispatch 覆盖 (强制 Volta D256 走 ncols=32) + 完整模型验收 (~1.5h)
- 副产品 (可复用): harness (`artifacts/t18_fa_harness.cu`, 直接 include 源头文件), ncu 日志 2 份, 5 个变体 exe

---

# T18 结案 (2026-09-23): **生产 A/B 未达门槛 (128k 点 ~0%) -> 不合并, 已回退**

## 生产 A/B (A = T12+T16 `453E2911`; B = T18 ncols=32+dispatch `2D1D56C4`; 轮换顺序)

| 点 | 轮 | A | B | 相对 |
|---|---:|---:|---:|---:|
| short pp512 | 1/2 | 946.94 / 943.65 | 947.60 / 944.10 | +0.07 / +0.05% |
| short pp4096 | 1/2 | 937.66 / 926.30 | 931.04 / 930.12 | -0.71 / +0.41% |
| short pp8192 | 1/2 | 914.02 / 904.28 | 907.93 / 908.99 | -0.67 / +0.52% |
| tg128 | 1/2 | 26.63 / 25.67 | 26.55 / 26.30 | -0.3 / +2.4% (漂移) |
| **depth32k** | 1/2 | 680.03 / 678.53 | **688.94 / 690.04** | **+1.31 / +1.70% (均值 +1.50%)** |
| **pp32768** | 1/2 | 797.32 / 789.56 | **803.47 / 801.92** | **+0.77 / +1.57% (均值 +1.17%)** |
| **pp8192@depth128k** | 1/2 | 374.58 / 372.27 | 371.59 / 376.12 | **-0.80 / +1.03% (均值 +0.11%)** |

- 短点差在 ±0.5% 漂移内 (本机同 session 逐轮 A 自身漂移可达 1%)
- **主判据 128k 点: 均值 +0.1% < 门槛 +2% -> 不合并** (Q15 规则; depth32k 也未达 +2% 的复核线)

## 决定性证据: nsys 逐 launch 隔离长文阶段 (duration > 30ms, l~135k)

| cfg | kernel 变体 | grid | 实例数 | median | mean |
|---|---|---:|---:|---:|---:|
| A | `flash_attn_ext_f16<256,256,32,2>` | 384 | 1755 | 40.52 ms | 40.28 ms |
| B | `flash_attn_ext_f16<256,256,16,2>` | 768 | 1641 | 40.77 ms | 40.38 ms |

- dispatch 覆盖**确实生效** (B 的变体/grid 都对), 但长 l 下 **B 反而慢 0.3-0.6%**
- 128k 点构成 (B, nsys): FA 41.7% / cutlass GEMM 35.2% / q6_K dequant 6.7% / GDN 4.6% / q5_K 3.1% / 其他 ~8%
- 即: ncols=32 的收益在 l<=35k 真实 (~1.05x FA, 由 depth32k/pp32768 的 +1.2~1.5% 反推),
  但 **l>=100k 完全消失** (harness 在 l=100k 测得的 1.10x 在生产不复现)

## 根因 (harness 保真缺口)

- Stage 0 harness 用 fp16 K/V + 独立热循环, 只在 l=35072 与生产对过 (+2.5%); **长 l 未对**
- 生产 `-ctv q8_0`: 内核内无 dequant 代码, V 由独立 `dequantize_block_q8_0_f16` 先转 fp16 (1.2%, 可忽略)
  -> 不是 V 精度问题; 更可能是长 l 下内核转为 K/V 流式受限, 小 tile 预取深度不足抵消了 2 CTA 的优势
- 教训: FA 类 harness 必须在**目标 l** 上做保真对照 (不能只在短 l 对)

## 状态

- 部署 DLL 已回退 `453E2911` (T12+T16, 门槛通过版); T18 构建留档 `%TEMP%/v100/ggml-cuda-T18.dll` (`2D1D56C4`)
- 源码工作区: `fattn.cu` (dispatch 覆盖) + `fattn-mma-f16.cuh` (Volta ncols=32 行) 暂留未撤, 待裁决; 若否决则 `git checkout` 这两文件
- PPL/生成未跑 (门槛未达, 不需要)
- T18 全部产物: harness + 5 变体 exe + ncu 日志 2 + nsys 2 + 逐 launch CSV (artifacts/ 与 %TEMP%/v100)

---

# T18 最终处置 (2026-09-23, 用户 Q16): **撤回归档, 不合并**

- 用户裁决: "1% 就不要了, 还牵扯了注意力改动, 划不来" -> T18 全部改动撤销
- 已执行: `git checkout -- ggml/src/ggml-cuda/fattn.cu ggml/src/ggml-cuda/fattn-mma-f16.cuh`;
  工作区回到 5 文件交付态 (convert.cu/dequantize.cuh/fattn-common.cuh/gated_delta_net.cu/unary.cu);
  部署 DLL 保持 `453E2911` (T12+T16, 门槛通过版, 无需重建)
- 留档: `artifacts/t18-ncols32-REJECTED.patch` (1783 B, ncols=32 配置行 + Volta dispatch 覆盖)
  + harness/变体 exe/ncu 2/nsys 2/逐 launch 数据 (artifacts/ 与 %TEMP%/v100)
- 归档结论: 长文 (l>=100k) FA 对 tile 配置不敏感, 瓶颈是 K/V 流式/预取;
  后续任何 FA harness 必须先在**目标 l** 做生产保真对照

---

# T19 全曲线对照 (终期, 2026-09-23): OURS vs STOCK @ 同一 NEWBASE

## 口径

| 项 | 值 |
|---|---|
| NEWBASE (两边同一 base) | `e6ab7c1a4` (ggml-org master, fetch 时最新) |
| OURS | `D:\LLM\Backend\src\llama.cpp-my` = NEWBASE + `3fc05594a` (sm70 FA 调参) + `afbab1748` (交付) |
| STOCK | `D:\LLM\Backend\src\llama.cpp` = NEWBASE |
| 构建 | 双边全量 Release, `CMAKE_CUDA_ARCHITECTURES=70-real;89-real` |
| 部署 SHA256 | OURS `7F1B9B2403438803...` / STOCK `976E2CABF9EADC7D...` (部署 == 构建输出, 每臂跑前核对) |
| 运行 | 各臂从各自部署目录; `CUDA_VISIBLE_DEVICES=1`; `-ngl 99 -fa on -ctv q8_0 -ub 512`; 同 session |

## 命令

- pp: `llama-bench -m <model> -ngl 99 -fa on -ctv q8_0 -ub 512 -p 512,4096,8192 -n 0 -r 3`
      `... -p 32768 -n 0 -r 2` / `... -p 131072 -n 0 -r 2`
- tg: `... -p 0 -n 128 -d 0,4096,8192 -r 3` / `... -d 32768 -r 2` / `... -d 131072 -r 2`

## 实测 (ub512; 每点 = llama-bench 内部 r 次均值; 用户指示: 跳过 ub2048; tg d32768/131072 已按复测修正/标注)

| 点 | OURS t/s | STOCK t/s | Δ |
|---|---:|---:|---:|
| pp512 | 948.48 | 872.78 | **+8.7%** |
| pp4096 | 929.91 | 847.45 | **+9.7%** |
| pp8192 | 907.54 | 814.28 | **+11.5%** |
| pp32768 | 793.47 | 663.29 | **+19.6%** |
| pp131072 | 531.52 | 392.10 | **+35.6%** |
| tg128 d0 | 26.61 | 26.65 | -0.2% |
| tg128 d4096 | 25.47 | 25.56 | -0.4% |
| tg128 d8192 | 24.05 | 23.75 | +1.3% |
| tg128 d32768 | ~~18.93~~ **22.92** | ~~21.18~~ **22.87** | ~~-10.6%~~ **+0.2%** (复测修正) |
| tg128 d131072 | **10.76** (复测均值) | **11.57** (复测均值) | **-7.0% 存疑** (矩阵 -9.4%, 未定论) |

- prefill: 收益随上下文增长 (pp131072 +35.6%), 与 attention 占比上升一致 (T12+T16 + dequant + GDN 共同作用)
- decode: d<=8192 持平; **d32768 复测持平 (+0.2%)** (矩阵的 -10.6% 系测量状态异常, 已排除);
  **d131072 存疑 (-3~-10%, 未定论)**; 两臂 decode 内核集合相同 (含 `flash_attn_ext_vec`), 详见"T19 遗留项排查"

## 产物

- `artifacts/t19_pp.png` / `t19_tg.png` / `t19_delta.png` (300dpi) + `t19_data.csv` + `t19_plot.py` (可复现)
- 原始逐点输出: `%TEMP%/v100/t19_matrix_ub512.txt`

## 质量差异 (Phase E)

| 项 | OURS | STOCK |
|---|---|---|
| PPL (c512 chunks8 seed42) | **4.3562** | 4.3572 |
| 生成 A (创作型 prompt, seed42 temp0 200tok) | 确定性 (两次运行逐字节一致) | 首个生成 token 即分叉 |
| 生成 B (事实型 prompt, 同上) | 200 token **逐字节一致 (871/871)** | 同左 |

- PPL 差 **-0.0010** (OURS 略低); 已知来源 = GDN vec4 求和顺序 1-ulp 重结合 (交付表已记录)
- 生成 A 的分叉解释: greedy + 首 token 近并列 + 上述 1-ulp 级数值差 -> 轨迹完全分叉 (greedy 混沌, 非降智);
  两个输出均连贯
- **结论: PPL 差 <=0.001 且确定性 prompt 下 200 token 完全一致 -> 无降智证据**

## 交付物 (Phase A/B)

- 整理后的 5 文件已提交: `afbab1748` (注释清理, 零功能改动; 清理后 PPL/pp512 复测一致)
- rebase: `git rebase upstream/master` 干净通过 (无冲突, diff 规模与 rebase 前一致), 本地提交 2 个:
  `3fc05594a` (sm70 FA 调参) + `afbab1748` (交付); **未 push**
- 4 patch 基于 NEWBASE 重生成 + 双向 apply 校验通过 (`D:\LLM\Backend\patches\v100-*.patch`);
  校验时与交付提交的 3 行差 = sm70 配置提交 (非 4 组 patch 范围, 预期)
- 备份分支 `backup-t19-pre-rebase` (9403d528e) 保留在 fork 本地

---

## T19 遗留项排查 (2026-09-23): 长文 decode

### d32768: **排除** (矩阵值是异常值)

- nsys 逐 launch 对比 (两臂, `-p 0 -n 128 -d 32768 -r 1`): decode 段 (最后一个 >5ms kernel 之后)
  - OURS: 262368 kernels, GPU busy **5443.7 ms**, wall 5917.3 ms (gaps 8.0%)
  - STOCK: 262368 kernels, GPU busy **5446.3 ms**, wall 5897.9 ms (gaps 7.7%)
  - 逐内核一致: mul_mat_vec_q 3877.3 vs 3879.0 / **flash_attn_ext_vec 761.78 vs 761.81** / rms_norm 176.5 vs 176.1 /
    quantize_q8_1 134.4 vs 134.7 / GDN 50.13 vs 50.16 ...
  - 唯一差异 = silu 内核: OURS `unary_gated_op_kernel_f32_vec4` 47.70ms vs STOCK `unary_gated_op_kernel` 48.61ms (-1.9%, 设计内)
- 复测 (plain, 交替顺序, r=2): OURS **23.02 / 22.82**, STOCK **22.87 / 22.87** -> **+0.2% (持平)**
- 结论: 矩阵里的 18.93 vs 21.18 (-10.6%) 是当时测量状态异常, 非代码差异

### d131072: **存疑 (小回归或测量噪声, 未定论)**

- 复测 (plain, r=2, 两轮交替): 轮1 OURS 10.94 / STOCK 11.33 (-3.4%); 轮2 (反向) STOCK 11.80 / OURS 10.58 (-10.3%)
  -> 方向两轮一致 (OURS 慢), 幅度受热状态影响大 (绝对值为矩阵值 ±4%)
- nsys 逐 launch 对比失败: STOCK 的 d131072 profile 两次都在 prefill 尾部截断 (trace 缺 decode 段, 无 dropped 警告);
  OURS 侧完整: decode 128 token GPU busy 13.2s (MMVQ 6.07s + flash_attn_ext_vec 5.87s + 其他 1.3s)
- 已知约束: decode 段两臂用的内核集合相同 (MMVQ / flash_attn_ext_vec / rms_norm / GDN / silu);
  本 fork 对 decode 路径的唯一改动 (silu vec4) 在 d32768 实测更快; FA 的 mma/tile 配置与 stream-K 决策不参与 decode (vec 内核)
- 判定建议: 若要定论, 需 4-6 轮交替 A/B (约 30 min) 或在真实生产场景 (MTP, 长文) 下对比; 当前按"疑似小回归 (<=3%)"记档

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

### 8. T20 补充 (2026-09-23): S3 的低代价修法 (已验证)

用户追问"有无代价更低/无代价的修法" -> 找到并实现:

**修法**: `fattn-common.cuh` 非 stream-K 路径中, 对**小批量 (n_q <= 8, 即 decode/verify)**
不再用 `min(占用率, ntiles_KV)` + 波次搜索, 改用**与 batch 形状无关的固定切分块数**:
`parallel_blocks = ceil(blocks_per_wave / (ntiles_z_gqa*K->ne[2]*Q->ne[3]))` (参考 = 单 query tile);
prefill (n_q > 8) 保持原波次搜索不变。

**原理**: 切分块数固定 -> 交错切分的 stride 固定 -> 同一可见 KV 集合的分块/部分和完全相同;
多出来的 masked tile 整块落在空 block 里, 对 online-softmax 贡献严格 0 -> decode 与 verify 逐位一致。
(原实现里 pb 随 padded n_kv 变, 跨 256 边界时多出的 tile 会插进有效 block 之间, 改变累加顺序)

**验证**:
- 工具: `prefill 255` (原失败点, A n_kv=256 vs B n_kv=257->512) **逐位一致** (无需 PB_FORCE);
  `prefill 254` 同样一致
- server (S1+S2+S3', 无 PB_FORCE): 短上下文 900 token **n-max 1/2/3 与无 spec 逐 token 一致**;
  32k (prompt 32041) n-max3 **150 token 一致** (该组数据已跑完)
- 128k 按用户指示未测

**代价 (llama-bench tg128, 同 session 交错 base/lc/lc/base)**:
- d0: base 26.63/26.63 vs lc 26.42/26.43 -> **-0.8%**
- d32768: base 21.62/21.56 vs lc 21.49/21.49 -> **-0.5%**
- 即: 全部代价 = S2 (GDN 布局统一) 的 ~0.5-0.8%; S3' 的固定切分在新规则下 pb 与旧搜索最优值几乎相同
  (长文 ceil(640/24)=27 vs 旧 ~26), 短文虽块数变多但 FA 占比极小 -> 代价 ~0
- 此前 PB_FORCE=1 的 -31% 是因为块数被压到 1 (带宽不饱和); 固定块数在 pb>=2 时吞吐即饱和
  (pb=2..8 实测 23.0-23.2 与基线持平; 128k: pb=2/4/8 = 10.9/11.0/10.7 vs 基线 10.9-12.0, 热漂移主导)

**结论**: 三个源可以在 **~0.5-0.8% 总代价**下全部修掉 (不再需要 -31% 的 PB=1 方案);
QUESTIONS 里的方案 A/B 已合并 -> 建议直接采纳 S1+S2+S3' 全部固化。

### 9. T20 补充: 上游 / T19 / T20 三臂对照 (no spec, greedy, 250 token)

用户要求: 检查 T20 是否比 T19 引入更多与上游的不一致 (d0 + d8192)。四臂:
`stock` (上游 976E2CAB) / `t19` (交付 7F1B9B24) / `t20` (低代价修复版 8093C771, 默认) /
`t20fix` (T20 + S1/S2 env 全开)。命令: server `-c 4096|12288 -ctv q8_0 --seed 42` +
`/completion` (temp 0/top-k 1/seed 42/return_tokens; d8192 加 ignore_eos), 250 token。

| 对比 | d0 (creative prompt) | d8192 (8031 token prompt) |
|---|---|---|
| stock vs **t20** | **全等** | **全等** |
| stock vs t20fix | 全等 | 147 分叉 |
| stock vs t19 | **207 分叉** | 全等 |
| t19 vs t20 | 207 分叉 | 全等 |
| t19 vs t20fix | 207 分叉 | 147 分叉 |
| t20 vs t20fix | 全等 | 147 分叉 |

- **T20 (默认) 没有引入更多不一致**: 两个深度都与上游逐 token 全等; T19 在 d0 有 1 处分叉
- `t20fix` 的 d8192@147 分叉来自 S2 (GDN decode 改 vec4 布局) 的 1-ulp 级数值改动 ->
  只会翻转 <=1e-3 级 gap 的极端近并列; 而 T19 在 207 的翻转对应 0.068 nats 的 gap (说明 T19 的
  数值差更大, 来源应为 sm70 FA 配置 / T12+T16 归并)
- 结论: 所有构建之间的分叉都是"近并列 + ulp 级数值差"的硬币翻转, 数量级相当且都不影响质量门槛
  (PPL 4.3562 vs 4.3572); 修 S2 必然改变 decode 数值 (vec4-for-all 或 scalar-for-all 都会改一边),
  这是让 MTP spec 与 no-spec 逐 token 一致的不可避免代价
- 附带: 早前 d0 四 prompt (T19 vs stock): p1_factual @97 / p2_creative @207 分叉, p3_code / p4_math 全等
