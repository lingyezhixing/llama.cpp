import io

# ---------- RESULTS.md: T10 full report ----------
p = r'D:\LLM\Backend\v100-collab\RESULTS.md'
t = """
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
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('RESULTS.md: T10 appended')

# ---------- TASKS/T10: result ----------
p = r'D:\LLM\Backend\v100-collab\TASKS\T10-attention-recon.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace('状态: PROPOSED (analyst 2026-09-22, 已指派)',
              '状态: DONE (implementer 2026-09-22; 只测量, 0 行代码改动) -> 结论: 不值得移植 split-D/N32')
s += """
---

## Result (implementer, 2026-09-22)

已完成 1/2/3/4 全部要求, 完整数据见 RESULTS.md "T10" 章节, 原始 CSV 见 artifacts/t10_*_kern_sum.csv。

速览:
- attention 占比: pp4096 3.3% / pp32768 18.5% / pp4096@depth32k 32.0%
- attention 效率 (有用 causal FLOP): d0-pp4096 23.7 / pp32768 28.7 / depth32k 29.5 TF/s (ub512),
  **ub2048 同场景 39.9 TF/s**
- 变体: `flash_attn_ext_f16<256,256,32,2,0,0,0>` (mma, grid 恒 192x1x1, block 32x4x1), 无 vec/stream-k/sparse
- 深度不衰减 (逐 ubatch 25.9 -> 29.7 TF/s)
- **裁决: 不值得移植 split-D/N32** (ub2048 已达 39.9 TF/s, 超 1Cat 的 29-38 区间)
- 近路 (可选, 需 analyst 批准): 让 KV-split (parallel_blocks, 代码已存在但被 stream_k 分支绕过)
  在 ub512 生效 -> 预期 depth32k +5-8% / pp32768 +3-5% / pp8192 +1.2%, 0.5-1 天
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T10 updated')

# ---------- BOARD: T10 row + queue ----------
p = r'D:\LLM\Backend\v100-collab\BOARD.md'
s = io.open(p, encoding='utf-8').read()
old = "| T10 | 长上下文 attention 侦察 (只测量) | **PROPOSED (analyst 指派)** | nsys 分解 depth32k/pp32768 的 attention 占比/TFLOPS; 决定是否移植 1Cat split-D/N32 |"
new = "| T10 | 长上下文 attention 侦察 (只测量) | **DONE (implementer, 待 analyst 复核)** | attention: d0 3.3% / pp32768 18.5% / depth32k 32.0%; ub512 29.5 TF/s, **ub2048 39.9 TF/s** (grid 192->768) -> 不值得移植 split-D/N32; 近路 = ub512 下启用 KV-split (预期 depth32k +5-8%, 0.5-1 天) |"
assert old in s
s = s.replace(old, new, 1)

old2 = "| 3 | **T03 chunked GDN 重写** | prefill +3-4% | timebox 1.5 天 | APPROVED (中途 checkpoint) |"
new2 = """| 3 | **T03 chunked GDN 重写** | prefill +3-4% | timebox 1.5 天 | APPROVED (中途 checkpoint) |
| 3b | **T10 近路: ub512 下启用 FA KV-split** (parallel_blocks) | depth32k +5-8% / pp32768 +3-5% / pp8192 +1.2% | 0.5-1 天 | PROPOSED (待 analyst 裁: 并入 T03 前或后) |"""
assert old2 in s
s = s.replace(old2, new2, 1)

old3 = "2. **外部参考已归档: REFERENCE-1cat-vllm.md**"
new3 = """2b. **T10 已侦察完**: attention 不是 kernel 慢而是 ub512 并行度不足 (192 CTA / 80 SM / 1 CTA/SM);
   ub2048 同 kernel 39.9 TF/s = 1Cat 上限 -> 不移植 split-D/N32; 唯一近路是让 ub512 的 KV-split 生效
   (预期 depth32k +5-8%, 0.5-1 天)
2. **外部参考已归档: REFERENCE-1cat-vllm.md**"""
assert old3 in s
s = s.replace(old3, new3, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('BOARD.md updated')

# ---------- QUESTIONS: ask analyst about the KV-split follow-up ----------
p = r'D:\LLM\Backend\v100-collab\QUESTIONS.md'
t = """
---

## 2026-09-22 | T10 侦察完成 | implementer

Q6 (裁决请求): T10 结论 = "不值得移植 split-D/N32" (ub2048 同 kernel 已 39.9 TF/s, 超 1Cat 29-38 区间;
ub512 只有 29.5 是 grid=192/80SM/3 波次打尾造成)。
但侦察发现一个 0.5-1 天的近路, 请裁决是否做、何时做:

A. **ub512 下启用 FA KV-split**: `launch_fattn` 已内置 `parallel_blocks` KV 分块 + fixup
   (fattn-common.cuh:1188+), 但 `stream_k=true` 分支在效率 >=75% 时直接 `blocks_num.x = ntiles_dst`
   (1150 行判据), 绕过了它。改动 = 判定不启用 stream-K 时回退到 parallel_blocks 循环 (约 20 行启发式)。
   预期: grid 192 -> 384/768, 波次效率 80% -> 96%, attention -20~26%
   -> **depth32k +5-8% / pp32768 +3-5% / pp8192 +1.2% / pp4096 +0.7%** (ub512 口径, 长上下文才值)
   风险: 数值 = softmax 归并顺序变化 (1 ulp 级重结合, 同 GDN vec4 先例, PPL 门槛能抓住问题)
B. 不做, 直接进 T03 chunked (prefill 全局 +3-4%, 1.5 天) — 我建议的顺序是 A 先做 (小、可测、可放弃), 再做 B。
C. 并入 T04 (ub 提高后 attention 自然受益, 无需动 ub512 口径) — 但 T04 待用户口径。

我的建议: **A (0.5-1 天) -> B (T03 chunked)**, 因为 A 是 T10 唯一可落地的产出, 且能复用 T03 的
PPL/pp512/4096/8192 验收流程; 若你更看重"单一全局项", 也可以 B -> A。
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('QUESTIONS.md: Q6 appended')
