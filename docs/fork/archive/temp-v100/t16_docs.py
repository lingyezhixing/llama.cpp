import io

t = """
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
"""

io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

q = """
---

## 2026-09-23 | T12 重放 + T16 spec 修正 | implementer

**T12**: patch 已重放, PPL 4.3568 PASS, nsys 机制复现 (grid 192->80, fixup 1280x23.6us=30.2ms 与 T11 一致)。
完整 A/B (含 pp8192@depth128k 基线) 因夜间静音推迟; 工作区已含该 patch。

**T16 需要你确认的 spec 修正** (读码 + 资源实测, 详见 RESULTS/T16):

1. spec 的 "parallel_blocks PB, grid = ntiles_dst x PB" 在 **mma 内核上不存在那条路径**: mma 内核只读
   blockIdx.x (纯 stream-K 分解), `parallel_blocks` 是 vec/tile 路径的东西。我按等价语义实现为
   `GGML_CUDA_FATTN_PB/BLOCKS` 覆盖 `blocks_num.x` (PB=1 -> 192 = 对照; PB=2 -> 384; PB=4 -> 768),
   fixup 合并机制已确认支持任意 partial 数 (uniform/general 两个 kernel 都能处理 >2 段)。
2. 更重要的发现: 该内核 **smem 67584 B 全部来自 VKQ combine 缓冲** (4 warps x 32 cols x 132 half2),
   `occupancy=2` 本来是设计目标但被 smem 压到 **1 CTA/SM (4 warps)**; 与 issue 极限比慢 5-10x ->
   **FA 是延迟受限, 不是吞吐受限** (这解释了 24-32% MFU; 也说明 ub2048 的 39.9 TF/s 与 ub512 的 29.5
   之差主要是波次, 不是占用)。因此 T16 多了一条独立杠杆: **把占用提到 2 CTA/SM**:
   `nbatch_combine=64 + Q_in_reg=true` (smem 34816 -> 2 CTA/SM, 代价: regs 255 + spill ~480B/thread);
   另备 ncols=32 的免 spill 变体 (需改 dispatch, 待前者结果决定是否做)。
3. 若 PB 扫测 (grid 80/96/160/192/384/768) 无赢家, 请确认是否按 spec 的停损规则 (attention < 34 TF/s 即停)
   收敛为 "PB 路线关闭, 占用路线为主", 或你想调整顺序。

请求: ① 轻量扫测 (~3 min GPU) 的静音许可由用户给; ② 上条 spec 修正是否认可。
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(q)
print('QUESTIONS appended')
