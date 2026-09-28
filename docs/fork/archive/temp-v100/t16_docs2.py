import io

t = """
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
"""
io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

q = """
---

## 2026-09-23 | T16 部分结果: PB=2 胜出 (-14.7% vs T11-off) | implementer

轻量扫测 (depth8k, 512 launch) 数据见 RESULTS:
- **PB=2 (grid=2*ntiles_dst=384) = 2027us/launch, 比 T12 默认 grid=80 快 6.7%, 比 T11-off (grid=192) 快 14.7%**
- grid=96 (2 整 tile/CTA) 反而 +32% -> 你的 "无串行多 tile" 判断方向正确; 细粒度 (192->384) 是收益来源
- 换算 depth32k 口径 attention ~34.5 TF/s (门槛 36, 还差一步; 但已超你的停损线 34, 建议继续)
- fixup 代价 3.3% (已净算)

剩余 (待用户许可时段, 每次 ~15s):
1. **PB=4 (grid=768)**: 细粒度趋势是否延续 (fixup 代价 ~2x)
2. **占用变体 cfgA/B/C**: smem 67584 B 全来自 VKQ combine 缓冲 -> 1 CTA/SM; `nbatch_combine=64 + Q_in_reg`
   可到 2 CTA/SM (代价 spill); 这是与 PB 独立的正交杠杆
3. 若 PB=4 或 cfgA 有正收益 -> 组合 -> **完整验收** (depth32k/pp32768/pp8192@depth128k/pp512-8192/tg128/ub2048/PPL/生成)

问题: PB=2 若最终采纳, 启发式建议 = stream-K 触发时用 `nblocks = min(ntiles_KV*ntiles_dst, 2*ntiles_dst)`
(即 PB=2), 而非 `min(max_blocks, ...)`; ub2048 (eff 96%) 不触发, 不受影响。请确认。
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(q)
print('QUESTIONS appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T16-parallel-blocks.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## Result (partial, 2026-09-23)

- spec 修正: mma 内核无 parallel_blocks 路径 (只读 blockIdx.x); 等价实现 = 覆盖 `blocks_num.x`
  (`GGML_CUDA_FATTN_PB/BLOCKS`); fixup 机制支持任意 partial 数 (uniform 路径处理 k*ntiles_dst)
- 轻量扫测 (depth8k, 512 launch, nsys):
  grid192=1216ms / grid96=1604ms(差) / grid160=1129ms / grid80(T12)=1112ms / **grid384(PB=2)=1038ms**
  -> **PB=2 比 T11-off -14.7%, 比 T12 默认 -6.7%**; fixup 3.3%
- 未跑: PB=4, cfgA/B/C 占用变体 (nbatch_combine=64+Q_in_reg -> 2 CTA/SM), 完整验收
- 判定: 超过停损线 (>34 TF/s), 继续
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T16 updated')
