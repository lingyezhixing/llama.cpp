import io

sec = """

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
"""

p = r"D:\LLM\Backend\v100-collab\RESULTS.md"
s = io.open(p, encoding="utf-8", newline="").read()
s = s.rstrip() + sec
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok", len(s))
