# T16: ub512 长上下文 attention 并行度 (parallel_blocks KV 切分)

状态: **VERIFIED (2026-09-23 采纳入库)**; 见 Result (final)
预期: 若 attention 从 29.5 -> >=36 TF/s: 长文代表点 pp8192@depth128k >= +4%

## 动机 (用户常用上下文 128-150K)

- attention 二次方增长: pp32768 实测占比 18.5%; **pp8192@depth128k 该点占比估算 ~60%**
- ub512 下 attention 仅 29.5 TF/s; ub2048 同 kernel **39.9 TF/s** (-26% 时间被并行度浪费)
- T11 的 stream-K 方案只兑现 -5.6% (每 CTA 串行 ~2.4 个 tile, 单 block 效率下降抵消打尾收益)
- **原始方案中的 parallel_blocks (KV 切分, grid = ntiles x PB) 尚未实测** — 每 CTA 只做单个 tile
  的一段 KV, 无串行多 tile

## 目标

ub512 长上下文 attention 时间向 ub2048 靠拢: TF/s 从 31.3 (T12 后) / 29.5 (基线) -> **>=36 (-20%+)**

## 做法

- stream-K 判定不启用时, 强制 parallel_blocks PB=2/4 (或走既有 `parallel_blocks_test` 循环),
  grid = ntiles_dst x PB + fixup
- 保留 env 开关; 与 T12 二选一或组合, 取实测更优 (收益不叠加计算)

## 门槛 (用户 Q14: 长文验收 = pp8192@depth128k, 不做完整 pp128k; timebox 1 天)

1. **主判据 (长文代表点)**: **pp8192@depth128k >= +4%** (先建同 session 基线; 该点 attention 占比估算 ~60%)
2. 32K 副判据: pp32768 >= +1.5% 或 depth32k >= +2.5% (相对 T12 后基线)
3. attention TF/s (depth32k) 目标 >= 36 (T12 后 31.3; 基线 29.5); 中途 < 34 -> 停, 写 BLOCKED
4. pp512/4096/8192 不回退 > 0.3%; tg128 噪声内; PPL 门槛 + 生成检查
5. 完整 pp128k 不做

## 与 T14 的关系

T14 (ub2048) 用户不采纳; 本项是 ub512 口径下长上下文 attention 的唯一补偿方案。


---

## Result (partial, 2026-09-23)

- spec 修正: mma 内核无 parallel_blocks 路径 (只读 blockIdx.x); 等价实现 = 覆盖 `blocks_num.x`
  (`GGML_CUDA_FATTN_PB/BLOCKS`); fixup 机制支持任意 partial 数 (uniform 路径处理 k*ntiles_dst)
- 轻量扫测 (depth8k, 512 launch, nsys):
  grid192=1216ms / grid96=1604ms(差) / grid160=1129ms / grid80(T12)=1112ms / **grid384(PB=2)=1038ms**
  -> **PB=2 比 T11-off -14.7%, 比 T12 默认 -6.7%**; fixup 3.3%
- 未跑: PB=4, cfgA/B/C 占用变体 (nbatch_combine=64+Q_in_reg -> 2 CTA/SM), 完整验收
- 判定: 超过停损线 (>34 TF/s), 继续


## Result (2026-09-23, 暂停点)

- 8 配置扫测 (depth8k, 512 launch): **PB=2 (grid=384) 最优 = 1037.9ms, -6.7% vs T12 默认 / -14.7% vs T11-off**;
  PB=4 = 1073.5ms (过切); grid96 = 1604.3 (+32%); 占用变体 cfgA/cfgE = +75%/+637% (Q_in_reg spill), cfgD 编译失败
- 代码: `blocks_num.x = max(nblocks_stream_k, min(ntiles_KV*ntiles_dst, 2*ntiles_dst))` (PB=2 + 保底); env 覆盖保留
- 未做: 重建 + 短上下文检查 + 完整验收 (depth32k/pp32768/pp8192@depth128k/PPL/生成) + 入库
- 判定: 超过停损线 (attention >= 34 TF/s 估值 ~34.5), 建议按 PB=2 进入验收


## Result (final, 2026-09-23): **PASS -> 采纳 (PB=2), 与 T12 同一 patch 入库**

- 实现: mma 内核无 parallel_blocks 路径 (纯 stream-K, 只读 blockIdx.x) -> 等价实现 = 覆盖 `blocks_num.x`;
  `blocks_num.x = max(nblocks_stream_k, min(ntiles_KV*ntiles_dst, 2*ntiles_dst))` (PB=2 + 保底, 保护 decode)
- 扫测定论: PB=2 最优 (attention -14.7% vs T11-off / -6.7% vs T12 默认); PB=4 过切; 占用/更多 warp 全否证
- 验收 (合并 A/B): **pp8192@depth128k +11.28%** (主判据 >=+4%); depth32k +4.77% (副判据 >=+2.5%);
  pp32768 +2.18%; 短点无回退; PPL 4.3562 + 生成 OK
- 换算 attention: ~34.5 -> ~38 TF/s 级 (T12+T16 组合 vs BASE), 接近 analyst 的 V100 天花板估 (40-45)
