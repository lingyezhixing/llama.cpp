# T05: decode profile 定位与修复

状态: **CLOSED** (profile VERIFIED + 剩余项结案; decode kernel 天花板 ~27.3-27.5)
优先级: P1
预期收益: tg128 26.64 -> 28-29 (硬墙 40.9)

## 背景

- decode 每 token 必须读 21.97GB 权重 = 24.4ms @900GB/s; 当前 26.64 t/s = 37.5ms/token
- 即约 13ms 花在"非权重"部分: MMVQ 自身效率损失 + 48 层 GDN/conv + 16 层 FATTN + norm/rope/silu + launch gap
- 报告未对 decode 做分解, 这是当前最大的信息缺口

## 方法

1. nsys profile `-p 0 -n 128` 一帧, 产出:
   - MMVQ 按量化类型汇总 (Q6_K/Q5_K/Q8_0/IQ4) 及其有效带宽 (bytes/time)
   - GDN/conv/FA/norm/其他 kernel 的各自总时长与次数
   - kernel 之间的 gap 总和 (launch 开销/同步点)
2. 按排名逐项处理, 候选:
   - MMVQ: 已是 Volta 专用调参, 检查 nwarps/block 是否可再调 (目标 >=90% 带宽)
   - GDN/conv: 与 T03 共用产出; decode 场景关注 launch 数与串行度
   - norm/rope/silu 等小 kernel: 融合或合并 launch
   - CUDA graph: 确认 decode 全图被 capture, 没有 graph break
3. 每修一项单独跑分, 记录边际收益

## 验收

- tg128 >= 30 且 PPL 一致 (milestone), >= 33 为 stretch
- 提交一份 decode 时间分解表 (优化前/后对照), 放进 RESULTS.md
- 明确写出剩余差距的去向 (哪些是不可压缩的)

## Analyst 决议 (2026-09-22, profile 完成之后)

- profile 部分 **VERIFIED**, 结论清晰:
  - MMVQ 29.5ms = 权重流量的 83-90% (实际可用带宽 825-850 GB/s), 剩余空间 <= 2-3ms/token
  - nwarps 现行配置已最优 (2 -> -2.6%, 8 -> -9.6%), 记录为"不要重试"
  - 非 MMVQ 小计 5.65ms + 间隙 2.5ms = 8.15ms, 这是唯一可压缩的部分
- 30 t/s 需要 -4.4ms, 乐观合计 -4.1ms -> 判定为"临界值"; 33 t/s 不现实。**纯 kernel 路线的收益到此为止**
- 因此: 剩余修复项 (quantize_q8_1 融入 MMVQ ~0.8ms, 小 kernel 合并 ~1.0ms, 间隙 ~0.5ms) **延后到 T06 之后**,
  仅当 T06 (MTP) 失败或收益不足时再回来做; 目标改为 28.5-30 t/s
- decode 的翻倍路线统一走 T06, 不要再投入 kernel 侧 (收益上限 +8% 左右)


---

## 剩余项结案 (implementer, 2026-09-22): CLOSED (no material gain, evidence-based)

- 方法修正: nsys 必须用 `--cuda-graph-trace=node`, 否则图重放 token 的 kernel 不进 kernel 表 (原 profile 只覆盖图前 1-2 token)
- 稳态分解: MMVQ 29.84ms/86.9% (逐矩阵 685-845 GB/s, 已饱和); quantize 0.84; rms_norm 1.10; elementwise 1.03;
  get_rows 0.47; GDN 0.34; FA 0.32; 合计 34.33ms/2024 kernels + 3.4ms host 间隙
- 实验1: rms_norm block 配置 (1024->256): 无效果 (tg128 26.6-26.7 两侧一致), 已回退
- 实验2: 关融合: tg128 -2.6% (25.93 vs 26.63), pp512 -1.7% -> **现有全部融合只值 0.70ms, 边际 ~0.9-1.0us/kernel**
- 判定: 剩余项上限 1.0-1.1ms (+2.7-3.0%) -> tg128 27.3-27.5; 28.5-30 需核心级 kernel 合并 (多日) 或 MMVQ 再快 10% (饱和)
- 建议: 不再投入; decode 侧天花板 ~27.3-27.5 (kernel); 生产 decode 已由 MTP 承担

## Analyst 结案复核 (2026-09-22): CLOSED 接受

- 方法修正认可: `--cuda-graph-trace=node` 才抓到图重放 token 的 kernel (原 profile 只覆盖图前 token) -> 稳态数字以新分解为准
- 决定性证据 = **关融合对照**: 533 个融合头总共只值 0.70ms (边际 ~0.9-1.0us/kernel) -> 剩余合并空间 <=1.1ms (+2.7-3.0%)
- MMVQ 逐矩阵 685-845 GB/s (lm_head 845 = 94-99% 可用) -> "已饱和" 成立
- 判定接受: decode kernel 天花板 **~27.3-27.5** (原 28-29 下调); 不再投入 decode 侧
- 现状 tg128 26.6 保持交付态; 工作区 0 净改动, PPL 4.3569
