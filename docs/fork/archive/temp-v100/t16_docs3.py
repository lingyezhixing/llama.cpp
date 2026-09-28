import io

t = """
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

风险注记 (重要): 中止构建会删除 `build\\bin\\Release\\ggml-cuda.dll` (ninja 先删后链);
本次会话曾观察到部署路径 `D:\\LLM\\Backend\\llama.cpp-my\\ggml-cuda.dll` 被自动更新为新构建 (疑似硬链接),
已恢复为交付版 BASE (SHA 102BF84488F2FD43)。**恢复后必须每次构建/部署后核对部署 SHA**。

## 2026-09-23 暂停点状态 (恢复指引)

| 项 | 值 |
|---|---|
| 部署 DLL | **BASE 交付版** `102BF84488F2FD43` (已恢复, 用户可直接用) |
| 构建输出 | `build\\bin\\Release\\ggml-cuda.dll` **不存在** (被中止的 ninja 删除) -> 下次直接 `cmake --build` 重建 |
| 工作区 | 5 文件: 交付 4 文件 + `fattn-common.cuh` (T12+T16); `fattn-mma-f16.cuh` 干净 |
| 待测 DLL | `%TEMP%\\v100\\ggml-cuda-T16-final.dll` **未生成** (重建即可); 其余实验 DLL 9 个已归档 |
| nsys 报告 | `%TEMP%\\v100\\t16_*.nsys-rep` 9 个 (base-grid80/blocks96/160/192/pb2/pb4/cfgA/cfgB/cfgE) |
| 下一步 | ① 重建 -> ② 短上下文 PB=2 检查 (`-p 512,4096,8192 -n 128` + ub2048) -> ③ 完整验收 A/B (depth32k/pp32768/**pp8192@depth128k**/短点/tg128) -> ④ PPL+生成 -> ⑤ 入库+patch+文档 |
"""
io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

q = """
---

## 2026-09-23 | T16 定论: PB=2 胜出 (-14.7% vs T11-off), 占用路线否证 | implementer

完整 8 配置扫测数据见 RESULTS。**T16 结论**:

1. **PB=2 (grid = 2*ntiles_dst) 采纳**: 比 T12 默认 (grid=80) 再快 6.7%, 比 T11-off 快 14.7%;
   depth32k 口径 attention ~34.5 TF/s (你的门槛 36 还差一步, 但已超停损线 34)
2. PB=4 过切 (fixup 5.5%), 细粒度在 PB=2 饱和; grid=96 (+32%) 反证串行多整 tile 的代价
3. **占用/更多 warp 路线彻底否证**: Q_in_reg 必 spill (该配置行同时服务 ncols=64 的所有切分实例, 改动会破坏该族);
   cfgA +75% / cfgE +637% / cfgD 编译失败; cfgB 证明 combine=64 本身中性
4. 启发式落地形式 (已写进代码, 待验收):
   `blocks_num.x = max(nblocks_stream_k, min(ntiles_KV*ntiles_dst, 2*ntiles_dst))`
   —— 长 KV 用 PB=2, 但用 max 保底不减少块数 (decode 的 ntiles_dst=12 时保持 grid=80 的 KV-split, 不回退)

问题 (2 个):
- Q1: PB=2 的短上下文 (pp512/4096/8192) 若回退 >0.3%(门槛), 是否改为条件式 (如 `ntiles_KV >= 64` 才 PB=2)?
  我倾向先测, 用数据定条件。
- Q2: 完整验收的 A/B 组合请确认: **A = BASE 交付版, B = T12+T16(PB=2)** 同 session 交替 >=3 轮,
  点 = depth32k / pp32768 / pp8192@depth128k / pp512,4096,8192 / tg128 / ub2048 抽查;
  T12 单独的影响已知 (T11 实测 depth32k +2.0% / pp32768 +1.1%), 不再单独重验 T12。

(工作区/构建/部署状态见 RESULTS 的 "2026-09-23 暂停点状态"; 因用户夜间静音, 全部 GPU 与重 CPU 已停。)
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(q)
print('QUESTIONS appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T16-parallel-blocks.md'
s = io.open(p, encoding='utf-8').read()
s += """

## Result (2026-09-23, 暂停点)

- 8 配置扫测 (depth8k, 512 launch): **PB=2 (grid=384) 最优 = 1037.9ms, -6.7% vs T12 默认 / -14.7% vs T11-off**;
  PB=4 = 1073.5ms (过切); grid96 = 1604.3 (+32%); 占用变体 cfgA/cfgE = +75%/+637% (Q_in_reg spill), cfgD 编译失败
- 代码: `blocks_num.x = max(nblocks_stream_k, min(ntiles_KV*ntiles_dst, 2*ntiles_dst))` (PB=2 + 保底); env 覆盖保留
- 未做: 重建 + 短上下文检查 + 完整验收 (depth32k/pp32768/pp8192@depth128k/PPL/生成) + 入库
- 判定: 超过停损线 (attention >= 34 TF/s 估值 ~34.5), 建议按 PB=2 进入验收
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T16 updated')
