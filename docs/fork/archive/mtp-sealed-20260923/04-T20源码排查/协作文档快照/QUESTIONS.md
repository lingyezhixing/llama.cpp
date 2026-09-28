# QUESTIONS: implementer -> analyst

规则: implementer 在下面追加问题 (带日期和任务号), analyst 在问题下方直接写 `ANSWERED:` 回复。
不要修改已 ANSWERED 的条目。

---

---

## 2026-09-22 | T01 收尾 | implementer

Q1: T01 结论 = 调用侧无收益, 建议直接进 T02 (融合 kernel, 预计 1-2 天+)。
    在 analyst 细化 T02 方案期间, 我按 BOARD "T03/T04 随时可做" 先开 T03 (GDN, 预期 +3-4%) 可以吗?
    (若不可, 我待命等 T02 规格。)

ANSWERED (analyst): 可以, T03 已置 RUNNING。注意两点: ① GDN 的两个观测点分别是 "pp512 kernel 时间 -40%" 和 "整机 +2%", 以整机为准; ② 若中途发现 GDN 的串行结构改不动 (chunk 方案不可行), 不要硬凑, 写 BLOCKED + 原因, 转 T05/T06。
T02 规格已更新 (分阶段 go/no-go), 见 TASKS/T02-dequant-fusion.md。

Q2 (信息, 不需回答): "改数值路径换性能"在 prefill 上不成立: 实测 fp16 累加 (compute=16F) 更慢
    (gate/up 63.7 TF vs 32F 84.2 TF; down 97.4 vs 100-107 TF)。所以 NEEDS-USER Q2 对 T01/T02 无影响。

ANSWERED (analyst): 收到, 这项结论已记入 STATUS.md (fp16 累加在 V100 上更慢, 数值路径不需要动)。数据本身很好, 是个有价值的否证。

Q3 (信息): T01 的 T02 盈亏平衡线 (含 dequant 合计时间为分母): gate/up 56.3 TF, down 65.6, qkv 70.7, lm_head 67.8。
    建议 T02 的验收口径直接写成"整机 pp512 提升 >= X%", 因为融合 kernel 单独看 TFLOPS 会被 dequant 抵消。

ANSWERED (analyst): 同意, T02 验收改为整机 pp512, 盈亏平衡线作为 Stage 1 的 go/no-go 门槛。
---

## 2026-09-22 | T02 Stage1v1 后 + T05 分解后 | implementer

Q4: 请裁决下一步优先级 (我这边时间/算力都空闲, 随时可开工):
    A. **T02 Stage 1b**: 按可行性设计重写融合 kernel (BM=128/BN=64/BK=128 + B 双缓冲 + A 走 global fragment),
       目标 >=84 TF (= 整机 +8-12%)。预计 1-2 天, 风险高 (smem 96KB 约束下 5 个条件互相冲突)。
    B. **elementwise 向量化** (新任务, 未在板上): ub512 下 convert/silu/rms_norm/concat/add/conv 共 54ms (10%),
       实测都只跑 58-70% DRAM 带宽, 预计 +2-3% 整机, 风险低, 工作量半天。
    C. **decode 小 kernel 融合** (T05 的剩余部分): quantize_q8_1(0.86ms)+rms_norm(1.12)+scale(0.64) 融合,
       预计 decode +8-10%, 工作量 1 天, 需要动 mmvq/多 op 的图融合, 中等风险。
    我的建议: 先 B (确定性收益, 快), 再 A (主线, 大收益)。
---

## 2026-09-22 | T08 结案后 | implementer

Q5 (请裁决下一步优先级; 我这边空闲, 随时开工):
    A. **T03 chunked** (GDN 重写, prefill +3-4%): 剩余唯一 >=1% 的 prefill 项, 但 1-2 天 + 3 个高风险点。
    B. **T07 剩余 rms_norm 向量化** (+0.4%, 小时级) + **T05 剩余** (decode +5-8%, 用户 MTP 已覆盖有效吞吐)。
    C. 若认为 ub512 上调参类 (T04 ub2048 +21%) 应该兜底, 我可以直接做配置固化。
    我的建议: 先 B 的 rms_norm (确定性 + 快), 再 A (唯一大项, 1.5 天 timebox + 中途 checkpoint)。

补充事实 (供 analyst 更新 STATUS/ARCHIVE): T01/T02 矛盾已彻底解释, 建议把 T01 结论恢复为
"调用侧无空间" (T08 Step 1 已复核)。另: 真实负载下 SM 时钟 1447MHz (纯 GEMM 1530MHz),
这是 GEMM 侧剩下的主要"损失", 但属硬件/功耗管理, 不可控。T02 的 baseline 也是在转置方向测的,
其 66 TF 盈亏平衡线可修订为 66-70 TF (否决结论不变)。

---

## 2026-09-22 | T08 结案后 | implementer

Q5 (请裁决下一步优先级; 我这边空闲, 随时开工):
    A. **T03 chunked** (GDN 重写, prefill +3-4%): 剩余唯一 >=1% 的 prefill 项, 但 1-2 天 + 3 个高风险点。
    B. **T07 剩余 rms_norm 向量化** (+0.4%, 小时级) + **T05 剩余** (decode +5-8%, 用户 MTP 已覆盖有效吞吐)。
    C. 若认为 ub512 上调参类 (T04 ub2048 +21%) 应该兜底, 我可以直接做配置固化。
    我的建议: 先 B 的 rms_norm (确定性 + 快), 再 A (唯一大项, 1.5 天 timebox + 中途 checkpoint)。

补充事实 (供 analyst 更新 STATUS/ARCHIVE): T01/T02 矛盾已彻底解释, 建议把 T01 结论恢复为
"调用侧无空间" (T08 Step 1 已复核)。另: 真实负载下 SM 时钟 1447MHz (纯 GEMM 1530MHz),
这是 GEMM 侧剩下的主要"损失", 但属硬件/功耗管理, 不可控。T02 的 baseline 也是在转置方向测的,
其 66 TF 盈亏平衡线可修订为 66-70 TF (否决结论不变)。

ANSWERED (analyst, 2026-09-22): 采纳"先快后大", 另插入一项新侦察 (T10):

1. **T10 (新): 长上下文 attention 侦察** - 只测量 (nsys, ~1h), 与 T07 rms_norm 并行做。
   动机 (1Cat-vLLM 外部参考): 他们的 FA-V100 在 D256/GQA6 同形状上达到 prefill 29-38 causal TF/s;
   我们 pp4096@depth32k 比 pp4096 慢 31% (差 1935ms), 粗反推 attention ~14.5 TF/s -> 有 ~2x 嫌疑。
   先量出 attention 的 ms/占比/TFLOPS, 再决定是否值得移植 (潜在 +5-15% 长上下文)。
   规格见 TASKS/T10-attention-recon.md。
2. **T07 剩余 rms_norm**: 批准 (+0.4%, 小时级), 与 T10 并行。
3. **T03 chunked**: 批准, timebox 1.5 天, 中途 checkpoint (decay / 三角求解 / keep_rs_t 快照
   任一点卡住就停下写 BLOCKED, 不硬凑)。
4. **T05 剩余**: 排 T03 之后, 可选 (decode +5-8%; 对 MTP 轮也有加速, 但用户有效吞吐主要由 MTP 承担)。
5. **T04 ub**: 维持"用户口径决定", 补充外部证据: 1Cat 整个架构 = 大 M chunked prefill,
   dequant 占比 2% vs 我们 16.6% -> 若允许, 这是全局最大单项 (+21% 实测)。

你补充的三条事实 (T01 结论恢复 / 1447MHz 时钟 / 盈亏平衡线 66-70) 已全部并入 STATUS/ARCHIVE。
T08/T09 的结案复核已写入 TASKS/T08 与 ARCHIVE 第 9 节。

---

## 2026-09-22 | T10 侦察完成 | implementer

Q6 (裁决请求): T10 结论 = "不值得移植 split-D/N32" (ub2048 同 kernel 已 39.9 TF/s, 超 1Cat 29-38 区间;
ub512 只有 29.5 是 grid=192/80SM/3 波次打尾造成)。但侦察发现一个 0.5-1 天的近路, 请裁决是否做、何时做:

A. **ub512 下启用 FA KV-split**: `launch_fattn` 已内置 `parallel_blocks` KV 分块 + fixup
   (fattn-common.cuh:1188+), 但 `stream_k=true` 分支在波次效率 >=75% 时直接 `blocks_num.x = ntiles_dst`
   (1150 行判据), 绕过了它。改动 = 判定不启用 stream-K 时回退到 parallel_blocks 循环 (约 20 行启发式)。
   预期: grid 192 -> 384/768, 波次效率 80% -> 96%, attention -20~26%
   -> **depth32k +5-8% / pp32768 +3-5% / pp8192 +1.2% / pp4096 +0.7%** (ub512 口径, 长上下文才值)
   风险: 数值 = softmax 归并顺序变化 (1 ulp 级重结合, 同 GDN vec4 先例, PPL 门槛能抓住问题)
B. 不做, 直接进 T03 chunked (prefill 全局 +3-4%, 1.5 天) — 我建议的顺序是 A 先做 (小、可测、可放弃), 再做 B。
C. 并入 T04 (ub 提高后 attention 自然受益) — 但 T04 待用户口径。

我的建议: **A (0.5-1 天) -> B (T03 chunked)**, 因为 A 是 T10 唯一可落地的产出, 且能复用 T03 的
PPL/pp512/4096/8192 验收流程; 若你更看重"单一全局项", 也可以 B -> A。

ANSWERED (analyst, 2026-09-22): **批准 A 先做, 然后 B**。理由: A 小、可测、可放弃, 收益落在我们最弱的两项
(depth32k 649 / pp32768 775); B 是最大全局项但风险高, 排 A 之后。

A 的验收门槛 (严格执行, 不满足即回退):
1. 主判据 = **A/B 交替换 DLL >=2 轮**: depth32k 提升 >= 3% 且 pp32768 >= 2% 才算过; 单次测量不算
2. pp512/4096/8192 不得回退 > 0.5%; **tg128 不得回退** (若 KV-split 影响 n_q=1 路径立即回退)
3. PPL 门槛 |x-4.3572| <= 0.013 (预期 1 ulp 级, 同 GDN vec4 先例); 生成质量检查照做
4. 必须给 **nsys attention TF/s 前后对比** (验证波次效率 80% -> 96% 的机制成立), 不能只报端到端数字
5. ub2048 口径抽查一次不得回退 (KV-split 理论上只在低波次效率触发)
6. 改动最小化 (启发式 ~20 行), env 开关便于即时回退, patch 归档

若 A 失败: 写 BLOCKED + 数据, 不硬凑, 直接转 B (T03 chunked)。

---

## 2026-09-22 | T10 近路 (KV-split) 验收失败 | implementer

执行 Q6 的 A 并按 6 条门槛严格验收 -> **FAIL, 已回退, 按规则转 T03 chunked**。
完整数据见 RESULTS.md "T10 近路" 章节 (nsys 前后对比 + 2 轮交替 A/B 表)。

速览: nsys 机制成立 (grid 192->80, fixup kernel 出现, attention 1899.4->1792.7ms = -5.6%, 29.5->31.3 TF/s),
但端到端 depth32k **+2.0%** (门槛 3%) / pp32768 **+1.1%** (门槛 2%) -> 不达; 其它指标无回退, PPL 4.3568 安全。
根因: 波次效率模型高估 (每 block 2.4 tile 串行 + tile 接缝归并抵消); ub2048 的优势是并行度总量 + 更少 launch。

已开始 T03 chunked (timebox 1.5 天, 中途 checkpoint)。若你有 T03 的额外约束请写入 TASKS/T03。

---

## 2026-09-22 | T03 chunked 0.75 天 checkpoint | implementer

按你的要求给 0.75 天 checkpoint (数据见 RESULTS.md "T03 checkpoint" 与 TASKS/T03):

- **三个 checkpoint 全部通过 (数学层, 机器精度 1e-16)**: decay 递推 / 三角求解 (WY 恒等式 4.4e-16) /
  多 chunk 状态交接; 覆盖 L=1..64, D=8..128, T=256; 参考实现归档 `artifacts/t03_chunked_reference.py`
- 算法与实现要点已写清 (同一 (I+M) 三次前代即可, 不需显式求逆); 设计 = K1(chunk 并行)+K2(chunk 串行)
- 成本估算: 每 chunk 约 4.6M MAC vs 串行 3.1M MAC, 但现状 kernel 仅 ~0.5 TFLOPS (指令吞吐受限),
  目标 2 TFLOPS 级即可达 -50% -> 预期可行
- **CUDA 实现未开始** (本轮剩余时间不够实现+验证, 按"不硬凑"约定停在这里)
- 问题: 是继续实现 K1/K2 (需再一个较长 session), 还是先把 T03 的 timebox 顺延/换优先级?
  你若有对 chunk 长度 L (当前建议 64) 或 scratch 预算的约束, 请直接写进 TASKS/T03。

---

## 2026-09-22 | T03 CUDA 原型 V1 | implementer

补充 checkpoint 报告 (接上一条): CUDA 原型已能跑通, **正确性达标, 性能不达标**。

- 正确性: T=512 (16 个 chunk) 下 out/state 相对误差 5e-7 / 1.7e-7 (纯 fp32) -> 三个 checkpoint 在真实
  CUDA 代码上通过; 过程中修掉两个真 bug (C 掩码 / U 右端项多乘 beta, 后者会导致状态 32% 偏差)
- 性能: V1 = 4.60 ms per layer-ubatch vs 现状真 kernel 0.826 ms -> **慢 5.6x, 未集成**
- 瓶颈已定位 (bank conflict / occupancy / 低并行阶段 / sync 数), 修炼点明确
- 产物: `artifacts/t03_chunked_v1.cu`, `t03_check_state.py`; 工作区 0 行改动

问题: 接下来 (A) 继续性能重写 (~0.5-1 天, 目标先追平 0.826ms 再谈 -50%), 还是 (B) 先暂停 T03,
把这轮结论当成"算法可行、工程成本高于预期"记档, 转 T05 剩余/T04? 我倾向 A (瓶颈已定位, 值得一搏),
但如果你认为应该控制投入, 我就把 V1 归档后停。

ANSWERED (analyst, 2026-09-22): **批准 A (继续性能重写), 但设中间门槛。** 理由: 数学层已机器精度验证,
两个真 bug 已修, 瓶颈是经典 V1 问题 (bank conflict / occupancy / 低并行阶段), 不是 T02 式的架构墙;
且 T03 是 ub512 最后一个 >=1% 项。0.75 天 checkpoint 的提问 (是否继续 K1/K2) 一并按此答复。

门槛 (严格执行):
1. **集成前硬门槛**: V2 在 harness 达到 **<=0.60ms/layer-ubatch** (现状 0.826 的 -27%) 才准集成;
   若只到 0.60-0.83 (打平), **不集成** (代码风险换不来收益); 目标仍是 <=0.41 (=-50%)
2. **正确性广度 (V2 必须同时覆盖)**: L=16/32/64 对比; T 不整除 L 的尾块; 真实模型形状 (D=128 + 实际 head 数);
   多序列 batch 行为与现 kernel 等价 (否则从启用条件排除并回退); 现有 enable 条件
   (`n_tokens>1 && !keep_rs_t && !KDA && S_v==128`) 保持, 任何未覆盖情形回退标量路径
3. **资源约束**: smem 压到 <=48KB (目标 >=2 block/SM); 不得降低状态精度 (保持 fp32)
4. **集成后验收照旧**: 同 session A/B >=2 轮; pp512/4096/8192 + pp32768/depth32k + tg128 (decode 路径不动);
   PPL 门槛 `|x-4.3572| <= 0.013` + 生成质量检查
5. **timebox: 再给 1 天硬上限**; +0.5 天给进展报告 (即使不达标); 到点不达标 -> BLOCKED 归档, 转 T05/T04

若 V2 半路发现结构性瓶颈 (类似 T02), 立即停并写数字, 不硬凑。

---

## 2026-09-22 | T03 V2 进展 (0.5 天点) | implementer

按 gate 要求交 0.5 天点进展报告 (完整数据见 RESULTS "T03 V2 进展报告"):

- 迭代: V1 4.60ms -> V2a 3.29ms (pad 消 32-way bank conflict + C 阶段全并行) -> **V2b 1.644ms (L=16)**
- **资源门槛全过**: smem 43.0KB (<=48 ✓), 2 block/SM (16 warps ✓), 正确性 5.7e-7/1.7e-7 ✓ (T=512, 32 chunks)
- **速度门槛未达**: 1.644ms > 0.60ms (目标 0.41ms); 相对现状 kernel 0.826ms 仍慢 2.0x -> 按 gate #1 不集成
- 根因 (实测): L=16 每 chunk 约 276K MAC (1080 FMA/线程) 但实测 ~71K cycles/chunk -> ~66 cycles/FMA;
  阶段式 + 每 chunk 8 次块级 sync 导致延迟暴露, 25% occupancy 掩盖不住 -> 结构性问题, 非单点 bug
- 修炼方案 (需要一次结构性重写, ~0.3-0.5 天): 寄存器分块合并 KKT/QK/T1/A 为一个 2D 扫描;
  分块三角求解 (对角块串行 + 块间 matmul, W/U/R 三系统同做); 或改成 K1+K2 双 kernel 用 scratch 换并行度

问题: timebox 已用 ~0.9 天 (硬上限 1 天)。(A) 我继续做结构性重写 (再 ~0.3-0.5 天, 有把握过 0.60ms,
但会略超 1 天硬上限), 还是 (B) 就此 BLOCKED 归档 (算法已验证可行, 工程成本超预期), 转 T05 剩余/T04?
我倾向 A 的**最小版**: 只做"寄存器分块合并 KKT/QK/T1/A + 分块三角求解"两项 (~0.3 天), 若 0.5 天内仍不过
门槛就无条件归档。若你坚持硬上限, 我立刻转 B。

ANSWERED (analyst, 2026-09-22): 选 **B = BLOCKED 归档**。理由: 规则 (第 5 条) 就是"1 天硬上限到点不达标即归档";
现在已用 ~0.9 天, 速度门槛还差 2.7x (1.644 vs 0.60ms), 且本轮从 V1 到 V2b 的迭代已经用掉了两次"乐观估计被现实修正"
的机会 (T02/V1 都是先例)。按"不硬凑"原则停止, 不是否定你的方案, 而是控制投入。

这轮数据很有价值, 已记档:
- 算法层机器精度验证 (1e-16) + V1/V2a/V2b 正确性全过; 速度迭代 4.60 -> 3.29 -> 1.644ms (2.8x)
- 资源门槛全过 (smem 43KB, 2 blk/SM); 卡点 = 结构性延迟暴露 (~66 cycles/FMA, 每 chunk 8 次块级 sync)
- 修炼方案 (寄存器分块 2D 扫描 + 分块三角求解 / 或 K1+K2 双 kernel) 已写明, 恢复成本 ~0.3-0.5 天

**恢复条件 (留给最终统一回顾, 用户 Q7)**: 若回顾时判定值得, 按"最小版重写 + 0.5 天无条件归档"批准;
在那之前 T03 冻结, 不再投入。工作区保持 0 行改动, artifacts 保留完整 (v1/v2_L16/参考脚本/对拍脚本)。

---

## 2026-09-22 | Analyst 复议 (用户裁决): T03 恢复 | analyst

用户裁决 (覆盖上一条 BLOCKED): **T03 再给 0.5 天, 看有没有提升空间** -> 撤销归档, 状态改回 RUNNING。

- 批准: 最小版结构性重写 (寄存器分块合并 KKT/QK/T1/A 为一个 2D 扫描 + 分块三角求解);
  允许备选 K1+K2 双 kernel + 小 scratch (per-layer MB 量级, 走既有 workspace 机制)
- 硬条件:
  1. **timebox = 0.5 天, 到点无条件归档** (不再延期, 除非用户再次明确)
  2. 集成门槛不变: <=0.60ms/layer-ubatch; 0.60-0.83 (打平) 不集成
  3. 正确性/资源门槛不变 (L=16/32/64 对比已过可复用; 尾块/真实形状/多序列等价或回退;
     smem <=48KB 或双 kernel scratch <= MB 级; fp32 状态)
  4. 过程要求: 每个迭代给 cycles/FMA 与 ms; 若 ~0.25 天点 cycles/FMA 无 ~2x 改善 -> 提前停, 写数字
- 过门槛后按 TASKS/T03 集成验收流程 (同 session A/B >=2 轮 + pp512/4096/8192/32768/depth32k + tg128 + PPL)

---

## 2026-09-22 | T03 V3 提前停 (按规则 4) | implementer

按你的过程规则 4 执行: V3 (barrier 10 -> 4, OS 内联) **cycles/FMA 没有 2x 改善** (66 -> 80, 更差),
故提前停, 写数字 (完整见 RESULTS "T03 V3 结果"):

- V3 = 1.844ms/layer-ubatch, ~80 cycles/FMA, smem 41KB / 2 blk/SM (资源门槛仍过)
- **barrier 假设被否证** (同步减半以上, 时间反而 +12%)
- 真瓶颈 (诊断清楚): 四个点积阶段 (KKT/QK/T1/A) 是"每线程做长度 128 的串行点积", 循环上界运行时值
  导致不展开, 每次迭代的 L2/smem 延迟直接暴露 -> 算力利用率 ~1.3%; 现状 kernel 的 warp 协作+归约
  正是治这个的
- 修炼点明确 (warp 协作 + float4, 预计 ~6x -> 0.32-0.48ms, 应能过 0.60 门槛), 但需要一次重写,
  超出本轮 0.5 天可用余量

请裁: (A) 再批一次重写 (我估计 0.3-0.4 天, 只做"四个点积循环 warp 化 + 前代展开", 到点无条件归档),
或 (B) 维持你的"不再延期"立场 -> T03 现在无条件归档, 我转 T05 剩余/T04。
说明: 我倾向 A 的收益/风险比这次比前两轮好 (瓶颈已实测定位, 修法明确), 但完全尊重你控制投入的判断。

---

## 2026-09-22 | T03 结案: NO PATH FOUND | implementer

按用户"无限时间、确保没有误判"的指示, 我把每条路线都在**真实规模 (H=48)** 上实测到底 (完整数据见
RESULTS "T03 结案")。结论: **GDN kernel -50% 不可达, 建议 T03 结案**。

关键实测 (现状 0.814 ms/layer-ubatch; harness 复刻 0.922):
- 寄存器预取 / smem 双缓冲 staging: 都更慢 -> **不是访存延迟瓶颈** (40 warp/SM 已掩盖)
- C=2 列每 warp: harness 好 8%, 但**真实模型集成后 867us vs 814us = +6%** (warp 总数减半抵消) -> 已回退
- MB=12 提占用到 48 warp/SM: 1.01ms (寄存器 spill) -> **不是占用瓶颈**
- expf 消融: expf 总价值 14%; `__expf` 实测 -10% (kernel), 但按 gate 0.60-0.83 不入库
- warp 归约微基准: 延迟 144.5 cycles 但 ILP 完全可掩盖 -> 机制成立, 被 warp 总数抵消
- chunked 路线: 算法对 (1e-16/5e-7) 但 FLOP 多 1.4-2.4x, 要过门槛需效率 30x 提升 -> 不可行

建议: ① 关闭 T03 (状态 CLOSED, NO PATH) ② `__expf` 作为"低于门槛的已知选项"记档
③ 队列只剩 T05 剩余 (decode +5-8%, 分析已做) 与 T04 (ub +21%, 待用户口径)。
工作区已回退到 4 文件交付态, PPL 4.3569 ✓。

ANSWERED (analyst, 2026-09-22): **接受 T03 结案 (CLOSED, NO PATH)。** 排查质量认可 (真实规模 + 每条路线实测 + 回退),
"GDN ~0.8ms 接近实际下限" 与 chunked 的量化否决均成立; `__expf` 按"低于门槛选项"记档 (最终回顾可复议)。
V1-V3 与全部排查产物留在 artifacts, 将来换硬件/架构可复用。
至此**所有任务结束**; 下一步 = 最终统一回顾 (用户 Q7), 待用户发起。

---

## 2026-09-22 | T05 剩余结案: CLOSED (无实质收益) | implementer

按用户指示做了 T05 剩余 (quantize 融合 + 小 kernel 合并 + 间隙)。**结论: 无可接受风险的实质收益, 建议关闭。**

关键实测 (完整见 RESULTS "T05 剩余结案"):
1. **方法修正**: nsys 默认 `--cuda-graph-trace=graph` 会隐藏图重放 token 的 kernel -> 原 T05 profile 是"图前 token"数据。
   用 `--cuda-graph-trace=node` 重测得到稳态分解: MMVQ 29.84ms (86.9%) + quantize 0.84 + rms_norm 1.10 + elementwise 1.03
   + get_rows 0.47 + GDN 0.34 + FA 0.32 = 34.33ms/2024 kernels, host 间隙 3.4ms。
2. **MMVQ 已饱和**: 逐矩阵 685-845 GB/s (lm_head 845 = 94-99% 可用; 平均 677/825-850) -> 无 >=5% 空间。
3. **对照实验**: (a) rms_norm block 配置 1024->256 无效果 (已回退); (b) **关掉全部融合 tg128 仅 -2.6% (25.93 vs 26.63)**
   -> 现有 533 个融合头总共只值 0.70ms, **边际价值 0.9-1.0us/kernel**。
4. **定量判定**: 剩余 461 quantize + ~590 小 kernel 全合并也只有 1.0-1.1ms = +2.7-3.0% -> tg128 **27.3-27.5**;
   原估 -4.1ms/28.5-30 需核心级 kernel 合并 (多日, 高风险) 或 MMVQ 再快 10%。

建议: ① T05 剩余 CLOSED, decode kernel 侧天花板 ~27.3-27.5 ② 不再投入 decode 侧
③ 队列只剩 T04 (ub +21%, 待用户口径) 与最终统一回顾。工作区 0 净改动, PPL 4.3569 复核通过。

ANSWERED (analyst, 2026-09-22): **接受 T05 剩余结案 (CLOSED, 无实质收益)。** 关融合实验 + MMVQ 饱和证据充分;
decode kernel 天花板下调为 **~27.3-27.5**, 不再投入; 生产 decode 由 MTP 承担 (42.7 t/s)。
工作区保持交付态。至此 T03/T05 均彻底结束, **所有任务结束**, 待用户发起最终统一回顾 (Q7)。

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

ANSWERED (analyst, 2026-09-23): **spec 修正认可** (mma 内核无 parallel_blocks 路径, 等价实现 = 覆盖 `blocks_num.x`);
PB=2 数据也证实了方向。两个问题:

**Q1 (短上下文条件)**: 同意"先测量再定条件"。判据 = pp512/4096/8192 各 >=2 轮交替, 一致性回退 >0.3% 才加条件;
条件建议 `ntiles_KV >= 64` (KV >= 4096 才用 PB=2), 短 KV 保持 T12 的 grid=80。若短点无回退则不设条件 (简单优先)。

**Q2 (验收组合)**: 确认。A = BASE 交付版 (SHA `102BF844`), B = T12+T16, 同 session 交替 **>=3 轮**;
点 = depth32k / pp32768 / **pp8192@depth128k** / pp512,4096,8192 / tg128 / ub2048 抽查; T12 单独不重验 (已知 +2.0%/+1.1%)。
门槛 (B vs A): **长文点 >= +4% (主判据)**; depth32k >= +1.5%; pp32768 >= +0.8%; 短点 |Δ| <= 0.3%; tg128 噪声内;
ub2048 不回退; PPL 门槛 (B 已 4.3568) + 200 token 生成检查。
若长文点落在 **+2~4%** 且其余全过 -> 报 analyst 复核, 不自动拒; < +2% -> 拒。

**附 (重要)**: 构建/部署后**必须核对部署路径 SHA** (RESULTS 风险的硬链接疑点, 中止构建会删产物)。
夜间静音的时段许可由用户给, 我已把"等待用户时段"标进 BOARD。

---

## 2026-09-23 | 新提案: T18 (Volta FA mma 内核效率重写) - 长文 attention 是最大痛点 | implementer

用户明确表态: **长文降速是当前最头痛的问题, "能解决最好"** -> 提请立项评估。

### 动机 (全部本会话实测)

| 事实 | 数据 |
|---|---|
| 长文 attention 占比 | depth32k 32.0% (1899.4/5934.3ms) / pp32768 18.5% / **pp8192@depth128k 推算 ~65%** |
| FA 内核 MFU | 29.5 TF/s (ub512 depth32k) / 39.9 (ub2048) / **125 TF peak = 24-32%** |
| 命中 kernel | `flash_attn_ext_f16<256,256,32,2,0,0,0>` block=(32,4) 128 线程, dynSM **67584 B**, regs 254 -> **1 CTA/SM (4 warps)** |
| 停顿证据 | 每 KV chunk (32 KV x 64 q): mma issue 极限 ~1.0K cycles, LDS issue ~1-4K cycles, **实测 ~20K cycles -> 5-10x 停顿** |
| 波次已捡完 | T12+T16 (PB=2) 已把 grid/波次收益吃到 -14.7% (vs T11-off); PB=4 过切, grid96 +32% -> 负载均衡路线到头 |

结论: attention 的"慢"不是调度问题 (T10/T11/T12/T16 已解决到 -14.7%), 而是**内核本身延迟受限**;
要拿 1.5-2x (长文端到端 +25-30%) 必须改内循环。这是 llama.cpp 上游共享文件 (fattn-mma-f16.cuh), 属核心级改动。

### 已否证 (不要重复提)

- stream-K grid 数调优 (PB=2 已最优, -14.7%), PB=4/grid96/grid160/grid192 全测过
- 提占用: Q_in_reg 必 spill (cfgA +75%), nthreads=256/512 变体 (cfgE +637%, cfgD 编译失败);
  根因 = VKQ combine 缓冲 67584 B 占满 smem + 该配置行同时服务 ncols=64 的全部切分实例
- 不移植 1Cat 的 split-D/N32 (T10: ub2048 同内核已 39.9 TF/s >= 其 29-38)

### 候选方向 (按性价比排序, 建议 harness-first)

1. **K/V tile 软件流水线**: Volta 无 cp.async -> `nstages=0` -> K/V 装载同步, 每 ~1000 个 KV chunk
   暴露一次全局延迟。用寄存器做 double-buffer (预取下一 chunk 到 reg, 再做本轮 mma, 然后落 smem)。
   局部改动 (fattn-mma-f16.cuh 的 load_tile/iter), 是第 1 个该试的
2. **warp tile 扩展 / 操作数复用**: Volta 只有 m8n8k4, 操作数复用靠显式展开; 每 warp 覆盖更大的
   m/n 可摊薄 LDS 操作数流量 (当前每 mma 2 次 LDS.32, LDS 是首要瓶颈假设)。
   动 T_C_VKQ/T_B_KQ 布局, 风险中等 (tile 断言多)
3. **消掉 67584 B combine 缓冲**: 它是 smem 唯一大头 (输出 fp32->half2 经 smem 转置再写 global);
   改成直接按列片段写 global (上游注释说 smem 中转更快, 需实测) -> 若成立可 2 CTA/SM (免 Q_in_reg)
4. **纯配置探针**: ncols=32 的 Volta 配置行 + dispatch env (Q staging 减半, 寄存器压力减半;
   代价 = K 复用减半) - 0.5 天, 可以顺带做, 用来关闭/开启方向 3

### 建议的门槛结构 (照 T02/T03 先例)

- Stage 0 (0.5 天): 把该 kernel 抽成独立 harness (真实形状 DKQ=DV=256, ncols=64, l=32768),
  复刻 stock 时间 (2295us/launch @depth8k 量级) 作为基线
- Stage 1 gate: harness 内 **>= 1.25x** (attention 时间) 才准进集成; 不到 -> 写数字归档, 不硬凑
- Stage 2: 集成后验收 = depth32k / pp32768 / **pp8192@depth128k >= +15%** (相对 T12+T16 后基线) /
  短点 |delta|<=0.3% / tg128 / PPL 4.3572+/-0.013 + 200 token 生成
- timebox: 3 天, 中途 1.5 天 checkpoint

### 请求

请 analyst 裁决: (a) 是否立项 T18 及其编号/范围; (b) 门槛与 timebox 是否按上面;
(c) 若批, 我明天先做 T17 测量 + T12/T16 验收, 之后开 T18 Stage 0。

(用户侧: 长文优先; 当前生产 decode 已由 MTP 承担 42.7 t/s, 不受影响。)

ANSWERED (analyst, 2026-09-23): **立项 T18 批准**, 但修正三点 (spec 见 TASKS/T18):

1. **基线修正 (关键)**: Stage 1 的 1.25x 必须对**生产配置 (T12+T16, PB=2/grid=384, ~2027us/launch)** 算,
   不能对 stock (2375us) 算——否则 "1.25x" 实际只有 1.07x, 是假胜利
2. **门槛**: Stage 1 >= **1.20x** 才集成 (≈41.5 causal TF/s, 高于 1Cat 最好 38); 1.5 天 checkpoint <1.10x 即停;
   Stage 2 长文点 >= **+10%** (65% 占比 x 1.2x 的保守值)、depth32k >= +4%、pp32768 >= +2%、短点不回退、PPL/生成
3. **方向顺序**: ①K/V 寄存器流水 -> ④ncols=32 探针 -> ②消 combine smem (->2 CTA/SM) -> ③warp tile 扩展
   (风险最高放最后); 可依实测调整

**期望管理**: 1Cat 专用 FA-V100 也只有 29-38 TF/s, 我们 ub2048 已 39.9 -> V100 该形状天花板估 40-45 TF/s;
T18 概率上 >=1.2x 约 30%, 1.5x <10%——3 天硬上限 + 停损严格执行是前提。

**排期**: (a) 先等用户时段: T12+T16 合并验收 + T17 测量; (b) T18 Stage 0 harness 在静音时段可先写
(仅 CPU/代码, 不动工作区); GPU 阶段再排时段。批准无误。

---

## 2026-09-23 | T12+T16 合并验收: 全部门槛 PASS, 提请 VERIFIED | implementer

验收数据见 RESULTS "T12+T16 合并验收"。一行速览 (A=BASE -> B=T12+T16):
- **pp8192@depth128k 337.39 -> 375.45 = +11.28%** (主判据 +4%, 超 2.8x)
- depth32k 651.10 -> 682.16 = **+4.77%**; pp32768 774.19 -> 791.07 = **+2.18%**
- 短点 +0.03~0.56% (B 略快, 无回退); tg128 +0.06%; ub2048 -0.1%; PPL 4.3562; 生成 OK
- 128k 单轮: 你给的前两阶段都只有 <0.5% 轮间波动, 用户据此叫停了 128k 的剩余轮次 (省 20+ min 重负载);
  若你要 2 轮, 请说明 (需额外 ~10 min 重负载窗口)

提请: (a) T12/T16 标记 **VERIFIED**; (b) patch 已归档 `patches/v100-t12t16-fattn-split.patch`
(含 T12 启发式 + T16 PB=2 + 3 个 env 开关, 双向 apply 校验); (c) 后续按 BOARD 队列 = T17 测量 -> T18 Stage 0。

数值注: PPL 4.3568 -> 4.3562 (-0.0006), 来源 = PB=2 的 softmax 归并顺序 (KV 切 2 段), 与 GDN vec4 同类良性;
生成质量检查 (200 token) 无重复/乱码。

---

## 2026-09-23 | T17 结案: CLOSED (不投入) | implementer

- 微基准: 当前 Q8_0 内核 **767.2 GB/s** (Q6_K/Q5_K 参考线 707-825 同档); 我写的向量化候选 669.8 GB/s
  (逐位相同但更慢 -> Q8_0 的 34 字节错位块布局下, 上游的 smem 中转是更优解)
- nsys pp512: Q8_0 dequant 18.84ms / 1551.7ms = **1.2%**; 实模型折算 ~877 GB/s (>= 可用带宽 825-850)
- 判定: 不偏慢, 空间 <= ~0.3% < 门槛 +0.4% -> **记录数字关闭** (spec 第 3 条); 交付态不变
- 产物: `artifacts/t17_q8_bench.cu` (+exe, ks.csv)

队列剩余: **T18 Stage 0** (可静音做, 纯 CPU/代码)。

ANSWERED (analyst, 2026-09-23): **T12+T16 验收通过 -> VERIFIED (采纳入库)**; T17 结案接受。

- 数据复核通过: 主判据 **+11.28%** (门槛 +4%, 裕度 2.8x) / depth32k +4.77% / pp32768 +2.18% /
  短点无回退 / tg128 +0.06% / ub2048 -0.1% / PPL 4.3562 (|Δ|=0.001) / 生成 OK
- **128k 单轮豁免批准**: 长点轮间 <0.5% 且裕度 2.8x, 不补轮次 (豁免理由已记录)
- PPL 4.3568 -> 4.3562 (-0.0006): PB=2 softmax 归并顺序, 与 GDN vec4 同类良性, 接受
- patch `v100-t12t16-fattn-split.patch` + 5 文件 + DLL `453E2911` 已复核 (build == 部署, 我核对过)
- 热漂移注记已并入 PROTOCOL 测量口径 (长点轮换 A/B; tg128 用冷却后值)
- **下一步: T18 Stage 0** (harness, 纯 CPU/代码, 静音时段可开工; GPU 阶段等用户时段)

---

## 2026-09-23 | T18 Stage 0 DONE | implementer

- harness: `artifacts/t18_fa_harness.cu` (直接 include `fattn-mma-f16.cuh`, Stage 1 改动自动生效; 需 `--extended-lambda`)
- 形状: DKQ=DV=256, n_q=512, GQA 24/4, mask causal [n_kv, n_q]; 配置/ smem 从源文件读取 (smem 67584 ✓)
- **Stage 1 基线 (生产配置 grid=384+fixup) = 12.718 ms/launch @ n_kv=35072**; 对照 grid=192 = 15.218 ms (ratio 0.836)
  生产对照: 14.84ms (depth32k, grid=192); 相对增益 -16.4% vs 生产 ~-15% -> 保真
- 门槛: >=1.20x -> 目标 <= **10.60 ms**; 1.5 天 checkpoint <1.10x (<=11.56ms) 即停
- 下一步: Stage 1 方向 (1) K/V 寄存器软件流水线 (Volta 无 cp.async, nstages=0)

---

## 2026-09-23 | T18 Stage 1: **1.12x (未达 1.20x 门槛)**, 请裁: 归档 or 集成 | implementer

完整数据见 RESULTS "T18 Stage 1"。要点:

1. **可达最好: ncols=32 瓦片 (smem 35072 -> 2 CTA/SM = 8 warps) + PB=2/grid=768 = 1.12x @l=35072**;
   短 KV 更好 (l=12288: 1.22x), 长 KV 收敛 ~1.10x (l=100000)
2. ncu 诊断: ncols=64 是延迟受限 (occupancy 6.25%, Est.Speedup 61%); 提到 2 CTA/SM 后变成
   **L1/shared 数据通路受限 (74.5%)**, DRAM 8.5% / L2 96% 远未饱和
3. 突破 L1 的两条路都被硬件挡死: Q_in_reg -> 255 regs + 472B spill (实测 4x 慢);
   降 np -> 需 ncols=128 (smem 爆) 或 nwarps=1 (无并行)。**这是 Volta mma.m8n8k4 + 固定 cols_per_warp=32 布局的上限**,
   再往上要重写 tile/warp 布局 (多日, 超出 3 天 timebox)
4. Stage 2 预估: 128k +6.3% / depth32k +3.2% / pp32768 +1.85% -> 三门槛都差一点

请你裁决:
- **A (我倾向): 归档 T18 (不集成)** —— 按 spec 门槛 1.20x 未达; 1.12x 变体 + harness + ncu 日志全部留档,
  将来若要重写 tile 布局可直接复用
- **B: 破格集成 1.12x** —— 它是均匀正收益 (短 1.22x / 长 1.10x), 但需要额外改 dispatch (强制 Volta D256 走 ncols=32)
  + 完整模型验收 (~1.5h GPU); 收益 ~+6% 长文点, 达不到 Stage 2 的 +10%
- **C: 继续投入 tile 布局重写** —— 按现有诊断, 我估成功率 <20% 且在 timebox 外

(工作区已清理回 T12+T16 交付态; 部署 DLL 仍 453E2911; 本日 T12/T16 已全过, T17 已关闭)

ANSWERED (analyst, 2026-09-23): 裁决 **A (归档, 不集成)**; B 记为"可选补做项", C 不做。

- 依据: T18 spec 门槛 = 1.20x, 实测 **1.12x** 未达; Stage 2 三项预估 (+6.3%/+3.2%/+1.85%) 均差一点 -> 按 spec 归档
- **B 的评价 (保留)**: 1.12x 是均匀正收益 (短 KV 1.22x / 长收敛 1.10x) 且风险可控 (dispatch 强制 Volta D256 走
  ncols=32 + env 开关), 收益 ~**+6% 长文点**; 成本 ~1.5h GPU + 完整验收 -> **建议作为独立小项保留**,
  下次有窗口时优先做; 本次会话已收尾, 不自动执行
- C 不做: tile/warp 布局重写成功率 <20% 且超 timebox
- 归档内容: harness + 5 变体 exe + 2 份 ncu 日志 (artifacts), 将来复用
- 附带清理: 工作区 `fattn-mma-f16.cuh` 显示 M 但 `git diff` 为空 (CRLF/stat-dirty) ->
  建议 `git checkout -- ggml/src/ggml-cuda/fattn-mma-f16.cuh` 保证交付态干净

ANSWERED (analyst, 2026-09-23, 用户放宽 Q15): **T18 恢复 RUNNING**, 归档裁决撤销。新规则:

- **时间**: 探索期 **+2 天** (中途 1 天 checkpoint); 到点无论是否突破都进入集成阶段
- **探索方向** (未试的结构性路线, 按剩余价值): ①K/V 寄存器软件流水 (若确认 L1/shared 受限则优先级降) /
  ②消 67584B combine smem (可再提占用) / ③warp tile 布局重写 (原估 <20%, 现在有预算)
- **新采纳门槛 (替换 1.20x/1.10x)**: 集成后 **e2e 主判据 = pp8192@depth128k > +2% 即合并**;
  无回退要求照旧 (pp512-8192 |Δ|<=0.3%, tg128 噪声内, PPL 门槛 + 200 token 生成)
- **保底**: ncols=32 变体 (1.12x, 预估 128k +6.3%) —— 探索无突破则直接集成它, 只要实测 128k 点 > +2%
- 若实测 128k 点 < +2% 但 depth32k > +2% -> 报 analyst 复核 (不自动拒)

---

## 2026-09-23 | T18 终裁请求: 生产未达门槛 (128k +0.1%), 已回退部署 | implementer

结论 (完整数据见 RESULTS "T18 结案"):

1. **128k 点 (主判据) = +0.11%** (两轮: -0.80% / +1.03%, 漂移 ±1%) -> **未达 Q15 的 +2% 门槛**
2. depth32k **+1.50%** / pp32768 **+1.17%** (真实且两轮一致) / 短点与 tg128 无回退
3. **决定性证据**: nsys 逐 launch, l~135k 的 FA (>30ms): A(ncols=64) 40.28ms vs B(ncols=32) 40.38ms
   -> **长 l 下 B 慢 0.3-0.6%**; ncols=32 的收益在 l<=35k 真实, l>=100k 消失
4. 根因: harness (fp16 V + 独立热循环) 长 l 保真缺口; 生产 q8_0 V 是独立 dequant 内核 (1.2%, 无关)
5. 已按"未达门槛"处理: 部署 DLL 回退 `453E2911` (T12+T16); 源码 T18 两处改动暂留待裁

请裁决:
- **A (我倾向): 撤销 T18 改动** (git checkout fattn.cu / fattn-mma-f16.cuh), T18 归档为 "长文 FA 对 tile 配置不敏感,
  瓶颈在流式/预取"; 后续若再碰 FA, 先在目标 l (128k) 做 harness 保真对照
- **B: 破格保留** (depth32k +1.5% / pp32768 +1.17% / 128k 无损失, 零回退) —— 需补跑 PPL+生成 (0.5h) 再入库
- **C: 继续追长文**: 方向应转向 "长 l 流式/预取" 而非 tile 配置 (我估需要新 harness 口径 + 2-3 天, 且 P(>+2%) 未知)

(T17 已 CLOSED; T12+T16 交付不变; 工作区除 T18 两文件外干净)

---

## 2026-09-23 | T18 裁决结果: 用户选 A (撤回归档) | implementer

用户: "1% 就不要了, 还牵扯了注意力改动, 划不来" -> 选 A。
- 两处源码改动已 `git checkout` 撤销; 工作区 = 5 文件交付态; 部署 DLL = `453E2911`
- patch 留档 `artifacts/t18-ncols32-REJECTED.patch`; 无需你进一步动作 (归档结论已在 RESULTS/TASKS)

---

## 2026-09-23 | T20 完成, 请裁决修复采纳方式 | implementer

T20 (MTP 轨迹一致性独立排查) 已按用户要求独立完成 (不盲信旧报告; 旧报告 5 条主张 4 条被推翻)。
完整证据链见 RESULTS "T20"。要点:

**分叉源 3 个**:
- S1 FA VEC(n_q=1) vs TILE(n_q=2..4) - 旧报告 H2, 确认
- S2 **GDN vec4 布局分支** - 本 fork T03/A2 引入 (n_tokens>1 换 lane 行映射 -> warp 归约顺序变);
  旧报告"GDN batch-invariant"不成立 (旧 base 上恰好未触发)
- S3 **FA VEC split-K 的 padding 边界** - 上游设计属性 (padded n_kv 跨 256 -> parallel_blocks/交错切分变);
  server logits dump 证明首个差在采样点 242 (position 255, 首个 256 边界)

**验收 (三源全修)**: n-max 1/2/3 与无 spec **逐 token 完全一致** - 短 900 / 32k 150 / 128k 100 token;
回滚+快照路径工具级逐位验证精确; PPL 4.3562 (同交付); 确定性复测通过

**代价** (llama-bench 同 session 交错):
- S1: 短点 ~0 (nmax1/2/3 36.9/39.1/37.2 vs 36.9/39.5/36.9)
- S2: tg128 d0 **-0.6%** / d32768 **-1.3%** (交错复测)
- S3: tg128 d32768 **-31%** (23.10 -> 15.83; 128k 未测, 预计更差) - 因 KV 切分是长文 decode 的主要并行度来源

**请裁决 (三选一)**:
- **A (我倾向): 固化 S1+S2 (代码默认), S3 只留 env 可选** - 收益: 消除 fork 引入的 S2 + 恢复旧报告级修复;
  残余 = S3 的极端近并列翻转 (900 token 内 1-3 次, top1-top2 <=0.02 nats, 两条续写均合法), 生产可接受
- **B: S1+S2+S3 全固化** - 100% 逐 token 一致, 但长文 decode -31% @32k (与"速度不回退"验收冲突)
- **C: 全部不采纳, 仅归档** - 保留现状 (分叉最频繁, 900 token 内 3+ 次, 含 1e-3 级 FA 差异)

若选 A/B, 需要: 固化代码 (S2: 删除 `n_tokens > 1` 分支一律 vec4; S1: Volta 分支 `vec_limit` 2->16;
S3: env `GGML_CUDA_FATTN_PB_FORCE` 或更名) + 全量验收 (PPL/pp 曲线/tg/长文点/生成) + 重建部署核对 SHA。
工作区已备好实验 patch (`artifacts/t20-work.patch`), 部署已回滚 T19。

ANALYST NOTE (2026-09-23): T20 证据链复核 **通过** (独立复现/算子定位/内核强制矩阵/logits dump/回滚深度, 方法正确;
旧报告 4/5 条被推翻, **S2 确系本 fork A2 (GDN vec4) 引入**, 已记教训)。裁决仍由用户定, 但**掉速分析先给下列数据**:

1. **S2 零代价统一 (优先试)**: 保留标量访存, 把标量路径的 lane 行映射改为 vec4 的 `4*lane+r` (只改索引, 不改访存宽度)
   -> 归约顺序一致, 预期 decode 无代价。若成立则不需要"一律 vec4" (-0.6~1.3%); 失败再退回一律 vec4,
   或评估"直接回退 A2"(代价 = 失去 pp512 +1.0%, 收益 = decode 恢复 + 消除 S2 根因)。
2. **S3 替代修法 (探索, 不设前提)**: 目标 = **不改并行度**地消除 padding 边界扰动:
   a. decode 与 verify 对齐同一 `ntiles_KV`/切分边界 (对齐口径而非禁切分);
   b. VEC 内核内做与 `parallel_blocks` 无关的确定性 KV 归约顺序 (kernel 级, 成本待测);
   c. 都不可行 -> 接受 A 的 S3 残余 (仅近并列翻转), env 保留复现用。
3. **T19 遗留并入**: tg d131072 -3~-10% 一并定论 (4-6 轮交替 A/B; 检查与 S3/切分是否相关);
   同时给 **S1+S2 固化版** 的完整代价 (短/32k/128k decode + MTP 各档 + PPL)。
4. **产出**: nsys 逐 launch (定位 S3 -31% 的耗时去向) + 同 session 交错 A/B 表, 回本文件报数据后交用户裁决。

---

## 2026-09-23 | T20 补充: S3 找到低代价修法, 建议直接采纳全部三项 | implementer

上一条 QUESTIONS 的选项 A/B 已合并: S3 不再需要 -31% 的 PB=1, 改用
**小批量固定切分块数** (与 batch 无关; 空块贡献严格 0) -> decode/verify 逐位一致。

- 实测代价 (tg128, 交错): d0 **-0.8%** / d32768 **-0.5%** (全部来自 S2; S3' 代价 ~0)
- 验收: 短 900 token n-max 1/2/3 全等; 32k n-max3 150 token 全等 (128k 按用户指示未测)
- 详细见 RESULTS "T20" 第 8 节

**修订后建议: 直接采纳 S1+S2+S3' 全部固化** (总代价 ~0.5-0.8%), 不需要再二选一。
待用户确认后执行: 固化代码 (S1: Volta vec_limit 2->16; S2: 去掉 n_tokens>1 分支;
S3': 已实现, 去掉 PB_FORCE 调试 env 或保留) + 全量验收 + 重建部署核对 SHA。

ANALYST NOTE (2026-09-23, 对 §8/§9 复核): 方案合并**支持** (原理正确: 固定切分 stride -> 部分和一致,
空块贡献严格 0; ~0.5-0.8% 总代价可信)。固化前请补三项:
1. **最终验收必须含 128k 点** (轨迹一致性 + pp/tg + PPL; 长文是主场景, 不能省);
2. **显存核对** (预期 0 增量, 逐项记录);
3. **注明 S3' 是 NVIDIA 通用改动** (非 Volta 门控; 小批量 n_q<=8 全 arch 生效): 社区/其他卡测试时留意速度影响。
另: §9 的 T19 d0@207 翻转 (0.068 nats gap) 建议在最终报告注明来源归因 (sm70 FA 配置 / T12+T16 归并) 与质量结论 (PPL 不变)。
