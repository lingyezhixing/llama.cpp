# T09: 融合 2.0 验证 (量化 operand 版) - 已结案

状态: **REJECTED / CLOSED (T09-A FAILED, 融合永久关闭)**
优先级: P1 (若批准, 排在 T08 之后或并行)
预期收益: 若成立 +5-10% 整机 prefill; 若失败则永久关闭融合路线
成本: 1 天 (微基准 0.5 + CUTLASS 调研 0.5), 不写生产代码

## 为什么值得回头看 (T02 Gate A 的漏洞)

Gate A 的骨架用的是 **fp16 权重** (预反量化好的), 也就是说:

- 它测的是"我们的搬运/流水线 vs cutlass" -> 我们落后 25% (57.5 vs 77-84 TF), 结论正确
- 它**从未测过融合的正面机制**: 读量化权重 (Q6_K 25B/32値) 比 fp16 (64B/32值) 少 2.6x 的
  global/L1 读流量。而这个机制正是 T01 指出的唯一可能反超点
- "mma 路径单独 1.472ms > baseline 1.404ms" 只对**那个骨架**成立, 对"搬运效率达标 + 量化 operand"的
  设计不构成否决
- 反量化成本的估算 (4.4M 指令/SM = 44% issue 预算) 是**估算**, 口径存疑 (按 ~2 指令/权重写的注释,
  数值却对应 ~4 指令/权重); 用 LOP3/PRMT + half2 解包可做到 1-1.5 指令/权重, 即 11-17%
- 权重重复反量化 (BM=128 时每个 m-tile 一份, ub512 -> 4x) 这个维度没有被探索过

## 冷水的部分 (先说清楚, 避免重复 Gate A 的失望)

- 融合省的是 **global 读** 那 2.6x; 但 STS (写 smem) 和 LDS (读 fragment) 的流量不降,
  两者占 L1TEX 的大头 -> L1 真实节省可能只有 ~20%, 不是 2.6x
- 解包指令与 mma/HMMA 抢 issue 是真实的, 只能在"每权重 1-1.5 指令"的前提下才划算
- 还要追平 cutlass 的搬运效率 (Gate A 落后 25%), 这是历史级难题
- 综合: 期望是"小胜或打平", 不是翻盘; 概率估计 20-30%

## T09-A (0.5 天): 量化 operand 微基准 (决定性问题)

- 在 v4 骨架基础上: B (权重) 以 Q6_K 形态加载进 smem, 块内解包成 fp16 再喂 mma; 只做 gate/up 形状
- 两档 tile: BM=128 (权重重复 4x) 与 BM=512 (重复 1x, 但 grid 仅 136 blocks -> 观察 tail/并行度)
- **实测**解包指令数 (逐 SASS 计数, 不用估算); 解包用 LOP3/PRMT/half2 手写
- go/no-go:
  - >= 84 TF: 大胜 (相当于打平 cublasLt), 进入集成讨论 (整机 +8-11%)
  - 70-84 TF: 条件集成 (按形状开关)
  - 66-70 TF: 边缘, 交 analyst
  - < 66 TF: **永久关闭融合路线**, 不再投入

## T09-B (0.5 天): CUTLASS 2.x sm70 定制 mainloop 调研 (只读文档, 不写实现)

- 确认哪个 CUTLASS 版本仍支持 sm70 fp16 tensorop (2.x; 3.x 起砍掉 Volta)
- 自定义 iterator / "dequant staging" 的可行性; 有无先例 (fused quantized mainloop on sm70)
- 产出: 一页结论 (可行性 + 工作量估计 + 是否值得 3-7 天实现)

## 与其它任务的关系

- 与 T08 (cublasLt) 收益重叠 (都在 gate/up), 不要重复计入总账
- T09-A 失败则全域上限不变 (~1000-1050); 成功也只到 ~1050-1100, 不会到 1200

---

## Analyst 结案 (2026-09-22): REJECTED, 融合路线永久关闭

- 三条独立否决成立 (复核 RESULTS 的 T09-A 章节):
  1. BM=128 去解包 (mode 8) = 1.569ms > baseline 1.404ms -> 即使解包零成本也赢不了
  2. 解包成本 0.46-0.66ms (18-30%), Volta 无 cp.async 且与 mma 抢 issue, 不可重叠
  3. BM=512 (权重只解包一次) 寄存器/smem 物理不可能; 权重解包重复次数 = N/BN = 4 无法消除
- 根因 (实测): v9 的 staged 流量比 fp16 版少 25% (1.07 vs 1.42GB), 但时间几乎相同 ->
  **kernel 不是 DRAM 带宽受限, 是 smem staging 的延迟/发射受限**, "少读 2.6x" 落不到时间上
- 附带修正认知: ub512 prefill 的瓶颈不是 DRAM 带宽 (848GB/s), 而是 staging 延迟/发射
- T09-B (CUTLASS 调研) 按 implementer 建议**跳过**: cuBLAS 水平 staging (76.9 TF) 加 18-30% 解包
  = ~53-63 TF < 66 TF 盈亏平衡, 收益上限为负
- 决定: **融合路线永久关闭**; 后续 prefill 只剩 T08 (cublasLt, +2-5% 待证) / T07 小项 / T03 chunked
