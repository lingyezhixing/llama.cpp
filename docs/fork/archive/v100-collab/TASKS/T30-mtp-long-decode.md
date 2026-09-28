# T30: MTP 长文提速 (主候选: 验证路径 TILE->VEC) - 用户批准立项 2026-09-26

状态: **REJECTED (2026-09-26 Phase A 实测; 见文末最终结果与 analyst 复核)**; 曾 APPROVED (用户 2026-09-26: "mtp 到 128k 只剩 23 左右, 太慢了; 看看有没有办法加快");
执行者: implementer; 基线 = T19-L + T24 (`3eae5cdae`); 需 GPU 时段 (深夜静音规则 -> 与用户协调)
来源: T20 排查产出 ("MTP 验证路径 TILE->VEC, 纯速度, 从 T20 拆出", `OPTIONS §H`); 本轮范围 = **MTP 长文生成提速**, 128K 为主要痛点。

## 背景 (为什么 128K 慢)
- 用户实测: MTP3 (`--spec-type draft-mtp --spec-draft-n-max 3`) 生成速度 **128K 时仅 ~23 t/s** (implementer 先复测确认)
- 机制 (T20 已定位, 封存材料 `04-T20源码排查/调试开关与改动说明.md` S1):
  - `fattn.cu` Volta 分支上游 `vec_limit = 2`: decode (n_q=1) 走 **VEC** 核;
    verify (n_q=2..4, 该分支判据不满足) 走 **TILE 核 + V 物化 f16**
  - 128K 时 V 物化 (q8_0 -> f16, 17 个 full-attn 层) 量级 **~9GB 流量/轮** (估算, 需实测)
- 相关但不在本任务: ≥128K 时 MTP 接受率本身衰减 (T20 旧数据: 128k +46% / 32k +53% vs none);
  "K 的经济性" 属 T21, 本任务只做 **每轮速度**

## Phase A: 测量/定位 (先做, 不预设胜负)
1. 基线: server MTP3 greedy seed42, d0 / d32768 / d131072, 生成 100-200 token;
   记录 t/s、接受率、平均接受 token/轮; 同 session 轮换顺序 (热漂移)
2. nsys 一轮 @d131072: 按类拆时 (目标 verify GEMM / FA 核 (哪个核) / V 物化 copy / draft 步 / 采样接受 / 其他)
3. 选核微基准 (harness 或 nsys): verify 形状 (n_q=4, l≈128K, V q8_0) **VEC vs TILE(+物化) 单独计时**;
   **必须先回答: VEC 在 n_q=4/长 l 下的并行度是否够** (若 VEC 本身更慢, 预期 +5-8% 不成立 -> 如实报结果)
4. 产出: "一轮 128K MTP 的时间去向" 表 + VEC/TILE 对比数字

## Phase B: 实现 (仅在 Phase A 支持时)
- `fattn.cu` Volta 分支加选核开关 (小改动, env 门控): 让 verify (小 n_q) 走 VEC;
  **decode 路径与数值不动** (decode 本来就是 VEC)
- 交错 A/B (同 session, 轮换顺序) 测 d0/d32768/d131072 的 MTP3 t/s
- 若 VEC 慢但物化是主因: 记录并评估备选 (TILE 去物化 = 新内核/大工作量; 或按 l 阈值混合选核)

## 数值性质 (先声明)
- TILE->VEC **不降低精度**: 两边都是 f16 输入 + f32 累加, V 仍 q8_0; 不引入近似函数/低精度存储 (区别于 T13/T25)
- 但**会改变浮点结合** (KV 维归约顺序不同): 这是 T20 S1 分叉源 (层输出 maxabs ~2.5e-3 级, 近并列点可翻 token)
  -> 按用户 Q17 口径允许; 且 verify 与 decode 归一为同核 (VEC), 消除 S1, 与不开 MTP 的对齐只会更好
- decode (n_q=1) 路径不动 -> 无 spec 的 PPL 应逐位不变

## Phase C: 验收 (若采纳)
- MTP3: **d131072 (主)** + d32768 + d0, t/s (A/B 3 轮交替); 接受率同报
- 无 spec 控制点: d131072 t/s + PPL 逐位不变 (证明不影响 decode)
- PPL: `|PPL - 4.3567| <= 0.013` (**控制位**: 无 spec PPL 不经过 verify -> 应逐位不变)
- MTP 侧质量: 生成流畅/无乱码重复 (d0 + 128K) + **接受率对照不降** (明显下降 = logits 漂移过大 -> 回退); 显存中性
- 长文 prefill 防回归: pp512 + pp8192@depth128k
- 口径: 用户 Q17 = 不要求与老 MTP 逐位; 只要实现正确; 能与不开 MTP 对齐更好

## 交付物
- RESULTS "T30": 时间去向表 + VEC/TILE 对比 + (若实现) A/B t/s 表 + PPL/生成结论
- 结论: 是否采纳; 若采纳给默认行为 (env 默认/门槛) 与 patch

## 约束
- 不改 decode 数值路径; 不动 GDN/A4; 不重开已否决项 (T11/双 stream/融合等)
- 未达门槛就写 BLOCKED/REJECTED + 数据, 不硬凑


## T30 最终结果 (implementer, 2026-09-26): REJECTED

Phase A 测量结论: verify 形状 (n_q=4, l=128K, V q8_0) 下 **VEC 比 TILE 慢 3.1x** (nsys 直测 4.68 vs 1.50 ms/层),
即使 V 物化被完全消除 (VEC 窗口内物化实例 = 0); 提高 KV 切分 (PB/NBATCH=160/1024, 13 -> 160 片) 也慢 15%。

A/B 矩阵 (server MTP3, d131072=128270 tok, 150 token, 同 session A0->A3->A2->A1->A0b, tps 冷/热):
A0 基线 20.06/20.13; A3 (VEC+PB+NBATCH) 12.48/13.02; A2 (PB+NBATCH) 15.26/17.31; A1 (VEC) 12.78/13.81;
接受率不变 (0.3883 / 0.3835), d0 各配置 49.8-52.5 t/s, PPL 控制位 4.3567 **逐位不变** (decode 路径未动)。

按规格 "若 VEC 本身更慢 -> 如实报结果" 与 "未达门槛写 REJECTED + 数据" 结案, Phase B/C 未进入。
备选 (未做, 大工作量): 新 TILE 内核直读 q8_0 V, 上限 ~10% @128K。长文接受率衰减 (0.388 vs d0 0.808) 属 T21。
详见 `RESULTS.md` "T30 Phase A/B"。实验改动已回滚, 工作区干净; 复现脚本与 nsys 报告见 RESULTS 第 7 节。

### analyst 复核 (2026-09-26)

- **否决正确, 流程有效**: Phase A 判据干净 (VEC 慢 3.1x, 物化只占 ~10%), 预先声明的风险 (VEC 并行度) 命中 ->
  未浪费 Phase B/C; PPL 控制位逐位不变 = decode 未被误伤 (与预测一致)。
- **机理**: TILE 能共享 K/V tile (grid=(1,13) 但每 block 复用多 tile), VEC 在 n_q=4 下 cols_per_block=2 且全程仅 13 warp ->
  上游 "n_q*gqa>2 用 TILE" 的启发式是对的; 物化是 TILE 的合理代价而非浪费。
- **剩余空间 (按性价比)**: (a) **T21 接受率经济学**: 128K 接受率 0.388 vs d0 0.808 -> 每轮平均 2.16 vs 3.42 token,
  这是长文收益衰减的另一半 (用户要的"128K 提速"更可能来自这里 -> 建议重开 T21); (b) 新 TILE 内核直读 q8_0 V:
  上限 ~10% @128K, 需新内核 + 全量 FA 回归 (大工作量, 待用户判断); (c) 内核级重写靠近 K/V 带宽下限 (~8.2ms/轮 vs 当前 ~25ms/轮) -- 不做。
- 备注: 用户上报 128K ~23 t/s 与实测 19.5-20.1 的差异可能来自 prompt/生成长度/采样参数; 后续以此表口径为准。
