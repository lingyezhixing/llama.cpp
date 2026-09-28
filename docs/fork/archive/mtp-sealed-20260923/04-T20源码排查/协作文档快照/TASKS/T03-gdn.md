# T03: gated_delta_net 优化

状态: **CLOSED (NO PATH FOUND)** (2026-09-22 彻底排查结案); vec4 已入库, chunked 不可行, `__expf` 记档为低于门槛选项
最终结果: kernel 879.9 -> 826.3 us/层 (-6.1%); pp512 +1.0% (959.7); tg128 噪声内; PPL 4.3569

## 已做

1. **向量化行布局 (vec4)**: lane 拥有连续 4 行 -> state/k/q 用 float4 访存;
   按 `n_tokens > 1` 分支 (decode 单 token 走原标量路径)
2. **KDA 缺陷修复**: `use_vec4` 加 `&& !KDA` (旧索引会让 g 乘到错误的行; 本模型 kda=false 不触发)
3. 复核结论: 该 kernel 是总指令吞吐受限 (两次 warp 归约占 20/45 条指令), 微调到头

## 为什么暂停 chunked 重写

- 理论 kernel -60% -> 整机 +3-4%, 需要 1-2 天 + 3 个高风险点 (decay / 三角求解 / keep_rs_t 快照)
- 优先级低于 T02 (prefill +5-10%); 仅当 T02 失败且其他任务都完成后重启
- decode 侧: chunk 对单 token 无效; 若要优化 decode 的 GDN, 应减少 launch 而非重写

## 保留注意事项

- 不要为速度把状态精度降到 fp16/bf16, 除非用户单独批准
- patch: `D:\LLM\Backend\patches\v100-gdn-vec4.patch` (已更新, 含 n_tokens 分支)

---

## Result 补充 (implementer, 2026-09-22): vec4 对数值的影响已定位

问题: 交付版 PPL = 4.3569, 原始基线 4.3572, 偏移 0.0003 的来源。

3 路 DLL A/B (seed 42, 512ctx/8chunks) + 一次隔离构建 (unary.cu 回退后重建):

| DLL | 内容 | PPL |
|---|---|---|
| ggml-cuda-old.dll | 原始 | 4.3572 |
| ggml-cuda-fast.dll | + 向量化 dequant | 4.3572 (**逐位相同**) |
| 隔离构建 | dequant + GDN vec4 | 4.3569 |
| 交付 DLL | dequant + GDN vec4 + silu vec4 | 4.3569 |

- **偏移 100% 来自本任务的 vec4 行布局**; dequant 与 silu 均为逐位相同 (加回 silu 后 PPL 不再变化)
- 机制: vec4 把 lane 拥有的状态元素从 `r*32+lane` 改为 `4*lane+r` -> warp 内点积 `sum_r s[r]*q[r]`
  的求和分组顺序改变 -> fp32 加法重结合 -> 约 1 ulp -> 48 层累计成 0.0003 的 PPL 差
- 量级判定: 0.0003 (0.007%), 远小于验收门槛 0.013, 也远小于 PPL 自身 +/-0.24 的误差棒 -> **良性, 不需要返工**
- GDN 的 kernel 级收益不受影响 (879.9 -> 826.3 us/层)

---

## Analyst 约束 (chunked 重启, 2026-09-22)

- 目标: GDN kernel 40ms -> ~20ms (-50%), 整机 pp512 +3-4% (验收主口径)
- 三个 checkpoint (任一失败即 BLOCKED, 不硬凑):
  1. decay 递推与当前 scan 参考实现逐段一致 (随机输入数值对拍)
  2. 三角求解正确性 (对拍当前 kernel, 相对误差 <= 1e-5)
  3. keep_rs_t 快照语义 (chunk 边界状态传递) 与串行版一致
- 数值门槛: chunked 重结合漂移会大于 vec4, 但仍须 `|PPL - 4.3572| <= 0.013` + 200 token 生成检查;
  超门槛或降智 -> REJECTED
- 验收: pp512/4096/8192 + **pp32768/depth32k** (GDN 随上下文线性, 长上下文必须给) + tg128 不回退;
  端到端须**同 session A/B 交替 >=2 轮** (机器存在 ~1% 的 session 级漂移, 禁止跨 session 混比)
- decode 路径 (n_tokens==1) 不动, 改动走 n_tokens>1 分支 (同 vec4 的既定模式)
- timebox 1.5 天; 0.75 天处给 checkpoint 报告 (即使是半成品也要写数据)


---

## Checkpoint 报告 (implementer, 2026-09-22, 0.75 天点): 数学原型完成, 三 checkpoint 全过

- **checkpoint 1 (decay) / 2 (三角求解) / 3 (chunk 边界状态) 全部通过, 相对误差 1e-16 (机器精度)**
  - 覆盖 L=1..64, D=8..128, 单 chunk + 多 chunk (T=256/L=64)
  - 脚本归档 `artifacts/t03_chunked_reference.py` (含算法注释与验证代码)
- 算法 (per chunk): Ac=exp(cumsum(g)); Bt=β/Ac; A_beta=inv(I+tril(diag(β)KKT,-1))diag(β);
  W=A_beta K; U=A_beta(diag(1/Ac)V); Sn+=K^T(U-W Sn); S_out=Ac[-1]Sn;
  Qtilde=Q-tril(QK)W; C=QK-tril(QK)A_beta tril(KKT,-1); O=scale*Ac*(Qtilde Sn + tril(C)(Bt V))
- 实现要点: 不用显式求逆, 用同一 (I+M) 做 3 次前代 (L 个右端项) 得 W/U/R
- 设计: K1 (chunk 并行: KKT/前代/W/U/Qtilde/C/块内输出) -> K2 (chunk 串行: 状态扫描+状态项输出);
  仅 `n_tokens>1 && !keep_rs_t && !KDA && S_v==128` 启用, decode 路径不动
- 成本: 每 chunk 每 (seq,head) ≈ 4.6M MAC (L=64,D=128), 现状 kernel ~0.5 TFLOPS -> 目标 2 TFLOPS 即可 -50%
- **CUDA 实现未开始** (timebox 内剩余时间不够做完 + 验证, 按约定不硬凑); 下一步见 RESULTS "T03 checkpoint"


---

## CUDA 原型 V1 结果 (implementer, 2026-09-22): 正确性过, 性能不达标

- 正确性: harness (D=128,H=32,L=32,COLS=32) vs 串行参考, T=512/16 chunks -> out 5.3e-7 / state 1.7e-7 ✓
  (纯 fp32; 逐阶段 W/U/R/QK/T1/OS/C 与 numpy 对拍全过; 修掉 2 个真 bug: C 的 m<=t 掩码、
  U 右端项多乘 beta -> 正确形式 `U = A_tri @ (diag(1/Ac) V)`)
- 性能: V1 = 4.60 ms/layer-ubatch vs 真 kernel 0.826 ms -> **慢 5.6x, 未集成**
- 瓶颈: smem 32-way bank conflict (sM/sK stride=128 未 pad), occupancy 1 block/SM (73KB),
  串行前代 + C 阶段并行度低, 每 chunk 8 次 sync
- 产物: `artifacts/t03_chunked_v1.cu` (+ .exe + t03_check_state.py); 工作区 0 行改动
- 下一步: 性能重写 (pad + L=16 + 2D tiling + 提高 occupancy), 预计 ~0.5-1 天

## Analyst 裁决 (V1 后, 2026-09-22): 批准 A (性能重写), 设中间门槛

1. 集成前硬门槛: V2 <=0.60ms/layer-ubatch (现状 0.826 的 -27%) 才准集成; 打平 (0.60-0.83) 不集成; 目标 <=0.41 (-50%)
2. 正确性广度: L=16/32/64; 尾块 (T%L!=0); 真实模型形状 (D=128 + 实际 head 数); 多序列等价或回退
3. 资源: smem <=48KB (>=2 block/SM); 保持 fp32 状态精度
4. 集成后: 同 session A/B >=2 轮 + pp512/4096/8192 + pp32768/depth32k + tg128 + PPL 门槛
5. timebox 追加 1 天硬上限; +0.5 天进展报告; 不达标 BLOCKED 归档, 转 T05/T04
（与 QUESTIONS.md 的 ANSWERED 同步）


---

## V2 进展 (implementer, 2026-09-22, 0.5 天点)

- V1 -> V2a (pad 消 bank conflict + C 全并行) -> **V2b (L=16)**: 4.60 -> 3.29 -> **1.644 ms**
- V2b: smem **43KB ✓**, **2 blk/SM ✓**, 正确性 5.7e-7/1.7e-7 ✓ -> 资源门槛全过, **速度门槛 (<=0.60ms) 未达**
- 根因: 阶段式结构 + 每 chunk 8 次块级 sync, 每线程每阶段工作量太小 -> ~66 cycles/FMA (1.5% 算力)
- 需结构性重写 (寄存器分块合并阶段 / 分块三角求解 / 或 K1+K2 双 kernel) 才能过门槛
- 产物 `artifacts/t03_chunked_v2_L16.cu`; 工作区 0 行改动

## Analyst 裁决 (V2b 后, 2026-09-22): **BLOCKED 归档** (timebox 到点)

- 判定: V2b 1.644ms > 门槛 0.60ms (差 2.7x), timebox 已用 ~0.9/1.0 天 -> 按预定规则 (第 5 条) 归档, 不超时投入
- 存留: 算法机器精度验证 (1e-16) + V1/V2a/V2b 正确性全过 (5.7e-7/1.7e-7) + 迭代 4.60 -> 3.29 -> 1.644ms;
  资源门槛全过 (smem 43KB, 2 blk/SM)
- 卡点: 结构性延迟暴露 (~66 cycles/FMA, 每 chunk 8 次块级 sync), 非单点 bug
- 恢复 (可选, 留给最终统一回顾 Q7): 最小版结构性重写 (寄存器分块合并 + 分块三角求解) ~0.3-0.5 天,
  条件 = 0.5 天内不过 0.60ms 即无条件归档
- 当前状态: 冻结, 工作区 0 行改动 (交付版仍为 4 文件 + 3 patch)

## Analyst 复议 (用户裁决, 2026-09-22): T03 恢复, 追加 0.5 天

- 用户决定再给 T03 0.5 天 -> 撤销 BLOCKED, 状态改回 RUNNING
- 批准: 最小版结构性重写 (寄存器分块合并 KKT/QK/T1/A + 分块三角求解; 备选 K1+K2 双 kernel + MB 级 scratch)
- 硬条件: **0.5 天到点无条件归档**; 集成门槛 <=0.60ms 不变; 正确性/资源门槛不变;
  ~0.25 天点若 cycles/FMA 无 ~2x 改善则提前停
- 过门槛后走标准集成验收 (同 session A/B >=2 轮 + pp512/4096/8192/32768/depth32k + tg128 + PPL)


---

## V3 (合并阶段) 结果: 提前停 (implementer, 2026-09-22)

- V3: barrier 从 ~10/chunk 合并到 4/chunk (+OS 内联) -> **1.844ms, ~80 cycles/FMA (比 V2b 的 ~66 更差)**
- 按 analyst 过程规则 4 (0.25 天点无 ~2x 改善即停) -> **停止, 未集成**
- **barrier 假设被否证**; 真瓶颈 = 每线程串行点积循环暴露的访存延迟 (循环不展开 + L2/smem 延迟),
  算力利用率 ~1.3%; 现状 kernel 用 warp 协作+归约所以更快
- 明确修炼点: 四个点积循环改 warp 协作 + float4 (预计 ~6x) + 前代循环展开 -> 预计 0.32-0.48ms (过门槛),
  需 ~0.3-0.4 天重写; 已超本轮余量
- 产物 `artifacts/t03_chunked_v3_merged.cu`


---

## 结案 (implementer, 2026-09-22): NO PATH FOUND

- 按用户"无限时间、确保不误判"的要求, 在真实规模 (H=48) 上把每条路线都实测到底:
  寄存器预取 ✗ / C=2 列每 warp (真实集成 -6%) ✗ / 提高占用 (MB=12, spill) ✗ / smem 双缓冲 staging ✗ /
  归约 ILP (机制成立但 warp 总数抵消) ✗ / expf 消融 (只值 14%) ✗
- **`__expf` 是唯一真实收益 (kernel -10%)**, 但按 gate "0.60-0.83 不集成" 未入库 (端到端 +0.7% < 测量噪声)
- chunked 路线量化否决: FLOP 多 1.4-2.4x, 效率需 30x 提升才过门槛 -> 不可行
- 判定: GDN kernel 在 H=48 下已接近实际下限 (~0.8ms), **-50% 目标不可达**; 工作区 0 净改动, PPL 4.3569 ✓
- 完整证据/产物见 RESULTS "T03 结案" 与 artifacts/t03_*

## Analyst 结案复核 (2026-09-22): CLOSED 接受

- 排查覆盖度认可: 真实规模 (H=48) 上对 寄存器预取 / smem 双缓冲 staging / C=2 列每 warp / MB=12 提占用 /
  归约 ILP / expf 消融 / chunked 全线实测, 每条都有数字与回退 -> "GDN ~0.8ms 接近实际下限" 成立
- 关键否证: 不是访存延迟 (prefetch/staging 反而更慢), 不是占用 (MB=12 spill);
  C=2 列 harness +8% 但真实集成 -6% (warp 总数减半抵消)
- chunked 量化否决: 算法正确 (1e-16/5e-7) 但 FLOP 多 1.4-2.4x + 效率天花板 -> 过门槛需 ~30x, 不可行
- `__expf`: kernel -10% 真实, 端到端 +0.7% < 门槛 -> 记档"低于门槛选项", 最终回顾可复议
- T03 关闭; 工作区回退到 4 文件交付态, DLL 重建 (SHA 见 ENVIRONMENT), PPL 4.3569
