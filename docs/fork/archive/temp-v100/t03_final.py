import io, shutil, os

T = os.path.join(os.environ['TEMP'], 'v100')
A = r'D:\LLM\Backend\v100-collab\artifacts'
for f in ['t03_existing.cu', 't03_variant.cu', 't03_sweep.cu', 't03_stage.cu', 't03_sweep2.exe']:
    src = os.path.join(T, f)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(A, f))
print('archived harnesses')

p = r'D:\LLM\Backend\v100-collab\RESULTS.md'
t = """
---

# T03 结案: GDN kernel 加速路线全部否证, 无可行路径 (implements, 2026-09-22) | NO PATH FOUND

工作区已回退到交付态 (4 文件, 无本任务净改动), PPL 4.3569 复核通过。
产物: `artifacts/t03_existing.cu` (现状 kernel 复刻 + 分段计时), `t03_variant.cu` (C 列/warp),
`t03_sweep.cu` (occupancy 扫描 + expf 消融), `t03_stage.cu` (smem 双缓冲), `t03_chunked_v1/v2_L16/v3_merged.cu`。

## 全部尝试与实测 (harness H=48 真实 head 数; 现状 kernel 实测 0.814 ms/layer-ubatch -> harness 复刻 0.922)

| 尝试 | 机制假设 | 实测结果 | 判定 |
|---|---|---|---|
| 寄存器预取 (t+1 加载提前) | 全局加载延迟暴露 | ~1.0x (0.88/0.83) | ✗ 假设否证 |
| **C=2 列/warp** | 归约与加载延迟按列摊薄 | harness 0.85-0.90; **真实模型集成后 867us vs 814us = +6%** | ✗ 无常驻收益 (warp 总数减半抵消) |
| C=2 + MB=12 (48 warp/SM) | 占用不足 | 1.01ms (寄存器压到 40, spill) | ✗ 假设否证 |
| block 级 smem 双缓冲 staging | 用 smem 替代寄存器预取 | 1.15ms+ (且实现有 bug) | ✗ 更慢 |
| 分段 clock64 计时 | 定位瓶颈 | loads 4%, **kv_local+reduce1+expf 69%**, 其余 26% | 诊断有效 |
| warp 归约微基准 | 归约延迟 vs 吞吐 | 单次 144.5 cycles 延迟, **ILP 可完全掩盖** (C=8 -> 18 cycles/次) | 机制成立但需 ILP |
| **`__expf` 替换 `expf`** | 指令数 (expf ~20 条/次) | **0.922 -> 0.838 (-10%)**, state_rel 7.5e-8 -> 1.5e-7 | ✓ 真实但小 |
| 无 expf 消融 (数值故意错) | expf 的总代价上限 | 0.792 (-14%) | expf 全部价值只有 14% |

## 为什么没有 2x (被实测限定的结论)

1. **不是内存延迟瓶颈**: 预取和 smem staging 两条独立路径都失败 -> 40 warp/SM 的驻留已经掩盖了访存延迟
2. **不是占用瓶颈**: 强行上 48 warp/SM (MB=12) 反而 -10% (寄存器 spill)
3. **不是归约延迟瓶颈**: C=2 把每列归约减半, 但 warp 总数也减半, 真实模型实测净 -6%
4. **指令数上限**: expf 只占 14%, 去掉后仍有 0.79ms -> 剩余指令 (FMA/归约/寻址/存储) 的下限就在 0.75ms 级
5. 现状 kernel 在 H=48 下已接近"指令+延迟平衡"的实际下限 (~0.8ms); **analyst 的 -50% (0.41ms) 不可达**

## chunked 路线 (V1/V2b/V3) 的量化否决

- 算法正确 (numpy 1e-16 / CUDA 5e-7), 但 FLOP 比串行多 1.4-2.4x (列切分重复 + KKT/QK/T1/A 开销)
- 最好 V2b = 1.644ms (harness) = 现状 2x 慢; 要过 0.60 需把效率从 ~1.5% 提到 >25% 峰值 (30x) -> 不现实
- 结论: chunked 只是把"每 token 的串行点积"换成"每 chunk 的串行点积", 没有碰到真瓶颈

## 保留价值

- `__expf` 是一个**真实可行但低于集成门槛**的选项 (kernel -10%, 端到端约 +0.7%, state_rel 1.5e-7 远在 PPL 门槛内);
  按 analyst gate "0.60-0.83 不集成" 未集成, 若未来放宽门槛可一行启用 (记录在此)
- 三份 harness (现状 kernel 复刻 / C 列变体 / occupancy 扫描) 可复用, 分段 clock64 计时法可复用于其他 kernel 诊断
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T03-gdn.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## 结案 (implementer, 2026-09-22): NO PATH FOUND

- 按用户"无限时间、确保不误判"的要求, 在真实规模 (H=48) 上把每条路线都实测到底:
  寄存器预取 ✗ / C=2 列每 warp (真实集成 -6%) ✗ / 提高占用 (MB=12, spill) ✗ / smem 双缓冲 staging ✗ /
  归约 ILP (机制成立但 warp 总数抵消) ✗ / expf 消融 (只值 14%) ✗
- **`__expf` 是唯一真实收益 (kernel -10%)**, 但按 gate "0.60-0.83 不集成" 未入库 (端到端 +0.7% < 测量噪声)
- chunked 路线量化否决: FLOP 多 1.4-2.4x, 效率需 30x 提升才过门槛 -> 不可行
- 判定: GDN kernel 在 H=48 下已接近实际下限 (~0.8ms), **-50% 目标不可达**; 工作区 0 净改动, PPL 4.3569 ✓
- 完整证据/产物见 RESULTS "T03 结案" 与 artifacts/t03_*
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS updated')

p = r'D:\LLM\Backend\v100-collab\QUESTIONS.md'
t = """
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
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('QUESTIONS appended')
