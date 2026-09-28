import io, shutil, os

T = os.path.join(os.environ['TEMP'], 'v100')
A = r'D:\LLM\Backend\v100-collab\artifacts'
shutil.copy(os.path.join(T, 't03_chunked.cu'), os.path.join(A, 't03_chunked_v2_L16.cu'))
print('archived v2')

p = r'D:\LLM\Backend\v100-collab\RESULTS.md'
t = """
---

# T03 V2 进展报告 (0.5 天点): 资源门槛全过, 速度门槛未达 | implementer, 2026-09-22

产物: `artifacts/t03_chunked_v2_L16.cu` (V2b, L=16)。工作区仍 0 行改动。

## 迭代路径 (harness, D=128, H=32, T=512, per layer-ubatch)

| 版本 | 改动 | 时间 | smem | occupancy | 正确性 (out/state rel) |
|---|---|---:|---:|---|---|
| V1 | 初版 (L=32, COLS=32) | 4.60 ms | 73 KB | 1 blk/SM | 5.3e-7 / 1.7e-7 ✓ |
| V2a | sK/sM 加 pad (消 32-way bank conflict) + C 阶段全并行 | 3.29 ms | 77 KB | 1 blk/SM | 5.3e-7 / 1.7e-7 ✓ |
| **V2b** | **L=16** | **1.644 ms** | **43.0 KB ✓** | **2 blk/SM ✓** | **5.7e-7 / 1.7e-7 ✓** |
| (现状真 kernel) | - | 0.826 ms | - | - | - |

- 门槛检查: smem <=48KB ✓, >=2 block/SM ✓, 正确性 ✓ -> **但速度 1.644ms > 0.60ms 门槛 -> 按 gate #1 不集成**
- 相对 V1 已快 2.8x; 相对现状 kernel 仍慢 2.0x

## 仍慢的根因 (实测推算)

- L=16 时每 chunk 每 block 约 276K MAC / 256 线程 = ~1080 FMA/线程, 但实测每 chunk ~51us = ~71K cycles
  -> **~66 cycles/FMA**, 即算力只用了 1.5%
- 结构性原因: 每 chunk 8 个阶段各带 __syncthreads (L=16 时 32 chunk x 8 = 256 次 sync/block),
  每阶段每线程实际工作只有几十~几百条指令 -> 大量延迟暴露 (smem ~30 cycle + barrier),
  occupancy 只有 25% (16 warps/64) 不足以掩盖
- 结论: 不是单点 bug, 是 V1/V2 的**阶段式 + 块级同步**结构本身低效

## 下一版要改的 (若继续)

1. 减少阶段与同步: 把 KKT/QK/T1/A 合并为一次 2D 分块扫描 (寄存器分块, 每线程算一个 2x2/4x4 tile)
2. 前代求解改 **分块三角求解** (对角小块串行 + 块间 matmul 更新), 并让 W/U/R 三系统一起做
3. 状态更新/输出合并, 减少 smem 往返
4. 或改成 K1(chunk 并行, 大 matmul)/K2(chunk 串行, 轻量扫描) 两 kernel 结构, 用全局 scratch 换并行度
- 预期: 每次只做上述 1-2 项即可过 0.60ms; 但都需要一次结构性重写 (~0.3-0.5 天)

## timebox 记账

- 数学原型 ~0.4 天 + V1 ~0.35 天 + V2 ~0.15 天 = ~0.9 天 (硬上限 1 天)
- 按 analyst 规则: 这个 0.5 天点报告后, 若继续需要再投入 ~0.3-0.5 天做结构性重写;
  若不继续则 BLOCKED 归档 (算法已验证可行, 工程成本超预期)
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T03-gdn.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## V2 进展 (implementer, 2026-09-22, 0.5 天点)

- V1 -> V2a (pad 消 bank conflict + C 全并行) -> **V2b (L=16)**: 4.60 -> 3.29 -> **1.644 ms**
- V2b: smem **43KB ✓**, **2 blk/SM ✓**, 正确性 5.7e-7/1.7e-7 ✓ -> 资源门槛全过, **速度门槛 (<=0.60ms) 未达**
- 根因: 阶段式结构 + 每 chunk 8 次块级 sync, 每线程每阶段工作量太小 -> ~66 cycles/FMA (1.5% 算力)
- 需结构性重写 (寄存器分块合并阶段 / 分块三角求解 / 或 K1+K2 双 kernel) 才能过门槛
- 产物 `artifacts/t03_chunked_v2_L16.cu`; 工作区 0 行改动
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS updated')

p = r'D:\LLM\Backend\v100-collab\QUESTIONS.md'
t = """
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
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('QUESTIONS appended')
