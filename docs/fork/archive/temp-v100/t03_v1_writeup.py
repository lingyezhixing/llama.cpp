import io, shutil, os

T = os.path.join(os.environ['TEMP'], 'v100')
A = r'D:\LLM\Backend\v100-collab\artifacts'
shutil.copy(os.path.join(T, 't03_chunked.cu'), os.path.join(A, 't03_chunked_v1.cu'))
shutil.copy(os.path.join(T, 't03_chunked.exe'), os.path.join(A, 't03_chunked_v1.exe'))
shutil.copy(os.path.join(T, 't03_check_state.py'), os.path.join(A, 't03_check_state.py'))
print('archived prototype')

p = r'D:\LLM\Backend\v100-collab\RESULTS.md'
t = """
---

# T03 CUDA 原型 V1: chunked GDN kernel 正确性达成, 性能未达标 | implementer, 2026-09-22

产物: `artifacts/t03_chunked_v1.cu` (harness: 参考实现 + chunked V1 + 中间量对拍),
`artifacts/t03_check_state.py` (numpy 逐阶段校验)。工作区未改 (llama.cpp 0 行)。

## 正确性 (harness, D=128, H=32, L=32, COLS=32)

与"逐 token 串行参考"对比 (纯 fp32):

| tokens | chunks | out rel err | state rel err |
|---:|---:|---:|---:|
| 32 | 1 | 5.7e-7 | 1.4e-7 |
| 64 | 2 | 4.9e-7 | 1.5e-7 |
| 128 | 4 | 4.8e-7 | 1.8e-7 |
| 256 | 8 | 5.0e-7 | 1.8e-7 |
| 512 | 16 | **5.3e-7** | **1.7e-7** |

- 16 个连续 chunk 后仍只有 ~1e-7 相对偏差 -> **算法与实现正确, 三个 checkpoint 在实际 CUDA 代码上通过**
- 逐阶段 numpy 对拍 (W/U/R/QK/T1/OS/C, Ac/Bt) 全部到 fp32 精度; 过程中修掉两个真 bug:
  1. C 的内层求和缺 `m <= t` 掩码 (tril(QK)) -> 12% 输出误差
  2. **U 的右端项写成 `beta_t*Bt_t*v` (多乘一个 beta)**: 正确形式是 `U = A_tri @ (diag(1/Ac) V)`,
     即前代右端项 `v_t/Ac_t` (不是 `Bt_t*v_t`); 该错误使状态 32% 偏差

## 性能 (per layer-ubatch, H=32, T=512)

| 实现 | 时间 | 说明 |
|---|---:|---|
| harness 串行参考 (1 线程/列, 1 block/head) | 1.86 ms | 弱并行, 仅作正确性基线 |
| **chunked V1** | **4.60 ms** | 0.40x vs 上面那个参考 |
| **真 kernel (现状, 实测)** | **0.826 ms** | llama.cpp 现有实现 (grid H x n_seqs x 32) |

- V1 比真 kernel **慢 5.6x** -> **性能不达标, 未集成**
- 已定位的瓶颈 (下一版必修):
  1. **smem bank conflict**: `sM[jj*D + m]` / `sK[s*D + i]` 在连续线程上 stride=128 -> 32-way 冲突
     (T1/QK 两个主循环) -> 加 pad (D+1) 即可
  2. **occupancy**: 73KB smem -> 1 block/SM (8 warps = 12.5%); L=16 + pad 后约 43KB -> 2 blocks/SM
  3. 串行前代 + C 阶段只用 32/256 线程 -> 需并行化 (或 warp 级)
  4. 每 chunk 8 次 __syncthreads x 16 chunks; 可合并阶段
- 参考成本: 每 chunk 每 block ~0.56M MAC (COLS=32 时分块间还会重复 KKT/solve/QK), 理论上限远高于现状

## 结论与下一步

- 算法侧完成 (checkpoint 全过); 工程侧 V1 正确但慢, 需要一次性能重写 (换 2D tiling / 寄存器分块 /
  pad / 更高 occupancy), 预计还需要一个 ~半天到一天的 block
- timebox 记账: 数学原型 ~0.4 天 + CUDA V1 ~0.35 天; 性能 pass 未开始 -> 是否继续由 analyst 裁决
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('RESULTS.md appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T03-gdn.md'
s = io.open(p, encoding='utf-8').read()
s += """

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
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T03 updated')

p = r'D:\LLM\Backend\v100-collab\QUESTIONS.md'
t = """
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
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('QUESTIONS.md appended')
