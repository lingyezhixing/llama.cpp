import io

t = """
---

# T18 Stage 1: 占用 vs L1 的双瓶颈实测 (2026-09-23, implementer) -> **1.12x (门槛 1.20x 未达)**

基线 (harness, 生产配置): ncols=64 + PB=2/grid=384, l=35072 -> **12.660 ms/launch**

## 尝试矩阵 (全部 harness 实测, 同一形状/时钟)

| 变体 | smem | 占用 | 时间 (l=35072, 最优 grid) | 相对基线 |
|---|---|---:|---:|---:|
| ncols=64, grid=384 (基线) | 67584 | 1 CTA / 4 warps | 12.660 | 1.000 |
| ncols=64, nthreads=256 | - | - | 编译失败 (np=8 不受支持) | - |
| **ncols=32, grid=768 (PB=2)** | **35072** | **2 CTA / 8 warps** | **11.297** | **1.120x** |
| ncols=32, grid=1152 / 1536 | 35072 | 2 CTA | 11.821 / 11.424 | 1.071 / 1.109 |
| ncols=32 + Q_in_reg=true | 34816 | 2 CTA | 45.751 (REG 255 + STACK 472B spill) | 0.277 |
| ncols=16 | - | - | NO_DEVICE_CODE (Volta 内核限制 ncols>=32) | - |

## 随 KV 长度的收益趋势 (同一 harness)

| l (n_kv) | ncols=64 | ncols=32 | speedup |
|---:|---:|---:|---:|
| 12288 | 4.802 | 3.939 | **1.219x** |
| 35072 | 12.660 | 11.297 | **1.120x** |
| 70000 | 25.512 | 23.093 | **1.105x** |
| 100000 | 36.780 | 33.390 | **1.102x** |

-> 收敛于 **~1.10x**; 长文 (128k 点, attention 占比 ~65%) 折算 e2e **~+6.3%**;
depth32k (32%) ~+3.2%; pp32768 (18.5%) ~+1.85% -> **Stage 2 三道门槛 (+10%/+4%/+2%) 都会差一点**

## ncu 诊断链 (关键)

| | ncols=64 (1 CTA) | ncols=32 (2 CTA) |
|---|---|---|
| Occupancy (theoretical=achieved) | 6.25% (4 warps) | **12.5% (8 warps)** |
| Compute (SM) Throughput | 26.5% | 40.0% |
| **L1/TEX Cache Throughput** | 47.6% | **74.5% (新瓶颈)** |
| Mem Busy / DRAM / L2 | - / 2.9% / - | 71.3% / 8.5% / 96% |
| No Eligible / Issued per scheduler | 78.5% / 0.22 | 70.1% / 0.30 |
| ncu "Est. Speedup" (scheduler) | 61.3% | 28.7% |

- 结论: ncols=32 把内核从**延迟受限**推到**L1/shared 数据通路受限** (74.5%, 且 DRAM/L2 远未饱和)
- 下一步本应减少 LDS 操作数流量, 但两条路都被硬件挡死:
  1. **Q_in_reg** (把 Q 操作数常驻寄存器, 消掉每 KV chunk 重读 smem 的主项): ncols=32 下仍需 255 regs + 472B/thread spill
     (寄存器墙 255/thread), 实测 4x 变慢
  2. **降 np** (nwarps*cols_per_warp/ncols; 现在 np=4, 每 KV chunk 有跨 warp 归并): np=1 需要 ncols = nwarps*32
     -> ncols=128 -> Q staging 67.5KB (smem 爆) 或 nwarps=1 (单 warp/CTA 无并行) -> 不可行
- Volta 的 mma.m8n8k4 操作数流量与 tile 布局 (cols_per_warp=32 固定) 决定了这个上限; 突破需要重写 tile/warp 布局
  (T_C_KQ/T_C_VKQ 族) = 多日 + 高风险, 超出 T18 3 天 timebox 的合理范围

## 判定 (按 spec 门槛)

- Stage 1 gate: **1.20x -> 未达 (1.12x)**; 1.5 天 checkpoint (1.10x) 刚过
- Stage 2 预估: 三道门槛都差一点 (+6.3% vs +10%; +3.2% vs +4%; +1.85% vs +2%)
- **建议: T18 归档 (不集成)**; 1.12x 变体保留在 harness/artifacts 供未来参考;
  若 analyst/user 认为"均匀 +1.10~1.22x"值得入库, 剩余工作 = dispatch 覆盖 (强制 Volta D256 走 ncols=32) + 完整模型验收 (~1.5h)
- 副产品 (可复用): harness (`artifacts/t18_fa_harness.cu`, 直接 include 源头文件), ncu 日志 2 份, 5 个变体 exe
"""
io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

q = """
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
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(q)
print('QUESTIONS appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T18-fa-mma-rewrite.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## Stage 1 Result (2026-09-23): **1.12x (gate 1.20x 未达)**

- 最好变体: **ncols=32 瓦片 (smem 35072 -> 2 CTA/SM = 8 warps) + PB=2/grid=768** = 11.297 ms (基线 12.660) = **1.12x**;
  l=12288: 1.22x; l=100000: 1.10x (收敛)
- ncu: ncols=64 延迟受限 (occ 6.25%, Est.Speedup 61%) -> ncols=32 后 **L1/shared 通路受限 (74.5%)**, DRAM 8.5%
- 被硬件挡死: Q_in_reg (255 regs + 472B spill, 4x 慢); ncols=16 (内核限定 >=32); nthreads=256 (np=8 不支持); np=1 (需 ncols=128)
- Stage 2 预估: +6.3% / +3.2% / +1.85% (三道门槛都差一点) -> 建议归档; 见 QUESTIONS 的裁决请求 (A/B/C)
- 产物: harness + 5 变体 exe + ncu 日志 2 份 (artifacts/)
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T18 updated')
