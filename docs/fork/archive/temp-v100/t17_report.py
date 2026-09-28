import io, shutil

T = r'<TEMP>\v100'
A = r'D:\LLM\Backend\v100-collab\artifacts'
for f in ['t17_q8_bench.cu', 't17_q8_bench.exe', 't17_ks.csv']:
    try:
        shutil.copy(T + '\\' + f, A + '\\' + f)
    except Exception as e:
        print('skip', f, e)
print('artifacts copied')

t = """
---

# T17: Q8_0 权重反量化补测 (2026-09-23, implementer): **记录数字, 关闭 (不投入)**

方法: 新增微基准 `artifacts/t17_q8_bench.cu` (当前上游内核 verbatim vs 向量化候选) +
nsys pp512 (`-p 512 -n 0 -r 2`, 部署版 453E2911)。

## 1. 微基准 (512M 元素, 570MB 读 + 1074MB 写)

| 内核 | 时间 | 带宽 |
|---|---:|---:|
| 当前上游 (`dequantize_block_q8_0_f16`, warp/2048, smem 中转) | 2.143 ms | **767.2 GB/s** |
| 向量化候选 (warp/256, 每 lane 8 连续 int8 -> uint4) | 2.455 ms | 669.8 GB/s (更慢) |

- 正确性: 候选与当前内核**逐位相同** (0 / 536870912 mismatch) -> 数值无障碍, 但**无收益**
- 原因: Q8_0 块 = 34 字节 (2 字节对齐), 错位布局下 "smem 聚合读 + 按 char2 展开" 比直接向量化读更优;
  当前内核已在 767 GB/s = Q6_K/Q5_K 参考线 (707-825) 同档

## 2. nsys pp512 占比 (2 reps, kernel 合计 1551.7 ms)

| kernel | 时间 | 占比 |
|---|---:|---:|
| `dequantize_block_q8_0_f16<(bool)0>` | 18.84 ms (501 次) | **1.2% of kernel time** |
| (对照) dequant q6_K vec / q5_K vec | 182.3 / 83.4 ms | 11.8% / 5.4% |
| Q8_0 在 dequant 内部占比 | 18.84/284.6 = 6.6% | 低于其权重占比 13% (即相对更快) |

- 反推实模型有效带宽: Q8_0 权重 2.86GB 读 + 5.38GB 写 = 8.24GB/ubatch, 单 forward ~9.4ms -> **~877 GB/s**
  (>= 实测可用 825-850) -> **已无空间**
- 上限估算: 即使 Q8_0 dequant 完全免费也只有 +1.2% (单 forward); 现实收益 ~0.2-0.3% -> 低于 +0.4% 门槛

## 3. 判定 (spec 第 3 条)

- Q8_0 路径**不偏慢** (767 GB/s >= 参考线; 实模型折算 877 GB/s), 向量化候选更慢 ->
  **不实施, 记录数字关闭**; 不新增 patch, 工作区/交付态不变 (仍 5 文件 + 4 patch, DLL 453E2911)
"""
io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T17-q8-dequant.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## Result (2026-09-23, implementer): **CLOSED (不投入)**

- 微基准 (`artifacts/t17_q8_bench.cu`): 当前内核 **767.2 GB/s**, 向量化候选 669.8 GB/s (逐位相同但更慢;
  34 字节错位块布局下 smem 中转让聚合读更优)
- nsys pp512: Q8_0 dequant = 18.84ms / 1551.7ms = **1.2%** of kernel time; 实模型折算 **~877 GB/s** (已到带宽上限)
- 上限 +1.2% (全免费), 现实 ~+0.2-0.3% < 门槛 +0.4% -> 按 spec 第 3 条关闭, 不新增 patch
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T17 updated')

q = """
---

## 2026-09-23 | T17 结案: CLOSED (不投入) | implementer

- 微基准: 当前 Q8_0 内核 **767.2 GB/s** (Q6_K/Q5_K 参考线 707-825 同档); 我写的向量化候选 669.8 GB/s
  (逐位相同但更慢 -> Q8_0 的 34 字节错位块布局下, 上游的 smem 中转是更优解)
- nsys pp512: Q8_0 dequant 18.84ms / 1551.7ms = **1.2%**; 实模型折算 ~877 GB/s (>= 可用带宽 825-850)
- 判定: 不偏慢, 空间 <= ~0.3% < 门槛 +0.4% -> **记录数字关闭** (spec 第 3 条); 交付态不变
- 产物: `artifacts/t17_q8_bench.cu` (+exe, ks.csv)

队列剩余: **T18 Stage 0** (可静音做, 纯 CPU/代码)。
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(q)
print('QUESTIONS appended')

p = r'D:\LLM\Backend\v100-collab\BOARD.md'
s = io.open(p, encoding='utf-8').read()
old = "| T17 | Q8_0 权重反量化补测 | **APPROVED (Q14)** | spec: TASKS/T17-q8-dequant.md; 先测吞吐; 若 >=+0.4% 则按 A1 同法向量化 (逐位相同) |"
new = "| T17 | Q8_0 权重反量化补测 | **DONE -> CLOSED (不投入)** | 当前内核 767 GB/s (微基准) / ~877 GB/s (实模型折算), pp512 占比 1.2% -> 空间 <=0.3% < 门槛 +0.4%; 向量化候选逐位相同但更慢; `artifacts/t17_q8_bench.cu` |"
assert old in s
s = s.replace(old, new)
oldq = "| 2 | **T17 Q8_0 反量化补测** (spec: TASKS/T17-q8-dequant.md) | 上限 +0.5-1% | 0.5h (若实施 +1h) | **APPROVED (Q14)** |"
newq = "| 2 | ~~T17 Q8_0 反量化补测~~ | 实测无可利用空间 (767 GB/s 已同档) | 0.5h | **DONE (CLOSED 不投入)** |"
assert oldq in s
s = s.replace(oldq, newq)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('BOARD updated')
