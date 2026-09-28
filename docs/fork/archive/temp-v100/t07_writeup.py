import io

# ---------- RESULTS.md ----------
p = r'D:\LLM\Backend\v100-collab\RESULTS.md'
t = """
---

# T07 剩余项: rms_norm 向量化 | implementer, 2026-09-22 | **REJECTED (kernel 级负收益)**

实现: `rms_norm_f32_vec4<block_size, do_multiply, do_add>` (float4 两趟 + 对齐/整除条件 + 标量回退),
覆盖 plain / mul / mul+add 三条路径。patch 已归档 `artifacts/t07_rms_norm_vec4_REJECTED.patch`, **未入库 (已回退)**。

## Kernel 级 (nsys, pp4096, 同命令同口径, 2 个 forward 合计)

| 变体 (调用/ubatch) | before | after (vec4) | Δ |
|---|---:|---:|---:|
| `rms_norm_f32<256,true>` (mul, 80) | 54.86us x 1280 = 70.23ms | 58.46us x 1280 = 74.83ms | **+6.6%** |
| `rms_norm_f32<1024,true>` (mul, 129) | 33.89us x 2064 = 69.94ms | 37.44us x 2064 = 77.28ms | **+10.5%** |
| `rms_norm_f32<256,false>` (32) | 22.55us x 1536 = 34.64ms | 21.32us x 1536 = 32.74ms | -5.5% |
| **合计** | **174.81 ms** | **184.85 ms** | **+5.7% (更慢)** |

## 端到端 A/B (交替 4 轮, `-r 3`, ub512, 无 tg)

| 轮 | DLL | pp512 | pp4096 | pp8192 |
|---|---|---:|---:|---:|
| 1 | A (vec4) | 965.22 | 941.05 | 915.39 |
| 2 | B (base) | 961.20 | 935.82 | 913.10 |
| 3 | B (base) | 955.22 | 932.39 | 909.10 |
| 4 | A (vec4) | 958.49 | 933.07 | 908.60 |
| 均值 A | | 961.86 | 937.06 | 911.99 |
| 均值 B | | 958.21 | 934.11 | 911.10 |
| Δ | | +0.38% | +0.32% | +0.10% |

- 轮间漂移 (同 DLL 两轮) 达 0.4-0.75% -> 上表 Δ **在噪声地板内**, 与 kernel 级 (+5.7% 更慢 = 端到端 -0.12%) 方向矛盾
- 按 PROTOCOL "单次测量 +/-0.5% 不能作判据" + T07 自身验收条件 (kernel 级为准) -> **拒绝**

## PPL (门槛 4.3572 +/- 0.013)

- A (vec4): **4.3563**; B (4 文件基线): 4.3569 -> Δ 0.0006, 门槛内 -> 纯性能否决, 无质量问题

## 根因 (为什么没有收益)

1. `rms_norm_f32<1024,true>` (ncols=5120, 130 次/ubatch) **已达 DRAM 带宽极限**: 单次 512 行 x 5120 列 x 3 遍
   (2 读 1 写) = 31.5MB / 33.89us = **930 GB/s** -> 无空间, 向量化只带来负载不均 (ncols4=1280 在 block=1024
   下 256 线程要做第 2 轮)
2. `rms_norm_f32<256,*>` (ncols=128/256, 176 次/ubatch) 是**延迟受限** (约 57 GB/s): 单次 22-58us 中大头是
   固定开销/归约/DRAM 延迟, 不是指令条数 -> float4 减少指令无用
3. 上限估算: 即使做完美也只有 4-9ms/forward = **0.1-0.2% 端到端**, 低于测量噪声地板 -> 不可验证

## 结论

- rms_norm 不再投入 (T07 全部关闭: silu 已入库 +5%/kernel, rms_norm 拒绝)
- 小 kernel 的延迟受限问题属于 **T05 剩余项 (减少调用次数/融合)** 的范畴, 向量化解决不了
- 交付状态已回退到 4 文件, DLL 重新构建部署, 体检: PPL 4.3569 / pp512 962.1 / pp4096 937.4 /
  pp8192 912.7 / tg128 26.68
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('RESULTS.md: T07 rejected appended')

# ---------- TASKS/T07 ----------
p = r'D:\LLM\Backend\v100-collab\TASKS\T07-elementwise.md'
s = io.open(p, encoding='utf-8').read()
s = s.replace('状态: PARTIAL (第 1 项 silu 已入库; 剩余项可选)',
              '状态: DONE (silu 已入库; rms_norm 向量化 = REJECTED, kernel 级 +5.7%, 见下)')
s += """
---

## Result 补充 (implementer, 2026-09-22): rms_norm 向量化 = REJECTED

- Kernel 级 (nsys): 合计 174.81 -> **184.85 ms (+5.7% 更慢)**; mul 变体 +6.6%/+10.5%, 仅 plain 变体 -5.5%
- 端到端交替 4 轮: pp512 +0.38% / pp4096 +0.32% / pp8192 +0.10%, 但轮间漂移 0.4-0.75% -> 噪声级,
  与 kernel 级矛盾 -> 按验收规则拒绝
- PPL: 4.3563 (门槛内, 无质量问题)
- 根因: <1024> 变体已 930 GB/s = DRAM 极限; <256> 变体延迟受限 (57 GB/s, 固定开销主导);
  理论上限仅 0.1-0.2% 端到端, 低于噪声地板
- patch 归档 `artifacts/t07_rms_norm_vec4_REJECTED.patch`, 工作区已回退 (仍为 4 文件)
- **T07 全部关闭**
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T07 updated')

# ---------- BOARD ----------
p = r'D:\LLM\Backend\v100-collab\BOARD.md'
s = io.open(p, encoding='utf-8').read()
old = "| T07 | elementwise 向量化 | **PARTIAL (silu 已入)** | silu kernel -5%, 端到端噪声内; 剩余 rms_norm +0.4% 已批准 |"
new = "| T07 | elementwise 向量化 | **DONE (关闭)** | silu kernel -5.1% 已入库; **rms_norm 向量化 REJECTED** (kernel 级 +5.7%; <1024> 已 DRAM 极限 930GB/s, 理论上限 0.1-0.2% 低于噪声地板) |"
assert old in s, 't07row'
s = s.replace(old, new, 1)

old2 = "| 2 | T07 剩余: rms_norm 向量化 | prefill +0.4% | 小时级 | APPROVED (与 T10 并行) |"
new2 = "| 2 | ~~T07 剩余: rms_norm 向量化~~ | ~~prefill +0.4%~~ | 小时级 | **DONE = REJECTED** (kernel 级 +5.7% 更慢, 见 TASKS/T07 与 RESULTS) |"
assert old2 in s, 't07q'
s = s.replace(old2, new2, 1)

old3 = "| T05 剩余: quantize_q8_1 融合 + 小 kernel 合并 | decode +5-8% (MTP 轮同受益) | ~1 天 | OPTIONAL |"
new3 = "| T05 剩余: quantize_q8_1 融合 + 小 kernel 合并 (含 rms_norm 等小核的延迟受限问题) | decode +5-8% (MTP 轮同受益) | ~1 天 | OPTIONAL |"
assert old3 in s, 't05q'
s = s.replace(old3, new3, 1)

old4 = "6. 已完成: T01, T03, T05 profile, T07 第 1 项, T02/T08/T09 (全部否决)"
new4 = "6. 已完成: T01, T03, T05 profile, **T07 (全部关闭: silu 入库 + rms_norm 被否)**, T10 侦察, T02/T08/T09 (全部否决)\n7. 待办: T03 chunked (APPROVED, 1.5 天) -> T10 近路 KV-split (Q6 待裁) -> T05 剩余 (OPTIONAL) -> T04 (用户口径)"
assert old4 in s, 'done'
s = s.replace(old4, new4, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('BOARD updated (4 edits)')

# ---------- ENVIRONMENT ----------
p = r'D:\LLM\Backend\v100-collab\ENVIRONMENT.md'
s = io.open(p, encoding='utf-8').read()
old = "| 部署 DLL | `D:\\LLM\\Backend\\llama.cpp-my\\ggml-cuda.dll` SHA256 `6D0AC881...454CD1` (== build 输出) |"
new = "| 部署 DLL | `D:\\LLM\\Backend\\llama.cpp-my\\ggml-cuda.dll` SHA256 `B60722EB...50921E` (T07 rms_norm 实验回退后重建; 同源码重建仅构建元数据不同) |"
assert old in s, 'env-dll'
s = s.replace(old, new, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('ENVIRONMENT updated')
