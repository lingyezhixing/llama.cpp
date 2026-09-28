import io

# ---------- RESULTS.md ----------
p = r'D:\LLM\Backend\v100-collab\RESULTS.md'
t = """
---

# T10 近路: ub512 下启用 FA KV-split (stream-K 启发式) | implementer, 2026-09-22 | **REJECTED (BLOCKED)**

## 改动 (最小化, 已回退)

`fattn-common.cuh` `should_use_stream_k` +13 行:
- 新增 env 开关 `GGML_CUDA_FATTN_STREAM_K` (0=禁用/1=强制/未设=自动), 便于即时回退
- 新增条件: `NVIDIA && ntiles_dst > max_blocks && tiles_efficiency_percent < 96` -> 启用 stream-K
  (ub512 + depth32k: ntiles_dst=192, max_blocks=80, 效率 80% -> 触发; ub2048 的 96% 不受影响)
- 机制: grid 192 CTA/3 波 -> 80 CTA/1 波 + `flash_attn_stream_k_fixup_general<256,32,2>` 归并

## nsys 验证 (pp4096@depth32k, 与 T10 同口径)

| 项 | before | after |
|---|---:|---:|
| FA grid | (192,1,1) | **(80,1,1)** |
| fixup kernel | 无 | `stream_k_fixup_general<256,32,2>` 1280 x 23.55us = 30.1ms |
| eval attention | 1899.4 ms | **1792.7 ms (-5.6%)** |
| eval attention TF/s | 29.5 | **31.3** |

- 波次效率模型 (80% -> 100% 应 -20%) **高估**: 实测只 -5.6%。
  原因: stream-K 下每 block 串行做 ~2.4 个 tile + tile 接缝的 needs_fixup 归并, 单 block 效率下降,
  抵消了大部分打尾收益

## 端到端 A/B (交替换 DLL 2 轮, gate #1)

| 指标 | A 均值 (R1/R4) | B 均值 (R2/R3) | Δ |
|---|---:|---:|---:|
| pp4096 @d32768 | 666.35 (665.89/666.81) | 653.33 (654.28/652.38) | **+2.0%** |
| pp32768 | 788.67 (788.47/788.86) | 780.10 (781.82/778.38) | **+1.1%** |
| pp512 | 946.29 | 951.47 | -0.55% |
| pp4096 | 930.96 | 928.54 | +0.26% |
| pp8192 | 908.54 | 904.47 | +0.45% |
| tg128 | 25.15 | 24.89 | +1.0% (噪声 ±0.9) |
| ub2048 pp4096 (gate #5) | 1152.46 | 1153.39 | -0.1% (无回退) |

## 验收判定 (analyst 门槛)

| gate | 要求 | 实测 | 判定 |
|---|---|---|---|
| #1 depth32k | >= +3% | +2.0% | **FAIL** |
| #1 pp32768 | >= +2% | +1.1% | **FAIL** |
| #2 pp512/4096/8192 | 不回退 >0.5% | -0.55%/+0.26%/+0.45% | 噪声内 (无收益) |
| #2 tg128 | 不回退 | +1.0% (与机器漂移同量级) | PASS |
| #3 PPL | 4.3572 +/- 0.013 | A 4.3568 / base 4.3569 | PASS (数值机制安全) |
| #4 nsys TF/s 前后 | 必须给 | 29.5 -> 31.3, grid 192->80, fixup 出现 | PASS |
| #5 ub2048 | 不回退 | -0.1% | PASS |
| #6 最小改动/env 开关/patch | - | +13 行, `GGML_CUDA_FATTN_STREAM_K`, 已归档 | PASS |

## 结论 (按 Q6 规则)

- **A 失败**: 效果真实但只达门槛的 ~1/3~2/3 (attention -5.6%, depth32k +2.0%, pp32768 +1.1%)
  -> 不硬凑, 已回退 (工作区 0 行改动), patch 归档 `artifacts/t11_fattn_kvsplit_REJECTED.patch`
- 保留的事实: **ub512 的长上下文 attention 确实受 grid/波次限制**, 但 stream-K 这一条路只能回收 ~1/4;
  ub2048 的 39.9 TF/s 主要来自"每 launch 4x 并行度 + 4x 更少 launch 次数"的整体效应, 不是单纯打尾
- 机器状态注记: 本次 A/B 期间全指标比历史基线低 ~1% (pp512 ~951 vs 958-963; tg128 25.1-25.7 vs 26.6),
  故所有比较都用同 session 的交替 A/B, 绝对值不能与历史数字混用
- 下一步: 按 Q6 = **转 T03 chunked** (prefill 全局 +3-4%, timebox 1.5 天)
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('RESULTS.md: T11 appended')

# ---------- TASKS/T10 ----------
p = r'D:\LLM\Backend\v100-collab\TASKS\T10-attention-recon.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## 近路 Result (implementer, 2026-09-22): ub512 KV-split (stream-K) = REJECTED

- 改动 +13 行 (`should_use_stream_k`, env `GGML_CUDA_FATTN_STREAM_K`), nsys 机制验证通过:
  grid 192 -> 80, fixup kernel 出现, eval attention 1899.4 -> 1792.7ms (-5.6%, 29.5 -> 31.3 TF/s)
- 端到端 A/B 2 轮交替: depth32k **+2.0%** (gate >=3% FAIL), pp32768 **+1.1%** (gate >=2% FAIL);
  pp512/4096/8192 噪声内, tg128/ub2048 无回退, PPL 4.3568 安全
- 判定: 不达门槛 -> 已回退, patch 归档 `artifacts/t11_fattn_kvsplit_REJECTED.patch`, 转 T03
- 学习: 波次效率模型高估 (每 block 2.4 tile + seam 归并抵消大半); ub2048 的优势主要是并行度总量与
  更少 launch, 不是单纯打尾
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T10 updated')

# ---------- BOARD ----------
p = r'D:\LLM\Backend\v100-collab\BOARD.md'
s = io.open(p, encoding='utf-8').read()
old = "| 0 | **T10 近路: ub512 下启用 FA KV-split** (parallel_blocks) | depth32k +5-8% / pp32768 +3-5% / pp8192 +1.2% / pp4096 +0.7% | 0.5-1 天 | **APPROVED (先做, Q6 已答; 验收门槛见 QUESTIONS)** |"
new = "| 0 | ~~T10 近路: ub512 FA KV-split~~ | ~~depth32k +5-8%~~ | 0.5-1 天 | **DONE = REJECTED (不达门槛)**: attention -5.6%, depth32k +2.0% (<3%), pp32768 +1.1% (<2%); 已回退 |"
assert old in s, 'q0'
s = s.replace(old, new, 1)
old2 = "| FA split-D/N32 移植 (T10) | 不移植: ub2048 同 kernel 已 39.9 TF/s (超 1Cat 29-38); 真问题是 ub512 grid 调度 |"
new2 = old2 + "\n| ub512 FA KV-split (stream-K 启发式, T10 近路) | attention -5.6% (1899->1793ms, 29.5->31.3 TF/s), 但端到端只 +2.0% depth32k / +1.1% pp32768 (门槛 3%/2%) -> 已回退; 波次模型高估 |"
assert old2 in s, 'retry'
s = s.replace(old2, new2, 1)
old3 = "   - 长上下文 (KV-split 后预期): depth32k 649 -> ~680-700; pp32768 775 -> ~800-815"
new3 = "   - 长上下文: KV-split 已否决 -> depth32k/pp32768 维持 649/775 基线水平 (2026-09-22 复测: 653/780, 机器当轮低 ~1%)"
assert old3 in s, 'll'
s = s.replace(old3, new3, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('BOARD updated (3 edits)')

# ---------- QUESTIONS ----------
p = r'D:\LLM\Backend\v100-collab\QUESTIONS.md'
t = """
---

## 2026-09-22 | T10 近路 (KV-split) 验收失败 | implementer

执行 Q6 的 A 并按你的 6 条门槛严格验收 -> **FAIL, 已回退, 按规则转 T03 chunked**。
数据见 RESULTS.md "T10 近路" 章节 (含 nsys 前后对比 + 2 轮交替 A/B 表)。

速览: nsys 机制成立 (grid 192->80, fixup 出现, attention 1899->1793ms = -5.6%, 29.5->31.3 TF/s),
但端到端 depth32k **+2.0%** (门槛 3%) / pp32768 **+1.1%** (门槛 2%) -> 不达; 其它无回退, PPL 4.3568 安全。
根因: 波次效率模型高估 (每 block 2.4 tile 串行 + seam 归并抵消); ub2048 的优势主要是并行度总量/更少 launch。

现在开始 T03 chunked (timebox 1.5 天, 中途 checkpoint)。若你有 T03 的额外约束请直接写入 TASKS/T03。
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('QUESTIONS.md appended')
