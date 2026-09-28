import io

# ---------- BOARD ----------
p = r'D:\LLM\Backend\v100-collab\BOARD.md'
s = io.open(p, encoding='utf-8').read()

old = "| T10 | 长上下文 attention 侦察 (只测量) | **PROPOSED (analyst 指派)** | nsys 分解 depth32k/pp32768 的 attention 占比/TFLOPS; 决定是否移植 1Cat split-D/N32 |"
new = "| T10 | 长上下文 attention 侦察 (只测量) | **DONE (implementer, 待 analyst 复核)** | attention: d0 3.3% / pp32768 18.5% / depth32k 32.0%; ub512 29.5 TF/s, **ub2048 39.9 TF/s** (grid 192->768) -> 不值得移植 split-D/N32; 近路 = ub512 下启用 KV-split (预期 depth32k +5-8%, 0.5-1 天) |"
assert old in s, 'board-t10' ; s = s.replace(old, new, 1)

old2 = "| 3 | T03 chunked GDN 重写 | prefill +3-4% | timebox 1.5 天 | APPROVED (中途 checkpoint) |"
new2 = """| 3 | T03 chunked GDN 重写 | prefill +3-4% | timebox 1.5 天 | APPROVED (中途 checkpoint) |
| 3b | **T10 近路: ub512 下启用 FA KV-split** (parallel_blocks, 代码已存在) | depth32k +5-8% / pp32768 +3-5% / pp8192 +1.2% / pp4096 +0.7% | 0.5-1 天 | PROPOSED (Q6 请裁: 与 T03 的先后) |"""
assert old2 in s, 'board-q3' ; s = s.replace(old2, new2, 1)

old3 = "2. **外部参考已归档: REFERENCE-1cat-vllm.md**"
new3 = """2. **T10 侦察完成**: attention 不是 kernel 慢, 是 ub512 并行度不足 (grid=192 CTA / 80 SM / 1 CTA/SM /
   3 波次打尾 -> 80% 波次效率); ub2048 同 kernel 39.9 TF/s = 1Cat 上限 -> 不移植 split-D/N32。
   唯一近路: ub512 下让 KV-split (parallel_blocks) 生效 (预期 depth32k +5-8%, 0.5-1 天, Q6 待裁)
2. **外部参考已归档: REFERENCE-1cat-vllm.md**"""
assert old3 in s, 'board-guide' ; s = s.replace(old3, new3, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('BOARD.md updated (3 edits)')

# ---------- QUESTIONS ----------
p = r'D:\LLM\Backend\v100-collab\QUESTIONS.md'
t = """
---

## 2026-09-22 | T10 侦察完成 | implementer

Q6 (裁决请求): T10 结论 = "不值得移植 split-D/N32" (ub2048 同 kernel 已 39.9 TF/s, 超 1Cat 29-38 区间;
ub512 只有 29.5 是 grid=192/80SM/3 波次打尾造成)。但侦察发现一个 0.5-1 天的近路, 请裁决是否做、何时做:

A. **ub512 下启用 FA KV-split**: `launch_fattn` 已内置 `parallel_blocks` KV 分块 + fixup
   (fattn-common.cuh:1188+), 但 `stream_k=true` 分支在波次效率 >=75% 时直接 `blocks_num.x = ntiles_dst`
   (1150 行判据), 绕过了它。改动 = 判定不启用 stream-K 时回退到 parallel_blocks 循环 (约 20 行启发式)。
   预期: grid 192 -> 384/768, 波次效率 80% -> 96%, attention -20~26%
   -> **depth32k +5-8% / pp32768 +3-5% / pp8192 +1.2% / pp4096 +0.7%** (ub512 口径, 长上下文才值)
   风险: 数值 = softmax 归并顺序变化 (1 ulp 级重结合, 同 GDN vec4 先例, PPL 门槛能抓住问题)
B. 不做, 直接进 T03 chunked (prefill 全局 +3-4%, 1.5 天) — 我建议的顺序是 A 先做 (小、可测、可放弃), 再做 B。
C. 并入 T04 (ub 提高后 attention 自然受益) — 但 T04 待用户口径。

我的建议: **A (0.5-1 天) -> B (T03 chunked)**, 因为 A 是 T10 唯一可落地的产出, 且能复用 T03 的
PPL/pp512/4096/8192 验收流程; 若你更看重"单一全局项", 也可以 B -> A。
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('QUESTIONS.md: Q6 appended')
