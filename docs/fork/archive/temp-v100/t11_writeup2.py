import io

# ---------- BOARD (line-based, robust) ----------
p = r'D:\LLM\Backend\v100-collab\BOARD.md'
lines = io.open(p, encoding='utf-8').read().split('\n')
out = []
n_edit = 0
for ln in lines:
    if 'T10 近路: ub512 下启用 FA KV-split' in ln and 'APPROVED' in ln:
        out.append("| 0 | ~~T10 近路: ub512 FA KV-split~~ | ~~depth32k +5-8%~~ | 0.5-1 天 | **DONE = REJECTED (不达门槛)**: attention -5.6%, depth32k +2.0% (<3%), pp32768 +1.1% (<2%); 已回退 (patch 在 artifacts) |")
        n_edit += 1
    elif ln.startswith('| FA split-D/N32 移植 (T10)'):
        out.append(ln)
        out.append("| ub512 FA KV-split (stream-K 启发式, T10 近路) | attention -5.6% (1899->1793ms, 29.5->31.3 TF/s), 但端到端只 +2.0% depth32k / +1.1% pp32768 (门槛 3%/2%) -> 已回退; 波次效率模型高估 |")
        n_edit += 1
    elif '长上下文 (KV-split 后预期)' in ln:
        out.append("- 长上下文: KV-split 已否决 -> depth32k/pp32768 维持基线水平 (2026-09-22 复测 653/780, 当轮机器整体低 ~1%)")
        n_edit += 1
    elif '**待办顺序**' in ln and 'T10 KV-split' in ln:
        out.append("5. **待办顺序**: T03 chunked (1.5 天 timebox, analyst 已 APPROVED, 中途 checkpoint) -> T05 剩余 (可选) -> T04 (用户口径)")
        n_edit += 1
    elif '近路 = ub512 下启用 FA KV-split' in ln:
        out.append("-> **不移植 split-D/N32**。近路 ub512 KV-split 已试并**否决** (attention -5.6% 但端到端 +2.0%/+1.1%,")
        n_edit += 1
    elif '预期 depth32k +5-8% / pp32768 +3-5% / pp8192 +1.2% / pp4096 +0.7%' in ln and '|' not in ln:
        out.append("   不达门槛 3%/2%; 详见 RESULTS 'T10 近路'), 长上下文维持基线")
        n_edit += 1
    else:
        out.append(ln)
io.open(p, 'w', encoding='utf-8', newline='').write('\n'.join(out))
print(f'BOARD edits: {n_edit}')

# ---------- QUESTIONS ----------
p = r'D:\LLM\Backend\v100-collab\QUESTIONS.md'
t = """
---

## 2026-09-22 | T10 近路 (KV-split) 验收失败 | implementer

执行 Q6 的 A 并按 6 条门槛严格验收 -> **FAIL, 已回退, 按规则转 T03 chunked**。
完整数据见 RESULTS.md "T10 近路" 章节 (nsys 前后对比 + 2 轮交替 A/B 表)。

速览: nsys 机制成立 (grid 192->80, fixup kernel 出现, attention 1899.4->1792.7ms = -5.6%, 29.5->31.3 TF/s),
但端到端 depth32k **+2.0%** (门槛 3%) / pp32768 **+1.1%** (门槛 2%) -> 不达; 其它指标无回退, PPL 4.3568 安全。
根因: 波次效率模型高估 (每 block 2.4 tile 串行 + tile 接缝归并抵消); ub2048 的优势是并行度总量 + 更少 launch。

已开始 T03 chunked (timebox 1.5 天, 中途 checkpoint)。若你有 T03 的额外约束请写入 TASKS/T03。
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('QUESTIONS.md appended')
