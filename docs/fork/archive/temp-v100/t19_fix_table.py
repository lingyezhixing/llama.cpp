import io

p = r'D:\LLM\Backend\v100-collab\RESULTS.md'
s = io.open(p, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in s else '\n'

old_rows = ('| tg128 d32768 | 18.93 | 21.18 | **-10.6%** |' + nl +
            '| tg128 d131072 | 10.47 | 11.55 | **-9.4%** |')
new_rows = ('| tg128 d32768 | ~~18.93~~ **22.92** | ~~21.18~~ **22.87** | ~~-10.6%~~ **+0.2%** (复测修正) |' + nl +
            '| tg128 d131072 | **10.76** (复测均值) | **11.57** (复测均值) | **-7.0% 存疑** (矩阵 -9.4%, 未定论) |')
assert s.count(old_rows) == 1, s.count(old_rows)
s = s.replace(old_rows, new_rows)

old_bullet = ('- decode: d<=8192 持平; **d>=32768 OURS 慢 ~10% (待查)**: 该区间 decode 主要花在 FA 长 KV (stream-K 路径);' + nl +
              '  该决策来自上游 heuristic (本 fork 未改 decode 的 ncols 选择), 但差值可复现 (两臂各 r=2, 非漂移级)' + nl +
              '  -> 已记为遗留问题; 若用户在意长文 decode, 下一步 = nsys tg@d32768 逐 kernel 对比 (约 3 min)')
new_bullet = ('- decode: d<=8192 持平; **d32768 复测持平 (+0.2%)** (矩阵的 -10.6% 系测量状态异常, 已排除);' + nl +
              '  **d131072 存疑 (-3~-10%, 未定论)**; 两臂 decode 内核集合相同 (含 `flash_attn_ext_vec`), 详见"T19 遗留项排查"')
assert s.count(old_bullet) == 1, s.count(old_bullet)
s = s.replace(old_bullet, new_bullet)

old_cap = '## 实测 (ub512; 每点 = llama-bench 内部 r 次均值; 用户指示: 跳过 ub2048, 不做轮换重测)'
new_cap = '## 实测 (ub512; 每点 = llama-bench 内部 r 次均值; 用户指示: 跳过 ub2048; tg d32768/131072 已按复测修正/标注)'
assert s.count(old_cap) == 1
s = s.replace(old_cap, new_cap)

io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('RESULTS table corrected')
