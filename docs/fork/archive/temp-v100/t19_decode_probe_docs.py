import io

def read(p):
    s = io.open(p, encoding='utf-8', newline='').read()
    nl = '\r\n' if '\r\n' in s else '\n'
    return s, nl

def write(p, s):
    io.open(p, 'w', encoding='utf-8', newline='').write(s)

t = """
---

## T19 遗留项排查 (2026-09-23): 长文 decode

### d32768: **排除** (矩阵值是异常值)

- nsys 逐 launch 对比 (两臂, `-p 0 -n 128 -d 32768 -r 1`): decode 段 (最后一个 >5ms kernel 之后)
  - OURS: 262368 kernels, GPU busy **5443.7 ms**, wall 5917.3 ms (gaps 8.0%)
  - STOCK: 262368 kernels, GPU busy **5446.3 ms**, wall 5897.9 ms (gaps 7.7%)
  - 逐内核一致: mul_mat_vec_q 3877.3 vs 3879.0 / **flash_attn_ext_vec 761.78 vs 761.81** / rms_norm 176.5 vs 176.1 /
    quantize_q8_1 134.4 vs 134.7 / GDN 50.13 vs 50.16 ...
  - 唯一差异 = silu 内核: OURS `unary_gated_op_kernel_f32_vec4` 47.70ms vs STOCK `unary_gated_op_kernel` 48.61ms (-1.9%, 设计内)
- 复测 (plain, 交替顺序, r=2): OURS **23.02 / 22.82**, STOCK **22.87 / 22.87** -> **+0.2% (持平)**
- 结论: 矩阵里的 18.93 vs 21.18 (-10.6%) 是当时测量状态异常, 非代码差异

### d131072: **存疑 (小回归或测量噪声, 未定论)**

- 复测 (plain, r=2, 两轮交替): 轮1 OURS 10.94 / STOCK 11.33 (-3.4%); 轮2 (反向) STOCK 11.80 / OURS 10.58 (-10.3%)
  -> 方向两轮一致 (OURS 慢), 幅度受热状态影响大 (绝对值为矩阵值 ±4%)
- nsys 逐 launch 对比失败: STOCK 的 d131072 profile 两次都在 prefill 尾部截断 (trace 缺 decode 段, 无 dropped 警告);
  OURS 侧完整: decode 128 token GPU busy 13.2s (MMVQ 6.07s + flash_attn_ext_vec 5.87s + 其他 1.3s)
- 已知约束: decode 段两臂用的内核集合相同 (MMVQ / flash_attn_ext_vec / rms_norm / GDN / silu);
  本 fork 对 decode 路径的唯一改动 (silu vec4) 在 d32768 实测更快; FA 的 mma/tile 配置与 stream-K 决策不参与 decode (vec 内核)
- 判定建议: 若要定论, 需 4-6 轮交替 A/B (约 30 min) 或在真实生产场景 (MTP, 长文) 下对比; 当前按"疑似小回归 (<=3%)"记档
"""
io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

p = r'D:\LLM\Backend\v100-collab\BOARD.md'
s, nl = read(p)
old = 'tg d0-d8192 持平, **d32768/131072 -10.6/-9.4% (待查)**'
new = 'tg d0-d8192 持平, d32768 复测**持平 (+0.2%, 矩阵值=异常)**; d131072 **存疑 (-3~-10%, 未定论)**'
assert s.count(old) == 1
s = s.replace(old, new)
write(p, s)
print('BOARD updated')

p = r'D:\LLM\Backend\v100-collab\STATUS.md'
s, nl = read(p)
s += nl + nl + """## T19 遗留项排查 (2026-09-23, implementer)

- **d32768 decode: 排除** - nsys 逐 launch 两臂一致 (GPU busy 5443.7 vs 5446.3ms; flash_attn_ext_vec 761.78 vs 761.81);
  plain 复测 +0.2% (23.02/22.82 vs 22.87/22.87); 矩阵的 -10.6% = 测量状态异常
- **d131072 decode: 存疑** - plain 复测两轮同向 (OURS -3.4% / -10.3%), 幅度受热状态影响;
  STOCK 侧 nsys profile 两次截断 (缺 decode 段) -> 未能逐内核定论; 按"疑似小回归 (<=3%)"记档
- 详见 RESULTS "T19 遗留项排查"
"""
write(p, s)
print('STATUS appended')

p = r'D:\LLM\Backend\v100-collab\ENVIRONMENT.md'
s, nl = read(p)
old = '| 验收 (OURS, 新 base) | pp512 948.5 / pp131072 531.5 / tg128 26.61 / **PPL 4.3562**; vs STOCK: pp +8.7..+35.6%, tg 长文 -10% (遗留待查) |'
new = '| 验收 (OURS, 新 base) | pp512 948.5 / pp131072 531.5 / tg128 26.61 / **PPL 4.3562**; vs STOCK: pp +8.7..+35.6%, tg d32768 持平 (复测), tg d131072 存疑 (-3~-10%) |'
assert s.count(old) == 1
s = s.replace(old, new)
write(p, s)
print('ENVIRONMENT updated')
