import re, sys, os

# Parse a llama-server log produced with "-lv 4" (TRACE) and split one generation request into
# rounds / draft / accept / begin / residual using the SPC stats that speculative.cpp prints
# per request inside slot::print_timings().

path = sys.argv[1]
txt = open(path, encoding='utf-8', errors='replace').read()

re_eval = re.compile(r'eval time =\s*([\d.]+) ms /\s*(\d+) tokens \(.*?([\d.]+) tokens per second\)')
re_acc = re.compile(r'draft acceptance = ([\d.]+) \(\s*(\d+) accepted /\s*(\d+) generated\), mean len =\s*([\d.]+)')
re_stats = re.compile(
    r'statistics\s+(\S+): #calls\(b,g,a\) =\s*(\d+)\s+(\d+)\s+(\d+), '
    r'#gen drafts =\s*(\d+), #acc drafts =\s*(\d+), '
    r'#gen tokens =\s*(\d+), #acc tokens =\s*(\d+)'
    r'(?:, #mean acc len = ([\d.]+), #acc rate/pos = \(([^)]*)\))?'
    r'(?:, dur\(b,g,a\) = ([\d.]+), ([\d.]+), ([\d.]+) ms)?'
)

events = []
for i, line in enumerate(txt.splitlines()):
    m = re_stats.search(line)
    if m:
        events.append(('stats', i, m))
        continue
    m = re_eval.search(line)
    if m:
        events.append(('eval', i, m))
        continue
    m = re_acc.search(line)
    if m:
        events.append(('acc', i, m))

prev = None
req = 0
print(f"{'req':>3} {'gen_ms':>9} {'tokens':>7} {'rounds':>7} {'tok/rd':>7} {'acc/n':>9} "
      f"{'draft/rd':>9} {'accept/rd':>9} {'begin/rd':>9} {'resid/rd':>9}  {'rate/pos':<24} {'tps':>7}")
for kind, li, m in events:
    if kind == 'eval':
        gen_ms = float(m.group(1)); ntok = int(m.group(2)); tps = float(m.group(3))
    elif kind == 'acc':
        pass
    elif kind == 'stats':
        cur = dict(type=m.group(1),
                   cb=int(m.group(2)), cg=int(m.group(3)), ca=int(m.group(4)),
                   gen_d=int(m.group(5)), acc_d=int(m.group(6)),
                   gen_t=int(m.group(7)), acc_t=int(m.group(8)),
                   dur=tuple(float(x) for x in m.groups()[10:13]) if m.group(11) is not None else None,
                   rate=m.group(10) if m.group(9) else None)
        if prev is None:
            prev = dict.fromkeys(cur, 0); prev['dur'] = (0.0, 0.0, 0.0); prev['type'] = cur['type']
        d_ca = cur['ca'] - prev['ca']
        d_gen_d = cur['gen_d'] - prev['gen_d']
        d_acc_d = cur['acc_d'] - prev['acc_d']
        d_acc_t = cur['acc_t'] - prev['acc_t']
        if cur['dur'] and prev.get('dur') is not None:
            d_b = cur['dur'][0] - prev['dur'][0]
            d_d = cur['dur'][1] - prev['dur'][1]
            d_a = cur['dur'][2] - prev['dur'][2]
        else:
            d_b = d_d = d_a = float('nan')
        prev = cur
        if d_ca <= 0:
            continue
        req += 1
        resid = (gen_ms - d_d - d_a - d_b) / d_ca
        rp = ''
        if m.group(10):
            try:
                vals = [float(x) for x in m.group(10).split(',')]
                rp = '(' + ', '.join(f'{v:.3f}' for v in vals) + ')'
            except Exception:
                rp = m.group(10)[:24]
        acc_ratio = (d_acc_d / d_gen_d) if d_gen_d else 0.0
        print(f"{req:3d} {gen_ms:9.1f} {ntok:7d} {d_ca:7d} {ntok/d_ca:7.2f} "
              f"{d_acc_d:4d}/{d_gen_d:<4d} {d_d/d_ca:9.2f} {d_a/d_ca:9.2f} {d_b/d_ca:9.2f} {resid:9.2f}  "
              f"{rp:<24} {tps:7.2f}")

