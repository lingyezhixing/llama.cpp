import io, re, collections

p = r'<TEMP>\v100\nodes_raw.txt'
lines = io.open(p, encoding='utf-8').read().splitlines()
execs, cur, prev = [], [], None
for ln in lines:
    if not ln.startswith('NODE'):
        continue
    m = re.match(r'NODE\s+(\d+)\s+(.*?)\s{2,}(\S+)\s+(\S+)\s+ne=(\d+)(\s+FUSED)?$', ln)
    if m:
        n = dict(idx=int(m.group(1)), name=m.group(2).strip(), op=m.group(3), type=m.group(4),
                 ne=int(m.group(5)), fused=bool(m.group(6)))
    else:
        m2 = re.match(r'NODE\s+(\d+)\s+(.*?)\s+(\S+)\s+(\S+)\s+ne=(\d+)$', ln)
        n = dict(idx=int(m2.group(1)), name=m2.group(2).strip(), op=m2.group(3), type=m2.group(4),
                 ne=int(m2.group(5)), fused=False)
    if prev is not None and n['idx'] <= prev:
        execs.append(cur); cur = []
    cur.append(n); prev = n['idx']
execs.append(cur)
ex = execs[1]

def norm(s):
    s = re.sub(r'[-.]\d+', '-N', s)
    s = re.sub(r'\s*\(.*?\)', '', s)
    return s.strip()

cc = collections.Counter((n['op'], norm(n['name']), 'F' if n['fused'] else 'E') for n in ex)
print('%-14s %-42s %-2s %5s' % ('op', 'name', 'E/F', 'cnt'))
for (op, nm, fz), c in sorted(cc.items()):
    print('%-14s %-42s %-2s %5d' % (op, nm, fz, c))

print('\n--- ne by op (executed only) ---')
neo = collections.defaultdict(collections.Counter)
for n in ex:
    if not n['fused']:
        neo[n['op']][n['ne']] += 1
for op in sorted(neo):
    items = ', '.join('%d:%d' % (k, v) for k, v in sorted(neo[op].items()))
    print('%-16s %s' % (op, items))
