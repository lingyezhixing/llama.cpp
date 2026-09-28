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

for ei, ex in enumerate(execs):
    print('===== exec %d: %d nodes =====' % (ei, len(ex)))
    sc = [n for n in ex if n['op'] == 'SCALE']
    cache = [n for n in sc if n['name'].startswith('cache_')]
    z = [n for n in cache if n['ne'] == 0]
    print('SCALE total=%d  cache_*=%d (ne>0: %d, ne==0: %d)  ne_sum(cache)=%d' % (
        len(sc), len(cache), len(cache) - len(z), len(z), sum(n['ne'] for n in cache)))
    print('  cache names sample:', sorted(set(n['name'] for n in cache))[:4])
    print('  cache ne values:', collections.Counter(n['ne'] for n in cache).most_common(5))
    # sizes for the delta-net / conv path
    for key in ['conv_states', 'conv_input', 'state_predelta']:
        sub = [n for n in ex if n['name'].startswith(key)]
        if sub:
            print('  %-16s cnt=%d ne=%s' % (key, len(sub), sorted(set(n['ne'] for n in sub))))
    # total executed bytes estimate per op
    print()

# diff between exec1 and exec0 on (op, normalized name)
def norm(s):
    s = re.sub(r'[-.]\d+', '-N', s)
    s = re.sub(r'\s*\(.*?\)', '', s)
    return s.strip()
c0 = collections.Counter((n['op'], norm(n['name']), n['ne'], n['fused']) for n in execs[0])
c1 = collections.Counter((n['op'], norm(n['name']), n['ne'], n['fused']) for n in execs[1])
print('--- differences exec1 vs exec0 ---')
for k in sorted(set(c0) | set(c1)):
    a, b = c0.get(k, 0), c1.get(k, 0)
    if a != b:
        print('  %-16s %-34s ne=%-8d fused=%-5s  exec0=%d exec1=%d' % (k[0], k[1], k[2], k[3], a, b))
