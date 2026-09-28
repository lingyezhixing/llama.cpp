import io, re, collections

p = r'<TEMP>\v100\nodes_raw.txt'
lines = io.open(p, encoding='utf-8').read().splitlines()

# split into executions whenever idx resets
execs = []
cur = []
prev = None
for ln in lines:
    if not ln.startswith('NODE'):
        continue
    m = re.match(r'NODE\s+(\d+)\s+(.*?)\s{2,}(\S+)\s+(\S+)\s+ne=(\d+)(\s+FUSED)?$', ln)
    if m:
        n = dict(idx=int(m.group(1)), name=m.group(2).strip(), op=m.group(3),
                 type=m.group(4), ne=int(m.group(5)), fused=bool(m.group(6)))
    else:
        m2 = re.match(r'NODE\s+(\d+)\s+(.*?)\s+(\S+)\s+(\S+)\s+ne=(\d+)$', ln)
        n = dict(idx=int(m2.group(1)), name=m2.group(2).strip(), op=m2.group(3),
                 type=m2.group(4), ne=int(m2.group(5)), fused=False)
    if prev is not None and n['idx'] <= prev:
        execs.append(cur)
        cur = []
    cur.append(n)
    prev = n['idx']
execs.append(cur)

print('executions:', len(execs))
for ei, ex in enumerate(execs):
    rms = [n for n in ex if n['op'] == 'RMS_NORM']
    mm = [n for n in ex if n['op'] == 'MUL_MAT']
    exe = [n for n in ex if not n['fused']]
    print('exec %d: nodes=%d executed=%d fused=%d  MM=%d  RMS=%d' % (ei, len(ex), len(exe), len(ex)-len(exe), len(mm), len(rms)))

ex = execs[1]
exe = [n for n in ex if not n['fused']]
print('\n--- exec #1 (assumed real decode): executed by op ---')
for op, cnt in collections.Counter(n['op'] for n in exe).most_common():
    print('%-20s %5d' % (op, cnt))
print('\nfused spans:', len(ex) - len(exe))

print('\n--- MUL_MAT executed node names (pattern) ---')
def norm_name(s):
    return re.sub(r'[-.]\d+', '-N', s)
for (op, nm), c in collections.Counter((n['op'], norm_name(n['name'])) for n in exe if n['op'] == 'MUL_MAT').most_common():
    print('%-18s %-34s %5d' % (op, nm, c))
