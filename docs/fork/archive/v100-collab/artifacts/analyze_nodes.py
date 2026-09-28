import io, re, collections

p = r'<TEMP>\v100\nodes_raw.txt'
lines = io.open(p, encoding='utf-8').read().splitlines()
nodes = []
for ln in lines:
    if not ln.startswith('NODE'):
        continue
    m = re.match(r'NODE\s+(\d+)\s+(.*?)\s{2,}(\S+)\s+(\S+)\s+ne=(\d+)(\s+FUSED)?$', ln)
    if not m:
        print('UNPARSED:', ln)
        continue
    nodes.append(dict(idx=int(m.group(1)), name=m.group(2).strip(), op=m.group(3),
                      type=m.group(4), ne=int(m.group(5)), fused=bool(m.group(6))))

print('total nodes:', len(nodes))
ex = [n for n in nodes if not n['fused']]
print('executed (non-fused):', len(ex))
print('fused:', len(nodes) - len(ex))

print('\n--- executed nodes by op ---')
for op, cnt in collections.Counter(n['op'] for n in ex).most_common():
    print('%-20s %5d' % (op, cnt))

print('\n--- fused spans by first-node op ---')
for op, cnt in collections.Counter(n['op'] for n in nodes if n['fused']).most_common():
    print('%-20s %5d' % (op, cnt))

def norm_name(n):
    s = re.sub(r'[-.]\d+', '-N', n['name'])
    s = re.sub(r'\s*\(.*?\)', '', s)
    s = re.sub(r'\s*\(.*?\)\s*$', '', s)
    return s.strip()

key = lambda n: (n['op'], norm_name(n), 'F' if n['fused'] else '')
print('\n--- executed+fused nodes by (op, name-pattern) ---')
cnt = collections.Counter(key(n) for n in nodes)
for (op, nm, fz), c in cnt.most_common(80):
    print('%-18s %-30s %-1s %5d' % (op, nm, fz, c))
