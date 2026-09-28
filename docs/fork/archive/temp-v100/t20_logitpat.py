import json
from collections import defaultdict

def load_dump(path):
    groups = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 2 or parts[1] == "NULL":
            continue
        ctx = parts[0]
        toks = {}
        for p in parts[1:9]:
            t, v = p.split(":")
            toks[int(t)] = float(v)
        groups[ctx].append(toks)
    return groups

ga = load_dump(r"<TEMP>\v100\t20_logits_none.txt")
gb = load_dump(r"<TEMP>\v100\t20_logits_nmax1.txt")
la = max(ga.values(), key=len)
lb = max(gb.values(), key=len)
n = min(len(la), len(lb))

rows = []
for i in range(n):
    a, b = la[i], lb[i]
    common = set(a) & set(b)
    if not common:
        d = float("inf")
    else:
        d = max(abs(a[t] - b[t]) for t in common)
    rows.append((i, d, a, b))

ndiff = sum(1 for _, d, _, _ in rows if d > 0)
print(f"calls={n} calls_with_diff={ndiff} first_diff={next((i for i,d,_,_ in rows if d>0), None)}")

prev = -10
for i, d, a, b in rows:
    if d > 0 and (i - prev > 1 or d > 0.5):
        top_a = max(a, key=a.get)
        top_b = max(b, key=b.get)
        print(f"call {i}: maxdiff={d:.5f} top1_a={top_a}({a[top_a]:.4f}) top1_b={top_b}({b[top_b]:.4f})")
    if d > 0:
        prev = i

# difference growth over segments
import statistics
for lo in range(200, n, 100):
    seg = [d for i, d, _, _ in rows if lo <= i < lo + 100 and d > 0]
    if seg:
        print(f"segment {lo}-{lo+99}: ndiff={len(seg)} max={max(seg):.5f} mean={statistics.mean(seg):.5f}")
