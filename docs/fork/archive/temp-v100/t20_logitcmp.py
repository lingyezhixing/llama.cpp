import json, sys
from collections import defaultdict

def load_dump(path):
    groups = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 2 or parts[1] == "NULL":
            continue
        ctx = parts[0]
        toks = []
        for p in parts[1:9]:
            t, v = p.split(":")
            toks.append((int(t), float(v)))
        groups[ctx].append(toks)
    return groups

ga = load_dump(r"<TEMP>\v100\t20_logits_none.txt")
gb = load_dump(r"<TEMP>\v100\t20_logits_nmax1.txt")

for name, g in (("none", ga), ("nmax1", gb)):
    for ctx, lines in g.items():
        print(f"{name} ctx={ctx} lines={len(lines)} first_top1={lines[0][0]}")

# use the largest group (target ctx)
def target(g):
    return max(g.values(), key=len)

la = target(ga)
lb = target(gb)
n = min(len(la), len(lb))
print(f"comparing {n} sample calls")

first = None
for i in range(n):
    if la[i] != lb[i]:
        first = i
        break
print("first logits difference at sample call:", first)
if first is not None:
    for i in range(max(0, first - 2), min(n, first + 4)):
        print(f"--- call {i} ---")
        print("  none :", la[i])
        print("  nmax1:", lb[i])
