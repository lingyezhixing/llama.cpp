import json

def load(tag):
    return json.load(open(rf"<TEMP>\v100\{tag}.json", encoding="utf-8-sig"))

def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n

arms = ["stock", "t19", "t20", "t20fix"]
for depth in ("d0", "d8192"):
    data = {}
    for a in arms:
        try:
            data[a] = load(f"3w_{a}_{depth}")["tokens"]
        except FileNotFoundError:
            data[a] = None
    print(f"=== {depth} (250 tokens, greedy, no spec) ===")
    for a in arms:
        if data[a] is not None:
            print(f"  {a}: len={len(data[a])}")
    for i, a in enumerate(arms):
        for b in arms[i+1:]:
            if data[a] is None or data[b] is None:
                continue
            fd = first_diff(data[a], data[b])
            if fd is None:
                print(f"{a:7s} vs {b:7s}: IDENTICAL")
            else:
                print(f"{a:7s} vs {b:7s}: diverge at token {fd}")
