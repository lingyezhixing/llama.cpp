import json, sys

base = json.load(open(rf"<TEMP>\v100\t20_srv_{sys.argv[1]}.json", encoding="utf-8-sig"))
ref = base["tokens"]
print(f"{sys.argv[1]} : n={len(ref)} tps={base['tps']:.2f}")

for tag in sys.argv[2:]:
    d = json.load(open(rf"<TEMP>\v100\t20_srv_{tag}.json", encoding="utf-8-sig"))
    t = d["tokens"]
    n = min(len(ref), len(t))
    first = None
    for i in range(n):
        if ref[i] != t[i]:
            first = i
            break
    if first is None and len(ref) == len(t):
        print(f"{tag}: IDENTICAL ({len(t)} tokens) tps={d['tps']:.2f}")
    else:
        print(f"{tag}: DIVERGE at token {first} (len {len(t)}) tps={d['tps']:.2f}")
        if first is not None:
            lo = max(0, first - 5)
            hi = min(n, first + 6)
            print("  ctx  ref:", ref[lo:hi])
            print("  ctx  got:", t[lo:hi])
