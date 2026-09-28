import json, sys, io

def load(tag):
    return json.load(open(rf"<TEMP>\v100\t20_ns_{tag}.json", encoding="utf-8-sig"))

# analyze the earlier 4-prompt d0 comparison (T19 vs stock)
print("=== earlier d0 run: ours(T19) vs stock, 4 prompts ===")
for pid in ("p1_factual", "p2_creative", "p3_code", "p4_math"):
    a = load(f"ours_{pid}")["tokens"]
    b = load(f"stock_{pid}")["tokens"]
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    if first is None and len(a) == len(b):
        print(f"{pid}: IDENTICAL ({len(a)} tokens)")
    else:
        print(f"{pid}: DIVERGE at {first} (len ours={len(a)} stock={len(b)})")
        if first is not None:
            lo = max(0, first - 4)
            print("   ours :", a[lo:first + 5])
            print("   stock:", b[lo:first + 5])
