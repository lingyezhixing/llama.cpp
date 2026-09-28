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
data = {a: load(f"3w_{a}_d0")["tokens"] for a in arms}
print("=== d0 (creative prompt, 250 tokens, greedy) ===")
for a in arms:
    print(f"  {a}: len={len(data[a])}")

pairs = [(a, b) for i, a in enumerate(arms) for b in arms[i+1:]]
for a, b in pairs:
    fd = first_diff(data[a], data[b])
    if fd is None:
        print(f"{a:7s} vs {b:7s}: IDENTICAL")
    else:
        print(f"{a:7s} vs {b:7s}: diverge at token {fd}")

# divergence point details
for a, b in pairs:
    fd = first_diff(data[a], data[b])
    if fd is not None and fd < min(len(data[a]), len(data[b])):
        lo = max(0, fd - 3)
        print(f"--- {a} vs {b} @ {fd} ---")
        print(f"   {a}:", data[a][lo:fd+4])
        print(f"   {b}:", data[b][lo:fd+4])
