import json

def load(tag):
    return json.load(open(rf"<TEMP>\v100\t20_srv_{tag}_raw.json", encoding="utf-8-sig"))

a = load("none_fix_probs")
b = load("nmax1_fix_probs")
ca, cb = a["completion_probabilities"], b["completion_probabilities"]
ta, tb = a["tokens"], b["tokens"]
n = min(len(ca), len(cb), len(ta), len(tb))
first = next((i for i in range(n) if ta[i] != tb[i]), None)
print("first token diff:", first, "n =", n)

# compare top-1 logprob of the *chosen* token at each position (rounding: 4 decimals)
ndiff = 0
first_lp_diff = None
for i in range(min(first if first is not None else n, 560)):
    la = ca[i]["logprob"]
    lb = cb[i]["logprob"]
    if abs(la - lb) > 1e-9:
        ndiff += 1
        if first_lp_diff is None:
            first_lp_diff = i
        if ndiff <= 5 or i > (first if first else n) - 6:
            print(f"pos {i}: lp_a={la:.4f} lp_b={lb:.4f} tok_a={ca[i]['id']} tok_b={cb[i]['id']}")
print("total positions with lp difference:", ndiff, "first at", first_lp_diff)
