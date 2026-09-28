import json

d = json.load(open(r"<TEMP>\v100\t20_srv_none_fix_probs_raw.json", encoding="utf-8-sig"))
print("keys:", list(d.keys()))
cp = d.get("completion_probabilities")
print("n cp:", len(cp) if cp else None)
if cp:
    print("entry keys:", list(cp[0].keys()))
    for i in (556, 557, 558, 559):
        if i < len(cp):
            e = cp[i]
            print("--- pos", i, "id=", e.get("id"), "---")
            for p in (e.get("top_logprobs") or [])[:6]:
                print("  tok=", p.get("id"), "lp=%.4f" % p.get("logprob"), repr(p.get("token")))
