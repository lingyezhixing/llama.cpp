import json

d = json.load(open(r"<TEMP>\v100\t20_srv_none_fix_probs_raw.json", encoding="utf-8-sig"))
cp = d["completion_probabilities"]
for i in (285, 286, 287, 586, 587, 588):
    if i < len(cp):
        e = cp[i]
        print("--- pos", i, "id=", e.get("id"), repr(e.get("token")), "---")
        for p in (e.get("top_logprobs") or [])[:4]:
            print("   tok=", p.get("id"), "lp=%.4f" % p.get("logprob"), repr(p.get("token")))
