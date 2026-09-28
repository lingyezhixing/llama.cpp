import json

def load(tag):
    return json.load(open(rf"<TEMP>\v100\t20_srv_{tag}.json", encoding="utf-8-sig"))

r1 = load("none_r1")
rp = load("none_probs")
t1, tp = r1["tokens"], rp["tokens"]
n = min(len(t1), len(tp))
first = next((i for i in range(n) if t1[i] != tp[i]), None)
print(f"determinism none_r1 vs none_probs: first_diff={first} lens={len(t1)},{len(tp)}")

probs = rp.get("completion_probabilities") or rp.get("probs")
print("prob keys present:", [k for k in rp.keys() if "prob" in k.lower()])
idx = 207
print("token[206..209] none_probs:", tp[206:210])
if probs:
    print(f"n_prob_entries={len(probs)}")
    for i in (206, 207, 208):
        if i < len(probs):
            e = probs[i]
            top = e.get("top_logprobs") or e.get("probs") or []
            print(f"--- position {i} token={e.get('id')} ---")
            for p in top[:6]:
                print(f"   tok={p.get('id')} logprob={p.get('logprob'):.4f} tokstr={p.get('token','')!r}")
