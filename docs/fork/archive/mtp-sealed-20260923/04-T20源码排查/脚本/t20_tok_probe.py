import json, io, time, urllib.request

base = "http://127.0.0.1:8650"

def post(path, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, repr(e)

for i in range(300):
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as r:
            if b"ok" in r.read():
                break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("server not up")

raw = json.load(io.open(r"<TEMP>\v100\t20_quality4\raw_stock-nomtp_sample1.json", encoding="utf-8-sig"))
m = raw["choices"][0]["message"]
reason = m.get("reasoning_content") or ""
content = m.get("content") or ""
print("reason repr:", json.dumps(reason[:60], ensure_ascii=True))
print("content repr:", json.dumps(content[:60], ensure_ascii=True))

for name, text in [("reason", reason), ("content", content)]:
    st, res = post("/tokenize", {"content": text, "add_special": False})
    if isinstance(res, dict):
        print(name, "status", st, "n_tokens", len(res.get("tokens", [])))
    else:
        print(name, "status", st, "body", res[:300])

st, res = post("/tokenize", {"content": "hello world", "add_special": False})
print("plain", st, len(res.get("tokens", [])) if isinstance(res, dict) else res[:200])
st, res = post("/tokenize", {"content": content})
print("content (default add_special)", st, len(res.get("tokens", [])) if isinstance(res, dict) else res[:200])
