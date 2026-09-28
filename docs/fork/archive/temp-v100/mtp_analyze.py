import io, os, hashlib, difflib

D = r"D:\LLM\Backend\mtp"
files = {
    "原版-nomtp": "原版-nomtp.md",
    "原版-mtp": "原版-mtp.md",
    "修复-nomtp": "修复-nomtp.md",
    "修复-mtp": "修复-mtp.md",
}
texts = {}
out = io.open(os.path.join(os.environ["TEMP"], "v100", "mtp_result_analysis.txt"), "w", encoding="utf-8")

for k, fn in files.items():
    p = os.path.join(D, fn)
    raw = io.open(p, "rb").read()
    t = raw.decode("utf-8-sig")
    texts[k] = t
    body = t.strip()
    paras = [x for x in body.split("\n") if x.strip()]
    out.write("%-12s bytes=%d chars=%d paragraphs=%d md5=%s\n" % (
        k, len(raw), len(t), len(paras), hashlib.md5(raw).hexdigest()[:12]))

out.write("\n== pairwise SequenceMatcher ratio (whole text) ==\nkeys=%s\n" % list(texts))
for a in files:
    row = []
    for b in files:
        r = difflib.SequenceMatcher(None, texts[a], texts[b]).ratio()
        row.append("%.3f" % r)
    out.write("%-12s %s\n" % (a, " ".join(row)))

out.write("\n== structure markers ==\n")
markers = ["五月花", "华盛顿", "林肯", "南北", "广岛", "长崎", "布雷顿森林", "冷战", "苏联", "苏俄",
           "硅谷", "华尔街", "美元", "航母", "北约", "国会山", "太史公曰", "呜呼", "盛极",
           "仁义不施", "攻守之势", "灯塔", "陈涉", "印第安", "昭昭天命", "斯普特尼克", "登月", "QE", "次贷",
           "罗斯福", "里根", "福山", "马斯克", "苹果", "英伟达", "一带一路", "金砖", "特朗普", "特朗普", "新冠", "阿富汗", "伊拉克"]
out.write("%-14s" % "marker" + "".join("%-12s" % k for k in files) + "\n")
for m in markers:
    out.write("%-14s" % m + "".join("%-12s" % ("Y" if m in texts[k] else ".") for k in files) + "\n")

for k in files:
    t = texts[k].strip()
    paras = [x.strip() for x in t.split("\n") if x.strip()]
    out.write("\n===== %s (chars=%d) =====\n" % (k, len(t)))
    for i, p in enumerate(paras):
        out.write("  [%02d] %s\n" % (i, p[:60].replace("\n", " ")))

out.close()
print("written")
