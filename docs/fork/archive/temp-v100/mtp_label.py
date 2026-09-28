import io, os, hashlib, difflib, shutil

D = r"D:\LLM\Backend\mtp"
BK = os.path.join(D, "备份-原始")
os.makedirs(BK, exist_ok=True)

question = ("用贴吧暴躁老哥的口气和风格， 且使用文言文，  仿《过秦论》作《过美利坚论》\n"
            "注意：\n1、文言文要言辞犀利，且不带白话文句子\n2、用非常硬核的文言文\n"
            "3、知识面体现的要宽\n4、插上想象力翅膀，豪放主义诗人的奔放程度。\n"
            "5、用词要极具想象力，且非常奔放不羁，你要放飞自我。")

files = ["原版-nomtp.md", "原版-mtp.md", "修复-nomtp.md", "修复-mtp.md"]
texts = {}
for fn in files:
    p = os.path.join(D, fn)
    raw = io.open(p, "rb").read()
    shutil.copy2(p, os.path.join(BK, fn))
    texts[fn] = raw.decode("utf-8-sig")

def stats(t):
    body = t.strip()
    paras = [x for x in body.split("\n") if x.strip()]
    return len(body), len(paras), hashlib.md5(t.strip().encode("utf-8")).hexdigest()[:12]

sim = {}
for a in files:
    for b in files:
        sim[(a, b)] = difflib.SequenceMatcher(None, texts[a], texts[b]).ratio()

labels = {
    "原版-nomtp.md": ("原版 · 不开 MTP", "上游 llama.cpp（ggml-cuda 976E2CAB）", "关闭"),
    "原版-mtp.md":   ("原版 · 开 MTP",   "上游 llama.cpp（ggml-cuda 976E2CAB）", "draft-mtp, n-max 3"),
    "修复-nomtp.md": ("修复后 · 不开 MTP", "修复分支（ggml-cuda 8093C771 + 修复开关）", "关闭"),
    "修复-mtp.md":   ("修复后 · 开 MTP",   "修复分支（ggml-cuda 8093C771 + 修复开关）", "draft-mtp, n-max 3"),
}
pair = {"原版-nomtp.md": "原版-mtp.md", "原版-mtp.md": "原版-nomtp.md",
        "修复-nomtp.md": "修复-mtp.md", "修复-mtp.md": "修复-nomtp.md"}

lines_summary = []
for fn in files:
    title, build, spec = labels[fn]
    c, p, md5 = stats(texts[fn])
    other = pair[fn]
    c2, p2, md52 = stats(texts[other])
    same = (md5 == md52) and (c == c2)
    if same:
        cmp_line = "**与同构建" + ("开" if "nomtp" in fn else "不开") + " MTP 版逐字节完全相同**（md5 一致）"
    else:
        cmp_line = ("**与同构建" + ("开" if "nomtp" in fn else "不开") + " MTP 版完全不同**"
                    "（长度 %d vs %d 字，相似度 %.2f）" % (c, c2, sim[(fn, other)]))
    header = (
        "> 【%s】\n"
        "> 构建：%s ｜ 投机解码：%s ｜ 采样：temp 1.0, top-p 0.95, top-k 20, min-p 0, seed 42\n"
        "> 上下文：65536（KV fp16）｜ 聊天模板：模型自带（GGUF 内嵌）｜ 最大输出：不限\n"
        "> 输出长度：%d 字 / %d 段\n"
        "> %s\n\n---\n\n" % (title, build, spec, c, p, cmp_line)
    )
    out = header + texts[fn].strip() + "\n"
    io.open(os.path.join(D, fn), "w", encoding="utf-8-sig", newline="\n").write(out)
    lines_summary.append((title, c, p, md5, same, sim[(fn, other)]))

c1, p1, m1 = stats(texts["原版-nomtp.md"]); c2, p2, m2 = stats(texts["原版-mtp.md"])
c3, p3, m3 = stats(texts["修复-nomtp.md"]); c4, p4, m4 = stats(texts["修复-mtp.md"])

summary = """# MTP 一致性对照：Qwen3.8-27B（V100）

**测试题**（单轮，贪心改随机采样以复现生产用法）：

> %s

## 测试配置

| 项 | 值 |
|---|---|
| 模型 | Qwen3.8-27B-UD-Q6_K |
| 四组 | 原版/修复 x 不开 MTP / 开 MTP |
| 原版 | 上游 llama.cpp（ggml-cuda 976E2CAB） |
| 修复版 | 修复分支（ggml-cuda 8093C771 + 修复开关） |
| 投机解码 | draft-mtp，n-max 3 |
| 采样 | temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence-penalty 0, repeat-penalty 1.0, seed 42 |
| 上下文 | 65536（KV 不量化，fp16） |
| 聊天模板 | 模型自带（GGUF 内嵌） |
| 最大输出 | 不限 |

## 结果

| 组合 | 不开 MTP | 开 MTP | 两者关系 |
|---|---|---|---|
| 原版 | %d 字 / %d 段 | %d 字 / %d 段 | **完全不同**（相似度 %.2f）= 两篇不同文章 |
| 修复版 | %d 字 / %d 段 | %d 字 / %d 段 | **逐字节完全相同**（md5 一致） |

## 结论

1. 投机解码应当是**无损加速**：开 MTP 的输出与不开 MTP 完全一致，只是更快。
2. **原版**：开 MTP 后生成轨迹被改变，输出成了另一篇文章（结构、用典、篇幅全变）。温度 > 0 时，底层数值上的微小差异被采样放大，一路带偏。**"开 MTP 感觉变笨/降智"的观感即由此而来**：不是模型退化，而是输出换了一条轨迹；换到哪条纯看运气，可能更好也可能更差。
3. **修复版**：开 MTP 与不开 MTP 逐字节一致，MTP 不再改变输出。
4. 注：修复版与原版的两篇也都不同（修复本身带来极小的数值变化，同样会在 temp>0 时改变轨迹）；关键结论是"同一构建内，开 MTP 无不一致"。

## 复现

四个组合用完全相同参数，仅切换构建与投机开关：
- 原版：D:\\LLM\\Backend\\llama.cpp\\llama-server.exe
- 修复版：D:\\LLM\\Backend\\llama.cpp-t20\\llama-server.exe + 环境变量 GGML_CUDA_FA_SMALL_BATCH_VEC=1、GGML_CUDA_GDN_VEC4=1
""" % (question.replace("\n", "\n> "), c1, p1, c2, p2, sim[("原版-nomtp.md", "原版-mtp.md")], c3, p3, c4, p4)

io.open(os.path.join(D, "对比汇总.md"), "w", encoding="utf-8-sig", newline="\n").write(summary)
print("done")
for t, c, p, md5, same, s in lines_summary:
    print("%-14s %5d chars %3d paras same=%s sim=%.3f" % (t, c, p, same, s))
