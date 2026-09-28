import io

q = """

---

## 2026-09-23 | T20 补充: S3 找到低代价修法, 建议直接采纳全部三项 | implementer

上一条 QUESTIONS 的选项 A/B 已合并: S3 不再需要 -31% 的 PB=1, 改用
**小批量固定切分块数** (与 batch 无关; 空块贡献严格 0) -> decode/verify 逐位一致。

- 实测代价 (tg128, 交错): d0 **-0.8%** / d32768 **-0.5%** (全部来自 S2; S3' 代价 ~0)
- 验收: 短 900 token n-max 1/2/3 全等; 32k n-max3 150 token 全等 (128k 按用户指示未测)
- 详细见 RESULTS "T20" 第 8 节

**修订后建议: 直接采纳 S1+S2+S3' 全部固化** (总代价 ~0.5-0.8%), 不需要再二选一。
待用户确认后执行: 固化代码 (S1: Volta vec_limit 2->16; S2: 去掉 n_tokens>1 分支;
S3': 已实现, 去掉 PB_FORCE 调试 env 或保留) + 全量验收 + 重建部署核对 SHA。
"""

p = r"D:\LLM\Backend\v100-collab\QUESTIONS.md"
s = io.open(p, encoding="utf-8", newline="").read()
s = s.rstrip() + q
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("questions ok", len(s))
