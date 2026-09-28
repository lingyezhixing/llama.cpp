import io

q = """

---

## 2026-09-23 | T20 完成, 请裁决修复采纳方式 | implementer

T20 (MTP 轨迹一致性独立排查) 已按用户要求独立完成 (不盲信旧报告; 旧报告 5 条主张 4 条被推翻)。
完整证据链见 RESULTS "T20"。要点:

**分叉源 3 个**:
- S1 FA VEC(n_q=1) vs TILE(n_q=2..4) - 旧报告 H2, 确认
- S2 **GDN vec4 布局分支** - 本 fork T03/A2 引入 (n_tokens>1 换 lane 行映射 -> warp 归约顺序变);
  旧报告"GDN batch-invariant"不成立 (旧 base 上恰好未触发)
- S3 **FA VEC split-K 的 padding 边界** - 上游设计属性 (padded n_kv 跨 256 -> parallel_blocks/交错切分变);
  server logits dump 证明首个差在采样点 242 (position 255, 首个 256 边界)

**验收 (三源全修)**: n-max 1/2/3 与无 spec **逐 token 完全一致** - 短 900 / 32k 150 / 128k 100 token;
回滚+快照路径工具级逐位验证精确; PPL 4.3562 (同交付); 确定性复测通过

**代价** (llama-bench 同 session 交错):
- S1: 短点 ~0 (nmax1/2/3 36.9/39.1/37.2 vs 36.9/39.5/36.9)
- S2: tg128 d0 **-0.6%** / d32768 **-1.3%** (交错复测)
- S3: tg128 d32768 **-31%** (23.10 -> 15.83; 128k 未测, 预计更差) - 因 KV 切分是长文 decode 的主要并行度来源

**请裁决 (三选一)**:
- **A (我倾向): 固化 S1+S2 (代码默认), S3 只留 env 可选** - 收益: 消除 fork 引入的 S2 + 恢复旧报告级修复;
  残余 = S3 的极端近并列翻转 (900 token 内 1-3 次, top1-top2 <=0.02 nats, 两条续写均合法), 生产可接受
- **B: S1+S2+S3 全固化** - 100% 逐 token 一致, 但长文 decode -31% @32k (与"速度不回退"验收冲突)
- **C: 全部不采纳, 仅归档** - 保留现状 (分叉最频繁, 900 token 内 3+ 次, 含 1e-3 级 FA 差异)

若选 A/B, 需要: 固化代码 (S2: 删除 `n_tokens > 1` 分支一律 vec4; S1: Volta 分支 `vec_limit` 2->16;
S3: env `GGML_CUDA_FATTN_PB_FORCE` 或更名) + 全量验收 (PPL/pp 曲线/tg/长文点/生成) + 重建部署核对 SHA。
工作区已备好实验 patch (`artifacts/t20-work.patch`), 部署已回滚 T19。
"""

p = r"D:\LLM\Backend\v100-collab\QUESTIONS.md"
s = io.open(p, encoding="utf-8", newline="").read()
s = s.rstrip() + q
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok", len(s))
