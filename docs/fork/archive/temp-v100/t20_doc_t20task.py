import io

sec = """

---

## Result (2026-09-23, implementer) - **DONE, 暂停等用户裁决**

独立排查完成 (未盲信旧报告; 旧报告 5 条主张中 4 条被推翻, 见 RESULTS "T20" 第 7 节)。

### 分叉源 (3 个, 全部有独立证据链)
1. **S1 FA VEC/TILE** (旧报告 H2, 确认): Volta decode n_q=1 走 VEC, verify n_q=2..4 走 TILE -> 首 FA 层即分叉
2. **S2 GDN vec4 布局分支** (**本 fork T03/A2 引入, 旧报告漏项**): n_tokens>1 时 lane 行映射变 (4*lane+r vs r*32+lane)
   -> warp 归约顺序不同 -> ulp 级数据相关差异; 与 FA 无关, 旧报告"GDN batch-invariant"不成立
3. **S3 FA VEC split-K 的 padding 边界** (**新发现, 上游设计属性**): padded n_kv 跨 256 时 ntiles_KV/parallel_blocks
   变化 -> KV 交错切分顺序变 -> 1e-6 级差; decode 与 verify 的 padding 天然不同 -> 每次跨 256 边界一次扰动

### 修复与验收 (S1+S2+S3 全开)
- 短上下文 900 token / 32k 150 token / 128k 100 token: **n-max 1/2/3 与无 spec 逐 token 完全一致**
- 回滚/快照路径独立验证精确 (工具: batch+seq_rm+probe, rollback 1/2/3, rs=0..3, 全行逐位一致)
- PPL = 4.3562 (与交付一致); 确定性复测通过

### 代价 (llama-bench 同 session 交错)
- S1 ~0; S2 tg128 d0 -0.6% / d32768 -1.3%; **S3 -31% @d32768 (不能默认开)**
- 仅 S1+S2 时残余分叉 = 极端近并列 (<=0.02 nats) 翻转, 900 token 内 1-3 次, 两条续写均合法

### 交付物
- RESULTS "T20" (完整证据链/命令/原始数据); 工具增强 (batch-invariance --seq-a/--probe/--rollback, 未提交);
  实验 patch `artifacts/t20-work.patch`; 调试 DLL SHA `DE3F5E50...` (部署已回滚 T19)
- **待用户裁决**: 是否采纳修复 (S1+S2 固化 + S3 可选 env), 见 QUESTIONS
"""

p = r"D:\LLM\Backend\v100-collab\TASKS\T20-mtp-trajectory.md"
s = io.open(p, encoding="utf-8", newline="").read()
s = s.rstrip() + sec
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok", len(s))
