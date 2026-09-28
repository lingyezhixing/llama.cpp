import io

sec = """

### 9. T20 补充: 上游 / T19 / T20 三臂对照 (no spec, greedy, 250 token)

用户要求: 检查 T20 是否比 T19 引入更多与上游的不一致 (d0 + d8192)。四臂:
`stock` (上游 976E2CAB) / `t19` (交付 7F1B9B24) / `t20` (低代价修复版 8093C771, 默认) /
`t20fix` (T20 + S1/S2 env 全开)。命令: server `-c 4096|12288 -ctv q8_0 --seed 42` +
`/completion` (temp 0/top-k 1/seed 42/return_tokens; d8192 加 ignore_eos), 250 token。

| 对比 | d0 (creative prompt) | d8192 (8031 token prompt) |
|---|---|---|
| stock vs **t20** | **全等** | **全等** |
| stock vs t20fix | 全等 | 147 分叉 |
| stock vs t19 | **207 分叉** | 全等 |
| t19 vs t20 | 207 分叉 | 全等 |
| t19 vs t20fix | 207 分叉 | 147 分叉 |
| t20 vs t20fix | 全等 | 147 分叉 |

- **T20 (默认) 没有引入更多不一致**: 两个深度都与上游逐 token 全等; T19 在 d0 有 1 处分叉
- `t20fix` 的 d8192@147 分叉来自 S2 (GDN decode 改 vec4 布局) 的 1-ulp 级数值改动 ->
  只会翻转 <=1e-3 级 gap 的极端近并列; 而 T19 在 207 的翻转对应 0.068 nats 的 gap (说明 T19 的
  数值差更大, 来源应为 sm70 FA 配置 / T12+T16 归并)
- 结论: 所有构建之间的分叉都是"近并列 + ulp 级数值差"的硬币翻转, 数量级相当且都不影响质量门槛
  (PPL 4.3562 vs 4.3572); 修 S2 必然改变 decode 数值 (vec4-for-all 或 scalar-for-all 都会改一边),
  这是让 MTP spec 与 no-spec 逐 token 一致的不可避免代价
- 附带: 早前 d0 四 prompt (T19 vs stock): p1_factual @97 / p2_creative @207 分叉, p3_code / p4_math 全等
"""

p = r"D:\LLM\Backend\v100-collab\RESULTS.md"
s = io.open(p, encoding="utf-8", newline="").read()
s = s.rstrip() + sec
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("results ok", len(s))
