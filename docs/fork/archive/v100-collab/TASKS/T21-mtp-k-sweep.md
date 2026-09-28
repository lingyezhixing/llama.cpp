# T21: MTP K 扫描 + greedy drafting 检查 - 用户批准 2026-09-23

状态: **APPROVED (用户 2026-09-23)**; 执行者: implementer; 口径 = T19 交付, **无代码改动** (若判断需要改代码先报)
动机: skinny 实测"接受率随上下文衰减、k 过大反亏"; T20 §4 已在修复版给过 32k/128k n-max3 数据
(32k 24.26 vs none 15.85 = +53% / 128k 10.58 vs 7.23 = +46%), 但需在**交付版**上补全 K 扫描与生产建议;
并检查 drafter 采样模式 (greedy 化可再提接受率)。

## 1. K 扫描 (server; greedy seed42; return_tokens)
- 深度: d0 / d32768 / d131072 (长文优先); 生成 100-200 token
- 配置: none / n-max 1 / 2 / 3 (/ 4 若支持)
- 记录: tg t/s, 接受率, 平均每轮 token, 显存峰值 (recurrent state 行数 = 1+K)
- 热漂移: 同 session 交替, 长点轮换顺序

## 2. greedy drafting 检查
- 查 `--spec-draft-*` 现有旋钮与 `common/speculative.cpp` 的 draft-mtp 实现: drafter 是 greedy 还是采样?
- 若有 greedy 选项 (或可加 env): 测接受率变化 (skinny 报告 +10-25 点); 若无旋钮, 记录并给"是否值得改"判断

## 3. 交付物
- RESULTS 章节 "T21": 深度 x K x t/s x 接受率 表 + 显存表
- 结论: 每个深度推荐的生产 K; 是否建议改 drafter 采样 (预估收益/工作量)
