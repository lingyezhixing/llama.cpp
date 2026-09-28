- **更新 (2026-09-27 夜, 终审)**: 全分支终审 (12 提交) 结论 "With fixes", 3 条 Important 已修并复审通过 (全部 ADDRESSED, 无新增 Critical/Important):
  ① park 早退静默丢弃检查点 -> WRN + `anchors_skipped` 计数; ② `promote_prune` 改为按捕获链作用域 (防跨分支误删);
  ③ 新增非对齐恢复真机场景 (部分尾块匹配 + 部分块装载 + `seq_rm(C,-1)` 裁剪 + 非对齐锚点).
  最终 head `e7eea21ea` (14 提交); 测试 (最终 head 重跑): logic 18/18, 2B model 43/43, 3B control 43/43, 2B accept 42/42;
  归档已按最终 head 刷新 (`t32-stage2-model.txt` / `t32-stage2-accept.txt`), SDD 证据归档至 `artifacts\t32-stage2-sdd\`.
  未 push, 未部署 (生产仍 `ba41cccec`).
