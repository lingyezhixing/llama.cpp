- **T32 阶段 0+1 完成 (2026-09-27, SDD)**: 分支 `t32-stage1` (6 提交, 全分支审查通过, **未 push, 未部署**).
  区间序列化 API (写/读/append+重叠拒绝) + harness; H2D 2.37 GiB/s, q8_0 口径 50200 B/token,
  `--tree-chunk`=512, 100K 4.68 GiB (~2s H2D); 小模型正确性 23/23/23/22 全 PASS + 两个回归全绿.
  资产 artifacts/t32-stage1-{bench,correctness}.txt + t32-stage1-sdd/. 待用户: 合并/push/部署 + 阶段 2 计划.
  **用户约束: 不再跑 27B, 验证只用小模型.**
