## Result (stage 1 验证矩阵, 2026-09-27; 小模型, CUDA device 0)

命令: `test-t32-range.exe --mode correctness`; 2B (Qwen3.5-2B Q4_K_XL) 与 3B (Qwen2.5-Coder-3B IQ4_XS)
矩阵: np=1/3 x {f16, V-q8_0, V-q4_0, K+V-q8_0} x kvu 开/关 (18 run, 全部 exit 0)
归档: `artifacts/t32-stage1-correctness-matrix.txt` (每 run 完整 stderr + logits 距离)

结论:
- **np=1 (生产口径)**: 全量化组合 18/18 PASS, **logits 逐位相同 (max|diff| = 0.000000)**; 含"不恢复基线 vs
  分段恢复 vs 全量恢复"直接对比 (harness 新增 np=1 自比对路径).
- **np=3 非 unified**: 全量化组合 PASS, **logits 0.000000** (每 seq 独立 stream -> cell 布局一致).
- **np=3 unified**: 数据级检查全 PASS; 跨序列"逐 token 相等"判据改为显式 SKIP (cell 布局不同, 见下).
- unified 下量化 V 的 logits 距离: 2B q8_0 0.25 / 3B q4_0 0.42 / 3B q8_0 0.58-0.63; **连旧的全量恢复
  路径与基线也差 0.60** (只是碰巧没翻 token) -> 偏差来自 unified 共享 stream 下不同 cell 布局的
  FA 归约顺序, **不是 range API**; 数据级检查 (逐字节 payload 相等 / 恢复成功 / 重叠拒绝 / 截断清理)
  在所有配置全 PASS. 引擎自身测试惯例 (`test-save-load-state`) 对跨路径比较用 NMSE 容差而非逐位.
- 对树 (阶段 3, R1 非 unified) 无影响; 生产 np=1 逐位对齐.
- harness 变更: np=1 自比对路径 (基线 vs 分段 vs 全量), logits max|diff| 打印, unified 下布局敏感判据
  显式 SKIP; 提交 `e19dfca26` (待填) 之后所有矩阵 run exit 0.
