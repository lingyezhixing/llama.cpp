### T32 阶段 1 验证矩阵 (implementer, 2026-09-27; 小模型 device 0)

18 run: 2B/3B x np=1/3 x {f16, V-q8_0, V-q4_0, K+V-q8_0} x kvu 开/关, 全部 exit 0; 归档
`artifacts/t32-stage1-correctness-matrix.txt`.
- np=1 与 np>=2 非 unified: 数据级 + 逐 token 全 PASS, **logits 0.000000** (布局一致).
- np>=2 unified: 数据级全 PASS; 逐 token 跨序列相等判据显式 SKIP (不同 cell 布局 -> FA 归约顺序差异;
  实测 logits 距离 0.25-0.63, 旧的全量恢复路径同样偏离基线 0.60 -> 与 range API 无关).
- 结论: 生产口径 (np=1) 与树口径 (非 unified) 逐位对齐; unified 多序列不承诺跨布局逐位.
