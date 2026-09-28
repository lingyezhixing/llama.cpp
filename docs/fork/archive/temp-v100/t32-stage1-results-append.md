## T32 阶段 0+1: 区间序列化 API (implementer, 2026-09-27; SDD 子代理执行 + 两级审查)

- 分支 `t32-stage1` (从 `ba41cccec`), **6 提交, 未 push, 未部署**; 全分支审查通过 (1 轮终审修复 + 限定复审):
  `82674b9f5` harness(h2d)+CMake / `3d05ef570` 区间写 API / `df4bacd74` 修复(sentinel+idx 拒绝) /
  `4ae887b1c` 区间读 API(append/重叠拒绝) / `dfd8a7035` 区间吞吐基准 / `c5a91c15e` 终审修复(归档+mirrored 守卫+head 回退)
- **实测 (27B, 设备 1)**: H2D 2.37 GiB/s / D2H 2.60 GiB/s (32K 全量 state 2.10 GiB f16, PCIe x4);
  q8_0 口径 target KV **50200 B/token** -> 512 块 24.5 MiB, 100K 4.68 GiB (~2.0s H2D / ~1.8s D2H);
  区间 API 吞吐: chunk 512/1024/2048 读 12.8/23.2/41.5 ms/chunk (1896/2063/2309 MiB/s),
  写 20.3/34.8/60.1 ms/chunk (1198/1378/1594 MiB/s); ms/chunk 近似随 chunk 线性 -> **`--tree-chunk` 默认锁 512**
- **正确性 (小模型)**: 2B hybrid 23/23, 2B `-kvu` 23/23, 3B 纯 attention 22/22 全 PASS (含逐字节等价、分段 append==全量恢复、
  重叠拒绝、截断 blob 失败清理、契约反例); 归档 `artifacts/t32-stage1-correctness.txt`
- **回归 (c5a91c15e 树上复跑)**: `test-state-restore-fragmented` SUCCESS + `test-save-load-state` All tests passed, 均 exit 0
- 资产: `artifacts/t32-stage0-h2d.txt`, `t32-stage1-bench.txt`, `t32-stage1-correctness.txt`,
  `t32-stage1-worktree.patch` (ba41cccec..c5a91c15e), `t32-stage1-sdd/` (SDD 报告 + ledger 含全部裁决)
- 待用户决定: 合并回 master / push / 部署 (生产仍为 `ba41cccec`); 阶段 2 (树模块) 计划待出
