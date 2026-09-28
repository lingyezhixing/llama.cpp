- **更新 (2026-09-27, 整理并入 master)**: 分支 `t32-stage2` (14 提交, head `e7eea21ea`) 已整理为 2 个干净提交并入本地 master (fast-forward, 沿用阶段 1 惯例):
  `85fba497d server : add kv tree storage for attention KV` (模块 + tools/server CMake) / `4f2631088 tests : add kv tree storage harness` (harness + tests CMake);
  合并结果 tree hash `44728e479b1acbf6e6f0d045c9e9db596950909c` 与已验证状态逐字节一致; 合并后在 master 上重建并重跑: logic 18/18, 2B model 43/43, 2B accept 42/42 (全部 exit 0);
  分支 `t32-stage2`/`t32-stage2-tidy` 已删除 (原 14 提交历史保留于 reflog, head `e7eea21ea`, 未 push 过); 本地 master 领先 origin/master 2 个提交, **未 push**; 部署未动 (生产仍 `ba41cccec`).
