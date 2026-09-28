# T19-L: 交付瘦身 (丢 A2/A3, 保 A1/A4) - 用户批准 2026-09-23

状态: **APPROVED (用户 2026-09-23)**; 执行者: implementer
动机: 减少上游 rebase 维护面 + 消除 A2 带来的 S2 (MTP decode/verify 分叉源);
保留两项大收益 (A1 反量化 +20.6% / A4 FATTN +11.28%@128K)。

## 范围
- **保留**: A1 (`convert.cu` + `dequantize.cuh`) + A4 (`fattn-common.cuh`) + sm70 FATTN 调参提交 `3fc05594a`
- **移除**: A2 (`gated_delta_net.cu`) + A3 (`unary.cu`)
  - 预期代价: pp512 ~**-1.1%** (A2 -1.0% + A3 -0.1%); decode/长文不受影响 (A2 仅作用于 `n_tokens>1` prefill 路径)
  - 预期收益: patch 4->2, 文件 5->3; **S2 分叉源消失**; PPL 回到 ~4.3565 级 (撤销 A2 的 -0.0003, 仍远在门槛内)
- 旧 patch 归档不删除: 移入 `patches/retired/` (或 ENVIRONMENT 标注 retired)

## 步骤
1. `git checkout 3fc05594a -- ggml/src/ggml-cuda/gated_delta_net.cu ggml/src/ggml-cuda/unary.cu`
2. 重建 -> 部署 `D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll` -> **核对部署 SHA == build 输出**
3. 验收 (同 session): PPL 4.3572±0.013; pp512 (~-1% vs 948.5); pp32768 / pp131072 不回退; tg128 不回退; MTP3 快速 sanity
4. 重生成 2 个 patch (A1/A4, 基于 `e6ab7c1a4`) + 双向 apply 校验; A2/A3 patch 标 retired
5. 本地 commit (不 push); 更新 ENVIRONMENT/BOARD/STATUS (新 DLL SHA / patch 清单 / 验收数字)

## 注意
- 与 T21/T22 (零代码实测) 无冲突; 建议顺序: T21/T22 先跑 -> 本项 -> 冻结收尾
- 瘦身后 OURS vs STOCK 的差异 = A1 + A4 + sm70 调参 (长文 +35.6% 保持, 短点 ~+19% 级)

## Result (2026-09-24, implementer)

- 已入库: 交付提交 `d24474edd` (squash 后单条, 未 push); 工作区干净; 部署 DLL SHA256-16 `054BFFD625E37E04`
- 验收 (同 session A/B, ub512): pp512 **-1.5%** (948.1, 绝对值 ≈ 记录 948.5) / pp4096 -1.8% / pp8192 -1.2% /
  pp32768 **-0.9%** / pp131072 **-0.5%**; tg128 短点持平 (±0.2%), 长点 (d32768/d131072) 噪声内
- PPL **4.3567** (T19 重建 4.3562; 门槛 4.3572±0.013) OK; MTP3 sanity: d0 55.6 t/s acc 0.932 / 32k 49.9 t/s acc 1.0 OK
- patch: `artifacts/v100-dequant-vec.patch` + `artifacts/v100-t12t16-fattn-split.patch` (基于 `e6ab7c1a4`, 双向 apply 校验通过);
  A2/A3 -> `artifacts/retired/`
- 详见 RESULTS "T19-L"
