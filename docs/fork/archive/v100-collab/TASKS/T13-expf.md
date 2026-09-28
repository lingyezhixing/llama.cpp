# T13: `__expf` (GDN) 集成

状态: **CANCELLED (用户 2026-09-23: 砍, 保 100% 零漂移)**; 原为 Q10 采纳项, 规格保留备用
预期: GDN kernel 0.826 -> ~0.73ms/层 (T03 排查实测 -10%); e2e +0.7%

## 步骤

1. `gated_delta_net.cu` 内 `expf` -> `__expf` (改动点明确; 覆盖 kernel 内全部 exp 调用, 保持 fp32 状态)
2. kernel 级验证: GDN 时间前后对比 (nsys, >=2 forward), 目标 -10% 量级
3. **PPL 门槛 + 200 token 生成检查 (主验证点: fast-math 精度风险)**
4. 通过 -> 入库 + patch 归档; 不通过 (PPL 超门槛或生成异常) -> 回退 + 回报
