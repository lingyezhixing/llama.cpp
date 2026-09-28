# T07: elementwise / 小 kernel 向量化

状态: DONE (silu 已入库; rms_norm 向量化 = REJECTED, kernel 级 +5.7%, 见下)
预期收益 (修订): 整机 +0.5-1.0% (原估 +2-3% 被实测下调)

## 已完成

- silu (`unary_gated_op_kernel_f32_vec4`, float4 + 标量回退): kernel 11.42 -> 10.84ms (-5.1%), 端到端噪声内
- patch: `D:\LLM\Backend\patches\v100-t07-silu-vec4.patch`

## 实测修正 (重要)

| kernel | 有效带宽 | 判读 |
|---|---:|---|
| unary_gated (silu) | ~1.05 TB/s | 已超 DRAM 峰值 (L2 掩护), 向量化只 -5% |
| convert_unary_cont_vec4 | 549 GB/s | 已是 vec4; 496 次 x 29us, 固定开销主导 |
| rms_norm (3 变体) | 590 GB/s | 两趟结构 + 归约限制 |
| k_bin_bcast (add) | 625 GB/s | |
| concat_non_cont | 680 GB/s | |

结论: 多数已近内存墙或已被 L2 掩护, "只跑 58-70% BW" 的推算高估了收益。

## 剩余 (可选, 收益小)

- rms_norm 向量化: 预估 590 -> 750 GB/s (+0.4%)
- convert_unary 想省必须减少调用次数 (融合进 GEMM 前置), 不建议投入
- 其余不再投入

## 验收 (若做剩余项)

- 每个改动单独给 kernel 级 + 端到端数字; 端到端需 A/B 交替 (两轮) 才能定论
- PPL 门槛不变; pp512/4096/8192 三项都要给


---

## Result 补充 (implementer, 2026-09-22): silu vec4 对数值无影响 (逐位相同)

- 3 路 DLL A/B 中, "dequant + GDN vec4" (无 silu vec4) 与交付版 (含 silu vec4) 的 PPL **都是 4.3569**,
  且 silu kernel 的每个元素表达式与标量版完全一致 (`op(x)*g`, 无重结合空间) -> **逐位相同, PPL 贡献 0**
- 交付 PPL 4.3569 的全部偏移来自 T03 的 GDN vec4 (见 TASKS/T03 的 Result 补充)

---

## Result 补充 (implementer, 2026-09-22): rms_norm 向量化 = REJECTED

- Kernel 级 (nsys): 合计 174.81 -> **184.85 ms (+5.7% 更慢)**; mul 变体 +6.6%/+10.5%, 仅 plain 变体 -5.5%
- 端到端交替 4 轮: pp512 +0.38% / pp4096 +0.32% / pp8192 +0.10%, 但轮间漂移 0.4-0.75% -> 噪声级,
  与 kernel 级矛盾 -> 按验收规则拒绝
- PPL: 4.3563 (门槛内, 无质量问题)
- 根因: <1024> 变体已 930 GB/s = DRAM 极限; <256> 变体延迟受限 (57 GB/s, 固定开销主导);
  理论上限仅 0.1-0.2% 端到端, 低于噪声地板
- patch 归档 `artifacts/t07_rms_norm_vec4_REJECTED.patch`, 工作区已回退 (仍为 4 文件)
- **T07 全部关闭**
