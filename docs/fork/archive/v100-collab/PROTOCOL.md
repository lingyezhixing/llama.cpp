# V100 优化协作协议 (analyst <-> implementer)

最后整理: 2026-09-22 (analyst, 清理重复追加)

## 角色

- analyst: 分析、拆解任务、提出方向、定义验收口径、复核数据。不改代码、不跑 GPU 测试。
- implementer: 实现、编译、测试、跑分、记录原始数据。不擅自改目标与口径; 疑问写 QUESTIONS.md。
- 用户: 只在需要物理操作或决策时介入, 见 NEEDS-USER.md。

## 文件地图 (D:\LLM\Backend\v100-collab)

| 文件 | 内容 | 维护方 |
|---|---|---|
| PROTOCOL.md | 本文件: 协议与硬性口径 | analyst |
| BOARD.md | **唯一入口**: 恢复指引、任务表、队列、不要重试清单、关键判断 | analyst |
| STATUS.md | 当前有效事实: 硬件/模型/常数/已实测数字/上限结论 | analyst |
| ARCHIVE.md | 历史: 用户决策记录、预测修订链、已证伪实验详情 | analyst |
| REFERENCE-1cat-vllm.md | 外部参考: 1Cat-vLLM (V100 vLLM fork) 的验证与可借鉴项 | analyst |
| OPTIONS.md | **有提升的方案总表** (最终统一回顾材料: A 已落地 / B 待裁决 / C 工程外 / D 外部 / E 不推荐 / F 排序 / G 裁决清单) | analyst |
| ENVIRONMENT.md | 工作区状态、补丁、构建部署、工具脚本、备份 DLL | implementer 提供, analyst 整理 |
| TASKS/T0X-*.md | 任务规格 + 结果 | spec = analyst, Result = implementer |
| RESULTS.md | **原始时间顺序日志 (追加式, 不改历史)** | implementer |
| QUESTIONS.md | implementer 提问 + analyst 回答 | 各自追加 |
| NEEDS-USER.md | 需用户操作/决策的事项 | 各自追加 |

## 状态机

`PROPOSED -> APPROVED -> RUNNING -> BLOCKED -> DONE -> VERIFIED | REJECTED | SKIPPED`

## 测量口径 (硬性)

- 模型: `<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf`
- 设备: `CUDA_VISIBLE_DEVICES=1` (Tesla V100-SXM2-32GB)
- 基线命令: `llama-bench -m <model> -ngl 99 -fa on -ctv q8_0 -ub 512 -p 512,4096,8192 -n 128 -r 3`
- 每条结果必须记录: git hash、工作区 patch、SM clock/功耗、是否锁频、命令原文
- 噪声规则: 单次测量 ±0.5% 不可作为判据; 有疑义用 A/B 交替换 DLL (两轮)
- 热漂移 (2026-09-23 实测): 本机连续负载下 tg128 26.6 -> 24.4; **长点验收必须轮换 A/B 顺序**; tg128 用冷却后复核值
- **长短 prefill 防回归**: 每个验收必须同时给 pp512/4096/8192; 涉及注意力/显存/ub/上下文必须加 pp32768 或 pp4096@depth32k (基线 775 / 649); 只报短 prefill 提升不算通过; **长文代表点 (用户 Q14 指定): pp8192@depth128k (128K 深度 8k pp, 基线待测) - 长上下文相关改动必须给这一项; 不做完整 pp128k**
- 正确性: 当前 PPL = 4.3569 (重结合后), 原基线 4.3572; 门槛 `|PPL - 4.3572| <= 0.013` + 200 token 生成无乱码/重复

## 物理边界 (不做无效尝试)

- decode: 权重 21.97GB; 实际可用带宽 825-850 GB/s -> 纯 kernel 上限 **28-29 t/s**, 现状 26.6 已接近
- 有效 decode (含投机) 由用户生产配置承担: `--spec-type draft-mtp --spec-draft-n-max 3` 实测 25.3 -> 42.7 t/s, 不在工程范围
- prefill: cuBLAS GEMM 占 349ms/532ms 且已到刀口 (加权 72% 峰值, 调用侧仅 cublasLt 待仲裁);
  **融合路线已永久关闭** (T02 Gate A + T09-A: 去解包 1.569ms 仍慢于 baseline 1.404ms; 瓶颈是 smem staging
  的延迟/发射而非 DRAM 带宽, "少读 2.6x" 无法转成时间)
- dequant 路径本身已到 92% 实际带宽 (88.5ms/ubatch), **这里没有剩余空间**
- ub512 上限: ~1000-1050 (T08 + T07剩余 + T03 chunked); 1500 明确不可达

## 协作规则

1. 主线同一时间一个; 小任务可并行; 动同一文件前在 BOARD 标 RUNNING
2. 追加式写入, 不重写历史; 整理由 analyst 统一做
3. 每个任务结束必须给可比数字; 预期不成立就写 BLOCKED/REJECTED + 原因, 不硬凑
4. 数值路径: 允许微小漂移 (见门槛), 不接受导致模型降智的偏移
