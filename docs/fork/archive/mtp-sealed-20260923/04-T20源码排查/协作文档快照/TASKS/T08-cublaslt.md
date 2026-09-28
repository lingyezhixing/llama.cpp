# T08: cublasLt 集成 (per-shape algo)

状态: APPROVED (measurement-first, analyst 2026-09-22)
优先级: P1 (T02 REJECTED 后, prefill 唯一有意义的杠杆)
预期收益: 整机 prefill +2-5% (取决于逐形状加权, **尚未证实**)

## 背景与矛盾点 (必须先仲裁)

| 来源 | gate/up (512, 17408, 5120) 同形状数字 |
|---|---|
| T01 standalone ABAB (预热交错) | GemmEx DEFAULT_TENSOR_OP 84.2 vs cublasLt 84.1 (**无差**) |
| T02 Gate A harness | GemmEx 默认 77.4 vs ALGO8-15 83.0 vs cublasLt heuristic#0 84.3 (**+9%**) |

两次测量互相矛盾。若后者为真, 说明 llama.cpp 当前的默认 algo 调用不是最优。
**警告 (已实测)**: 把 ALGO8 提示盲目套进 llama.cpp -> pp512 从 958 掉到 310 (-68%);
algo hint 是形状相关的, 绝不能全局设置。

## Step 1: 权威 ABAB (先做, 0.5 天, 不碰 llama.cpp)

- 同一个 harness、同一 layout、预热 + 交错 >= 5 轮, 测**模型全部 GEMM 形状**:
  `gemmEx 默认` vs `cublasLt heuristic algo` (记录 algo/tile/splitK 号)
- 输出逐形状表 + 按 FLOPs 加权的整机增益估算
- 判定: 加权增益 >= 3% 才进 Step 2; < 3% 直接 REJECTED (不值得碰调用侧)

## Step 2: 集成 (仅 Step 1 达标后)

- 用 cublasLt API, 逐形状 query heuristic 并缓存 (键: m,n,k,layout); 若某形状比默认慢则永久回退该形状
- env flag 开关 (如 `GGML_CUDA_LT=1`), 全形状回归通过前默认关闭
- 严禁全局 algo hint (已证 -68%)
- 注意 batched 3D 调用 (llama.cpp 现有 batched 路径) 也要覆盖或明确排除

## 验收

- 端到端 A/B (交替 DLL, 两轮): pp512 提升 >= 2%, 且 pp4096/8192 不回退
- PPL 门槛不变 (|PPL-4.3572| <= 0.013)
- 逐形状表无回退; 任何形状回退则该项回退, 不硬上

## 风险

- cublasLt 需要 workspace 管理; 与现有 4MB handle workspace 的交互要验证
- 逐形状 heuristic 首次调用有开销; 需在 warmup/加载阶段完成
- 形状数量多 (64 层 x 每层 3-5 个 GEMM + lm_head), 缓存与回落逻辑要简单可靠

---

## Result (implementer, 2026-09-22): Step 1 = REJECTED (加权 -1.11%, 不达 +3% 门槛)

- 完整数据见 RESULTS.md 的 "T08 Step 1" 章节; harness: `artifacts/t08_arb.cu` (+ t08_one/t08_interf 诊断)
- 真实方向 (llama.cpp 用的 m=out_dim/n=tokens) 下, 全部 8 个形状: cublasLt 最好也只是打平,
  gate/up 反而 -2.9%; 按模型时间加权 **-1.11% of GEMM -> -0.67% 整机 prefill** -> 判定 REJECTED
- **Step 2 (集成) 不做**; llama.cpp 调用侧保持现状 (0 行改动)
- T01/T02 矛盾已定位: ① T02 harness 用转置方向 (m=tokens) -> 那里 cublasLt 确实 +6.7%;
  ② math mode 差异 (llama.cpp 的 TF32_TENSOR_OP 已经是最优); ③ 预热/时钟 (真实混合负载 1447MHz vs 纯 GEMM 1530MHz)
- 修正 T02/T09 的盈亏平衡线: 真实方向下约 66-70 TF (原 66 是转置方向 baseline), 两个否决结论不变
- 副产品否证 (记入不要重试): 显式 ALGO0..15 在真实方向全部慢于 DEFAULT (最好 62.6 vs 82.1 TF)

## Analyst 复核 (2026-09-22): REJECTED 接受, 结案

- 数据与判定复核通过: 真实方向 8 形状、5 轮交错、加权 -1.11% of GEMM -> 不进 Step 2, 调用侧 0 行改动, 正确
- T01 结论恢复为 "调用侧无空间"; T01/T02 矛盾解释 (转置方向 + math mode + 时钟) 已并入 STATUS/ARCHIVE 第 9 节
- 队列更新: T08 关闭; 下一批 = T10 attention 侦察 + T07 rms_norm (并行) -> T03 chunked (timebox) -> T05 可选
