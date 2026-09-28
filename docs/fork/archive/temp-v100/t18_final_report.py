import io

t = """
---

# T18 结案 (2026-09-23): **生产 A/B 未达门槛 (128k 点 ~0%) -> 不合并, 已回退**

## 生产 A/B (A = T12+T16 `453E2911`; B = T18 ncols=32+dispatch `2D1D56C4`; 轮换顺序)

| 点 | 轮 | A | B | 相对 |
|---|---:|---:|---:|---:|
| short pp512 | 1/2 | 946.94 / 943.65 | 947.60 / 944.10 | +0.07 / +0.05% |
| short pp4096 | 1/2 | 937.66 / 926.30 | 931.04 / 930.12 | -0.71 / +0.41% |
| short pp8192 | 1/2 | 914.02 / 904.28 | 907.93 / 908.99 | -0.67 / +0.52% |
| tg128 | 1/2 | 26.63 / 25.67 | 26.55 / 26.30 | -0.3 / +2.4% (漂移) |
| **depth32k** | 1/2 | 680.03 / 678.53 | **688.94 / 690.04** | **+1.31 / +1.70% (均值 +1.50%)** |
| **pp32768** | 1/2 | 797.32 / 789.56 | **803.47 / 801.92** | **+0.77 / +1.57% (均值 +1.17%)** |
| **pp8192@depth128k** | 1/2 | 374.58 / 372.27 | 371.59 / 376.12 | **-0.80 / +1.03% (均值 +0.11%)** |

- 短点差在 ±0.5% 漂移内 (本机同 session 逐轮 A 自身漂移可达 1%)
- **主判据 128k 点: 均值 +0.1% < 门槛 +2% -> 不合并** (Q15 规则; depth32k 也未达 +2% 的复核线)

## 决定性证据: nsys 逐 launch 隔离长文阶段 (duration > 30ms, l~135k)

| cfg | kernel 变体 | grid | 实例数 | median | mean |
|---|---|---:|---:|---:|---:|
| A | `flash_attn_ext_f16<256,256,32,2>` | 384 | 1755 | 40.52 ms | 40.28 ms |
| B | `flash_attn_ext_f16<256,256,16,2>` | 768 | 1641 | 40.77 ms | 40.38 ms |

- dispatch 覆盖**确实生效** (B 的变体/grid 都对), 但长 l 下 **B 反而慢 0.3-0.6%**
- 128k 点构成 (B, nsys): FA 41.7% / cutlass GEMM 35.2% / q6_K dequant 6.7% / GDN 4.6% / q5_K 3.1% / 其他 ~8%
- 即: ncols=32 的收益在 l<=35k 真实 (~1.05x FA, 由 depth32k/pp32768 的 +1.2~1.5% 反推),
  但 **l>=100k 完全消失** (harness 在 l=100k 测得的 1.10x 在生产不复现)

## 根因 (harness 保真缺口)

- Stage 0 harness 用 fp16 K/V + 独立热循环, 只在 l=35072 与生产对过 (+2.5%); **长 l 未对**
- 生产 `-ctv q8_0`: 内核内无 dequant 代码, V 由独立 `dequantize_block_q8_0_f16` 先转 fp16 (1.2%, 可忽略)
  -> 不是 V 精度问题; 更可能是长 l 下内核转为 K/V 流式受限, 小 tile 预取深度不足抵消了 2 CTA 的优势
- 教训: FA 类 harness 必须在**目标 l** 上做保真对照 (不能只在短 l 对)

## 状态

- 部署 DLL 已回退 `453E2911` (T12+T16, 门槛通过版); T18 构建留档 `%TEMP%/v100/ggml-cuda-T18.dll` (`2D1D56C4`)
- 源码工作区: `fattn.cu` (dispatch 覆盖) + `fattn-mma-f16.cuh` (Volta ncols=32 行) 暂留未撤, 待裁决; 若否决则 `git checkout` 这两文件
- PPL/生成未跑 (门槛未达, 不需要)
- T18 全部产物: harness + 5 变体 exe + ncu 日志 2 + nsys 2 + 逐 launch CSV (artifacts/ 与 %TEMP%/v100)
"""
io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

q = """
---

## 2026-09-23 | T18 终裁请求: 生产未达门槛 (128k +0.1%), 已回退部署 | implementer

结论 (完整数据见 RESULTS "T18 结案"):

1. **128k 点 (主判据) = +0.11%** (两轮: -0.80% / +1.03%, 漂移 ±1%) -> **未达 Q15 的 +2% 门槛**
2. depth32k **+1.50%** / pp32768 **+1.17%** (真实且两轮一致) / 短点与 tg128 无回退
3. **决定性证据**: nsys 逐 launch, l~135k 的 FA (>30ms): A(ncols=64) 40.28ms vs B(ncols=32) 40.38ms
   -> **长 l 下 B 慢 0.3-0.6%**; ncols=32 的收益在 l<=35k 真实, l>=100k 消失
4. 根因: harness (fp16 V + 独立热循环) 长 l 保真缺口; 生产 q8_0 V 是独立 dequant 内核 (1.2%, 无关)
5. 已按"未达门槛"处理: 部署 DLL 回退 `453E2911` (T12+T16); 源码 T18 两处改动暂留待裁

请裁决:
- **A (我倾向): 撤销 T18 改动** (git checkout fattn.cu / fattn-mma-f16.cuh), T18 归档为 "长文 FA 对 tile 配置不敏感,
  瓶颈在流式/预取"; 后续若再碰 FA, 先在目标 l (128k) 做 harness 保真对照
- **B: 破格保留** (depth32k +1.5% / pp32768 +1.17% / 128k 无损失, 零回退) —— 需补跑 PPL+生成 (0.5h) 再入库
- **C: 继续追长文**: 方向应转向 "长 l 流式/预取" 而非 tile 配置 (我估需要新 harness 口径 + 2-3 天, 且 P(>+2%) 未知)

(T17 已 CLOSED; T12+T16 交付不变; 工作区除 T18 两文件外干净)
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(q)
print('QUESTIONS appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T18-fa-mma-rewrite.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## 最终 Result (2026-09-23): 生产未达门槛 -> 不合并, 已回退

- 生产 A/B: 128k 点 **+0.11%** (门槛 +2%) / depth32k +1.50% / pp32768 +1.17% / 短点 tg128 无回退
- nsys 逐 launch (l~135k, >30ms): A 40.28ms vs B 40.38ms -> **长 l 下 ncols=32 慢 0.3-0.6%** (harness 1.10x 不复现)
- 128k 构成: FA 41.7% / GEMM 35.2% / dequant 9.8% / GDN 4.6% / 其他 8.7%
- 根因: harness 长 l 保真缺口 (只在 l=35k 对过生产); 长 l 瓶颈 = K/V 流式, 非 tile 配置
- 处置: 部署回退 `453E2911`; 源码两处改动待裁 (QUESTIONS); T18 归档
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T18 updated')
