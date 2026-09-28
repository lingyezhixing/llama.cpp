# T18: Volta FA mma 内核效率重写 (长文 attention 延迟受限)

状态: **RUNNING (用户 2026-09-23 放宽 Q15: 继续探索, 不突破则集成最好变体; 门槛改为 e2e >2%)**
动机: implementer 实测 FA 内核 24-32% MFU; 每 KV chunk 实测 ~20K cycles vs issue 极限 ~3-5K
-> 延迟受限; T10/T11/T12/T16 已把调度/波次吃到 -14.7%, 内循环是剩下的大头

## 目标与门槛 (analyst 修正版)

- **Stage 0** (0.5 天, CPU 可先做): 抽独立 harness (真实形状 DKQ=DV=256, ncols=64, l=32768);
  **基线必须用生产配置 (T12+T16, PB=2/grid=384, ~2027us/launch @depth8k)**, 不是 stock (2375us)
- **Stage 1 gate (harness)**: **>= 1.20x** (attention 时间, vs 生产配置基线) 才准进集成; stretch 1.5x;
  1.5 天 checkpoint: < 1.10x -> 停, 写数字归档 (不集成, 不硬凑)
- **Stage 2 (集成后, 同 session A/B >=3 轮)**: **pp8192@depth128k >= +10%**; depth32k >= +4%;
  pp32768 >= +2%; pp512/4096/8192 |Δ| <= 0.3%; tg128 不回退; PPL 4.3572±0.013 + 200 token 生成
- timebox: 3 天 + 1.5 天 checkpoint

## 范围与约束

- 仅限 sm70 的 `flash_attn_ext_f16<256,256,32,2,...>` (D256) 配置族; 其他 arch/配置不受影响
- `fattn-mma-f16.cuh` 是共享文件: 最小 diff + env gate (+ #ifdef); 集成只在 Stage 1 通过后
- 方向顺序建议: ①K/V 寄存器软件流水 (局部, 先试) -> ④ncols=32 探针 (0.5 天, 为 ② 开关) ->
  ②消 67584B combine smem (->2 CTA/SM, 免 spill; 结构性收益最大) -> ③warp tile 扩展 (风险最高, 最后);
  实现者可依实测调整顺序
- 已否证不要重复: stream-K grid 调优 (PB=2 最优), 提占用 (Q_in_reg/nthreads spill), split-D/N32 移植

## 期望管理 (analyst)

- 1Cat 专用 FA-V100 实测 29-38 TF/s; 我们 ub2048 已 39.9 -> V100 该形状实际天花板估 **~40-45 TF/s**
- T18 合理预期: 34.5 -> 40-45 (+16~30% attention; 长文点 e2e +9~18%); >=1.2x 概率估 ~30%, 1.5x <10%
- 注意: T02/T09 内核级改造两次失败的前车之鉴 -> 概率不高估; 但本次有"停顿 5-10x"的定量诊断,
  依据比那两次硬


---

## Stage 0 Result (2026-09-23): DONE

- harness `artifacts/t18_fa_harness.cu`: 直接 include 源头文件; 真实形状; grid=192 对照 + grid=384 (生产) + uniform fixup;
  正确性 max_abs 2.6e-4; 编译需 `--extended-lambda` + ggml_abort stub
- 基线: **12.718 ms/launch @ n_kv=35072** (grid=384); 对照 grid=192 = 15.218 ms (0.836); 生产 l=35072 = 14.84ms (保真 +2.5%)
- 门槛换算: Stage 1 >= 1.20x -> <= 10.60 ms; checkpoint 1.5d < 1.10x (<= 11.56 ms) 即停
- 附带: 每 launch 固定 ~70us + 每 KV token ~0.43us (l >= 12k)


---

## Stage 1 Result (2026-09-23): **1.12x (gate 1.20x 未达)**

- 最好变体: **ncols=32 瓦片 (smem 35072 -> 2 CTA/SM = 8 warps) + PB=2/grid=768** = 11.297 ms (基线 12.660) = **1.12x**;
  l=12288: 1.22x; l=100000: 1.10x (收敛)
- ncu: ncols=64 延迟受限 (occ 6.25%, Est.Speedup 61%) -> ncols=32 后 **L1/shared 通路受限 (74.5%)**, DRAM 8.5%
- 被硬件挡死: Q_in_reg (255 regs + 472B spill, 4x 慢); ncols=16 (内核限定 >=32); nthreads=256 (np=8 不支持); np=1 (需 ncols=128)
- Stage 2 预估: +6.3% / +3.2% / +1.85% (三道门槛都差一点) -> 建议归档; 见 QUESTIONS 的裁决请求 (A/B/C)
- 产物: harness + 5 变体 exe + ncu 日志 2 份 (artifacts/)

## 用户放宽裁决 (2026-09-23, Q15): 继续探索 + 集成保底

- 时间: **探索期 +2 天** (中途 1 天 checkpoint); 到点无论是否突破都进入集成阶段
- 探索方向 (未试的结构性路线): ①K/V 寄存器软件流水 (若确认 L1/shared 受限则优先级降) /
  ②消 67584B combine smem (可再提占用) / ③warp tile 布局重写 (原估 <20%, 现在有预算)
- **新采纳门槛 (替换 1.20x/1.10x)**: 集成后 **e2e 主判据 pp8192@depth128k > +2% 即合并**;
  无回退照旧 (pp512-8192 |Δ|<=0.3%, tg128 噪声内, PPL 门槛 + 200 token 生成)
- 保底: ncols=32 变体 (1.12x, 预估 128k +6.3%) —— 探索无突破则直接集成它
- 若实测 128k 点 < +2% 但 depth32k > +2% -> 报 analyst 复核 (不自动拒)


---

## 最终 Result (2026-09-23): 生产未达门槛 -> 不合并, 已回退

- 生产 A/B: 128k 点 **+0.11%** (门槛 +2%) / depth32k +1.50% / pp32768 +1.17% / 短点 tg128 无回退
- nsys 逐 launch (l~135k, >30ms): A 40.28ms vs B 40.38ms -> **长 l 下 ncols=32 慢 0.3-0.6%** (harness 1.10x 不复现)
- 128k 构成: FA 41.7% / GEMM 35.2% / dequant 9.8% / GDN 4.6% / 其他 8.7%
- 根因: harness 长 l 保真缺口 (只在 l=35k 对过生产); 长 l 瓶颈 = K/V 流式, 非 tile 配置
- 处置: 部署回退 `453E2911`; 源码两处改动待裁 (QUESTIONS); T18 归档


## 用户裁决 (Q16): 撤回归档 (2026-09-23)

- 理由: "1% 就不要了, 还牵扯了注意力改动, 划不来"
- 已撤销 `fattn.cu` (dispatch 覆盖) + `fattn-mma-f16.cuh` (ncols=32 配置行); 部署 DLL 保持 `453E2911`
- 留档 `artifacts/t18-ncols32-REJECTED.patch`; 状态 -> **CLOSED**
