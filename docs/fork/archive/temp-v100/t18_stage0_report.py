import io, shutil

T = r'<TEMP>\v100'
A = r'D:\LLM\Backend\v100-collab\artifacts'
for f in ['t18_fa_harness.cu', 't18_fa_harness.exe']:
    shutil.copy(T + '\\' + f, A + '\\' + f)
print('harness archived')

t = """
---

# T18 Stage 0: FA mma 独立 harness (2026-09-23, implementer): **DONE**

产物: `artifacts/t18_fa_harness.cu` (+exe)。做法: **直接 include `fattn-mma-f16.cuh`** (不是 verbatim 拷贝) ->
Stage 1 修改源文件时 harness 自动跟随, 迭代最快。编译: `build_harness.cmd <exe> <cu> --extended-lambda`
(+ `ggml_abort` stub)。

## 形状与方法

- 真实形状: DKQ=DV=256, ncols1/ncols2=32/2, n_q=512 (ub512), GQA 24/4, head_dim 256, mask [n_kv, n_q] causal
- 两个配置: **grid=192 (1 tile/CTA, 无 fixup, = T11-off 对照)** 与 **grid=384 (PB=2 + uniform fixup = 生产配置)**
- 配置/共享内存/步幅全部从源文件读取 (nthreads=128, nbatch_fa=32, combine=128, Q_in_reg=0, smem=67584 ✓ 与生产一致)
- 正确性: grid384+fixup vs grid192 -> max_abs 2.6e-4 (mean 1.0e-5) ✓

## 时间 (20 iters, 单 launch 均值)

| n_kv | grid=192 | grid=384+fixup | ratio | 生产对照 (grid=192) |
|---:|---:|---:|---:|---|
| 12288 | 6.308 ms | 4.926 ms | 0.781 | 5.29 ms (nsys per-instance, depth8k 末 ubatch) |
| 24576 | 10.704 ms | 8.930 ms | 0.834 | - |
| **35072** | **15.218 ms** | **12.718 ms** | **0.836** | **14.84 ms** (depth32k avg, T11-off) |

- 保真度: l=35072 绝对时间差 +2.5% (时钟/干扰), 相对增益 -16.4% vs 生产 ~-15% ✓ -> **可直接用于 Stage 1 门槛**
- **Stage 1 基线 (生产配置) = 12.718 ms/launch @ l=35072; 门槛 >=1.20x -> 目标 <= 10.60 ms**
- 附带发现: 每 launch 固定开销 ~70us (生产分布拟合), 每 KV token ~0.43us (l>=12k 基本线性)
- 生产 per-launch 分布 (depth8k, 512 launch): l=512 -> 0.29ms ... l=12288 -> 5.29ms (16 层 x 每 l 16 次)
"""
io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

q = """
---

## 2026-09-23 | T18 Stage 0 DONE | implementer

- harness: `artifacts/t18_fa_harness.cu` (直接 include `fattn-mma-f16.cuh`, Stage 1 改动自动生效; 需 `--extended-lambda`)
- 形状: DKQ=DV=256, n_q=512, GQA 24/4, mask causal [n_kv, n_q]; 配置/ smem 从源文件读取 (smem 67584 ✓)
- **Stage 1 基线 (生产配置 grid=384+fixup) = 12.718 ms/launch @ n_kv=35072**; 对照 grid=192 = 15.218 ms (ratio 0.836)
  生产对照: 14.84ms (depth32k, grid=192); 相对增益 -16.4% vs 生产 ~-15% -> 保真
- 门槛: >=1.20x -> 目标 <= **10.60 ms**; 1.5 天 checkpoint <1.10x (<=11.56ms) 即停
- 下一步: Stage 1 方向 (1) K/V 寄存器软件流水线 (Volta 无 cp.async, nstages=0)
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(q)
print('QUESTIONS appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T18-fa-mma-rewrite.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## Stage 0 Result (2026-09-23): DONE

- harness `artifacts/t18_fa_harness.cu`: 直接 include 源头文件; 真实形状; grid=192 对照 + grid=384 (生产) + uniform fixup;
  正确性 max_abs 2.6e-4; 编译需 `--extended-lambda` + ggml_abort stub
- 基线: **12.718 ms/launch @ n_kv=35072** (grid=384); 对照 grid=192 = 15.218 ms (0.836); 生产 l=35072 = 14.84ms (保真 +2.5%)
- 门槛换算: Stage 1 >= 1.20x -> <= 10.60 ms; checkpoint 1.5d < 1.10x (<= 11.56 ms) 即停
- 附带: 每 launch 固定 ~70us + 每 KV token ~0.43us (l >= 12k)
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T18 updated')
