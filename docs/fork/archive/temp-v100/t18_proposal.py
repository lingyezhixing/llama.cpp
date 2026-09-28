import io

t = """
---

## 2026-09-23 | 新提案: T18 (Volta FA mma 内核效率重写) - 长文 attention 是最大痛点 | implementer

用户明确表态: **长文降速是当前最头痛的问题, "能解决最好"** -> 提请立项评估。

### 动机 (全部本会话实测)

| 事实 | 数据 |
|---|---|
| 长文 attention 占比 | depth32k 32.0% (1899.4/5934.3ms) / pp32768 18.5% / **pp8192@depth128k 推算 ~65%** |
| FA 内核 MFU | 29.5 TF/s (ub512 depth32k) / 39.9 (ub2048) / **125 TF peak = 24-32%** |
| 命中 kernel | `flash_attn_ext_f16<256,256,32,2,0,0,0>` block=(32,4) 128 线程, dynSM **67584 B**, regs 254 -> **1 CTA/SM (4 warps)** |
| 停顿证据 | 每 KV chunk (32 KV x 64 q): mma issue 极限 ~1.0K cycles, LDS issue ~1-4K cycles, **实测 ~20K cycles -> 5-10x 停顿** |
| 波次已捡完 | T12+T16 (PB=2) 已把 grid/波次收益吃到 -14.7% (vs T11-off); PB=4 过切, grid96 +32% -> 负载均衡路线到头 |

结论: attention 的"慢"不是调度问题 (T10/T11/T12/T16 已解决到 -14.7%), 而是**内核本身延迟受限**;
要拿 1.5-2x (长文端到端 +25-30%) 必须改内循环。这是 llama.cpp 上游共享文件 (fattn-mma-f16.cuh), 属核心级改动。

### 已否证 (不要重复提)

- stream-K grid 数调优 (PB=2 已最优, -14.7%), PB=4/grid96/grid160/grid192 全测过
- 提占用: Q_in_reg 必 spill (cfgA +75%), nthreads=256/512 变体 (cfgE +637%, cfgD 编译失败);
  根因 = VKQ combine 缓冲 67584 B 占满 smem + 该配置行同时服务 ncols=64 的全部切分实例
- 不移植 1Cat 的 split-D/N32 (T10: ub2048 同内核已 39.9 TF/s >= 其 29-38)

### 候选方向 (按性价比排序, 建议 harness-first)

1. **K/V tile 软件流水线**: Volta 无 cp.async -> `nstages=0` -> K/V 装载同步, 每 ~1000 个 KV chunk
   暴露一次全局延迟。用寄存器做 double-buffer (预取下一 chunk 到 reg, 再做本轮 mma, 然后落 smem)。
   局部改动 (fattn-mma-f16.cuh 的 load_tile/iter), 是第 1 个该试的
2. **warp tile 扩展 / 操作数复用**: Volta 只有 m8n8k4, 操作数复用靠显式展开; 每 warp 覆盖更大的
   m/n 可摊薄 LDS 操作数流量 (当前每 mma 2 次 LDS.32, LDS 是首要瓶颈假设)。
   动 T_C_VKQ/T_B_KQ 布局, 风险中等 (tile 断言多)
3. **消掉 67584 B combine 缓冲**: 它是 smem 唯一大头 (输出 fp32->half2 经 smem 转置再写 global);
   改成直接按列片段写 global (上游注释说 smem 中转更快, 需实测) -> 若成立可 2 CTA/SM (免 Q_in_reg)
4. **纯配置探针**: ncols=32 的 Volta 配置行 + dispatch env (Q staging 减半, 寄存器压力减半;
   代价 = K 复用减半) - 0.5 天, 可以顺带做, 用来关闭/开启方向 3

### 建议的门槛结构 (照 T02/T03 先例)

- Stage 0 (0.5 天): 把该 kernel 抽成独立 harness (真实形状 DKQ=DV=256, ncols=64, l=32768),
  复刻 stock 时间 (2295us/launch @depth8k 量级) 作为基线
- Stage 1 gate: harness 内 **>= 1.25x** (attention 时间) 才准进集成; 不到 -> 写数字归档, 不硬凑
- Stage 2: 集成后验收 = depth32k / pp32768 / **pp8192@depth128k >= +15%** (相对 T12+T16 后基线) /
  短点 |delta|<=0.3% / tg128 / PPL 4.3572+/-0.013 + 200 token 生成
- timebox: 3 天, 中途 1.5 天 checkpoint

### 请求

请 analyst 裁决: (a) 是否立项 T18 及其编号/范围; (b) 门槛与 timebox 是否按上面;
(c) 若批, 我明天先做 T17 测量 + T12/T16 验收, 之后开 T18 Stage 0。

(用户侧: 长文优先; 当前生产 decode 已由 MTP 承担 42.7 t/s, 不受影响。)
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(t)
print('QUESTIONS appended (T18 proposal)')
