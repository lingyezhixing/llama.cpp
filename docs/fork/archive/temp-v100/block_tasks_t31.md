
## 执行前整理 + 数据复用审计 (implementer, 2026-09-26 深夜, 不跑 GPU)

用户指示: 先整理任务 + 复用既有数据, 能直接得结论最好。

### 1. 既有数据可推出的结论 (全部来自已测数据, 无新运行)

**(a) 一轮时间预算** (T30 的 tps + 日志 mean len; ms/轮 = ms/token x tok/轮)

| 深度 | ms/token | tok/轮 | 轮时间 | 深度相关 | 深度无关 |
|---|---|---|---|---|---|
| d0 (l~5) | 20.3 | 3.39 | 69ms | ~1ms | ~68ms |
| d32768 | 35.6 | 2.33 | 83ms | ~12ms | ~71ms |
| d131072 | 49.8-54.9 | 2.16 | 108-119ms | ~45ms | ~69ms |

- 深度相关 = verify 的 FA TILE (25ms@128K, T30 nsys 实测) + V 物化 (8ms) + draft 侧 FA (~5ms) + ~7ms 杂项
- 深度无关 ~69ms/轮 = verify 权重 (MMVQ ~30ms, 22GB/850GB/s 下限附近) + draft 权重/head (~5ms) + **~30-35ms 未知**

**(b) H1 (draft 步) 上界 ~5-10ms/轮** (由 T30 实测 1.6ms/层@128K + MTP 层/head 已知权重规模推出)
- => **H1 < 15ms 判据 => B1 (draft 窗口化) 上限只有 ~+4-5%, 不能单独达成 +20% => 降级为次选**

**(c) verify 侧剩余 ~25-33ms/轮** (FA 距带宽下限 3x + V 物化) **但路线已被 T30 否**
- 换 VEC 慢 3.1x / 提高切分慢 15%; 只剩"新 TILE 内核直读 q8_0 V" (~+10%, 工作量高), 单独列项

**(d) 主项 = ~30-35ms/轮的未解释开销 (H2/H3), 现有数据无法分解**
- 迁移证据: llama-cli 的 nsys 在 decode 空档里 GPU 完全空闲 (5.6s 内 0 个 kernel),
  但 profiler 自身串行化放大 host 侧, 不能用来量化 server
- **关键发现: `speculative.cpp` 已内置 t_begin/t_draft/t_accept 计时 (`gen_perf = true` 恒开),
  每请求在 `slot::print_timings()` 打 `SPC_TRC` (默认 verbosity 3 不显示)**
- => **Phase A 零代码方案: server 加 `-lv 4`** (LOG_LEVEL_TRACE=4 <= thold), 每请求即得:
  `statistics draft-mtp: #calls(b,g,a) = ..., #acc tokens = ..., #mean acc len = ..., #acc rate/pos = (...), dur(b,g,a) = begin, draft, accept ms`

**(e) 缺失的关键基线: 无 spec @128K tps** (T19 只标了 d131072 "存疑")
- 由 (a) 估 ~67ms/token ≈ **15 t/s** -> 即 MTP3 @128K (19-20) 相对无 spec 只有 +20-30% (d0 为 +88%)
- 长文衰减 = 固定开销 + 接受率 (0.388 vs 0.808) 两项

### 2. Phase A 已就绪 (零代码, 等 GPU 时段, 预计 ~15 min)

- `%TEMP%\v100\t31_phaseA.ps1 -NMax 3|1|0` (BinDir 默认 `llama.cpp-t24` = 生产同构建):
  每个配置: d0 150tok + d131072 150tok x2 (第二次 cache 命中, 差分干净); server 加 `-lv 4`,
  `GGML_CUDA_GDN_REPLAY=1`, 其余生产口径 (ub512, greedy seed42)
- `%TEMP%\v100\t31_spc.py <log>`: 解析 statistics / eval time / draft acceptance, 按请求做差分, 输出:
  轮数, tok/轮, draft ms/轮, accept ms/轮, begin ms/轮, 残差 (verify+采样+其他) ms/轮, 逐位置接受率
- 判据 (规格): 哪项 >15ms/轮 即主攻方向; 三配置同时给 K 结构探针 + 无 spec 基线

### 3. 整理后的执行顺序

1. Phase A 三连 (~15 min): H1/H2/H3 分解 + K 结构 + 无 spec 基线
2. 若 H2/H3 为主 (预期): 走 B2 (采样/接受设备端化 = 并入 T29), B1 降级
3. K 策略由结构探针定: 固定开销若在"每轮" => 不节流 K (少投多轮反而多付固定开销); 若在"每步" => 节流可能有益
4. verify 侧新内核 (TILE 直读 q8_0 V) 放最后, 单独列项 (~+10%)

### 4. 待用户 / 风险

- GPU 时段 (今晚未跑, 遵守夜间静音)
- B2 若改采样路径: 按规格硬性要求 greedy 逐 bit + 生产采样统计一致 (风险/工作量中等)
