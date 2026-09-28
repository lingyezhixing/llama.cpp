# T31: MTP 轮内开销 (128K) - 用户批准集中攻克 2026-09-26

状态: **PARKED (用户 2026-09-27: ROI 不足 -> 转 T32; B2 延后; 见文末裁决)**; 曾 APPROVED (用户 2026-09-26: "接下来就剩 MTP 轮内开销了" -> 集中攻克);
执行者: implementer; 基线 = T24 部署版 (`299DAFC7` + `GGML_CUDA_GDN_REPLAY=1` = 生产配置); 需 GPU 时段
并入: **T29** (设备端采样/接受; 条件已满足 -> 本任务 Phase C)
不列入: 投机组合 (ngram+MTP 为 first-wins 回退链, 用户 2026-09-26 明确否)

## 前置事实 (T30 + 本轮代码分析)
- MTP3 @d131072 = **19.5-20.1 t/s**、接受率 0.388、平均 2.16 token/轮 -> **一轮 ~111ms**
- verify 侧已测 (nsys, 仅 kernel 窗口): MMVQ 49% / FA TILE 30% / V 物化 10% / 其它 10% -> 约 56-60ms/轮
- **未解释开销 ~50ms/轮** = draft 步 + 采样/接受 + 同步 + server/CPU (T30 只在 llama-cli 测过, server 侧未测)
- 已知机制 (代码): draft = ctx_dft 上 3 次自回归 decode (每步 1-2 层 attention over 128K +
  device top_k(10) 采样, `common/speculative.cpp:1400-1411`); 目标 logits 每轮 D2H (4 token x 248K vocab ~4MB)
  + CPU 采样接受; 每次 llama_decode 一次 synchronize; MTP prefill 钩子 (每 ubatch) 已属摊薄项

## Phase A: 分解 (0.5h, 先做)
- server 侧 (非 llama-cli) nsys profile 一段 128K MTP3 生成 (150 token, greedy seed42)
- 计时打点 (host): t_draft / t_verify / t_sample_accept / t_sync / t_loop (SPC 统计 + 外部计时)
- 产出: "128K 一轮 111ms 去向表" (ms/轮 + 占比); 明确 H1 (draft 步) / H2 (采样+同步+CPU) / H3 (其它)
- 判据: 哪项 >15ms 即为主攻方向

## Phase B: 按 A 结果二选一 (0.5-1 天)
**B1 (若 H1 为主): draft 窗口化** -- ctx_dft 的注意力/KV 限制到窗口 (4K/8K, env 门控)
- 依据: MTP 输入 h_tgt 已含全上下文信息, MTP 层自身长程注意力冗余; 窗口只保留"近处抄模式"功能
- 预期: draft 每步 attention ~1.5ms -> ~0.1ms; draft KV 大幅缩小 (显存 -)
- 风险: 改变训练分布 -> acceptance 可能小降, 必须 A/B
- 同相位备选: **长文 K 节流** (K=1/2, 零代码先试) -- 低接受率场景少投几步可能更划算

**B2 (若 H2 为主): 采样/接受优化** -- 目标侧采样设备端化 / 减少同步 / 4 位置批量处理
- 硬性: 采样路径改动不得改变输出分布 (greedy 下逐 token 一致 + 生产采样统计一致)

## Phase C: (若 B 不够) 并入 T29
- 设备端采样/接受全链路; 或 draft 追赶/verify 异步流水化 (改动 common/speculative.cpp 时序, 风险中)

## 验收
- 主: **MTP3 @d131072 t/s** (同 session A/B 3 轮交替; 目标 +20% 起)
- 次: @d32768 / @d0 防回归; 接受率同报
- 质量: PPL 控制位 (无 spec -> 应逐位不变) + 128K 生成 200 token 无乱码/重复
- 显存: 中性或下降 (窗口化应降)
- prefill 防回归: pp512 + pp8192@depth128k
- 口径: Q17 (不要求与老 MTP 逐位; 输出仍由 target 采样决定, 不降智)

## 交付物
- RESULTS "T31": A 分解表 + B/C 的 A/B 表 + 结论 (采纳/默认开关/patch)
- 未达门槛 -> BLOCKED/REJECTED + 数据, 不硬凑

## 约束
- 不改 decode/verify 数值路径 (B2 采样改动需保证 greedy 逐位); 不动 GDN/A4
- 不放松精度红线; 不重开已否决项 (T11/双 stream/融合/T21 K 扫描等)


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

## T31 Phase A 结果 (implementer, 2026-09-26): 分解完成, B1/B2 均不达 +20%

零代码方案 (`-lv 4` + `t31_spc.py`) 一次跑完 5 个配置。128K, MTP3, 生产采样, 热请求:
**轮 = 114.9ms / 3.12 token (38.4ms/token, 27.0-27.6 tps)** 构成:
verify+host ~90ms (78%) / draft 17.9ms (16%) / 目标侧采样 ~6.8ms (6%) / accept+begin ~0.3ms (0%)。

- 前置事实修正: 无 spec 单 token decode @128K = 77.4ms (热) -> verify 实际 ~90ms, "未解释 ~50ms" = verify 本身
  (T30 的 llama-cli 窗口估算 56-60ms 偏低); MTP3 相对无 spec 是 -50% (机制本身高效)
- H1 (draft 17.9ms) 可动部分只有 FA 的 ~5ms -> **B1 (窗口化) 上限 ~+4% -> 建议否决**
- H2 (accept) 实测 ~0.3ms; 目标侧采样 6.8ms -> **B2 (设备端采样, =T29) 上限 ~+5%**
- K: MTP3 最优 (MTP1 greedy 16.3-19.0 / MTP6 prod 20.2) -> 保持 K=3, 不节流; pos4-6 接受率已塌 (0.23/0.19/0.12)
- **口径修正: 生产采样下 MTP3 @128K = 27.0-27.6 tps** (greedy 19-20 只是严格接受的特例; acc 0.694 vs 0.388)
- 建议: 唯一能到 +20% 的是 verify 的 FA + V 物化内核 (33ms/轮, TILE 直读 q8_0 V + 快速 FA) -> 新内核任务立项;
  B2 作为唯一低风险 +5% 项可选。详见 `RESULTS.md` "T31 Phase A"。

## Phase A analyst 复核 (2026-09-27): 通过; 决策建议
- 数字独立重算全部对上 (主表, 见 RESULTS "T31 Phase A - analyst 复核"); 修正: 第 2 节 MTP3 行 pos 值为累计,
  128K 单请求 = greedy (0.638/0.362/0.159) / prod (0.917/0.729/0.438); "38.4ms/token" 应为 36.0-37.0 (2.10-2.15x)
- 冷热异常记档 (无 spec/MTP1 冷值偏高 17-30%, 原因未明; 跨配置以热值为基准成立)
- 结论: **B1 否决** (可动仅 ~2-5ms, 上限 +2~4%); **B2 可选** (~+5%, 设备端采样, 并入 T29 或独立小项);
  **内核路径 (FA/V) 是唯一可能 +20% 的项, 但无已证路径** (T30/T18 前车) -> 若要立, 先过 l=128K 微基准门
- 口径重定基: 生产采样 (temp0.6) @128K = 27.0-27.6 tps (非 greedy 19-20); 目标改为"用户实际采样配置固定 +20%"
- 待用户裁决: (a) 立 verify 内核任务 (带微基准门); (b) 做 B2 (+5%); (c) 暂停 T31, 转 T32 优先

## T31 裁决 (用户 2026-09-27): PARKED -> 转 T32 主攻
- 量化: 低风险全包 (B2 + V 直读) ~+12% = 31 tps (+3.4 tps); FA 快速化无已证路径 (T30/T18 前车) -> ROI 不足
- 对照 T32: 一次 128K re-prefill = 257s (本轮日志实测 128270 tok @ ~500 t/s); 会话切换 260s -> 2-6s
- **B2 (设备端采样, +1.4 tps) 留作廉价待办**; 内核任务不立 (除非先过 l=128K 微基准门)
- 归档保留: Phase A 分解 + analyst 复核 + 口径修正 (生产采样 27.6 tps) 供 T04 / 后续引用