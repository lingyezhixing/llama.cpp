# T20: MTP 轨迹一致性独立排查 (根因定位) - 用户指定, T19 后最高优先

状态: **ISSUED (用户指令 2026-09-23)**; 执行者: implementer
用户要求: **不得盲信 2026-09-22 的旧排查报告**; 旧报告仅作"信息参考源"(可能有错误结论);
必须从现象复现开始独立排查, 自建证据链, 每条旧结论都要能独立确认或推翻。
完成后**暂停等判断**, 不做后续任务。

## 0. 背景与参考资产 (仅作线索, 不作结论)
- 旧报告: `D:\LLM\Backend\MTP-轨迹一致性分析与修复报告.md` (2026-09-22)
- 修复 patch 备份: `D:\LLM\Backend\mtp-backup\mtp-work.patch`
  (UTF-16 编码, apply 前需转 UTF-8/LF; 内容 = `fattn.cu` 的 `GGML_CUDA_FA_SMALL_BATCH_VEC` +
  `GGML_CUDA_FA_DEBUG` + `examples/CMakeLists.txt` 接线, 共 3 处)
- 差分诊断工具源码: `D:\LLM\Backend\mtp-backup\batch-invariance\` (batch-invariance.cpp 9.5KB + CMakeLists)
- **当前部署构建不含旧修复** (T19 清理/rebase 后丢失; HEAD `afbab1748`, DLL `7F1B9B240343`):
  基线 = 未修复状态, 应能复现旧报告"修复前"的分叉
- 旧报告主张 (待独立验证, **不作为前提**):
  H1 机制: 提交进上下文的 token 永远是 target 自己采样 -> 无"降智"机制
  H2 分叉主因: Volta FA kernel 选择 (decode n_q=1 走 VEC; verify n_q=2..16 走 TILE + V 物化 f16)
  H3 残留: n-max 3 分叉 = "rs=3 循环态快照恢复不精确" (旧报告自认为推断, 未定位)

## 1. 目标
1. 独立复现/证伪: 当前构建下 `--spec-type draft-mtp` 各 n-max 与无 spec 的轨迹一致性行为
2. 独立定位: 找出**所有**导致分叉的算子/状态路径 (不预设 FA 或快照; 旧报告可能漏项或结论有误)
3. 给出结论 + 修复 (若为可修 bug): **n-max 1/2/3 greedy 与无 spec 逐 token 一致** (短上下文 + 32k + 128k),
   速度不回退 (现状 n-max3 ~2x), 显存中性
4. 明确回答"生产 MTP3 是否存在质量风险"——用证据, 不是引用旧报告

## 2. 阶段
### Phase 0 恢复工具 (~0.5h)
- 转码 apply `mtp-work.patch` (注意: T12/T16 在 `fattn-common.cuh` (stream-K/PB=2), env 在 `fattn.cu`,
  两者交互需验证: small-batch VEC 选择 vs launch_fattn 的 blocks_num.x 覆盖)
- 恢复 `examples/batch-invariance/`; 构建开 `LLAMA_BUILD_EXAMPLES=ON`
- env: `GGML_CUDA_FA_SMALL_BATCH_VEC=1` (强制小批量 VEC), `GGML_CUDA_FA_DEBUG=1` (打印 n_q/kernel)

### Phase 1 独立复现 + 全算子差分 (0.5-1 天)
- server greedy A/B (temp0/top-k1/seed42, >=900 token): none vs n-max 1/2/3 (先不设 env)
  - 记录: 分叉位置 / 分叉 token 的 top1-top2 概率 / 是否等价措辞; 独立启动两次验证确定性
- 差分工具: 同前缀 A=1 (decode) vs B=2/4 (verify batch) 逐层 hidden + h_nextn + logits 对比,
  **逐算子列出所有 batch-variant 项** (FA/GDN/conv/norm/FFN/sampler/KV), 不信任旧报告的"唯一不一致=FA"
- 独立验证 H1 (机制, 代码 + 行为双重): 输出 token 是否可能来自 draft

### Phase 2 根因定位 (1-2 天)
- 用 state hash / 状态对比定位**第一个不一致点** (含调用链)
- 候选假设 (含旧报告之外的):
  a. FA VEC/TILE 数值差 (H2)
  b. n_rs_seq>2 循环态快照恢复 (H3): 行索引 off-by-one? conv 与 GDN 快照不对称? 快照拷贝融合?
  c. KV cache 回滚 (verify 批次 KV vs 顺序解码 KV)
  d. sampler chain 状态 (accept 调用顺序 / grammar)
  e. 本 fork A2 (GDN vec4) 与 verify batch (n_tokens=4 -> vec4 路径) 的交互
  f. ubatch split (`llama-memory-recurrent.cpp:445`) 行分配
  g. 其他 (差分工具发现的算子)
- 每个假设给出"支持/排除"的实测证据

### Phase 3 修复 + 验收 (0.5-1 天)
- 修复 (最小 diff + env 开关), 或若证实为纯数值: 给误差上界 + 分叉概率量化 + 生产安全建议
- 验收: greedy A/B n-max 1/2/3 vs 无 spec **逐 token 一致** (短 + 32k + 128k, >=2 类 prompt)
- batch-invariance 工具全等; PPL 4.3572±0.013; pp512/短点不回退; tg128 不回退; 显存中性 (记录)
- 决定: `GGML_CUDA_FA_SMALL_BATCH_VEC` 是否转默认 (需与 T12/T16 共存验证)

## 3. 停损与时间盒
- 时间盒 **3 天**, 1.5 天 checkpoint: 若根因未定位, 报告"第一个不一致点 + 已知/未知清单"
- 若根因在快照系统深处: 至少给机制文档 + 安全默认建议 (VEC 强制 + n-max 2), 不硬改
- 用户要求: 完成后**暂停等判断**

## 4. 交付物
- RESULTS 章节 "T20 MTP 轨迹一致性": 全部命令 + 原始数据 + 分叉点分析 + 根因证据链
- 修复 patch (若有) + 重建部署 (核对 SHA)
- 结论: 生产 MTP3 质量风险判定 (有/无/边界)
- 更新 BOARD / STATUS / ENVIRONMENT

---

## Result (2026-09-23, implementer) - **DONE, 暂停等用户裁决**

独立排查完成 (未盲信旧报告; 旧报告 5 条主张中 4 条被推翻, 见 RESULTS "T20" 第 7 节)。

### 分叉源 (3 个, 全部有独立证据链)
1. **S1 FA VEC/TILE** (旧报告 H2, 确认): Volta decode n_q=1 走 VEC, verify n_q=2..4 走 TILE -> 首 FA 层即分叉
2. **S2 GDN vec4 布局分支** (**本 fork T03/A2 引入, 旧报告漏项**): n_tokens>1 时 lane 行映射变 (4*lane+r vs r*32+lane)
   -> warp 归约顺序不同 -> ulp 级数据相关差异; 与 FA 无关, 旧报告"GDN batch-invariant"不成立
3. **S3 FA VEC split-K 的 padding 边界** (**新发现, 上游设计属性**): padded n_kv 跨 256 时 ntiles_KV/parallel_blocks
   变化 -> KV 交错切分顺序变 -> 1e-6 级差; decode 与 verify 的 padding 天然不同 -> 每次跨 256 边界一次扰动

### 修复与验收 (S1+S2+S3 全开)
- 短上下文 900 token / 32k 150 token / 128k 100 token: **n-max 1/2/3 与无 spec 逐 token 完全一致**
- 回滚/快照路径独立验证精确 (工具: batch+seq_rm+probe, rollback 1/2/3, rs=0..3, 全行逐位一致)
- PPL = 4.3562 (与交付一致); 确定性复测通过

### 代价 (llama-bench 同 session 交错)
- S1 ~0; S2 tg128 d0 -0.6% / d32768 -1.3%; **S3 -31% @d32768 (不能默认开)**
- 仅 S1+S2 时残余分叉 = 极端近并列 (<=0.02 nats) 翻转, 900 token 内 1-3 次, 两条续写均合法

### 交付物
- RESULTS "T20" (完整证据链/命令/原始数据); 工具增强 (batch-invariance --seq-a/--probe/--rollback, 未提交);
  实验 patch `artifacts/t20-work.patch`; 调试 DLL SHA `DE3F5E50...` (部署已回滚 T19)
- **待用户裁决**: 是否采纳修复 (S1+S2 固化 + S3 可选 env), 见 QUESTIONS
