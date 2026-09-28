# T32 子问题 3: RAM+SSD 树状 KV 存储 - 设计 v2 (implementer, 2026-09-27; 用户逐节确认)

状态: 阶段 0-5 已完成 (阶段 5: fork 验收模式 + 全回归 + 短 soak, 分支 t32-stage4/t32-stage5 未合并/未 push); 磨损优化/持久化待后续

## 0. 范围与已定决定
- 需求 (用户 2026-09-27 权威): 整序列独立 blob -> **树状存储**; 介质 = VRAM 工作区 / RAM 热层 /
  **SSD 第三级存储** (非持久化; 持久化/启动挂载 = 后续独立任务); 用**引用次数 + 重叠度**把高引用枝干
  留 RAM、靠叶部分溢写 SSD; 合理淘汰; 最大化 KV 复用减少计算; **A/B 场景一次成型都要通过**
- 引擎路线 R1 (用户选定): 引擎新增"区间序列化"API; 不开 `--kv-unified`、不加 seq、VRAM 零增长
- 已排除: VRAM 内活分支共享 (S2a, 用户 2026-09-27 纠偏); 持久化/启动索引重建 (后续任务)

## 1. 架构与数据模型
### 1.1 三个角色
- VRAM = 推理工作区 (一次一条活序列; 树不做 VRAM 内共享)
- RAM = 热层 (高引用块 + 热锚点)
- SSD = 冷层 (叶块 + 冷锚点; 目录/上限可配)

### 1.2 块链 = 树
- **块 (chunk)**: 粒度 `--tree-chunk` 默认 512 token (阶段 1 实测锁定: 512/1024/2048 的 ms/chunk 近似随字节线性增长, 固定开销不主导, 512 保住复用粒度; 见附录); 身份 = 链式内容哈希
  `h_i = H(h_{i-1}, tokens(chunk_i))`; 相同前缀 -> 相同块 -> 天然 radix 树; 分叉 = 哈希链分叉
- 块元数据: token 内容 / 位置区间 [a,b) / target KV 区间 blob / draft KV 区间 blob (MTP 时) /
  refcount / heat / last_used / tier
- 匹配用哈希定位, 命中后逐 token 校验 (防碰撞), 校验失败按缺失处理
- 键性质: attention KV 是前缀的纯函数 (任意分叉/去重); recurrent 不是纯函数 -> 可恢复点 = 状态锚点 (见 4)
- 分叉/切分/合并/去重全部是元数据操作, 不搬数据; 块内分叉最多重复半块 (≤512 token)

### 1.3 状态锚点 (recurrent 快照)
- 内容 = GDN recurrent 快照 (162 MiB 常数, S1.5 后) + spec 状态 (MTP, ~20KB)
- 键 = 该位置的前缀哈希; 元数据: pos / kind (tip | message | ondemand) / refcount / heat / tier
- **硬约束**: 锚点只能建在确实捕获过状态的位置 (序列末端 / slot 现有检查点 / 重放后), 不能凭空生成

### 1.4 序列簿记
- {tip 块哈希, 长度 L, last_used, pin, 锚点列表}

## 2. 引擎接口 (R1)
### 2.1 新增 API (`llama.h`; 只针对 attention KV, 不碰 recurrent)
```c
size_t llama_state_seq_get_size_range_ext(ctx, seq_id, p0, p1, flags);
size_t llama_state_seq_get_data_range_ext(ctx, dst, size, seq_id, p0, p1, flags);
size_t llama_state_seq_set_data_range_ext(ctx, src, size, seq_id, append, flags);
```
### 2.2 语义
1. 只做 attention; recurrent 继续用现有 `PARTIAL_ONLY` 接口 (末端 park 直接捕获; 中间锚点用 slot 检查点 blob)
2. 写侧: `state_write` 本就逐 cell 写 `pos`, 只加 `pos ∈ [p0,p1)` 过滤 (SWA 掩码照旧); payload 格式不变
3. 读侧: append=true 跳过现有 `state_read_meta` 开头的 `seq_rm(dest,-1,-1)`
   (server-context 现状 :2344); 但 payload 中任何已存在于目标序列的位置直接报错
   (stock `find_slot` 不会复用同位置 cell); 其余位置分配空闲 cell
4. **位置重叠 = 报错返回 0** (防静默覆盖); 池满/分配失败返回 0, 由上层降级
5. draft 上下文同款可用 (ctx_dft 也是 llama_context)
6. `llama_memory_hybrid` 只把 range 调用转发给 attention 组件, 保证语义单一
### 2.3 兼容与验证
- 现有 `get_size_ext/get_data_ext/set_data_ext` 一行不改 (旧 prompt cache / 检查点路径不受影响)
- 引擎自测: a) 纯 attention 模型: 全区间序列化 == 全量序列化 (逐字节); b) 分段 append 恢复 ==
  一次性全量恢复 (生成逐位一致); c) 重叠 append 明确报错; d) hybrid (Qwen3.5-2B) 一遍
  (SWA 无模型, 由现有回归覆盖); e) 现有 save/load 测试全绿

## 3. 存取路径
### 3.1 Park (存)
1. 从位置 0 逐块算链式哈希 -> 找已存最深块边界 B; 缺失段 [B, L) 逐块序列化:
   target KV `get_data_range(ctx_tgt, seq, a, b)`; draft KV 同款对 `ctx_dft`; 新块先落 RAM, refcount=1;
   链上已有块 refcount/heat 更新
2. **末端锚点**: `get_data_ext(PARTIAL_ONLY)` 捕获 L 处状态 (必存, 免费) + spec 状态 (MTP)
3. **中间锚点**: 收编 slot 现有检查点 (消息边界 / 尾部锚点), 按 4 的稀疏化规则过滤
   (>=step, 冲突留靠前) + 按需锚点 (3.2 重放后生成);
   锚点数量 ≈ L/step (100K: 8K->13 个=2.2GB, 32K->4 个=0.7GB); 超预算时按 5.3 先淘汰锚点
4. 序列簿记 (tip 哈希/长度/last_used); slot 清空 (沿用现有流程)
### 3.2 Restore (取) — 锚点为准
- **恢复点公式**: `C = 满足"新请求 tokens[0,C) == 已存前缀 [0,C)" 的最深锚点`
  (不是最长块匹配; C 之后 KV 不装, 反正会被重算覆盖)
- 步骤: 1) 逐块哈希 + 逐 token 校验 (最后一块内可部分匹配), 收集路径锚点;
  2) 选 C; **无可用锚点 -> 放弃树恢复, 全量 prefill** (日志/指标, 不静默);
  3) `seq_rm(slot)` -> 逐块 `set_data_range(append=true)` 装 [0,C) (RAM memcpy / SSD 读文件;
  最后一块可整块装载) -> `seq_rm(slot, C, -1)` 裁掉 C 之后多余 cell ->
  `set_data_ext(PARTIAL_ONLY)` 装状态@C -> 重建 slot 最小检查点表 (来自路径锚点; 末端锚点必含,
  供剥思考后的回退用);
  4) 交回现有流程: 从 C 批量 prefill 剩余 prompt, 之后照旧
- 若本次是"分叉且无锚点": 重放 [C', C) 后**顺手捕获 C 处状态存为新锚点** (自愈)
### 3.3 降级边界 (全部可见)
- slot prompt 经 context shift / 截断 (位置不从 0 起) -> 跳过 park (WRN + 指标), 下次全量 prefill
- SSD 读失败 -> 该块视为缺失: 退到更浅锚点, 否则全量; 绝不静默丢数据
- 池满 / 分配失败 -> 恢复失败 -> 全量 prefill
### 3.4 Server 集成与旋钮
- 接缝: `launch_slot_with_task` 的"存当前/取目标"两处 + idle slot 保存处; `server_prompt_cache` 由树替换,
  开发期 `--kv-tree` 开关与原路径 A/B 对照 (默认关)
- 旋钮: `--tree-ram` (默认 8192 MiB) / `--tree-disk` (目录, 如 E:\llm-tree) + `--tree-disk-limit`
  (默认 65536 MiB) / `--tree-chunk` (默认 512, 阶段 1 实测锁定, 见附录) / `--tree-anchor-step` (默认 32768, 阶段 0/1 定)
  / `--tree-debug`
- VRAM 零变化 (不开 unified、不加 seq); v1 I/O 与 H2D 同步 (切换本来就阻塞); 调度器单线程

## 4. 锚点策略 (用户提案细化)
- **来源**: ① 序列末端 (必存) ② slot 现有检查点 (user 消息边界 / 尾部锚点) ③ 按需 (重放后捕获)
- **稀疏化**: 候选按位置排序, 贪心保留: `pos - 上一保留锚点 >= --tree-anchor-step` (旋钮,
  **默认 32768**, 阶段 0 基准后在 16K/32K 间定; 8K 起可选); **冲突保留靠前者** (它同时覆盖后面,
  重放距离 <= step)
- **分叉点提升 (随引用更新)**: 位置 P 成为分叉点且被引用 (refcount>=2 或实际被恢复过 1 次) ->
  在 P 建锚点; 删除 `(前一锚点, P)` 之间无分叉价值的中间锚点 (refcount<2 且非末端);
  若 P 与前一锚点距离 < step -> 不建 (已覆盖, 符合"只保留前一个")
- **覆盖性**: 有覆盖区间内分叉重放距离 <= step (32K ≈ 44s / 8K ≈ 11s @719 t/s 批量 prefill);
  长回合无消息边界时靠按需锚点自愈 (第一次分叉付重放, 之后免费)
- **锚点成本 (修正: 早期写的 2.5% 是算错的)**: 锚点 170MB / step token; target KV ≈ 52KB/token
  -> 8K 间距 = KV 的 ~40%, 16K ~20%, **32K ~10% (默认)**; 收益 = 每次分叉少重放 <= step token
  (8K≈11s / 16K≈22s / 32K≈44s @719 t/s); 默认值阶段 0 后定
- **可选兜底**: 真网格 (server 解码时按格强制建点) = 同样的 step 成本 (8K=+40% / 32K=+10%); 先不做

## 5. 统计 / 放置 / 淘汰
### 5.1 统计
- 块: refcount (引用序列数) / heat (加载次数) / last_used / tier
- 锚点: refcount (覆盖的序列数) / heat / kind
### 5.2 RAM/SSD 放置
- RAM 优先: refcount>=2 的枝干块 + heat 高 + 靠近锚点
- SSD 优先: refcount==1 且无后继的叶块
- 新块先落 RAM; RAM 紧张按评分降级到 SSD; SSD 命中提升回 RAM (**单份权威, 移动不复制**)
### 5.3 淘汰次序 (空间不足)
1. 状态锚点 (可重放重建, 最便宜) - 先删最冷、覆盖价值最低的
2. KV 叶块 (refcount<=1 且无后继) - 按 recency/heat
3. 整条叶序列: 从 tip 沿链向上递减 refcount, 降到 0 的块删除
4. 仍不够 -> **拒绝本次保存/恢复** (显式 WRN + 指标, 绝不静默)
- 共享枝干 (refcount>=2) 最后动
### 5.4 pin 纪律
- 选定恢复目标后立即 pin 其路径上的块/锚点; 淘汰全程跳过 pinned;
  全 pinned 且空间不够 -> 放弃当前工作序列 (延迟成本), 绝不牺牲目标 (立即成本)

## 6. 错误处理与可观测
### 6.1 失败模式
| 场景 | 动作 | 可见性 |
| 链不匹配 / 无锚点 | 全量 prefill | INF + 指标 |
| SSD 读失败 | 块缺失 -> 更浅锚点或全量 | WRN + 指标 |
| SSD 写失败 | 保留 RAM 或丢弃新块 (该序列不存) | WRN + 指标 |
| 池满 / 分配失败 | 恢复失败 -> 全量 | WRN + 指标 |
| 位置不从 0 / 截断 | 跳过 park | WRN + 指标 |
| payload 校验失败 | 丢弃该块 | WRN + 指标 |
### 6.2 指标
命中次数 / 复用 token 数 / 搬运字节与耗时 / 重放 token 数 / 锚点数 / 淘汰数 / RAM+SSD 占用 / 降级次数
### 6.3 日志
复用与降级原因必须 WRN/INF 可见; 不静默

## 7. 验收判据
### 7.1 A 场景 (两长会话共享枝干)
- 构造: 2 x ~100K, 共享 ~90K; 交替 6 轮
- 判据: (a) 每次切回恢复 = 装 [0,C) + prefill [C,末尾), C = 该会话 tip;
  (b) 输出与无树/全量 prefill 基线**逐位一致** (greedy, **replay=0**; replay=1 有 T24 已知近并列
  运行间不确定性, 不作判据);
  (c) H2D 时间记录 (100K KV ~5GB; 阶段 0 实测值为准);
  (d) RAM <= `--tree-ram` (默认 8192 MiB), SSD <= `--tree-disk-limit` (默认 65536 MiB)
### 7.2 B 场景 (多短会话)
- 构造: N=8, 各 5-10K, 共享前缀 ~4K; 轮换
- 判据: 前缀块 refcount >= N 且只存一份; 总占用 ≈ 前缀 + Σ独有; 冷叶落 SSD; 输出逐位一致 (replay=0)
### 7.3 引擎 / 回归 / 反例
- 2.3 的 a-e; 现有 save/load 测试全绿; 生产构建不受影响
- 反例: 无锚点 / 位置不从 0 / SSD 读失败 / 重叠 append -> 全部正确降级
- 实测项: agent 长会话在 135K ctx 下是否触发 context shift/截断 (触发则该类会话无复用, 记录占比)

## 8. 非目标 (本次不做)
- 持久化 / 启动挂载; VRAM 内共享; 真网格 (可选); 异步 I/O (v1 同步); 会话 ID (仍按内容匹配)
- iswa / hybrid-iswa 模型 (SWA 分层) 不支持 range API: 调用返回 0 (与 hybrid-idx 同策略, 显式失败)

## 9. 分期 / 风险 / 回退
- **阶段 0 (无 server 改动; 现有 API 测 H2D 带宽, 定 A 场景预期与 `--tree-anchor-step` 16K vs 32K)**:
  真实模型 32K 序列 ~1.9GB D2H/H2D 往返, 线性外推 100K (~5GB -> 秒数)
- 阶段 1: 引擎 API + 单测 (无 server 改动); 出口加区间 API 吞吐基准 (chunk 512/1024/2048,
  32K 区间足够看固定开销) -> 定 `--tree-chunk` 默认值 + 是否需要批量装载路径
- 阶段 2: 树模块 (块链/锚点/放置/淘汰) + harness (短序列正确性)
- 阶段 3: server 集成 (`--kv-tree` 开关) + A/B 验收
- 阶段 4: 淘汰/分层打磨 + 指标
- 风险: 读侧 append 语义需实测 (重叠/复用, 阶段 1 已验证); range payload 与全量 state 共用同一 blob magic,
  装载方无法区分 -> 阶段 2/3 前建议给 range payload 独立 magic; 切换阻塞时长 (v1 同步 I/O);
  位置不从 0 的会话无复用; 锚点提升抖动 (加冷却: 同位置提升限一次)
- 回退: `--kv-tree` 关闭即回原 prompt cache; 生产保持 T24 部署
- **已知缺口 (D2, 记录)**: 树锚点载荷不含 MTP/spec 状态 (`kv_tree_anchor_in` 无 spec 字段, 模块无 spec io, 树恢复后不还原 spec 状态). 2B/3B 验收不涉及; 27B/MTP 部署阶段补 (或恢复后显式重置 spec 状态).
- **D12 (记录, 阶段 3 裁决)**: §3.2 步骤 3 的"按路径锚点重建 slot 最小检查点表"阶段 3 未实现 (`prompt.checkpoints` 清空, `res.anchors` 保留为预留接口); 缺失只影响 reasoning 回退/SWA 裁剪的快速路径, 正确性由全量重算回退保证 -> 阶段 4 实现. **阶段 4 已实现**: 按树锚点载荷重建 `prompt.checkpoints`; recurrent 上下文无法 partial seq_rm 时改用 tail 语义 `pos_min = pos_max = pos - 1`, PART 上下文仍 `pos_min = 0` (D18, 见 artifacts/t32-tree-plan-stage4.md); 重建拷贝另有 256 MiB 字节上限, 超限从最浅丢弃 (D24).
- **D25 (记录, 阶段 5)**: 双档间距: park 采纳的 MESSAGE 候选用 `--tree-checkpoint-anchor-step` (默认 32768); 分叉捕获的 ONDEMAND 锚点用 `--tree-checkpoint-fork-step` (默认 8192); 两者独立、非负, 0 = 无最小间距. 理由: 分叉点重放代价低 (8K 重放 ~11s), 可密; 猜测型锚点稀疏以省存储.
- **D26 (记录, 阶段 5)**: heal-on-miss: `restore` 在 `C < 0 且 deep > 0` 时返回 `heal = deep`; server 在 miss 时把 `tree_heal` 保留过 `prompt_clear` (仅 cache_prompt=true), 重放至 heal 位置时捕获锚点. 中断安全: 仅 `n_tokens == heal` 精确相等时捕获, 引擎复验 `pos_max == pos - 1`, 选槽时 `tree_heal` 无条件清零.
- **D27 (记录, 阶段 5)**: 删除 `promote_prune` 死代码: 存锚时 `prev` 按构造取同链 pos 之下最深的锚点 (与 `fork_step` 无关, 0 时也成立), `(prev, pos)` 之间不可能有锚点, 剪枝窗口恒为空; 猜测锚点的取舍由淘汰评分承担 (heat==0 的猜测天然先淘汰).
- 记录 (阶段 5 评审): 树流程运行时 context shift 实际关闭 (验收一律 `--no-context-shift`; `pos_max` 守卫不能证明缓存窗口从 0 开始).
- 记录 (阶段 5 评审): `m.deep == tokens.size()` 的 heal (全部 prompt token 有块匹配但无锚点覆盖) 不会捕获 (仅性能; 同阶段 3 heal-on-hit 备注).

## 附: 关键数字 (设计依据)
- 块: 512 token -> target KV 24.5 MiB 即 25.7 MB (阶段 1 实测 q8_0 V, 50200 B/token, range API 仅 attention,
  不含 recurrent 常数), draft KV ~2MB (target ~50.2KB/token / draft 4.1KB/token)
- 锚点: 162 MiB = 170MB/个 (MTP on, S1.5 后); step=8K -> 21.2MB/1K token = KV 的 ~42%
  (16K ~21% / 32K ~10%); 100K 序列: 8K 间距 13 个 = 2.2GB, 32K 间距 4 个 = 0.7GB
- 速度: prompt eval ~719 t/s (实测 32354 tok/45.0s); 8K 重放 ≈ 11s; 128K 全量 re-prefill ~260s (实测)
- H2D: 2.37 GiB/s / D2H 2.60 GiB/s (阶段 0 实测: 27B Q6_K 全量 state f16, 32K 2.10 GiB, PCIe x4)
- 100K 树存储搬运 (阶段 1 实测口径, q8_0 target KV = 50200 B/token, 4.68 GiB):
  按全量 state 带宽 (2.37/2.60 GiB/s) 外推 H2D ~2.0s / D2H ~1.8s, 仅作全量口径参考;
  按 range API chunk 512 实测吞吐 (H2D 1896 MiB/s / D2H 1198 MiB/s) 则 restore ~2.5s / save ~4.0s
- range API 区间吞吐 (阶段 1 实测, 27B q8_0, 32K, 完整输出 artifacts/t32-stage1-bench.txt):
  chunk 512/1024/2048 -> 读 12.8/23.2/41.5 ms/chunk (1896/2063/2309 MiB/s),
  写 20.3/34.8/60.1 ms/chunk (1198/1378/1594 MiB/s); ms/chunk 近似随 chunk 翻倍 -> 固定开销不主导,
  `--tree-chunk` 维持默认 512 (复用粒度最好)
- **D31 (阶段 5b)**: capture_anchor 间距只计 TIP/ONDEMAND 锚点, MESSAGE 猜测不压制动证据; D32: 主动网格填充未做, 待 step_skips 数据.
- **近并列说明**: tree 路径与全量 prefill 路径的浮点累加顺序不同, logits 有 ~0.01-0.05 nats 级确定性差异; 贪心采样在近并列处会翻转 (T24 replay 现象). soak 用 logit 差 < 0.05 判 near-tie (记录), 更大才算 mismatch.
