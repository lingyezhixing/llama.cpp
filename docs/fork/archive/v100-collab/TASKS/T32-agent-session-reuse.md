# T32: agent/长会话复用专题 (用户 2026-09-26 提出; 2026-09-27 决定直接采用系统级树)

状态: **RUNNING; S1 实现+实测完成, 但改动已于 2026-09-27 按用户指示"放弃当前全部修改"回滚 (快照 `artifacts/t32-s1-worktree.patch` + temp); S2a 已停 (analyst 误读, 用户纠偏); 需求三条见"范围澄清", 待重写方案**; 执行者: analyst(设计)/implementer(实现+GPU);
动因: T31 封顶 ~+12% (31 tps) vs 一次 128K re-prefill = 257s / 会话切换 260s -> 2-6s (ROI 对比后转 T32);
来源: 用户 agent 工作流实测现象 (长任务下轮从头 prefill; 两长会话轮换互相淘汰)

## 范围澄清 (用户 2026-09-27 权威定义; 覆盖此前所有 "VRAM 活分支共享" 解释)
需求 = 三条, **VRAM 内复用不在其中**:
1. 每个 user 回合结束思考被剔除, 常出现"剔除后从头 prefill" -> 查/修 (即子问题 1)
2. n 个序列跑在不足 n 个槽位里时, 完工序列保存到 RAM 是 llama 默认动作; 默认 RAM 不够大 -> 轮换时目标被删 -> 从头 prefill -> 修 (即子问题 2)
3. 现保存机制 = **整个序列**存入 RAM 或 SSD; 要求改为**树状存储在 RAM+SSD**:
   - 枝干优先存 RAM; 溢出到 SSD 时优先溢出**叶**
   - 工作序列从槽位删除时: **先尝试上树; 不能则新建树**
   - 新序列**从 RAM/SSD 的树上复用**后进槽位计算
- **纠偏**: "S2a 活分支 fork (VRAM 内 seq_cp 共享活分支前缀)" = analyst 误读, **停止投入**; 其产物 (harness/COW 结论/
  `--kv-unified` 前提/多序列数值不一致) 仅归档参考, 不属本任务需求。**原 "S2b: 节点级 parking / 按区间序列化" 才是子问题 3 的正主, 现升为主线。**

## 子问题 1: 检查点被挤掉导致长回合从头 prefill
- 现象: 思考剔除后 (preserve_thinking=false 或客户端侧剔除), 长 agent 回合的下一轮触发大范围重算;
  短回合正常 (从上一轮 user 末尾恢复)
- 机制 (代码事实):
  - 模板 (Qwen3.8-27B 内嵌): `preserve_thinking` 未定义=永远保留; =false 时只保留"最后一个真实 user 之后"的
    assistant 思考 (tool 块由 delimiter 正确识别为 TOOL, `common/parsers/qwen3-coder.cpp:32-38`)
  - 分叉 D = U_k 结束后第一个 token; 最小重算 = 本回合 assistant+tool 内容 (去思考)
  - server 会在每个请求 prompt 末尾前 4+ub/4 处专门建检查点 (`server-context.cpp:3560-3576`, PR 20288)
    = 正好 U_k 末尾; 但每个 tool 步骤请求也各建一个 -> 32 上限很快满 -> 压缩循环删"距前一检查点 <8K 的旧检查点"
    -> U_k 末尾检查点可能被提前挤掉 -> 退到更早 -> 重算范围翻倍; 极端 do_reset
    (日志 `forcing full prompt re-processing due to lack of cache data`)
- **根因确认 (2026-09-27, analyst; 证据链完整)**: 用户启动命令带
  `--chat-template-file D:/LLM/Backend/Chat-Template/llama.cpp/Qwen3.5-chat_template.jinja`;
  该自定义模板第 103 行**硬编码** `{% if loop.index0 > ns.last_query_index %}` 保留思考, **完全不实现 `preserve_thinking`**
  -> 能力探测 `supports_preserve_reasoning=false` (caps.cpp:529-538) -> 服务端 "默认开启" (arg.cpp:958-961) 无从生效;
  实际行为 = 每个新 user 消息到来时剥掉上一轮所有 assistant 思考 (旧思考在上下文里消失)
- 证据: (1) opencode DB (`opencode.db`): 每条 assistant 消息均存有 reasoning part, 且**轮内**被发回
  (下一步 prompt 增长 ≈ 上一步输出 + 工具内容); (2) 跨 user 边界 prompt 缩短 ≈ 上一轮思考量 (两次实测 824 / 6329 token),
  且分叉点 = 上一轮首条 assistant 消息 (llama.cpp prompt cache 命中位置 10654/15209 = 该轮 user 之后);
  (3) 内嵌模板 (GGUF, 已抽取) 第 119 行 = `preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index`;
  自定义模板无该条件, 且也缺 reasoning_effort 注入 (用户 opencode variants low/medium/xhigh 被忽略)
- **修复 (零服务端代码, 待用户应用)**: (a) 去掉 `--chat-template-file` 用内嵌模板 (顺带恢复 reasoning_effort); 或
  (b) 自定义模板第 103 行改成内嵌版同款条件。修复后 prompt 追加式增长 -> 长回合 re-prefill 从根上消失;
  代价 = 思考常驻使上下文变大 (容量问题更依赖 T32-2/树)
- **用户选用模式 (2026-09-27 决定): 删除自定义模板 -> 内嵌模板 + `--no-reasoning-preserve`** (理由: 上下文不足, 必须每轮剥思考):
  语义 = 只保留"最后一个真实 user 之后"的思考 (当前回合, 含 tool 子步); **代价 = 每个新 user 边界 re-prefill 上一轮内容**
  (T32-1 现象以更小规模回归; S1 的 pin 把恢复点钉在 D-4 使其最小化: 实测 2762 tok/5.1s @32K+36 步场景)。
  缓解建议 (按性价比): (1) `reasoning_effort` low/medium -- 内嵌模板生效, 源头减思考, **保持追加式零 re-prefill**;
  (2) opencode compaction 触限一次性改写; (3) 控制单轮工具输出体积。注意: 切内嵌后 system prompt 多一行 reasoning
  指令 (及变体切换) -> 首次/每次变体切换一次全量 prefill (一次性)
  **用户 2026-09-27 明确重申: 上下文是硬约束, 剥思考为既定策略, 不再讨论"保留思考"方案; 一切优化的目标 =
  最小化剥思考的代价 (S1 pin / S1.5 瘦身 / S2a fork 复用), 不是取消剥离。**
- 对策候选:
  a) 客户端不剔思考 (prompt 只追加 -> 零重算; 代价 = 上下文变大)
  b) fork 补丁: pin "最后一个 user 消息末尾"的检查点, 不参与压缩/逐出 (小改 server-context.cpp)

## 子问题 2: prompt-cache 容量不足导致两长会话轮换互相淘汰
- 现象: 两长会话严格轮换时, 每次切换都从头 prefill
- 机制 (代码事实, `server-task.cpp:1711-1900`):
  - `alloc` 前所有 idle slot 都被 save (`server-context.cpp:2436`)
  - 淘汰 = **FIFO pop_front** (最早保存); 恢复的条目会被移出缓存, 下次停放重新入队尾 (近似 LRU)
  - token 上限动态: `max(n_ctx, limit_size/每token字节)` -> 调大 `--cache-ram` 同时放宽
  - 单条状态 > limit -> 跳过保存 (WRN); `load` 有 `f_keep<0.25` 保护
  - 128K 会话状态 = ~7GB -> 8GB 只能停 1 个; 两长会话轮换 = 每次互相淘汰 = 每次全量 prefill
- 诊断日志 (WRN, 默认可见):
  - `making room for prompt cache entry, removing oldest entry` (thrash 实锤)
  - `cache token limit ... reached` / `prompt state size ... exceeds cache size limit, skipping` / `found better prompt`
- 对策: `--cache-ram 16384` (两长会话, 需实测 RSS; 31GB RAM 留余量) / 避免长短会话轮换 / 更短上下文用双 slot

## 子问题 3: 三级会话存储 (用户 2026-09-26 提出)
目标形态: L1 VRAM 工作区 (KV 池) -> L2 RAM 热复用区 (2 长 + 若干短轮换) -> L3 磁盘冷复用区 (被 L2 淘汰的序列)

现状对照 (代码事实):
- L1: 已存在 (KV 池按 `-c` 预分配; parking 不缩小池)
- L2: `--cache-ram` (默认 8192MiB); **FIFO 淘汰** (pop_front = 最早保存); token 上限动态
  `max(n_ctx, limit_size/每token字节)`; 命中恢复后条目移出缓存 (下次停放重新入队尾)
- L3: **API 已存在但不自动**: `--slot-save-path` + `POST /slots/{id}?action=save|restore|erase` + `{"filename"}`
  (`server-context.cpp:5290/5326`); 无 "RAM 满 -> 溢写磁盘"、无 "miss -> 自动回读"
- 身份判断: 默认无会话 ID; L2 靠 token **LCP 匹配** (`get_common_prefix`, `f_keep >= 0.25` 且 f_keep/f_sim 双高);
  显式: 请求 `id_slot` 固定槽位; L3 用 **filename** 精确命名 (身份由客户端定义)
- 触发时机: 保存 = 任务分配槽位时存当前 prompt (`server-context.cpp:1647`) + 新任务启动时存所有 idle slot
  (`:2441`); 加载 = 分配槽位时尝试换入更匹配条目 (`:1649`); **从头 prefill 是结果不是触发条件**
  (miss 后 LCP 小 -> n_past≈0)

实现路径:
- 阶段 1 (**零 server 改动, 推荐先做**): agent 客户端切换会话时调 slots save/restore (文件名 = 会话 id);
  L2 提到 `--cache-ram 16384`; 预期两长会话轮换从 260s 全量 -> ~2-4s (磁盘读 + H2D @PCIe x4)
- 阶段 2 (可选 fork): `server_prompt_cache` 自动分层 (evict -> 磁盘, miss -> 回读, 身份键); 中等工作量

## 阶段 2 设计草图 (2026-09-26 analyst; 用户迭代中, 未定稿)
统一抽象:
- **一个索引 (RAM 常驻) + 后端链 (RAM -> Disk -> Delete)**; 索引存 每条目 token 列表 (~0.5MB/128K)/大小/最后接触/pin/所在层
  -> 跨层相似度匹配都在内存里做 (磁盘不用遍历); 磁盘文件头本身也含 token 列表 (格式: MAGIC|VERSION|n_tok|tokens|state)
- **pin 纪律**: 选定目标后先 pin; 一切淘汰跳过 pinned; 全 pinned 则拒绝本次溢写/切换, 绝不删除目标 (消除连锁淘汰病理)
- 切换状态机: pin(T) -> park(current, L2 无位则直写 L3/溢写最老 unpinned) -> restore(T, L2 命中 H2D / L3 读文件流式 H2D) -> unpin(T)
- 成本: 切换 ~4-8s (D2H+H2D 各 ~2s @PCIe x4, 加磁盘读写); 对比全量 prefill 260s

容量数学与不足策略:
- N 槽位保留 K 个会话 (每个大小 S) 需 tier 总容量 >= (K-N+1)*S; 用户场景 N=1, K=2, S~7GB -> **>=14GB**
  (8GB RAM 单独不够: 需 L3 >=6GB 或 RAM=16GB)
- 不足时: 明确降级 = 拒绝切换 (显式报错, 保留现状) 或允许丢失 (该会话下次全量 prefill); 必须日志/API 可见, 不静默
- 不建议"开机即按 K 预分配": 容量规划交给配置 (cache-ram / slot-save-path 目录)

想法 A (RAM+SSD 当一块): 逻辑统一 (单索引/单淘汰/pin/双后端) 即成立; 实现两种:
- (a) 显式分层 (推荐): 可预测, RAM 层纯内存拷贝, 磁盘层显式读写
- (b) 写穿磁盘 + 靠 OS page cache: 实现最简 (RAM 层白送), 但淘汰/延迟不可控, 车内 NVMe 上可接受性待测

### 树状存储 v2 (用户 2026-09-26/27 深化) + analyst 可行性分析
> **[2026-09-27 用户纠偏] 本节及以下 "VRAM = 活分支工作区 / 分支=sequence + seq_cp 活分支共享" 的解释作废。
> 用户需求 = RAM+SSD 上的**树状存储组织** (枝干优先 RAM / 溢出先叶 / 上树-新建-复用), 与 VRAM 内复用无关。**

澄清 (2026-09-27, 已被上条纠偏): 树 = **系统级全局管理** (服务端全局 forest + 全局索引/淘汰; 分支 = llama sequence;
槽位只是"当前服务哪个分支"的 worker; VRAM = 活分支工作区). **不是每个槽位一棵树**。轮换 = 空根森林的退化情形;
树在概念上取代轮换, 但**停放/分层机制不消失而是升级为节点粒度** (冷分支的区间 KV + checkpoint 序列化, 引用计数, 叶淘汰)。
用户设计:
- 引擎只认相似度不认"同一会话" -> 用前缀树组织 KV; attention 天然可树; GDN(recurrent) = 树上各深度处的检查点
- 检查点位置若规范化 (同一前缀总在相同位置建点) -> 检查点可跨分支复用; 否则需淘汰机制控密度防膨胀, **按叶淘汰**
- 新序列: 与 VRAM 相似度 > 阈值 -> 直接回滚复用; 否则 VRAM 序列尝试上树 (不行则新建树), 新序列再从树中最大复用
- 枝干优先 RAM, 向 SSD 溢出优先叶; 无 0.25 门槛, 能复用就复用
analyst 可行性 (代码事实):
- **attention KV 跨分支共享已具备**: `seq_cp` 在同 stream 下只做 `cells.seq_add` (零拷贝共享, 按 seq 成员引用计数)
  (`src/llama-kv-cache.cpp:451-490`, TODO tag `[TAG_KV_CACHE_SHARE_CELLS]`; 跨 stream 才复制)
  -> 树可建在"一个分支 = 一个 llama sequence"之上, **不必改 cell 模型**
- **recurrent 不能共享**: 每分支的 GDN 状态是独立 blob。两条路: (a) 分支保持存活各占一份 (144MB/分支 @replay=1,
  10 分支 = 1.4GB VRAM); (b) 检查点 + 重放 (每个检查点也是 144MB) -> 密度/容量是主成本, 需淘汰策略
- **检查点规范化可实现**: 模板消息边界对相同前缀确定性一致; 或每 N token 网格; 密度 vs 重放长度是权衡
- **无 0.25 门槛成立**: 树恢复 = "到最近祖先 + 重放 suffix", 顺理成章
- 树 vs 轮换边界: 收益 = 共享前缀大小 x (分支数-1); **会话间无共享时树 ≈ 轮换** (根为空), 无额外收益
- 约束: 存活分支数受 `n_seq_max` 预算限制 (提高有开销); 服务端需新组件 (分支生命周期/树索引/按叶淘汰/分层)
- 工程: 因 KV 共享已存在, 不是 SGLang 级重写; 粗估增量 1-2 周 (最小可用版)
分期建议:
- **S0/S1 合并 (现在, 无 GPU)**: 检查点规范化 + 去 0.25 门槛 + 跨 blob 最近祖先恢复 (含轮换修复: 次序/pin) -- 立即缓解
- **S1 (便宜)**: 检查点规范化 + 去掉 0.25 + LCP 最近祖先恢复 (无 KV 共享, 前缀在各停放状态中重复)
- **S2 (真树)**: 分支=sequence + seq_cp 共享 + 枝干 VRAM/RAM + 叶淘汰 + SSD 分层; **用户 2026-09-27 决定直接启动
  (跳过决策门, 理由: 最差=轮换, 收益>=0)**
- 决策门 (原计划) 已跳过; 前缀共享度现在只决定**收益大小**: 无共享时系统级树行为=轮换 (容量/开销同), 有共享才赢
- 注: 树不替代 S0/S1 的修复 (思考剔除的重算/唯一内容的容量数学不变); 节点级 parking (按区间序列化) 是新格式, 属 S2b
已决定直接做 (用户 2026-09-27); 前缀共享度只决定**收益大小** (无共享时树行为=轮换, 有共享才放大)

## S2 系统级树管理: 正式设计 v1 (2026-09-27, 用户决定直接采用树方案)
决定: 用户 2026-09-27 "直接用树比较好" -- 理由: 最差=轮换 (收益>=0); 先剪枝叶 + 合理淘汰可保留大量高复用短前缀,
降低计算与等待。**跳过纯轮换路线, 由树方案取代。**

analyst 注记 (3 个不能忽略的工程点):
1. "最差=轮换" 只对**稳态行为**成立, 不对**实现风险**成立: 新增复杂度会推迟交付并引入缺陷 -> 必须分阶段,
   每阶段可独立验证 + 回退到当前生产 (T24 构建)
2. recurrent 检查点**按节点收费** (144MB/节点); 共享前缀只付一次 -> "保留大量短前缀"的成本 = 节点数 x 144MB,
   密度策略是核心 (不是每个节点都必须存点, 可用祖先检查点 + 重放)
3. 活分支数受 `n_seq_max` 限制 (每活分支一个 seq id) -> 需显式预算 + 超限淘汰最冷分支

### 存储角色定位 (2026-09-27 澄清, 用户问)
目标态: **VRAM = 推理工作区 ("推理槽", live 分支的 KV+状态, 唯一能算的地方); RAM = 索引 + 热停放; SSD = 冷停放**。
但不"彻底":
1. attention KV 没有独立的"存储实体" -- cells 在 VRAM 是唯一原身; 冷分支必须 S2b 区间序列化后才真正落 RAM/SSD
   (S2a 阶段冷分支仍是整状态 blob = 轮换的存储模型)
2. live 分支数仍由 VRAM 决定: 共享只让 trunk 计一份; 每个活分支的唯一后缀 KV + GDN 状态 (144MB) 仍必须常驻 VRAM
3. recurrent (GDN) 不能像 KV 那样"按需取用": 要么每分支常驻 144MB, 要么节点检查点 (144MB/点), 要么**祖先检查点 + 重放**
   (消耗算力, 非零成本) -> recurrent 是"快照/重放"模型, 不是"存储指针"模型
4. 每次"从 RAM/SSD 激活" = H2D 搬运 (~2s/GB @PCIe x4) + (可能)尾部重放; 树降低的是**重复存储**, 不消除搬运
5. **VRAM 红线: 中性或下降** -- 树不把存储复制进 VRAM; 只提高"同一 VRAM 的复用效率" (共享前缀计一份)。
   唯一会使 VRAM 上升的是主动选择: 保留更多活分支 (每分支 + 唯一后缀 KV + 144MB 状态) / 把热检查点缓存进 VRAM (不做);
   另需监控共享 cells 的池碎片化。每阶段验收强制核对 VRAM (基线 = T24 部署 -420MiB)

### 架构 (系统级, 全局)
- **tree_index (RAM, 全局)**: node {id, parent, token 区间 [p0,p1), tokens/hash, refcount, last_used, tier, checkpoint?}
- **branch**: 每活分支 = 一个 llama sequence; 新分支复用祖先 cells = `seq_cp` (同 stream 零拷贝); 分支死亡 -> 释放 seq,
  cells 按 refcount 回收
- **术语**: 共享 = **零拷贝引用去重** (同一份 cell 被多序列引用, 引用计数), **不是压缩** (无编码/变换, 字节不变);
  仅 attention KV 可共享 (K/V 是前缀位置的纯函数); GDN 状态不可共享 (逐 token 递归, 依赖被 draft 的 token 历史)
- **checkpoint manager**: 节点 recurrent 状态 (144MB); 位置规范化 (消息边界 / 每 N token); T24 replay 管短回滚;
  超预算 -> 只在分支点保留
- **eviction (全局)**: 叶优先 (整支删除); 共享节点按 refcount; 评分 = recency + 重算成本 + refcount
- **tiering (全局)**: 活分支 cells 在 VRAM; 冷分支 -> (a) 分支级整状态 blob (现有格式, 先做) / (b) 节点级区间序列化 (后做, 去重省 RAM/SSD)
- **与 slots 正交**: slot = 当前服务哪个分支的 worker; 分支可在 slot 间移动

### 分阶段 (每阶段独立可用, 收益单调不减)
- **S1 (前置, 便宜)**: 检查点规范化 + 去掉 0.25 门槛 + 跨停放 blob 的最近祖先恢复 (无共享; 立即获得部分复用 + 修 T32-1/2)
- **S2a**: 活分支树 (VRAM): 分支=seq + `seq_cp` 共享 + 叶淘汰; 冷分支仍用整状态 blob (RAM/SSD)
  收益: 活分支间共享长前缀 + 修剪死尾
- **S2b**: 节点级冷存储: 区间 KV + checkpoint 序列化 (新格式) + 启动索引重建
  收益: 停放态共享去重 (大量高复用短前缀的 RAM/SSD 占用大降)
- **验收 (每阶段)**: 正确性 = 与 blob 路线逐 token A/B; VRAM/RAM 占用; 切换耗时; agent 端到端 (长任务下一轮 TTFT);
  PPL 控制位; 生产不受影响 (dev 分支)

### 风险与对策
- 正确性 (引用计数 / cells 共享 / T24 回滚交互) -> 逐 token A/B vs 现有 blob 路线; 保守 pin 策略
- 复杂度 (~1-2 周) -> 分阶段; 生产保持 T24 构建不动
- n_seq_max 超限 -> 显式上限 + 淘汰最冷活分支
- 检查点预算 -> 密度策略 (分支点优先) + 用祖先 + 重放替代

### 可行性评估 (analyst 2026-09-27) + 去风险实验清单
结论: 分层可行, 但不能一次性承诺:
- **S1: 高把握 (90%+)** -- 全部基于现有机制 (检查点创建/选择, load 门槛, LCP 匹配), 改动小, 立即可验证
- **S2a: 中等 (60-75%)** -- 核心假设 "`seq_cp` 同 stream 零拷贝共享 + 多活分支并发 decode 正确" **未经实践检验**
  (代码里 `[TAG_KV_CACHE_SHARE_CELLS]` 仍是 TODO); 另有 `n_seq_max` 与 recurrent 144MB/分支 的预算约束
- **S2b: 重且后置** -- 新序列化格式 (区间 KV + checkpoint) / 索引重建; 仅在 S2a 实测收益 + 短前缀共享频次确认后启动
去风险实验 (0.5 天, 零 server 改动, harness/CLI):
1. `n_seq_max>=4`: 序列 A 预填前缀 P -> `seq_cp(A->B)` -> A/B 各自续写 -> 与"不共享"路线逐位对比 (共享 cells 批处理正确性)
2. 填满 cache 触发淘汰/重分配: 验证 refcount>0 的共享前缀 cells 不被误删; 分支死亡后正常回收
3. hybrid 交互: 共享 attention 前缀的两序列 (GDN recurrent 各自独立) 并发 decode 正确; T24 replay 回滚不受影响
4. 预算实测: 2/4 活分支的 VRAM (recurrent 144MB x 分支 + 唯一 KV + n_seq_max 抬高开销)
判据: 1-3 全对 + 4 在预算内 -> S2a 开工; 任一项失败 -> 记录数据, 退回 S1 作为终点

## 业内实现参考调研 (analyst 2026-09-27) -- 树/分层/混合模型都有先例, 不必从零发明
**纯 attention 树 (成熟)**
- SGLang **RadixAttention / `radix_cache.py`**: 就是"前缀树 + LRU 叶淘汰 + 引用计数", 逻辑可直接移植 (端口, 非拷码);
  `TreeNode` 还自带 `host_value` / `host_ref_counter` / `write_through_pending_id` = **节点级主机层存储已有先例** (对应我们 S2b)
- SGLang session-aware radix cache: 已含 **Mamba 分支** (复用状态挂在注册叶上, 两级 LRU); 淘汰策略 lru/lfu/slru/**tlru**
  (arXiv:2510.15152, 专为 agentic 多轮设计: 保"下一轮 TTFT 需要的尾部", 其余先淘汰) -- 与 T32 场景高度对口
- vLLM: APC 按块哈希 (hash + 块链, 无显式树); `--kv-offloading-size/backend` 自带 CPU 卸载
**混合 (attention + recurrent) 前缀缓存 (年轻但有直接同构)**
- vLLM `mamba_cache_mode` = none/**all** (每 block 位置都存状态)/**align** (仅对齐位置, 开前缀缓存时默认):
  与我们"规范化检查点 + 尾部重放"同思路
- vLLM issue #45238 (2026): align 模式在共享前缀负载下命中率 **0%** -- 因为每请求只存"末尾前一个对齐点"的状态,
  落在请求私有 token 上就白存; 其修复方向 (b) = **全局固定间隔 M 存点** (= 我们 S1 的检查点规范化), 直接佐证 S1 设计
- vLLM PR #54637: GDN hybrid (Qwen3.5 家族, 同我们模型族) prefix caching + **MTP** 的 V2 支持 -- 同构组合可参考
- vLLM `--replayssm` 内核: "缓存最近 SSM 输入, 跳过每步全状态写, 只在 checkpoint flush 时写回" -- 与 T24 ReplaySSM
  同名同思路 (Mamba2 Triton; 需 none/align 模式); 注意 vLLM issue #39809: mamba 前缀缓存 + MTP 有连环 bug (警示:
  我们的组合要进测试矩阵)
**判读**: (1) 树/哈希索引、叶淘汰、节点级主机层、agentic 淘汰策略全部有现成逻辑可抄 (SGLang 最完整, vLLM 最贴近
hybrid+MTP); (2) 真正无先例的只有 **llama.cpp cells/seq + hybrid memory 管理器 + 区间序列化** 这一层胶水; (3) 因此
S2 的风险从"发明算法"降级为"移植 + 接引擎", 冒烟实验依然是必要门 (引擎层共享语义各系统不同)。
来源: SGLang radix_cache.py (v0.5.16/main) + SGLang eviction docs + vLLM cache config docs + vLLM #45238/#54637/#39809。

## 子问题 3 补充: 容量极限与淘汰次序的设计规则 (2026-09-26 讨论)

### 容量需求分析 (2026-09-26 修正: 上一版 "容量=1 即可" 是错的)
- 事实: `server-context.cpp:1647-1651` 是 save(current) 先于 load(target); 容量=1 时 alloc(current) 会 pop_front
  挤掉唯一目标 -> 目标 miss -> **立刻**全量 prefill
- 但改成 load-first 也救不了容量=1: load 会把目标写进 VRAM, **覆盖正在工作的序列**; 工作序列必须在被覆盖前
  写到某处, 而它的目的地必须在"目标的停放副本仍存在"时就已经空出来 -> **切换瞬间必须同时存在 2 份状态副本**
- 结论: **无损双序列轮换需要 2 个停放容量 (总量, 可跨层)**; 推广: **停放格数 = N (N = 轮换序列数)**,
  稳态 N-1 格被占 + 1 格空 (空位是切换的必需品; 无空位 -> 必须丢最不常用的)
  - 可拆法: **1 RAM 格 + 1 磁盘格即可** -- 出序列存到"空闲的那层", 入序列从"被占用的那层"读; 角色每轮交替
    (磁盘在这里不是缓存层级, 而是第二格; 每轮切换一次磁盘往返 ~4s)
  - 顺序优化 (load-first) 仍有价值: 容量 >= 2 时提前腾格 / 避免不必要的淘汰; 但**不是容量=1 的解法**
  - 总量=1 时: 唯一选择 = 放弃工作序列 (延迟成本, 将来回来才付 260s), 绝不放弃目标 (现在就要付)
  - 内存双格 (16GB) 是最顺的形态; 8GB RAM + 磁盘第二格是退路; 只有 8GB 单格 = 必然 thrash


### 淘汰优先级 (空间仍不足时)
1. pin 目标 > 一切 (选定目标后立即 pin, 所有淘汰跳过 pinned)
2. 优先用"消费目标"腾出的空间
3. 淘汰最老的 unpinned (L2 -> 溢写 L3)
4. L3 满 -> 删最老的 unpinned
5. 全被 pin 或仍不够 -> **放弃当前工作序列** (延迟成本: 将来回来才付)
   - 绝不放弃目标 (立即成本: 现在就要 260s 全量)
   - 用户直觉 "当前写不进去就直接删" 正确, 且应只是最后手段

### L2/L3: 索引/策略统一, 介质分层
- 单一 manager + 单索引 (tokens/recency/pin/tier); 放置规则 = 按 recency 降级 (RAM -> SSD -> 删除)
- 这样"热序列落 SSD"不会发生 (最近使用的留 RAM, 只有最冷的被降级); 访问成本差异 (RAM hit ~2s vs SSD hit ~2-4s) 保留层级
- 不做"平铺随机放置"的统一池 (写寿命/延迟无谓消耗)

### 树状存储 v1 -> 删除 (2026-09-26); **v2 见上节 (已采用 2026-09-27)**

### 目标形态与预期 (用户 2026-09-26 愿景 + analyst 校准)
用户愿景: 16GB RAM + 32GB SSD (~5 个满 ctx 停放格) + VRAM np=2 + unified KV;
短序列同步工作, 长序列交替切换, SSD 写入量可控。

校准 (数字):
- 停放格: 128K 会话状态 ≈ 7.2GB -> 16GB RAM ≈ 2 格; 32GB SSD ≈ 4 格; 合计 ~6 格 (用户"5 个"约数成立, 留余量)
- VRAM np=2 **不可能同时驻两份满 128K**: KV 2x6.71 + 权重 20.46 + compute ~1.3 = **35.2 GiB > 32 GiB**
  - 可行组合: 2 x ~64K 并发 (28.2 GiB) / 1 长 128K + 1 短 ~32K (30.1 GiB)
  - 建议 `-c` 设 ~150-170K (unified KV 总池), 超出部分靠停放切换
- 长序列 = 必然交替 (一次一个, 切换 ~4-6s: D2H + H2D @PCIe x4), 与用户描述一致
- SSD 写入: RAM 双格覆盖"两个长会话轮换"**零 SSD 写入**; 第 3 个长会话起才溢写 (~7GB/次), 磨损可忽略
- 短序列并发: np=2 权重读一次摊两份 -> 聚合 decode 近 2x 单流 (26.6 -> ~45-48), MTP 在 np=2 已验证 (T24)

**与 T32-1 的边界 (预期管理)**: 三级缓存只解决"会话切换"; agent 每轮新对话的重算来自**思考剔除导致上一轮 content 无法复用**
(分叉点 D = 上一轮 user 消息之后)。即使分级完美, 每轮仍会重算"上一轮内容"; 要零重算只有**不剔思考** (prompt 只追加,
代价 = 上下文变大)。两条收益是独立叠加的, 不要混淆预期。

### 待细化 (2026-09-27 继续): 多序列切换 + 分级淘汰 (低容量边界)
- 目标: 在任意 RAM/SSD 容量组合下做到**稳定切换 + 分级淘汰 (写入/删除)**
- 高容量情形简单 (全部驻留); **低容量边界是问题集中区**, 需逐档推敲, 例如:
  - 8GB+0 (1 格) / 8GB+小磁盘 (1.5 格) / 2 格但序列 7+7+短序列 / N 序列 > 格数 / 长短混合
  - 写入盘时空间不足 / 读回失败 / 半途中断的回退
- 已确立的规则 (沿用):
  - 停放格数 = N (N 个轮换序列); 稳态 N-1 占 + 1 空 (**空位是切换的必需品**)
  - 磁盘 = 第二格, 角色每轮交替 (不是缓存层级)
  - pin 目标 > 先消费目标 > 淘汰最老 unpinned (L2->L3 溢写) > L3 满删最老 unpinned > 最后弃工作序列
  - 保目标 (立即成本), 弃工作序列 (延迟成本)
- 待定: 淘汰评分 (recency / 大小 / 重算收益) / 索引重建 (启动时扫盘) / 与现有 `server_prompt_cache` 的接口 / 多 slot 并发切换


## 交付物 (2026-09-27 更新)
- **S1: DONE (2026-09-27)** -- A/B/D 已实现 + T32-1/2 实测通过 (见文末 "S1 结果"); patch 待用户批准提交
- **冒烟实验: DONE + 已停 (2026-09-27 纠偏)** -- "S2a 活分支 fork" 判为 analyst 误读, 产物仅归档 (见 "范围澄清")
- **待决**: S1.5 检查点瘦身 / **树状 RAM+SSD 存储设计 (子问题 3 正主, 待重写)** / 模板应用 (T32-1 源头) / T24 replay 复查
- S2a/S2b 每阶段逐 token A/B vs blob 路线, 生产保持 T24 构建; 子问题 2 配置实测并入 T04


## S1 执行计划 (implementer 整理, 2026-09-27; 设计以本文档上文为准, 本节只做落地细化; 代码事实已逐条核对)

### 0. 两条新实测/容量事实 (来自 T31 生产日志 + 代码, 影响设计)

**(1) 单个 context checkpoint 实测大小 = 162 MiB + ~4.1 KiB/token** (ctv q8_0, ub512, MTP on)
- `t31_mtp3.err`: n_tokens=1 -> 161.77 MiB; n_tokens=127754 (= prompt_end-516) -> 663.73 MiB;
  n_tokens=128266 (= prompt_end-4) -> 665.74 MiB; 512 token 增量 = 2.01 MiB -> 4.02 KiB/token
- 构成推断: `data_tgt` (PARTIAL_ONLY, 近似常数 ~162MB, 与 48 层 GDN recurrent 量级一致; T24 记录 144-157MB)
  + `data_dft` (draft/MTP KV, 随长度线性 ~4.1KiB/token)
- 后果 1: **128K 处每个检查点 ~666 MiB**; 32 上限 = 21 GB; 全局固定网格 M=8192 需 16 点 = 10.6 GB
  -> **S1 不做固定网格** (vLLM align 的 "固定间隔 M 存点" 每点成本模型与我们不同: 我们每点 666MB, 必须服从 RAM 预算);
  pin 集必须极小 (见 A)
- 后果 2: 检查点位置实测命名 (ub=512): prompt_end-516 与 prompt_end-4 (:3565 {4+n_ubatch, 4});
  turn 首请求 (prompt 末尾是当前 user 消息) 时这两个点 = "U_k 末尾锚点" = 下一轮思考剔除的分叉 D 附近

**(2) prompt cache blob 含 checkpoints** (server-task.cpp:1722-1735 计入大小, :1782 拷贝; load 时 :1862 随 tokens 移回)
- 128K blob ~= 7.2G (全量 state, server-context.cpp:304) + 检查点 (如 4 个 = 2.6G) ~= **9.8G**
- => 默认 `--cache-ram 8192` 时整条 blob 被 `prompt state size ... exceeds cache size limit, skipping` 拒收
  -> **T32-2 的轮换成本部分是 "跳过" 而非 "FIFO 淘汰"** (诊断第一步就抓这行 WRN)
- => `--cache-ram 16384` 只够 1 条 128K blob; 双长会话需 2 条 = 19.6G, RAM 31G 可行但紧 -> RSS 实测仍是 P1

### 1. 变更清单 (S1, 建议施工顺序 A -> B -> D -> C 验证)

**A. 检查点锚定 (pin) + 淘汰纪律** (修 T32-1 主症)
- `common/common.h:1165` `common_prompt_checkpoint` 加 `bool pinned = false;`
- `server-context.cpp:2309 create_checkpoint`: 新参数标明 "prompt 尾锚点";
  判定在调用点 :3633: `near_prompt_end` (:3586) 且 `slot.task->n_tokens() - last_user_pos <= n_ubatch + 8` (U 是最后一条消息)
  -> 只 pin offset=4 那一个 (end-4; 1 点/回合, +666MB@128K)
- 淘汰循环 :2316-2329 (min-step 压缩) 与 :2331-2339 (32 上限 FIFO) 都跳过 pinned;
  32 上限循环改为 "删最老 unpinned, 全 pinned 则跳出"; 创建新 pin 时统一解除旧 pin (只保留最新回合锚点, pin 数 <= 1-2)
- 验收: 复现脚本里长回合后 `restored context checkpoint` 落在 U_k 锚点 (而非 `forcing full prompt re-processing`)

**B. L2 选择改 max-LCP + 去 0.25 门槛** (修 T32-2 命中率)
- `server-task.cpp:1813` 删除 `f_keep_cur < 0.25f` 跳过; :1817 现条件 "f_keep/f_sim 双指标同时严格变好" 会漏掉
  "LCP 更长但 f_keep 更低" 的候选 -> 改独立排序: 主序 `lcp_cur`, 次序 `f_keep_cur`; 当前 slot 自身仍作基线 (不 load)
- 保留**绝对下限**参数 (新 env, 默认 0 = 关闭, 与文档口径 "能复用就复用" 一致; 供实验扫): load 代价与 blob 大小成正比
  (全量 state 读回+H2D ~2s), 与 LCP 无关
- 验收: 两长会话轮换 + 同一会话暂停回来, `found better prompt`/n_prompt_cached 上升

**C. 跨 blob 最近祖先恢复** (无新代码, 由 A+B 生效; 只做验证)
- 链路已具备: load 把 blob 的 tokens+checkpoints 移入 (:1862); 之后 :3219 n_past = LCP(blob, new);
  :3349-3385 找 `pos_max <= pos_next` 且 `pos_min < pos_min_thold` 的检查点回退
- 前提: blob 里有点 <= 新分叉点; A 的锚点随 blob 自动保存/恢复; 无点则退化为 do_reset (正确但慢)
- 验收: 会话 A -> B -> A, 日志 `restored context checkpoint` 命中; 逐 token A/B 与不切换一致

**D. 轮换次序 (select-before-save + protect)** (修 T32-2 互相淘汰)
- 现状 :1647-1649 `prompt_save(current)` 先于 `prompt_load(target)`; save 的 FIFO 逐出可能删掉即将 load 的目标
- 改: (1) 新增 `server_prompt_cache::select(tokens)` (B 的准则, const, 不改状态);
  (2) `server_slot::prompt_save` (:299) 加 protect 参数 -> 传给 `alloc`, pop_front/update 循环跳过 protect;
  (3) :1642-1656 改 select -> save(protect) -> load(selected)
- 边界: "fully contained in current" 删除循环 (:1737-1748) 若删到 protect: 安全 (current 的 LCP 必 >= 它), 加日志;
  容量不足时行为 = 维持现状 (不静默丢), 与容量数学一致 (双长会话需 2 格)
- 验收: 两长会话切换 260s -> ~2-6s; thrash WRN 消失

**E. 配置/文档** (并入 T04): `--cache-ram` 实测 RSS (16384 起步); `n_ctx_checkpoints` 是否需要提高 (S1 后测);
`checkpoint_min_step` 无需改 (锚点已有 :3554 / :3624-3628 的 last_user/near_end 例外)

### 2. P0 诊断/验收脚本 (短 GPU ~15-30min) `t32_repro.ps1` (待写)
- server: 先生产同构建 (llama.cpp-t24), `-lv 4` + `LLAMA_SERVER_SLOTS_DEBUG=1`, `--cache-ram 8192`, `-c 135168`,
  ub512, replay=1, np1; S1 后换 fork 构建, 同脚本转绿 = 验收
- 场景 1 (T32-1): 模拟 agent 长回合: U_1 -> 20+ tool step (/v1/chat/completions 带 tools;
  assistant/tool 消息; 最后切换 `chat_template_kwargs={"preserve_thinking": false}`) -> 抓 n_prompt_cached 与检查点生死
- 场景 2 (T32-2): 两条长会话 (id_slot=-1) 严格轮换 3 次 -> 抓 `making room` / `exceeds cache size limit, skipping` /
  `cache size/token limit reached`
- 必须走 /v1/chat/completions (message_delimiters 由 chat template 提供, server-common.cpp:1384), 裸 /completion 无 user span

### 3. S2a 冒烟 (0.5d, 零 server 改动) -- 开工门
- 基建: `tests/test-save-load-state.cpp` 已有 `test_seq_cp_host/device/scatter` (:307/:373/:438) -> 照其模式写临时 harness
  (私有 fork; 不进 tests/ 收尾, 避免 PR 面)
- 4 项: (1) A 填前缀 P -> seq_cp(A->B) -> A/B 并发 batch decode 独立续写 vs 不共享基线逐 token 比;
  (2) 填满 cache 触发 cells 淘汰/重分配: refcount>0 共享前缀不被误删 + 分支死亡回收;
  (3) hybrid: 共享 attention 前缀 + GDN 独立 + T24 replay 交互; (4) VRAM: 2/4 活分支预算
- 门: 1-3 全对 + 4 在预算内 -> S2a 开工; 任一失败 -> 记录数据, S1 作为终点

### 4. 待决 (不阻塞 A/B/D 编码)
1. A 的 pin 集: 只 pin end-4 (1 点/回合) vs {end-516, end-4} (2 点) -- 建议前者
2. B 的绝对下限默认值: 0 (纯 max-LCP) vs 512 token -- 实验扫
3. 网格 (S1.5): 粗网格 (M >= 32K, 每点 666MB) 是否要作为配置项 -- 建议 S1 先不引入

### 设计注记 (implementer, 2026-09-27): 检查点去重/密度 与 "瘦身" 提案 (待验证, 影响 S1.5/S2b)

**实测拆分 (两组独立数据交叉验证, 结论确定)**: 单个 context checkpoint =
**162 MiB 常数 (data_tgt, PARTIAL_ONLY = GDN recurrent) + ~4.1 KiB/token (data_dft = draft/MTP 的 KV)**
- checkpoint: n=1 -> 161.77; 127754 -> 663.73; 128266 -> 665.74; Δ512 = 2.01 MiB = 4.02 KiB/token
- prompt_save@154: total 169.75 MiB, draft = 0.605 MiB; 外推 draft@127754 = 501.5 MiB
  vs checkpoint@127754 - 162 = 501.96 MiB (0.1% 吻合)
- => 128K 每点 666MB 里 504MB 是 draft 的 attention KV; 而 attention KV 是前缀函数, 回滚只需裁后缀 (seq_rm),
  不需要快照。GDN recurrent 本体只是常数 162MB

**提案 S1.5 (待验证)**: 检查点只存 data_tgt (+ draft 的 partial, 若有), 不存 draft 全量 KV; restore 时对 ctx_dft
`seq_rm(n_past..end)` + 后缀本就要重放 (draft 与 target 同步重算) -> 每点 666MB -> ~162MB (4x)
- 收益: 128K 16 点网格 10.6G -> 2.6G; 单锚点 0.65G -> 0.16G; blob 内检查点负担同降
- 风险: 若 MTP 层含 GDN (partial 非空), 其常数状态仍需保留 (仍是常数级); 需量
  `llama_state_seq_get_size_ext(ctx_dft, PARTIAL_ONLY vs NONE)` + 正确性逐 token A/B
- 验证归入 S2a 冒烟 (item 3 hybrid 交互附带)

**去重前提 (回答"共享段检查点是否一致")**: 仅"规范化位置"上同一前缀 => 语义一致 (确定性内核; 批切分或有的
ULP 级浮点差可忽略), 可按 (树节点=前缀内容, 位置) 去重只存一份。非规范化触发 (prompt 尾锚点 end-516/end-4)
按"请求在哪结束"定位, 不同分支切分 -> 不同位置 -> 才是膨胀来源; 修法 = 把锚点挂到"它覆盖的前缀节点"上
(内容寻址), 而不是挂分支。min-step 门/淘汰是列表历史的函数, S1 的 pin 纪律与树管理的叶淘汰就是解决这个。

**分叉点检查点**: agent 场景自然分叉点 = user 消息边界 (下一轮思考剔除在 U_k 末尾分叉), 与规范化位置天然重合,
不需预测未来; 未知分叉 (客户端编辑/prompt 变体) 用"最近祖先 + 重放"兜底, 重放距离 = 是否需要网格的理由。

## S1 结果 (implementer, 2026-09-27): A/B/D 已实现, T32-1 与 T32-2 均实测验证

**实现** (工作区, 未提交; 补丁 `artifacts/t32-s1-worktree.patch` (32.7KB), 基于 `3eae5cdae`)
- A. pin 锚点: 聊天路径新增 `last_msg_is_user` (server-common.cpp -> task.params, 由消息数组判定, tool 消息不算);
  轮次首请求的 prompt_end-4 检查点置 `pinned`; 淘汰循环 (min-step 压缩 / 32 上限) 跳过 pinned; 只保留最新 pin;
  **同位置 supersede 继承 pin** (实测发现的最初版本会把 pin 冲掉, 已修)
- B. L2 选择: `load` 改 max-LCP (主序), 去掉 0.25 f_keep 门槛
- D. 轮换: `alloc` 保护"与来任务 LCP 最长"的条目不被 park 逐出; blob 只存 {最老 + pinned + 最新 2} 检查点 (size 计费同步)
- 诊断: 检查点创建打 pin 判定 (TRC), 加载打 lcp

**实测** (`t32_repro.ps1`: 场景 A = 32K 前置 + 36 步 agent 回合 (assistant/tool 真角色) + 下一轮助手内容替换 (思考剔除等价);
场景 B = 两会话 (各 ~5.5-5.9K) 轮换; 均 `-lv 4` + `LLAMA_SERVER_SLOTS_DEBUG=1`, MTP off)
- **T32-1 (A36)**: 基线 D 锚点 (n_tokens=32350) 被 "too close" 压缩删除 (日志 L1295), 最终恢复退到 31838 (D-516),
  prompt_n=3274 / 5.68s; 修复版锚点存活, 恢复 32350 (**D-4**), prompt_n=2762 / 5.10s (-512 token, -10%)
- **T32-2 (B, cache-ram 1100)**: 基线每次切换 5.4-5.7K token / 6.7-7.3s, 6x `making room ... removing oldest entry (1022-1029 MiB)`
  = park 逐出目标 + 恢复落回会话起点 (305); 修复版除前两次外每次切换 **71 token / 0.6s** (最终 4 token / 0.23s),
  7x `found better prompt lcp = 5.7-5.9K, f_keep = 1.0`, 恢复在 blob 末尾 end-4 -> **~12x TTFT**
- **规模效应**: 未裁剪 blob = state 424MB + ~10 检查点 x 149.6MB ~= 1.9GB (超限 -> skip/thrash); 裁剪后 ~1.02GB (可保存)
- **正确性**: A 22 个 + B 8 个请求在基线与修复版输出 (content + reasoning_content) **逐位一致**
- 检查点结构复核: 无 MTP = 149.6 MiB 常数 (GDN recurrent); MTP on = +4.1KiB/token (draft KV) -> S1.5 瘦身依据

**产物**: `artifacts/t32-s1-worktree.patch`, `t32_repro.ps1`, `t32_log.py`, 日志 `t32_{A36base,A36fix3,B1100base,B2fix}.err`,
jsonl `t32_{A20base2,A20fix2,B2base2,B2fix2}.jsonl` (含逐请求 choice 便于回归对比)

**下一步 (待用户)**: (1) 提交 S1 到 fork (需用户批准); (2) S1.5 检查点瘦身 (drop data_dft, restore 走 seq_rm) + 网格/密度评估;
(3) S2a 冒烟 (seq_cp 4 项); (4) B 的后缀复算: 目前 71 token/切换 = 4 rewind + ~67 增量, 已近最优


## S2a 冒烟实验结果 (implementer, 2026-09-27): 门通过; 附带发现 T24 ReplaySSM 回滚不确定性 (与共享无关)

工具: `tests/test-t32-smoke.cpp` (临时 harness, 私有 fork, 不进 PR; 目标 `test-t32-smoke`);
模型: Qwen3.5-2B (hybrid=1) 做正确性; Qwen3.8-27B 做预算。
方法: 共享前缀 P (~206 tok) + 4 分支各自续写 32 步 (teacher-forced; 每步记录 argmax + 全量 logits 对比);
`ref` = 4 条独立序列各自 prefill; `shared` = prefill 一次 + `seq_cp` x3 (同批 4 seq 一起 decode, 形状一致)。

- **item1 (共享前缀正确性)**: PASS, 全部 step logits 逐位一致 (max diff 0.000e+00)
- **item2a (删掉 seq0 对共享前缀的 cells)**: PASS, 逐位一致 (refcount 保护正确; 其它分支不受影响)
- **item2b (杀掉一个分支, 再从另一分支 seq_cp 重建)**: PASS, 逐位一致
- **item3 (深度回滚 + replay)**: 共享本身正确, 但**发现引擎侧问题**: `ref-vs-ref` (无共享, 同配置跑两遍) 也偶发差异
  -> T24 ReplaySSM 回滚路径存在运行间不确定性: 回滚后 logits 偶发差 ~0.17-0.25 (仅在 top-2 差 ~0.003 时翻转 token);
  复现率 ~1/6 (value-diff), token 翻转更罕见; **replay=0 时 6/6 逐位一致; rb<=2 时无损**;
  生产同款 (n_rs_seq=3, rb=3) 6 次无 token 翻转, 1/6 value-diff; `GGML_CUDA_GDN_REPLAY_CHECK=1` 未报错
  -> 建议: (a) 记为 T24 后续调查 (独立复现命令: `test-t32-smoke -m <model> --nrs 3 --rb 3`, replay=1);
  (b) S2a 正确性验证用 replay=0 或对"近并列"容忍; (c) 生产采样 temp0.6 下影响被随机采样掩盖
- **item4 (VRAM 预算)**: 27B, prefix 206 tok, n_ctx 8192: 1/2/4 分支 peak VRAM = 21747 / 21965 / 22407 MiB
  -> 每个共享前缀的活分支 ~= **+220 MiB** (189.8 MiB/分支 = GDN recurrent 状态不可共享 + 少量 cells; 与设计预期一致)
  -> 推论: 32GB 卡 - 权重 20.95GB ~= 10GB 可用于分支; 长分支的额外成本 = 各自唯一后缀 KV (共享前缀只算一份)

**结论**: 冒烟门通过 (1/2a/2b 逐位一致; item3 的分歧与共享无关, 属既有引擎路径; item4 预算合理) -> S2a 可开工。
**更正 + np=2 实验 (implementer, 同日)**: 上面"np N + unified KV 天然共享前缀 cells"的说法**不成立**。代码事实:
`llama_kv_cache::find_slot` 只复用"空 cell / 单序列 cell", 不按 token 去重; 跨序列共享必须显式 `seq_cp`
(服务器目前仅用于并行子任务: `server_slot::copy_state_to`, server-context.cpp:722/3829)。
实验 (`t32_np2.ps1`, np=2, 两会话共享 23K token 前缀, `id_slot` 固定): A slot0 prefill 23072 tok / 28.4s;
B slot1 **仍是全量 prefill 23076 tok / 28.5s** (无共享); 后续轮 A/B 各 33 tok / 0.6s (in-slot 增量);
VRAM 全程 25503 MiB 不变 (池预分配, 无法从 VRAM 看共享)。
结论: (a) **零代码已可用**: `-np 2` + 客户端 `id_slot` 固定 = 两个会话同时常驻, 交替轮换变成增量 (前提: 两会话总量
<= `-c` 池); (b) 共享前缀/去重要靠 S2a 的 `seq_cp` 分支 (省池子 + 省 B 的一次全量 prefill) -> S2a 第一步 =
"新请求与活分支 LCP 大 -> 直接 fork (seq_cp) 而非重算"。

产物: `artifacts/t32-smoke-test-t32-smoke.cpp`, `artifacts/t32_smoke_b{1,2,4}.err`


## S2a 第一步实施设计 (implementer, 2026-09-27): "活分支 fork" 原语 (先设计后落码)

目标 (最小闭环): 新请求到来时, 若某个**空闲 slot** 的序列与新 prompt 有很长的公共前缀 (LCP),
不再全量 prefill, 而是把该序列当作"活分支"进行一次 **fork**: 共享 [0,LCP) 的 attention cells + 复用其检查点回退
recurrent 状态 + 只解码后缀。收益: 省计算 (LCP 部分零解码) + 省显存 (共享 cells 只一份)。

机制草案 (尽量复用现有路径, 不新造):
```
fork(src_slot, dst_slot, LCP):
  1. mem.seq_rm(dst.id, -1, -1)                    // 清 dst
  2. mem.seq_cp(src.id, dst.id, 0, LCP)            // attention cells [0,LCP) 零拷贝共享 (refcount)
  3. dst.prompt = src.prompt.clone(); dst.prompt.tokens.keep_first(LCP)
     dst.prompt.checkpoints = src 的 checkpoints 中 <= LCP 的部分 (拷贝)
  4. 交给现有流程: 下一轮 batch 会自己算 n_past = LCP -> 检查点回退 (S1 机制) -> 解码后缀
     (即 fork 后 dst 的 prompt 状态 = "从 src 接过前缀", 其余全部沿用原逻辑)
```
触发条件 (v1): src idle; LCP >= 阈值 (如 512); dst 即将被覆盖 (它原来的状态可先走 prompt cache park, S1 已修);
不在 src 正在 decode 时执行 (队列单线程时序天然保证)。

待确认的代码点 (落码前必须核实, 防 COW 陷阱):
1. `llama_memory_recurrent::seq_cp` 在 p0=0,p1=LCP 时对 **recurrent tail cell** 的处理: 是否无条件共享 tail?
   若共享, 后续 `load_tgt(dst_id)` (检查点回退) 会写入共享 cell -> 需要先解除共享 (COW), 否则污染 src。
   需要读 `seq_rm`/`state_read` 的 COW 逻辑, 或在 fork 时改为"先 load 检查点再 seq_cp attention"的顺序。
   (注: `server_slot::copy_state_to` 只在"同长 prompt"的并行子任务用, 不涉及回退, 所以现成路径没暴露这个问题)
2. 检查点的 `load_tgt` 目标 seq 是否支持 `dst.id` (函数签名带 seq_id, 应可) + draft 侧 `seq_rm(dst.id, P, -1)` (S1.5 已实现同款)
3. `src.prompt.checkpoints` 的 pos 语义是否全局 (是: pos 是序列位置, 与 seq id 无关)
4. 与 spec/T24 的交互: fork 后 spec 状态 (data_spec, 20KB) 需要从 src 的检查点恢复 (S1 已接上) + smoke 已验证 cp+replay 共享正确

验收 (每步都要):
- 逐 token A/B: fork 路径 vs "独立 prefill"基线 (Qwen3.5-2B harness 已具备该对比模式, 扩一个 fork 场景即可)
- 复用现有 test-t32-smoke 的 item0/item1 对比框架; MTP 开/关各跑
- VRAM: fork 后 dst 不重复占 [0,LCP) cells (n_seq_max 允许时)
- 生产不动的回归: 现有 S1/S1.5 A/B (A36, B) 重跑不变

风险与回退:
- 最大风险 = 共享 cell 的写污染 (上面第 1 点) -> 先用 harness 证明 `seq_cp + load 检查点` 不污染 src (可直接加进 test-t32-smoke)
- n_seq_max 预算: 活分支数受 n_seq_max 限制 (提高有开销); v1 只在"现有 slot 之间"fork, 不新增 seq
- 若 COW 路径不可行: 退化为"只 fork 全前缀扩展 (LCP == src 长度)" 的简单情形 (无检查点回退, 直接共享当前状态)


## S2a 补充 (implementer, 2026-09-27): fork 原语已证明; 但发现"多序列布局数值不一致"未解问题

**1. fork 原语 (item5/6): PASS** — 两类都逐 token 匹配单序列基线, 且源分支不受污染:
- 全前缀扩展 (LCP == src 长度): `seq_rm(dst) + seq_cp(src,dst,0,LCP)` -> 继续解码 ✓
- 检查点回退 (LCP < src 长度): 再 `state_seq_set_data_ext(dst, ckpt, PARTIAL_ONLY)` ✓
  (源码已核实: `state_read_meta` 内部先 `seq_rm(dst,-1,-1)` 解除共享再分配私有 cell -> COW 安全)
- **前提: 必须 unified KV**。同 stream 下 `seq_cp` = `cells.seq_add` 零拷贝; 跨 stream 有 `is_full` 断言且是数据拷贝。
  注意: 我们生产与此前 T32 测试都是 `kv_unified = false` (日志确认); S2a 必须在 `--kv-unified` 下运行。

**2. 未解问题: 不同"布局路径"下 logits/token 不一致 (hybrid 与纯 attention 都复现)**
- GT = 单序列 (n_seq_max=1, batch=1); `n_seq_max=4-solo` (仅用 seq0) 与 GT **完全一致** -> 配置本身无害
- ref = 4 个独立 prefill + 交错解码; shared = 1 prefill + seq_cp x3 + 交错; shared/seq = 共享但逐 seq 单步; shared/priv = 共享 + 每分支私有 recurrent
- 结果 (first-diff, 32=全对):
  - Qwen3.5-2B (hybrid): ref=32/15/14/32, shared/il=17/32/14/32, shared/seq=32/32/14/32, shared/priv=17/32/14/32
  - Qwen2.5-Coder-3B (纯 attention): ref=32/32/32/**19**, shared 三条路径全 32/32/32/32 ✓
- 特征: 分歧只在少数位置 (14-19), 首次分歧步 logits 差 ~0.1-0.5, 之后 token 翻转级联; **共享不是唯一原因**
  (纯 attention 下共享全对, 独立 prefill 路径反而错一个分支); 也不是必现 (多数分支全对)
- 需要判定: (a) 引擎 bug (cell 布局/分配史影响 kernel 读取或归约顺序) 还是 (b) 可接受的"不同 kernel/归约顺序 -> 数值不可比"
- 影响: 树/共享分支的**验收标准** (不能简单沿用"与原路径逐位一致"); 生产是 np=1 单序列, 从未跑过多序列解码

**3. 建议 (待裁决)**
- a. 最小复现判定引擎 bug: 纯 attention, 4 seqs **相同位置区间**, 交错批量 vs solo (harness item1b 已具备; 可再缩小)
- b. 若属"数值不可比": 树验收改为 (i) 首次分歧前 logits 一致到某容差 (如 1e-4), 或 (ii) 采样输出分布等价 + PPL 控制位
- c. 工程路线不受影响的短期项: `-np 2 --kv-unified` + 客户端 `id_slot` 固定 (顺序请求) 已能覆盖"两会话轮换"场景;
  fork 落码保持**顺序解码** (避免交错批), 按 (b) 的容差验证


## S1.5 重新落地 (implementer, 2026-09-27 晚): 检查点瘦身 + blob 检查点裁剪 (不带 A/B/D, 按用户决定)

用户决定: A (pin, 长上下文只差 512 token 的回退) **不修**; S1.5 = 必须 (100K 序列 MTP 检查点吃爆 RAM: slot 上限 32 点 x 555MiB = 17.3 GiB)。
本次只重放 dropped 快照中的 S1.5 + alloc 裁剪 (共 43 行改动, 无 pin / max-LCP / 0.25 阈值 / park 保护), 补丁 rtifacts/t32-s1.5-worktree.patch。

改动:
1. create_checkpoint: draft (MTP) KV 不存快照; 仅当 PARTIAL_ONLY < NONE 才存 (纯 attention draft 的 KV 是前缀函数, 恢复时裁剪重建)
2. 
estore checkpoint: 先算 pos_next/n_past 再 load_tgt; data_dft.empty() -> seq_rm(ctx_dft, n_past, -1) (失败则整段清空 + WRN)
3. server_prompt_cache::alloc: blob 只存 {最老 + 最新 2} 检查点 (size 计费同步)

实测 (同 S1 口径; BinDir = src\llama.cpp-my\build\bin\Release; MTP3; replay=1):
- **A36 (cr600)**: 输出 38/38 与 	32_A36mBase/	32_A36mSlim **逐位一致**; 152 个检查点全部 = **161.769 MiB 常数** (基线 162.97-302.63)
- **B (cr1300)**: 输出 8/8 与 	32_BmBase/BmSlim/B2base2/B2fix2 逐位一致; 53 点全部 161.77; blob 逐出线 1180 -> 950 MiB
- B 轮换仍 5.4-5.7K token/7.7s (B/D 未带, 预期); S1 全量版实测 71 tok/0.6s -> 待下一步重放
- k2-final prompt_n = 3274 (A 未带, 会退到 D-516; S1 全量版 2762)
- **100K 外推**: 每点 555 -> 162 MiB; slot 32 点上限 17.3 GiB -> 5.1 GiB; 再加 --ctx-checkpoints 8 -> 1.3 GiB
- 验证日志: 	32_A36s15.err/jsonl, 	32_Bs15.err/jsonl (temp/v100; 待归档)


## Result (stage 0, 2026-09-27): H2D 基准 harness + 实测 (27B)

工具: `tests/test-t32-range.cpp` (`--mode h2d`; 私有 fork 工作区, 未提交; 目标 `test-t32-range` 用 `llama_build` 注册, 不进 ctest)。
模型/环境: Qwen3.8-27B-UD-Q6_K, `CUDA_VISIBLE_DEVICES=1` (V100), `-ngl 99 -fa on -np 2 -c 65536 -b 512 -ub 512 --mode h2d --n 32000`。

4 个数 (完整输出: `artifacts/t32-stage0-h2d.txt`):
- fill: 32000 tokens in 40.2 s (796.0 t/s)
- size: 2150.4 MiB (2254815072 bytes = 2.100 GiB, 70463 B/token)
- D2H: 0.81 s -> **2.60 GiB/s**
- H2D: 0.89 s -> **2.37 GiB/s**

100K 外推: 按实测尺寸线性外推 6.56 GiB -> H2D ~2.8 s / D2H ~2.5 s (往返 ~5.3 s); 按 brief 的 5.0 GiB 口径则 ~2.1 s / ~1.9 s。
spec 附录 H2D 行已替换为实测: 2.37 GiB/s H2D / 2.60 GiB/s D2H, 比 "~2s/GB 保守口径" 快 ~5x (新保守口径可用 ~0.5 s/GiB)。

偏差备注: 32K state 实测 2.10 GiB, 比 spec 估算 (~1.9GB) 大 ~16%, 100K 外推相应变大 (6.56 GiB vs ~5GB); 树存储预算建议按实测 70463 B/token 重算。


## Result (stage 1, 2026-09-27): 区间状态 API + 吞吐基准 + 回归 + 默认值锁定

工具/模型: `tests/test-t32-range.cpp` (`--mode range-bench`, 本阶段新增; `--mode h2d|correctness` 为 stage 0 与本阶段); Qwen3.8-27B-UD-Q6_K,
`CUDA_VISIBLE_DEVICES=1`, `-ngl 99 -fa on -ctv q8_0 -np 2 -c 65536 -b 512 -ub 512 --mode range-bench --n 32000` (完整输出: `artifacts/t32-stage1-bench.txt`)。
每次自带 32K prefill (40.8-40.9s); payload 1532.0 MiB = 50200 B/token (range API 仅 attention, 不含 recurrent; 生产量化 q8_0 V)。

| chunk | chunks | write s | write MiB/s | write ms/chunk | read s | read MiB/s | read ms/chunk |
|-------|--------|---------|-------------|----------------|--------|------------|---------------|
| 512   | 63     | 1.28    | 1198.1      | 20.30          | 0.81   | 1896.4     | 12.82         |
| 1024  | 32     | 1.11    | 1377.8      | 34.75          | 0.74   | 2062.5     | 23.21         |
| 2048  | 16     | 0.96    | 1594.1      | 60.06          | 0.66   | 2308.8     | 41.47         |

判读: ms/chunk 每翻倍 x1.71-1.81 (非持平) -> 固定开销不主导; range 读达全量 H2D (2.37 GiB/s) 的 78%-95%,
range 写仅达全量 D2H (2.60 GiB/s) 的 45%-60% (写计时含每 chunk 新 blob 分配, 口径保守)。
决定: `--tree-chunk` 维持默认 **512** (复用粒度最好; 上调 2048 只省 100K 序列约 1.4s 搬运); 不强制批量装载路径。
修正 stage 0 备注: 树存储预算按 **50200 B/token** (q8_0 target range payload), 不是 70463 B/token (f16 全量含 recurrent)。

回归 (Qwen3.5-2B-UD-Q4_K_XL, V100):
- `test-state-restore-fragmented`: `SUCCESS - state restore works with fragmented KV cache`, 退出码 0
- `test-save-load-state`: `All tests passed.`, 无 SKIP, 退出码 0 (Test 9 故意损坏 state 的负路径 E 日志属预期)

spec 对应改动 (`artifacts/t32-tree-storage-design.md`):
- 1.2 / 3.4: `--tree-chunk` "阶段 0 后可能上调 / 阶段 1 出口定" -> "阶段 1 实测锁定, 默认 512"
- 附录: target ~52KB/token -> 实测 50200 B/token (q8_0), 块 512 -> 24.5 MiB (25.7 MB);
  100K 外推 6.56 GiB (f16 全量) -> 4.68 GiB (q8_0 target) -> H2D ~2.0s / D2H ~1.8s; 加 range API ms/chunk 行
- `--tree-anchor-step` 维持 32K: H2D 实测 0.42 s/GiB, 远快于 "2s/GB" 保守口径, 无需按成本表重估

补丁: `artifacts/t32-stage1-worktree.patch` (`git diff ba41cccec..HEAD`, 阶段 1 全分支)。

正确性回归归档 (最终复审修复后重跑, 2026-09-27): `artifacts/t32-stage1-correctness.txt` -
计划 Task 2 Step 2 的三条命令原样重跑, 2B hybrid / 3B 纯 attention / 2B `-kvu` 全 PASS, 退出码 0;
含新增的截断 append 清理 (`state_clear_append`) 与非法参数拒绝检查。

## Result (stage 1 验证矩阵, 2026-09-27; 小模型, CUDA device 0)

命令: `test-t32-range.exe --mode correctness`; 2B (Qwen3.5-2B Q4_K_XL) 与 3B (Qwen2.5-Coder-3B IQ4_XS)
矩阵: np=1/3 x {f16, V-q8_0, V-q4_0, K+V-q8_0} x kvu 开/关 (18 run, 全部 exit 0)
归档: `artifacts/t32-stage1-correctness-matrix.txt` (每 run 完整 stderr + logits 距离)

结论:
- **np=1 (生产口径)**: 全量化组合 18/18 PASS, **logits 逐位相同 (max|diff| = 0.000000)**; 含"不恢复基线 vs
  分段恢复 vs 全量恢复"直接对比 (harness 新增 np=1 自比对路径).
- **np=3 非 unified**: 全量化组合 PASS, **logits 0.000000** (每 seq 独立 stream -> cell 布局一致).
- **np=3 unified**: 数据级检查全 PASS; 跨序列"逐 token 相等"判据改为显式 SKIP (cell 布局不同, 见下).
- unified 下量化 V 的 logits 距离: 2B q8_0 0.25 / 3B q4_0 0.42 / 3B q8_0 0.58-0.63; **连旧的全量恢复
  路径与基线也差 0.60** (只是碰巧没翻 token) -> 偏差来自 unified 共享 stream 下不同 cell 布局的
  FA 归约顺序, **不是 range API**; 数据级检查 (逐字节 payload 相等 / 恢复成功 / 重叠拒绝 / 截断清理)
  在所有配置全 PASS. 引擎自身测试惯例 (`test-save-load-state`) 对跨路径比较用 NMSE 容差而非逐位.
- 对树 (阶段 3, R1 非 unified) 无影响; 生产 np=1 逐位对齐.
- harness 变更: np=1 自比对路径 (基线 vs 分段 vs 全量), logits max|diff| 打印, unified 下布局敏感判据
  显式 SKIP; 提交 `e19dfca26` (待填) 之后所有矩阵 run exit 0.

## Result (stage 2)

- 树模块 `tools/server/server-kv-tree.{h,cpp}` + harness `tests/test-t32-tree.cpp` 完成 (分支 `t32-stage2`)
- 模式: logic (fake IO 单测: 哈希链/去重/稀疏化/淘汰/拒绝) / model (2B 真机: tip/fork/自愈/sparsify/ssd) / accept (mini A/B)
- 结论: park/restore 逐 token 与基线一致 (np=1); 分叉按最深可用锚点恢复, 自愈后第二次免费;
  SSD 单份权威往返逐位一致; 淘汰次序 (锚点 -> 叶块 -> 叶序列 -> 拒绝) 与 pin 纪律生效, 拒绝可见
- 归档: artifacts/t32-stage2-model.txt, artifacts/t32-stage2-accept.txt
- 未接 server (`--kv-tree` 属阶段 3); 生产未动

## Result (阶段 4, 2026-09-27): soak 长跑 + D12/D13 + 全回归

- 分支 `t32-stage4` (base master `daf4186d3`, head `c0619da14`, **未合并/未 push**); 决定 D14-D23 见 `artifacts/t32-tree-plan-stage4.md`
- 交付: D12 恢复后按树锚点重建 `prompt.checkpoints` (recurrent 上下文 tail 语义 `pos_min = pos_max = pos - 1`, PART 保持 `pos_min = 0`, D18) + 聚合日志 `kv tree stats:` (D20); D13 构造时清理 `blocks/anchors` 并报 `cleared N stale files` (D17); D22 `disk_errors` 单计数; D23 锚点间距链作用域; harness `soak` 模式 (np=2 churn + 逐位抽检 + RSS/句柄/IO + erase + kill -9 重启)
- 证据: `artifacts/t32-stage4-soak-30.txt` (30 min: rounds=1703 rebuilt=336 cmp 340/340, `RESULT soak: 0 failure(s)`; 重启 `files_before=79 cleared_lines=1 files_after=0`), `t32-stage4-soak-smoke.txt` (5 min, 0 failure), `t32-stage4-{logic,model,accept,ab,overlap,b,b3,neg,heal,ref}.txt` (全 0 FAIL / 0 failure), srv 原始日志 `t32-stage4-logs/`
- D14 (SSD 磨损优化) 延后; D15 60 min soak 用户裁决跳过; D11 不做 (D16); tidy/merge 未做, 分支保留



## Result (阶段 5, 2026-09-27): 分叉锚点 (D25-D30) + fork 验收 + 全回归 + 短 soak

- 分支 `t32-stage5` (base `42ee7a6f7`, head `04954b468`, **未合并/未 push**); D25 双档间距 (猜测 `--tree-checkpoint-anchor-step` 32768 / 分叉 `--tree-checkpoint-fork-step` 8192), D26 heal-on-miss, D27 删除 `promote_prune`, D28 `anchors_skipped_step`/`step_skips=`, D29 选项改名 (决定见 `artifacts/t32-tree-plan-stage5.md`, 设计记录见 `t32-tree-storage-design.md` §9)
- 新 `fork` 验收模式 (脚本 `t32-stage3-ab.ps1`): `FORK METRICS captured=[8192,16384] restored=[8192,8192,16384]` -- miss-heal 捕获 8192 (req2), 第二分叉重放捕获 16384 (req4), 两次复用 (req3/req5 恢复 8192/16384); 5/5 请求 tree vs full prefill 逐位一致
- 全回归 (cuda1): logic 74/0, model 64/0, accept 42/0; ab/overlap/b/b3/neg/heal/ref/fork 全 `RESULT 0 failure(s)`; 5 min soak 302 轮 0 failure (rebuilt=143, cmp 60/60, 重启清理 61->0)
- 场景修正 (fork 模式, 阈值未动): ctx 16384 -> 32768 (b1=20501 token 超限); `--slot-prompt-similarity 0` (默认 0.1 的 stock LCP 原地复用使树不参与, f_keep>=0.5)
- 归档 `artifacts/t32-stage5-*.txt` + `t32-stage5-logs/`; 未合并/未 push, 分支保留 (D30)

- **媒体原生进树 (2026-09-28 夜间, 分支 t32-media, 未合并/未部署)**: kv 树现支持含媒体 (图像/音频/视频) 的 prompt 的 park/match/restore. 提交: `5722fac1e`+`2e341d9e9` (树存储侧: `kv_tree_media` 跨度 + token/位置映射 + 媒体块对齐 + 身份入块哈希 + park/match), `70157be75` (检索侧测试), `7d5b11bb1`+`587272960` (服务器接线: park/restore/erase/heal 四处去掉 has_mtmd 门 + `get_tokens_raw()` + `keep_first` 相邻媒体块边界修复). 设计文档 `artifacts/t32-media-tree-spec.md`, 计划 `artifacts/t32-media-tree-plan.md`, 证据 `artifacts/t32-media-e2e-20260928.txt`.
- **E2E (0.8B-MTP + mmproj-F16, cuda0, `--image-min-tokens 1024`, 贪心)**: 同一含图会话第二次请求 `parked 1094 -> restored 1059` (恢复点越过 1024-token 图像块), prompt_n 1063->25; 换一张图同位置 -> restore miss + 全量 prefill (无错误复用); 两图相邻 (A 后 B) -> parked 2157 / restored 2090, prompt_n=22; 5 个请求输出与无树 baseline **逐字节一致**; 日志无 abort/assert; 单 server 显存 ~1.9GB (device 0, 8188MiB), 跑完已全部停止.
- **关键机制**: 媒体块身份 = `mtmd_input_chunk_get_id()` (原始字节 sha256) 折叠进块哈希; token 索引与位置分离 (M-RoPE: 图像 N token 仅占 n_pos=max(nx,ny) 个位置, 全块共享起始位置); 块边界对齐媒体块 (不切块); 恢复点永远落在块边界; 树不存媒体字节, 恢复时用请求自带 chunk 重建槽 prompt.
- **未做/限制**: 媒体 + stock checkpoints / `n_cache_reuse` / spec 状态仍按上游门控; 媒体块原子 (超大视频 = 超大块); 尚未合并 master、未部署生产 (生产仍为撤回前的 mmproj 绕过构建).

- **媒体树白天轮补测 + 崩溃修复 (2026-09-28, cuda1)**: 分支 t32-media 新增 `d9a62d8ef` (无回滚能力的模型上树恢复留 1 token) + `1bfb1b38a` (leave_one 上限不得超过已验证前缀, 审查发现并修复). 证据 `artifacts/t32-media-day2-20260928.txt`.
- **发现并修复的真实缺陷**: hybrid 模型无回滚快照 (0.8B-MTP 不开 spec => `n_rs_seq=0`, seq_rm_type=FULL) 时, 树恢复到整段 prompt (C == task.n_tokens) -> 服务器按 TAG_PROMPT_LOGITS 减 1 再 `seq_rm` -> recurrent 无法回滚 set_partial 恢复的状态 -> `failed to remove sequence ...` abort (纯文本也复现). 修法: 目标或 draft 上下文无 PART/RS 能力时 `leave_one` (恢复点上限 min(deep, n_tokens-1)); 树恢复点恰在 n_past 时跳过 checkpoint 搜索; 媒体 prompt 也重建 checkpoints (单位改为 tok/pos 分离), 使有回滚能力的模型 (27B+MTP) 保持深复用. 修复波经审查: 1 Critical + 1 Important + 1 Minor 全部修复并复审通过 (mutation 验证测试有效).
- **白天轮矩阵结果**: ① 不设 `--image-min-tokens`: 小图块也能 deep restore (prompt_n 25), 输出与自带 baseline 逐字节一致; ② 强制落盘 (`--tree-ram 16`): `ram=0 B, disk=53.9MB`, 从 SSD 恢复 1059 tokens, 输出一致; ③ np=2 跨 slot: A 链被 C 挤掉后回来 `restored 1053`, prompt_n=4; ④ 媒体 soak (np=2, 强制落盘) 90 请求 alive, 恢复路径确定 (vs 第2轮 diff=0), aborts=0, disk_err=0; ⑤ 27B 生产配置 (Q6_K+q8_0 V+mmproj+MTP) 第3/4轮 `prompt_n=1 cached=1125 ~145ms`; ⑥ 纯文本无 spec 重复请求 prompt_n=4 (与 stock 一致, 不崩); ⑦ harness logic 130/0, 2B model 65/0.
- **已记录现象**: 全量 prefill vs 分片缓存中恢复 (np=2) 的浮点归约顺序差异 -> 0.8B q4_k 上 logit 差 ~0.15-0.19 nats 可翻转贪心近并列 (措辞变化, 语义不变; 每条路径自身确定). np=1 与 27B 未观察到输出差异.
