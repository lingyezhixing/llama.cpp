
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
