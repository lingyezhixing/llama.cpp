# T24 ReplaySSM：实现记录与 MTP 问题的最终结论（2026-09-26）

本目录是 2026-09-23 封存的续篇。封存当时的结论是"二选一，本次放弃"；
这一篇补上了其中一条轴（状态/回滚轴）的**精确实现**，并把整个 MTP 问题定性收口。

- 实现提交：`3eae5cdae`（本地，未 push），基线 `d24474edd`（T19-L 交付）
- 完整 diff：`t24-replayssm-final-20260926.patch`
- 原始数据：`数据/`（1000 token 贪心 token 流、64 token 三臂、自检运行）

---

## 一、T24 是什么

老实现（上游 + 本 fork）做投机回滚的方式是**存状态快照**：
S 缓存每层保留 `n_rs_seq+1 = 4` 个平面，op 每 token 写快照，回滚时按 `rs_idx` 选平面。

ReplaySSM 换成**存输入 + 重算**：
S 缓存只留 **1 个 committed 平面**；每 token 的 raw inputs（k/v/g/beta，conv 之后的值）
存进每层的记录张量；回滚 = 从 committed 平面**重放** accepted 前缀（fold），
用的是与 verify 主循环**完全相同**的递推代码（同一个 `gdn_state_update` inline）。

收益：MTP3 场景 VRAM **−420 MiB**（21598 vs 22018）；PPL 不变（4.3567）；
代价：tg ≈ **−1.9%**（fold 的重放计算）。

开关（默认关，不开 = 老行为）：

| 环境变量 | 作用 |
|---|---|
| `GGML_CUDA_GDN_REPLAY=1` | 启用 replay 路径 |
| `GGML_CUDA_GDN_REPLAY_CHECK=1` | 额外启用"fold vs 上批提交态"逐位自检（多占 ~96MB，默认关） |

## 二、与老实现的差异（共 4 处，不止存储形式）

1. **存的东西**：4 个状态平面 → 1 个状态平面 + 每 token raw inputs（约 265KB/层）。
2. **发布语义**：记录批提交的是"本批 token **之前**"的状态（pinned），由下一批的 fold
   推进到 accepted 前缀；不可记录的大批（prefill，nt > T_cap）直接提交 after-batch 状态。
3. **断链批判据**：pos 回退/重处理批必须不 fold（p=0），走 fresh 语义。
   老实现靠 cell 链天然得到；replay 必须显式判断。
4. **conv 缓存仍走老的快照 + 回滚索引**，没有改成 replay —— 严格说是混合方案。

## 三、修掉的两个 bug（都对应一个确定性分叉）

### bug 1：断链批不该 fold
真实 prefill（pos 0）会重新处理 warmup 已算过的 pos 0..1。这种批在内存里的状态链是断的，
正确语义是**从清零状态整批重算**（老实现/base 的行为）。而 `get_rec_p` 仍按
`rec_n - rollback` 给出 p=2，fold 在**被清零的**状态上又重放 2 条记录 ⇒ 多算 2 个 token
⇒ 状态小扰动 ⇒ 在 index 44 的近并列处采样翻成另一个 token。

修复：`get_rec_p(i, pos0)` 增加连续性判据，只有 `pos0 == rec_pos0[seq] + p`
（真正接续上一个 accepted 前缀）才 fold，否则 p=0；`rec_commit` 记录每批首 token 位置。

### bug 2：`s_copy` 覆盖破坏了 conv 的回滚
conv 缓存和状态 gather 共用 `s_copy`。为了状态 gather 曾把 replay 下返回值强制为 0，
结果 conv 抓取也变成"永远取最新行" ⇒ conv 历史把**被拒绝草稿的原始输入**算了进去
⇒ q/k/v 末位漂移 ⇒ 同样是采样翻转。

修复：`s_copy` 还原为上游公式；conv 索引**提前**抓取（赶在状态 gather 消费/重置 `rs_idx`
之前），状态 gather 因内部 reset 自然拿到 pinned 行。

## 四、验证结果（全绿）

| 项目 | 结果 |
|---|---|
| d0 长跑（1000 贪心 token，短 prompt，ctx4096/q8_0） | **老 MTP vs replay MTP：逐 token 完全相同** |
| 同上，base（不开 MTP） vs 两者 | 都在 token 38 分叉（已知选核轴，与 T24 无关，见 §六） |
| MTP3 64 token（常识 prompt） | ON = OFF = **64/64 与 base 一致** |
| n-max=1：ON vs OFF | 逐 token 相同（整理后复跑通过） |
| 不开 MTP：ON vs OFF | 逐 token 相同（整理后复跑通过） |
| n-max=2：ON vs OFF | 逐 token 相同（64 token，draft 44 accepted 41） |
| 同 slot 连续请求（0 裁剪续写 / 2 token 部分回滚 / 换 prompt 整段重算） | ON = OFF 全部 6 段逐 token 一致 |
| 多次 save/restore（恢复→续写→再存档→再恢复→续写） | 各段与不中断运行逐 token 一致 |
| 恢复后部分裁剪（r=1/2/3，与 fresh prefill 对照） | 一致 |
| 长生成到上下文上限（ctx4096, 4500 请求 → 4082 token） | ON = OFF 逐 token 一致，均 truncated |
| 多序列 `-np 2`（交替续写 3 轮） | ON = OFF 逐 token 一致；自检 0 mismatch |
| 多序列 `-np 3` / `-np 4`（交错、含不同 slot 组合） | ON = OFF 逐 token 一致；`-np 4` 两轮 + 自检 0 mismatch |
| 并发（两 slot 同批，各自 32 token） | 两次 OFF 自身可复现；ON == OFF 逐 token，draft/accepted 统计也相同 |
| `-np 4` slot2 save/restore 后继续 | 与不中断运行逐 token 一致 |
| **长程** `-np 1` 1000 token / `-np 2` 4x250=1000 token/slot / `-np 4` 4x125=500 token/slot | ON = OFF 逐 token；长程自检 0 mismatch |
| 并发长程（两 slot 同批，各 500 token）+ 自检 | 自检 0 mismatch；token 层不可跨配置比对（见下注） |
| VRAM `-np 4` | OFF 19348 MiB → ON **17978 MiB（省 ~1.37 GiB）** |
| **DFlash**（Qwen3.6-35B-A3B，`qwen35moe` + `draft-dflash` Q8，`--spec-draft-n-max 6`，np=1） | greedy 900 token（3 轮续写）/ 生产采样(temp0.6/top-p0.95/top-k20) 256 token：ON = OFF 逐 token，draft/accepted 统计完全相同；自检 0 mismatch |
| DFlash VRAM / 吞吐 | OFF 25132 → ON **24782 MiB（省 350 MiB）**；吞吐 3 组交错测量在 ±5% 噪声内（均值几乎相同），**无速度结论** |
| **DFlash + np>1**（`-np 2` / `-np 4`，交替续写） | np=2: 2 slot x 200 x 2 轮 = 800 token ON = OFF + draft/accepted 统计相同；np=4: 4 slot x 128 x 2 轮 = 1024 token 同样一致 |
| DFlash + np>1 + 自检 / 生产采样 / 显存 | np=2 顺序与**并发混合批**自检均 0 mismatch；np=2 生产采样(temp0.6) 800 token 逐 token 一致；VRAM np=2 OFF 25657 → ON **24957 MiB（省 ~700 MiB）** |
| PPL（512/8chunk/seed 42） | **ON = OFF = 4.3567**（= T19-L 基线） |
| 自检模式（全层全批 fold vs 上批提交态） | **0 mismatch** |
| VRAM @MTP3 | **−420 MiB** |
| tg @MTP3（交错 4 次平均） | 71.6 → 70.2 t/s（−1.9%） |
| pp512 / tg128（非 MTP 路径） | 无回退（整理前实测） |
| slot save/restore 后继续（MTP3，16/40/60 token） | 恢复后输出与不中断运行**逐 token 一致** |

复现口径：

- d0：`llama-server -m <model> -np 1 -ngl 99 -fa on --seed 42 -c 4096 -ctv q8_0`，
  请求 `prompt="Write a long detailed essay about the history of computing. Be verbose."`,
  `n_predict=1000, cache_prompt=false, temperature=0, top_k=1, seed=42, return_tokens=true`；
  三臂 = 不开 spec / `--spec-type draft-mtp --spec-draft-n-max 3`（env 关）/ 同前 + `GGML_CUDA_GDN_REPLAY=1`。
- 自检：在 replay 臂上加 `GGML_CUDA_GDN_REPLAY_CHECK=1`，日志里出现
  `fold did not reproduce the committed state` 即为失败。

## 五、调试教训（为什么花了两天）

真正的 bug 只有两个，但定位过程被"不可靠仪器"主导：

| 陷阱 | 后果 |
|---|---|
| CUDA 设备端 printf 会重复/乱序 | 出现"同一 lid 打印两次""哨兵被覆盖"等不可能现象，把结论带向"记录在两次 launch 间被清零" |
| host 侧 `ggml_backend_tensor_get` 相对异步计算滞后 | dump 显示"上批写的东西本批读不到"，其实是读数时机；150ms 重读也不变，进一步误导 |
| 双 context（target + draft）+ 预留图（只 build 不执行，nt=1/512） | 日志互相交织，一度把草稿 context 当嫌疑人 |
| `set_fold_input` 每批被调 96 次（48 层 ×2 input） | half 翻转偶数次、读写半区错开；需按 (ubatch 指针, pos0) 幂等 |
| `n_tokens` 是 int64 但 printf 用 `%d` | 后续参数整体错位 4 字节，误判"p 恒为 0" |

破局靠三件事：**① host 侧可信**（kernel 把"它看到什么"回写进 fold 张量，host 读回，
不再依赖设备 printf）；**② 用 git diff 对上游审自己**（暴露 `s_copy` 覆盖）；
**③ 位置连续性判据**（对比批的 pos0 序列与 accepted 前缀）。

## 六、MTP"降智"问题的最终结论

> 本节是摘要；完整推理与阐述见 `../06-MTP降智问题的完整阐述.md`。
> 下文的"三层误差源"对应完整版的 §2，"翻转机制"对应 §3，
> "不变量"对应 §5，"三条路线"对应 §7。

### 6.1 三层误差源，量级与性质都不同

| 层 | 性质 | 量级 | 我们怎么处理 |
|---|---|---|---|
| 权重/激活/KV **量化** | **系统性偏置**（改变的是模型函数本身） | ~1e-3 及以上 | PPL/benchmark 门禁接受 |
| 训练数值 vs 推理数值、跨引擎/版本/显卡 | 零均值底噪（结合律） | ~1e-6 | 无法定义绝对基准，只能文档化 |
| **MTP verify 批 vs decode 批** | 零均值、同量级、**同环境内可对比** | ~1e-6 | 分两条轴：状态轴（T24，已完成）/ 选核轴（T20，封存） |

关键：MTP 并没有"额外引入一类误差"，它只是**制造了同环境 on/off 对照**，
把这个天然存在的误差类暴露了出来。

### 6.2 翻转机制与"降智"观感的来源

1e-6 的末位差沿网络被 Jacobian 放大（可达 1e-4~1e-2），在 top1/top2 的 gap
小于**当前累积扰动**处翻转（封存实测 d0@207 的 gap 是 0.068 nats，就是这么翻的）；
翻转后自回归放大成完全不同的轨迹，但**两条轨迹都是合法采样**。

因此：

- 期望上无偏（变好/变差概率相同）——大样本聚合指标不退化；
- 但**方差在难任务上被放大**：高度训练的内容 margin 大、近并列少，轨迹稳定；
  小众/难任务本来在窄脊上，任何扰动都翻，长链把一次错翻放大成整段崩；
- 社群说的"降智"本质是**方差在难任务上炸开**，不是均值下移。
  单次对比看不出系统性降智；要证明降智必须用多 prompt/多样本的聚合指标。

### 6.3 没有绝对基准，"正确"只能定义为不变量

浮点不满足结合律，IEEE 只规定单次运算舍入、不规定归约顺序；
上游 nomtp 也只是一个**约定**（换卡/换版本/换 batch size 它自己也漂），
训练时的前向又是另一套算术。能承诺的只有：

1. 同配置确定性；
2. 同构建内 batch-invariance（开 MTP 完全透明）；
3. 分布等价（PPL/benchmark，量化偏置也在这里被门禁）；
4. 跨构建/跨硬件：文档化，不承诺。

### 6.4 两条修复轴

- **状态/回滚轴**（T24，本目录）：已做到"老 == 新逐位"，自检 0 mismatch。
  这是**无损重构**，不改变行为，省 420MB。
- **选核轴**（T20，见 `../04-T20源码排查/`）：让 verify 和 decode 走同一套算术。
  封存里已实现并验收（n-max 1/2/3 与不开 spec 逐 token 一致，代价 0.5~0.8%），
  但它的做法是**让 decode 搬家去和 verify/prefill 对齐**（S2），并把 pb 钉常量（S3'），
  所以 **no-MTP 自身的数值也会变**（封存实测 t20fix vs stock 在 d8192@147 分叉）。
  即：它不是"恢复旧值"，而是"两条路一起换到同一套新算术"。

### 6.5 三条路线与取舍

| 路线 | MTP 透明 | no-MTP 数值 | 代价 | 状态 |
|---|---|---|---|---|
| A 现状（仅 T24） | 否（近并列仍翻） | 不变 | 0 | 已交付（开关默认关） |
| B 照搬 T20 修复 | **是**（验收过） | **变**（≠ 现在生产、≠ 上游） | −0.5~0.8% | 补丁在封存，未移植 |
| C decode-preserving 对齐 | 可能（S3 有边界残差风险） | **不变** | 未知 | 未做；S1 免费、S2 小改可行，S3 需新机制 |

## 七、未覆盖与后续

1. ~~整理后待复跑~~ **已复跑（2026-09-26）**：n-max=1 / 不开 MTP 逐 token 一致；VRAM @MTP3 −420 MiB（22019 → 21599）。
2. ~~会话状态存取~~ **已修（2026-09-26，提交 `3eae5cdae`）**：
   原问题 = replay 下 S 缓存只有 1 个 pinned 平面，记录张量与 pending accepted 数未序列化，
   prompt cache / slot 保存恢复后会从"少一个 accepted 前缀"的状态继续（静默错误）。
   修复 = 状态文件追加 replay 块（pinned 平面走普通 cell 行 + 记录张量 +
   `rec_gen`/`half`/`gen_of`/`n`/`pos0`/pending）；恢复后首个 batch 用 pending 计数 fold 一次；
   `LLAMA_STATE_SEQ_VERSION` 3→4（旧状态文件显式报错）。
   验证：slot save/restore（MTP3，n_predict=16/40/60）恢复后输出与不中断运行逐 token 一致；
   跨模式（records 存档 → records 关闭读取）会显式报错而非静默出错。
   注：40 token 那个用例在修复前是**可复现失败**的（TREF/TB 第 8 个 token 分叉）。
3. ~~多序列~~ **已支持（2026-09-26，提交 `3eae5cdae`；此前 `3eae5cdae` 曾临时加门禁）**：
   原问题 = 记录按 cell（序列）分 bank，但 fold 记账（`rec_gen`/`rec_half`/`rec_written_half`）是
   **全 ubatch 共享**的 → 序列交错时某序列的记录永远等不到 fold（p=0），其 committed 平面永久落后
   （可复现证据：`-np 2` 交替续写，ON vs OFF 在 slot1 第 2 轮分叉）。
   改为 **per-sequence 记账**：记录 bank = 序列 id（不再等于 cell，cell 移动也不影响）；每序列独立
   的 gen/have_rec、读半区、写半区；**每序列自己交替半区**（读/写永不撞同一半区）；保存/恢复按
   bank;`LLAMA_STATE_SEQ_VERSION` 4→5。fold 块布局改为 `[_, p_i, 读半区_i, 写半区_i, bank_i, 自检, skip]`
   （4*n_seq_max+3 槽）。
   **踩坑记录**：改布局时 kernel 侧 `n_fold = (fold->ne[0]-3)/4` 的除数一度仍是 3 → `np≥3` 时
   n_fold 偏大 → 自检计数器越界写 → 静默错（np=2 恰好整除而"通过"）。教训：布局常量必须单点定义/或
   在 host 与 kernel 双侧断言一致。
   注：**并发（混合批）的 token 级比对不能跨运行**——批的合并点取决于到达时序，OFF 自身换个
   stagger（30ms→200ms）在 token 314 就分叉、同 stagger 两次也不同。并发场景的可靠判据是
   **自检 0 mismatch**（与批合并无关）＋短程恰好同批时序时的逐 token 一致。
4. 回滚边界与 pending 组合（同 `3eae5cdae`）：
   - `seq_rm` 的部分回滚在 replay 下**以记录覆盖为界**（最后一批 recording 批的 `rec_n`，
     或恢复状态的 accepted 前缀 `pending`，且 ≤ `n_rs_seq`），否则返回 false；
   - 恢复后的部分回滚按 `p = pending - rollback` 组合（修复前 pending 分支忽略 rs_idx →
     连续性判据把 p 清零 → 状态落后 pending-r 个 token，静默错）。
   注：`common_context_seq_rm` 对 seq_rm=false 是 **GGML_ABORT**（上游既有行为，回滚 > `n_rs_seq`
   时同样会 abort）；实测 server 的 SWA/checkpoint 守卫会让"部分裁剪"要么 ≤ 3 token、要么整段
   重算，故该组合路径在 server 上未被触发（属内存 API 层语义修复）。
5. 自检（`GGML_CUDA_GDN_REPLAY_CHECK=1`）恢复后的**假阳性已修**（同 `3eae5cdae`）：
   恢复时 diag 张量在本进程尚未写入（全 0），首个 fold 与全 0 比对 → 报
   `fold did not reproduce`（实测 37,748,352 = 全部元素）。修复 = 携带恢复状态的第一个 batch
   跳过自检（fold 张量新增 skip 位），该 batch 仍正常写 diag，之后的检查恢复有效。
   验证：slots3 + CHECK=1 从 3 条 mismatch 降到 0 条。
6. 未覆盖矩阵：KDA 模型（无权重）；EAGLE3 / DSpark（机制与 DFlash 完全相同，`need_n_rs_seq` 同一分支，但无权重可测）。
   DFlash 已实测（2026-09-26）：正确性全过 + 省 350 MiB + 吞吐无差异（噪声内）；
   DFlash + np=2/4（含并发混合批 + 自检 + 生产采样）也全过，np=2 省 ~700 MiB。
7. 路线 C 的可行性验证（decode 中性对齐 + S3 残差量化）。
