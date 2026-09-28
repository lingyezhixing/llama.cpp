# T24: ReplaySSM - GDN 投机状态 raw-input 重放 (用户批准 2026-09-23)

状态: **DEBUGGING (2026-09-24 重启: 用户决定继续给时间, 收益不小)**; 工作区已回灌归档 patch, 回到排查终点; 根因已修复 + 链路已验证; 唯一阻塞 = "4 个 recording batch 后崩溃" 未定位; 集成门槛不变 (解决 -> 集成); 详见文末 "重启与崩溃排查计划"
动机: 当前投机回滚用"每位置全量 state 快照" (n_rs_seq=K, 每份 144MB); ReplaySSM 改为只存
1 份 committed state + 每 token 原始输入记录 (k/v/g/beta, ~1.71MiB/token), accept 后按与 verify
**同一有限精度路径**重放 accepted 前缀。
收益: MTP3 省 **~430MB** 显存 + 去掉每轮多份 state 写; 与窗口宽度正交。

## 参考 (仅参考, 独立实现; 不盲信)
- 本地: `D:\LLM\Backend\src\ninfer-v100-geoffwatts\docs\maintainer\replayssm-gdn.md`
  (§4 失败模式: fold 必须与 verify 逐 bit 同路径; §6.2 记录格式 1.71MiB/token)
- sglang 参考: `sglang/python/sglang/kernels/ops/attention/fla/gdn_replayssm_spec_fold.py` (建议先抓单文件细读)
- 现状代码: `src/llama-memory-recurrent.cpp` (n_rs_seq/rollback), `ggml/src/ggml-cuda/gated_delta_net.cu`,
  `ggml/src/ggml-cuda/ggml-cuda.cu` (snapshot cpy 融合 ~3453-3462)

## 目标与验收
1. **逐位等价**: replay 恢复的 state == 现快照路径 (state hash / memcmp; 覆盖 rollback 1..K, rs=0..K)
2. MTP3 行为不回退: 接受率 / 速度 / 生成 sanity (T20 已知近并列分叉不算回归)
3. **显存实测**: 峰值对比 (预期 -400MB 级 @MTP3); 报告记录
4. PPL 4.3572±0.013; 短点/长点 (含 pp131072 或 depth32k) 不回退
5. 时间盒 **3-4 天**; 1.5 天 checkpoint = 记录格式 + fold 数值一致性打通

## 风险 / 注意
- fold 必须复现 GDN 内核同一有限精度路径 (T19-L 后为 scalar-only, 简化)
- T25 (state fp16) 已否 (降低精度不采纳) -> 无耦合; fold 面对的就是当前 f32 state 路径
- **若 fold 无法做到逐位等价 -> 放弃本项** (不做部分采纳)

## Result (checkpoint 1/2, 2026-09-24, implementer)

**fold 逐位一致性打通** (1.5 天 checkpoint 的核心目标): 独立 harness 证明 fold 内核与 verify 内核
是同一有限精度 transition - `fold(m) == 快照 slot (T-m)` 在 m=0..4 / 8 用例 (base, extreme_g,
tiny_k, big_k, multiseq2, beta0_g0, KDA, KDA+multiseq2) **全部逐位相同**; record side copy 逐位相同;
64 轮 verify+fold 长链 vs 纯序列基线 **逐位相同**。详见 RESULTS "T24"。
-> "fold 无法逐位等价则放弃" 的条件未触发, 可进入集成阶段。

集成方案 (待确认): record 缓冲 (每层 [k,v,g,beta] x T_cap) + committed state 单独一份 +
图内 fold op (每 batch 开头, m = T - rollback) + conv 列记录与 exact gather; state cache 只留 1 份 (-430MB @MTP3)。

## 集成实现计划 (2026-09-24, implementer) - 直通 M2, 开发期开关

设计定稿 (与上面"待确认"版的差异: **fold 并入现有 GDN op/kernel, 不新增 op**; 逐位一致由 Phase A 保证):
- record 布局 = op 输入布局的逐位拷贝 (per layer): `k [S_k,H_q,T_cap,B]`, `v [S_v,H_v,T_cap,B]`,
  `g/beta [*,H_v,T_cap,B]`, `T_cap = n_max+1`; 本模型 ≈ 8288 floats/layer/token (33KB) -> 48 层 ≈ 1.5MiB/token
- **双缓冲**: record 缓冲 = 2 x T_cap 槽 (每 batch 交替) - 避免"fold 读旧记录"与"本批写新记录"跨 block 竞争
  (k 在 GQA 组内共享: 同组 3 个 value head block 会同时读写同一 k)
- fold 时机: 每个 spec-verify batch 开头, 在 GDN kernel 内从 committed state 重放 `p = T - rollback` 条记录,
  结果 in-place 写回 committed (每 (h,col) 只碰自己的元素), 再处理本批 token
- 全接受时 p = T (fold(T) == verify final state, Phase A 已证) -> 统一走 fold, 无特例
- S 缓存只留 1 份 (**省 432MiB @MTP3**); R (conv) 仍用快照 (只省 ~17MB, 推迟到 M3)
- 非 spec / prefill 大 batch (T > T_cap): 无记录 -> 原路径 (直接写 committed)
- rollback 超 record 范围: 沿用服务器 checkpoint 回退 (不变)
- 开发期开关 `GGML_CUDA_GDN_REPLAY=1` (默认关 = T19-L 行为); 验证后固化并删除开关

改动点 (依赖序):
1. `llama-memory-recurrent.{h,cpp}`: record 缓冲分配 (n_rs_seq>0 时) + accessors;
   形状由 hparams 推 (S_v=ssm_d_state, H_v=ssm_d_inner/ssm_d_state, H_q=ssm_n_group, KDA=n_embd_head_kda!=0)
2. `ggml.h` / `ggml.c`: `GGML_OP_GATED_DELTA_NET` 增加可选第 7 输入 (records tensor, 可空)
3. `ggml-cuda/gated_delta_net.cu`: kernel 加 record 写 + fold 阶段 (门控); 融合结构体加 {rec, rec_read, fold_n}
4. `ggml-cuda/ggml-cuda.cu`: 融合匹配扩展 (把 record 指针 + 每批 [half, p] 交给 kernel)
5. `models/delta-net-base.cpp` `build_recurrent_attn`: 传 records/fold 输入; S 不再按 rs_idx 选平面 (恒 0)
6. `llama-graph.cpp`: 新 fold 输入类型 (set_input 写 [half, p_seq*] 到小 i32 tensor)

验收: (a) **逐位**: 同 prompt/seed 下新旧构建的 state dump / 贪心输出 / PPL 全等; (b) **VRAM -432MiB @MTP3**;
(c) MTP3 接受率/速度不回退; (d) pp/tg/PPL 不回退

## Result (checkpoint 2, M1/M2 集成, 2026-09-24, implementer)

**已实现** (全部由 `GGML_CUDA_GDN_REPLAY=1` 门控, 默认关 = T19-L 行为):
- record 缓冲: 每层一个打包张量 `[k|v|g|beta, 2*T_cap, mem_size]` F32 (本模型 8288 floats/token/layer,
  双缓冲 ~12.7MB @T_cap=4, 48 层), 形状由 hparams 推 (S_v/H_v/H_q/KDA)
- S 缓存 replay 模式只留 1 份 (省 3/4); R (conv) 仍走快照 (M3 推迟)
- `ggml_gated_delta_net` 增加可选第 7/8 输入: `rec` (per-cell view, F32) + `fold` (I32 [half, p_0..p_n-1])
- kernel: fold 阶段 (从 committed state 重放 p 条 record, 写回 state) + 主循环 record 写 + 非记录批次
  (n_tokens > T_cap) 提交 batch 末 state
- 图: `delta-net-base.cpp::build_recurrent_attn` replay 分支; `llm_graph_input_rs::fold` (set_input 写
  [half, p_seq*], 含 gen/half 跟踪); `s_copy` replay 时恒选 plane 0
- 融合匹配未改 (record 指针走 op 输入 view, 不再需要扩融合结构体)

**验证**:
- **env OFF 与 T19-L 生产部署逐位一致** (64 token 贪心全等 + MTP 接受计数一致) -> 无回归
- **env ON 仍分叉**: 前 5 token 相同, 第 6 个起不同 (= 第 2 个 verify batch 的 fold 首次生效处)
- 已修: 非记录批次未提交最终 state (修复前第 3 token 即分叉, 修复后推迟到第 6) -> 改善但未解

**下步 (待查)**:
1. record 写入/读取一致性 (已静态核对偏移: k=iq1*S_v+i, v=h*S_v+col, g/b=h; 但未实测)
2. 建议实验: `--spec-draft-n-max 1` (T=2) 对照; 或 kernel printf 对比 fold 的输入/输出
3. 或: 扩展 harness 验证生产 kernel 的 record 写入 (harness 原来用自己的 side copy, 没覆盖生产写入路径)
4. 注: MTP draft 是独立 context (`creating MTP draft context against the target model`), 不干扰目标内存

**产物**: `artifacts/t24-replayssm-wip-20260924.patch` (9 文件 +434/-36); 测试目录
`D:\LLM\Backend\llama.cpp-t24`; A/B 脚本 `%TEMP%\v100\t24_ab.ps1` + `t24_srv_run.cmd` (端口 8477,
CUDA_VISIBLE_DEVICES=1, MTP3 配置)。复现:
```
powershell -File %TEMP%\v100\t24_ab.ps1 -BinDir <dir> -Replay <0|1> -OutFile <txt>   # 比 tokens 逐位
```

### 补充 (2026-09-24 晚, implementer): 已定位到 record 写入路径

修了 `get_rec_p` 的代数 bug (gen 在 set_input 内先自增, 检查应为 `gen_of == rec_gen` 而非 `+1 ==`,
原写法导致 p 恒 0 = fold 从未生效)。修复并重建 llama.dll + 重建 server 后, **输出与修复前逐位相同**,
说明 fold 虽然跑了但**对 state 无影响**。结合 record 缓冲分配时被 `ggml_backend_buffer_clear(buf, 0)`
清零 -> **推断 fold 读到的是全 0 记录** (k=v=0, beta=0, g=0 -> delta=0, expf(0)=1 -> s 不变 = no-op)。
即: **record 写入没有真正落到 fold 读取的位置**。已静态核对 (未发现错误):
- 写: `wk[iq1*S_v+i]`, `wv[h_idx*S_v+col]`(lane0), `wg[h_idx]`(lane0), `wb[h_idx]`(lane0)
- 读: 同一公式, `rec_whalf = fold[0]`, 读 `1-rec_whalf`, `half_stride = (ne[1]/2)*nb1 = 4*8288`
- `t_cap = ne[1]/2 = 4`, `nb1 = nb[1]/4 = 8288`, `nb2 = nb[2]/4`
- `rec_write = n_tokens <= t_cap` (4<=4 真)

下步建议 (按优先级):
1. 在 kernel 里临时 printf 第一条 record 的 k[0]/v[0]/g/beta 与 `rec_whalf`/`rec_p` (用 env 门控只打一次),
   直接确认写入是否发生/地址是否正确
2. 检查 `src_rec->data` 在融合路径下是否是**设备指针** (record view 是 cache 张量的 view, 应为设备地址);
   同样确认 `src_fold->data` (host 输入) 是否被 scheduler 换成设备副本
   3. 检查 record 张量是否真的在 CUDA buffer 里 (memory_breakdown 打印 REC 行可确认; 当前 server 日志未含该行)

## analyst 复核 (2026-09-24, 用户中止后)

- **索引复核: 通过** (映射 vs [S,H,T,B] / GQA 同值重复写无害 / p<->rollback 等价; 静态排查未见 bug:
  融合路径传 GDN 节点本体、fold 与 s_copy 同 H2D 机制、rs_z 稳态 -1 不清状态、rec_l 同分配、读写公式对称)
- **关键**: "fold 未生效" 有三个不可区分状态 (p=0 / 读零 / 半区错), 先证实再修
- **定向实验 (见 QUESTIONS "A")**: 0 = 启动日志 `REC:` 行 (免 GPU, 1 分钟; `0.00 MiB` = rec_enabled false =
  replay 分支从未执行, 分叉来自 rec_replay 截平面 + K 快照路的 OOB/无回滚); 1 = set_input 日志 (p/half/gen);
  2 = kernel 一次性打印 (写后回读 vs fold 读); 3 = env ON 无 MTP 二分
- **代码审查发现**: `rec_replay` (截平面) 与 `rec_enabled()` (建图) 不一致 = OOB; 建议 assert/同源
- 工作区 WIP 未提交; 若回滚先归档最新 patch (`artifacts/t24-replayssm-wip-20260924.patch` 为旧版)

## 用户裁决与执行口径 (2026-09-24)

- **裁决**: 给**长时间窗口**排查; 能解决就集成, 否则回滚放弃 (无硬时限, 但建议每 ~0.5 天回报一次进展: 定位/未定位)。
- **排查顺序** (QUESTIONS "A", 从便宜到贵, 不要跳步):
  0. 启动日志 `REC:` 行 (免 GPU, 1 分钟; `0.00 MiB` = rec_enabled false = replay 分支从未执行);
  1. `set_input` 每批日志 (n_seq_tokens/will_record/gen/gen_of/rs_idx/rec_n/p/half, 限 20 行);
  2. kernel 一次性打印 (写后回读 vs fold 读; `rec.fold[0..4]`; 仅 layer0/block0/seq0);
  3. env ON **无 MTP** 二分 (纯 decode, 免 printf; 一致 -> p/回滚问题; 分叉 -> 链路问题);
  4. (可选) `--spec-draft-n-max 1` -> T=2 缩窗。
- **集成门槛** (全满足才固化; 届时删 `GGML_CUDA_GDN_REPLAY` 开关, 默认走 replay):
  1. **逐位等价**: env ON vs env OFF 同 prompt/seed 贪心输出全等 (+ 可选 state dump / PPL 全等);
  2. **显存实测**: MTP3 峰值相对 T19-L 降 ~432MiB 级 (mem_size=1, 按 cells 缩放);
  3. **不回退**: MTP3 接受率/速度, pp512/pp32768/pp131072, tg 短/长点, PPL (预期仍 4.3567);
  4. 通过后: 重建 + 部署核对 SHA -> 重生成 patch / 更新文档 -> 本地 commit。
- **回滚口径** (若放弃): 先从当前工作区重生成并归档最新 WIP patch (`artifacts/t24-replayssm-wip-<date>.patch`),
  再 `git checkout -- .` 回 HEAD `d24474edd`; 部署三处不动; 通道按放弃格式标记 (参照 T20 封存)。
- **注意**: 排查期间部署 DLL 保持 T19-L; 实验只在 `llama.cpp-t24` 目录; A/B 必须同 session 轮换。

### 修复 (2026-09-24 深夜, implementer): 根因 = hybrid set_input 不调用 rs 的 set_input

**根因**: `llm_graph_input_mem_hybrid{, _k, _iswa}::set_input` 自己复制了一份 s_copy 写入逻辑,
**从不调用 `llm_graph_input_rs::set_input`** -> 我加的 fold 参数写入代码从未执行 -> fold 张量恒为 0 ->
`p` 恒 0 -> fold 从不生效 (与"输出与修复前逐位相同"完全吻合)。

**修复**: 抽出 `llm_graph_input_rs::set_fold_input(ubatch)` (必须在 s_copy 消耗 rollback 之前调用),
在 `llm_graph_input_rs::set_input` 与 3 个 hybrid set_input 顶部调用。

**验证 (kernel printf + host log 双端)**:
- host: fold 参数正确 (`p=2/1`, `half` 交替) ✓
- device: fold 读到**真实非零记录** (`rk0=0.004570 rv0=-0.052227 rg0=-0.881120 rb0=0.804258`) ✓
- record 写入: `whalf` 正确交替 (96 次 half=1 / 96 次 half=0) ✓
- 结论: record 写入 + 双缓冲 + fold 读取链路已通

**新问题**: 加修复后首次运行在完成 4 个 recording batch 后**服务端崩溃** (HTTP 连接被关闭,
tokens=0; 日志尾部无 C++ 异常/断言信息 -> 疑似 CUDA illegal access 或 device assert)。
下步: (1) 复现并取 CUDA 错误信息 (server 日志完整版 / `CUDA_LAUNCH_BLOCKING=1`);
(2) 怀疑点: fold 的 state 写 (fused plane vs 非融合 tail)、p/half 在 draft context 的取值、
或 `s_shard` 写越界; (3) 之后再做逐位 token 对比 (本轮因崩溃未完成)。

注: 生产部署与 env OFF 路径仍未受影响 (env OFF == T19-L 逐位一致已两次复现)。

## 结束记录 (2026-09-24, analyst)

- **最终结果**: 未集成 -> 回滚放弃 (用户口径: 能解决就集成, 否则回滚)。
- **根因 (已找到并修复, 最有价值的产出)**: `llm_graph_input_mem_hybrid{, _k, _iswa}::set_input` 自带 s_copy 写入,
  **从不调用 `llm_graph_input_rs::set_input`** -> fold 参数写入从未执行 -> fold 恒 0 -> p=0 -> fold 从不生效
  (与 "env ON 分叉@token6 / 修 gen 后输出不变" 完全吻合)。修复 = 抽出 `set_fold_input()`, 在 rs 与 3 个 hybrid set_input 顶部调用。
- **已验证 (printf 双端)**: p/half 正确; fold 读到非零真实记录; 记录写入 half 交替正确 -> 链路已通。
- **未解决**: 加修复后 4 个 recording batch 后服务端崩溃 (无 C++ 异常; 疑 CUDA illegal access/device assert);
  候选: fold 的 state 写 (fused vs tail)、draft context 的 p/half、s_shard 越界。逐位 token 验收未做。
- **执行**: WIP 归档 -> `artifacts/t24-replayssm-wip-20260924-final.patch` (37,432 B; 旧 `-20260924.patch` 为早期版, 已被取代);
  工作区 `git checkout -- .` 回 `d24474edd`, 干净; 部署 DLL `054BFFD625E37E04` (T19-L) 未动, 已核对。
- **若要恢复**: `git apply artifacts/t24-replayssm-wip-20260924-final.patch` 即回到排查终点 (含 printf 插桩)。

## 重启与崩溃排查计划 (2026-09-24, 用户: 收益不小, 继续给时间)

- 工作区已回灌归档 patch (`t24-replayssm-wip-20260924-final.patch`), 回到排查终点; 部署 T19-L 不动; 实验仍在 `llama.cpp-t24`。
### 收益明细核算 (analyst, 2026-09-24; 每 cell/序列, n_max=3 -> n_rs_seq=3)

- state 快照: 现状 (1+n_rs_seq) = 4 份 x 144MiB = **576MiB**
- T24: 1 份 = **144MiB** + 新增 record 缓冲 48 层 x (8288 floats x 8 槽 x 4B) = **12.1MiB**
- 净: 576 -> **~156MiB**, **省 ~420MiB/序列** (任务书 "-432MiB" 只算 S cache, 加记录缓冲后净 ~420)
- 缩放: 随 `--parallel` 的 cells 数线性; n_max=1 时 288 -> ~150MiB (省 ~138)
- R (conv) 不变 (~23MiB/序列; M3 可再省 ~17MiB); 另: state 副本写少 3/4, decode ~1%

- 唯一阻塞 = 修复后 "完成 4 个 recording batch 后服务端崩溃" (无 C++ 异常; 疑 CUDA illegal access/device assert)。
- 排查顺序 (按性价比):
  1. **先去掉 printf 插桩重建再复现** (崩溃发生在带插桩的构建; CUDA device printf 不支持 `%p`, 且插桩本身可能有副作用)
  2. `CUDA_LAUNCH_BLOCKING=1` 复现 -> 取崩溃前最后一个 kernel + 完整 stderr (定位 launch)
  3. `compute-sanitizer --tool memcheck` 短跑 (10-20 token 单请求) -> 直接报越界地址/内核名 (最有效)
  4. 二分: `--spec-draft-n-max 1` (T=2); env ON **无 MTP** (无回滚, p 恒 = rec_n=1); 若都稳 -> 与回滚/多 token 相关
  5. 候选 (按怀疑度): (a) fold 的 state 写与融合 cpy 的交互 -> 用 `GGML_CUDA_DISABLE_FUSION=1` 对照 (非融合走 op tail);
     (b) draft context 是否也有 rec/fold 活动 (查两边启动日志 REC 行 + host 日志); (c) `s_shard`/`rec.ptr` 越界;
     (d) p/half 在 1/2/3-token 非满批次的取值
  6. 定位并修 -> 逐位 token 对比 (env ON vs OFF) -> 显存/MTP/pp/tg/PPL 全量验收 -> 集成 (删 env 开关)
- 提醒: env OFF 已两次逐位等 T19-L; 崩溃排查期间生产部署保持 T19-L。

### 崩溃定位 (2026-09-24, implementer)

`CUDA_LAUNCH_BLOCKING=1` 复现:
```
ggml-cuda.cu:108: CUDA error
E CUDA error: an illegal memory access was encountered
E   in function ggml_cuda_graph_evaluate_and_capture (ggml-cuda.cu:4395)
E   cudaGraphLaunch(graph->instance, cuda_ctx->stream())
```
崩溃**仅在 fold 真正生效后**出现 (修复前 fold 从不运行, 不崩), 故越界来自 fold 相关路径。

**首要怀疑 (与 1 平面布局有关)**: replay 模式下 S 缓存只有 1 个平面 (`n_rows_s = mem_size`),
但 `build_rs` (llama-graph.cpp:3478) 仍按 **(1+n_rs_seq) 平面**布局寻址:
- `state_zero` 视图: `rs_zero*states->nb[1]` (rs_z 可能是跨平面行号)
- `states_extra` cpy: 写到 `(rs_head + n_seqs)*s->nb[1]` (n_rs > n_seqs 时越界)
这两处都在 CUDA graph 内执行 -> 与 "graph launch 时 illegal access" 吻合。

**建议修复**: replay 模式下跳过 `build_rs` 的 rs_z 清零与 extra-state cpy (1 平面时二者都不需要:
初始状态由分配时的 buffer clear 保证为 0, 且无跨平面行)。或者把 rs_z 限制在 mem_size 内。

其它已排除: record 张量读写偏移 (已在 device 端验证非零且 half 交替正确)、fold 的 p 取值 (p<=rec_n<=T_cap)。

### 崩溃排查第二轮 (2026-09-24/25, implementer): 竞态, 非融合

1. **排除融合**: 只关 GDN snapshot-cpy 融合 (`gdn->src[6] != nullptr -> return 0`) 后**仍然崩溃**;
   全关融合 (`GGML_CUDA_DISABLE_FUSION=1`) 下第一次跑完 (draft_n=66/acc=40), 第二次又崩 -> 说明
   与融合无关, 之前那次"融合修复"只是接受模式不同造成的巧合。
2. **确认竞态**: `compute-sanitizer --tool memcheck --cuda-graph-trace=node` 下**同配置跑完 64 token
   且无任何 memcheck 报错** (draft_n=51/acc=46) -> sanitizer 的串行化掩盖了问题 -> 时序相关竞态。
3. **数值正确性**: 全关融合时 env OFF == T19-L ref **逐位一致** (证明融合本身 bit-exact);
   env ON 的对比因崩溃/未跑完尚未拿到 (sanitizer 那次跑完但 tokens 未落盘, 待补)。
4. **怀疑方向** (待验证): `fold` 输入张量的 host->device 拷贝与 GDN kernel 的竞争:
   GDN 用 PDL (`ggml_cuda_pdl_sync`) 只等前一个 *kernel*, 不等 `cudaMemcpyAsync`; 若 kernel 先读到
   *未初始化*的 device 输入缓冲 -> `fold[0]` 为垃圾 -> `rec_whalf` 越界 -> illegal access。
   验证方法: (a) 在 kernel 里对 `rec_half`/`rec_p` 加范围 clamp 后重跑 (若不再崩 = 确认);
   (b) 或把 fold 参数改走 op 输入之外的通路 (如写进 record 张量头部, 由 kernel 读)。
   临时缓解: 在 kernel 里 `rec_half = rec_half & 1; rec_p = min(max(rec_p,0), t_cap)` (仅防御, 非修复)。

注: 生产部署与 env OFF 路径不受影响 (env OFF == T19-L 逐位一致已多次复现)。

### 根因确认 (2026-09-25, implementer): fold 输入张量的 device 副本没被填充

关闭 CUDA graph (`cmake -B build -DGGML_CUDA_GRAPHS=OFF`) 后一次抓到:
```
T24DBG bad fold params: half=1866222143 p=0 t_cap=4
```
`half=0x6F3A...` = 未初始化显存 -> **`fold` 输入张量的 host->device 拷贝没有发生** (或 kernel 读到的
不是拷贝目标), kernel 读到垃圾 -> `rec_whalf` 越界 -> illegal memory access。
完全解释"时崩时不崩": 读到 0 时不崩, 读到垃圾时崩。

**为什么 s_copy 同样机制却没事**: 待查 (可能因为它被普通 op `get_rows` 消费, 而 `fold` 只被
`GGML_OP_GATED_DELTA_NET` 消费, 而该 op 在 CUDA 上走的是**执行期融合**路径 ->
scheduler 建 split 时的输入拷贝插入逻辑可能没覆盖到它)。

**修复方向** (择一):
1. 不依赖图输入拷贝: 把 fold 参数直接写进 record 张量头部 (持久 device 缓冲), 用
   `ggml_backend_tensor_set()` 由 host 直接写入 (48 层 x 8 字节/批, 开销可忽略), kernel 从
   record 张量读参数;
2. 或查清 scheduler 输入拷贝为何没覆盖 src[7] (对比 s_copy 的路径), 从机制上修。

**当前代码状态**: kernel 里有防御性 clamp (`half` 非法->0, `p` 越界->0) + 诊断 printf,
保证不崩但会让 fold 失效 (输出会错), 属临时保护, 不是修复。

**构建注意**: build 目录当前被重配为 `-DGGML_CUDA_GRAPHS=OFF` (为定位问题); 修复验证后需
`cmake -B build -DGGML_CUDA_GRAPHS=ON` 恢复。生产部署与源码未受影响。

### 方案 1 实施 + 新问题 (2026-09-25, implementer)

已按方案 1 改造 (patch: `artifacts/t24-replayssm-wip2-20260925.patch`):
- memory: 新增持久 device 张量 `cache_rec_fold` (I32 [1+n_seq_max], 与 record 同缓冲, 每 memory 一个);
- graph: `set_fold_input` 改用 `ggml_backend_tensor_set()` 直写该张量 (不再走图输入拷贝);
  移除 host `fold` 张量 / `can_reuse` 检查 / `build_rs_inp_impl` 里的创建;
- `delta-net-base.cpp`: op 第 7 输入改为 `mctx_cur->get_rec_fold_t()`;
- kernel: 保留 clamp 作防御, 移除调试 printf; ggml.c 断言放宽为 `ne[0] >= 1+n_seqs`。

**新问题**: env ON 运行在 **prefill 阶段静默崩溃** (日志停在 "processing task", 无 CUDA error/assert 输出;
两次都一样; env OFF 未测)。怀疑点:
1. `ggml_backend_tensor_set()` 在该张量上不可用 (buffer/backend 未就绪, 或 RS buffer 非 CUDA);
2. 持久张量作为 op 输入时 scheduler 的处理 (非 input 张量, 应视为常量, 但需确认);
3. 崩溃点可能在 hybrid set_input 调用链 (mctx 空/悬垂)。

下步: (a) 加 stderr 打印确认 set_fold_input 是否执行到 tensor_set; (b) 用 cdb/WinDbg 或
`GGML_ABORT` 定位; (c) 检查 `rec_fold_t->buffer` 与 backend 类型。

### 重大教训 + 当前状态 (2026-09-25, implementer)

**教训 (代价极大)**: 本项目的 impl-DLL 构建中 **`llama-server-impl.dll` 静态链接了 libllama 副本**。
改 `src/` 后只重建 `llama.dll` 时, server 用的是**旧布局的 libllama** -> 新增类成员被读成垃圾指针
(实测: 第二个 context 的 `rec_fold_t` = 垃圾指针 + `ne0=1858399458896`), 引发难以理解的崩溃。
-> **结论: 任何 `src/` 改动后必须重建 `llama-server` (build_server.cmd), 不能只建 `llama` 目标。**
-> 受此污染的早期结论 (如"hybrid set_input 不调用 rs 的 set_input"、"host 修复无效") 需要重测复核。

**当前状态 (重建 server 后)**:
- `set_fold_input` 每批调用一次 (日志确认 4 批 4 次, 无重复), 参数正确: `half` 交替, `p0` = 0/2/0/4;
- `fold_t` 有效 (buffer/data/ne0=2 都正常);
- 崩溃点推进到**第 4 批的计算阶段** = fold 第一次真正重放 (p=4) 时; 服务端仍无错误输出 (硬崩)。
- 下步: (a) 关 CUDA graph 再抓一次 (现在布局一致了, sanitizer/错误信息可能可用);
  (b) 用 cdb 抓 crash dump 定位; (c) 逐段 bisect fold (先去掉 state 写, 再只读不写);
  (d) 复核 record 张量内容是否真的被写入 (之前 device 端 printf 已确认过一次, 但现在布局变了要重测)。

注: build 目录当前仍是 `-DGGML_CUDA_GRAPHS=OFF` (定位用), 修复验证后需改回 ON。

### 根因 2 找到: conv(R) gather 的 s_copy 被 replay 模式改坏 (2026-09-25)

修复根因 1 (stale mctx) 后: **跑完 64 token, 接受计数与 ref 完全一致 (51/46), 前 44 个 token 逐位相同,
只在 index 44 有一个 token 不同 (198 vs 220), 之后又收敛**; ON 两次运行完全一致 -> 确定性差异。

**根因**: replay 模式下 `s_copy()` 被改为恒返回 0 (让 S 状态 gather 读 committed 平面)。
但 **conv(R) 缓存仍用旧的快照回滚** (M3 推迟), 它的 gather 复用同一个 s_copy ->
现在读 **plane 0 (最新 conv 状态) 而非 plane r (回滚 r token)**。只在部分接受的轮次出错,
下一轮快照写回后自愈 -> 单个 token 翻转 + 收敛, 与观测完全吻合。

**修复方案** (下次直接做):
1. `llm_graph_input_rs` 增加 `s_copy_conv` (+ main/extra 视图), 在 `build_rs_inp_impl` 里创建 (host 输入张量);
2. `set_fold_input` (replay 模式) 里对每个 i 调一次 `mctx_cur->s_copy(i)` (消费 rs_idx), 把值写进
   `s_copy_conv`; 这样随后的常规 s_copy 循环自然返回 0 -> S gather 读 committed 平面 (正确),
   而 conv gather 用 s_copy_conv 得到旧的 rollback 平面 (正确);
   非 replay 模式 set_fold_input 提前返回 (fold_t == nullptr), 一切照旧;
3. `build_rs(inp, s, state_size, n_seqs, get_rows)` 增加 `bool conv=false` 参数, `build_conv_state`
   (delta-net-base.cpp:463) 传 true 使用 s_copy_conv;
4. can_reuse 加 s_copy_conv 尺寸检查。

预期: 修完后 ON == ref 逐位全等。之后再跑显存/MTP3/PPL 验收。

### conv 索引修复已落地, 但分叉未消失 (2026-09-25)

已实施 s_copy_conv 方案 (rs 输入新增 s_copy_conv(+main/extra), set_fold_input 里消费 rs_idx 并写入
rollback 值, build_rs 加 conv 参数, build_conv_state 传 rec_enabled(), can_reuse 加检查)。
结果: **仍然是 index 44 的同一个 token 分叉 (198 vs 220)**, 其它全同 -> conv 索引不是原因 (或不是唯一原因)。

**已排除**: 崩溃/竞态 (ON 两次完全一致)、S 状态 fold 数学 (Phase A 逐位)、conv gather 平面选择 (已修)。
**剩余怀疑**:
1. **draft context 的 fold**: 若 draft 也有 rec_enabled, 它的 p/rollback 语义可能不同; 但其状态只影响
   drafts 不影响 target tokens (spec 是精确的), 除非影响 accept 结构;
2. **S 状态在某个特定 batch 有 1-ulp 级差异**: 建议用 printf 直接对比: 在 kernel 里对
   (固定 layer, 固定 batch) 打印 fold 的输入 state[0] 与结果 state[0], 在 env OFF 下打印对应的
   snapshot 输入/结果, 逐位对比定位第一个不同的 batch;
3. **target 的 rollback 值本身**: 建议在 set_fold_input 里打印 p/rollback (env 门控) 与
   server 侧的 accept 数对照 (可从 server 日志的 draft acceptance 行推出每轮 accept)。

### 代码生成假设排除 + 差异定位到数据层 (2026-09-25)

实验: 把 kernel 的 `replay` 从编译期模板参数改为**运行期参数** (env OFF 的路径现在也包含
fold/record 代码, 只是运行期不执行)。
- **env OFF(+fold 代码) == ref 逐位一致** -> 添加代码不改变主循环浮点结果, 代码生成假设**排除**;
- env ON 仍在 index 44 分叉 (198 vs 220), 完全一致 -> 差异在**数据**层。

**已排除**: 崩溃/竞态、fold 数学 (Phase A)、conv gather 平面、代码生成/FMA 收缩、主循环实例差异。
**逻辑链上"应该全等"但实际不等**, 说明某个环节的等价性假设有漏洞。下一步用 printf 做**逐 batch 数据对比**:
- ON 侧: 在 fold 完成后打印 (层=0, block 0, lane 0) 的 `s_shard[0]` + `rec_p` + `rec_half`;
- OFF 侧: 在 snapshot 写入处打印同一位置的值 + `target_slot`;
- 对应关系: ON 的 fold(p) 应等于 OFF 的 snapshot slot (T-p); 逐 batch 对齐找**第一个不同的 batch**,
  再回溯那一批的输入 (base state / records / conv)。

实现提示: 用 `__device__` 全局计数器给 batch 编号; printf 仅 block(0,0,0) thread(0,0) 打印一次/层。

### printf 追踪结果: fold 控制参数全部正确 (2026-09-25)

坑: `n_tokens` 是 int64_t, printf 用 `%d` 导致后续参数**整体错位 4 字节**, 一度误判为
"p 恒为 0 / half=2,3,4" (实为错位后的假象)。加 `(int) n_tokens` 后真实值:
- `p` = 4 / 3 / 1 / 0 (非零, 与 accept 计数一致), `half` = 0/1 正确交替, `wr` 正确;
- prefill (nt=16) wr=0 + END 提交路径触发 ✓; 记录批次 wr=1 走 fold 提交 ✓。

结论: **fold 的运行参数、记录半区、gen 判定都正确** -> 差异在**数值**层 (fold 结果 vs 快照值)。
下一步 (下次开工直接做): 在 fold 结束后打印 `s_shard[0]` (ON), 在 snapshot 写入处打印同一逻辑位置
的值 (OFF), 按 batch 对齐找出第一个数值不同的 batch; 再回溯该 batch 的 base/records。

注意: 追踪 printf 目前留在 kernel 里 (T24BISECT 块), 验证完必须删除。

### 数值对比实验: 日志顺序不可靠, 必须加 host 侧 launch id (2026-09-25)

实验: kernel 打印 fold 的 base/res (T24RES) 和 snapshot 的 slot/v (T24SNAP), 想按 batch 对齐比较。
结果: **无法用日志顺序对齐** -
- 层是独立 op (grid 无 z 维), 每 batch 有 48 个 launch; target 与 draft 两个 context 交错;
- CUDA device printf 缓冲按刷新点输出, 顺序与 launch 顺序不保证一致 -> 分组错位 (B 组数 78 != 72,
  且部分 slot 值解析为空);
- 另一个坑: 中间张量由 graph 分配器复用 -> `state` 指针在各层之间相同, **不能用来区分层** (只能区分 context)。

已获得的有效线索 (待对齐后确认):
- A[4..7] 的 base 与早先 T24END(nt=16) 的值吻合, 说明 prefill 的 END 提交路径正确;
- ON 侧前几个 verify batch 的 `p=0` (base==res, 空折叠), 之后 p=4/1。**需要确认这些 p=0 是否合法**
  (设计上 rollback 超过上一 batch 记录数时回退到 p=0 并用平面值, 若服务器 rollback 覆盖整个
  prefill chunk 则合法; 否则是 rec_n/gen 判定错误)。

**下一步 (必须做)**: 在 `launch_gated_delta_net` 里用 host 静态计数器生成 launch id, 经
`gdn_rec_params` 传给 kernel 并打印; 按 launch id 对齐后:
1. 校验每个 recording batch 的 p 是否等于上一 batch 的 accepted 数;
2. 比较 ON 的 fold 结果 res 与 OFF 的 snapshot(B[k-1] slot 4-p) 逐位相等, 找出第一个不同的 batch。

注意: 所有 T24BISECT 追踪块 (T24FOLD/T24RES/T24SNAP/END) 验证完必须删除。

### launch-id 方案受阻: device printf 日志本身不可信 (2026-09-25)

加了 host 静态计数器 (gdn_rec_params.lid, 每次 GDN launch +1) 并打印 lid, 结果:
- ON: 3840 条 RES 行, 但 lid 大量重复 (0,0,0,0,1,1,1,...) -> 分组全乱 (2881 组, 大小 1);
- OFF: 14976 条 SNAP 行 -> 14017 组。

结论: **本环境下 CUDA device printf 的输出缓冲会重复/乱序** (3840 次 launch 全部在同一个 stream 上,
printf 缓冲按刷新点转储, 行序和内容都不保证)。**任何基于日志顺序/行数的对齐都不可信**。

**改用的正解 (下次直接做, 自包含, 不需要任何对齐)**:
让 ON 运行同时具备快照路径, 在**同一次 launch 内部**比较:
1. 内存 S 缓存回到 `(1+n_rs_seq)` 平面, 图传 `K = n_rs_seq`;
2. kernel 里快照写入偏移一个平面 (`state + (target_slot+1)*state_slot_stride`), 使平面 0 专供 fold 提交;
3. fold 完成后, 比较 `s_shard[0]` 与平面 `(rec_n[seq] - p)` 处的快照值 (rec_n = 上一 batch 的 token 数,
   经 fold 张量扩一维传入);
4. **只打印不匹配** (日志量极小, 出现即证明问题)。
若全部匹配 -> fold/记录链路数学正确, 问题在别处 (平面基值/conv/调度); 若某 batch 不匹配 -> 直接拿到
第一个坏 batch 的 (layer, batch, p, 两个数值)。

同时提醒: T24BISECT 追踪块验证完必须删除; printf 用 `%d` + `(int)` 强转, 不要用 `%lld`。

### 根因确认: set_fold_input 每 batch 被调用 96 次, half 翻转偶数次永不交替 (2026-09-25)

自检方案 (fold 结果 vs 同 launch 内的快照值, 平面 0 专供 fold 提交, 快照写平面 1..K) 一次命中:
**3456 次检查全部不匹配, fold 结果恒为 0**。

探针证据 (block 0,0,0 thread 0, layer 0):
- 写入侧 batch1: `T24WR lid=0 half=1 k0=-0x1.77276p-9 base=1024578000 wk=1024598600` -> 写到 half=1 半区 ✓
- 读取侧 batch2 fold: `T24IN lid=48 p=2 half=1 k0=0x0p+0 base=1024578000 kk=1024578000` -> 读 (1-1)=half 0 半区 -> 全零 ✗

**根因**: `set_fold_input` 每 batch 被调用 **96 次** (48 个 GDN 层 x 2 个输入: rs + conv),
每次调用都跑 `rec_begin_batch()` -> `rec_gen++` 且 `rec_half = 1 - rec_half`。
96 次翻转是偶数 -> **half 每批回到原值, 写入半区与读取半区永远错开** -> fold 读空 -> 状态错 -> index 44 分叉。
(此前"每批恰好调用一次"的结论来自被 printf 缓冲污染的日志, 是错的。)

**修复方案 (下次直接做)**:
把副作用从 `set_fold_input` 里挪到每 batch 只跑一次的地方:
1. 在 recurrent context 的 `init_batch(ubatch)` 里: 先按现有逻辑算好 `p` (此时 rec_gen 还是上一批的, gen 校验通过) 与
   `half`, 存进成员 (如 `rec_fold_data` vector); 然后调 `rec_begin_batch(will_record)`;
2. `set_fold_input` 退化为**幂等**操作: 只把已存好的值 `ggml_backend_tensor_set` 到 fold 张量
   (外加消费 rs_idx 写 s_copy_conv, 重复消费同一值无害);
3. 校验: ON == ref 逐位一致 + 自检 0 mismatch + PPL/速度;
4. 之后清理所有 T24BISECT 追踪块和 host launch id。
注意: 改 src/ 后必须删掉 `llama-server-impl.dll` 再 build_server (cmake 不会自动重链),
且 `ggml-base.dll` 等所有 DLL 都要拷到测试目录。

### half 交替 bug 已修复 + 新现象 (2026-09-25)

修复: `init_batch` (每 ubatch 一次) 设 `rec_fold_pending=true`, `set_fold_input` 只在
`rec_fold_take()` 返回 true 时执行副作用 (rec_begin_batch/rec_commit/fold 张量写入)。
结果:
- **自检 0 mismatch** (3456 -> 0): fold 结果与同 launch 内快照逐位一致, half 交替正确;
- 但 ON != ref: **首个 token 差异提前到 index 5** (ref 494 vs on 314), accepted 42 vs 46, draft_n 62 vs 51。
- 修复前是 index 44 分叉 (那时 fold 读空、几乎被跳过)。

解读: 修复让 fold 真正生效 -> 目标/草稿状态都变了。accept 模式变化说明**草稿 context 的状态路径**
与 OFF 不一致 (accept 数由草稿质量决定)。目标侧状态由自检验证过机制正确, 但"语义正确"未被验证
(自检是同一 kernel 内自洽, 不能证明与 OFF 快照路径等价)。

**下一步 (下次直接做)**:
1. 关掉 MTP (`--spec-type`/`--spec-draft-n-max` 去掉) 跑 ON 与 OFF 各一次 -> 无 rollback 时两者应逐位一致。
   若一致 -> 目标路径正确, 问题锁定在草稿 context; 若不一致 -> 目标路径本身还有问题;
2. 草稿侧排查点: 草稿 context 的 rec 状态是否被 server 用不同方式 rollback (rs_idx 语义),
   以及草稿 context 是否也应启用 replay (它有自己的 memory 实例)。

### 关键对照: 无 MTP 时 ON == OFF 逐位一致 (2026-09-25)

实验: 关掉 spec (`--spec-type/--spec-draft-n-max` 去掉, 端口 8478, 脚本 t24_ab_ns.ps1), ON/OFF 各跑一次 64 token:
- **NOSPEC ON==OFF: True** (逐位一致) -> 记录/半区/提交/fold 整条状态链路在无 rollback 时完全正确。

结论: 剩余差异**只在 rollback 路径**。带 MTP 时 ON 首个 token 差异在 index 5, accepted 42 vs 46, draft_n 62 vs 51
-> accept 模式变化说明**草稿 context 的 rollback 处理与 OFF 不一致** (草稿质量由草稿状态决定)。

**下一步候选**:
1. 用 `--spec-draft-n-max 1` 做最小复现 (频繁小 batch rollback), 对比 ON/OFF;
2. 重点查 server 对**草稿 context** 的 rollback 语义: 目标侧 rollback = rs_idx, 草稿侧据 upstream 代码是
   `n_to_drop = n_drafted_last - n_accepted - 1` (多一个 -1)。若草稿 rollback 会跨过上一 batch 边界,
   我的 `p = rec_n - rollback` (rollback>=rec_n 时回退 p=0 用平面值) 与 OFF 的 `s_copy = r*size + src0`
   (读快照平面 r) 语义不等价 -> 需要让 ON 侧在 rollback 超界时也能回退到正确状态;
3. 也可先只保留目标侧 replay (草稿 context 关掉 replay) 验证是否 ON==ref。

脚本: `%TEMP%\v100\t24_ab_ns.ps1` + `t24_srv_run_ns.cmd` (无 spec, 端口 8478, 日志 t24_ns_<r>.log)。

### 最小复现 + 差异机制假设 (2026-09-25)

`--spec-draft-n-max 1` (2-token batch, 频繁 rollback): **ON != OFF, 首个 token 差异 index 8**。
无 spec 时 ON==OFF 逐位一致; 有 spec 就分叉 -> 差异**只在 rollback 路径**。

机制假设 (待验证): 两条路径在 **rollback >= 上一 batch token 数** 时语义不等价:
- OFF (参考/上游行为): `s_copy = r*size + src0`, 读快照平面 r。若 r >= T_prev, 该槽位上一 batch
  **没写过** -> 读到更早 batch 留下的陈旧值 (上游就这样, 且实测输出正常);
- ON (T24): p = rec_n - rollback = 0 -> fold 跳过 -> 用平面 0 = **最后一次提交的已提交状态**
  (语义上更"正确", 但与 OFF 读到的陈旧值不同) -> 逐位不一致。

=> 关键问题: 目标侧 rollback 是否可能 >= T_prev? upstream 目标侧 r = n_drafted - n_accepted
(草稿侧才多一个 -1), 若至少采样的那个 token 总被接受则 r <= T_prev-1, 两条路径等价;
若 r == T_prev 真会发生, 则 T24 必须复现快照语义才能与上游逐位一致 (1 层记录做不到 -> 设计约束)。

**下一步 (决定性)**: host 侧打点 (`set_fold_input` 内, 每 batch 一行: seq/rs_idx/rec_n/p), 在 n-max-1
场景跑 ON, 统计 r >= rec_n 的出现次数与位置, 对照首个分叉 batch (index 8 附近)。若从不出现 ->
差异另有原因; 若出现 -> 按上面的机制修 (或确认上游语义后决定 T24 的等价策略)。

### analyst 复核 (2026-09-25): 里程碑与剩余问题的性质

- **里程碑**: 无 rollback 时 ON == OFF 逐位一致 (64 token) -> 记录/半区/提交/fold 整链在无回滚语义下**正确**;
  half bug 根因 (set_fold_input 每批 96 次 -> 偶数翻转) 定位+修复正确; 同 launch 自检 0 mismatch。
- **剩余差异只在 rollback 路径, 性质 = "语义等价"而非链路 bug**:
  OFF 参考读快照平面 r; r >= T_prev 时该平面本批未写 -> 读到**陈旧历史值** (上游 best-effort);
  T24 的 p = rec_n - r -> 0 -> 用 committed 状态 (语义更正确)。
  **1 层记录无法复现陈旧平面值** (可能来自任意久远批次, 需无界历史) -> 若该情形真发生, 逐位等价在设计上不可达
  (除非保留快照平面 = 收益归零)。
- **决定性实验 (先做, 便宜)**: 按 **context 分开**打点 (target/draft): 每批 `seq/rs_idx/rec_n/p`; n-max=1 跑 ON,
  统计 `r >= rec_n` 出现位置是否与首个分叉 (index 8) 对齐; 并核对 server 目标侧 r 公式 (`n_drafted - n_accepted`,
  草稿 -1) 是否恒保证 r < rec_n (若恒保证, stale 只可能来自草稿 context, 或多次 seq_rm 被 pending 丢弃)。
- **若确认发生 (预置选项, 待用户裁决)**:
  (a) **分 context 二分 (最便宜)**: "草稿侧不启用 replay" 跑一次; 差异只在一侧 -> 该侧保持快照, 另一侧 replay;
  (b) **放宽门槛**: 角落情形用 committed 状态 (文档化 "比上游更正确"), 输出 token 仍全为 target 采样 (分布不变),
      需用户批准 + 全量质量验收;
  (c) 严格逐位 -> 只能保留快照平面 (收益归零), 不值得。
- **构建纪律**: 改 `src/` 必须重建 server (删 `llama-server-impl.dll`), 否则库布局不一致产生垃圾指针
  (已确认坑; 受污染的早期结论需重测)。

### 修复 2: pending 标志必须放在 `prepare` 里 (2026-09-25) - n-max-1 已逐位等价

发现: `llama_memory_hybrid::init_batch` **不调用** `mem_recr->init_batch`, 它调用 `mem_recr->prepare(ubatches)`。
所以上一版把标志放在 recurrent 的 `init_batch` 里**从未生效** (T24P host 日志 0 行, "0 mismatch" 是空判)。
改为在 `llama_memory_recurrent::prepare()` 开头 `rec_fold_pending = true;` 后:

- **n-max=1 (2-token batch, 频繁 rollback): ON == OFF 逐位一致** (64 token) ✓✓
- **MTP3: 仍不等, 首个 token 差异 index 44** (198 vs 220), 但 **draft_n=51 / accepted=46 与 ref 完全一致** (accept 模式已对齐)
- kernel 自检 (fold vs 同 launch 快照 at slot tn-p): **6144 mismatch** -> fold 结果与该快照不等的 batch 真实存在,
  且 n-max-1 时状态差异小到没翻 token, MTP3 时在 index 44 翻了一个。

判读: 剩余差异 = **fold 结果 != 快照路径在 slot (tn-p) 的值**, 且 p>0 (r < tn) -> 不是"陈旧平面"机制,
是 fold 本身在部分 batch 上与快照路径不等价。

**下一步 (决定性, 下次直接做)**: 把 mismatch 打印扩成 `lid/p/tn/fold/snap(tn-p)/slot0(上一 batch 末状态)` 并只在 layer 0 打:
- 若 `fold == slot0` -> fold 重放了**整批**而不是 p 条 (p/记录数 bug);
- 若 `fold` 等于某个别的 plane 值 -> 记录半区/偏移错位;
- 若都不等 -> 记录内容本身与上一批实际消费的 token 不一致 (写侧 t/偏移)。

另: T24P host 日志没打出来 (LLAMA_LOG_INFO 未出现在 server 日志), 需要确认 server 日志等级或改用 fprintf(stderr)。

### analyst 复核 (2026-09-25, 修复 2 后)

- **陈旧平面假设被证伪**: 剩余 mismatch 在 `p>0 且 r<tn` 时出现 -> 不是 "读未写槽位", 是 fold 结果与快照在
  正常回滚范围内不等价。之前 "0 mismatch" 是**空判** (flag 放 init_batch 从未生效) -> 教训: **自检必须打印执行次数**,
  不能只看 mismatch 数 (0 也可能是没跑)。
- **现状**: n-max=1 逐位等价 ✓; MTP3 差 index 44 且 accept/draft_n 已对齐 -> 目标侧状态在某个 batch 漂移。
- **分类打印必须同时给 bit/ULP 差** (实现者计划 + 这条):
  - |Δ| <= 1-2 ulp 且普遍存在 -> **代码生成差异** (fold 块 vs 主循环块 FMA 收缩/调度不同) -> 修法: 把状态更新抽成
    共享 `__device__ inline` 函数, fold 与主循环调用同一实现;
  - 大差/特定 batch -> 平面/偏移/slot 索引 (含 debug 构建的 +1 平面) 或记录内容错位;
  - 同时先做 **checker 自证**: 打印检查次数 + 控制组 (把 fold 换成读快照, mismatch 必须为 0)。
- **另发现潜在 bug (与本分叉可能无关但必须修)**: `rec_fold_pending` 放在 `prepare()` = **每 batch 一次**且 prepare
  只是模拟 (之后 restore); 真正的**每 ubatch** 钩子是 `llama_memory_recurrent_context::apply()` (llama-context 的
  `do { ... } while (mctx->next())` 每 ubatch 调一次)。当前放置 -> 多 ubatch 的 batch 只有第一个 ubatch 会
  gen++/toggle/commit -> **prefill 尾 ubatch <= 4 token 时会漏记录/漏提交 -> 下一批 p=0 状态卡住**。
  建议移到 `apply()` (或 find_slot 的实调用路径), 并加回归: prompt 长度使尾 ubatch <= 4。
- 提醒: `LLAMA_LOG_INFO` 未出现在 server 日志 -> 改用 `fprintf(stderr)` 或确认日志级别。

### 修复 3+4 结果 (2026-09-25): OFF 无回归; ON 仍差 44; 自检恒失败 -> checker 需自证

已实施: (3) 状态更新抽成共享 `__device__ __forceinline__ gdn_state_update` (fold 与主循环共用, 消除 ULP 级 codegen 差异);
(4) per-ubatch 标志同时放在 `prepare()`/`next()`/`apply()` (多 ubatch batch 的每个 ubatch 都会 gen++/toggle/commit)。

结果:
- **OFF == ref 逐位一致** (无回归, 关键!) 
- **ON 仍 != ref, 首个差异 index 44** (198 vs 220), accepted/draft_n 与 ref 一致
- 自检 mismatch 数: 3456 = 72 batch x 48 层 = **每一次检查都失败** -> 但 n-max-1 场景 (同样恒失败) 的 token 却逐位一致
  => **自检的期望值大概率写错了** (不是 fold 真错), 必须按 analyst 要求做 **checker 自证**:
  1. 打印检查执行次数 (证明真跑了);
  2. 控制组: 把 fold 结果替换成直接读该 slot 的快照 -> mismatch 必须为 0;
  3. 打印 mismatch 的 bit/ULP 差 (分类: <=2 ulp -> codegen; 大差 -> 索引/内容错位)。
- 排查 slot 映射时注意: 写入侧是 `((replay?1:0)+target_slot)*stride`, 读取侧我用了 `(tn-p+1)*stride`;
  两者语义需逐一核对 (可能差一个 plane, 这会解释"恒失败")。

**下一步**: 先做 checker 自证 + 打印 ULP; 若确认 slot 映射错 -> 修 checker; 若真错 -> 按 ULP/大差分类定位。

### analyst 复核 (2026-09-25, 修复 3+4 后)

- **关键信号**: 自检 mismatch = 3456 = 72 batch x 48 层 = **每次都失败**, 但 n-max=1 (同样恒失败) token 逐位一致
  -> 几乎可以肯定 **checker 的期望值/索引错了** (测量仪器坏了), 先修仪器再谈定位。OFF == ref 逐位一致
  (fix 3 共享 inline 未改 OFF 数值) 是好消息。
- **修 checker 的两个快速判据**:
  1. **特例校验**: 找 `p == tn` 的 batch -> 此时 fold 结果必须等于**上一批末状态** = 平面 1 (debug 构建 +1 偏移);
     若这类也失败 -> 基址/记录/参数错; 若这类通过而 p<tn 失败 -> slot 映射 `(tn-p+1)` 或 `tn` 来源错;
  2. 打印实际使用的 `tn` 并与 host 侧 rec_n (每批一行) 对照; 打印 `state` 基址 (确认是 cache 基址, 不是 op tail / get_rows 副本)。
- **若 checker 一时修不好, 换无仪器方案 (更稳)**: 每 batch 结束后 dump S 缓存 plane 0 (ON) 与对应回滚平面 (OFF),
  逐 batch 比对 -> 直接拿到**第一个状态不同的 batch** 及其 p/rollback; 再回看那一批的记录写入 (printf 该批 t=0..3 的
  k/v/g/beta) 与 fold 读取值, 定点定位。此法不依赖 kernel 内自检逻辑。
- 备注: flag 同时放 `prepare()/next()/apply()` 依赖"每 ubatch 恰好 take 一次"; 建议加 take 计数断言 (每 ubatch == 1),
  防 "set_input 被调用两次" 类回归。

### 决定性证据: p==tn 特例下 fold 仍为 0 -> 记录链路仍未生效 (2026-09-25)

按 analyst 特例判据 (p == tn -> fold 必须等于上一批末状态 = slot0 = 平面 1):
```
lid=48 p=2 tn=2 fold=0x0.0000000000000p+0 snap=-0x1.5203p-12 slot0=-0x1.5203p-12 slot1=0x1.1ca6fcp-12
```
- `snap == slot0` -> **checker 索引这次是对的** (plane tn-p+1 = 1 = slot0), 仪器问题排除;
- **fold = 精确 0** -> fold 基址 (= 上一批 fold 提交 = 初始 0) + 记录读出全 0 -> **记录是空的**。

结合历史探针 (batch1 half=1, batch2 half=1, 未交替) => **half 交替仍未生效**。
最可能原因: 本轮把 flag 同时放在 `prepare()/next()/apply()` -> **同一 ubatch 被 take 多次** ->
`rec_begin_batch` 多次翻转 -> 偶数次 -> 半区不交替 (回到旧的坏状态)。

**修复 (下一步)**: flag **只**放在 `apply()` (analyst 指定), 且加 **take 计数断言** (每 ubatch 恰好 1 次):
在 memory 里 `rec_take_count++`, 在每次 ubatch 结束时校验 == 1; 若 >1 -> 说明置位点重复。
修好后重跑: 期望 fold == slot0 (特例) + 自检 0 mismatch + MTP3 ON==ref。

### analyst 确认 (2026-09-25, 特例判据结果)

- **仪器排除**: `snap == slot0` 证明 checker 索引正确 (`tn-p+1 = 1 = slot0`) -> 之前 "恒失败" 是**真失败**, 不是 checker 错。
- **根因方向确认**: `fold == 0` 精确为 0 (记录读出全 0 + 基址 0) + batch1/batch2 half 均 =1 (未交替)
  -> **half 交替仍坏**: flag 三处放置 (prepare/next/apply) 使同一 ubatch 被 take 多次 -> 偶数次翻转。
- **同意修复方案**: flag **只**放 `apply()`; take 计数断言 (每 ubatch == 1; 注意 `set_input` 每 ubatch 会被调 ~96 次,
  take 必须由 pending 门控, 断言在 ubatch 边界校验)。
- 修好后按序验证: (1) 特例 `p==tn` -> `fold == slot0`; (2) 自检 mismatch = 0 **且检查次数 > 0**; (3) n-max=1 + MTP3 ON == ref 逐位。

### 最后一处根因定位: 非记录批不断翻转导致读半区错开 (2026-09-25) - 决定性

修复 per-ubatch 幂等键 (指针+首位置 `rec_fold_take(ub, pos0)`) 后:
- **自检 mismatch 3456 -> 192**; T24P host 日志打通 (fprintf(stderr)) 且 readhalf 基本正常交替;
- 192 = 4 batch x 48 层 = **恰好是"上一批为非记录批"的转换** (prefill 的 nt=16 与 nt=4 尾块, 两个 context 各 2 次);
- 日志证据: `nt=2 wr=1 readhalf=0` -> `nt=16 wr=0 readhalf=1` (非记录批不翻转) -> `nt=4 wr=1 readhalf=1`
  -> 该批 kernel 用 `(1-rec_half_c)` 读, 与上一**记录**批的写半区错开 -> fold 读空。
- MTP3 仍差 index 44; OFF == ref 无回归。

**根因**: kernel 读半区公式 `(1 - rec_half_c)` 隐含假设"本批已翻转", 但 `rec_begin_batch` 只在 `will_record`
时翻转 -> 上一批为非记录批时读半区错。

**修复 (下一步, 最后一步)**: 在读半区上不要依赖翻转, 由 host 显式给出:
- 方案 A (最小): fold 张量再加一项 `read_half`; host 每次写参数时同时写"最近一次记录批的写半区";
  kernel 的 fold 用 `read_half` 而非 `(1 - rec_half_c)`;
- 方案 B: 非记录批也翻转 half 但写侧不写 (会破坏另一半的语义, 不推荐)。
修好后验收: 自检 0 mismatch + MTP3 ON == ref + n-max-1 ON == OFF + OFF == ref (无回归)。

### read_half 显式化后仍 192 mismatch + 44 分叉 (2026-09-25)

已实现: fold 参数 `fold[0] = write_half | (read_half<<8)`; memory 记 `rec_written_half` (最近一次记录批的写半区);
kernel 的 fold 直接用 `read_half` 定位记录 (不再依赖翻转)。build + 跑 MTP3 ON:
- **mismatch 仍 192** (4 batch x 48 层), MTP3 仍差 index 44, OFF 侧无回归。
- 192 = 4 个 batch: 与"上一批为非记录批"的 batch 数一致 (T24P: nt=16 p=2 wr=0 等)。

**重要怀疑 (analyst 早已提示)**: 自检读的"快照平面"在 **op dst 的 tail** 里 (`state = dst + S_v*H*n_tokens*n_seqs`),
而 flush 到 S 缓存的 cpy 只搬 plane 0。不同形状的 batch (nt=16 vs nt=4) 的 dst 分配不同 -> tail 里的
plane 1..K 可能被别的 batch 覆盖 -> 这 192 很可能是**仪器假象**, 不一定是 fold 真错。
而 MTP3 的 44 分叉说明**确实还有真实差异** (或在别的环节)。

**下一步 (无仪器方案, analyst 建议的 plan B, 最稳)**: 不再依赖 kernel 自检:
1. 加 host 侧 (或 kernel) 在**每批结束**后 dump S 缓存 plane 0 的少量元素到磁盘 (ON), 在 OFF 侧 dump
   对应回滚平面 (r*size) 的同一元素 -> 逐 batch 比对, 找**第一个状态不同的 batch**;
2. 拿到该 batch 的 (p, rollback, nt) 后, 只 printf 那一批 t=0..3 的写入 k/v/g/beta 与 fold 读取值。
  此法绕开 tail/快照可被覆盖的问题, 直接给绝对证据。

### analyst 复核 (2026-09-25, read_half 后)

- **进展确认**: take-key (指针+首位置) 修复使 mismatch 3456 -> **192** ✓; 192 = 4 batch x 48 层 = "上一批为非记录批" 的转换,
  与 host 日志 readhalf 异常一一对应 -> **read_half 显式化方向正确** (Option A 对)。
- **192 的可信度**: 同意你的怀疑——debug 自检读的是 **op dst tail** 里的平面, 而 replay 只把 plane 0 经 cpy 刷进 S 缓存,
  tail 的 plane 1..K 是**临时显存** (不同形状 batch 会复用/覆盖) -> 192 很可能是**仪器假象**。二选一:
  (a) 想让自检可信: 把 debug 快照写进**缓存平面** (稳定内存) 而非 op tail; 或 (b) 直接上 plan B (无仪器)。
- **plan B 细节 (建议)**: 每个 ubatch 结束后 dump **committed plane 0** (ON) 与 OFF 对应回滚平面 `r*size + src0` 的同位置
  少量元素 (如每层 h=0,col=0 的 4 个 float) -> 找第一个不同的 batch; 拿到 (p, r, nt) 后只对那一批 printf 记录写/读值。
- **判读备忘**: 非记录批 (nt>t_cap) 上 p 可以 >0 (先 fold 上一记录批的 accepted 前缀再处理大批) 是设计内行为 ✓;
  非记录批不写记录、结束提交全程状态 ✓; 下一记录批 p 由 gen 校验 (非记录批 bump 不 commit -> p=0) ✓。
  若 plan B 的第一个坏 batch 落在这类转换上, 重点查该批 fold 的 **base (plane 0 是否 = 上一记录批提交值)** 与 read_half。

### 绝对证据: 读写地址一致但内存为 0 -> 记录写入未落盘/被覆盖 (2026-09-25)

read_half 修复后, 同一 layer 的读写地址已对齐:
```
T24WR lid=0  half=1 k0=-0x1.77276p-9 wk=1024598600   (base+half_stride, 写入侧探针在写块内, 证明 rec_write=true 且块执行)
T24IN lid=48 half=1 kk=1024598600 k0=0x0.0p+0 v0=0   (同一地址, 读出全 0)
```
=> **写入未持久化**。192 mismatch 全部集中在 lid 48..95 (= 第二个 batch 的 48 层), p=2/tn=2, fold=0。

**最后待查 (唯一剩下的问题)**:
1. batch1 写完后, batch2 读之前, 记录张量内容为何是 0?
   - 候选: 记录张量与某个每批被 clear/覆写的缓冲重叠? 图里是否有 op 把 state_zero/别的 cpy 写到记录张量上?
   - 候选: 写入的 *cell* 与读取的 *cell* 不同 (图侧 per-layer view 偏移 `kv_head*nb[2]` 与 kernel 里 `seq*nb2` 的组合);
   - 候选: 写入宽度/偏移 (wk[i] 的 i 范围, rows_per_lane/warp_size) 只写了别的元素。
2. 快速判据: 在 batch2 读之前用 host 侧 `ggml_backend_tensor_get` 把该记录张量的前几个元素抓回 host 打印;
   或直接在 kernel 的 **写块之后** 加 `__threadfence()` + 同批内再读一次 (若同批内也读到 0 -> 写入地址错;
   若同批读到非 0、下批读到 0 -> 被别的 op 清零/覆盖)。

### analyst 复核 (2026-09-25, 绝对证据后 - 写未持久化)

- **判读**: 同址写入 (探针值 -0x1.77276p-9) + 下批读出全 0 -> "store 未落盘/被清" 三类原因, 按概率:
  1. **两次 launch 间不是同一块内存** (指针数值相同但分配被复用/换过): 查 host 侧 `rec_l[il]->data` / `rec_view->data`
     每批是否稳定, 与 kernel 打印的 `rec.ptr` 对照; 并在 batch1 后、batch2 前用 host `ggml_backend_tensor_get`
     读回前几个元素 (你的判据 2)。
  2. **store 被丢掉**: `rec.rec` 是 `const float *` 且 `rec` 为 const 传值, 写路径是 const-cast; 一般可行但值得用
     `__threadfence()` + 同批写后回读确认 (你的判据 1)。同批回读也为 0 -> 地址算错或 store 被优化。
  3. **被别的 op 清零/覆盖**: 重点找"写 0"的 op —— `build_rs` 每层每批执行的 `ggml_scale_inplace(state_zero, 0)`
     (S 缓存 rs_z 行) 以及任何仍按 `(1+n_rs_seq)` 行数寻址的 view/cpy。**replay 模式 S 张量只有 mem_size 行**,
     若有路径仍按 (1+n_rs_seq) 行写 (尤其 debug 构建里 K=n_rs_seq 的快照写若走融合 cache 路径), 会越过 S 张量
     落进**同 buffer 相邻张量 = record 张量** (分配顺序 r, s, p?, rec)。检查法: dump record 张量前几个元素 +
     S 张量尾部几行, 看越界写痕迹; 核对 `rs_z/head/n_rs` 在 replay 下能否产生越界 view。
- 两个判据 (同批回读 / host 读回) 足够先二分 1+2 vs 3, 再按结果查 OOB。

### 情况评估 (2026-09-25, 用户询问)

**能不能修好: 能, 而且根因已高度收敛**。剩余问题只有一个具体事实: 记录张量写入不持久 (地址一致、写块执行过、下批读出 0),
192 mismatch 全集中在第二批 48 层。

**最可能根因 (analyst 候选 3 的具体化)**: 我加的**诊断性快照写** (replay 时 `+1` 平面偏移, 写 plane 1..K) 超出了
op 状态 **tail** 的分配范围 (tail 只按 1 个平面分配, 即 `S_v*H*n_tokens*n_seqs`), 越界写把相邻的**记录张量清零**。
-> 这是**诊断仪器的副作用**, 不是 T24 设计问题。

**一步验证**: 把诊断快照写入重定向到 S 缓存 (诊断期已扩到 5 平面) 或直接去掉诊断快照, 改用 plan B
(host 侧每批 dump S 缓存 plane 0 逐批比对) -> 若 192 消失且 ON == ref -> 修复完成。

**时间**: 1-2 个构建-运行周期 (~20-40 分钟) 可判定; 若仍有残余, 还有 3 条明确候选 + 3 个预置退路
(按 context 二分 / 放宽门槛 / 保留快照), 且**不修好也不影响交付** (T19-L 已冻结, OFF 无回归, 生产未动)。

**为什么耗了 4 小时 (复盘)**: (1) impl-DLL 重链陷阱 -> 垃圾指针污染多轮结论; (2) CUDA device printf 缓冲重复/乱序 ->
日志顺序对齐全部作废; (3) 自检标志放错位置 (init_batch 不生效) -> "0 mismatch" 空判。三条现已全部解决
(host fprintf 可信 / 地址已核对 / take-key 幂等)。

### 决定性: 写入落盘但跨 launch 丢失 -> 记录张量所在 buffer 可疑 (2026-09-25)

同 launch 写-回读探针 (哨兵 12345):
```
T24RB lid=0 save=0x0.0p+0 rb=-0x1.77276p-9   <- 我写的 12345 被真实写入的 k_reg[0] 覆盖 -> 真实写入落盘 OK
T24IN lid=48 ... kk=1024598600 k0=0x0.0p+0    <- 下一 batch 同一地址读出 0
```
=> **记录张量内容在两次 GDN launch 之间丢失**。S 缓存显然持久 (否则 OFF 路径不可能逐位正确)。
下一步: 对比 `rec_l[]` 与 `s_l[]` 的分配 (buft 类型/ctx_map/是否模型 buffer)。

### 再进一步: 写入落盘且批内持久; 跨批(第二 batch 读)为 0 (2026-09-25)

- 分配对比: `rec_l[]` 与 `s_l[]` 都用**同一个** `ctx_for_buft(buft)` 创建的 context/buffer (src/llama-memory-recurrent.cpp:118,131,143)
  -> 同 buffer 类型, 都是持久张量, **不是分配问题**; S 缓存持久 (OFF 路径正确) => 记录理论上也应持久。
- 哨兵实验细节: 同一 lid=0 探针出现 4 次 (save=0,0,0,12345) -> **我写的 12345 在后续执行中仍在** => 批内写入确实持久;
  每次 `rb` 都被真实写入的 `k_reg[0]` 覆盖 => 真实写入落盘。
- 但 batch2 (lid=48) 在**同一地址**读出 0。

**结论**: 写入落盘且批内持久, 但**跨到下一个 batch 时该地址变成 0** -> 有某个 op/地址计算在批间把那块内存清零。
**下一步 (最直接)**: 在 host 侧用 `ggml_backend_tensor_get` 在**每批结束后**把 `rec_l[0]` 前 8 个 float 抓回打印 (ON 运行),
即可看到**从哪一批开始变 0**; 再对照该批前后的 op (state_zero / cpy / view 偏移) 定位清零者。

### 反转: host 侧 dump 证明记录**跨批持久** (2026-09-25)

host 侧每 ubatch 开头 dump `rec_l[0]` 前 3 个 float (偏移 0 = half 0):
```
gen=0 pos=0 : 0 0 0        (batch1 前)
gen=1 pos=0 : 0 0 0        (batch2 前) <- batch1 写的是 half 1, 所以 half 0 为 0, 正常
gen=2 pos=16: 0 0 0        (batch3 前, 非记录批)
gen=3 pos=20: 非零         (batch4 前) <- batch3 写了 half 0 -> 可见
gen=4 pos=21: 同 gen=3     (batch5 前, 写 half 1 -> half 0 不变, 正常)
gen=5..11  : 每两批更新一次 (双缓冲交替, 完全符合预期)
```
=> **记录写入落盘且跨批持久, 双缓冲交替正确**。"记录被清零"假设**被推翻**。
=> 那么 kernel fold 在 batch2 读到 0 的原因只能在这几类里:
  (a) fold 读的 launch 与写不是同一个 memory/context (例如草稿 context 的 records 本来就是空的);
  (b) 读的半区/偏移与写不同 (需 dump **half 1 区域** 对照, 即偏移 half_stride*4 字节);
  (c) 设备可见性/顺序问题 (kernel 读到旧值)。
**下一步**: 把 host dump 扩成同时读 half 0 与 half 1 (偏移 0 与 `T_cap*n_rec*4`), 逐批对照 kernel 侧 T24IN 的读数,
即可判定 (a)/(b)/(c)。

### 决定性: 只有 **batch 1 的写入丢失**, 之后各批全部持久 (2026-09-25)

host dump 双半区:
```
gen=1 pos=0 : h0=0 h1=0        <- batch1 记了 nt=2 hr, 但 half1 仍为 0 (写入丢失)
gen=3 pos=20: h0!=0 h1=0       <- batch3 写 half0 可见
gen=4 pos=21: h0!=0 h1!=0      <- batch4 写 half1 可见
gen=5..     : 正常交替持久
```
=> **只有第一次执行的写入丢失**; 后续批次写入/持久/双缓冲全部正确。
最可能: 记录张量的 data 指针在首个 graph 之后发生变化 (buffer 重分配/移动), 首写落在旧地址。
**下一步 (一步判定)**: 在 dump 里同时打印 `rec_l[0]->data` 指针, 若 gen=0/1 与 gen>=3 不同 -> 确认重分配;
修法: 确保记录张量在首次 decode 前完成分配 (或在重分配后重置 rec 状态)。

### 关键区分: 目标 context 记录持久 OK; 丢写的是另一 context (草稿) 的首批 (2026-09-25)

新增 dump 指针后:
- `ptr=1024578000` **全程不变** -> 无重分配 (该假设排除);
- T24RB lid=0: `save=0x1.81c8p+13 (=12345, 我上次写的哨兵)` -> **该地址的写入跨执行持久** (!),
  说明地址 1024598600 本身没问题;
- dump 序列 `gen=0 pos=0 / gen=1 pos=0 / gen=2 pos=16 / gen=3 pos=20 / gen=4 pos=21 ...`
  -> **两个 context 的序列交错** (两个 prefill 都从 pos=0 开始): 其中 pos=16/20/21/24 的**目标侧记录全部持久 ✓**
  (gen=3 h0 非零 = 目标 prefill 尾块 nt=4 的写入, 可见);
- 而 T24IN (lid=48, tn=2/p=2) 读到 0 的那个 context = **草稿** (它的批次是 2~3 token), 其 dump 行 `gen=1 pos=0 h1=0`
  = **草稿首批 (2 token, 记录批) 写 half1 后为 0 -> 草稿侧首批写入丢失**。

**结论**: 目标侧记录链路完全正确; 问题只在**草稿 context 的首个记录批**。下一步: 对草稿 context 单独打点
(prefill 16+4 与后续 2-token 批的 gen/half/写半区序列), 查它首批写 half 与读 half 的对应关系
(可能草稿的首批在 prefill 之后, 但 rec_half 状态被 prefill 的非记录批影响)。

## Final result: T24 判定为 FAIL (不发货), 2026-09-25

### 本轮决定性证据 (host 侧读回 kernel 实际所见, 绕开不可靠的 device printf)
在 fold 张量里加回写槽: kernel 每个 launch 的 block(0,0,0)/thread0 写 `fold[4] = wh | rd<<8 | nt<<16 | wr<<24`, `fold[5] = lid`;
host 在每次 take 读回并打印 (T24KR):
```
take b2:  saw_wh=0 rd=0 nt=0 wr=0 lid=0     <- b1 的 48 个 launch 没留标记
take b3:  saw_wh=1 rd=1 nt=16 wr=0 lid=95   <- b2 最后 launch (48..95), 参数完全正确
take b4:  saw_wh=0 rd=1 nt=4  wr=1 lid=143  <- b3 (96..143) 正确
take b5:  saw_wh=1 rd=0 nt=4  wr=1 lid=191  <- b4 正确
```
=> **从 b2 起, kernel 看到的 fold 参数 (写半区/读半区/nt/rec_write) 与 host 完全一致且交替正确**;
   lid 步长恒为 48 (每批 48 个 GDN op) => 批次数与 launch 数自洽; 只有 b1 的标记读回为 0 (可能 host 异步读滞后,
   也可能 b1 走的是旧非 replay 路径), 而 b1 的 fold p=0, 记录缺失对后续无影响 (b2 的 fold 退化为 no-op, 结果恰好正确)。

### 状态轨迹逐项复核结论 (全部自洽)
- b1(nt=2,记录): 发布"起始态"(pinned), 记录写 half1;
- b2(nt=16,不记录): fold p=2 读 half1 => no-op; 主循环后发布 after-batch (走 `!rec_write` 分支) => 与 OFF 的 snapshot 语义一致;
- b3(nt=4,记录): fold p=0 (记录 stale, 正确跳过), 发布 pinned; b4 fold p=4 从 pinned 重放 b3 四条记录 => "after b3"
  = OFF 的 slot0 => 数值路径相同 (fold 与主循环共用同一 `gdn_state_update`);
- 生成期 (n-max=3): p ∈ {1,3,4} 的 fold 全部落在"上一个批次的 accepted 前缀"上, 与 pos0 序列 (20,21,24,28,...) 自洽。
=> **未找到状态/半区/记录语义上的错误**; n-max=1 时 ON==OFF 逐位一致, n-max=3 时仅在 index 44 差一个 token
   (198 vs 220), accepted/draft_n 与 ref 相同 => 典型的"末位数值差异被采样放大" (见 MTP封存文档)。

### 判定理由与决定
- 无法做到逐位一致 (MTP n-max=3 场景), 无法用现有证据证明"绝对不降智";
- 因此 **T24 不发货**: 保持 `GGML_CUDA_GDN_REPLAY` 默认关闭, 生产行为与 T19-L 交付版本完全一致 (零风险);
- 代码与诊断全部留在工作区 (uncommitted), 供后续会话继续; 结论 + 全部否证/证据链在本文档。

### 已确认否证 (本轮, 不要再试)
- 记录被清零 (host 全层双半区 dump: 写入/持久/双缓冲交替均正确);
- 张量重分配 (ptr 全程不变);
- host 读滞后 (150ms 重读完全一致);
- 快照/诊断平面越界; 代码生成/FMA 差异 (env OFF 带 fold 代码逐位一致);
- 草稿 context 影响 (其 GDN record 张量根本没被使用: MTP 草稿图只跑 MTP head, 无 delta-net op)。

## 判据修正 + 新证据: 以"不开 MTP 的原始模型"为基准 (用户 2026-09-25)

用户判据: 不要求与老 MTP 逐位一致 (老 MTP 本身可能与原始模型不同); 只要实现正确; 能与不开 MTP 对齐更好。
实测 (同 prompt/贪心/seed42/64 token):

| 运行 | 与 base (无 MTP) 对比 |
|---|---|
| 老 MTP (OFF, snapshot) | **64/64 完全一致, 无首个差异** |
| Replay MTP (ON) | **63/64, index 44 起不同 (220 vs 198)** |

=> 老 MTP 在本场景就是"正确基准"; **T24 replay 实现仍然错误**, 不能发货。

### 已定位到的新嫌疑 (未验证完)
delta-net 图仍跑上游 `build_rs` 状态机制, replay 模式只改了一半:
- `s_copy()` 已改为选择 row 0 (pinned 态) ✓
- 但 `rs_zero = get_rs_z()` (= `is_full ? 0 : mem->rs_z`) 仍会执行 `state_zero` (`ggml_scale_inplace(...,0)`)
  清 row `rs_zero` -> **可能把 pinned committed 态 (plane 0) 清零**, 在 fold 之前;
- `get_head()`/`get_n_rs()` 也仍是上游语义。
=> 下一步: 在 replay 模式下把 `state_zero` 关掉 (rs_zero 传 -1 或指向诊断平面), 并核对 `is_full` 触发时机;
   然后用 base(无 MTP) 64/64 作为最终验收判据。

### 现状
- 生产/交付不受影响: `GGML_CUDA_GDN_REPLAY` 默认关闭 => 走老 MTP (== base 64/64);
- T24 代码与全部诊断仍在工作区 (uncommitted), host 侧工具链 (T24ALL/T24SLP/T24KR/T24PATH/T24ENT) 齐全。

## 定位到最可能的具体根因: full batch 的 state_zero + fold 叠加 (2026-09-25 深夜)

上游 `build_rs` 会执行 `state_zero`: 对 `rs_zero = get_rs_z()` 行做 `ggml_scale_inplace(...,0)`;
`get_rs_z() = is_full ? 0 : mem->rs_z` (llama-memory-recurrent.cpp:1339), 而上下文构造里 `is_full(true)` (同文件:1286)。
上游语义: 当 batch 为 "full"(>= n_rs 槽) 时, 状态被清零并从零重算整批 —— **这正是 b2(16 token) 与 base/老MTP 一致的原因**
(二者都在 b2 把 row0 清零后从零跑 16 个 token)。

**但我们的 replay 在 b2 仍然 fold p=2 (b1 的 warmup 记录)**: 从**被清零的**状态上重放 2 条记录 =>
多加了 2 个 token 的错误贡献 => 状态出现小扰动 => 在 index 44 的近似并列采样处翻成另一个 token。
这解释了: n-max=1 逐位一致 (那条路径上没有这种 full-batch + p>0 组合), MTP3 确定性差 1 token, 老MTP==base 64/64。

**下一步 (一行级修复, 下轮先验证再改)**:
- 在 `get_rec_p()` 中, 当 batch 为 full (上游会清零 state) 时返回 p=0 (不 fold); 等价于 "被清零的状态不能 fold 旧记录";
- 同时核对 `is_full` 的真实语义/设置点 (是否恒 true), 以及 `get_head()/get_n_rs()` 在 replay 下的取值;
- 验收: 无 MTP base 64/64 + 在位 fold-vs-snapshot 自检 0 mismatch + n-max=1/3 逐位。

## T24 最终结果: DONE - ReplaySSM 与"不开 MTP 的原始模型"逐位一致 (2026-09-26)

### 两个真正的根因 (都会导致 index 44 那个确定性分叉)
1. **重启批不该 fold**: b2 (真实 prefill, pos 0) 重新处理了 warmup (b1) 已处理的 pos 0..1,
   此时状态链在内存侧是"断开/全新"的,正确的语义是从**清零的 fresh 行**开始整批重算 (这也解释了
   老 MTP/base 在 b2 把整批从零跑的行为)。而 `get_rec_p` 仍按 `rec_n - rollback` 给出 p=2,fold 在
   *被清零的* 状态上重放了 2 条记录 => 多算 2 个 token。
   修复: `get_rec_p(i, pos0)` 增加"连续性"判据 - 仅当 `pos0 == rec_pos0[seq] + p` 才 fold,
   否则 p=0 (rec_commit 现在记录每批首 token 位置 `rec_pos0`)。
2. **`s_copy` 覆盖破坏了 conv 的 rollback**: 我曾把 `s_copy` 改成 replay 下恒返回 0,
   但 conv cache (`r_l`) 也走同一个 `s_copy` 取回滚索引 => conv 的历史不再回滚,把被拒绝草稿的
   原始输入也算进去了 (状态仍对, 但 q/k/v 有微小漂移 -> 末端采样翻转)。
   修复: 还原 `s_copy` 为上游公式; conv 抓取 (`s_copy_conv`, 本次新增的本份) 在状态 gather 之前
   调用, 拿到回滚感知的行号; 状态 gather 因为 `s_copy` 内部会 reset `rs_idx`, 自然拿到 row 0 (pinned 面)。

### 最终验收 (全部通过, 2026-09-26, 最终构建)
| 项目 | 结果 |
|---|---|
| MTP3 replay(ON) vs base(不开 MTP) | **64/64 逐位一致** |
| MTP3 OFF vs base | 64/64 逐位一致 |
| n-max=1: ON vs OFF | 64/64 逐位一致 |
| 不开 MTP: ON vs OFF | 64/64 逐位一致 |
| PPL (512/8chunk/seed42) | **ON = OFF = 4.3567** (= T19-L 基线, 逐位相同) |
| fold 自检 (opt-in, 全层全批 fold vs 上批提交态) | **0 mismatch** |
| VRAM @MTP3 | **-420 MiB** (21598 vs 22018 MiB) |
| tg @MTP3 (交错 4 次平均) | -1.9% (71.6 -> 70.2 t/s) |

### 最终形态
- S 缓存 replay 模式下只有 **1 个 committed 平面** (`n_rows_s = mem_size`), 记录张量存 raw inputs;
- 开关: `GGML_CUDA_GDN_REPLAY=1` 启用 replay (默认关 = 老 MTP 行为); 
  `GGML_CUDA_GDN_REPLAY_CHECK=1` 额外启用"fold vs 上批提交态"逐位自检 (默认关, 自检会多 48MB/layer 的拷贝);
- 所有 T24BISECT 诊断已清理; 最终形态 = 提交 `3eae5cdae` (11 文件, +920/-78; 四提交 squash 后已推 fork origin/master);
  patch 归档 `artifacts/t24-replayssm-final-3eae5cdae.patch` (68,494 B, 反校验 OK)。

### analyst 验收复核 (2026-09-26, squash+push 后复查)

- **证据链可信**: 判据已换到 host 侧读回 (绕开 device printf) + opt-in 自检 (全层全批 0 mismatch) + PPL 与 T19-L 逐位相同
  -> 测量仪器问题全部解决; "与 base 逐位" / "ON==OFF 逐位" 在验收场景成立。
- **性质提醒**: 这是**权衡**不是纯增益 —— 换显存 (np=1 -420MiB; np=4 -1.37GiB), 代价 **tg -1.9%** (71.6 -> 70.2 @MTP3);
  开启与否按 VRAM/速度偏好决定 (开关默认关 = T19-L 行为, 零风险)。
- **覆盖范围 (最终)**: MTP n-max 1/2/3 x np 1..4 + 并发 + 长程 (np1 1000 / np2 4x250x2 / np4 4x125x4) + session save/restore
  + PPL 4.3567; 另一模型 Qwen3.6-35B-A3B + DFlash (n-max=6) x np 1..4 + 生产采样 (draft/accepted 统计相同);
  并发场景 token 级不可跨运行比对 (OFF 自身随批合并时序分叉), 以自检为准。
- **"与不开 MTP 一致" 的范围**: 结论基于逐 token 贪心验收; T20 已知长跑/近似并列点仍可能出现翻转 (FA 侧 S1/S3 未动),
  不是普遍逐位结论; 要更强保证需长文/needle 回归 (T27)。
- **状态**: 四提交 squash 为 `3eae5cdae` 已推 fork origin/master (analyst 核实 HEAD == origin/master); 最终构建 `llama.cpp-t24`
  21:08 (`ggml-cuda.dll` = `299DAFC748DE5D91`; 源码末改 20:58 < 构建 21:08 -> 与该提交对应); patch 重生成并反校验 OK。
- **状态更新 (2026-09-26)**: 部署已完成 (用户批准): 生产 `llama.cpp-my` = `299DAFC748DE5D91` + `GGML_CUDA_GDN_REPLAY=1` (用户级);
  备份 `deploy-backup\llama.cpp-my-t19l-20260926\` (T19-L `054BFFD6` 9 文件, 可回滚); 部署后 PPL 冒烟 4.3567。
  **仅剩**: 是否纳入交付基线 (交付快照/patch 集/文档)。
- 未覆盖: EAGLE3/DSpark/KDA (无权重); np>=5; 无 SWA hybrid + 极端小裁剪 (理论边界)。

### 后续发现 (2026-09-27, T32 冒烟测试附带; 待立案)
- 回滚 + replay 路径存在**运行间不确定性**: ref-vs-ref (无共享, 同配置跑两遍) 亦偶发 value-diff
  (logits ~0.17-0.25, 复现率 ~1/6; token 翻转仅出现在 top-2 差 ~0.003 的近并列点);
  **replay=0 时 6/6 逐位一致; rb<=2 无损**; 生产配置 (n_rs_seq=3, rb=3) 6 次无 token 翻转, 1/6 value-diff;
  `GGML_CUDA_GDN_REPLAY_CHECK=1` 未报 (自检未覆盖此场景)。
- 复现: `test-t32-smoke -m <model> --nrs 3 --rb 3` (replay=1, 需 GPU); 待用户决定是否立案复查
  (生产已开 `GGML_CUDA_GDN_REPLAY=1`; temp 1.0 采样下 token 翻转概率低, 但已知不确定性应记录)。
