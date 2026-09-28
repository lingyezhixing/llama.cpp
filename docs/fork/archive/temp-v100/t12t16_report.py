import io

# ---------- RESULTS ----------
t = """
---

# T12+T16 合并验收 (2026-09-23, implementer): **全部门槛 PASS, 已采纳入库**

构建: `453E29111E5E29C9...` (T12 stream-K 启发式 + T16 PB=2); 工作区 5 文件; patch `patches/v100-t12t16-fattn-split.patch`
A/B: A = BASE 交付版 (`102BF844`), B = T12+T16; 同 session, 长点按轮次**轮换 A/B 顺序** (消除热漂移系统性偏差)

| 门槛 | 要求 | 实测 (A -> B) | 判定 |
|---|---|---:|---|
| **pp8192@depth128k (主判据)** | >= +4% | 337.39 -> **375.45** = **+11.28%** | **PASS** (超 2.8x) |
| depth32k (`-p 4096 -d 32768`) | >= +1.5% | 651.10 -> 682.16 = **+4.77%** | **PASS** |
| pp32768 | >= +0.8% | 774.19 -> 791.07 = **+2.18%** | **PASS** |
| pp512 | |Δ| <= 0.3% | 945.5 -> 950.8 = +0.56% (B 更快) | PASS |
| pp4096 | |Δ| <= 0.3% | 932.2 -> 932.5 = +0.03% | PASS |
| pp8192 | |Δ| <= 0.3% | 909.2 -> 912.1 = +0.32% (B 更快) | PASS |
| tg128 | 噪声内 | 26.667 -> 26.682 = +0.06% | PASS |
| ub2048 (抽查) | 不回退 | 1151.4 -> 1149.9 = -0.1% | PASS |
| PPL | 4.3572 ± 0.013 | **4.3562** | PASS |
| 200 token 生成 | 无乱码/重复 | 连贯 (Flash Attention 技术总结), 无重复行 | PASS |
| 显存增量 | 记录 | fixup partials = `nblocks*64*130*4B ≈ 12.8MB`/launch (CUDA pool 复用); 128k 实跑无 OOM | PASS |

细节:
- 128k 点: 单轮 (用户确认前两阶段波动 <0.5% 后停测, 省 20+ min 重负载): A 337.39 / B 375.45
- 长点稳定性: depth32k A 650.0-652.8 / B 680.9-682.9; pp32768 A 771.8-777.8 / B 787.8-795.1 -> 轮间 <0.5%
- 短点 3 轮: pp512 A 929.4/953.1/954.2 (r1 低离群), B 954.9/952.0/945.6
- **热漂移**: 本机为笔记本机箱, 连续负载下 tg128 从 26.6 逐步掉到 24.4 (A/B 同步) -> 长点必须轮换顺序; tg128 采用冷却后复核值
- **数值**: PPL 4.3568 (T12 单独) -> 4.3562 (PB=2 后) = -0.0006 漂移, 来源 = KV 切 2 段后 softmax 归并顺序变化 (与 GDN vec4 同类, 良性; |diff| 远小于门槛 0.013); 生成质量检查通过
- 采纳后当前最快: pp512 950.8 / pp4096 932.5 / pp8192 912.1 / pp32768 791.1 / depth32k 682.2 / **pp8192@depth128k 375.5** / tg128 26.68 / PPL 4.3562
"""
io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

# ---------- TASKS/T12 ----------
p = r'D:\LLM\Backend\v100-collab\TASKS\T12-t11-adopt.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## Result (2026-09-23, implementer): **PASS -> 采纳入库**

- 合并验收 (A=BASE `102BF844`, B=T12+T16 `453E2911`, 长点轮换顺序): **pp8192@depth128k +11.28%** (主判据 +4%),
  depth32k **+4.77%** (+1.5%), pp32768 **+2.18%** (+0.8%), 短点 +0.03~0.56% (无回退), tg128 +0.06%,
  ub2048 -0.1%, PPL **4.3562** (门槛 4.3572±0.013), 200 token 生成无重复/乱码
- 显存: stream-K/PB=2 fixup partials ≈ 12.8MB/launch (pool 复用); 128k 深度实跑无 OOM
- 交付: patch `patches/v100-t12t16-fattn-split.patch` (T12+T16 合并, 双向 apply 校验通过);
  工作区 5 文件; DLL `453E29111E5E29C9`; env 开关保留 (STREAM_K/BLOCKS/PB, 便于 T18 与回归)
- 注: T11 原 REJECTED patch 被本合并 patch 取代 (artifacts 保留历史)
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T12 updated')

# ---------- TASKS/T16 ----------
p = r'D:\LLM\Backend\v100-collab\TASKS\T16-parallel-blocks.md'
s = io.open(p, encoding='utf-8').read()
s += """

## Result (final, 2026-09-23): **PASS -> 采纳 (PB=2), 与 T12 同一 patch 入库**

- 实现: mma 内核无 parallel_blocks 路径 (纯 stream-K, 只读 blockIdx.x) -> 等价实现 = 覆盖 `blocks_num.x`;
  `blocks_num.x = max(nblocks_stream_k, min(ntiles_KV*ntiles_dst, 2*ntiles_dst))` (PB=2 + 保底, 保护 decode)
- 扫测定论: PB=2 最优 (attention -14.7% vs T11-off / -6.7% vs T12 默认); PB=4 过切; 占用/更多 warp 全否证
- 验收 (合并 A/B): **pp8192@depth128k +11.28%** (主判据 >=+4%); depth32k +4.77% (副判据 >=+2.5%);
  pp32768 +2.18%; 短点无回退; PPL 4.3562 + 生成 OK
- 换算 attention: ~34.5 -> ~38 TF/s 级 (T12+T16 组合 vs BASE), 接近 analyst 的 V100 天花板估 (40-45)
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T16 updated')

# ---------- QUESTIONS ----------
q = """
---

## 2026-09-23 | T12+T16 合并验收: 全部门槛 PASS, 提请 VERIFIED | implementer

验收数据见 RESULTS "T12+T16 合并验收"。一行速览 (A=BASE -> B=T12+T16):
- **pp8192@depth128k 337.39 -> 375.45 = +11.28%** (主判据 +4%, 超 2.8x)
- depth32k 651.10 -> 682.16 = **+4.77%**; pp32768 774.19 -> 791.07 = **+2.18%**
- 短点 +0.03~0.56% (B 略快, 无回退); tg128 +0.06%; ub2048 -0.1%; PPL 4.3562; 生成 OK
- 128k 单轮: 你给的前两阶段都只有 <0.5% 轮间波动, 用户据此叫停了 128k 的剩余轮次 (省 20+ min 重负载);
  若你要 2 轮, 请说明 (需额外 ~10 min 重负载窗口)

提请: (a) T12/T16 标记 **VERIFIED**; (b) patch 已归档 `patches/v100-t12t16-fattn-split.patch`
(含 T12 启发式 + T16 PB=2 + 3 个 env 开关, 双向 apply 校验); (c) 后续按 BOARD 队列 = T17 测量 -> T18 Stage 0。

数值注: PPL 4.3568 -> 4.3562 (-0.0006), 来源 = PB=2 的 softmax 归并顺序 (KV 切 2 段), 与 GDN vec4 同类良性;
生成质量检查 (200 token) 无重复/乱码。
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(q)
print('QUESTIONS appended')

# ---------- BOARD ----------
p = r'D:\LLM\Backend\v100-collab\BOARD.md'
s = io.open(p, encoding='utf-8').read()
old12 = "| T12 | 采纳 T11: KV-split 重放集成 | **RUNNING (实现+PPL 通过; 完整 A/B 待跑)** | patch 已重放, PPL 4.3568, nsys 机制复现 (grid 192->80 + fixup 30.2ms); 验收点: depth32k >=+1.5% / pp32768 >=+0.8% / pp8192@depth128k >=+2% / pp512-8192 / tg128 / PPL; 与 T16 合并为一次 A/B (A=BASE, B=T12+T16) |"
new12 = "| T12 | 采纳 T11: KV-split 重放集成 | **DONE -> 待 analyst VERIFIED** | 合并验收全过: **pp8192@depth128k +11.28%** / depth32k +4.77% / pp32768 +2.18% / 短点 +0.03~0.56% / tg128 +0.06% / ub2048 -0.1% / PPL 4.3562 / 生成 OK; patch `v100-t12t16-fattn-split.patch` |"
assert old12 in s
s = s.replace(old12, new12)
old16 = "| T16 | ub512 attention parallel_blocks | **RUNNING (实现完成 + PB=2 胜出; 验收待跑)** | **发现: mma 内核是纯 stream-K (只读 blockIdx.x), 无 parallel_blocks 路径** -> 等价实现 = 覆盖 blocks_num.x; 扫测定论: **PB=2 (grid=2*ntiles_dst) -14.7% vs T11-off / -6.7% vs T12 默认**, PB=4 过切, 占用/更多 warp 路线全部否证 (spill); 主判据 pp8192@depth128k 待验收 |"
new16 = "| T16 | ub512 attention parallel_blocks | **DONE -> 待 analyst VERIFIED** | mma 内核无 parallel_blocks 路径 (纯 stream-K) -> 等价实现 = 覆盖 blocks_num.x; **PB=2 -14.7% vs T11-off / -6.7% vs T12 默认**; PB=4 过切, 占用/更多 warp 全否证; 验收 (合并) 全过: **128k +11.28%** 等; 与 T12 同一 patch 入库 |"
assert old16 in s
s = s.replace(old16, new16)
oldq1 = "| 1 | **T12+T16 合并验收** (spec: TASKS/T12, TASKS/T16) | depth32k +2.0% / pp32768 +1.1% / pp8192@depth128k 待测; T16 部分 = PB=2 再快 6.7% | 0.5 天 (夜间断点: 待用户许可时段) | **RUNNING** |"
newq1 = "| 1 | ~~T12+T16 合并验收~~ | 实测 **128k +11.28% / depth32k +4.77% / pp32768 +2.18%** (全过) | 0.5 天 | **DONE (待 VERIFIED)** |"
assert oldq1 in s
s = s.replace(oldq1, newq1)
oldq3 = "| 3 | **T16 parallel_blocks** (spec: TASKS/T16) | PB=2 实测 attention -14.7% vs T11-off (估值 ~34.5 TF/s); 验收并入序 1 | 1 天 (已用 ~0.5 天) | **RUNNING (实现完成)** |"
newq3 = "| 3 | ~~T16 parallel_blocks~~ | PB=2 采纳, 验收并入序 1 (全过) | 1 天 | **DONE (待 VERIFIED)** |"
assert oldq3 in s
s = s.replace(oldq3, newq3)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('BOARD updated')

# ---------- ENVIRONMENT ----------
e = """
## 交付快照更新 (2026-09-23, T12+T16 采纳后)

| 项 | 值 |
|---|---|
| 工作区 | `D:\\LLM\\Backend\\src\\llama.cpp-my` **5 个已修改文件**: 原 4 文件 + `fattn-common.cuh` (T12+T16) |
| patch | `patches\\v100-{dequant-vec,gdn-vec4,t07-silu-vec4}.patch` + **`v100-t12t16-fattn-split.patch`** (双向 apply 校验通过; T11 原 REJECTED patch 被取代, artifacts 留档) |
| 部署 DLL | `D:\\LLM\\Backend\\llama.cpp-my\\ggml-cuda.dll` SHA256 `453E29111E5E29C9...` (T12+T16; 每次构建后核对, 有硬链接疑点) |
| 验收 (本次) | **pp8192@depth128k 375.45 (+11.28%)** / depth32k 682.2 (+4.77%) / pp32768 791.1 (+2.18%) / pp512 950.8 / pp4096 932.5 / pp8192 912.1 / tg128 26.68 / PPL **4.3562** |
| env 开关 (保留) | `GGML_CUDA_FATTN_STREAM_K` (0/1), `GGML_CUDA_FATTN_BLOCKS=N`, `GGML_CUDA_FATTN_PB=N` - 实验/回归用, 未设时走新默认 (stream-K 启发式 + PB=2 + 保底) |
| 实验 DLL 备份 | `%TEMP%\\v100\\ggml-cuda-T12-BASE.dll` (交付前基线), `ggml-cuda-T12T16.dll` (当前采纳版), 其余 8 个实验变体 |
"""
io.open(r'D:\LLM\Backend\v100-collab\ENVIRONMENT.md', 'a', encoding='utf-8', newline='').write(e)
print('ENVIRONMENT appended')
