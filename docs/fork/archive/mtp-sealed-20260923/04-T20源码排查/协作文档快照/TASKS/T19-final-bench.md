# T19 (v2): 同步最新上游 + 公平对照 + 交付定稿 (用户口径修订 2026-09-23)

状态: **ISSUED (analyst v2)**; 执行者: implementer; 用户已下达 (GPU 长时任务, 按用户窗口执行)
v2 变更: 对照不再用"不同 commit"直接比; 改为 **两边统一同步到同一最新 base** 后各自编译 (用户指定)

## 0. 口径 (用户指定, 不要改)

| 角色 | 源目录 | 部署目录 | 远端 |
|---|---|---|---|
| **STOCK (官方原版)** | `D:\LLM\Backend\src\llama.cpp` | `D:\LLM\Backend\llama.cpp` | origin = ggml-org/llama.cpp |
| **OURS (我们的修改版)** | `D:\LLM\Backend\src\llama.cpp-my` | `D:\LLM\Backend\llama.cpp-my` | origin = lingyezhixing/llama.cpp, upstream = ggml-org/llama.cpp |

- **公平对照定义**: 两边同步到**同一 NEWBASE** (最新 upstream master) 后各自编译;
  OURS = NEWBASE + fork 本地提交 (sm70 调参 + 交付提交), STOCK = NEWBASE
- **不 push**: 只做本地 commit / fetch / rebase / 构建 / 部署
- fork 当前本地提交: `4046f5c8c` (sm70 FATTN 调参); 工作区 5 文件 = 未提交的交付改动

## 1. Phase A: 整理 + 本地提交

- 清理 5 文件 diff 注释: 保留非显然不变量 (16B 对齐回退 / Q5_K 布局 / KDA 排除原因 / 行所有权映射 / PB+stream-K 语义);
  删冗余/教程式/重复; 英文, 1-2 行, 不硬折行 (AGENTS.md 风格)
- env 开关保留 (`GGML_CUDA_FATTN_STREAM_K` / `_BLOCKS` / `_PB`), 注释精简
- **零功能改动**: 自查 `git diff` 只允许注释行变化
- 提交前快速 sanity: 重建 ggml-cuda + PPL + pp512 (确认清理未破坏; 数值会随后续同步再测)
- 本地 commit (1 个提交, 5 文件): 消息示例
  `ggml-cuda : V100 optimizations (K-quant vec dequant, GDN vec4, silu vec4, FATTN stream-K + PB=2)`
- 记录提交哈希到 ENVIRONMENT

## 2. Phase B: 同步最新上游 + rebase

1. vanilla: `git fetch origin` -> ff-only 到最新 master -> 记录 **NEWBASE 全哈希**
2. fork: `git fetch upstream` -> `git rebase NEWBASE` (重放 2 个本地提交: sm70 调参 + 交付提交)
   - **重复检测**: 若上游已含 sm70 FATTN 调参 (等效提交或同代码) -> 该提交 rebase 会空/冲突;
     确认后丢弃 (`git rebase --skip`) 并记录"上游已含"
   - 冲突重点: `fattn-common.cuh` (上游活跃), `gated_delta_net.cu`, `convert.cu/dequantize.cuh`
   - **若某项改动与上游重复或无法安全 rebase -> 停下报告 analyst, 不硬改**
   - 完成后: `git log --oneline -5` 记录; 工作区干净; 本地提交数 = 上游没有的部分
3. 重建 fork (新 base) -> PPL (门槛 4.3572±0.013 需按新 base 复核) + pp512/tg128 sanity + 快速 A/B
   (新 base 可能含上游 CUDA 改动, 绝对值会变; 记录新基线)
4. 重建 vanilla (同一 NEWBASE) -> 同样 sanity
5. 部署: vanilla 构建输出 -> `D:\LLM\Backend\llama.cpp`; ours -> `D:\LLM\Backend\llama.cpp-my`;
   每边核对**部署 DLL SHA == 该边 build 输出 SHA** (ours 防硬链接陷阱)
6. 重新生成 4 个 patch (基于 NEWBASE: `git diff NEWBASE..<交付提交> -- <文件组>`) + 双向 apply 校验
7. 记录: NEWBASE / fork HEAD / 两边 DLL SHA / PPL / sanity 到 ENVIRONMENT

## 3. Phase C: 基准矩阵 (OURS vs STOCK, ub512 与 ub2048 各一遍; 同 session 配对; 每点 r>=2)

- 固定: `-m <model> -ngl 99 -fa on -ctv q8_0 -ub {512,2048}`, `CUDA_VISIBLE_DEVICES=1`;
  每臂从各自部署目录运行 (ours: `llama.cpp-my`; stock: `llama.cpp`), 运行前核对 DLL SHA
- pp: `-p 512,4096,8192 -n 0 -r 3` / `-p 32768 -n 0 -r 2` / `-p 131072 -n 0 -r 2`
- tg: `-p 0 -n 128 -d 0,4096,8192 -r 3` / `-d 32768 -r 2` / `-d 131072 -r 2`
- 热漂移协议: 长点必须轮换臂顺序; 禁止跨 session 混比
- OOM 预案: ub2048 @131072 若 OOM -> 降到 98304 (或实际最大可行点), 图表/表格明确标注, 不静默丢点
- 原始输出全部落 RESULTS 新章节 "T19 全曲线对照" (命令 + 每点 r 次原始值 + 均值)
- 顺序建议: 短点 -> 32768 -> 131072 (131072 最后)

## 4. Phase D: 拟合绘图 (python 3.12 + matplotlib 3.10 已确认可用)

- 图1 `t19_pp.png`: x=pp tokens (log2), y=pp t/s; 2 子图 (ub512/ub2048); OURS/STOCK 散点 + 拟合曲线 (log-x 平滑单调)
- 图2 `t19_tg.png`: x=depth (log2), y=tg128 t/s; 同上
- 图3 `t19_delta.png`: OURS vs STOCK 每点 +% 柱状 (两 ub 并列)
- 数据表 `t19_data.csv` + 脚本 `artifacts/t19_plot.py` (可复现); PNG 300dpi -> `artifacts/`
- 可选归因臂 CLEAN (仅当需要拆 sm70 与交付补丁时): fork @ NEWBASE + sm70-only 构建

## 5. Phase E: 质量差异

- PPL: OURS / STOCK 同文本 `-c 512 --chunks 8 --seed 42` (预期 OURS 4.3562 级, 按新 base 复核)
- 生成: 同 prompt / `--seed 42` / temp0 / 200 token, OURS vs STOCK (同一 NEWBASE, 最干净): 逐字对比 + token 一致率
- 结论段: 若 PPL 差 <=0.001 且生成一致 -> "无降智证据"; 有差异给证据与解释

## 6. 交付物

- 整理后的 5 文件 + 本地提交 (rebase 后) + 4 patch 重生成 + 两边部署 (SHA 核对)
- RESULTS "T19" 章节; artifacts 3 图 + CSV + 脚本; 质量分析段
- 更新 BOARD / STATUS / ENVIRONMENT (NEWBASE, fork HEAD, 各臂 SHA)

## 7. 时间预算与安全

- Phase A ~20min; Phase B (fetch/rebase/双构建/双部署/校验) ~1-1.5h; Phase C ~2-3h (131072 占大头); D+E ~1h
- 不动 4060; 长跑注意散热与臂顺序轮换; 用户夜间静音规则仍适用; 网络操作 (fetch) 前后注意冲突备份 (`git stash` 不必要, 工作区已提交)
