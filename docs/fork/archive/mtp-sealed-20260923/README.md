# MTP 一致性试验封存（2026-09-23）

- 环境：Qwen3.8-27B-UD-Q6_K / 单卡 V100-SXM2-32GB / llama.cpp（本 fork）
- 起因：群友反馈"27B 开 MTP 明显降智"
- 状态：**2026-09-26 已收口**。续篇见 `05-T24-ReplaySSM/`：
  - **状态/回滚轴**：用 ReplaySSM（T24）做到"老 MTP == 新 MTP 逐 token 相同"，属无损重构，
    省 420MB VRAM（提交 `c16b1a4c8`，本地未 push；开关默认关）；
  - **选核轴**（即本篇的 S1/S2/S3）：修复仍封存在此（§四），可让 MTP 完全透明，
    但代价是 no-MTP 数值也变（两条路一起换到同一套新算术，不是恢复旧值）；
  - **整体定性**（三层误差源、翻转机制、"降智"=难任务上方差放大而非均值下移、
    "正确"只能定义为不变量）见 05 的第六节；
    完整推理与阐述另见根目录 `06-MTP降智问题的完整阐述.md`。
- 原结论（下文保留）：保留速度优化就不可能与原版逐位一致；放弃加速才可能，
  且"开 MTP 完全透明"与"no-MTP 与原版逐位一致"在原理上二选一（卡在 FA 的 pb 上）。

---

## 一、四臂对照结果（实测）

题目（单轮，见 `01-四臂试验结果/对比汇总.md`）：

> 用贴吧暴躁老哥的口气和风格，且使用文言文，仿《过秦论》作《过美利坚论》…

| 组合 | 不开 MTP | 开 MTP | 两者关系 |
|---|---|---|---|
| **原版**（上游 976E2CAB） | 2015 字 / 23 段 | 1758 字 / 21 段 | **完全不同**（相似度 0.10，两篇不同文章） |
| **修复版**（T20 8093C771 + 开关） | 1242 字 / 14 段 | 1242 字 / 14 段 | **逐字节完全相同**（md5 一致） |

结论：

1. **原版开 MTP 会换一条生成轨迹**（结构、用典、篇幅全变）。所谓"降智"不是模型退化，而是底层末位数值差异被采样放大成"另一个 token"，之后一路分叉。
2. **修复版开/关 MTP 逐字节一致**，MTP 完全透明。修复代价：约 -0.5~0.8% 性能（全来自 S2），PPL 不变（4.3562 vs 4.3572）。
3. 修复版的 no-MTP 输出与原版也不同（见第三节）。

## 二、机制：三个数值分叉源

采样温度 > 0 时，logits 的 ~1e-6 级差异就能在某一步"近似并列"的候选处跳到另一个 token，之后整条轨迹全变。

| 编号 | 分叉源 | 代码位置 | 修复 |
|---|---|---|---|
| S1 | FA 小批 kernel 选择：decode(n_q=1) 走 VEC，verify(n_q=2~4) 走 TILE | `ggml-cuda/fattn.cu`（Volta 分支） | `vec_limit 2→16`，让 verify 也走 VEC |
| S2 | GDN vec4 布局分支 `vec4 = n_tokens > 1`（本 fork A2 引入；decode 走标量、prefill 走 vec4） | `ggml-cuda/gated_delta_net.cu` | 一律 vec4（调试开关 `GGML_CUDA_GDN_VEC4`） |
| S3 | FA split-K 切分数 pb 随 padded KV 长度变化（跨 256 边界就变，**上游固有**） | `ggml-cuda/fattn-common.cuh` | n_q≤8 钉成"与位置/批型无关"的常量 |

实测证据（batch-invariance 工具 + server logits dump，见 `04-T20源码排查/原始数据/`）：

- 不修时，MTP-verify 与 no-spec decode 的 logits 首个差异出现在 position 255（首个 256 边界）
- 修后验收：n-max 1/2/3 与无 spec 在短(900 token) / 32k(150) / 128k(100) 全部**逐 token 一致**
- 三臂对照（greedy, 250 token）：stock vs T19 在 d0@207 分叉；stock vs T20(默认) 在 d0/d8192 全等；stock vs t20fix(S1+S2 开关) 在 d8192@147 分叉

## 三、为什么"修复版的 no-MTP"对不上"原版的 no-MTP"

因为**本 fork 的 prefill 优化已经改变了末位数值**，decode 再忠实也读不回原版：

- A2 的 GDN vec4 用在 prefill（n_tokens>1）→ GDN 递推状态/KV 末位就与原版不同
- T12/T16 的 FA stream-K / PB 调优也在 prefill 路径
- 证据：T19 生产版（decode 走的正是"非 vec4"+VEC+原版 pb 的忠实路径）与原版照样分叉（greedy d0@207）

所以：**保留加速 ⇒ 与原版逐位一致不可能**（这是硬结论）；**放弃加速（干净上游 + 极小补丁）⇒ 可以做到 no-MTP 与原版逐位一致**。

## 四、pb 是绕不开的开关（最终结论）

- pb 决定 KV 切几段再合并，直接决定末位数值；
- 上游的 pb 是**随序列长度/批型**由波次搜索定的 → 同一位置在 255 和 256 处可能用不同 pb；
- 而 MTP verify 一次算 2~4 个 token，`launch_fattn` 每次只算**一个** pb → 与 decode 的"位置相关 pb"必然对不齐；
- 想反过来（保上游 pb、让 verify 对齐）只能逐 token 验证 = 没有加速。

| 选择 | 代价 | 结果 |
|---|---|---|
| **钉 pb**（现修复版做法） | ~0-0.8% | 开 MTP 100% 透明；no-MTP ≠ 原版 |
| **不钉 pb**（最小补丁做法） | 0 | no-MTP == 原版（逐位）；MTP 在跨 256 边界处仍有 ulp 残差 |

最理想但未做的方案：**干净上游 + 仅 S1**（不碰 decode 路径）= no-MTP 逐位等于原版 + MTP 只差 S3 那一档边界残差（发作频率未实测）。若要"最小上游补丁"，就是 `vec_limit` + `pb` 两处。

## 五、构建与部署（当时状态）

| 名称 | 目录 | ggml-cuda SHA | 说明 |
|---|---|---|---|
| 原版（上游） | `D:\LLM\Backend\llama.cpp` | `976E2CAB` | 对照基准 |
| 修复版 | `D:\LLM\Backend\llama.cpp-t20` | `8093C771` | 含 S3'；S1/S2 用环境变量开启 |
| T19 交付（生产） | `D:\LLM\Backend\llama.cpp-my` | `7F1B9B24` | 未含修复，输出不受本次影响 |

模型：`<models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf`

**修复版必须带的环境变量**（否则等于 T19）：

```
GGML_CUDA_FA_SMALL_BATCH_VEC=1
GGML_CUDA_GDN_VEC4=1
```

四条启动命令见 `02-复现命令.md`。

## 六、有用的副产品

### 调试开关（默认全关）

| 环境变量 | 位置 | 作用 |
|---|---|---|
| `GGML_CUDA_FA_SMALL_BATCH_VEC` | `fattn.cu` | 小批强制走 VEC 核（S1 开关） |
| `GGML_CUDA_GDN_VEC4` | `gated_delta_net.cu` | 0/1 强制 GDN 布局（S2 开关） |
| `GGML_CUDA_FATTN_PB_FORCE` | `fattn-common.cuh` | 强制 KV 切分数（研究 pb 用） |
| `GGML_CUDA_FA_FORCE` | `fattn.cu` | 强制 FA kernel 类型 |
| `GGML_DBG_LOGITS` | `common/sampling.cpp` | dump 采样点 logits（对齐两条路用） |

### 工具

- `examples/batch-invariance`（源码在 `04-T20源码排查/工具源码/`）：支持 `--seq-a/--probe/--rollback`，逐行比对两条序列/两种批型的 logits，专治"批型相关数值差"
- `03-测试脚本/t20_quality_run.ps1` + `t20_quality_render.ps1`：四臂自动跑 server + 渲染带标注的 md（含推理/回答 token 数统计，走 `/tokenize`）
- `03-测试脚本/mtp_analyze.py` + `mtp_label.py`：结果对比分析与标注

### 踩过的坑

- **PowerShell 5.1 发中文/非 ASCII body**：`Invoke-RestMethod -Body $jsonString` 会编码坏 → 必须 `-Body ([System.Text.Encoding]::UTF8.GetBytes($json))`（否则 server 返回 400，"推理长度"统计会全 -1）
- **PS 5.1 读无 BOM 的 UTF-8 文件**：`Get-Content` 默认按 ANSI 读 → 中文乱码/JSON 解析失败 → 要 `-Encoding UTF8`，或写文件时存 UTF-8 with BOM
- **含中文的 .ps1**：必须存 UTF-8 with BOM，否则 PS 5.1 解析报错
- **llama-server 的 thinking**：`--jinja` 下 `reasoning_content` 与 `content` 自动分离（剥离 thinking 只需取 content）
- **内置模板 vs `--chat-template-file`**：行为不同（prompt 40 vs 82 token，思考风格也不同）
- **贪婪 + 无输出上限 + 开放题**：会复读狂奔停不下来（实测一条题 10 分钟不结束）→ 改 `--temp 1.0` 或加 `-n` 限制
- **长同义 prompt 会立刻 EOS**：长文测试必须加 `ignore_eos=true`

## 七、遗留（如果再捡起来）

1. 建"干净上游 + 仅 S1"版，实测 S3 边界残差发作频率（决定"最小补丁版"能不能两个都拿到）
2. S2' 备选：让 vec4 布局的归约顺序与标量路径一致（当时评估"可能零代价"，未实测）
3. 若要上报上游：最小补丁 = `vec_limit` + `pb` 两处（需先想清楚 S3 的取舍口径）

> 2026-09-26 注：以上是**选核轴**的工作项；**状态轴**已在 `05-T24-ReplaySSM/` 完成。
> 05 另提出路线 C（decode-preserving 对齐：S1 免费、S2 小改可行、S3 需新机制），
> 用于同时满足"no-MTP 数值不变"与"MTP 透明"这两个目标。

## 八、目录导览

```
MTP封存-2026-09-23/
├── README.md                    <- 本文件（全部经验）
├── 01-四臂试验结果/               <- 四份带标注的结果 md + 对比汇总 + 原始导出
├── 02-复现命令.md                <- 四条启动命令与参数
├── 03-测试脚本/                  <- 四臂 runner/render + 分析脚本 + 分析报告
├── 04-T20源码排查/
│   ├── 调试开关与改动说明.md       <- 三个分叉源与开关的代码说明
│   ├── 工具源码/                 <- t20-work.patch、batch-invariance 源码
│   ├── 脚本/                     <- T20 排查用的全部 harness（binv/3way/perf/logits）
│   ├── 原始数据/                 <- logits dump、验收、性能、batch-invariance 原始输出、长 prompt
│   └── 协作文档快照/              <- v100-collab 全部 md 快照（RESULTS 的 T20 章节、T20 任务书等）
├── 05-T24-ReplaySSM/            <- 续篇（2026-09-26）：状态轴实现记录 + 最终结论
│   ├── README.md                <- ReplaySSM 实现、两个 bug、验证数据、定性结论、三条路线
│   ├── t24-replayssm-final-20260926.patch
│   └── 数据/                    <- 1000 token 贪心 token 流（base/老/new）、64 token 三臂、自检运行
└── 06-MTP降智问题的完整阐述.md    <- 问题的完整推理：浮点无基准、三层误差源、翻转机制、
                                      期望无偏 vs 方差放大、"正确"=不变量、三条路线与结论
```

原始工作目录（未动）：`D:\LLM\Backend\mtp\`（可自行删除）；协作文档原件：`D:\LLM\Backend\v100-collab\`。
