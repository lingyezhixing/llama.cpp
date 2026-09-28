# ENVIRONMENT: 工作区、补丁、构建部署、工具

最后整理: 2026-09-22 (analyst; 内容来自 implementer 的 SESSION RESUME)

## 当前交付快照 (implementer, 2026-09-22 收尾, 上下文压缩后先看这里)

| 项 | 值 |
|---|---|
| 工作区 | `D:\LLM\Backend\src\llama.cpp-my` @ `4046f5c8c`, **4 个已修改文件, 0 个未跟踪**, `git diff --check` 干净 |
| 改动量 | convert.cu +55/-2, dequantize.cuh +77, gated_delta_net.cu +56/-21, unary.cu +32 |
| patch | `patches\v100-{dequant-vec,gdn-vec4,t07-silu-vec4}.patch` (纯 ASCII/LF; 对 HEAD 前向 + 对工作区反向双向验证通过) |
| 部署 DLL | `D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll` SHA256 `453E29111E5E29C9...` (T12+T16 入库; 与 build 输出已核对一致; 历次记录见文末) |
| 验收 | PPL **4.3569** (门槛 4.3572+/-0.013); pp512 **956.6**, tg128 **26.61**; ub512 口径 |
| PPL 偏移 | 4.3572 -> 4.3569 全部来自 T03 GDN vec4 (求和重结合, 良性); dequant/silu 逐位相同 |
| 构建脚本 | `cmd /c %TEMP%\v100\build_ggml_cuda.cmd` -> 拷贝 `build\bin\Release\ggml-cuda.dll` |
| 归档 | `v100-collab/artifacts/` 56 个文件 (全部 harness 源 + 关键日志); `%TEMP%\v100` 已清理 504MB |

## 工作区状态 (D:\LLM\Backend\src\llama.cpp-my, 未提交, 4 个文件)

| 文件 | 内容 | patch 归档 |
|---|---|---|
| `ggml/src/ggml-cuda/dequantize.cuh` | Q6_K/Q5_K 向量化反量化设备函数 | `patches\v100-dequant-vec.patch` |
| `ggml/src/ggml-cuda/convert.cu` | 向量化反量化 kernel + launcher + 16B 对齐回退 | 同上 |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | vec4 行布局 (按 n_tokens 分支) + KDA 修复 | `patches\v100-gdn-vec4.patch` |
| `ggml/src/ggml-cuda/unary.cu` | silu float4 向量化 (带标量回退) | `patches\v100-t07-silu-vec4.patch` |
| ~~`docs/v100-performance-target.md`~~ | 目标/可行性文档 - **用户已删除 (2026-09-22)**; 不再出现在工作区 | - |

- 注意: `update_llama.bat` 会 `git reset --hard`, 更新后需 `git apply` 上述 patch
- 基线 commit: `4046f5c8c` (含 sm70 FATTN 调参补丁)
- **patch 校验 (2026-09-22, implementer)**: 3 个 patch 已重新生成并用
  `git apply --check --cached` (对 HEAD 前向可应用) + `git apply --check -R` (对当前工作区反向可应用)
  双向验证通过; 文件为纯 ASCII/LF (此前版本是 PowerShell `>` 重定向写出的 UTF-16, `git apply` 无法读取, 已修)
- `git diff --check` 无空白错误; 工作区只有这 4 个已修改文件, 无未跟踪文件 (目标文档已由用户删除)
- **构建非逐字节可复现**: 同一源码重建后 DLL 与上次构建仅差 4 字节, 全在 PE header/.rdata (构建元数据),
  `.text` 代码段逐字节相同, PPL 一致 -> 视为等价构建 (2026-09-22 复核)

## 构建与部署

```
构建: cmd /c %TEMP%\v100\build_ggml_cuda.cmd     (vcvars64 + cmake --build build --config Release --target ggml-cuda)
部署: Copy-Item D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\ggml-cuda.dll D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll -Force
```

- 已部署 DLL = 当前最新构建 (含上述 4 项改动, T05 实验回退后重建), 2026-09-23 复核:
  - build\bin\Release\ggml-cuda.dll == D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll (同一次构建, SHA 已核对)
  - SHA256 `102BF84488F2FD43DBA4E6E89DA39B8010956EE6B9A0B17BBFBBA5584FD1835C`
    (历次: `6D0AC881` T08 期 -> `B60722EB` T07 rms 回退 -> `4135CDF3` T11 回退 -> `9B389126` T03 回退
     -> `102BF844` T05 回退; 均为同源码重建, 只有构建元数据差异, .text 相同)
  - 功能复核: PPL **4.3569** (门槛 |x-4.3572| <= 0.013)
  - 注意: 机器存在 ~1% 的 session 级漂移, 比较必须同 session 交替 A/B
- 备份: `%TEMP%\v100\ggml-cuda-fast.dll` (仅 dequant patch), `%TEMP%\v100\ggml-cuda-old.dll` (原始)

## 验证门槛 (每次改完必过)

1. PPL: `llama-perplexity -m <model> -f %TEMP%\v100\ppl_text.txt -c 512 --chunks 8 -ngl 99 -fa on -ctv q8_0 --seed 42`
   -> 期望 4.3569 或 4.3572 (|diff| <= 0.013)
2. 性能: `llama-bench -m <model> -p 512,4096,8192 -n 128 -ub 512 -fa on -ctv q8_0 -r 3`
3. 有疑义时 A/B 交替换 DLL 复测 (单次 ±0.5% 噪声不可作判据)

## 工具与脚本

| 路径 (%TEMP%\v100\) | 用途 |
|---|---|
| `build_ggml_cuda.cmd` | 构建 |
| `server_test.ps1` | 服务端 pp 测试 (-Tag -Port -ExtraArgs) |
| `server_decode_test.ps1` | 服务端 decode 测试 (含/不含 MTP) |
| `t01_bench.cu/.exe` | cuBLAS/cuBLASLt 形状扫描 |
| `t01_ab.cu/.exe` | 预热 + A/B 对照 |
| `t02_fused.cu/.exe` | 融合 kernel 微基准 (mode: 1=dequant, 2=mma, 8=A load) |
| `gemm_sweep.exe` / `gemm_conc.exe` / `dequant_bench.exe` | GEMM 扫描 / 并发 / 反量化带宽 |
| `ncu_*.ps1` (t01/t01b/gdn/t02/gateA) | ncu 提权脚本 (UAC) |
| `ppl_text.txt` | PPL 测试文本 |
| `t08_arb.cu/.exe` | **T08 Step 1 主仲裁**: 8 个模型形状 x (def_tf32/def_tensor/def_dflt/lt_best) x 5 轮交错 |
| `t08_one.cu/.exe` | 单形状 (gate/up) llama.cpp-exact 调用, 预热/时钟诊断 |
| `t08_interf.cu/.exe` | 解包-写回干扰诊断 (mode 0-3) |
| `t08_query.py` / `t08_times.py` / `t08_kerncmp.py` / `parse_log.py` | nsys sqlite 查询 / kernel 时间聚类 / kernel 对比 / cuBLAS 日志解析 |

## 归档与清理 (2026-09-22, implementer)

- `v100-collab/artifacts/` 现含 **56 个** 源文件/脚本/日志: T02 全部 harness 变体 (v1-v7, peak, tile_dbg, bm64)、
  T09 (q6k_v9_bm64.cu, pack112.py, verify_formula.py, sass_v9.txt)、T08 全部 (arb/one/interf + 工具脚本 + 原始日志)、
  T01 (t01_bench.cu, t01_ab.cu)、关键 nsys kernel-summary CSV 7 份、ncu 日志 4 份、server 测试脚本、构建脚本
- `%TEMP%\v100` 已清理: 删除 201 个可再生成文件 (504 MB: Q6_K/Q5_K raw/bin 数据 374MB、nsys sqlite/rep、
  中间 DLL、.lib/.exp、旧日志); 保留构建脚本、常驻 harness (t01/t08)、基线备份 DLL、kernel-summary CSV、ppl_text.txt
- 大文件原数据可用 `artifacts/extract_q6k.py` + `artifacts/pack_q6k.py` + `artifacts/pack112.py` 重新生成
- T05 结案产物: `nodes_raw.txt` (图节点 dump) + `analyze_nodes*.py` (v1-v4) + `q_dec8*.py` / `q_sig.py` +
  `trace_ops.patch` / `build_harness.cmd` (均在 artifacts/)

## nsys / ncu 注意

- nsys: 无需提权, `--trace=cuda --cuda-event-trace=false`; `-n > 2` 会丢数据, 用 `-n 1/2`;
  **抓 CUDA graph 重放必须 `--cuda-graph-trace=node`** (默认 graph 模式会隐藏图内 kernel - T05 结案教训)
- ncu: 需要提权 (脚本用 Start-Process -Verb RunAs); 抓 llama.cpp 时注意锁频对绝对值的影响, 只看比例

## 2026-09-23 暂停点 (T12/T16 夜间断点, implementer)

| 项 | 值 |
|---|---|
| 部署 DLL | **BASE 交付版** SHA256 `102BF84488F2FD43...` (已恢复; T12/T16 未入库) |
| 构建输出 | `build\bin\Release\ggml-cuda.dll` 不存在 (被中止的 ninja 删除; cmake --build 会重链) |
| 工作区 | 5 文件 = 交付 4 + `fattn-common.cuh` (含 T12 stream-K 启发式 + T16 PB=2 + env 覆盖); `fattn-mma-f16.cuh` 干净 |
| 待测 DLL | `%TEMP%\v100\ggml-cuda-T16-final.dll` 未生成 (重建即得); 实验 DLL: T12-BASE/T12-KV/T16-env/T16-KV+PB/T16-cfgA/B/C/E (9 个) |
| nsys 报告 | `%TEMP%\v100\t16_*.nsys-rep` 9 个 + `t16_ks_*.csv` 聚合; 查询脚本 `t16_fa.py` / `t16_parse.py` / `t16_sweep*.ps1` |
| 恢复步骤 | (1) `build_ggml_cuda.cmd` 重建 -> (2) 部署并核对 SHA -> (3) 短上下文 PB=2 检查 -> (4) 完整验收 A/B -> (5) PPL+生成 -> (6) 入库/patch/文档 |

**部署陷阱 (实测)**: 构建后部署路径 `D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll` 曾被自动更新为新构建
(疑似与构建输出硬链接; 恢复前请 `fsutil hardlink list` 再验)。**每次构建/部署后必须核对部署 SHA**;
交付版备份在 `%TEMP%\v100\ggml-cuda-T12-BASE.dll` (注意 `ggml-cuda-delivered.dll` 是 9B389126 的旧快照)。

**T16 实验结论要点 (供恢复)**: mma FA 内核 = 纯 stream-K (只读 blockIdx.x), 无 parallel_blocks 路径;
正确等价实现 = 覆盖 `blocks_num.x`; 扫测 8 配置 -> **PB=2 (grid = 2*ntiles_dst) 最优 (-14.7% vs T11-off,
-6.7% vs T12 默认)**; PB=4 / grid96 / 占用变体 (Q_in_reg 必 spill) 全部否证; env 开关:
`GGML_CUDA_FATTN_STREAM_K` (T12), `GGML_CUDA_FATTN_BLOCKS` / `GGML_CUDA_FATTN_PB` (T16) - 均为实验开关, 入库前需决定去留。

## 明日任务清单 (2026-09-23 夜整理; **已全部执行完毕**: T12/T16/T17/T18; 本节仅历史记录; 当前任务见 `TASKS/T19-final-bench.md`)

### 0. 开工检查 (5 min)
- 重建并部署: `cmd /c %TEMP%\v100\build_ggml_cuda.cmd` (build 输出缺失, 会重链) ->
  `Copy-Item build\bin\Release\ggml-cuda.dll D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll -Force`
- **核对部署 SHA** (必须): BASE = `102BF84488F2FD43`; T12+T16 构建应为新 SHA (记下来);
  若构建进程/部署异常, 从 `%TEMP%\v100\ggml-cuda-T12-BASE.dll` 恢复交付版
- 验证工作区 = 5 文件 (交付 4 + fattn-common.cuh), `fattn-mma-f16.cuh` 干净

### 1. T12+T16 合并验收 (analyst 门槛; A=BASE, B=T12+T16, 同 session 交替 >=3 轮)
1. **短点先行**: `-p 512,4096,8192 -n 128 -r 3` (>=2 轮交替) + tg128 + `-ub 2048 -p 4096` 抽查
   - 一致性回退 >0.3% -> 加条件 `ntiles_KV >= 64` (KV >= 4096 才用 PB=2); 无回退 -> 保持简单 (不加条件)
2. **长点** (r=2, 交替): depth32k (`-p 4096 -n 0 -d 32768`) >= +1.5%; pp32768 (`-p 32768 -n 0`) >= +0.8%;
   **pp8192@depth128k (`-p 8192 -n 0 -d 131072`) >= +4% = 主判据** (先建同 session 基线)
   - 若长点落 +2~4% 且其余全过 -> 报 analyst 复核 (不自动拒); < +2% -> 拒
3. PPL (`-c 512 --chunks 8 --seed 42`, 期望 4.3568; 门槛 4.3572±0.013) + 200 token 生成检查
4. 通过 -> patch 归档 (T12 去 REJECTED 名, T16 新建) + TASKS/T12+T16 写 Result + BOARD/ENVIRONMENT 更新

### 2. T17: Q8_0 反量化补测 (GPU, 轻 ~1 min)
- 微基准: Q8_0 反量化吞吐 vs Q6_K/Q5_K 参考 707-825 GB/s; nsys: pp512 中 Q8_0 路径占比
- go 条件: 明显偏慢 (<~700 GB/s) 且向量化可省 >=2ms/ubatch (>=+0.4% pp512) -> 实施 (逐位相同) + 验收
  (PPL/pp512-8192/pp32768/depth32k/**pp8192@depth128k 不回退**); 否则记录数字关闭

### 3. T18 Stage 0 (静音可做; 纯 CPU/代码, 不动工作区)
- 抽独立 harness: 真实形状 DKQ=DV=256, ncols=64, l=32768 (取 T10/T11 的 nsys 口径),
  **基线 = 生产配置 (T12+T16, PB=2/grid=384, ≈2027us/launch @depth8k)**, 不是 stock (2375us)
- 先复刻基线 -> 再进 Stage 1 方向① (K/V 寄存器软件流水线); 1.20x 才集成, 1.5 天 checkpoint <1.10x 停
- 方向顺序 (analyst): ①K/V 寄存器流水 -> ④ncols=32 探针 (0.5 天) -> ②消 67584B combine smem (2 CTA/SM) -> ③warp tile 扩展

### 环境提醒
- 交付版 DLL 备份: `%TEMP%\v100\ggml-cuda-T12-BASE.dll` (102BF844); 实验中 DLL 9 个在 `%TEMP%\v100\`
- nsys 图内 kernel 必须 `--cuda-graph-trace=node`; 长点比较必须同 session 交替 A/B
- 用户夜间静音: GPU 重负载/长时间构建都先要时段

## 交付快照更新 (2026-09-23, T12+T16 采纳后)

| 项 | 值 |
|---|---|
| 工作区 | `D:\LLM\Backend\src\llama.cpp-my` **5 个已修改文件**: 原 4 文件 + `fattn-common.cuh` (T12+T16) |
| patch | `patches\v100-{dequant-vec,gdn-vec4,t07-silu-vec4}.patch` + **`v100-t12t16-fattn-split.patch`** (双向 apply 校验通过; T11 原 REJECTED patch 被取代, artifacts 留档) |
| 部署 DLL | `D:\LLM\Backend\llama.cpp-my\ggml-cuda.dll` SHA256 `453E29111E5E29C9...` (T12+T16; 每次构建后核对, 有硬链接疑点) |
| 验收 (本次) | **pp8192@depth128k 375.45 (+11.28%)** / depth32k 682.2 (+4.77%) / pp32768 791.1 (+2.18%) / pp512 950.8 / pp4096 932.5 / pp8192 912.1 / tg128 26.68 / PPL **4.3562** |
| env 开关 (保留) | `GGML_CUDA_FATTN_STREAM_K` (0/1), `GGML_CUDA_FATTN_BLOCKS=N`, `GGML_CUDA_FATTN_PB=N` - 实验/回归用, 未设时走新默认 (stream-K 启发式 + PB=2 + 保底) |
| 实验 DLL 备份 | `%TEMP%\v100\ggml-cuda-T12-BASE.dll` (交付前基线), `ggml-cuda-T12T16.dll` (当前采纳版), 其余 8 个实验变体 |


## T18 归档后状态 (2026-09-23, implementer)

| 项 | 值 |
|---|---|
| 工作区 | `D:\LLM\Backend\src\llama.cpp-my` **5 个修改文件** (原 4 + `fattn-common.cuh` T12+T16); T18 两文件已 `git checkout` 撤销 |
| patch | 4 个 (dequant-vec / gdn-vec4 / t07-silu-vec4 / **v100-t12t16-fattn-split**); T18 被拒 patch: `artifacts/t18-ncols32-REJECTED.patch` |
| 部署 DLL | `453E2911...` (T12+T16; 每次构建后核对, 防硬链接意外) |
| 验收 (T12+T16) | pp8192@depth128k 375.5 / depth32k 682.2 / pp32768 791.1 / pp512 ~950 / tg128 ~26.6 / PPL 4.3562 |
| T18 产物 | harness `artifacts/t18_fa_harness.cu` + 变体 exe 5 个 + ncu 日志 2 + nsys rep 2 (`%TEMP%/v100/t18_d128k_{A,B}.nsys-rep`) |
| 遗留经验 | FA 类 harness 必须在**目标 l** 做生产保真对照 (T18 在 l=35k 对过, 长 l 未对 -> 误判 1.10x) |


## T19 交付快照 (2026-09-23, implementer)

| 项 | 值 |
|---|---|
| NEWBASE | `e6ab7c1a4` (两边同一 base; vanilla ff-only, fork rebase 干净通过) |
| fork HEAD | `afbab1748` (交付提交, rebase 后; 其下 `3fc05594a` sm70 FA 调参); **未 push**; 备份分支 `backup-t19-pre-rebase` (9403d528e) |
| vanilla HEAD | `e6ab7c1a4` |
| 部署 | OURS `D:\LLM\Backend\llama.cpp-my` SHA `7F1B9B2403438803...`; STOCK `D:\LLM\Backend\llama.cpp` SHA `976E2CABF9EADC7D...` (均 == 各自 build 输出; 全量 exe+dll 已同步) |
| patch | `patches\v100-{dequant-vec,gdn-vec4,t07-silu-vec4,t12t16-fattn-split}.patch` 基于 NEWBASE 重生成, 双向 apply 校验通过 (与交付提交 3 行差 = sm70 提交, 预期) |
| 验收 (OURS, 新 base) | pp512 948.5 / pp131072 531.5 / tg128 26.61 / **PPL 4.3562**; vs STOCK: pp +8.7..+35.6%, tg d32768 持平 (复测), tg d131072 存疑 (-3~-10%) |
| 图/数据 | `artifacts/t19_{pp,tg,delta}.png` + `t19_data.csv` + `t19_plot.py` |
| 备份 DLL | `%TEMP%/v100/ggml-cuda-T19-{OURS,STOCK}.dll`; 历史 `ggml-cuda-T12T16.dll` (453E2911) |


## T20 资产与前置 (2026-09-23, analyst)

| 项 | 值 |
|---|---|
| 旧排查报告 (**仅参考源, 不盲信**) | `D:\LLM\Backend\MTP-轨迹一致性分析与修复报告.md` (2026-09-22) |
| 修复 patch 备份 | `D:\LLM\Backend\mtp-backup\mtp-work.patch` (UTF-16, apply 前需转 UTF-8/LF; fattn.cu env + examples 接线) |
| 差分诊断工具 | `D:\LLM\Backend\mtp-backup\batch-invariance\` (batch-invariance.cpp + CMakeLists.txt) |
| 当前构建 | **不含旧修复** (HEAD `afbab1748`, DLL `7F1B9B240343`); 基线 = 未修复状态 |
| 注意 | T12/T16 (stream-K/PB=2, `fattn-common.cuh`) 与 env (`fattn.cu`) 的交互需验证 |
| 时间盒 | 3 天 (1.5 天 checkpoint); **完成后暂停等判断** |

## T20 状态 (2026-09-23, implementer) - MTP 轨迹排查后

| 项 | 值 |
|---|---|
| 工作区 | `D:\LLM\Backend\src\llama.cpp-my` @ `afbab1748` + **T20 实验改动 (未提交)**: `fattn.cu` (FA_SMALL_BATCH_VEC + FA_DEBUG + FA_FORCE env), `gated_delta_net.cu` (GDN_VEC4 env), `fattn-common.cuh` (FATTN_PB_FORCE env), `common/sampling.cpp` (GGML_DBG_LOGITS dump), `examples/CMakeLists.txt` + `examples/batch-invariance/` (新工具, 未跟踪) |
| 实验 patch | `artifacts/t20-work.patch` (tracked 改动); batch-invariance 工具源码在工作区 + `D:\LLM\Backend\mtp-backup\batch-invariance\` (原版) |
| 构建配置 | `build` 已改为 `LLAMA_BUILD_EXAMPLES=ON` (T20 需要 `llama-batch-invariance` 目标) |
| T20 调试构建 | ggml-cuda.dll SHA `DE3F5E50A3BAE907...` (含全部 T20 env, 默认行为 = T19); 备份 `%TEMP%\v100\ggml-cuda-T20-debug.dll`; llama-common.dll (含 logits dump) 17:12 构建, 未单独备份 (可从工作区重建) |
| **部署 (已回滚 T19)** | `D:\LLM\Backend\llama.cpp-my`: ggml-cuda `7F1B9B2403438803...` + llama-common `7A9F5E651C4C8E31...` (T19 备份 `%TEMP%\v100\llama-common-T19.dll`); 冒烟 pp512 936 (r=1) |
| T20 新 env (仅调试, 默认全关) | `GGML_CUDA_GDN_VEC4=0/1` (GDN 布局覆盖), `GGML_CUDA_FA_FORCE=vec/tile/mma` (强制 FA 内核), `GGML_CUDA_FATTN_PB_FORCE=N` (非 stream-K 强制 KV 切分块数), `GGML_DBG_LOGITS=<path>` (采样点 top-8 logits dump, 在 llama-common) |
| 工具/脚本 (%TEMP%\v100) | `t20_binv_run.ps1` (batch-invariance 跑批), `t20_server_ab.ps1` (server A/B: -Tag/-ExtraArgs/-PromptFile/-EnvVars/-NProbs/-NPredict/-Ctx), `t20_cmp2.py` (逐 token 对比), `t20_logitcmp.py` / `t20_logitpat.py` (logits dump 对比), `t20_perf.ps1` (修复代价 A/B), `t20_mkprompt.py` (长 prompt 生成), `t20_cmp.py`/`t20_probs*.py` |
| 原始数据 | `%TEMP%\v100\t20_*`: `binv_*.txt` (工具输出), `srv_*.json` + `*_raw.json` + `*.log` (server A/B), `logits_*.txt` (dump), `k32_*`/`k128_*` (长上下文), `t20_perf.txt`, `t20_ppl_fix.txt` |
| 长 prompt | `t20_prompt32k.txt` (180117 chars ~ 32k token) / `t20_prompt128k.txt` (720103 chars ~ 128k token), 纯 ASCII 合成文本 (含结尾指令) |
| 注意 | 部署陷阱同前 (构建输出会覆盖部署路径, 每次核对 SHA); 若继续 T20 工作需先部署 T20 调试 DLL (`%TEMP%\v100\ggml-cuda-T20-debug.dll` + 重建 llama-common) |

## T20 收尾补充 (2026-09-23, implementer)

| 项 | 值 |
|---|---|
| T20 低代价修复构建 | ggml-cuda.dll SHA `8093C771A20EDB1C...` (S3' 固定切分已内置; S1/S2 env 默认关); 备份 `%TEMP%\v100\ggml-cuda-T20-lowcost.dll` |
| 三臂测试目录 | `D:\LLM\Backend\llama.cpp-t20` (T19 全量 exe/dll + T20 ggml-cuda 8093C771; 用于 stock/T19/T20 对照, 非生产) |
| 部署状态 | `D:\LLM\Backend\llama.cpp-my` = **T19 交付** (ggml-cuda `7F1B9B24...` + llama-common `7A9F5E65...`) — 生产不受影响 |
| 三臂脚本 | `%TEMP%\v100\t20_3way.ps1` (d0) / `t20_3way_d8192.ps1` (8k) / `t20_3way_cmp*.py`; 结果 `3w_*.json` |
| 长 prompt | `t20_prompt8k.txt` (45143 chars ~ 8031 token, 纯 ASCII) |
| 待办 | 用户裁决是否采纳修复 (S1+S2 固化 + S3' 已内置) -> 若采纳: 改代码 + 全量验收 + 重建部署核对 SHA |
