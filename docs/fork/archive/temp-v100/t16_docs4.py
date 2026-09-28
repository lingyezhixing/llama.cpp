import io

t = """
## 2026-09-23 暂停点 (T12/T16 夜间断点, implementer)

| 项 | 值 |
|---|---|
| 部署 DLL | **BASE 交付版** SHA256 `102BF84488F2FD43...` (已恢复; T12/T16 未入库) |
| 构建输出 | `build\\bin\\Release\\ggml-cuda.dll` 不存在 (被中止的 ninja 删除; cmake --build 会重链) |
| 工作区 | 5 文件 = 交付 4 + `fattn-common.cuh` (含 T12 stream-K 启发式 + T16 PB=2 + env 覆盖); `fattn-mma-f16.cuh` 干净 |
| 待测 DLL | `%TEMP%\\v100\\ggml-cuda-T16-final.dll` 未生成 (重建即得); 实验 DLL: T12-BASE/T12-KV/T16-env/T16-KV+PB/T16-cfgA/B/C/E (9 个) |
| nsys 报告 | `%TEMP%\\v100\\t16_*.nsys-rep` 9 个 + `t16_ks_*.csv` 聚合; 查询脚本 `t16_fa.py` / `t16_parse.py` / `t16_sweep*.ps1` |
| 恢复步骤 | (1) `build_ggml_cuda.cmd` 重建 -> (2) 部署并核对 SHA -> (3) 短上下文 PB=2 检查 -> (4) 完整验收 A/B -> (5) PPL+生成 -> (6) 入库/patch/文档 |

**部署陷阱 (实测)**: 构建后部署路径 `D:\\LLM\\Backend\\llama.cpp-my\\ggml-cuda.dll` 曾被自动更新为新构建
(疑似与构建输出硬链接; 恢复前请 `fsutil hardlink list` 再验)。**每次构建/部署后必须核对部署 SHA**;
交付版备份在 `%TEMP%\\v100\\ggml-cuda-T12-BASE.dll` (注意 `ggml-cuda-delivered.dll` 是 9B389126 的旧快照)。

**T16 实验结论要点 (供恢复)**: mma FA 内核 = 纯 stream-K (只读 blockIdx.x), 无 parallel_blocks 路径;
正确等价实现 = 覆盖 `blocks_num.x`; 扫测 8 配置 -> **PB=2 (grid = 2*ntiles_dst) 最优 (-14.7% vs T11-off,
-6.7% vs T12 默认)**; PB=4 / grid96 / 占用变体 (Q_in_reg 必 spill) 全部否证; env 开关:
`GGML_CUDA_FATTN_STREAM_K` (T12), `GGML_CUDA_FATTN_BLOCKS` / `GGML_CUDA_FATTN_PB` (T16) - 均为实验开关, 入库前需决定去留。
"""
io.open(r'D:\LLM\Backend\v100-collab\ENVIRONMENT.md', 'a', encoding='utf-8', newline='').write(t)
print('ENVIRONMENT appended')
