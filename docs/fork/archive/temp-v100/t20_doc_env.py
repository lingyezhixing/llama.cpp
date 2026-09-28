import io

sec = """

## T20 状态 (2026-09-23, implementer) - MTP 轨迹排查后

| 项 | 值 |
|---|---|
| 工作区 | `D:\\LLM\\Backend\\src\\llama.cpp-my` @ `afbab1748` + **T20 实验改动 (未提交)**: `fattn.cu` (FA_SMALL_BATCH_VEC + FA_DEBUG + FA_FORCE env), `gated_delta_net.cu` (GDN_VEC4 env), `fattn-common.cuh` (FATTN_PB_FORCE env), `common/sampling.cpp` (GGML_DBG_LOGITS dump), `examples/CMakeLists.txt` + `examples/batch-invariance/` (新工具, 未跟踪) |
| 实验 patch | `artifacts/t20-work.patch` (tracked 改动); batch-invariance 工具源码在工作区 + `D:\\LLM\\Backend\\mtp-backup\\batch-invariance\\` (原版) |
| 构建配置 | `build` 已改为 `LLAMA_BUILD_EXAMPLES=ON` (T20 需要 `llama-batch-invariance` 目标) |
| T20 调试构建 | ggml-cuda.dll SHA `DE3F5E50A3BAE907...` (含全部 T20 env, 默认行为 = T19); 备份 `%TEMP%\\v100\\ggml-cuda-T20-debug.dll`; llama-common.dll (含 logits dump) 17:12 构建, 未单独备份 (可从工作区重建) |
| **部署 (已回滚 T19)** | `D:\\LLM\\Backend\\llama.cpp-my`: ggml-cuda `7F1B9B2403438803...` + llama-common `7A9F5E651C4C8E31...` (T19 备份 `%TEMP%\\v100\\llama-common-T19.dll`); 冒烟 pp512 936 (r=1) |
| T20 新 env (仅调试, 默认全关) | `GGML_CUDA_GDN_VEC4=0/1` (GDN 布局覆盖), `GGML_CUDA_FA_FORCE=vec/tile/mma` (强制 FA 内核), `GGML_CUDA_FATTN_PB_FORCE=N` (非 stream-K 强制 KV 切分块数), `GGML_DBG_LOGITS=<path>` (采样点 top-8 logits dump, 在 llama-common) |
| 工具/脚本 (%TEMP%\\v100) | `t20_binv_run.ps1` (batch-invariance 跑批), `t20_server_ab.ps1` (server A/B: -Tag/-ExtraArgs/-PromptFile/-EnvVars/-NProbs/-NPredict/-Ctx), `t20_cmp2.py` (逐 token 对比), `t20_logitcmp.py` / `t20_logitpat.py` (logits dump 对比), `t20_perf.ps1` (修复代价 A/B), `t20_mkprompt.py` (长 prompt 生成), `t20_cmp.py`/`t20_probs*.py` |
| 原始数据 | `%TEMP%\\v100\\t20_*`: `binv_*.txt` (工具输出), `srv_*.json` + `*_raw.json` + `*.log` (server A/B), `logits_*.txt` (dump), `k32_*`/`k128_*` (长上下文), `t20_perf.txt`, `t20_ppl_fix.txt` |
| 长 prompt | `t20_prompt32k.txt` (180117 chars ~ 32k token) / `t20_prompt128k.txt` (720103 chars ~ 128k token), 纯 ASCII 合成文本 (含结尾指令) |
| 注意 | 部署陷阱同前 (构建输出会覆盖部署路径, 每次核对 SHA); 若继续 T20 工作需先部署 T20 调试 DLL (`%TEMP%\\v100\\ggml-cuda-T20-debug.dll` + 重建 llama-common) |
"""

p = r"D:\LLM\Backend\v100-collab\ENVIRONMENT.md"
s = io.open(p, encoding="utf-8", newline="").read()
s = s.rstrip() + sec
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok", len(s))
