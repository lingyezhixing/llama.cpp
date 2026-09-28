import io

sec = """

## T20 收尾补充 (2026-09-23, implementer)

| 项 | 值 |
|---|---|
| T20 低代价修复构建 | ggml-cuda.dll SHA `8093C771A20EDB1C...` (S3' 固定切分已内置; S1/S2 env 默认关); 备份 `%TEMP%\\v100\\ggml-cuda-T20-lowcost.dll` |
| 三臂测试目录 | `D:\\LLM\\Backend\\llama.cpp-t20` (T19 全量 exe/dll + T20 ggml-cuda 8093C771; 用于 stock/T19/T20 对照, 非生产) |
| 部署状态 | `D:\\LLM\\Backend\\llama.cpp-my` = **T19 交付** (ggml-cuda `7F1B9B24...` + llama-common `7A9F5E65...`) — 生产不受影响 |
| 三臂脚本 | `%TEMP%\\v100\\t20_3way.ps1` (d0) / `t20_3way_d8192.ps1` (8k) / `t20_3way_cmp*.py`; 结果 `3w_*.json` |
| 长 prompt | `t20_prompt8k.txt` (45143 chars ~ 8031 token, 纯 ASCII) |
| 待办 | 用户裁决是否采纳修复 (S1+S2 固化 + S3' 已内置) -> 若采纳: 改代码 + 全量验收 + 重建部署核对 SHA |
"""

p = r"D:\LLM\Backend\v100-collab\ENVIRONMENT.md"
s = io.open(p, encoding="utf-8", newline="").read()
s = s.rstrip() + sec
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("env ok", len(s))
