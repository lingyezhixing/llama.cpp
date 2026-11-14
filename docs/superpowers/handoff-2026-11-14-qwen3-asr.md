# Qwen3-ASR 交接文档（2026-11-14）

Plan: docs/superpowers/plans/2026-11-13-qwen3-asr.md
Spec: docs/superpowers/specs/2026-11-13-qwen3-asr-design.md

## 状态

Task 1-14 全部完成并提交；spec 第 13 节的成功标准已逐条核对（两项 1.7B 延迟目标因本机只有 0.6B 未测）。工作区干净。

| Task | 提交 | 内容 |
|---|---|---|
| 1 | ebc17ca28 | 公共类型、语言归一化、输出解析、JSON |
| 2 | 6336dfd59 | 流式 multipart 上传落盘 |
| 3 | 8fa541c98 | 对齐器 GGUF 键与转换 |
| 4 | 92340e6a6 | 分类头加载、图、查询 API |
| 5 | 5b99ace08 | 对齐器分词、时间戳修复、前向 |
| 6 | 7db0cd880, 37fbbaf0b | 引擎、能力、flags、props、路由；404 body 修复 |
| 7 | 8411d689c | 离线任务管理器与异步 API |
| 8 | 1e77764c6 | 真实离线转写 |
| 9 | ce55bf538 | 离线时间戳 |
| 10 | dd03de408 | 实时 WebSocket 协议 |
| 11 | 280c5f785 | 实时重解码与稳定提交 |
| 12 | 8520935b0 | 长流段重置 |
| 13a | e9ebda430 | 离线/实时上下文互斥 |
| 13b | 66bc7e453 | 实时抢占离线的门控测试 |
| 14 | 7705a9447 | 2 小时离线 / 1 小时实时 / 既有端点回归 / README 与 spec 文档 |

## 环境要点

- 构建：build 目录，Ninja Multi-Config，BUILD_SHARED_LIBS=ON，CUDA sm_89-real；编译前需载入
  vcvars64.bat（路径 E:/Program files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build）。
- 跑 GPU 必须传 --device CUDA0 --mmproj-device CUDA0，否则会挑到 sm_70 的 V100 并报 no kernel image。
- 使用的模型：Qwen3-ASR-0.6B（q8 文本 + mmproj）与 Qwen3-ForcedAligner-0.6B（q8 + mmproj），
  转换命令见 tools/server/tests/convert_qwen3_aligner.py 与 convert_hf_to_gguf.py --mmproj。
- 音频 fixture：16 kHz 单声道 PCM16 WAV，可用 ffmpeg 从 tools/mtmd/test-2.mp3 生成。

## 代码地图

server-asr-common（类型/JSON/语言）、server-asr-upload（multipart 落盘）、
server-asr-audio（miniaudio 解码、随机访问、100 ms 能量）、server-asr-job（runner 接口与任务管理器）、
server-asr-offline（分块规划、假 runner、真实 runner）、server-asr-align（分词/时间戳/前向）、
server-asr-engine（能力检测、懒建上下文、jobs/realtime、上下文互斥）、
server-asr-realtime 与 server-asr-realtime-runner（协议、静音检测、稳定提交）、
server-asr-api（/v1/asr/* handler）、server-http 的 post_stream。

## 关键决策

1. 两条独立 API：/v1/asr/transcriptions（202 + 轮询）与 /v1/asr/realtime（WS）；既有端点不变。
2. 专用 ASR llama_context（懒创建、seq=1），借用 server 的 mtmd；对齐器自带 mtmd 与上下文。
3. 对齐器为 Qwen3VL + 分类头变体（cls.output 5000 类），走逐 token embedding，不用词表 logits。
4. 稳定提交：safe = min(LCP, len(raw) - holdback)，holdback 同时管 prompt 回滚与提交边界。
5. 段重置：优先静音锚点（全提交、无重叠），否则最低能量 100 ms + 1 s 重叠 + 首帧去重。
6. 实时优先：realtime_active 让离线在 chunk 边界暂停，实时结束后先恢复暂停任务。

## 与 spec/plan 的偏离（已写入文档）

1. 404 body：httplib 对 >=400 都调用 error handler，原实现无条件覆盖 404；改为仅 body 为空时填默认值。
2. 对齐器输入以官方 encode_timestamp 为准：audio + 每个词后跟两个 ts，结尾再一对，保持因果注意力。
3. 实时元数据只在段首出现：前 12 个 token 内见到 <asr_text> 才剥离，否则整段视为正文。
4. 假 runner 用环境变量：LLAMA_SERVER_ASR_FAKE_RUNNER=1、LLAMA_SERVER_ASR_FAKE_STEP_MS。
5. cancel 返回枚举 NOT_FOUND/CANCELLED/CANCELLING/DELETED，便于 DELETE 映射 404/200/202/204。
6. 新增 server-asr-audio.*；Python 测试需加 --confcutdir=tools/server/tests/unit 绕开预设下载。

## 验证（实测通过）

- C++：ctest -L behavior -R "test-asr|test-asr-aligner" 3/3；test-asr 181+ 断言无需模型。
- Python 非 heavy 全套（--confcutdir=tools/server/tests/unit -m "not heavy"）：23 passed，73 s。
- Heavy：2 小时离线 258 s（220 MiB 上传、>=20 chunk）；1 小时实时 571 s（3600 次解码）；
  10 分钟实时 102 s。
- 抢占：280 s 任务运行中开实时，0.06 s 内 paused，结束后回到 running 并 succeeded，
  processing_ms 不含暂停。
- 回归：/v1/chat/completions 带 input_audio、/v1/audio/transcriptions、离线任务运行中再发聊天，3 passed。
- 环境变量：QWEN3_ASR_MODEL、QWEN3_ASR_MMPROJ、QWEN3_ASR_ALIGNER_MODEL、QWEN3_ASR_ALIGNER_MMPROJ、
  QWEN3_ASR_AUDIO、QWEN3_ASR_ALIGN_AUDIO、QWEN3_ASR_DEVICE=CUDA0。
- 性能（0.6B q8，RTX 4060）：30 s 音频离线 0.9 s；离线约 0.03 s / 音频秒；实时约 0.16 s / 音频秒。

## 收尾建议

- 两项 1.7B 延迟目标未测（本机无 1.7B 权重）：可按 spec 13 节用 --hf 拉 1.7B Q8 复测。
- 后续可选：日/韩分词细化、v1 之后的重复文本清理。

## 已知限制

- 日/韩分词为逐字近似；v1 不做重复文本清理。
- 对齐器单次 <=180 s；时间戳模式 chunk 上限由两个上下文共同决定。
- ASR 不经过 router（统一 501）；仅 WAV/MP3/FLAC；CPU 可跑但慢。
- 1.7B 模型的延迟目标未实测。
