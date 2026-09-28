# T22: 内置 ngram 查表投机实测 (零代码) - 用户批准 2026-09-23

状态: **APPROVED (用户 2026-09-23)**; 执行者: implementer; 口径 = T19 交付 (`afbab1748` / DLL `7F1B9B240343`), **无代码改动**
动机: HyperQwen 的 LABD (context-lookup) 灵感; 已确认 llama.cpp 内置 `--spec-type ngram-map-k4v`
(ngram-map/cache/mod, 注册 `common/common.h:178-182`, impl `common/speculative.cpp`)。
预期: 复述/代码类 **+20-40%**, 普通 prose +2-5%; 显存中性。

## 1. 前置确认 (~0.5h)
- 列出现有 `--spec-type` 取值 (含 ngram-*), 确认 `ngram-map-k4v` 可跑; 记录其参数
- 确认能否多类型组合 (如 `draft-mtp,ngram-map-k4v`) —— 可组合则单列一组测试
- 无损性: 接受判据仍是 target 采样相等 (`common/sampling.cpp`); greedy 应逐 token 一致
  (注意 T20 已知的 S2/S3 类 ulp 近并列翻转: 记录分叉位置即可, 不作为失败)

## 2. 测试矩阵 (server; greedy seed42 为主 + 一组 temp>0 参考; 同 session 交替)
| 场景 | prompt 类型 | 深度 |
|---|---|---|
| 复述/复制 | 逐字续写 / 清单 / 缩进结构 | d0 |
| 代码 | 函数续写 / 文档字符串 | d0 |
| 普通 prose | 创作 / 问答 | d0 |
| 长文复述 | 复制类 | d32k / d128k |

- 对照: none / ngram-map-k4v / MTP3 (生产) / (若可组合) MTP3+ngram
- 指标: tg t/s, 接受率, 每轮 token, TTFT, 显存峰值 (nvidia-smi)

## 3. 交付物
- RESULTS 章节 "T22": 命令 + 表格 + 无损性核对
- 结论: 哪些场景值得开 (生产建议); 与 MTP 组合是否可行; 显存/质量记录
