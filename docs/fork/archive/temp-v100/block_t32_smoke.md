
## S2a 冒烟实验结果 (implementer, 2026-09-27): 门通过; 附带发现 T24 ReplaySSM 回滚不确定性 (与共享无关)

工具: `tests/test-t32-smoke.cpp` (临时 harness, 私有 fork, 不进 PR; 目标 `test-t32-smoke`);
模型: Qwen3.5-2B (hybrid=1) 做正确性; Qwen3.8-27B 做预算。
方法: 共享前缀 P (~206 tok) + 4 分支各自续写 32 步 (teacher-forced; 每步记录 argmax + 全量 logits 对比);
`ref` = 4 条独立序列各自 prefill; `shared` = prefill 一次 + `seq_cp` x3 (同批 4 seq 一起 decode, 形状一致)。

- **item1 (共享前缀正确性)**: PASS, 全部 step logits 逐位一致 (max diff 0.000e+00)
- **item2a (删掉 seq0 对共享前缀的 cells)**: PASS, 逐位一致 (refcount 保护正确; 其它分支不受影响)
- **item2b (杀掉一个分支, 再从另一分支 seq_cp 重建)**: PASS, 逐位一致
- **item3 (深度回滚 + replay)**: 共享本身正确, 但**发现引擎侧问题**: `ref-vs-ref` (无共享, 同配置跑两遍) 也偶发差异
  -> T24 ReplaySSM 回滚路径存在运行间不确定性: 回滚后 logits 偶发差 ~0.17-0.25 (仅在 top-2 差 ~0.003 时翻转 token);
  复现率 ~1/6 (value-diff), token 翻转更罕见; **replay=0 时 6/6 逐位一致; rb<=2 时无损**;
  生产同款 (n_rs_seq=3, rb=3) 6 次无 token 翻转, 1/6 value-diff; `GGML_CUDA_GDN_REPLAY_CHECK=1` 未报错
  -> 建议: (a) 记为 T24 后续调查 (独立复现命令: `test-t32-smoke -m <model> --nrs 3 --rb 3`, replay=1);
  (b) S2a 正确性验证用 replay=0 或对"近并列"容忍; (c) 生产采样 temp0.6 下影响被随机采样掩盖
- **item4 (VRAM 预算)**: 27B, prefix 206 tok, n_ctx 8192: 1/2/4 分支 peak VRAM = 21747 / 21965 / 22407 MiB
  -> 每个共享前缀的活分支 ~= **+220 MiB** (189.8 MiB/分支 = GDN recurrent 状态不可共享 + 少量 cells; 与设计预期一致)
  -> 推论: 32GB 卡 - 权重 20.95GB ~= 10GB 可用于分支; 长分支的额外成本 = 各自唯一后缀 KV (共享前缀只算一份)

**结论**: 冒烟门通过 (1/2a/2b 逐位一致; item3 的分歧与共享无关, 属既有引擎路径; item4 预算合理) -> S2a 可开工。
**顺带的产品级洞察**: `-np N` + unified KV 下, 多个 slot 的相同前缀**本来就共享 cells** (与冒烟 item1 同机制),
即"两个长会话同时常驻"可能直接用 `-np 2` + 客户端 `id_slot` 固定就成立 (省掉一切切换), 前提是 VRAM 装得下
(共享前缀下: 共享部分只一份) -> 下一步做服务器级验证实验 (零代码)。

产物: `artifacts/t32-smoke-test-t32-smoke.cpp`, `artifacts/t32_smoke_b{1,2,4}.err`
