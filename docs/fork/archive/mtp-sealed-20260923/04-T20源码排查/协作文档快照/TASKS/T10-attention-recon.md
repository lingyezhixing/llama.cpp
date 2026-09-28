# T10: 长上下文 attention 侦察 (只测量, 不改代码)

状态: DONE (implementer 2026-09-22; 只测量, 0 行代码改动) -> 结论: 不值得移植 split-D/N32
预期收益: 未知; 侦察若显示 attention >= 15% 且 TFLOPS 低, 长上下文潜在 +5-15%
成本: ~1 小时 (nsys), 与 T07 rms_norm 并行

## 动机 (外部参考)

1Cat-vLLM (REFERENCE-1cat-vllm.md 第 1.2 节) 的 FA-V100 在 D256/GQA6 (与我们相同形状):
prefill 29-38 causal TF/s, 长 prefill 专用核 77 logical TF/s, split-D/N32 相对 generic FA2 改进 1.23-1.6x。

我们的线索:
- pp4096@depth32k = 649 t/s vs pp4096 = 936 (-31%, 差约 1935ms)
- 粗反推: 该差额对应 attention ~14.5 TF/s -> 有 ~2x 空间的嫌疑 (待证实)
- pp32768 = 775 t/s 的衰减同理

## 任务 (只测量)

1. nsys 分解 **pp4096@depth32k**: `flash_attn_ext*` 的 ms / GFLOP / TFLOPS / 占比
   (对照 pp4096 无深度)
2. nsys 分解 **pp32768**: attention 占比与 TFLOPS
3. 记录实际命中的 FATTN kernel 名字/变体 (sm70)
4. 结论必须明确回答: 值不值得移植 1Cat 的技术 (split-D/N32), 给出工程量与预期区间

## 产物

- RESULTS.md 追加 (原始数字)
- `artifacts/` 存 nsys kernel-summary CSV

## 验收

- 给出 attention 的 ms / GFLOP / TFLOPS / 占比 (两个上下文各一份)
- 明确结论: "值得动" (给预期) 或 "不值得" (给依据)
- 不改任何代码

---

## Result (implementer, 2026-09-22)

已完成 1/2/3/4 全部要求, 完整数据见 RESULTS.md "T10" 章节, 原始 CSV 见 artifacts/t10_*_kern_sum.csv。

速览:
- attention 占比: pp4096 3.3% / pp32768 18.5% / pp4096@depth32k 32.0%
- attention 效率 (有用 causal FLOP): d0-pp4096 23.7 / pp32768 28.7 / depth32k 29.5 TF/s (ub512),
  **ub2048 同场景 39.9 TF/s**
- 变体: `flash_attn_ext_f16<256,256,32,2,0,0,0>` (mma, grid 恒 192x1x1, block 32x4x1), 无 vec/stream-k/sparse
- 深度不衰减 (逐 ubatch 25.9 -> 29.7 TF/s)
- **裁决: 不值得移植 split-D/N32** (ub2048 已达 39.9 TF/s, 超 1Cat 的 29-38 区间)
- 近路 (可选, 需 analyst 批准): 让 KV-split (parallel_blocks, 代码已存在但被 stream_k 分支绕过)
  在 ub512 生效 -> 预期 depth32k +5-8% / pp32768 +3-5% / pp8192 +1.2%, 0.5-1 天


---

## 近路 Result (implementer, 2026-09-22): ub512 KV-split (stream-K) = REJECTED

- 改动 +13 行 (`should_use_stream_k`, env `GGML_CUDA_FATTN_STREAM_K`), nsys 机制验证通过:
  grid 192 -> 80, fixup kernel 出现, eval attention 1899.4 -> 1792.7ms (-5.6%, 29.5 -> 31.3 TF/s)
- 端到端 A/B 2 轮交替: depth32k **+2.0%** (gate >=3% FAIL), pp32768 **+1.1%** (gate >=2% FAIL);
  pp512/4096/8192 噪声内, tg128/ub2048 无回退, PPL 4.3568 安全
- 判定: 不达门槛 -> 已回退, patch 归档 `artifacts/t11_fattn_kvsplit_REJECTED.patch`, 转 T03
- 学习: 波次效率模型高估 (每 block 2.4 tile + seam 归并抵消大半); ub2048 的优势主要是并行度总量与
  更少 launch, 不是单纯打尾
