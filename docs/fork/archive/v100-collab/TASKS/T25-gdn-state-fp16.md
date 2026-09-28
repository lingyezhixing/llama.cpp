# T25: GDN recurrent state fp16 (用户批准 2026-09-23)

状态: **REJECTED (用户 2026-09-23: 降低精度不采纳)**; 保留存档 (state fp32->fp16 为有损存储, 违反质量红线)
动机: 显存极紧; vLLM 同款 (`--mamba-ssm-cache-dtype float16`) 报告 PPL 不变。
收益: state 每份 144MB -> 72MB (**MTP3 共 4 份: -288MB**; 无 MTP: -72MB) + 读写减半 (decode ~0.5% 级)。

## 范围
- 状态张量 dtype F32 -> F16 (定位线索: `src/llama-model.cpp` 约 2603-2604 硬编码; `llama-memory-recurrent.cpp` 分配)
- GDN 内核 (`gated_delta_net.cu`): state 读写按 half (load -> float 计算, store -> half); conv state 是否同改待评估
- MTP 快照/回滚路径同样适配 (state 行拷贝)

## 验收 (质量红线优先)
1. **PPL** 4.3572±0.013 (预期 ±0.001 内)
2. **长文质量**: 128k 深度生成 sanity; 建议复用 T27 的 needle/PPL 语料做回归; 与 f32 版对比
   (greedy 允许近并列翻转, 记录即可)
3. 显存实测 (-72MB/行); 速度不回退 (短点 / 长点 / MTP)
4. 时间盒 **1-2 天**

## 风险
- f16 state 跨长文累积舍入: 若长文质量有可观测下降 -> 放弃 (用户红线)
- **与 T24 的耦合**: 若两者都做, 建议本项先行 (T24 的 fold 需匹配最终 dtype)
