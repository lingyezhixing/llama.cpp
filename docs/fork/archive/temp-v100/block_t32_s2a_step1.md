
## S2a 第一步实施设计 (implementer, 2026-09-27): "活分支 fork" 原语 (先设计后落码)

目标 (最小闭环): 新请求到来时, 若某个**空闲 slot** 的序列与新 prompt 有很长的公共前缀 (LCP),
不再全量 prefill, 而是把该序列当作"活分支"进行一次 **fork**: 共享 [0,LCP) 的 attention cells + 复用其检查点回退
recurrent 状态 + 只解码后缀。收益: 省计算 (LCP 部分零解码) + 省显存 (共享 cells 只一份)。

机制草案 (尽量复用现有路径, 不新造):
```
fork(src_slot, dst_slot, LCP):
  1. mem.seq_rm(dst.id, -1, -1)                    // 清 dst
  2. mem.seq_cp(src.id, dst.id, 0, LCP)            // attention cells [0,LCP) 零拷贝共享 (refcount)
  3. dst.prompt = src.prompt.clone(); dst.prompt.tokens.keep_first(LCP)
     dst.prompt.checkpoints = src 的 checkpoints 中 <= LCP 的部分 (拷贝)
  4. 交给现有流程: 下一轮 batch 会自己算 n_past = LCP -> 检查点回退 (S1 机制) -> 解码后缀
     (即 fork 后 dst 的 prompt 状态 = "从 src 接过前缀", 其余全部沿用原逻辑)
```
触发条件 (v1): src idle; LCP >= 阈值 (如 512); dst 即将被覆盖 (它原来的状态可先走 prompt cache park, S1 已修);
不在 src 正在 decode 时执行 (队列单线程时序天然保证)。

待确认的代码点 (落码前必须核实, 防 COW 陷阱):
1. `llama_memory_recurrent::seq_cp` 在 p0=0,p1=LCP 时对 **recurrent tail cell** 的处理: 是否无条件共享 tail?
   若共享, 后续 `load_tgt(dst_id)` (检查点回退) 会写入共享 cell -> 需要先解除共享 (COW), 否则污染 src。
   需要读 `seq_rm`/`state_read` 的 COW 逻辑, 或在 fork 时改为"先 load 检查点再 seq_cp attention"的顺序。
   (注: `server_slot::copy_state_to` 只在"同长 prompt"的并行子任务用, 不涉及回退, 所以现成路径没暴露这个问题)
2. 检查点的 `load_tgt` 目标 seq 是否支持 `dst.id` (函数签名带 seq_id, 应可) + draft 侧 `seq_rm(dst.id, P, -1)` (S1.5 已实现同款)
3. `src.prompt.checkpoints` 的 pos 语义是否全局 (是: pos 是序列位置, 与 seq id 无关)
4. 与 spec/T24 的交互: fork 后 spec 状态 (data_spec, 20KB) 需要从 src 的检查点恢复 (S1 已接上) + smoke 已验证 cp+replay 共享正确

验收 (每步都要):
- 逐 token A/B: fork 路径 vs "独立 prefill"基线 (Qwen3.5-2B harness 已具备该对比模式, 扩一个 fork 场景即可)
- 复用现有 test-t32-smoke 的 item0/item1 对比框架; MTP 开/关各跑
- VRAM: fork 后 dst 不重复占 [0,LCP) cells (n_seq_max 允许时)
- 生产不动的回归: 现有 S1/S1.5 A/B (A36, B) 重跑不变

风险与回退:
- 最大风险 = 共享 cell 的写污染 (上面第 1 点) -> 先用 harness 证明 `seq_cp + load 检查点` 不污染 src (可直接加进 test-t32-smoke)
- n_seq_max 预算: 活分支数受 n_seq_max 限制 (提高有开销); v1 只在"现有 slot 之间"fork, 不新增 seq
- 若 COW 路径不可行: 退化为"只 fork 全前缀扩展 (LCP == src 长度)" 的简单情形 (无检查点回退, 直接共享当前状态)
