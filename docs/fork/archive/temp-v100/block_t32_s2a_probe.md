
## S2a 补充 (implementer, 2026-09-27): fork 原语已证明; 但发现"多序列布局数值不一致"未解问题

**1. fork 原语 (item5/6): PASS** — 两类都逐 token 匹配单序列基线, 且源分支不受污染:
- 全前缀扩展 (LCP == src 长度): `seq_rm(dst) + seq_cp(src,dst,0,LCP)` -> 继续解码 ✓
- 检查点回退 (LCP < src 长度): 再 `state_seq_set_data_ext(dst, ckpt, PARTIAL_ONLY)` ✓
  (源码已核实: `state_read_meta` 内部先 `seq_rm(dst,-1,-1)` 解除共享再分配私有 cell -> COW 安全)
- **前提: 必须 unified KV**。同 stream 下 `seq_cp` = `cells.seq_add` 零拷贝; 跨 stream 有 `is_full` 断言且是数据拷贝。
  注意: 我们生产与此前 T32 测试都是 `kv_unified = false` (日志确认); S2a 必须在 `--kv-unified` 下运行。

**2. 未解问题: 不同"布局路径"下 logits/token 不一致 (hybrid 与纯 attention 都复现)**
- GT = 单序列 (n_seq_max=1, batch=1); `n_seq_max=4-solo` (仅用 seq0) 与 GT **完全一致** -> 配置本身无害
- ref = 4 个独立 prefill + 交错解码; shared = 1 prefill + seq_cp x3 + 交错; shared/seq = 共享但逐 seq 单步; shared/priv = 共享 + 每分支私有 recurrent
- 结果 (first-diff, 32=全对):
  - Qwen3.5-2B (hybrid): ref=32/15/14/32, shared/il=17/32/14/32, shared/seq=32/32/14/32, shared/priv=17/32/14/32
  - Qwen2.5-Coder-3B (纯 attention): ref=32/32/32/**19**, shared 三条路径全 32/32/32/32 ✓
- 特征: 分歧只在少数位置 (14-19), 首次分歧步 logits 差 ~0.1-0.5, 之后 token 翻转级联; **共享不是唯一原因**
  (纯 attention 下共享全对, 独立 prefill 路径反而错一个分支); 也不是必现 (多数分支全对)
- 需要判定: (a) 引擎 bug (cell 布局/分配史影响 kernel 读取或归约顺序) 还是 (b) 可接受的"不同 kernel/归约顺序 -> 数值不可比"
- 影响: 树/共享分支的**验收标准** (不能简单沿用"与原路径逐位一致"); 生产是 np=1 单序列, 从未跑过多序列解码

**3. 建议 (待裁决)**
- a. 最小复现判定引擎 bug: 纯 attention, 4 seqs **相同位置区间**, 交错批量 vs solo (harness item1b 已具备; 可再缩小)
- b. 若属"数值不可比": 树验收改为 (i) 首次分歧前 logits 一致到某容差 (如 1e-4), 或 (ii) 采样输出分布等价 + PPL 控制位
- c. 工程路线不受影响的短期项: `-np 2 --kv-unified` + 客户端 `id_slot` 固定 (顺序请求) 已能覆盖"两会话轮换"场景;
  fork 落码保持**顺序解码** (避免交错批), 按 (b) 的容差验证
