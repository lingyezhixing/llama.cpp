import io, os, shutil

T = os.path.join(os.environ['TEMP'], 'v100')
A = r'D:\LLM\Backend\v100-collab\artifacts'
for f in ['dec4gr.sqlite', 'dec8.sqlite', 'q_dec4gr.py', 'q_dec4gr2.py', 'q_sig.py', 'q_dec8.py', 'q_dec8b.py',
          'analyze_nodes.py', 'analyze_nodes2.py', 'analyze_nodes3.py', 'analyze_nodes4.py', 'nodes_raw.txt',
          'trace_ops.patch', 'build_harness.cmd']:
    src = os.path.join(T, f)
    if os.path.exists(src):
        try:
            shutil.copy(src, os.path.join(A, 't05r_' + f) if f in ('dec4gr.sqlite', 'dec8.sqlite') else os.path.join(A, f))
        except Exception as e:
            print('skip', f, e)
print('artifacts copied')

t = """
---

# T05 剩余 (quantize 融合 + 小 kernel 合并) 结案: 无可接受风险的实质收益 | CLOSED (evidence-based)

工作区 0 净改动 (rms_norm 实验已回退), PPL 4.3569 复核通过, 4 文件交付态。

## 方法修正 (重要, 影响后续所有 decode profile)

- nsys 默认 `--cuda-graph-trace=graph`: **图重放的 token 的 kernel 根本不进 kernel 表**。
  llama-bench tg128 里每个 token 走 cudaGraphLaunch (实测 7 次 graphLaunch / 8 token, 每个 ~1.5-2.4ms),
  默认 trace 只留下图之前 1-2 个 token 的 kernel 事件 -> analyst 原 T05 profile 是"首 token / 图前"数据。
  **正确做法: `nsys profile --trace=cuda --cuda-graph-trace=node`** (本次已用)。
- 首 token 与稳态差异已被量化 (见下): `scale_f32` 0.64 -> 0.15ms (96 个 ne=786432 的 state 清零 SCALE 只在首 token/rs_z>=0 时出现)。

## 稳态单 token 分解 (graph-replay 窗口, 2024 kernels, busy 34.33ms; wall 37.7ms)

| 类别 | calls | ms | %busy |
|---|---:|---:|---:|
| mul_mat_vec_q (全部) | 461 | 29.84 | 86.9% |
| quantize_q8_1 | 461 | 0.84 | 2.4% |
| rms_norm (3 变体) | 305 | 1.10 | 3.2% |
| elementwise 合计 (add/cpy/silu/sigmoid/softplus/concat/rope/fwht) | 460 | 1.03 | 3.0% |
| k_get_rows | 97 | 0.47 | 1.4% |
| gated_delta_net | 48 | 0.34 | 1.0% |
| flash_attn_ext_vec | 16 | 0.32 | 0.9% |
| scale_f32 (predelta 2/层) | 96 | 0.15 | 0.4% |
| set_rows (KV q8_0) | 32 | 0.11 | 0.3% |
| **kernel 合计** | **2024** | **34.33** | 100% |
| host/graph-submit 间隙 | - | ~3.4 | - |

## MMVQ 已饱和 (逐矩阵实测带宽)

| 矩阵 (Q6_K/Q5_K/Q8_0) | grid | calls | avg us | 实测 GB/s |
|---|---:|---:|---:|---:|
| lm_head Q8_0 (248320 行) | 248320 | 1 | 1598 | **845** (=94-99% 可用) |
| ffn down / attn o_proj Q6_K | 5120 | 55 | 98 | 746 |
| ffn gate 或 up Q6_K | 17408 | 25 | 102 | 713 |
| gate+up 融合 GLU Q6_K | 17408 | 26 | 213 | 685 |
| ffn Q5_K 部分 | 17408 | 26 | 82 | 752 |

- MMVQ 平均 677 GB/s (可用 825-850), 分矩阵 685-845 -> 剩余空间约 5-8% 且分布在各小矩阵,
  改 vec_dot/访存模式属研究级改动; 与 analyst "纯 kernel 路线到顶" 结论一致。

## 两个对照实验 (均为可逆, 已回退/未入库)

1. **rms_norm block 配置** (env GGML_CUDA_RMS_NORM_BS, 把 5120 宽 norm 从 block=1024 换成 256; 129 个单行 norm 占 0.73ms, 5.62us/次):
   - BS=1024: tg128 26.69/26.60, pp512 968.6/964.0
   - BS=4096 (即 <256>): tg128 26.63/26.68, pp512 963.2/962.5
   - **结论: 无效果 (噪声内)** -> 单行 norm 的 5.6us 是"跟在带宽饱和 kernel 之后的延迟+节点开销", 不是 block 配置问题。未入库。
2. **GGML_CUDA_DISABLE_FUSION=1** (量化"融合机制"的边际价值):
   - 关闭: tg128 25.93/25.97, pp512 950.4/945.2
   - 开启: tg128 26.63 (-2.6% when disabled), pp512 963.6 (-1.7%)
   - **结论: 现有全部融合 (GLU/norm/softplus/ssm_conv/GDN, 每 token 533 个融合头) 总共只值 0.70ms**;
     折算 **每消除 1 个 kernel 的边际价值约 0.9-1.0us**。

## 结案判定 (定量)

- 剩余可压缩项上限 = 461 (quantize) + ~590 (小 kernel) 个 kernel, 按 0.9-1.0us/kernel 折算 = **1.0-1.1ms = +2.7-3.0%** -> tg128 **27.3-27.5**。
- analyst 原估 "quantize 0.8 + 小 kernel 1.0 + 间隙 0.5 = -4.1ms / 28.5-30 t/s" 中的 0.8ms 与 1.0ms 两项,
  按实测边际价被高估约 2-4 倍; 间隙 3.4ms 是 host 侧 (logits 取回 + graph submit), 与 kernel 数几乎无关。
- 要到 28.5-30 必须把 ~1500 个小 kernel 合并进大 kernel (核心级重写, 多日, 高风险), 或 MMVQ 再快 10% (已饱和)。
- **建议: T05 剩余 CLOSED (无实质收益/风险比不成立); decode 侧 kernel 天花板约 27.3-27.5;**
  decode 的实际翻倍已由 MTP (42.7 t/s) 提供, 不建议再投 kernel 侧。

## 产物

- `artifacts/t05r_dec4gr.sqlite` (graph-node 级 trace, 稳态 token), `t05r_dec8.sqlite` (默认 trace, 对照)
- `artifacts/q_dec4gr*.py`, `q_sig.py`, `analyze_nodes*.py`, `nodes_raw.txt` (node->kernel 对应与分解脚本)
- `artifacts/trace_ops.patch` = GGML_CUDA_TRACE_OPS 诊断补丁 (打印图内 node 名/op/尺寸; 仅诊断, 未入库)
"""
io.open(r'D:\LLM\Backend\v100-collab\RESULTS.md', 'a', encoding='utf-8', newline='').write(t)
print('RESULTS appended')

p = r'D:\LLM\Backend\v100-collab\TASKS\T05-decode-profile.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## 剩余项结案 (implementer, 2026-09-22): CLOSED (no material gain, evidence-based)

- 方法修正: nsys 必须用 `--cuda-graph-trace=node`, 否则图重放 token 的 kernel 不进 kernel 表 (原 profile 只覆盖图前 1-2 token)
- 稳态分解: MMVQ 29.84ms/86.9% (逐矩阵 685-845 GB/s, 已饱和); quantize 0.84; rms_norm 1.10; elementwise 1.03;
  get_rows 0.47; GDN 0.34; FA 0.32; 合计 34.33ms/2024 kernels + 3.4ms host 间隙
- 实验1: rms_norm block 配置 (1024->256): 无效果 (tg128 26.6-26.7 两侧一致), 已回退
- 实验2: 关融合: tg128 -2.6% (25.93 vs 26.63), pp512 -1.7% -> **现有全部融合只值 0.70ms, 边际 ~0.9-1.0us/kernel**
- 判定: 剩余项上限 1.0-1.1ms (+2.7-3.0%) -> tg128 27.3-27.5; 28.5-30 需核心级 kernel 合并 (多日) 或 MMVQ 再快 10% (饱和)
- 建议: 不再投入; decode 侧天花板 ~27.3-27.5 (kernel); 生产 decode 已由 MTP 承担
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS updated')

t = """
---

## 2026-09-22 | T05 剩余结案: CLOSED (无实质收益) | implementer

按用户指示做了 T05 剩余 (quantize 融合 + 小 kernel 合并 + 间隙)。**结论: 无可接受风险的实质收益, 建议关闭。**

关键实测 (完整见 RESULTS "T05 剩余结案"):
1. **方法修正**: nsys 默认 `--cuda-graph-trace=graph` 会隐藏图重放 token 的 kernel -> 原 T05 profile 是"图前 token"数据。
   用 `--cuda-graph-trace=node` 重测得到稳态分解: MMVQ 29.84ms (86.9%) + quantize 0.84 + rms_norm 1.10 + elementwise 1.03
   + get_rows 0.47 + GDN 0.34 + FA 0.32 = 34.33ms/2024 kernels, host 间隙 3.4ms。
2. **MMVQ 已饱和**: 逐矩阵 685-845 GB/s (lm_head 845 = 94-99% 可用; 平均 677/825-850) -> 无 >=5% 空间。
3. **对照实验**: (a) rms_norm block 配置 1024->256 无效果 (已回退); (b) **关掉全部融合 tg128 仅 -2.6% (25.93 vs 26.63)**
   -> 现有 533 个融合头总共只值 0.70ms, **边际价值 0.9-1.0us/kernel**。
4. **定量判定**: 剩余 461 quantize + ~590 小 kernel 全合并也只有 1.0-1.1ms = +2.7-3.0% -> tg128 **27.3-27.5**;
   原估 -4.1ms/28.5-30 需核心级 kernel 合并 (多日, 高风险) 或 MMVQ 再快 10%。

建议: ① T05 剩余 CLOSED, decode kernel 侧天花板 ~27.3-27.5 ② 不再投入 decode 侧
③ 队列只剩 T04 (ub +21%, 待用户口径) 与最终统一回顾。工作区 0 净改动, PPL 4.3569 复核通过。
"""
io.open(r'D:\LLM\Backend\v100-collab\QUESTIONS.md', 'a', encoding='utf-8', newline='').write(t)
print('QUESTIONS appended')
