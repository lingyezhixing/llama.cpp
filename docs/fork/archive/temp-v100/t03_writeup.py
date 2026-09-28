import io, shutil, os

# ---------- 1) archive the validated numpy reference ----------
src = os.path.join(os.environ['TEMP'], 'v100', 't03_math4.py')
dst = r'D:\LLM\Backend\v100-collab\artifacts\t03_chunked_reference.py'
s = io.open(src, encoding='utf-8').read()
hdr = '''# T03 chunked GDN: validated numpy reference (implementer, 2026-09-22)
# Status: all checkpoints PASS at machine precision (rel err ~1e-16):
#   L=1/2/8/32/64, D=8..128, single-chunk + multi-chunk (T=256, L=64)
# Algorithm (per chunk, per (seq, head)):
#   Ac      = exp(cumsum(g))                    # within-chunk cumulative decay
#   Bt      = beta / Ac                         # normalized betas
#   KKT     = K K^T
#   A_beta  = inv(I + tril(diag(beta) KKT, -1)) @ diag(beta)   # <-- triangular solve (checkpoint 2)
#   W       = A_beta @ K ; U = A_beta @ (diag(1/Ac) V)
#   Sn_new  = Sn_in + K^T (U - W Sn_in)         # normalized state; S_out = Ac[-1]*Sn_new (checkpoint 3)
#   QK      = Q K^T
#   Qtil    = Q - tril(QK, 0) @ W
#   C       = QK - tril(QK, 0) @ A_beta @ tril(KKT, -1)
#   O       = scale * Ac[:,None] * (Qtil @ Sn_in + tril(C) @ (Bt[:,None]*V))
# Notes:
#   - decay (checkpoint 1): only Ac (cumsum+exp) per chunk; no cross-chunk decay state needed
#   - implementation can replace the explicit inverse by forward substitution:
#     W_t = beta_t (k_t - sum_{s<t} (k_t.k_s) W_s),  U_t = Bt_t v_t - beta_t sum_{s<t}(k_t.k_s) U_s,
#     and one more solve with the same I+M matrix for R = A_beta @ tril(KKT,-1) (for C)
'''
io.open(dst, 'w', encoding='utf-8', newline='').write(hdr + s)
print('archived reference ->', dst)

# ---------- 2) RESULTS.md ----------
p = r'D:\LLM\Backend\v100-collab\RESULTS.md'
t = """
---

# T03 (chunked GDN 重写): 0.75 天 checkpoint 报告 | implementer, 2026-09-22

状态: **数学原型 + 三个 checkpoint 全部通过 (机器精度)**; CUDA 实现未开始 (下一 block)。

## Checkpoint 结果 (analyst 要求的三项)

| # | checkpoint | 方法 | 结果 |
|---|---|---|---|
| 1 | decay 递推与 scan 参考一致 | numpy 对拍 (随机输入, L=1..64, D=8..128) | **通过, 相对误差 ~1e-16** |
| 2 | 三角求解正确性 | WY 恒等式数值验证 + 全流程对拍 | **通过**: WY 恒等式 4.4e-16; 全流程 1e-16 |
| 3 | keep_rs_t / chunk 边界状态传递 | 多 chunk 状态交接 (T=256, L=64) 与串行版对拍 | **通过, 1.1e-15** |

验证脚本归档: `artifacts/t03_chunked_reference.py` (可直接当 CUDA 移植的 ground truth)。

## 推导出的算法 (per chunk, per (seq, head))

```
Ac      = exp(cumsum(g))                                  # 块内累计 decay (checkpoint 1)
Bt      = beta / Ac
KKT     = K K^T
A_beta  = inv(I + tril(diag(beta) KKT, -1)) @ diag(beta)  # L x L 三角求解 (checkpoint 2)
W       = A_beta @ K            # L x D
U       = A_beta @ (diag(1/Ac) V)
Sn_new  = Sn_in + K^T (U - W Sn_in)                       # 归一化状态 (checkpoint 3)
S_out   = Ac[L-1] * Sn_new                                # 反归一化交给下一 chunk
QK      = Q K^T
Qtil    = Q - tril(QK, 0) @ W
C       = QK - tril(QK, 0) @ A_beta @ tril(KKT, -1)
O       = scale * Ac * ( Qtil @ Sn_in + tril(C) @ (Bt * V) )
```
- 关键点: **decay 是标量 (非 KDA)**, 所以把状态按块内累计 decay 归一化后, 块内递推无 decay,
  只需每块一次 cumsum+exp 和一次标量反归一化 -> checkpoint 1 极简
- 实现时不需要显式求逆: 用前代 (forward substitution) 三次, 同一个 (I+M) 矩阵, L 个右端项:
  `W_t = beta_t (k_t - sum_{s<t} (k_t.k_s) W_s)`, U 同理, 第三组解出 `R = A_beta @ tril(KKT,-1)` (给 C 用)
- 成本估算 (per chunk per (seq,head), L=64, D=128): KKT 0.52M + 三角求解 ~0.4M + 状态 2.1M + 输出 1.6M
  ≈ 4.6M MAC = 9.2 MFLOP (对比串行 kernel 每块 3.1M MAC) -> FLOP 多 1.5x 但全是规整 matmul,
  现状 kernel 只有 ~0.5 TFLOPS (指令吞吐受限), 目标 2 TFLOPS 级即可满足 -50%

## 设计 (待实现)

1. **K1 (chunk 并行)**: per (seq, head, chunk) 计算 KKT / 三次前代 / W / U / QK / Qtil / C / 块内输出项
   `tril(C)@(Bt*V)`; 存 W, U, Qtil, C (或 C 的块内输出) 到 scratch
2. **K2 (chunk 串行, (seq,head) 并行)**: 状态扫描: 逐 chunk `Sn += K^T(U - W Sn)` + 状态项输出
   `Qtil@Sn_in` (需 K, W, U, Qtil; L=64/D=128 时每 chunk 约 2.1M+1.05M MAC)
3. 合计 2 个 kernel + scratch (per chunk: W,U 各 L*D*4B + C L*L*4B ≈ 80KB @L=64,D=128)
4. 限制: 仅当 `n_tokens > 1 && !keep_rs_t && !KDA && S_v == 128` 走新路径; 其余保持现有 kernel
   (decode 单 token 路径完全不动)
5. 预期: GDN 40ms -> ~20ms (整机 pp512 +3-4%); 长上下文 (pp32768/depth32k) 同比例受益

## 下一步 (未完成, 不硬凑)

- CUDA 实现 K1/K2 + 板载三角求解 (warp/block 级前代)
- 验收: kernel 级 (目标 -50%) + 端到端 pp512/4096/8192/32768/depth32k + tg128 不回退
  + PPL |x-4.3572| <= 0.013 + 200 token 生成检查; 同 session A/B 交替 >=2 轮
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('RESULTS.md: T03 checkpoint appended')

# ---------- 3) TASKS/T03 ----------
p = r'D:\LLM\Backend\v100-collab\TASKS\T03-gdn.md'
s = io.open(p, encoding='utf-8').read()
s += """

---

## Checkpoint 报告 (implementer, 2026-09-22, 0.75 天点): 数学原型完成, 三 checkpoint 全过

- **checkpoint 1 (decay) / 2 (三角求解) / 3 (chunk 边界状态) 全部通过, 相对误差 1e-16 (机器精度)**
  - 覆盖 L=1..64, D=8..128, 单 chunk + 多 chunk (T=256/L=64)
  - 脚本归档 `artifacts/t03_chunked_reference.py` (含算法注释与验证代码)
- 算法 (per chunk): Ac=exp(cumsum(g)); Bt=β/Ac; A_beta=inv(I+tril(diag(β)KKT,-1))diag(β);
  W=A_beta K; U=A_beta(diag(1/Ac)V); Sn+=K^T(U-W Sn); S_out=Ac[-1]Sn;
  Qtilde=Q-tril(QK)W; C=QK-tril(QK)A_beta tril(KKT,-1); O=scale*Ac*(Qtilde Sn + tril(C)(Bt V))
- 实现要点: 不用显式求逆, 用同一 (I+M) 做 3 次前代 (L 个右端项) 得 W/U/R
- 设计: K1 (chunk 并行: KKT/前代/W/U/Qtilde/C/块内输出) -> K2 (chunk 串行: 状态扫描+状态项输出);
  仅 `n_tokens>1 && !keep_rs_t && !KDA && S_v==128` 启用, decode 路径不动
- 成本: 每 chunk 每 (seq,head) ≈ 4.6M MAC (L=64,D=128), 现状 kernel ~0.5 TFLOPS -> 目标 2 TFLOPS 即可 -50%
- **CUDA 实现未开始** (timebox 内剩余时间不够做完 + 验证, 按约定不硬凑); 下一步见 RESULTS "T03 checkpoint"
"""
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('TASKS/T03 updated')

# ---------- 4) QUESTIONS ----------
p = r'D:\LLM\Backend\v100-collab\QUESTIONS.md'
t = """
---

## 2026-09-22 | T03 chunked 0.75 天 checkpoint | implementer

按你的要求给 0.75 天 checkpoint (数据见 RESULTS.md "T03 checkpoint" 与 TASKS/T03):

- **三个 checkpoint 全部通过 (数学层, 机器精度 1e-16)**: decay 递推 / 三角求解 (WY 恒等式 4.4e-16) /
  多 chunk 状态交接; 覆盖 L=1..64, D=8..128, T=256; 参考实现归档 `artifacts/t03_chunked_reference.py`
- 算法与实现要点已写清 (同一 (I+M) 三次前代即可, 不需显式求逆); 设计 = K1(chunk 并行)+K2(chunk 串行)
- 成本估算: 每 chunk 约 4.6M MAC vs 串行 3.1M MAC, 但现状 kernel 仅 ~0.5 TFLOPS (指令吞吐受限),
  目标 2 TFLOPS 级即可达 -50% -> 预期可行
- **CUDA 实现未开始** (本轮剩余时间不够实现+验证, 按"不硬凑"约定停在这里)
- 问题: 是继续实现 K1/K2 (需再一个较长 session), 还是先把 T03 的 timebox 顺延/换优先级?
  你若有对 chunk 长度 L (当前建议 64) 或 scratch 预算的约束, 请直接写进 TASKS/T03。
"""
io.open(p, 'a', encoding='utf-8', newline='').write(t)
print('QUESTIONS.md appended')
