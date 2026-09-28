
## T24 部署完成 (2026-09-26, implementer; 用户 2026-09-26 批准)

- 目标: 生产目录 `D:\LLM\Backend\llama.cpp-my` (原 T19-L, ggml-cuda.dll `054BFFD625E37E04`)
- 覆盖 9 个文件 (整套 ggml/llama/server/mtmd): ggml-base, ggml-cpu, ggml-cuda, ggml, llama-common,
  llama-server-impl, llama-server.exe, llama.dll, mtmd; 其余 28 个 dll/exe 与构建产物逐字节相同, 未动
- 部署后生产 ggml-cuda.dll = `299DAFC748DE5D91` (llama.cpp-t24, 与 `3eae5cdae` 对应); 9 个文件 hash 全部与构建一致
- 备份: `D:\LLM\Backend\deploy-backup\llama.cpp-my-t19l-20260926\` (9 个旧文件) +
  `ggml-cuda-t24-final-299DAFC7.dll` (T24 最终构建的副本)
- 开关: `setx GGML_CUDA_GDN_REPLAY 1` (用户级, 新进程生效); 关闭 = `setx GGML_CUDA_GDN_REPLAY 0`
- 部署时无 llama 进程占用; 部署后 PPL 冒烟 = 4.3567 (replay 惰性, 与基线逐位一致)
- 生效需重启生产 server; 未开 spec / draft-simple / ngram 时 replay 惰性 (n_rs_seq=0)
- 状态文件: LLAMA_STATE_SEQ_VERSION 3->5, 旧 prompt cache / slot 状态文件不兼容 (显式报错)

## T30 Phase A/B: MTP 长文提速 (verify TILE->VEC + KV 切分) - REJECTED (2026-09-26, implementer)

规格: `TASKS/T30-mtp-long-decode.md`。基线 = T19-L + T24 (`llama.cpp-t24` @`3eae5cdae`), replay=1 (= 生产配置),
server MTP3 greedy seed42, 150 token, d131072 = `t20_prompt128k.txt` = 128270 tok。

### 1. 基线 (同 session, 3 轮轮换；口径 ub512)

| 深度 | prompt_n | tps (轮1/2/3) | accept | draft (acc/n) | prefill |
|---|---|---|---|---|---|
| d0 | 5 | 50.11 / 49.67 / 49.17 | 0.8077 | 105/130 | 0.26-0.36s |
| d32768 | 32041 | 30.80 / 30.23 / 28.07 | 0.445-0.457 | 85/191 | 44.6s (718 t/s) |
| d131072 | 128270 | 19.47 / 18.45 / 18.04 | 0.3883 | 80/206 | 260.1s (493 t/s) |

提示文件实际长度: `t20_prompt32k.txt` = 32041 tok, `t20_prompt128k.txt` = 128270 tok; 峰值显存 ~28.6GB (ctx 135168)。

### 2. nsys 一轮 128K verify 时间去向 (llama-cli, TILE 路径)

verify (n_q=4) 的 FA = `flash_attn_tile<256,256,4,2>`, grid=(1,13), blockX=32, **1.50ms/层**

| 类 | 时间 (2 轮窗口) | 占比 | 备注 |
|---|---|---|---|
| MMVQ (GEMM) | 80.3ms | 49% | 权重带宽受限, 合理 |
| FA TILE (verify) | 49.7ms | 30% | 全程只有 13 warp |
| V 物化 q8_0->f16 | 16.8ms | 10% | 17 层/轮 |
| GDN + rms + conv + 其它 | ~16ms | 10% | |

注: nsys 下 kernel 时长放大 ~1.4-1.5x (窗口 163ms/2 轮 vs 服务端 ~50ms/轮), 只取相对占比。
`llama-cli` 在 nsys 下每轮有数秒 CPU 停顿 (serve 不受影响), 故只用于 kernel 级计时。

### 3. 实现 (Phase B 试做, env 门控; 已回滚)

- `GGML_CUDA_FA_VEC_VERIFY=4`: Volta 上让 verify (2..4 查询) 走 VEC (q8_0 直读, 免物化)
- `GGML_CUDA_FATTN_VERIFY_PB` / `GGML_CUDA_FATTN_VERIFY_NBATCH`: 提高 verify 的 KV 切分 (13 -> 160 / 1024)
- 两者都只作用于 n_q in (1,4]; decode (n_q=1) 与 prefill 完全不动

### 4. A/B 矩阵 (server, d131072=128270 tok, 150 token, 同 session 顺序 A0->A3->A2->A1->A0b)

| 配置 | vec | pb | nbatch | tps 冷 / 热 | accept |
|---|---|---|---|---|---|
| A0 (基线) | - | - | - | **20.06 / 20.13** | 0.3883 |
| A3 | 4 | 160 | 1024 | 12.48 / 13.02 (-36%) | 0.3883 |
| A2 | - | 160 | 1024 | 15.26 / 17.31 (-15%) | 0.3883 |
| A1 | 4 | - | - | 12.78 / 13.81 (-33%) | 0.3835 |
| A0b (基线) | - | - | - | 18.22 / 19.24 | 0.3883 |

- d0 全部 49.8-52.5 t/s (不受影响, 符合预期); 接受率基本不变 (A1: 79/206 vs 80/206)
- PPL 控制位 (A3 全 env 开): 4.3567 **逐位不变** -> 证明 decode 路径确实未被动过
- nsys 直测 (VEC verify 窗口, 17 层): `flash_attn_ext_vec<256,2,1,8>` **4.68ms/层** vs TILE 1.50ms/层 = **3.1x 慢**;
  且该窗口内 V 物化实例 = **0** (确认物化确实被消除) -> 仍然大幅更慢

### 5. 结论: REJECTED (Phase A 判据达成, 未进入 Phase B/C)

1. VEC 在 n_q=4/长 l 下更慢 3.1x: `cols_per_block=2`, 不能像 TILE 那样共享 K/V tile;
   与上游启发式 (Volta 上 `n_q*gqa_ratio_eff>2` 用 TILE) 一致
2. 提高 KV 切分 (-15%) 也慢: 每块 setup / 部分结果归并 / 尾部波次开销 > 并行收益
3. V 物化只占 ~10%; 去掉它的收益远小于换核损失
4. 规格预先声明命中: "若 VEC 本身更慢, 预期 +5-8% 不成立 -> 如实报结果"
5. 另一角度: 128K 的 MTP 接受率只有 0.388 (vs d0 0.808), 才是长文收益衰减的另一半, 属 T21 (K 经济学) 范围

### 6. 未做 (备选, 大工作量, 不建议平推)

- 新 TILE 内核直接读 q8_0 V (去物化): 上限 ~10% @128K; 需新内核 + 全量 FA 回归
- verify 的 K/V 带宽下限 = 410MB/层 x 17 = ~7GB/轮 ≈ 8.2ms; 当前 TILE 1.5ms/层 (=25ms/轮) 距下限 ~3x,
  但 A2 证明"加并行"不是到达下限的途径 (需内核级重写)

### 7. 产物与清理

- 脚本: `%TEMP%\v100\` 下 `t30_base.ps1` (基线), `t30_ab.ps1` (A/B), `t30_matrix.ps1`, `t30_nsys_cli.ps1`,
  以及 `t30_kern.py` / `t30_decode.py` / `t30_round.py` / `t30_grid.py` / `t30_tl.py` / `t30_vec.py` / `t30_vecwin.py`
- 原始: `t30_base.jsonl`, `t30_ab.jsonl`, `t30_cli128k.nsys-rep/.sqlite` (TILE), `t30_cli128k_vec.nsys-rep/.sqlite` (VEC)
- 实验改动 (`fattn.cu` + `fattn-common.cuh`) 已 `git checkout` 回滚, 工作区干净; build 目录已重建为回滚后版本
- `D:\LLM\Backend\llama.cpp-t30\` = 实验构建 (`CF4B2B5E`, 含 knobs), 仅供复现, 勿用于生产
