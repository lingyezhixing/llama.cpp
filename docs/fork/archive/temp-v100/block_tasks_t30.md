
## T30 最终结果 (implementer, 2026-09-26): REJECTED

Phase A 测量结论: verify 形状 (n_q=4, l=128K, V q8_0) 下 **VEC 比 TILE 慢 3.1x** (nsys 直测 4.68 vs 1.50 ms/层),
即使 V 物化被完全消除 (VEC 窗口内物化实例 = 0); 提高 KV 切分 (PB/NBATCH=160/1024, 13 -> 160 片) 也慢 15%。

A/B 矩阵 (server MTP3, d131072=128270 tok, 150 token, 同 session A0->A3->A2->A1->A0b, tps 冷/热):
A0 基线 20.06/20.13; A3 (VEC+PB+NBATCH) 12.48/13.02; A2 (PB+NBATCH) 15.26/17.31; A1 (VEC) 12.78/13.81;
接受率不变 (0.3883 / 0.3835), d0 各配置 49.8-52.5 t/s, PPL 控制位 4.3567 **逐位不变** (decode 路径未动)。

按规格 "若 VEC 本身更慢 -> 如实报结果" 与 "未达门槛写 REJECTED + 数据" 结案, Phase B/C 未进入。
备选 (未做, 大工作量): 新 TILE 内核直读 q8_0 V, 上限 ~10% @128K。长文接受率衰减 (0.388 vs d0 0.808) 属 T21。
详见 `RESULTS.md` "T30 Phase A/B"。实验改动已回滚, 工作区干净; 复现脚本与 nsys 报告见 RESULTS 第 7 节。
