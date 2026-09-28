# T12: 采纳 T11 (ub512 FA KV-split) - 重放集成

状态: **VERIFIED (2026-09-23 采纳入库)**; 见下方 Result
预期收益: depth32k +2.0% / pp32768 +1.1% (实测); pp512/4096/8192 ~0%; attention -5.6%

## 步骤

1. 重放 `artifacts/t11_fattn_kvsplit_REJECTED.patch` (约 13 行, 含 `GGML_CUDA_FATTN_STREAM_K` env 开关;
   保留开关便于即时回退)
2. 全套验收 (同 session A/B 交替 **>=3 轮**):
   - depth32k >= +1.5%, pp32768 >= +0.8%
   - **pp8192@depth128k (代表点, 先测基线) >= +2%**
   - pp512/4096/8192 各 |Δ| <= 0.3% (原 -0.55% 需确认是噪声)
   - tg128 噪声内; ub2048 抽查不回退 (回归保护)
3. PPL 门槛 `|x-4.3572| <= 0.013` + 200 token 生成检查
4. 通过 -> 入库 + patch 更名归档 (去掉 REJECTED); 不通过 -> 写数据回报, 不硬上
5. 记录显存增量 (stream-k fixup partials, 预期 MB 级)


---

## Result (2026-09-23, implementer): **PASS -> 采纳入库**

- 合并验收 (A=BASE `102BF844`, B=T12+T16 `453E2911`, 长点轮换顺序): **pp8192@depth128k +11.28%** (主判据 +4%),
  depth32k **+4.77%** (+1.5%), pp32768 **+2.18%** (+0.8%), 短点 +0.03~0.56% (无回退), tg128 +0.06%,
  ub2048 -0.1%, PPL **4.3562** (门槛 4.3572±0.013), 200 token 生成无重复/乱码
- 显存: stream-K/PB=2 fixup partials ≈ 12.8MB/launch (pool 复用); 128k 深度实跑无 OOM
- 交付: patch `patches/v100-t12t16-fattn-split.patch` (T12+T16 合并, 双向 apply 校验通过);
  工作区 5 文件; DLL `453E29111E5E29C9`; env 开关保留 (STREAM_K/BLOCKS/PB, 便于 T18 与回归)
- 注: T11 原 REJECTED patch 被本合并 patch 取代 (artifacts 保留历史)
