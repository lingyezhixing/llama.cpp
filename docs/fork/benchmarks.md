# Benchmarks

Reference for the private llama.cpp fork this repository (see
FORK-NOTES.md). Target model: Qwen3.8-27B-UD-Q6_K (hybrid GDN + attention). Numbers are
from one machine unless a row says otherwise; the raw log is `archive/v100-collab/RESULTS.md`.

## Hardware and methodology

| item | value |
|---|---|
| primary GPU | Tesla V100-SXM2-32GB, sm70, `CUDA_VISIBLE_DEVICES=1` |
| driver / cuBLAS | 581.80 / cuBLAS 120901 (CUDA 12.9) |
| GPU clocks | prefill 1522-1530 MHz SM, MEM 877 MHz, 150-278 W, SM util 99-100%; mixed-load median 1447 MHz |
| secondary GPU / small test models | RTX 4060 Laptop (device 0): T32 stage-3 acceptance, 0.8B media E2E; small models: Qwen3.5-2B-UD-Q4_K_XL, Qwen2.5-Coder-3B-IQ4_XS, 0.8B + mmproj-F16, Qwen3.6-35B-A3B |
| build | Windows/MSVC Release, `CMAKE_CUDA_ARCHITECTURES=70-real;89-real` |

source: archive/v100-collab/RESULTS.md (T01/T05/T19/T30/T31), FORK-NOTES.md.

Commands: llama-bench pp `-ngl 99 -fa on -ctv q8_0 -ub 512 -p 512,4096,8192 -n 0 -r 3`,
`-p 32768 -n 0 -r 2`, `-p 131072 -n 0 -r 2`; tg `-p 0 -n 128 -d 0,4096,8192 -r 3`,
`-d 32768 -r 2`, `-d 131072 -r 2`.
- llama-perplexity: `-f <text> -c 512 --chunks 8 -ngl 99 -fa on -ctv q8_0 --seed 42` (band 4.3572 +- 0.013).
- server MTP: `llama-server -ngl 99 -fa on -ctv q8_0 -ub 512 --spec-type draft-mtp --spec-draft-n-max 3`;
  greedy seed42, 150 tokens unless stated; "prod" = temp0.6/top-p0.95/top-k20.

Method notes: one llama-bench point carries about +-0.5% run noise; the project rule is
same-session alternating A/B with >=2 rounds (no cross-session comparison). Long points
(32768 / 131072) are 1-2 reps; thermal drift is real (tg128 26.6 -> 24.4 in one session).
"GPU time" is kernel time; "wall" includes host/graph gaps.

## V100 ggml-cuda optimizations

Prefill/decode curve, OURS vs STOCK on the same base `e6ab7c1a4` (ub512):

| point | STOCK t/s | OURS t/s | delta |
|---|---:|---:|---:|
| pp512 / pp4096 / pp8192 | 872.78 / 847.45 / 814.28 | 948.48 / 929.91 / 907.54 | +8.67% / +9.73% / +11.45% |
| pp32768 / pp131072 | 663.29 / 392.10 | 793.47 / 531.52 | +19.63% / +35.56% |
| tg128 d0 / d4096 / d8192 | 26.65 / 25.56 / 23.75 | 26.61 / 25.47 / 24.05 | -0.15% / -0.35% / +1.26% |
| tg128 d32768 | 22.87 | 22.92 | +0.22% (matrix -10.6% excluded; retest parity) |
| tg128 d131072 | 11.57 | 10.76 | -7.0% (unresolved; retests -3.4% / -10.3%) |

source: `archive/v100-collab/artifacts/t19_data.csv` (RESULTS.md | T19 full curve).

FATTN split acceptance (BASE `102BF844` -> T12 stream-K + T16 PB=2 `453E2911`):

| gate | before -> after | delta |
|---|---|---|
| pp8192@depth128k (main) | 337.39 -> 375.45 | +11.28% |
| depth32k (`-p 4096 -d 32768`) | 651.10 -> 682.16 | +4.77% |
| pp32768 | 774.19 -> 791.07 | +2.18% |
| pp512 / pp4096 / pp8192 | 945.5 / 932.2 / 909.2 -> 950.8 / 932.5 / 912.1 | +0.56% / +0.03% / +0.32% |

source: archive/v100-collab/RESULTS.md | T12+T16 merged acceptance. Also: tg128 26.667 -> 26.682 (+0.06%);
ub2048 pp4096 1151.4 -> 1149.9 (-0.1%); PPL 4.3562. Knobs: `GGML_CUDA_FATTN_STREAM_K` / `_BLOCKS` / `_PB`.
Attention TF/s: 29.5 baseline (ub512) -> 31.3 after stream-K -> ~34.5 estimate after PB=2 (gate 36 not reached);
ub2048 diagnostic 39.9. Attention share: pp32768 18.5%, pp8192@depth128k estimated ~60%.

Nsys kernel sweep (depth8k, 512 launches): blocks96 = 1604.3 ms (+31.9%, 2 tiles/CTA); 192 =
1216.2 baseline (1 tile/CTA); 160 = 1128.8 (-7.2%); 80 = 1112.2 (T12, -8.5%); 768 =
1073.5 (PB=4, -11.7%); 384 = 1037.9 (PB=2 adopted, -14.7%); fixup 0/0/23.0/12.1/58.8/34.2 ms.
source: archive/v100-collab/RESULTS.md | T16 sweep, T10/T16 attention share.

Per-optimization decomposition (kernel vs end-to-end vs PPL):

| optimization | kernel-level | e2e effect | PPL |
|---|---|---|---|
| K-quant vec dequant (Q6_K/Q5_K); Q8_0 (closed) | Q6_K/Q5_K 825 GB/s (92% of ~900), ffn gate/up 0.202 ms vs cuBLAS 1.199 ms; Q8_0 current 767.2 GB/s vs vectorized candidate 669.8 GB/s (bit-identical) | inside OURS curve (not isolated); Q8_0 share 1.2% -> no gain | 4.3572, bit-identical |
| FATTN stream-K + PB=2 | attention -14.7% vs T11-off (depth8k) | pp8192@depth128k +11.28%, depth32k +4.77% | 4.3568 -> 4.3562 |
| GDN vec4 (retired in T19-L) | 879.9 -> 826.3 us/layer (-6.1%) | pp512 +1.0..1.3%; tg noise | 4.3572 -> 4.3569 |
| silu vec4 (retired) | 11.42 -> 10.84 ms (-5.1%) | noise | no change |
| rms_norm vec4 (rejected) | 174.81 -> 184.85 ms (+5.7%, slower) | noise | 4.3563 vs 4.3569 |
| T19-L retirement of A2+A3 | - | pp512 -1.5%, pp4096 -1.8%, pp8192 -1.2%, pp32768 -0.9%, pp131072 -0.5%, tg +-0.2% | 4.3562 -> 4.3567 |

source: archive/v100-collab/RESULTS.md | T01, T02, T03, T07, T12, T16, T17, T19-L; T03 also archive/v100-collab/TASKS/T03-gdn.md.
MMA config facts (T16/T18): sm70 mma FA = 67584 B smem + 254 regs -> 1 CTA/SM (4 warps); ~1K
cycles mma + 1-4K LDS per KV chunk vs ~20.7K measured -> latency bound. T18 ncols=32 (35072 B
smem, 2 CTA/SM): 1.12x harness (1.22x @12k, 1.10x @100k), +0.11% @128k production -> REJECTED;
T16 occupancy variants Q_in_reg spill (cfgA +75%, cfgE +637%), nthreads=512 compile fail.

PPL chain (Qwen3.8-27B-UD-Q6_K, 512 ctx / 8 chunks / seed 42): pristine 4.3572;
+ vector dequant (Q6_K/Q5_K) 4.3572 (bit-identical); + GDN vec4 4.3569; + silu vec4 4.3569;
+ stream-K (T12) 4.3568; + PB=2 (T16) 4.3562; T19 OURS / STOCK 4.3562 / 4.3572;
T19-L (A2+A3 retired) 4.3567; T24 replay ON = OFF 4.3567 (bit-identical). 4.3572 -> 4.3569 = GDN vec4 (fp32
regroup, ~1 ulp); dequant/silu bit-identical.
source: archive/v100-collab/RESULTS.md | PPL offset trace, T19, T19-L, T24.

## Decode and MTP

Stable single-token decode (graph-replay window, 2024 kernels, 34.33 ms busy):

| class | calls | ms | % busy |
|---|---:|---:|---:|
| mul_mat_vec_q (all) | 461 | 29.84 | 86.9% |
| quantize_q8_1 + other small kernels (rms_norm, elementwise, get_rows, GDN, FA vec, scale, set_rows) | 1515 | 4.36 | 12.7% |
| host/graph submit gaps | - | ~3.4 | - |

source: archive/v100-collab/RESULTS.md | T05 revisit. Wall 37.7 ms/token = 26.5 t/s; an earlier profile gave
kernel sum 35.15 ms + 2.5 ms gaps (6.6%). tg128 baseline 26.62-26.68 t/s.

MMVQ bandwidth (config optimal): lm_head Q8_0 845 GB/s, ffn down Q6_K 746, gate/up Q6_K 713,
GLU Q6_K 685, Q5_K 752; average 677 of available 825-850 GB/s. Tuning sweeps (tg128): nwarps
4 = 26.62 (current), 2 = 25.93, 8 = 24.07; rows_per_block 2 = 26.70 (noise), 4 = 25.57. All
fusion = 0.70 ms, marginal 0.9-1.0 us/kernel; decode ceiling without MTP ~27.3-27.5 t/s.
source: archive/v100-collab/RESULTS.md | T05.

MTP baselines (server, greedy seed42, 150 tokens): d0 (prompt_n=5) 50.11/49.67/49.17 tps,
accept 0.8077, prefill 0.26-0.36 s; d32768 (32041) 30.80/30.23/28.07, accept 0.445-0.457,
prefill 44.6 s (718 t/s); d131072 (128270) 19.47/18.45/18.04, accept 0.3883, draft 80/206,
prefill 260.1 s (493 t/s); VRAM peak 28177-28595 MB, ctx 135168.

source: `archive/v100-collab/artifacts/t30_base.jsonl` (RESULTS.md | T30). A later same-build A/B session saw
the d131072 baseline at 18.22-20.13; treat the point as the 18.0-20.1 range. T30 Phase B
(verify TILE -> VEC + KV split) was REJECTED: VEC 4.68 ms/layer vs TILE 1.50 (3.1x slower)
even with V materialization eliminated; larger KV split also slower; PPL 4.3567 bit-identical.
source: `archive/v100-collab/artifacts/t30_ab.jsonl` (RESULTS.md | T30).

T31 round breakdown, 128K hot requests (prod = temp0.6/top-p0.95/top-k20):

| config | tps | accept | tok/round | round ms | draft ms | resid ms |
|---|---:|---:|---:|---:|---:|---:|
| MTP3 greedy | 19.97 | 0.388 | 2.17 | 108.1 | 17.82 | 90.3 |
| MTP3 prod | 27.57 | 0.694 | 3.12 | 112.6 | 17.21 | 95.4 |
| MTP1 greedy | 19.04 | 0.644 | 1.67 | 87.0 | 6.33 | 80.6 |
| no spec | 12.92 | - | 1.00 | 77.4 | - | - |
| MTP6 prod | 20.26 | 0.393 | 3.33 | 163.4 | 35.35 | 128.1 |

source: archive/v100-collab/RESULTS.md | T31 + analyst review (independently recomputed from raw logs).
MTP3 prod round composition: verify + host/sync ~90 ms (78%), draft 17.9 ms (16%), target
sampling 6.8 ms (6%), accept + begin ~0.3 ms; per-token 36.0-37.0 ms (2.10-2.15x vs no
spec); K=3 stays optimal. Per-position acceptance @128K, single request: MTP3 greedy
0.638/0.362/0.159; MTP3 prod 0.917/0.729/0.438; MTP6 prod 0.778/0.600/0.356/0.222/0.200/0.156
(analyst review; earlier table values were cumulative over d0+128K).

User production reference (llama-server, short prompt, n_predict=128, temp 0): no spec
25.3 t/s; draft-mtp n-max=3 42.2-42.8 t/s; acceptance 0.617, mean accepted len 2.82.
source: archive/v100-collab/RESULTS.md | T05 user-side info.

## ReplaySSM

State accounting and measured effects:

| item | snapshot path | ReplaySSM | effect |
|---|---|---|---|
| S state, n_max=3 | 4 x 144 MiB = 576 MiB | 144 MiB + 12.1 MiB records = ~156 MiB | save ~420 MiB/seq (n_max=1: 288 -> ~150, save ~138) |
| measured VRAM @MTP3 | 22018 MiB | 21598 MiB | -420 MiB (np=4 -1.37 GiB; 35B: -350 MiB np=1 / -700 MiB np=2) |
| tg @MTP3 | 71.6 t/s | 70.2 t/s | -1.9% (interleaved, 4 runs) |
| conv R cache; state version | ~23 MiB/seq; 3 (upstream) | unchanged (M3 deferred); 5 then 6 | old state files refuse to load |
| state blob after conv-plane fix | 161.8 MiB/checkpoint (27B) | ~178.6 MiB | +conv rollback planes |

source: archive/v100-collab/TASKS/T24-replayssm.md and RESULTS.md | T24. 0.8B smoke after the fix:
blob 22,580,436 -> 26,561,748 B, generation normal.

Correctness and determinism:

| item | result |
|---|---|
| fold Phase A; MTP3 replay ON vs base (no spec) | fold bitwise equal (8 cases, 64-round chain); 64/64 tokens |
| MTP3 OFF vs base; n-max1 ON vs OFF; no-MTP ON vs OFF | 64/64 each |
| PPL ON = OFF; fold self-check | 4.3567 bit-identical; 0 mismatch (`GGML_CUDA_GDN_REPLAY_CHECK=1`) |
| T32 smoke finding | ref-vs-ref value-diff in ~1/6 runs (logits 0.17-0.25); replay=0: 6/6 bitwise; rb<=2: no loss; production config (n_rs_seq=3, rb=3) 6 runs no token flip, 1/6 value-diff |

source: archive/v100-collab/TASKS/T24-replayssm.md; the ref-vs-ref finding is in RESULTS.md | T32 smoke.
Rollback findings: replay must not fold a restart batch (`pos0 == rec_pos0 + p`
continuity check), and overriding `s_copy` broke conv rollback; both were fixed.
Known limits: all GDN layers must be on CUDA; row split breaks the record layout
(FORK-NOTES.md, Portability notes).

## KV tree (T32)

Range state API and storage (Qwen3.8-27B-UD-Q6_K, q8_0 V, PCIe x4):

| item | value |
|---|---|
| full state H2D / D2H; 32K size | 2.37 GiB/s / 2.60 GiB/s; 2150.4 MiB = 2.100 GiB, 70463 B/token (f16, incl recurrent) |
| q8_0 target payload; 100K extrapolation | 50200 B/token, 512-token chunk = 24.5 MiB; 4.68 GiB -> ~2.0 s H2D / ~1.8 s D2H |
| range read / write, chunk 512/1024/2048 | read 12.8 / 23.2 / 41.5 ms/chunk (1896/2063/2309 MiB/s); write 20.3 / 34.8 / 60.1 ms/chunk (1198/1378/1594 MiB/s) |
| `--tree-chunk` default | 512 (locked by this measurement); harness fill 32000 tok in 40.2 s (796 t/s) |

source: `archive/v100-collab/artifacts/t32-stage1-bench.txt` (RESULTS.md | T32 stage 0-1).

Correctness and reuse acceptance:
- stage1: 2B hybrid 23/23, 2B -kvu 23/23, 3B attention 22/22 (exit 0). Matrix (18 runs,
  np=1/3 x quant x kvu): np=1 logits bit-identical (0.000000); np=3 non-unified 0.000000;
  np=3 unified: data PASS, cross-sequence check SKIP (logits 0.25-0.63; old full-restore 0.60).
- stage2 accept: two 4096-token sessions with 3072 shared -> 10 blocks = 6 shared (ref=2,
  stored once) + 2+2 tails, 12/12 tip@4096, tokens_reused 49152, stored 103,443,240 B
  < 2x(seq+anchor); four 1024-token sessions with 512 shared -> 5 blocks (1+4), 4/4 tip@1024.
- stage3 (device 0 = RTX 4060 Laptop): calib 417,408,732 B; ab parked=23 restored=12 miss=0,
  rammax=138,760,464, diskmax=497,257,704; overlap 0/0; b restored=12 ref4=2; b3 parked=11
  restored=0 (D11); neg 65 files removed (visible failure, request OK); heal captured=1;
  ref x2 30/30 identical.
- stage5 fork: captured=[8192,16384] restored=[8192,8192,16384]; tree vs full prompt_n
  req3 2066 vs 10258, req5 2070 vs 18454; 5/5 bitwise.
source: `archive/v100-collab/artifacts/t32-stage1-correctness.txt`, `archive/v100-collab/artifacts/t32-stage1-correctness-matrix.txt`,
`archive/v100-collab/artifacts/t32-stage2-accept.txt`, `archive/v100-collab/artifacts/t32-stage3-accept.txt`, `archive/v100-collab/artifacts/t32-stage5-fork.txt`; device note from
`archive/v100-collab/artifacts/t32-stage3-sdd/task-5-report.md`.

Soak and long-run (V100 unless noted):
- 30 min (stage4): rounds=1703 parked=1194 restored=432 rebuilt=336 failed=0 evict_refused=96
  diskmax=536,460,856 cmp_ok=340 cmp_bad=0; rss 1902->1894 MB, handles 245->271,
  wr 87767 / rd 25003 MB; restart files 79 -> 0.
- 5 min: stage4 rounds=293 parked=211 restored=94 rebuilt=79 cmp 58/58; stage5 rounds=302
  parked=219 restored=254 rebuilt=143 cmp 60/60; merged 6 min cmp_ok=70 cmp_tie=1 cmp_bad=0,
  restart 51 -> 0.
- near-tie policy: round 315 had one tree-vs-full greedy flip (logit delta ~0.015 nats;
  full-path top-2 gap 0.002); delta < 0.05 nats counts as tie, not mismatch.
source: `archive/v100-collab/artifacts/t32-stage4-soak-30.txt`, RESULTS.md | T32 stages 4-5 and merged soak.

27B + MTP + tree (production config): round 1 full prefill (prompt_n 10260-17433,
13.2-22.8 s); round 2 conversion (restore at 1024 guess anchor, capture at fork); round 3+
steady prompt_n=6, prompt_ms 291-300, wall 2.5-3.0 s per request (reuse >99.9%, ~7x
end-to-end); MTP acceptance 18-21 / 30-39 (same as round 1); VRAM 24839 MB stable,
SSD 4.4 GB / 125 files, fails=0.
source: archive/v100-collab/RESULTS.md | T32 27B runs (archive/v100-collab/artifacts/t32-27b-mtp-tree-3rounds.txt).

Whole-blob reuse A/B (S1, V100, MTP off): A36 (32K + 36 steps) baseline restore falls to
D-516 (31838), prompt_n=3274 / 5.68 s; fix keeps the anchor at D-4 (32350), prompt_n=2762 /
5.10 s (-512 tok, -10%). B (two sessions rotating, cache-ram 1100) baseline 5.4-5.7K tok /
6.7-7.3 s per switch with 6x "removing oldest entry"; fix 71 tok / 0.6 s per switch, final
4 tok / 0.23 s (~12x TTFT). S1.5 only (checkpoint slim): with MTP 162.97-302.63 MiB ->
constant 161.769 MiB (152 points; B 53 points all 161.77); blob eviction 1180 -> 950 MiB;
100K extrapolation 555 -> 162 MiB per point (32 points: 17.3 -> 5.1 GiB; 8 checkpoints
1.3 GiB). Correctness: A 22 + B 8 requests bit-identical; S1.5 A36 38/38, B 8/8 bitwise.
source: archive/v100-collab/RESULTS.md | T32 S1 and S1.5 sections.

Media E2E (0.8B-MTP + mmproj-F16, RTX 4060): same image session, 2nd request parked
1094 -> restored 1059 (prompt_n 1063 -> 25); different image -> restore miss + full
prefill; two adjacent images parked 2157 / restored 2090 (prompt_n 22); 5 requests
byte-identical to the no-tree baseline. Known gap: anchors carry no MTP/spec state (D2).
Also measured: `-np 2` alone gives no cross-sequence sharing (slot B still full-prefills
23076 tok in 28.5 s); inside a slot, incremental eval is 33 tok / 0.6 s.
source: archive/v100-collab/RESULTS.md | T32 media E2E (archive/v100-collab/artifacts/t32-media-e2e-20260928.txt) and np=2 experiment.

## A/B summaries

t30_base.jsonl (baseline, MTP3, replay=1):

| id | tps per round | median | accept | full prompt_ms |
|---|---|---|---|---:|
| d0 | 50.111 / 49.674 / 49.165 | 49.674 | 0.8077 | 264.4-360.1 |
| d32768 | 30.796 / 30.233 / 28.073 | 30.233 | 0.445-0.457 | 44,628.7 / 45,152.3 |
| d131072 | 19.473 / 18.453 / 18.040 | 18.453 | 0.3883 | 215,689.0 (96745 tok) / 260,074.9 (128270 tok) |

source: `archive/v100-collab/artifacts/t30_base.jsonl`.

t30_ab.jsonl (Phase B matrix, d131072 cold/hot):

| tag | config | cold tps | hot tps | accept |
|---|---|---:|---:|---:|
| A0 | baseline | 20.063 | 20.133 | 0.3883 |
| A1 | VEC verify | 12.777 | 13.809 | 0.3835 |
| A2 | PB=160, nbatch=1024 | 15.264 | 17.305 | 0.3883 |
| A3 | VEC + PB + nbatch | 12.476 | 13.021 | 0.3883 |

source: `archive/v100-collab/artifacts/t30_ab.jsonl`. A0b (baseline repeat, drift): cold 18.221, hot 19.243,
accept 0.3883.

T32 S1 A/B (tool-step prompt_ms median, first prefill excluded):

| scenario | metric | baseline | fix | note |
|---|---|---|---|---|
| A20 (20 steps) | step median | 698.45 ms | 697.50 ms | k2-final both 1530 tok; 2938.0 vs 2969.0 ms |
| A36 (36 steps) | step median | 1007.6 ms | 799.05 ms | k2-final 3274 tok / 6702.4 ms -> 2762 tok / 5466.6 ms |
| A36, S1.5 only | step median | 1007.6 ms | 755.75 ms | k2-final still 3274 tok (no pin), 6125.0 ms |
| B rotation | switch median | 6766.45 ms (5480-5718 tok) | 617.65 ms (71 tok) | final switch 4 tok / 211.5-215.0 ms |

source: `archive/v100-collab/artifacts/t32_A20base2/A20fix2`, `archive/v100-collab/artifacts/t32_A36mBase/A36mSlim/A36s15`, `archive/v100-collab/artifacts/t32_B2base2/B2fix2` jsonl
(plus `BmBase/BmSlim/Bs15`, same pattern on the MTP3 run). Tag map: `A20base2`/`A36mBase`/
`B2base2`/`BmBase` = pre-fix; `A20fix2`/`B2fix2`/`A36mSlim` = S1 full; `A36s15`/`Bs15` =
S1.5 only. In the shorter A20 case the baseline anchor survived, so base and fix are equal.

## Caveats

- Single machine, single V100 (plus RTX 4060 Laptop for T32 stage-3 acceptance and 0.8B
  media E2E). No multi-GPU, Linux, or MIG numbers; row split breaks ReplaySSM and layer
  split was never run (FORK-NOTES.md).
- Run-to-run variance about +-0.5% per llama-bench point; session drift up to ~1%; thermal
  drift makes sustained comparisons order-sensitive. Long-context points are single runs.
- Not measured: other quantization types (Q8_0 was tested and closed), other model families
  (only 2B/3B/0.8B/35B spot checks), np>=5, EAGLE3/DSpark/KDA (no weights), SWA hybrid edge
  cases, the 60 min soak, and media with stock checkpoints/spec gating.
- Harness fidelity matters: T18 harness gains (1.10-1.22x) did not reproduce at 100k+ in
  production; FA harnesses must be validated at the target length.

Conflicting or corrected figures:

| item | value A | value B | handling |
|---|---|---|---|
| T19 tg points | d32768 matrix 18.93 vs 21.18 (-10.6%) | d32768 retest 22.92 vs 22.87 (+0.2%); d131072 -7.0% vs retests -3.4% / -10.3% | matrix excluded; d131072 unresolved (suspected <=3% regression) |
| T30 d131072 baseline | 18.04-19.47 tps (t30_base) | 19.5-20.1 summary; A0/A0b 18.22-20.13 | session drift; use the 18.0-20.1 range |
| T31 corrections | token cost 38.4 ms/token ("-50%" vs no spec); per-position values cumulative 0.725-0.779 / 0.451-0.504 / 0.280-0.354 | 36.0-37.0 ms/token (2.10-2.15x); single request 0.638 / 0.362 / 0.159 (greedy) | analyst correction authoritative; use single-request values |
| T31 cold vs hot | no-spec cold 100.3 ms vs hot 77.4 ms (+30%) | MTP3 cold ~ hot (0-3%) | unexplained; hot values used for cross-config comparison |
| T18 / T16 | harness 1.10-1.22x; adopted estimate ~34.5 TF/s | production 128k +0.11%; gate was 36 TF/s | harness gap at long l; gate not reached but e2e acceptance passed (T18 rejected) |
| T19-L session bias | same-session T19 rebuild pp512 962.8 | historical T19 948.5 | use same-session deltas, not absolute values |
| T02 vs T01 cuBLAS | T01 first: 84.2 TF, "no call-side room" | T02 retest: default 77.4, cublasLt 84.3 | 7-9% existed only in the transposed harness; llama.cpp direction showed no gain (T08 rejected) |

source: archive/v100-collab/RESULTS.md | T16, T18, T19, T19-L, T30, T31; archive/v100-collab/TASKS/T16-parallel-blocks.md,
archive/v100-collab/TASKS/T18-fa-mma-rewrite.md.
