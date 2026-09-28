# Development history

Scope: this document covers the private V100/27B effort on this llama.cpp fork from 2026-09-22 to 2026-09-28, as recorded in the v100-collab channel log. The machine is an RTX 4060 Laptop 8 GB (mmproj only) plus a Tesla V100-SXM2-32GB (sm70, `CUDA_VISIBLE_DEVICES=1`, the only compute device used). The target is `Qwen3.8-27B-UD-Q6_K.gguf` (qwen35 hybrid: 48 gated-delta-net blocks + 17 full-attention blocks, 21.97 GB) at the fixed acceptance setting ub512. The end goal is agent-session KV reuse: stop re-prefilling long agent conversations. Facts are taken from the listed sources; where a source is ambiguous the text says "per channel log".

## Timeline

### 2026-09-22: bootstrap and benchmarks (T01-T09)

The collaboration fixed its ground rules first: acceptance at ub512, a PPL gate `|PPL - 4.3572| <= 0.013` plus a 200-token generation check, and A/B alternating runs because single measurements within +/-0.5% are noise. Baseline: pp512 950 t/s, tg128 26.64 t/s, PPL 4.3572 (the pre-existing K-quant vector dequant was already in the working tree; the handoff report had pp512 788 -> 950 after that change, +20.6%).

T01 confirmed the cuBLAS wall (84-107 TF after warm-up; the reported "75.6 TF / 60%" was an ncu locked-clock artifact). T02 (fused dequant + fp16 MMA) failed Gate A: the fp16 skeleton reached 57.5 TF and the mma path alone (1.472 ms) was already slower than the dequant+cuBLAS baseline (1.404 ms). T09-A (quantized operand) then failed three independent ways; the fusion route was declared permanently closed (user Q8). T08 measured cublasLt per shape in the real orientation: weighted -1.11% of GEMM, and every explicit algo hint was slower, closing the cuBLAS call side. T05 profiled decode: MMVQ is 29.84 ms of a 34.33 ms busy token (86.9%) at 685-845 GB/s, i.e. saturated; the kernel-only ceiling is 27.3-27.5 t/s. T07 vectorized silu (-5.1% kernel) but rejected rms_norm vectorization (kernel total 174.81 -> 184.85 ms, +5.7%). User Q4 skipped T06 because production already ran `--spec-type draft-mtp --spec-draft-n-max 3` at 25.3 -> 42.7 t/s.

### 2026-09-22/23: V100 ggml-cuda optimizations (T07/T10-T19)

The adopted sm70 work: A1 K-quant vector dequant (471 -> 707-825 GB/s; pp512 788 -> 950, +20.6%), A3 silu float4 (-5.1% kernel), and A4 FATTN long-context split. T10 measured attention at d0 3.3% / pp32768 18.5% / depth32k 32.0% and found the real problem was ub512 scheduling (grid 192 = 3 waves over 80 SMs), not kernel efficiency (ub2048 with the same kernel: 39.9 TF/s vs 29.5). T11 (stream-K KV split) initially missed its gates (depth32k +2.0%, pp32768 +1.1% vs 3%/2%) and was reverted, but user Q10 adopted it as a positive gain; T12 replayed the integration and T16 added PB=2 (attention -14.7% vs T11-off, -6.7% vs T12 default). Combined acceptance: pp8192@depth128k +11.28% (user-defined long-context point), depth32k +4.77%, pp32768 +2.18%, short points flat, PPL 4.3562. T17 closed the Q8_0 dequant follow-up (767 GB/s measured, <=0.3% theoretical headroom, below the +0.4% gate). T18 rewrote the Volta FA mma kernel (ncols=32); it passed the harness at 1.12x but production showed only +0.11% at 128k (nsys: ncols=32 is 0.3-0.6% slower at l~135k), so user Q16 cancelled it. T19 produced the full OURS vs STOCK curve: prefill +8.7/+9.7/+11.5/+19.6/+35.6% (pp512..pp131072), PPL 4.3562 vs 4.3572, decode flat through d32768; d131072 stayed "uncertain (-3..-10%)" per channel log. T19-L (2026-09-24) slimmed the delivery to A1 + A4 + sm70 MMA configs and retired A2/A3; this is the base commit `e92f20b1` (`ggml-cuda : V100 optimizations (K-quant vec dequant, FATTN split, sm70 MMA configs)`) that all private work now sits on.

### 2026-09-22: GDN investigation (T03/T13/T25)

T03 examined `gated_delta_net`. Step 1 adopted the vec4 row layout (lane owns 4 consecutive rows, float4 loads; branch on `n_tokens > 1`) and fixed a KDA indexing defect (`use_vec4 && !KDA`): kernel 879.9 -> 826.3 us/layer (-6.1%), pp512 +1.0%, PPL 4.3569. The kernel is instruction-throughput limited (two warp reductions ~20 of ~45 instructions). The chunked rewrite passed correctness (algorithm error 1e-16, V1-V3 harness passes) but stayed far slower at the real scale: 4.60 / 3.29 / 1.644 / 1.844 ms vs the existing 0.826 ms per layer-ubatch, with 1.4-2.4x more FLOPs; experiments showed it was neither memory-latency bound nor occupancy bound. T03 closed as NO PATH; GDN is near its practical floor (~0.8 ms). `__expf` gave -10% kernel / +0.7% end-to-end but was recorded below the drift gate; user Q12 cancelled T13 to keep 100% zero drift. T25 (fp16 GDN state, 144 -> 72 MiB) was rejected by the user: precision reduction is not accepted. GDN state snapshots later became the cost driver behind ReplaySSM.

### 2026-09-23..27: MTP work (T06/T20/T21/T22/T30/T31)

T06 was skipped by the user (production already used MTP n-max 3; 25.3 -> 42.7 t/s). T20 independently re-investigated MTP trajectory consistency and found three fork sources: S1 Volta FA dispatch (VEC for decode n_q=1, TILE for verify), S2 this fork's own GDN vec4 lane mapping (`n_tokens > 1` changes the warp reduction order; the old report's "GDN is batch invariant" did not hold), and S3 upstream FA VEC split-K padding near KV boundaries. With all three fixed, MTP output was token-identical to no-spec (900/150/100 tokens at d0/32k/128k); cost was S2 tg -0.6..-1.3% and S3 -0.5..-0.8%. The user did not adopt the fixes; they were rolled back and the material sealed. T21 (K sweep) and T22 (ngram lookup) were approved then declined (2026-09-23, explicit no on 2026-09-26). T30 (2026-09-26) tried to speed up long-context MTP by switching the verify path from TILE to VEC: VEC was 3.1x slower at n_q=4 / l=128K (4.68 vs 1.50 ms/layer) and larger KV splitting (PB/NBATCH 13 -> 160) was 15% slower; rejected and rolled back. T31 (2026-09-26/27) decomposed a 128K MTP3 round (~112-115 ms): verify+host 90 ms (78%), draft 17.9 ms, target sampling 6.8 ms, accept ~0; no-spec decode is 77.4 ms/token (12.9 t/s), so MTP3 is ~2.1x. K=3 stays optimal. Draft windowing (B1) was rejected by analysis (+2..4% ceiling); device-side sampling (B2, merged T29) is ~+5% and was left as a cheap todo. The user parked T31 on 2026-09-27: one 128K re-prefill costs ~257 s versus 2-6 s of reuse, so the effort moved to T32.

### 2026-09-23..26: ReplaySSM (T24)

T24 replaced per-position GDN state snapshots (K+1 copies of ~144-147 MiB) with one committed state plus per-token raw-input records (~1.71 MiB/token) replayed through the same fold path as verify. Two root causes were fixed: a restart batch (position rollback) must not fold, and `s_copy` overwrote the conv rollback state (upstream formula restored plus `s_copy_conv`). Multi-sequence accounting was made per-sequence (record bank = sequence id; fold block layout `4*n_seq_max+3`; SEQ_VERSION 3 -> 5). Acceptance was ON==OFF token-identical across MTP n-max 1/2/3 x np 1..4 and a second model (Qwen3.6-35B-A3B + DFlash n-max 6) including production sampling, concurrency, long runs and save/restore; PPL 4.3567 on both sides; self-check 0 mismatches. Trade-off: VRAM np=1 -420 MiB / np=4 -1.37 GiB for tg -1.9% (71.6 -> 70.2). It is opt-in via `GGML_CUDA_GDN_REPLAY=1` (default off = T19-L behavior) and was deployed on 2026-09-26. Later (T32 smoke, 2026-09-27) run-to-run nondeterminism on the rollback+replay path was found (ref-vs-ref ~1/6 value diffs, token flips only near ties; replay=0 is clean); recorded as a follow-up item.

### 2026-09-27: range state API (T32 stage 1)

T32 was opened for agent-session reuse. Stage 1 added `llama_state_seq_get_data_range_ext` / `llama_state_seq_set_data_range_ext` (save/load a position range of one sequence) with an overlap-rejecting append API and a CPU/GPU harness. Measured: H2D 2.37 GiB/s, q8_0 KV ~50,200 B/token, block size `--tree-chunk` 512; a 100K-context range is 4.68 GiB (~2 s H2D). Small-model correctness 23/23/23/22 PASS plus both regressions green. The work was merged as a range API commit plus a tests commit; the stable tree work reuses it.

### 2026-09-27: KV tree storage (T32 stages 2-5)

The tree stores stable KV prefixes content-addressed in blocks (`tools/server/server-kv-tree.{h,cpp}`): content-hash block chain, anchors, sparse checkpoints, one authoritative RAM+SSD copy, eviction order with pin/refusal. The server integration (`--kv-tree`, default off) parks sequences at stable prefixes, restores the longest matching prefix, erases on request end, and heals on restore misses. Stages: storage module (harness logic 18/18, 2B model 43/43/42/42, 3B attention control 43/43); server integration acceptance calib 417,408,732 B -> 133 MiB RAM, parked=23 restored=12 miss=0 with 12/12 bit-identical; long-run hardening (D12 checkpoint rebuild with recurrent-tail semantics, D13 startup clear, 256 MiB checkpoint cap) verified by a 30 min soak (rounds=1703, rebuilt=336, cmp 340/340, restart cleanup 79 -> 0); stage 5 split checkpoint anchors (`--tree-checkpoint-anchor-step`, 32768) from fork anchors (`--tree-checkpoint-fork-step`, 8192), added heal-on-miss and a `step_skips` counter (fork mode 5/5 bit-identical). A merged soak showed one tree-vs-full divergence at round 315; diagnosis found ~0.015 nats difference where the full path's top-2 gap was 0.002, so diffs < 0.05 nats are now recorded as near-ties (final run `cmp_tie=1 cmp_bad=0`). The early T32-S1 experiments (pin anchors, max-LCP, park protection) measured the payoff on a short agent trace: a rotated session went from 5.4-5.7K tokens / 6.7 s to 71 tokens / 0.6 s (~12x TTFT), and checkpoint blobs shrank 1.9G -> 1.02G. The 27B+MTP+tree production smoke reached a steady third round with prompt_n=6 (four sessions), prompt_ms 291-300, wall 16-25 s -> 2.7 s (>99.9% reuse, ~7x). S1 was later dropped per user instruction (snapshot archived) and re-landed only as S1.5 checkpoint slimming plus checkpoint pruning.

### 2026-09-28: media-aware tree

An interim fix that disabled the tree whenever `--mmproj` was loaded (all prompts were flagged as media) was reverted in favor of native media support. The tree now handles image/audio/video prompts: media identity (`mtmd_input_chunk_get_id()` sha256) is folded into the block hash, token index and position are separated for M-RoPE, block boundaries never split a media chunk, and media bytes are not stored (the slot prompt is rebuilt from the request chunks on restore). E2E (0.8B-MTP + mmproj-F16, greedy, cuda0): the same image session parked 1094 -> restored 1059 tokens with prompt_n 1063 -> 25; a different image at the same position caused a restore miss and full prefill; two adjacent images parked 2157 / restored 2090 (prompt_n=22); all 5 outputs were byte-identical to the no-tree baseline. A real crash found on the way (a hybrid model without rollback capability restored to the full prompt, then `seq_rm` aborted) was fixed by leaving one token in that case and rebuilding media prompt checkpoints with token/position separation.

### 2026-09-28: production incidents and fixes

Production (27B, np=1, MTP, tree, ctx 184320, started by LLM-Manager) began answering some requests with 0 tokens after agent-style repeated rollbacks of the same message. A request that used the stock checkpoint path (not the tree) evaluated 1 token and produced only stop/EOS. Diagnostic builds were deployed; it turned out production does run ReplaySSM because `GGML_CUDA_GDN_REPLAY=1` is a user-level environment variable inherited by the launcher. Root cause: the conv rollback cache R has `1+n_rs_seq` per-token planes, but `state_write_data` saved only the rs_idx-selected plane; after restore rs_idx is zeroed and the next partial rollback (for example the `[TAG_PROMPT_LOGITS]` n_past-- path) read stale planes, corrupting the conv state so the first token stopped. The fix saves all planes of R (and PLE `p_l`), bumping `LLAMA_STATE_SEQ_VERSION` 5 -> 6; the 27B checkpoint blob grows 161.8 -> ~178.6 MiB. A companion guard disables replay when any recurrent layer is not on CUDA (records are only written by the CUDA kernel). The diagnostic commits were dropped and the history rebuilt as two clean commits; a Vulkan call-site fix for the new `ggml_gated_delta_net` signature was added after the upstream rebase. The rebase itself (2026-09-28, upstream +23 commits) had zero conflicts. The private range was then rebuilt into the 8 groups below. Known remaining items per channel log: hybrid-iswa `state_write/read` ignores PARTIAL_ONLY (SWA-hybrid models only), T24 rollback nondeterminism review, and delivery-baseline inclusion.

## Task ledger

T11 has no task card (tracked in BOARD/ARCHIVE); T23 and T26-T29 exist only as external-candidate entries in OPTIONS.md and are not listed here.

| ID | Title | Outcome | Key result |
|---|---|---|---|
| T01 | cuBLAS GEMM efficiency check | verified | Wall 84-107 TF; call side has no room (T08 re-check); "75.6 TF" was a locked-clock artifact. |
| T02 | Fused dequant + fp16 MMA GEMM | rejected | Gate A: skeleton 57.5 TF; mma path alone 1.472 ms > 1.404 ms baseline. |
| T03 | gated_delta_net optimization | closed (no path) | vec4 adopted (-6.1% kernel, pp512 +1.0%); chunked V1-V3 1.644-4.60 ms vs 0.826 ms; `__expf` -10% kernel recorded. |
| T04 | ub / server config pinning | deferred | ub2048 measured 1152-1157 t/s (+21%); acceptance stays ub512; enable server `-ub 2048` when wanted. |
| T05 | decode profile | closed | 34.33 ms busy token: MMVQ 29.84 ms (86.9%, saturated); ceiling 27.3-27.5 t/s. |
| T06 | MTP speculative decoding | skipped (user) | Production already used draft-mtp n-max 3: 25.3 -> 42.7 t/s. |
| T07 | elementwise vectorization | done (partial rejected) | silu -5.1% kernel (adopted, later retired); rms_norm +5.7% kernel -> rejected. |
| T08 | cublasLt (per-shape) | rejected | Real orientation, 8 shapes, 5 interleaved rounds: -1.11% weighted; zero llama.cpp changes. |
| T09 | Fusion 2.0 (quantized operand) | rejected | 37.0-40.9 TF; even without unpacking 1.569 ms > 1.404 ms baseline; fusion closed forever. |
| T10 | Long-context attention recon | done | attention d0 3.3% / pp32768 18.5% / depth32k 32.0%; ub512 29.5 TF/s vs ub2048 39.9; no split-D/N32 port. |
| T11 | ub512 FA KV-split | rejected, then adopted | First gates missed (depth32k +2.0% / pp32768 +1.1%); user Q10 adopted -> T12. |
| T12 | Adopt T11 (KV-split replay) | adopted/verified | pp8192@depth128k +11.28%; depth32k +4.77%; pp32768 +2.18%; PPL 4.3562. |
| T13 | `__expf` (GDN) | cancelled (user) | +0.7% end-to-end was not worth any numerical drift; spec archived. |
| T14 | ub2048 adoption | deferred | Predicted +21% pp512 / ~+25% at 128K; declined Q14 (sealed), then deferred 2026-09-26. |
| T15 | Dual-stream dequant overlap | dropped | +0.5% (bandwidth saturated: dequant 707 + GEMM 150 ~ 95% peak); no formal A/B. |
| T16 | ub512 attention parallel_blocks | adopted/verified | PB=2: attention -14.7% vs T11-off, -6.7% vs T12 default; merged with T12. |
| T17 | Q8_0 dequant follow-up | closed | Current kernel 767.2 GB/s (~877 GB/s model-equivalent); vectorized candidate slower; <=0.3% < +0.4% gate. |
| T18 | Volta FA mma rewrite | closed (user Q16) | Production 128k +0.11% (< +2% gate); ncols=32 slower 0.3-0.6% at l~135k; reverted. |
| T19 | Final delivery + full curve | done | OURS vs STOCK pp +8.7/+9.7/+11.5/+19.6/+35.6% (512..131072); PPL 4.3562 vs 4.3572; d131072 uncertain. |
| T19-L | Delivery slimming (drop A2/A3) | done | pp512 -1.5%; long points -0.5..-0.9%; PPL 4.3567; MTP3 no regression; produced base commit `e92f20b1`. |
| T20 | MTP trajectory consistency | done, not adopted | 3 fork sources fixed (S1 FA dispatch, S2 GDN vec4, S3 upstream split-K padding); user declined; rolled back. |
| T21 | MTP K sweep | declined | User said no (2026-09-23; explicit no 2026-09-26). |
| T22 | ngram lookup | declined | Same user decision as T21; no measurements taken. |
| T24 | ReplaySSM | adopted + deployed | Snapshot -> records (~1.71 MiB/token); VRAM -420 MiB (np1) / -1.37 GiB (np4); tg -1.9%; ON==OFF token-identical; opt-in env var. |
| T25 | GDN state fp16 | rejected (user) | Precision reduction violates the quality line. |
| T30 | MTP long-decode speedup | rejected | Verify VEC 3.1x slower (4.68 vs 1.50 ms/layer at 128K); PB/NBATCH 13->160 -15%; rolled back. |
| T31 | MTP round overhead (128K) | parked (user) | Round ~112-115 ms: verify+host 90 (78%), draft 17.9, sampling 6.8; production sampling 27.0-27.6 t/s; B2 ~+5% left. |
| T32 | Agent session reuse | running | S1: 5.4-5.7K tok/6.7 s -> 71 tok/0.6 s (~12x TTFT); tree stages 1-5 + 5b accepted; 27B+MTP+tree third round prompt_n=6, 291-300 ms, ~7x wall; media tree E2E byte-identical. |

## Final commit series

The private work on top of `ggml-cuda : V100 optimizations` (`e92f20b1`) is organized as 8 commits. They replace the original 21-commit series (`e92f20b1..32290e987` per channel log).

1. `llama` ReplaySSM: records instead of snapshots, plus the all-GDN-layers-on-CUDA guard, the conv rollback plane fix (SEQ_VERSION 6) and the Vulkan call-site fix for the new signature.
2. `server`: drop draft KV from context checkpoints.
3. `llama`: range state API (`llama_state_seq_get_data_range_ext` / `set_data_range_ext`).
4. `server`: KV tree storage (the `server-kv-tree` module, storage hardening, size logging).
5. `server`: KV tree integration (options, server park/restore/erase/heal wiring, checkpoint rebuild and fork anchor seeding).
6. `server`: media-aware KV tree (media blocks/matching/restore plus the server media prompt wiring).
7. tests: the range state harness, KV tree storage harness, capture tests and harness extensions.
8. docs: fork notes/rebase workflow and the multi-GPU limitation note.

Mapping of the current range subjects to the new groups:

| Old commit subject (from `git log e92f20b1..HEAD`) | New group |
|---|---|
| `llama : replay GDN state from records instead of snapshots (ReplaySSM)` | 1 |
| `llama : disable replay when a recurrent layer is not on CUDA` | 1 |
| `llama : save all conv rollback planes with the recurrent state` | 1 |
| `ggml-vulkan : fix gated delta net call after the replay signature change` | 1 |
| `server : drop draft KV from context checkpoints` | 2 |
| `llama : add range state API for attention KV` | 3 |
| `server : add kv tree storage for attention KV` | 4 |
| `server : harden the kv tree storage for long runs` | 4 |
| `kv tree : log sizes in MiB` | 4 |
| `common : add kv tree server options` | 5 |
| `server : add kv tree server integration` | 5 |
| `common : add kv tree checkpoint and fork step options` | 5 |
| `server : rebuild checkpoints and seed fork anchors from the kv tree` | 5 |
| `kv tree : media-aware blocks, matching and restore` | 6 |
| `server : wire media prompts into the kv tree` | 6 |
| `tests : add range state API harness` | 7 |
| `tests : add kv tree storage harness` | 7 |
| `tests : add kv tree capture tests` | 7 |
| `tests : extend the kv tree harness` | 7 |
| `docs : add fork notes and rebase workflow` | 8 |
| `docs : note the multi-gpu limits` | 8 |

Notes: option commits travel with the integration group and storage hardening with the storage group, as planned in the channel. The work previously existed in other groupings (for example a single-commit T24 squash and separate stage branches, per channel log).

## Rejected and retired work

| Item | Status | Why |
|---|---|---|
| T02 fused dequant + fp16 MMA | rejected | mma path alone (1.472 ms) already slower than the 1.404 ms baseline; dequant also consumed the issue budget. |
| T09-A quantized-operand fusion | rejected | 37.0-40.9 TF; unpacking costs 18-30% and cannot overlap; "even unpacked" 1.569 ms > baseline. |
| T08 cublasLt / algo hints | rejected | Real orientation weighted -1.11%; every explicit algo 0..15 slower than DEFAULT; global hint gave pp512 -68%. |
| T03 chunked GDN | closed, no path | V1-V3 at 1.644-4.60 ms vs 0.826 ms current, 1.4-2.4x FLOPs; not memory/occupancy bound. |
| t07 rms_norm_vec4 | rejected | Kernel total 174.81 -> 184.85 ms (+5.7% slower); ceiling 0.1-0.2%, below the noise floor. |
| t11 fattn KV-split | rejected first, later adopted | Original gates missed (depth32k +2.0% < 3%, pp32768 +1.1% < 2%); user Q10 adopted it and T12/T16 integrated it. |
| t18 ncols32 FA | closed (user Q16) | Production 128k +0.11% (< +2% gate); at l~135k it is 0.3-0.6% slower; both source edits reverted. |
| t30 knobs (VEC verify, PB/NBATCH) | rejected | VEC 3.1x slower at n_q=4/128K; PB/NBATCH 13->160 also -15%; experiment code rolled back. |
| gdn vec4 (A2) | retired (T19-L) | Rebase surface plus the MTP S2 divergence source; pp512 +1.0% did not justify keeping it. |
| silu vec4 (A3) | retired (T19-L) | +0.1% end-to-end; dropped together with A2 to slim the patch set. |
| T13 `__expf` | cancelled (Q12) | +0.7% did not justify any numerical drift. |
| T15 dual stream | dropped | +0.5% only; bandwidth saturated (dequant 707 + GEMM 150 ~ 95% peak). |
| T14 ub2048 | deferred | +21% measured but config-level; declined Q14 (sealed for reference), deferred 2026-09-26. |
| T20 MTP fixes | not adopted | User decision; all fixes rolled back, materials sealed. |
| T25 fp16 GDN state | rejected | Precision reduction violates the quality red line. |
| rms_norm block config (T05) | reverted | No effect (tg128 identical on both arms). |
| FORCE_MMQ dp4a / fp16 accumulate / workspace / algo hints | disproven | pp512 -41% / gate/up 63.7 vs 84.2 TF / no gain / -68% global; do-not-retry list. |

## Pointers

Companion drafts in this folder:

- `pitfalls.md` - measurement pitfalls, heat drift, do-not-retry list, harness fidelity lessons.
- `benchmarks.md` - verified numbers and acceptance points per configuration.
- `kv-tree.md` - KV tree design, options and limitations.
- `replayssm.md` - ReplaySSM design, env vars, state format and limitations.

Raw channel log (source of record):

- `archive/v100-collab/` - ARCHIVE.md, STATUS.md, RESULTS.md, BOARD.md, OPTIONS.md, PROTOCOL.md, ENVIRONMENT.md, QUESTIONS.md, TASKS/.
