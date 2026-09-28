# Task 3 Report: range-API throughput bench, regressions, default locking (T32 stage 1)

Date: 2026-09-27
Repo: D:\LLM\Backend\src\llama.cpp-my (branch t32-stage1)
Commit: dfd8a7035 "t32 : add range state API throughput bench" (not pushed)

## What I implemented

1. Added `run_range_bench()` to `tests/test-t32-range.cpp` after `run_correctness()`, verbatim from the brief,
   plus the `range-bench` dispatch in `main`. Only `tests/test-t32-range.cpp` changed in the repo (+55 lines).
2. Built targets `test-t32-range test-state-restore-fragmented test-save-load-state` via
   `<TEMP>\v100\build_test_t32.cmd` (compiled clean; one pre-existing C4297 warning in llama.cpp).
3. Ran the three bench runs (27B Q6_K, GPU 1, production `-ctv q8_0`) and the two regressions (2B Q4_K_XL, GPU 1).
4. Archived results and updated the channel docs; generated the branch patch.

## Bench outputs (full stderr, verbose warning block included)

Commands (chunk 512 shown; 1024/2048 identical except `--chunk`):
```
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-t32-range.exe -m <models>\Qwen3.8-27B\Qwen3.8-27B-UD-Q6_K.gguf -ngl 99 -fa on -ctv q8_0 -np 2 -c 65536 -b 512 -ub 512 --mode range-bench --n 32000 --chunk 512
```

### chunk=512, exit 0
```
0.01.066.170 W model has unused tensor blk.64.attn_norm.weight (size = 20480 bytes) -- ignoring
0.01.066.183 W model has unused tensor blk.64.post_attention_norm.weight (size = 20480 bytes) -- ignoring
0.01.066.200 W model has unused tensor blk.64.attn_q.weight (size = 51609600 bytes) -- ignoring
0.01.066.206 W model has unused tensor blk.64.attn_k.weight (size = 5570560 bytes) -- ignoring
0.01.066.212 W model has unused tensor blk.64.attn_v.weight (size = 5570560 bytes) -- ignoring
0.01.066.237 W model has unused tensor blk.64.attn_output.weight (size = 25804800 bytes) -- ignoring
0.01.066.243 W model has unused tensor blk.64.attn_q_norm.weight (size = 1024 bytes) -- ignoring
0.01.066.249 W model has unused tensor blk.64.attn_k_norm.weight (size = 1024 bytes) -- ignoring
0.01.066.255 W model has unused tensor blk.64.ffn_gate.weight (size = 73113600 bytes) -- ignoring
0.01.066.261 W model has unused tensor blk.64.ffn_down.weight (size = 73113600 bytes) -- ignoring
0.01.066.267 W model has unused tensor blk.64.ffn_up.weight (size = 73113600 bytes) -- ignoring
0.01.066.275 W model has unused tensor blk.64.nextn.eh_proj.weight (size = 43008000 bytes) -- ignoring
0.01.066.282 W model has unused tensor blk.64.nextn.enorm.weight (size = 20480 bytes) -- ignoring
0.01.066.288 W model has unused tensor blk.64.nextn.hnorm.weight (size = 20480 bytes) -- ignoring
0.01.066.310 W model has unused tensor blk.64.nextn.shared_head_norm.weight (size = 20480 bytes) -- ignoring
0.20.299.883 I cmn          init: llama threadpool init, n_threads = 8
[t32-bench] n=32000 chunk=512 chunks=63 fill=40.8s total=1532.0 MiB
[t32-bench] write: 1.28 s (1198.1 MiB/s, 20.30 ms/chunk)
[t32-bench] read:  0.81 s (1896.4 MiB/s, 12.82 ms/chunk)
```

### chunk=1024, exit 0
```
0.00.989.776 W model has unused tensor blk.64.attn_norm.weight (size = 20480 bytes) -- ignoring
0.00.989.788 W model has unused tensor blk.64.post_attention_norm.weight (size = 20480 bytes) -- ignoring
0.00.989.804 W model has unused tensor blk.64.attn_q.weight (size = 51609600 bytes) -- ignoring
0.00.989.810 W model has unused tensor blk.64.attn_k.weight (size = 5570560 bytes) -- ignoring
0.00.989.816 W model has unused tensor blk.64.attn_v.weight (size = 5570560 bytes) -- ignoring
0.00.989.837 W model has unused tensor blk.64.attn_output.weight (size = 25804800 bytes) -- ignoring
0.00.989.843 W model has unused tensor blk.64.attn_q_norm.weight (size = 1024 bytes) -- ignoring
0.00.989.848 W model has unused tensor blk.64.attn_k_norm.weight (size = 1024 bytes) -- ignoring
0.00.989.854 W model has unused tensor blk.64.ffn_gate.weight (size = 73113600 bytes) -- ignoring
0.00.989.860 W model has unused tensor blk.64.ffn_down.weight (size = 73113600 bytes) -- ignoring
0.00.989.866 W model has unused tensor blk.64.ffn_up.weight (size = 73113600 bytes) -- ignoring
0.00.989.873 W model has unused tensor blk.64.nextn.eh_proj.weight (size = 43008000 bytes) -- ignoring
0.00.989.879 W model has unused tensor blk.64.nextn.enorm.weight (size = 20480 bytes) -- ignoring
0.00.989.885 W model has unused tensor blk.64.nextn.hnorm.weight (size = 20480 bytes) -- ignoring
0.00.989.903 W model has unused tensor blk.64.nextn.shared_head_norm.weight (size = 20480 bytes) -- ignoring
0.18.659.220 I cmn          init: llama threadpool init, n_threads = 8
[t32-bench] n=32000 chunk=1024 chunks=32 fill=40.8s total=1532.0 MiB
[t32-bench] write: 1.11 s (1377.8 MiB/s, 34.75 ms/chunk)
[t32-bench] read:  0.74 s (2062.5 MiB/s, 23.21 ms/chunk)
```

### chunk=2048, exit 0
```
0.00.982.286 W model has unused tensor blk.64.attn_norm.weight (size = 20480 bytes) -- ignoring
0.00.982.298 W model has unused tensor blk.64.post_attention_norm.weight (size = 20480 bytes) -- ignoring
0.00.982.315 W model has unused tensor blk.64.attn_q.weight (size = 51609600 bytes) -- ignoring
0.00.982.321 W model has unused tensor blk.64.attn_k.weight (size = 5570560 bytes) -- ignoring
0.00.982.327 W model has unused tensor blk.64.attn_v.weight (size = 5570560 bytes) -- ignoring
0.00.982.348 W model has unused tensor blk.64.attn_output.weight (size = 25804800 bytes) -- ignoring
0.00.982.354 W model has unused tensor blk.64.attn_q_norm.weight (size = 1024 bytes) -- ignoring
0.00.982.359 W model has unused tensor blk.64.attn_k_norm.weight (size = 1024 bytes) -- ignoring
0.00.982.365 W model has unused tensor blk.64.ffn_gate.weight (size = 73113600 bytes) -- ignoring
0.00.982.371 W model has unused tensor blk.64.ffn_down.weight (size = 73113600 bytes) -- ignoring
0.00.982.377 W model has unused tensor blk.64.ffn_up.weight (size = 73113600 bytes) -- ignoring
0.00.982.385 W model has unused tensor blk.64.nextn.eh_proj.weight (size = 43008000 bytes) -- ignoring
0.00.982.392 W model has unused tensor blk.64.nextn.enorm.weight (size = 20480 bytes) -- ignoring
0.00.982.398 W model has unused tensor blk.64.nextn.hnorm.weight (size = 20480 bytes) -- ignoring
0.00.982.416 W model has unused tensor blk.64.nextn.shared_head_norm.weight (size = 20480 bytes) -- ignoring
0.17.996.541 I cmn          init: llama threadpool init, n_threads = 8
[t32-bench] n=32000 chunk=2048 chunks=16 fill=40.9s total=1532.0 MiB
[t32-bench] write: 0.96 s (1594.1 MiB/s, 60.06 ms/chunk)
[t32-bench] read:  0.66 s (2308.8 MiB/s, 41.47 ms/chunk)
```

### Summary table
| chunk | chunks | write s | write MiB/s | write ms/chunk | read s | read MiB/s | read ms/chunk |
|-------|--------|---------|-------------|----------------|--------|------------|---------------|
| 512   | 63     | 1.28    | 1198.1      | 20.30          | 0.81   | 1896.4     | 12.82         |
| 1024  | 32     | 1.11    | 1377.8      | 34.75          | 0.74   | 2062.5     | 23.21         |
| 2048  | 16     | 0.96    | 1594.1      | 60.06          | 0.66   | 2308.8     | 41.47         |

Full-state references (stage 0): H2D 2.37 GiB/s = 2426.9 MiB/s; D2H 2.60 GiB/s = 2662.4 MiB/s.
Range read reaches 78% (512) to 95% (2048) of full-state H2D. Range write reaches 45% (512) to 60% (2048) of D2H.

## Regression outputs

Commands:
```
$env:CUDA_VISIBLE_DEVICES=1
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-state-restore-fragmented.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on
& D:\LLM\Backend\src\llama.cpp-my\build\bin\Release\test-save-load-state.exe -m <models>\Qwen3.5-2B-UD-Q4_K_XL.gguf -ngl 99 -fa on
```

### test-state-restore-fragmented, exit 0
stderr:
```
0.01.740.091 I cmn          init: llama threadpool init, n_threads = 8
main : processed prompt on seq 0, 1, 2 (70 tokens each)
main : saved seq 1 state, 21064092 bytes
main : cleared seq 1 to create fragmentation
main : restored state into seq 1, 21064092 bytes
main : successfully decoded with restored state, generated: '

'
main : SUCCESS - state restore works with fragmented KV cache
```
stdout: empty

### test-save-load-state, exit 0
stderr:
```
0.01.576.225 I run_save_load_tests_for_model: no prompt provided, generating 100 (n_batch) random tokens
0.01.576.268 I run_save_load_tests_for_model: the input prompt is 100 tokens
0.01.817.251 I cmn  common_promp: saved session before last token to dump_state.bin, n_new = 100
0.03.628.646 E state_read_data: mismatched key type (1 != -1, layer 7)
0.03.673.975 E state_seq_set_data: error loading state: failed to restore kv cache
0.04.008.047 E state_read_data: mismatched key type (1 != -1, layer 7)
0.04.056.609 E llama_state_seq_load_file: error loading sequence state file: failed to restore kv cache
```
stdout:
```
=== Test 1: baseline ===
86139 160292 118276 148926 154867 14235 1600 95726 8846 733 37 15934 4120 151466 323 220

=== Test 2: sequence removal isolation ===
PASS

=== Test 3: state load ===
86139 160292 118276 148926 154867 14235 1600 95726 8846 733 37 15934 4120 151466 323 220
PASS

=== Test 4: seq copy (host) ===
86139 160292 118276 148926 154867 14235 1600 95726 8846 733 37 15934 4120 151466 323 220
PASS

=== Test 5: seq copy (device) ===
86139 160292 118276 148926 154867 14235 1600 95726 8846 733 37 15934 4120 151466 323 220
PASS

=== Test 6: seq copy (host, scatter) ===

PASS

=== Test 7: seq copy (device, scatter) ===

PASS

=== Test 8: state blob round-trip ===

PASS

=== Test 9: state restore failure ===

PASS

All tests passed.
```
No SKIPs. The E logs are the intentional negative paths (Test 9 `corrupt_state`, plus the corrupted session-file load path); suite summary is "All tests passed." with exit code 0.

## --tree-chunk decision

**Keep the default at 512.**

Decision rule applied as given:
- `ms/chunk` does NOT stay flat: write 20.30 -> 34.75 -> 60.06, read 12.82 -> 23.21 -> 41.47.
- Per doubling the growth is x1.71/x1.73 (write) and x1.81/x1.79 (read): approximately doubling, so byte transfer
  dominates and the fixed part does not dominate (implied fixed share ~29% write / ~19% read at chunk 512).
- Raising to 2048 would save only ~1.4 s of transfer on a 100K sequence (write 4.0 -> 3.0 s, read 2.5 -> 2.1 s)
  while quadrupling worst-case intra-block fork duplication and coarsening matching; 512 keeps the best reuse granule.

No batch-load path is required by this decision; the range write bandwidth gap (45-60% of D2H) is recorded as a
stage-2 optimization candidate in the archive, not as a risk item.

## Spec lines changed (`D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md`)

1.2 (line 19), before -> after:
```
- **块 (chunk)**: 粒度 `--tree-chunk` 默认 512 token (阶段 0 基准后可能上调到 1024/2048); ...
- **块 (chunk)**: 粒度 `--tree-chunk` 默认 512 token (阶段 1 实测锁定: 512/1024/2048 的 ms/chunk 近似随字节线性增长, 固定开销不主导, 512 保住复用粒度; 见附录); ...
```
3.4 (line 85), before -> after:
```
`--tree-chunk` (默认 512, 阶段 1 出口定)
`--tree-chunk` (默认 512, 阶段 1 实测锁定, 见附录)
```
Appendix, before -> after:
```
- 块: 512 token -> target KV ~26MB, draft KV ~2MB (target ~52KB/token / draft 4.1KB/token)
- 块: 512 token -> target KV 24.5 MiB 即 25.7 MB (阶段 1 实测 q8_0 V, 50200 B/token, range API 仅 attention,
  不含 recurrent 常数), draft KV ~2MB (target ~50.2KB/token / draft 4.1KB/token)
```
```
~40% (16K ~20% / 32K ~10%)
~42% (16K ~21% / 32K ~10%)   [derived from the measured 50200 B/token]
```
```
- H2D: 2.37 GiB/s / D2H 2.60 GiB/s (...); 100K 尺寸线性外推 6.56 GiB -> H2D ~2.8s / D2H ~2.5s
- H2D: 2.37 GiB/s / D2H 2.60 GiB/s (阶段 0 实测: 27B Q6_K 全量 state f16, 32K 2.10 GiB, PCIe x4)
- 100K 树存储搬运 (阶段 1 实测口径, q8_0 target KV = 50200 B/token): 4.68 GiB -> H2D ~2.0s / D2H ~1.8s
  (替代原 f16 全量外推 6.56 GiB -> ~2.8s / ~2.5s; 那份不适用于树存储)
- range API 区间吞吐 (阶段 1 实测, 27B q8_0, 32K, 完整输出 artifacts/t32-stage1-bench.txt):
  chunk 512/1024/2048 -> 读 12.8/23.2/41.5 ms/chunk (1896/2063/2309 MiB/s),
  写 20.3/34.8/60.1 ms/chunk (1198/1378/1594 MiB/s); ms/chunk 近似随 chunk 翻倍 -> 固定开销不主导,
  `--tree-chunk` 维持默认 512 (复用粒度最好)
```

Step-5 item 4 check: H2D measured 0.42 s/GiB, far faster than the "2s/GB" conservative figure, so
`--tree-anchor-step` stays at 32K (16K vs 32K re-evaluation not triggered). Recorded in TASKS.

## Files changed

Repo (commit dfd8a7035, 1 file, +55):
- `tests/test-t32-range.cpp` - added `run_range_bench()` + `range-bench` dispatch

Channel (outside repo):
- `D:\LLM\Backend\v100-collab\artifacts\t32-stage1-bench.txt` (new; commands + full outputs + analysis + decision)
- `D:\LLM\Backend\v100-collab\artifacts\t32-tree-storage-design.md` (1.2, 3.4, appendix)
- `D:\LLM\Backend\v100-collab\TASKS\T32-agent-session-reuse.md` (appended `## Result (stage 1, 2026-09-27)`)
- `D:\LLM\Backend\v100-collab\artifacts\t32-stage1-worktree.patch` (`git diff ba41cccec..HEAD`, 34070 bytes, 13 files,
  generated after the commit; stages 0+1 range API + harness + bench)

## Self-review findings and concerns

1. Size mismatch vs brief's expected shape: brief expected `total≈1860 MiB`; measured 1532.0 MiB. Expected: the
   brief's figure predates the production `-ctv q8_0` quantization. Measured = 1532.0 MiB = 50200 B/token.
2. The range payload contains no recurrent constant: `llama_memory_hybrid::state_write_range` forwards to
   `mem_attn` only (src/llama-memory-hybrid.cpp:197) and the correctness test asserts
   `range(0,L) == full - partial + 8`. So per-token = total/n_tokens directly; recurrent remains the separate
   ~162 MiB anchor term. The stage-0 note "budget 70463 B/token" is superseded by 50200 B/token in the TASKS append.
3. Range write bandwidth (45-60% of full-state D2H) is below the "same ballpark" bar. The timed loop includes a
   fresh per-chunk blob allocation (first-touch page faults) and a size walk by design (brief's verbatim code), so
   the number is a conservative lower bound; production reusing buffers should be faster. Recorded as a stage-2
   optimization candidate; does not change the 512 decision.
4. The doubling ratios (1.71-1.81) are not exactly 2.0, so "fixed overhead negligible" is only approximately true
   (~29% write / ~19% read fixed share at 512). Decision branch 1 was chosen because the growth is clearly not flat;
   the nuance is documented in the archive and spec instead of hidden.
5. `test-save-load-state` prints two `E` lines (corrupted-state restore failure and corrupted session-file load);
   these are the expected negative paths of Test 9 and the suite exits 0 with "All tests passed." - not a failure.
6. Section 9's stage-1 planning sentence (line 159) still reads as a plan ("定 --tree-chunk 默认值 + 是否需要批量
   装载路径"); left untouched because the brief scoped default changes to 1.2/3.4/appendix, and the result is
   recorded in TASKS and the appendix.
7. GPU runs: all exit 0, `CUDA_VISIBLE_DEVICES=1`, `-ngl 99 -fa on`, `-ctv q8_0` verbatim. No failures, no reruns.
8. No push performed; repo working tree clean after the commit (channel files are outside the repo).
