# ReplaySSM

Motivation: speculative decoding rolls the GDN recurrent state back after rejected draft
tokens. The stock design keeps `1 + n_rs_seq` full state snapshots (one per rollback
step, ~144 MiB each per sequence for the 27B model). ReplaySSM instead keeps a single
committed state and records the raw inputs (k, v, g, beta) of each batch; a rollback
becomes "replay the accepted prefix of the last recording batch from the records", which
the GDN kernel does in registers before processing the new batch. This replaces per-token
snapshot planes with one plane plus a small per-batch record block.

When enabled:

- `GGML_CUDA_GDN_REPLAY=1` (default off: the old snapshot path, T19-L behavior).
- Only when `n_rs_seq > 0` and the GDN record widths can be derived from hparams.
- Every recurrent layer must be offloaded to CUDA: the records are written and replayed
  by the CUDA kernel only. Otherwise the constructor logs a WARN and disables replay.
- Optional self-check: `GGML_CUDA_GDN_REPLAY_CHECK=1` (diagnostic only, extra memory).

Implementation: `src/llama-memory-recurrent.{h,cpp}` owns the record tensors and the
bookkeeping; `src/llama-graph.cpp` fills the fold parameters; the CUDA kernel in
`ggml/src/ggml-cuda/gated_delta_net.cu` records and replays.

## Record layout

- One packed F32 tensor per layer: `[k | v | g | beta, 2*T_cap, banks]`.
  The bank dimension is the cell/sequence dimension (`mem_size`); a sequence always
  addresses its records through its own bank (`bank = seq_id`, chosen by the host), not
  through the cell the batch happens to land in.
- `T_cap = n_rs_seq + 1`: a recording batch can cover at most `T_cap` tokens, which is
  exactly the largest rollback distance the engine supports.
- Block widths come from the model hyperparameters: `k = H_q*d`, `v = H_v*d`,
  `g = H_v*d` (KDA) or `H_v` (non-KDA), `beta = H_v`, where `d = ssm_d_state`,
  `H_v = ssm_d_inner/d`, `H_q = ssm_n_group`. Records are enabled only when these and
  `ssm_d_conv` are positive.
- Double buffering: the `2*T_cap` dimension is split into two halves. Each sequence
  alternates its write half per recording batch; the next fold reads the half the
  previous recording batch wrote. The two halves are per sequence, so the records a
  fold reads and the records the current batch writes can never collide, however the
  batches of different sequences interleave.
- Fold parameters live in a small persistent I32 tensor (`rec_fold_t`), one per memory:
  `[unused, p_i ..., read_half_i ..., write_half_i ..., bank_i ..., self-check, skip check]`.
  The `p` slot packs the replay count in the low byte and the previous batch's token
  count `tn` in bits 8..15 (`tn` is used by the self-check). The struct is filled once
  per ubatch by the graph and pushed straight into the device tensor with
  `ggml_backend_tensor_set`; it is not a normal graph input, because the scheduler's
  input-copy mechanism does not deliver tensors consumed only by the fused
  `gated_delta_net` op.
- With the check env var on, one extra `rec_diag` tensor per layer holds the state
  committed by the last forward pass, for the fold self-check.
- Allocation: one record tensor per recurrent layer plus one fold tensor per memory
  (single-device assumption), allocated with the state buffers and cleared at allocation;
  the diag tensor exists only when the check env var is set.

## Fold protocol

- At the start of a recording batch (`n_tokens <= T_cap`) the GDN kernel replays `p`
  records of the sequence from the single committed state plane, using the same
  finite-precision update as the main loop (`gdn_state_update` is shared by both code
  paths), and writes the result to the state plane. It then processes the current batch
  and, if the batch records, writes its raw inputs to the other half.
- Non-recording batches (`n_tokens > T_cap`, typical prefill) write no records and commit
  the state after the batch themselves; the host marks the records as stale, and a
  rollback over such a batch falls back to the server checkpoint path unchanged.
- `p` is computed on the host:
  - Normal case: `p = rec_n - rs_idx` (accepted prefix of the previous recording batch),
    only while the records are current (`rec_have_rec`) and `rollback < rec_n`; a
    rollback requested with stale records logs an error and folds 0.
  - After a state restore: the blob carries a pending prefix; `p = pending - rs_idx`.
  - Continuity gate: the fold runs only when `pos0 == rec_pos0 + p` for the batch's
    first token, i.e. the batch continues exactly after the replayed prefix. A batch
    that restarts at an earlier position folds nothing (`p = 0`), which also covers
    prefill batches that re-process positions already covered by warmup records.
  - Full acceptance is not a special case: `p = T` replays the whole previous batch and
    must reproduce the state that pass committed.
- Self-check (opt-in): when `skip_check == 0`, `rec_diag` is present and `p == tn`
  (the whole previous batch was accepted), the kernel compares the fold result with the
  state committed by the previous pass, accumulates the number of mismatching elements
  into the fold block, and the host logs an error if it is nonzero. After a state
  restore the check is skipped once (`rec_check_bad`): there is no committed pass to
  compare against.
- Side effects run once per ubatch: the same memory appears in several graph inputs (rs
  and the hybrid variants), so a take flag (`rec_fold_take`) guards the commit and the
  fold-parameter write.
- State-save contract: in replay mode the S cache has a single committed plane per cell.
  The state blob carries the records and the replay bookkeeping as an extra `RSPL`
  block: per sequence the last write half, `rec_n`, `rec_pos0` and the replay prefix `p`,
  plus the record banks of every layer. A restore sets the pending prefix so the next
  fold rebuilds the committed state; the check is skipped once. Old state blobs are not
  interchangeable across the version bump (see below).

## Rollback correctness

- S (the GDN state): one plane per cell in replay mode. Its gather reads row 0, the
  pinned committed plane; rollback information travels in the records, not in the plane.
- R (the conv cache): still snapshot-based. `r_l` always has `mem_size * (1 + n_rs_seq)`
  rows, i.e. one plane per rollback step, and its gather uses the rollback-aware index
  `rs_idx * size + src`. Conv inputs are not recorded, so R cannot be replayed from the
  record block (deferred M3).
- PLE (the second conv history, `p_l`): same per-step planes, saved and restored with R.
- Ordering: `set_fold_input` captures the conv gather indices first (`s_copy_conv`);
  consuming them resets `rs_idx`, so the regular state gather that runs afterwards reads
  row 0 (the committed plane). The conv gather uses the rollback-aware indices, the fold
  uses the replay count, and the state gather uses the committed plane.
- Why S can be a single plane: the records hold the exact raw inputs and both the fold
  and the main loop call the same shared update function, so bit-exact replay is
  possible inside one launch. R has no recorded inputs, so it keeps its planes.

## The conv-plane restore bug (2026-09-28)

Production symptom: with the tree and context checkpoints in use, an agent request after
a rollback evaluated 1 token and generated 0 tokens (first token was EOS), producing
empty responses.

Concrete failure chain:

```
tree restore 27476 tokens -> [diag] state_read: tail 0 pos 27475 pending 4
-> [TAG_PROMPT_LOGITS] forces n_past-- -> seq_rm [27475, inf): ROLLBACK rs_idx=1
-> [diag] get_rec_p: pos0 27475 p 3 (continuity holds)
-> prompt eval 1 token, eval 0 tokens (first token stop)
```

Root cause: in replay mode the conv cache keeps one rollback plane per step, but the
state save wrote only the plane selected by the rollback index and the read side restored
that one plane and then cleared `rs_idx`. A partial rollback after the restore (the
1-token `TAG_PROMPT_LOGITS` rollback) gathered plane 1..n, which still held rows from an
unrelated sequence or an older run, so the conv state was wrong and the model emitted
EOS immediately.

Fix (the `llama : save all conv rollback planes with the recurrent state` commit of the
final series):

- In replay mode `state_write_data` writes and `state_read_data` reads all
  `1 + n_rs_seq` R planes (and the PLE `p_l` planes), not only the selected one.
- `LLAMA_STATE_SEQ_VERSION` is bumped 5 -> 6, so old state files and RAM blobs are
  rejected instead of being misread.
- The earlier `llama : disable replay when a recurrent layer is not on CUDA` commit
  disables replay with a WARN when any recurrent layer is not on CUDA, because the host
  bookkeeping would otherwise accept record rollbacks that the kernels never produce.

Size effect (measured on the 0.8B model, n_rs_seq = 3):

- State blob `22,580,436 -> 26,561,748` bytes (22.58 -> 26.56 MB), i.e. `+3` planes at
  `1.327 MiB` each.
- 27B class: each checkpoint grows by the planes that were not saved before. The full
  set is `1 + n_rs_seq` planes (4 at `n_rs_seq = 3`), about 5.6 MiB each; a checkpoint
  goes from ~161.8 MiB to ~178.6 MiB.

## Verification

- `test-backend-ops -b CUDA0 -o GATED_DELTA_NET` (plus the cache-fusion variant) runs the
  op on CUDA in snapshot mode, checking the fork signature against upstream changes;
  36/36 after the rebase, alongside `SSM_SCAN`. Replay itself is covered at model level.
- Command examples: `test-backend-ops -b CUDA0 -o GATED_DELTA_NET`;
  `test-t32-tree -m <gdn model> --mode model` (or `--mode accept`) with
  `GGML_CUDA_GDN_REPLAY=1`, optionally with the check env var on top.
- `test-t32-tree` model and accept modes with `GGML_CUDA_GDN_REPLAY=1` exercise the real
  snapshot/restore and rollback paths on a small GDN model, comparing generation against
  a baseline; the T24 harness (fold vs snapshot over m = 0..4/8 cases) proved bit-exact
  fold equivalence before integration.
- `GGML_CUDA_GDN_REPLAY_CHECK=1` turns on the in-kernel fold self-check (0 mismatch
  expected, errors are logged). It adds one per-layer state copy and is diagnostic only,
  not a production guard.
- Historical acceptance (T24, MTP n-max 1/2/3, np 1..4): replay ON and OFF both match
  the no-MTP baseline 64/64 greedy, PPL 4.3567 identical, VRAM -420 MiB at MTP3 np=1,
  throughput -1.9%. Known residual: rollback + replay can show run-to-run value
  differences (~1/6, logits 0.17-0.25) at near-ties; replay=0 is bit-identical in the
  same runs, and rollbacks of 2 tokens or less were lossless.

## Known limits

- Non-replay (env off) keeps the old hazard: the state save stores a single conv plane,
  so a restore followed by a partial rollback can read stale planes from another
  sequence. Replay is the workaround, and it is not the default.
- Row split breaks ReplaySSM: the kernel derives the record layout from the tensor
  strides, which a row split changes. Use layer split with all GDN layers on CUDA; any
  other placement disables replay with a warning (older builds could silently keep stale
  state, which is why the guard exists).
- CUDA only: the records are written and folded by the CUDA kernel. Other backends
  (Vulkan, CPU, Metal) only pass through the extra arguments with no record support.
- Version incompatibility: `LLAMA_STATE_SEQ_VERSION` is 6. State files and slot saves
  are not interchangeable with upstream (3) or with pre-fix fork builds (5); the tree
  disk tier is wiped at startup, but checkpoints inside a running server are RAM-only.
- Records cover at most `T_cap = n_rs_seq + 1` tokens per batch. Longer batches write no
  records and fall back to server checkpoints on rollback; a rollback larger than the
  recorded batch also falls back.
- The self-check is opt-in and one-shot: it skips the first batch after a restore and
  adds memory, so it is not a safety net for every path.
- Performance is a trade, not a pure win: VRAM drops by ~420 MiB at MTP3 np=1 while tg
  drops ~1.9%; enable per VRAM/speed preference.
- Coverage: validated on the Qwen3.5/3.6 GDN family (including another model with
  DFlash), np up to 4 and a session save/restore matrix. EAGLE3, DSpark and KDA were not
  covered (no weights); np >= 5 and extreme small rollbacks without an SWA hybrid are
  untested edges.

Source:
- archive/v100-collab/TASKS\T24-replayssm.md
- archive/v100-collab/TASKS\T32-agent-session-reuse.md
- archive/v100-collab/RESULTS.md
- archive/v100-collab/STATUS.md
- archive/v100-collab/ENVIRONMENT.md
- FORK-NOTES.md
- src\llama-memory-recurrent.h
- src\llama-memory-recurrent.cpp
- src\llama-graph.cpp
- ggml\src\ggml-cuda\gated_delta_net.cu
- include\llama.h (LLAMA_STATE_SEQ_VERSION)
- final-series commits: `llama : save all conv rollback planes with the recurrent state` (all conv planes) and `llama : disable replay when a recurrent layer is not on CUDA` (non-CUDA replay guard)
