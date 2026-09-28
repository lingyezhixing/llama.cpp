# Pitfalls and lessons

Things that cost time during the V100/27B effort. Read this before touching the same areas.

## The production incident (2026-09-28)

Symptom: agent requests returned empty answers (1 prompt token evaluated, 0 generated, first
token EOS).

Root cause (full chain in `replayssm.md`): in ReplaySSM mode the conv cache R keeps
`1 + n_rs_seq` rollback planes, but the state save wrote only the plane selected by the
rollback index. A restore read that plane back and reset the index, so the next partial
rollback read stale planes from an unrelated sequence and corrupted the conv state.

Why the test matrix missed it:

- The failing combination is "restore a saved state, then immediately partially roll back".
  The T24/T32 harnesses tested rollback (rb <= 2) and state save/restore separately, not the
  two in sequence.
- The server only takes that path when a restore lands exactly at the prompt end and
  `[TAG_PROMPT_LOGITS]` forces a 1-token rollback. Tree heal anchors plus the agent retry
  pattern (sending the same message again) made it deterministic in production.

Lessons:

- Feature interactions need their own tests. When two mechanisms both manage
  positions/rollback indexes (state blobs and sequential rollback), test the cross product,
  not the parts.
- Asymmetries between tensors are suspect. S had moved to a single committed plane while R
  still kept planes; both sides were self-consistent, which is why review did not catch it.
- Bump `LLAMA_STATE_SEQ_VERSION` whenever a state blob layout changes, or old blobs are
  silently misread (5 -> 6 for this fix).

## Logging traps

- `LLAMA_LOG_INFO` from the library is filtered in the server at the default verbosity
  (visible only with `-lv 4`). Production diagnostics from library code must use
  `fprintf(stderr, ...)` or WARN level; server code should use `SRV_INF/SRV_WRN/SRV_ERR`
  (visible from INFO up). A diagnostic round was lost to this.
- `LOG_*`/`SRV_*` are variadic macros: a message with no arguments needs the
  `"%s", "text"` idiom.

## Environment traps

- An LLM-Manager launcher with `env: {}` does NOT mean a clean environment: it inherits
  user-level variables. `GGML_CUDA_GDN_REPLAY=1` had been set with `setx` during the T24
  deployment, so production ran ReplaySSM even though the launcher config showed nothing.
  Check both the launcher row and
  `[Environment]::GetEnvironmentVariable('GGML_CUDA_GDN_REPLAY','User')`.
- Env vars read once into `static`/constructor state need a restart to change; the tuning
  knobs (`GGML_CUDA_FATTN_*`) are documented in `FORK-NOTES.md`.

## Windows and deployment traps

- `core.autocrlf=true` plus an editor holding stale buffers reverted two files during the
  session (`FORK-NOTES.md`, `ggml-vulkan-debug.cpp`). Keep the worktree clean before git
  surgery, close editors, and re-check `git status` after external writes.
- The production server locks its DLLs while running; a deploy needs a stop/start via
  LLM-Manager. Back up the old binaries, copy, verify SHA256, and never kill the user's
  server process.
- `--tree-disk` is wiped at startup. Never point it at a directory that holds other data.
- `-dev cudaN` is a device *index*: in this machine index 1 is the V100, index 0 the RTX
  4060. Do not assume the order.

## Measurement traps (condensed from the channel log)

- Single llama-bench points have about +-0.5% noise; session and thermal drift reach 1%+
  (tg128 26.6 -> 24.4 in one session). Use same-session alternating A/B with >= 2 rounds.
- Harness fidelity: the T18 FA harness showed 1.10-1.22x while production at 128k showed
  +0.11%. Validate harnesses at the target length.
- `ncu` under locked clocks produced a false "75.6 TF / 60%" cuBLAS number.
- Near-tie handling: tree-vs-full greedy flips happened where the logit gap was smaller
  than the numeric difference; the soak treats < 0.05 nats as a tie, not a mismatch.

## Do-not-retry list

| idea | result |
|---|---|
| fused dequant + fp16 MMA GEMM (T02/T09) | rejected: the mma path alone was slower than dequant+cuBLAS |
| cublasLt algo hints / global hints (T08) | rejected: -1.11% weighted; explicit hints slower |
| rms_norm vec4 (T07) | rejected: +5.7% kernel time |
| chunked GDN prefill (T03) | closed: 2-8x slower at real scale |
| `__expf` in GDN (T13) | cancelled: +0.7% did not justify numerical drift |
| fp16 GDN state (T25) | rejected by owner: precision red line |
| ub2048 default (T04/T14) | deferred: +21% measurable but changes the acceptance config |
| MTP trajectory fixes S1/S2/S3 (T20) | rolled back by owner; sealed material archived |
| MTP verify VEC / PB=160 (T30) | rejected: VEC 3.1x slower at n_q=4/128k |
| Volta FA ncols=32 (T18) | rejected: +0.11% at 128k production |

## Known latent issues (as of this archive)

- Non-replay mode keeps the same conv-plane hazard that caused the production incident: the
  save writes a single R plane. The fix covers replay mode only (storing all planes in
  non-replay mode would cost about 600 MiB per 27B checkpoint). Do not run "restore, then
  partial rollback" with MTP on non-replay builds.
- `llama_memory_hybrid_iswa` state write/read ignores `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`;
  SWA hybrid models could restore a wrong recurrent state (not triggered on qwen35).
- `server_tokens::pos_last()` media branch is only correct for M-ROPE models; for
  NORMAL/HUNYUANVL position types the park guard skips (with a warning) a prompt that ends
  in an image.
- Multi-GPU: layer split is expected to work but was never run; row split breaks ReplaySSM
  (the record layout derives from tensor strides).
- `--cache-ram`/prompt cache is superseded by `--kv-tree` (the tree branch ignores it).
- The T24 rollback+replay path can show run-to-run value differences (~1/6 of runs,
  near-ties only, replay=0 clean); recorded as a follow-up, not fixed.
