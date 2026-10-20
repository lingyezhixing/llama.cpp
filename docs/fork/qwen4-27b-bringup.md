# Qwen4-27B bring-up checklist

Private fork checklist. Not upstream-supported. Use at your own risk.

Status: pre-release. No Qwen4-27B weights or config are available yet.
The working hypothesis is that Qwen4-27B is a dense hybrid model derived from
Qwen3.5/3.8: GDN layers plus QSA full-attention layers, PLE n-gram at layer 2,
and hyper-connections. Treat every value below as unconfirmed until the official
config is published.

## 0. Working assumptions

Replace these with the official values as soon as they exist.

| item | working assumption | source to confirm |
|---|---|---|
| architecture | `qwen4` or a `qwen4exp` derivative | `config.json`, HF model card |
| trunk | dense 27B | official model card |
| layers | 64 | `num_hidden_layers` |
| hidden | 5120 | `hidden_size` |
| Q / KV heads | 24 / 4 | `num_attention_heads`, `num_key_value_heads` |
| attention head dim | 256 | `head_dim` |
| full attention interval | 4 | `full_attention_interval`, `layer_types` |
| GDN | 16 key heads, 48 value heads, d_state 128 | `linear_*`, `ssm_*` |
| QSA | 4 indexer heads x 128, top_k 2048, compress 4 | `indexer_*` |
| HC | 4 streams, low rank TBD | `hc_count`, `hc_lowrank` |
| PLE | layer 2, ngram 3, 8 heads per ngram | `ple_*`, `ngram_*` |
| MTP | 1 block | `mtp_*` |
| context | 262144 | `max_position_embeddings` |
| vocab | 248320 | `vocab_size` |

## 1. Pre-release preparation

These items do not depend on the final tensor names.

- [ ] Keep `qwen4-opt` clean. Rebase it on `upstream/master` before release day.
- [ ] Create a safety branch and tag the current upstream base.
- [ ] Record the current Qwen3.8-27B V100 baseline on the target machine.
- [ ] Keep the existing Qwen3.8-27B production build and its benchmark data.
- [ ] Prepare a synthetic dense-hybrid fixture with toggles:
  - dense FFN vs MoE
  - HC on/off
  - QSA on/off
  - PLE on/off
  - MTP on/off
- [ ] Prepare a RAM/VRAM calculator for weights, KV, indexer cache, PLE, and
      compute buffers.
- [ ] Prepare a release-day directory for configs, logs, benchmark JSON, and
      hashes.
- [ ] Prepare a rollback plan for the production server.

## 2. Release-day intake

Do this before writing any model code.

- [ ] Download `config.json`, `generation_config.json`, tokenizer files,
      chat template, model card, and `model.safetensors.index.json`.
- [ ] Record SHA256 of every downloaded metadata file.
- [ ] Extract and record every field in the official text config.
- [ ] Compare official values with the working assumptions in section 0.
- [ ] Decide whether to extend `qwen4exp` or add a new architecture.
- [ ] Inspect tensor names and shapes from the index file. Do not guess names.
- [ ] Inspect the chat template and special tokens.
- [ ] Inspect the vision config and mmproj layout, if the model is multimodal.
- [ ] Inspect the MTP head and whether it is shared or separate.
- [ ] Create branch `qwen4-27b-bringup` from `qwen4-opt`.

Stop if any of these differ from the working hypothesis:

- dense vs MoE
- HC count or HC low rank
- QSA indexer layout
- PLE layer or ngram size
- attention head dim
- tokenizer or vocab size
- MTP structure

## 3. Conversion and metadata

- [ ] Convert BF16 first. Do not start with a quantized file.
- [ ] Add or update the architecture enum in `src/llama-arch.cpp`.
- [ ] Add tensor names and shapes in `src/llama-arch.cpp` and `src/llama-arch.h`.
- [ ] Add hparams in `src/llama-hparams.h` and the model loader.
- [ ] Add the model factory case in `src/llama-model.cpp`.
- [ ] Add the model class and graph in `src/models/`.
- [ ] Reuse `conversion/qwen4exp.py` where possible. Add aliases only after
      checking the official tensor names.
- [ ] Reuse Qwen3.5 GDN, Qwen4Exp QSA, Qwen4Exp HC, and Qwen4Exp PLE code paths
      where they match.
- [ ] Add a dense FFN path if the model is not MoE.
- [ ] Add a dense/no-HC path if the model does not use HC.
- [ ] Verify the converted tensor count and total size against the HF index.
- [ ] Verify the tokenizer round-trip and chat template rendering.
- [ ] Verify the vision projector type if mmproj is used.
- [ ] Convert a small slice or a synthetic fixture before the full model.

Useful references:

- `conversion/qwen4exp.py`
- `src/models/qwen4exp.cpp`
- `src/llama-arch.cpp`
- `src/llama-model.cpp`
- `src/llama-hparams.h`
- `src/llama-memory-hybrid-idx.{h,cpp}`
- `tests/test-llama-archs.cpp`

## 4. CPU bring-up

- [ ] Run `test-llama-archs` for the new architecture with a small fixture.
- [ ] Load the real model with `-ngl 0` and a short prompt.
- [ ] Check layer assignment: GDN layers, QSA layers, PLE layer, MTP block.
- [ ] Check for shape asserts, NaNs, and early EOS.
- [ ] Run a fixed prompt and compare logits with the HF reference.
- [ ] Run greedy generation and compare tokens with the HF reference.
- [ ] Run `llama-perplexity` on a fixed text and record the value.

Suggested commands:

```sh
build/bin/Release/test-llama-archs.exe -a <arch> -b CPU -v 4
build/bin/Release/llama-cli.exe -m <model.gguf> -ngl 0 -p "Hello" -n 16 --jinja
build/bin/Release/llama-perplexity.exe -m <model.gguf> -f <text> -c 512 --chunks 8 -ngl 0
```

## 5. CUDA and V100 bring-up

### 5.1 Build

- [ ] Build Release with CUDA 12.9 and `CMAKE_CUDA_ARCHITECTURES=70-real;89-real`.
- [ ] Build a second binary with `GGML_CUDA_FORCE_MMQ=ON` only if the model is
      MoE. Test it separately; do not use it for the dense baseline.
- [ ] Verify `llama-cli --version` and `--list-devices`.
- [ ] Run `test-backend-ops` for the affected ops.

### 5.2 Op tests

- [ ] `GATED_DELTA_NET`
- [ ] `SSM_SCAN` if used
- [ ] `FLASH_ATTN_EXT` with head dim 256
- [ ] `LIGHTNING_INDEXER` with 4 heads x 128
- [ ] `TOP_K`, `CUMSUM`, `SET_ROWS`
- [ ] `DSV4_HC_PRE`, `DSV4_HC_POST`, `DSV4_HC_COMB`
- [ ] `MUL_MAT_ID` with the official expert count
- [ ] `GET_ROWS` for the PLE table

Example:

```sh
build/bin/Release/test-backend-ops.exe -b CUDA0 -o GATED_DELTA_NET
build/bin/Release/test-backend-ops.exe -b CUDA0 -o FLASH_ATTN_EXT
build/bin/Release/test-backend-ops.exe -b CUDA0 -o LIGHTNING_INDEXER
build/bin/Release/test-backend-ops.exe -b CUDA0 -o DSV4_HC_POST
```

### 5.3 V100-specific checks

- [ ] Confirm QSA sparse attention is not active on sm70. The current gate is
      `turing_mma_available()` in `ggml/src/ggml-cuda/fattn.cu`.
- [ ] Confirm the dense QSA fallback is correct before measuring speed.
- [ ] Confirm `hc_count == 4` if the HC kernels are used. `dsv4-hc.cu` hardcodes
      four streams.
- [ ] Confirm the GDN path is the generic fallback. The MMA path requires
      Ampere+ in `ggml/src/ggml-cuda/gated-delta-net-mma.cu`.
- [ ] Confirm `per_layer_token_embd` is host-resident and not counted in VRAM.
- [ ] Confirm the indexer cache is allocated and sized as expected.
- [ ] Confirm the KV cache type and indexer cache type.

### 5.4 First GPU run

```sh
build/bin/Release/llama-cli.exe -m <model.gguf> -ngl 99 -fa on \
  -ctk q8_0 -ctv q8_0 -p "Hello" -n 16 --jinja
```

- [ ] No OOM at the smallest context.
- [ ] Output matches the CPU reference within tolerance.
- [ ] No NaNs, no repeated tokens, no early EOS.
- [ ] Check VRAM, RSS, and PLE page cache.

## 6. Memory and placement

- [ ] Estimate weights, KV, indexer cache, PLE, and compute buffers separately.
- [ ] Keep PLE out of VRAM. Use lazy read or `-ot per_layer_token_embd=CPU`.
- [ ] For MoE, tune `-ncmoe` from the smallest value that fits.
- [ ] Always use `--no-op-offload` when expert weights are in CPU memory.
- [ ] Tune `--moe-cache-mib` only after the baseline works.
- [ ] Test q8_0 KV before f16 for long context.
- [ ] Test 32K, 128K, and 262K contexts only after the short-context path is
      correct.

Suggested starting point for a 32 GB V100 with a dense 27B model:

```sh
-ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 512
```

Suggested starting point if the model is MoE:

```sh
-ncmoe 32 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -ub 256 --no-op-offload
```

## 7. Correctness gates

Do not claim support until all gates pass.

- [ ] Gate A: model loads and generates on CPU.
- [ ] Gate B: V100 logits match CPU within the accepted tolerance.
- [ ] Gate C: QSA, GDN, HC, PLE, MoE, and MTP paths are each exercised.
- [ ] Gate D: 3 fixed prompts match the HF reference.
- [ ] Gate E: PPL is within the expected band.
- [ ] Gate F: needle-in-haystack at 10%, 50%, and 90% depth.
- [ ] Gate G: multi-turn prefix reuse and slot save/restore.
- [ ] Gate H: MTP on/off produces the same tokens at temperature 0.
- [ ] Gate I: quantized model passes the same gates as BF16.

## 8. Performance baseline

- [ ] `llama-bench` pp512, pp4096, pp8192.
- [ ] `llama-bench` tg128 at d0, d4096, d32768, d131072.
- [ ] Server: short prompt, 128 generated tokens, MTP off.
- [ ] Server: short prompt, 128 generated tokens, MTP on.
- [ ] Server: 32K, 128K, 262K contexts if memory allows.
- [ ] Record TTFT, prompt tok/s, decode tok/s, acceptance, VRAM, RSS.
- [ ] Compare with the Qwen3.8-27B baseline on the same machine.
- [ ] A/B the existing V100 patches one at a time.

Keep the same-session alternating A/B rule. Do not compare across sessions.

## 9. Failure playbook

### Load fails

- Check `LLM_ARCH_*`, tensor names, tensor shapes, and hparams.
- Check the tokenizer and vocab size.
- Check whether the model is dense or MoE.

### CPU works, CUDA differs

- Compare CPU and CUDA logits layer by layer.
- Check QSA mask construction and indexer cache.
- Check HC mix/combine and the final HC head.
- Check PLE row hashing and the ngram window.
- Check MTP draft state.

### V100 is slow

- Confirm QSA sparse is not silently expected on sm70.
- Confirm the dense QSA fallback cost.
- Check the 256x256 FA config and FATTN split.
- Check GDN generic kernel time.
- Check MoE host scheduling and `--no-op-offload`.
- Check whether `GGML_CUDA_FORCE_MMQ` helps the MoE build.

### OOM

- Reduce context.
- Use q8_0 KV.
- Increase `-ncmoe`.
- Reduce `--moe-cache-mib`.
- Use a smaller quant.
- Move PLE to host mmap if it is in VRAM.

### Output degenerates

- Check QSA block selection and mask.
- Check recurrent state rollback.
- Check PLE hash constants and EOS reset.
- Check MTP draft state.
- Check quantized KV compatibility.

## 10. Do not do

- Do not hardcode tensor names from the Qwen4Exp preview.
- Do not assume dense vs MoE without the official config.
- Do not assume HC count is 4 without checking the config.
- Do not claim V100 sparse QSA support. The current gate excludes sm70.
- Do not benchmark cold caches or mix sessions.
- Do not compare different quantizations as if they were the same.
- Do not enable `GGML_CUDA_FORCE_MMQ` for the dense baseline without A/B.
- Do not merge speculative Qwen4-27B code before Gate C passes.

## 11. Deliverables

- [ ] `docs/fork/qwen4-27b.md` with support status and known limits.
- [ ] Conversion command log with hashes.
- [ ] Benchmark JSON and CSV.
- [ ] Patch list with measured effects.
- [ ] Known-issues list.
- [ ] Rollback instructions for the production server.
