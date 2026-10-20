# Qwen4-27B release-day worksheet

Fill this in on release day. Use together with
`docs/fork/qwen4-27b-bringup.md`.

Do not edit the checklist while filling this. Record decisions here, then
update the checklist only after the bring-up is done.

## 1. Snapshot

| item | value |
|---|---|
| date / time | |
| upstream commit | |
| fork branch | |
| model repo | |
| model revision | |
| config.json sha256 | |
| generation_config.json sha256 | |
| tokenizer.json sha256 | |
| chat template sha256 | |
| safetensors index sha256 | |
| mmproj sha256 | |
| test machine | |

## 2. Official config vs working assumptions

| field | official value | working assumption | match? | action |
|---|---|---|---|---|
| architectures | | Qwen4... | | |
| model_type | | qwen4 / qwen4exp | | |
| num_hidden_layers | | 64 | | |
| hidden_size | | 5120 | | |
| num_attention_heads | | 24 | | |
| num_key_value_heads | | 4 | | |
| head_dim | | 256 | | |
| full_attention_interval | | 4 | | |
| layer_types | | GDN + QSA | | |
| linear_num_key_heads | | 16 | | |
| linear_num_value_heads | | 48 | | |
| linear_key_head_dim | | 128 | | |
| linear_value_head_dim | | 128 | | |
| linear_conv_kernel_dim | | 4 | | |
| hc_count | | 4 | | |
| hc_lowrank | | TBD | | |
| indexer_n_heads | | 4 | | |
| indexer_head_dim | | 128 | | |
| indexer_budget | | 2048 | | |
| indexer_compress_ratio | | 4 | | |
| ngram_size | | 3 | | |
| heads_per_ngram | | 8 | | |
| ple_layer_ids | | layer 2 | | |
| ple_embed_dim | | 5120 | | |
| num_experts | | 0 (dense) | | |
| num_experts_per_tok | | 0 (dense) | | |
| moe_intermediate_size | | n/a | | |
| mtp_num_hidden_layers | | 1 | | |
| max_position_embeddings | | 262144 | | |
| vocab_size | | 248320 | | |
| rope_theta | | 10000000 | | |
| partial_rotary_factor | | 0.25 | | |
| mrope_section | | 11, 11, 10 | | |
| vision depth | | 27 | | |
| vision hidden | | 1152 | | |
| vision out_hidden | | 5120 | | |

Stop and reassess if any of these differ:

- dense vs MoE
- HC count or HC low rank
- QSA indexer layout or top-k
- PLE layer or ngram size
- attention head dim
- tokenizer or vocab size
- MTP structure

## 3. Tensor inventory

| item | value |
|---|---|
| total tensors | |
| total bytes | |
| per-layer tensors | |
| QSA tensors | |
| HC tensors | |
| PLE tensors | |
| MTP tensors | |
| vision tensors | |
| unknown tensors | |

Tensor name deltas against `qwen4exp`:

```text

```

## 4. Decisions

| decision | value | reason |
|---|---|---|
| extend `qwen4exp` or add new arch | | |
| dense FFN or MoE | | |
| HC on/off | | |
| QSA on/off | | |
| PLE on/off | | |
| MTP on/off | | |
| V100 QSA sparse expected | no, dense fallback | |
| quantization plan | | |
| target context | | |
| target KV type | | |
| target `-ncmoe` | | |
| target `--moe-cache-mib` | | |

## 5. Gate results

| gate | result | notes |
|---|---|---|
| A: CPU load and generate | | |
| B: V100 logits match CPU | | |
| C: QSA / GDN / HC / PLE / MoE / MTP exercised | | |
| D: HF reference prompts | | |
| E: PPL band | | |
| F: needle at 10 / 50 / 90% | | |
| G: prefix reuse and slot restore | | |
| H: MTP on/off tokens match | | |
| I: quantized model gates | | |

## 6. Benchmark results

### llama-bench

| test | value | notes |
|---|---|---|
| pp512 | | |
| pp4096 | | |
| pp8192 | | |
| tg128 d0 | | |
| tg128 d4096 | | |
| tg128 d32768 | | |
| tg128 d131072 | | |

### Server

| test | TTFT | prompt tok/s | decode tok/s | acceptance | VRAM | RSS |
|---|---|---|---|---|---|---|
| short prompt, MTP off | | | | | | |
| short prompt, MTP on | | | | | | |
| 32K | | | | | | |
| 128K | | | | | | |
| 262K | | | | | | |

### Comparison with Qwen3.8-27B

| point | Qwen3.8-27B | Qwen4-27B | delta |
|---|---|---|---|
| pp512 | | | |
| pp8192 | | | |
| tg128 d0 | | | |
| tg128 d32768 | | | |
| server short decode | | | |

## 7. Issues and rollback

Blockers:

```text

```

Workarounds:

```text

```

Rollback:

```text

```

## 8. Final status

| item | value |
|---|---|
| supported on CPU | |
| supported on CUDA | |
| supported on V100 | |
| recommended quant | |
| recommended context | |
| recommended flags | |
| known limits | |
| follow-up work | |
