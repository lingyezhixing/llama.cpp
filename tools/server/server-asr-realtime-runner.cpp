#include "server-asr-realtime.h"

#include "server-asr-engine.h"
#include "server-common.h"

#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

namespace {

constexpr int64_t REALTIME_DECODE_INTERVAL_MS = 1000;
constexpr int32_t REALTIME_MAX_NEW_TOKENS     = 32;

struct gen_result {
    std::vector<llama_token> tokens;
    int64_t prompt_tokens = 0;
    int32_t generated     = 0;
    bool    capped        = false;
};

std::string build_prompt(const std::string & context, const std::string & marker,
                         const std::string & forced_language, const std::string & prefix_text) {
    std::string prompt;
    prompt += "<|im_start|>system\n" + context + "<|im_end|>\n";
    prompt += "<|im_start|>user\n" + marker + "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";
    if (!forced_language.empty()) {
        prompt += "language " + forced_language + "<asr_text>";
    }
    prompt += prefix_text;
    return prompt;
}

// drops the "language X<asr_text>" prefix at the token level
std::vector<llama_token> strip_metadata(const std::vector<llama_token> & tokens, llama_token marker_id, bool forced) {
    if (forced) {
        return tokens;
    }

    const auto it = std::find(tokens.begin(), tokens.end(), marker_id);
    if (it == tokens.end()) {
        // still printing the metadata: nothing is safe to commit
        return {};
    }

    return std::vector<llama_token>(it + 1, tokens.end());
}

} // namespace

class server_asr_realtime_runner_impl : public server_asr_realtime_runner {
public:
    explicit server_asr_realtime_runner_impl(server_asr_engine * engine) : engine(engine) {}

    bool start(const server_asr_realtime_config & config, server_asr_error & err) override {
        err = {};

        this->config  = config;
        this->forced  = server_asr_normalize_language(config.language);
        this->holdback = server_asr_holdback(config.stability);

        segment_audio.clear();
        commit = {};

        commit.holdback_tokens = holdback;

        audio_ms          = 0;
        processed_audio_ms = 0;
        processing_ms     = 0;
        pending_utf8.clear();

        usage = {};
        steps = 0;

        if (engine->ensure_context(err) == nullptr) {
            return false;
        }

        llama_context * ctx = engine->ensure_context(err);
        llama_memory_clear(llama_get_memory(ctx), true);

        const llama_vocab * vocab = engine->vocab();

        metadata_tokens = common_tokenize(vocab, "<asr_text>", false, true);
        if (metadata_tokens.size() != 1) {
            err = { "internal_error", "the tokenizer has no single <asr_text> token", "" };
            return false;
        }
        metadata_id = metadata_tokens[0];

        started_at = std::chrono::steady_clock::now();

        return true;
    }

    bool push(const float * samples_16k, size_t n_samples, server_asr_realtime_update & update,
              server_asr_error & err) override {
        err = {};
        update = {};

        if (samples_16k == nullptr || n_samples == 0) {
            return true;
        }

        segment_audio.insert(segment_audio.end(), samples_16k, samples_16k + n_samples);
        audio_ms += (int64_t) (n_samples * 1000 / 16000);

        if (audio_ms - processed_audio_ms < REALTIME_DECODE_INTERVAL_MS) {
            return true;
        }

        return decode(update, err, false);
    }

    bool finish(server_asr_realtime_update & update, server_asr_realtime_stats & stats,
                server_asr_error & err) override {
        err = {};
        update = {};

        if (audio_ms > processed_audio_ms || !segment_audio.empty()) {
            if (!decode(update, err, true)) {
                return false;
            }
        }

        // commit everything that is left: the stream ends here
        flush_remaining(update);

        stats.processing_ms = (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started_at).count();
        stats.usage = usage;
        stats.usage.audio_ms = audio_ms;

        return true;
    }

    void cancel() override {
        segment_audio.clear();
        commit = {};
        pending_utf8.clear();
    }

private:
    server_asr_engine * engine;

    server_asr_realtime_config config;
    std::string forced;
    int32_t     holdback = 8;

    std::vector<float> segment_audio;
    server_asr_commit_state commit;

    std::vector<llama_token> metadata_tokens;
    llama_token metadata_id = 0;

    std::string pending_utf8;

    int64_t audio_ms           = 0;
    int64_t processed_audio_ms = 0;
    int64_t processing_ms      = 0;
    int64_t steps              = 0;

    server_asr_usage usage;

    std::chrono::steady_clock::time_point started_at;

    static int32_t server_asr_holdback(const std::string & stability) {
        return server_asr_realtime_holdback_tokens(stability);
    }

    void append_delta(server_asr_realtime_update & update, const std::string & text) {
        if (!text.empty()) {
            update.deltas += text;
        }
    }

    // detokenizes [begin, end) and moves the commit pointers forward
    void emit_span(size_t end, server_asr_realtime_update & update) {
        const size_t base  = commit.segment_base_tokens.size();
        const size_t begin = base + (size_t) commit.emitted_in_segment;

        if (end <= begin) {
            return;
        }

        const llama_vocab * vocab = engine->vocab();

        std::vector<llama_token> span(commit.segment_raw_tokens.begin() + (ptrdiff_t) begin,
                                      commit.segment_raw_tokens.begin() + (ptrdiff_t) end);
        std::string text = common_detokenize(vocab, span, false);

        std::string complete;
        const size_t held = server_asr_utf8_holdback(pending_utf8 + text, complete);

        pending_utf8.clear();
        if (held > 0) {
            pending_utf8 = (pending_utf8 + text).substr(complete.size());
        }

        append_delta(update, complete);

        commit.emitted_in_segment = (int64_t) (end - base);
        commit.committed_tokens  += (int64_t) span.size();
        usage.committed_output_tokens += (int64_t) span.size();
    }

    void flush_remaining(server_asr_realtime_update & update) {
        const size_t raw_size = commit.segment_raw_tokens.size();
        emit_span(raw_size, update);

        if (!pending_utf8.empty()) {
            append_delta(update, pending_utf8);
            pending_utf8.clear();
        }
    }

    bool decode(server_asr_realtime_update & update, server_asr_error & err, bool flush) {
        llama_context * ctx = engine->ensure_context(err);
        if (ctx == nullptr) {
            return false;
        }

        mtmd_context * mctx = engine->mctx();
        const llama_vocab * vocab = engine->vocab();

        if (mctx == nullptr || vocab == nullptr) {
            err = { "internal_error", "the ASR engine has no multimodal context", "" };
            return false;
        }

        const std::string marker = mtmd_get_marker(mctx);
        const std::string prefix_text = common_detokenize(vocab, commit.prompt_prefix_tokens, false);
        const std::string prompt = build_prompt(config.context, marker, forced, prefix_text);

        mtmd_bitmap * bitmap = mtmd_bitmap_init_from_audio(segment_audio.size(), segment_audio.data());
        if (bitmap == nullptr) {
            err = { "internal_error", "failed to build the realtime bitmap", "" };
            return false;
        }

        struct chunks_guard {
            mtmd_input_chunks * p = nullptr;
            ~chunks_guard() { if (p != nullptr) { mtmd_input_chunks_free(p); } }
        } chunks{mtmd_input_chunks_init()};

        mtmd_input_text mtext = { prompt.c_str(), prompt.size(), false, true };
        const mtmd_bitmap * bitmaps[] = { bitmap };
        const int32_t tokenize_ret = mtmd_tokenize(mctx, chunks.p, &mtext, bitmaps, 1);
        mtmd_bitmap_free(bitmap);

        if (tokenize_ret != 0) {
            err = { "internal_error", "failed to tokenize the realtime prompt", "" };
            return false;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        struct batch_guard {
            llama_batch_ext * p = nullptr;
            ~batch_guard() { llama_batch_ext_free(p); }
        } batch{llama_batch_ext_init(ctx)};

        const int32_t n_batch = (int32_t) llama_n_batch(ctx);
        const size_t n_chunks = mtmd_input_chunks_size(chunks.p);

        llama_pos n_past  = 0;
        int64_t n_prompt  = 0;
        bool have_logits  = false;

        for (size_t i = 0; i < n_chunks; i++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.p, i);

            switch (mtmd_input_chunk_get_type(chunk)) {
                case MTMD_INPUT_CHUNK_TYPE_TEXT:
                    {
                        size_t n_tokens = 0;
                        const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
                        n_prompt += (int64_t) n_tokens;

                        size_t k = 0;
                        while (k < n_tokens) {
                            llama_batch_ext_clear(batch.p);

                            const size_t slice = std::min<size_t>((size_t) n_batch, n_tokens - k);
                            int32_t last_idx = -1;

                            for (size_t j = 0; j < slice; j++, k++) {
                                const int32_t idx = llama_batch_ext_add_token(batch.p, 0, tokens[k]);
                                if (idx < 0) {
                                    err = { "internal_error", "the realtime prompt does not fit the batch", "" };
                                    return false;
                                }
                                const llama_pos p = n_past++;
                                llama_batch_ext_set_pos(batch.p, idx, &p);
                                last_idx = idx;
                            }

                            if (i + 1 == n_chunks && k == n_tokens) {
                                llama_batch_ext_set_output_logits(batch.p, last_idx, true);
                                have_logits = true;
                            }

                            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.p) != 0) {
                                err = { "internal_error", "the realtime prompt decode failed", "" };
                                return false;
                            }
                        }
                    }
                    break;

                case MTMD_INPUT_CHUNK_TYPE_AUDIO:
                    {
                        n_prompt += (int64_t) mtmd_input_chunk_get_n_tokens(chunk);

                        if (mtmd_encode_chunk(mctx, chunk) != 0) {
                            err = { "internal_error", "failed to encode realtime audio", "" };
                            return false;
                        }

                        float * embd = mtmd_get_output_embd(mctx);
                        if (mtmd_helper_decode_image_chunk(mctx, ctx, chunk, embd, n_past, 0, n_batch, &n_past,
                                    nullptr, nullptr) != 0) {
                            err = { "internal_error", "the realtime audio prefill failed", "" };
                            return false;
                        }
                    }
                    break;

                default:
                    break;
            }
        }

        if (!have_logits) {
            err = { "internal_error", "the realtime prompt produced no logits", "" };
            return false;
        }

        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        const llama_token eos = llama_vocab_eos(vocab);
        const llama_token eot = llama_vocab_eot(vocab);

        gen_result gen;
        gen.prompt_tokens = n_prompt;

        while (gen.generated < REALTIME_MAX_NEW_TOKENS) {
            const llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (id == eos || id == eot) {
                break;
            }

            gen.tokens.push_back(id);
            gen.generated++;

            llama_batch_ext_clear(batch.p);

            const int32_t idx = llama_batch_ext_add_token(batch.p, 0, id);
            if (idx < 0) {
                err = { "internal_error", "the generated token does not fit the batch", "" };
                llama_sampler_free(smpl);
                return false;
            }

            const llama_pos p = n_past++;
            llama_batch_ext_set_pos(batch.p, idx, &p);
            llama_batch_ext_set_output_logits(batch.p, idx, true);

            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.p) != 0) {
                err = { "internal_error", "the realtime decode failed", "" };
                llama_sampler_free(smpl);
                return false;
            }
        }

        llama_sampler_free(smpl);

        gen.capped = gen.generated >= REALTIME_MAX_NEW_TOKENS;

        // stable commit: only the common prefix with the previous decode is safe
        const std::vector<llama_token> text_tokens =
            strip_metadata(gen.tokens, metadata_id, !forced.empty());

        commit.segment_raw_tokens.clear();
        commit.segment_raw_tokens.insert(commit.segment_raw_tokens.end(),
                commit.prompt_prefix_tokens.begin(), commit.prompt_prefix_tokens.end());
        commit.segment_raw_tokens.insert(commit.segment_raw_tokens.end(),
                text_tokens.begin(), text_tokens.end());

        size_t begin = 0;
        size_t end   = 0;
        if (server_asr_stable_commit(commit, begin, end)) {
            emit_span(end, update);
        }

        const size_t keep = commit.segment_raw_tokens.size() > (size_t) std::max(0, commit.holdback_tokens)
            ? commit.segment_raw_tokens.size() - (size_t) commit.holdback_tokens
            : 0;
        commit.prompt_prefix_tokens.assign(commit.segment_raw_tokens.begin(),
                                           commit.segment_raw_tokens.begin() + (ptrdiff_t) keep);
        commit.prev_segment_raw_tokens = commit.segment_raw_tokens;

        usage.input_tokens  += n_prompt;
        usage.output_tokens += gen.generated;
        usage.total_tokens   = usage.input_tokens + usage.output_tokens;
        usage.decode_steps  += 1;

        steps++;
        processed_audio_ms = audio_ms;
        (void) flush;

        return true;
    }
};

std::unique_ptr<server_asr_realtime_runner> server_asr_make_realtime_runner(server_asr_engine * engine) {
    return std::make_unique<server_asr_realtime_runner_impl>(engine);
}
