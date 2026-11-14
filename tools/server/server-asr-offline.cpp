#include "server-asr-offline.h"

#include "server-asr-audio.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>

// 180 s of audio in 60 s chunks, so progress can be checked between them
static constexpr int64_t FAKE_CHUNK_MS  = 60000;
static constexpr int64_t FAKE_TOTAL_MS  = 180000;
static constexpr int     FAKE_CHUNKS    = 3;
// keeps a job in RUNNING long enough for a test to observe or cancel it
static constexpr int     FAKE_STEP_MS   = 150;

// tests widen the window with LLAMA_SERVER_ASR_FAKE_STEP_MS
static int fake_step_ms() {
    if (const char * env = std::getenv("LLAMA_SERVER_ASR_FAKE_STEP_MS"); env != nullptr) {
        const int value = std::atoi(env);
        if (value > 0) {
            return value;
        }
    }
    return FAKE_STEP_MS;
}

bool server_asr_fake_runner::probe(const std::string & audio_path, int64_t & audio_total_ms, server_asr_error & err) {
    std::error_code ec;
    if (audio_path.empty() || !std::filesystem::exists(audio_path, ec)) {
        err = { "invalid_audio", "the audio file is missing", "file" };
        return false;
    }

    // the fake audio body still has to look like a container
    if (!server_asr_audio_container_supported(audio_path)) {
        err = { "unsupported_audio_format", "the audio must be WAV, MP3 or FLAC", "file" };
        return false;
    }

    audio_total_ms = FAKE_TOTAL_MS;
    return true;
}

server_asr_run_status server_asr_fake_runner::run(
        const server_asr_job_input & input,
        server_asr_run_state & state,
        const std::atomic<bool> & cancel,
        const std::function<bool()> & pause_requested,
        const std::function<void(int64_t, const server_asr_usage &)> & progress,
        server_asr_error & err) {
    while (state.next_chunk < FAKE_CHUNKS) {
        if (cancel.load()) {
            return server_asr_run_status::CANCELLED;
        }
        if (pause_requested()) {
            return server_asr_run_status::PAUSED;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(fake_step_ms()));

        state.next_chunk++;
        state.audio_processed_ms = std::min<int64_t>(FAKE_TOTAL_MS, (int64_t) state.next_chunk * FAKE_CHUNK_MS);

        state.usage.audio_ms      = state.audio_processed_ms;
        state.usage.chunks        = (int64_t) state.next_chunk;
        state.usage.input_tokens += 100;
        state.usage.output_tokens += 10;
        state.usage.total_tokens  = state.usage.input_tokens + state.usage.output_tokens;

        state.result.text          = "fake transcription";
        state.result.language      = input.language.empty() ? "English" : input.language;
        state.result.language_code = input.language.empty() ? "en" : "";

        if (input.timestamps) {
            state.result.items.clear();
            for (int c = 0; c < (int) state.next_chunk; c++) {
                state.result.items.push_back({ "fake", c * FAKE_CHUNK_MS, (c + 1) * FAKE_CHUNK_MS });
            }
        }

        progress(state.audio_processed_ms, state.usage);
    }

    err = {};
    return server_asr_run_status::DONE;
}
//
// chunk planning
//

// constants from the design, section 8.7
static constexpr double ASR_AUDIO_TOKENS_PER_SEC  = 13.0;
static constexpr double ASR_OUTPUT_TOKENS_PER_SEC = 7.0;
static constexpr double ASR_PROMPT_OVERHEAD       = 64.0;
static constexpr double ASR_SAFETY_FACTOR         = 0.85;
static constexpr double ASR_ALIGNER_TOKENS_PER_SEC = 18.0; // 6 items/s, 3 tokens each

static constexpr int64_t ASR_MAX_CHUNK_NO_TS_MS = 1200 * 1000;
static constexpr int64_t ASR_MAX_CHUNK_TS_MS    = 180 * 1000;
static constexpr int64_t ASR_ENERGY_WINDOW_MS   = 100;
static constexpr int64_t ASR_SEARCH_MS          = 5000;

int64_t server_asr_max_chunk_ms(int32_t n_ctx_seq, int32_t context_tokens, bool timestamps, int32_t aligner_ctx) {
    if (n_ctx_seq <= 0) {
        return 0;
    }

    const double usable = n_ctx_seq * ASR_SAFETY_FACTOR - ASR_PROMPT_OVERHEAD - context_tokens;
    if (usable <= 0.0) {
        return 0;
    }

    double max_sec = usable / (ASR_AUDIO_TOKENS_PER_SEC + ASR_OUTPUT_TOKENS_PER_SEC);

    if (timestamps) {
        if (aligner_ctx <= 0) {
            return 0;
        }
        const double aligner_usable = aligner_ctx * ASR_SAFETY_FACTOR - ASR_PROMPT_OVERHEAD;
        if (aligner_usable <= 0.0) {
            return 0;
        }
        const double aligner_sec = aligner_usable / (ASR_AUDIO_TOKENS_PER_SEC + ASR_ALIGNER_TOKENS_PER_SEC);
        max_sec = std::min(max_sec, aligner_sec);
    }

    int64_t chunk_ms = (int64_t) (max_sec * 1000.0);
    chunk_ms = std::min(chunk_ms, timestamps ? ASR_MAX_CHUNK_TS_MS : ASR_MAX_CHUNK_NO_TS_MS);

    return chunk_ms;
}

void server_asr_compute_chunk_ranges(
        int64_t audio_total_ms,
        int32_t n_ctx_seq,
        int32_t context_tokens,
        bool    timestamps,
        int32_t aligner_ctx,
        const std::vector<float> & energy_per_100ms,
        std::vector<server_asr_chunk_range> & out,
        server_asr_error & err) {
    out.clear();
    err = {};

    if (audio_total_ms <= 0) {
        err = { "invalid_audio", "the audio is empty", "" };
        return;
    }

    const int64_t max_chunk_ms = server_asr_max_chunk_ms(n_ctx_seq, context_tokens, timestamps, aligner_ctx);
    if (max_chunk_ms < 1000) {
        err = { "context_too_small", "the context size cannot hold even a one second chunk", "" };
        return;
    }

    auto energy_at = [&](int64_t start_ms) -> float {
        const size_t index = (size_t) (start_ms / ASR_ENERGY_WINDOW_MS);
        if (index >= energy_per_100ms.size()) {
            return 0.0f;
        }
        return energy_per_100ms[index];
    };

    int64_t start = 0;

    while (start < audio_total_ms) {
        const int64_t remaining = audio_total_ms - start;
        if (remaining <= max_chunk_ms) {
            out.push_back({ start, audio_total_ms });
            break;
        }

        const int64_t target = start + max_chunk_ms;

        // search around the target, never touching the current start or the end
        int64_t lo = std::max(start + 1000, target - ASR_SEARCH_MS);
        int64_t hi = std::min(audio_total_ms - 1000, target + ASR_SEARCH_MS);

        lo = (lo / ASR_ENERGY_WINDOW_MS) * ASR_ENERGY_WINDOW_MS;
        hi = (hi / ASR_ENERGY_WINDOW_MS) * ASR_ENERGY_WINDOW_MS;

        int64_t cut = target;
        if (hi > lo) {
            float best_energy = energy_at(lo);
            int64_t best_ms = lo;

            for (int64_t ms = lo; ms <= hi; ms += ASR_ENERGY_WINDOW_MS) {
                const float energy = energy_at(ms);
                if (energy < best_energy) {
                    best_energy = energy;
                    best_ms     = ms;
                }
            }
            cut = best_ms;
        }

        if (cut <= start) {
            cut = target;
        }
        if (cut > audio_total_ms) {
            cut = audio_total_ms;
        }

        out.push_back({ start, cut });
        start = cut;
    }
}
#include "server-asr-engine.h"

#include "mtmd-helper.h"

#include <memory>
#include <string>
#include <vector>

namespace {

enum class chunk_status {
    OK,
    PAUSED,
    CANCELLED,
    FAILED,
};

struct chunk_result {
    std::string text;
    std::string language;
    std::string language_code;
    server_asr_usage usage;
    int64_t prompt_tokens = 0;
    int32_t generated     = 0;
    bool    capped        = false;
};

// gluing two Latin words needs a space, as documented in the design
bool needs_space(char prev, char next) {
    auto is_latin = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    };
    return is_latin(prev) && is_latin(next);
}

int32_t max_new_tokens_for(int32_t n_ctx, int64_t n_prompt_tokens) {
    const int64_t usable = (int64_t) (n_ctx * 0.85) - n_prompt_tokens;
    if (usable <= 0) {
        return 0;
    }
    return (int32_t) std::min<int64_t>(2048, usable);
}

} // namespace

class server_asr_offline_runner_impl : public server_asr_offline_runner {
public:
    server_asr_offline_runner_impl(server_asr_engine * engine, int32_t aligner_ctx)
        : engine(engine), aligner_ctx(aligner_ctx) {}

    bool probe(const std::string & audio_path, int64_t & audio_total_ms, server_asr_error & err) override {
        err = {};

        if (!server_asr_audio_container_supported(audio_path)) {
            err = { "unsupported_audio_format", "the audio must be WAV, MP3 or FLAC", "file" };
            return false;
        }

        server_asr_audio audio;
        if (!audio.open(audio_path, 16000, err)) {
            return false;
        }

        audio_total_ms = audio.total_ms();
        if (audio_total_ms <= 0) {
            err = { "invalid_audio", "the audio has no duration", "file" };
            return false;
        }

        return true;
    }

    server_asr_run_status run(
            const server_asr_job_input & input,
            server_asr_run_state & state,
            const std::atomic<bool> & cancel,
            const std::function<bool()> & pause_requested,
            const std::function<void(int64_t, const server_asr_usage &)> & progress,
            server_asr_error & err) override {
        err = {};

        llama_context * ctx = engine->ensure_context(err);
        if (ctx == nullptr) {
            return server_asr_run_status::FAILED;
        }

        mtmd_context * mctx = engine->mctx();
        const llama_vocab * vocab = engine->vocab();
        if (mctx == nullptr || vocab == nullptr) {
            err = { "internal_error", "the ASR engine has no multimodal context", "" };
            return server_asr_run_status::FAILED;
        }

        server_asr_audio audio;
        if (!audio.open(input.audio_path, 16000, err)) {
            return server_asr_run_status::FAILED;
        }

        std::vector<float> energy;
        if (!audio.compute_energy(energy, err)) {
            return server_asr_run_status::FAILED;
        }

        const int32_t n_ctx   = (int32_t) llama_n_ctx(ctx);
        const int32_t n_batch = (int32_t) llama_n_batch(ctx);

        const int32_t context_tokens = server_asr_count_context_tokens(vocab, input.context);
        const int32_t align = input.timestamps ? aligner_ctx : 0;

        const int64_t max_chunk_ms = server_asr_max_chunk_ms(n_ctx, context_tokens, input.timestamps, align);
        if (max_chunk_ms < 1000) {
            err = { "context_too_small", "the context size cannot hold even a one second chunk", "" };
            return server_asr_run_status::FAILED;
        }

        std::vector<server_asr_chunk_range> ranges;
        server_asr_compute_chunk_ranges(audio.total_ms(), n_ctx, context_tokens, input.timestamps, align,
                energy, ranges, err);
        if (!err.empty()) {
            return server_asr_run_status::FAILED;
        }

        for (size_t i = state.next_chunk; i < ranges.size(); i++) {
            if (cancel.load()) {
                return server_asr_run_status::CANCELLED;
            }
            if (pause_requested()) {
                return server_asr_run_status::PAUSED;
            }

            std::vector<float> samples;
            if (!audio.read(ranges[i].start_ms, ranges[i].end_ms, samples, err)) {
                return server_asr_run_status::FAILED;
            }

            // the context is shared with a realtime session
            std::lock_guard<std::mutex> ctx_lock(engine->context_mutex());

            chunk_result chunk;
            chunk_status status = transcribe_chunk(ctx, mctx, vocab, input, samples, n_ctx, n_batch, 0, chunk,
                    cancel, pause_requested, err);

            if (status == chunk_status::OK && chunk.capped) {
                // the cap is a heuristic; one retry with the whole remaining context
                const int32_t hard_cap = n_ctx - (int32_t) chunk.prompt_tokens - 1;
                chunk_result retry;

                status = transcribe_chunk(ctx, mctx, vocab, input, samples, n_ctx, n_batch, hard_cap, retry,
                        cancel, pause_requested, err);

                if (status == chunk_status::OK) {
                    if (retry.capped) {
                        err = { "output_limit_reached", "a chunk hit the generation cap twice", "" };
                        return server_asr_run_status::FAILED;
                    }
                    chunk = std::move(retry);
                }
            }

            if (status == chunk_status::PAUSED) {
                // the partial chunk result is discarded; resume re-runs this chunk
                return server_asr_run_status::PAUSED;
            }
            if (status == chunk_status::CANCELLED) {
                return server_asr_run_status::CANCELLED;
            }
            if (status != chunk_status::OK) {
                return server_asr_run_status::FAILED;
            }

            if (!state.result.text.empty() && !chunk.text.empty() &&
                    needs_space(state.result.text.back(), chunk.text.front())) {
                state.result.text += " ";
            }
            state.result.text += chunk.text;

            if (state.result.language.empty() && !chunk.language.empty()) {
                state.result.language      = chunk.language;
                state.result.language_code = chunk.language_code;
            }

            if (input.timestamps) {
                if (!align_chunk(input, chunk.text, samples, ranges[i].start_ms, state, cancel, err)) {
                    if (cancel.load()) {
                        return server_asr_run_status::CANCELLED;
                    }
                    return server_asr_run_status::FAILED;
                }
            }

            state.usage.input_tokens  += chunk.usage.input_tokens;
            state.usage.output_tokens += chunk.usage.output_tokens;
            state.usage.total_tokens  += chunk.usage.total_tokens;
            state.usage.chunks        += 1;
            state.usage.audio_ms       = ranges[i].end_ms;

            state.next_chunk         = i + 1;
            state.audio_processed_ms = ranges[i].end_ms;

            progress(state.audio_processed_ms, state.usage);
        }

        return server_asr_run_status::DONE;
    }

private:
    server_asr_engine * engine;
    int32_t             aligner_ctx = 0;

    // maps one chunk's text to absolute item spans and adds the aligner usage
    bool align_chunk(
            const server_asr_job_input & input,
            const std::string & text,
            const std::vector<float> & samples,
            int64_t audio_offset_ms,
            server_asr_run_state & state,
            const std::atomic<bool> & cancel,
            server_asr_error & err) const {
        server_asr_aligner * aligner = engine->aligner();
        if (aligner == nullptr) {
            err = { "timestamps_not_available", "timestamps requested but no aligner model is loaded", "timestamps" };
            return false;
        }

        if (text.empty()) {
            return true;
        }

        server_asr_align_request request;
        request.samples   = samples.data();
        request.n_samples = samples.size();
        request.text      = text;
        request.language  = server_asr_normalize_language(input.language);

        server_asr_align_result result;
        if (!aligner->align(request, result, cancel, err)) {
            return false;
        }

        for (auto & item : result.items) {
            item.start_ms += audio_offset_ms;
            item.end_ms   += audio_offset_ms;
            state.result.items.push_back(item);
        }

        state.usage.add(result.usage);

        return true;
    }

    chunk_status transcribe_chunk(
            llama_context * ctx,
            mtmd_context * mctx,
            const llama_vocab * vocab,
            const server_asr_job_input & input,
            const std::vector<float> & samples,
            int32_t n_ctx,
            int32_t n_batch,
            int32_t max_new_override,
            chunk_result & out,
            const std::atomic<bool> & cancel,
            const std::function<bool()> & pause_requested,
            server_asr_error & err) const {
        const std::string forced = server_asr_normalize_language(input.language);

        std::string prompt;
        prompt += "<|im_start|>system\n" + input.context + "<|im_end|>\n";
        prompt += "<|im_start|>user\n" + std::string(mtmd_get_marker(mctx)) + "<|im_end|>\n";
        prompt += "<|im_start|>assistant\n";
        if (!forced.empty()) {
            prompt += "language " + forced + "<asr_text>";
        }

        mtmd_bitmap * bitmap = mtmd_bitmap_init_from_audio(samples.size(), samples.data());
        if (bitmap == nullptr) {
            err = { "internal_error", "failed to build the audio bitmap", "" };
            return chunk_status::FAILED;
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
            err = { "internal_error", "failed to tokenize the ASR prompt", "" };
            return chunk_status::FAILED;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        struct batch_guard {
            llama_batch_ext * p = nullptr;
            ~batch_guard() { llama_batch_ext_free(p); }
        } batch{llama_batch_ext_init(ctx)};

        const size_t n_chunks = mtmd_input_chunks_size(chunks.p);
        llama_pos n_past        = 0;
        int64_t   n_prompt      = 0;
        bool      have_logits   = false;

        for (size_t i = 0; i < n_chunks; i++) {
            if (cancel.load()) {
                return chunk_status::CANCELLED;
            }
            if (pause_requested()) {
                return chunk_status::PAUSED;
            }

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
                                    err = { "internal_error", "the ASR prompt does not fit the batch", "" };
                                    return chunk_status::FAILED;
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
                                err = { "internal_error", "the ASR prompt decode failed", "" };
                                return chunk_status::FAILED;
                            }
                        }
                    }
                    break;

                case MTMD_INPUT_CHUNK_TYPE_AUDIO:
                    {
                        n_prompt += (int64_t) mtmd_input_chunk_get_n_tokens(chunk);

                        if (mtmd_encode_chunk(mctx, chunk) != 0) {
                            err = { "internal_error", "failed to encode the audio", "" };
                            return chunk_status::FAILED;
                        }

                        float * embd = mtmd_get_output_embd(mctx);
                        if (mtmd_helper_decode_image_chunk(mctx, ctx, chunk, embd, n_past, 0, n_batch, &n_past,
                                    nullptr, nullptr) != 0) {
                            err = { "internal_error", "the audio prefill failed", "" };
                            return chunk_status::FAILED;
                        }
                    }
                    break;

                default:
                    break;
            }
        }

        if (!have_logits) {
            err = { "internal_error", "the ASR prompt produced no logits", "" };
            return chunk_status::FAILED;
        }

        const int32_t max_new = max_new_override > 0
            ? max_new_override
            : max_new_tokens_for(n_ctx, n_prompt);

        if (max_new <= 0) {
            err = { "context_too_small", "no room is left for the transcription", "" };
            return chunk_status::FAILED;
        }

        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

        const llama_token eos = llama_vocab_eos(vocab);
        const llama_token eot = llama_vocab_eot(vocab);

        std::string raw;
        int32_t n_generated = 0;
        chunk_status status = chunk_status::OK;

        while (n_generated < max_new) {
            if (cancel.load()) {
                status = chunk_status::CANCELLED;
                break;
            }
            if (pause_requested()) {
                status = chunk_status::PAUSED;
                break;
            }

            const llama_token id = llama_sampler_sample(smpl, ctx, -1);
            if (id == eos || id == eot) {
                break;
            }

            raw += common_token_to_piece(vocab, id, false);
            n_generated++;

            llama_batch_ext_clear(batch.p);

            const int32_t idx = llama_batch_ext_add_token(batch.p, 0, id);
            if (idx < 0) {
                err = { "internal_error", "the generated token does not fit the batch", "" };
                status = chunk_status::FAILED;
                break;
            }

            const llama_pos p = n_past++;
            llama_batch_ext_set_pos(batch.p, idx, &p);
            llama_batch_ext_set_output_logits(batch.p, idx, true);

            if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.p) != 0) {
                err = { "internal_error", "the ASR decode failed", "" };
                status = chunk_status::FAILED;
                break;
            }
        }

        llama_sampler_free(smpl);

        if (status != chunk_status::OK) {
            return status;
        }

        out.prompt_tokens          = n_prompt;
        out.generated              = n_generated;
        out.capped                 = n_generated >= max_new;
        out.usage.input_tokens     = n_prompt;
        out.usage.output_tokens    = n_generated;
        out.usage.total_tokens     = n_prompt + n_generated;

        const server_asr_parse_result parsed = server_asr_parse_output(raw, forced);
        out.text          = parsed.text;
        out.language      = parsed.language;
        out.language_code = parsed.language_code;

        return chunk_status::OK;
    }
};

std::unique_ptr<server_asr_offline_runner> server_asr_make_offline_runner(server_asr_engine * engine, int32_t aligner_ctx) {
    return std::make_unique<server_asr_offline_runner_impl>(engine, aligner_ctx);
}
