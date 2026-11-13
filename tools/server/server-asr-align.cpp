#include "server-asr-align.h"

#include "mtmd.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

//
// UTF-8 helpers
//

// decodes one code point, advances by its length, gives 0xFFFD on a bad byte
static uint32_t utf8_next(const std::string & text, size_t & i) {
    const unsigned char c0 = (unsigned char) text[i];
    size_t len = 1;
    uint32_t cp = c0;

    if (c0 >= 0xF0) {
        len = 4;
        cp  = c0 & 0x07;
    } else if (c0 >= 0xE0) {
        len = 3;
        cp  = c0 & 0x0F;
    } else if (c0 >= 0xC0) {
        len = 2;
        cp  = c0 & 0x1F;
    } else if (c0 >= 0x80) {
        i += 1;
        return 0xFFFD;
    }

    if (i + len > text.size()) {
        i = text.size();
        return 0xFFFD;
    }

    for (size_t k = 1; k < len; k++) {
        const unsigned char ck = (unsigned char) text[i + k];
        if ((ck & 0xC0) != 0x80) {
            i += 1;
            return 0xFFFD;
        }
        cp = (cp << 6) | (ck & 0x3F);
    }

    i += len;
    return cp;
}

static bool cp_is_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0B || cp == 0x0C ||
           cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

static bool cp_is_punct(uint32_t cp) {
    if (cp >= '!' && cp <= '/') return cp != '\'';
    if (cp >= ':' && cp <= '@') return true;
    if (cp >= '[' && cp <= '`') return true;
    if (cp >= '{' && cp <= '~') return true;
    if (cp == 0xAD) return true;
    if (cp >= 0xA1 && cp <= 0xBF) return true;
    if (cp == 0xD7 || cp == 0xF7) return true;
    if (cp >= 0x2000 && cp <= 0x206F) return true; // general punctuation, includes spaces handled above
    if (cp >= 0x2E00 && cp <= 0x2E7F) return true;
    if (cp >= 0x3000 && cp <= 0x303F) return true; // CJK punctuation
    if (cp >= 0xFF00 && cp <= 0xFF0F) return true; // fullwidth punctuation
    if (cp >= 0xFF1A && cp <= 0xFF20) return true;
    if (cp >= 0xFF3B && cp <= 0xFF40) return true;
    if (cp >= 0xFF5B && cp <= 0xFF65) return true;
    return false;
}

static bool cp_is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) ||
           (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) ||
           (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF);
}

static bool cp_is_kana(uint32_t cp) {
    return (cp >= 0x3040 && cp <= 0x309F) || // hiragana
           (cp >= 0x30A0 && cp <= 0x30FF) || // katakana
           (cp >= 0x31F0 && cp <= 0x31FF) || // katakana phonetic extensions
           (cp >= 0xFF66 && cp <= 0xFF9D);   // halfwidth katakana
}

static bool cp_is_hangul(uint32_t cp) {
    return (cp >= 0xAC00 && cp <= 0xD7AF) || // syllables
           (cp >= 0x1100 && cp <= 0x11FF) || // jamo
           (cp >= 0x3130 && cp <= 0x318F);   // compatibility jamo
}

// splits one whitespace-free segment, punctuation already removed
static void split_segment(const std::string & seg, bool split_kana, bool split_hangul, std::vector<std::string> & out) {
    std::string latin;

    auto flush = [&]() {
        if (!latin.empty()) {
            out.push_back(latin);
            latin.clear();
        }
    };

    size_t i = 0;
    while (i < seg.size()) {
        const size_t start = i;
        const uint32_t cp = utf8_next(seg, i);

        const bool single = cp_is_cjk(cp) ||
                            (split_kana   && cp_is_kana(cp)) ||
                            (split_hangul && cp_is_hangul(cp));

        if (single) {
            flush();
            out.push_back(seg.substr(start, i - start));
        } else {
            latin.append(seg, start, i - start);
        }
    }

    flush();
}

std::vector<std::string> server_asr_align_tokenize(const std::string & text, const std::string & language) {
    const std::string lang = server_asr_normalize_language(language);

    // nagisa and soynlp are approximated by one item per kana or hangul syllable
    const bool split_kana   = lang == "Japanese";
    const bool split_hangul = lang == "Korean";

    std::vector<std::string> out;
    std::string segment;

    auto flush = [&]() {
        if (!segment.empty()) {
            split_segment(segment, split_kana, split_hangul, out);
            segment.clear();
        }
    };

    size_t i = 0;
    while (i < text.size()) {
        const size_t start = i;
        const uint32_t cp = utf8_next(text, i);

        if (cp_is_space(cp)) {
            flush();
        } else if (!cp_is_punct(cp)) {
            segment.append(text, start, i - start);
        }
        // punctuation is dropped, the apostrophe is kept by cp_is_punct
    }

    flush();
    return out;
}

void server_asr_align_fix_timestamps(std::vector<int64_t> & times) {
    const size_t n = times.size();
    if (n == 0) {
        return;
    }

    // longest non-decreasing subsequence
    std::vector<int32_t> dp(n, 1);
    std::vector<int32_t> parent(n, -1);

    for (size_t i = 1; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            if (times[j] <= times[i] && dp[j] + 1 > dp[i]) {
                dp[i]     = dp[j] + 1;
                parent[i] = (int32_t) j;
            }
        }
    }

    size_t best = 0;
    for (size_t i = 1; i < n; i++) {
        if (dp[i] > dp[best]) {
            best = i;
        }
    }

    std::vector<char> keep(n, 0);
    for (int32_t idx = (int32_t) best; idx != -1; idx = parent[idx]) {
        keep[idx] = 1;
    }

    std::vector<double> result(times.begin(), times.end());

    size_t i = 0;
    while (i < n) {
        if (keep[i]) {
            i++;
            continue;
        }

        size_t j = i;
        while (j < n && !keep[j]) {
            j++;
        }

        const size_t anomaly = j - i;

        bool   has_left  = false;
        bool   has_right = false;
        double left      = 0.0;
        double right     = 0.0;

        for (size_t k = i; k > 0; k--) {
            if (keep[k - 1]) {
                left     = result[k - 1];
                has_left = true;
                break;
            }
        }
        for (size_t k = j; k < n; k++) {
            if (keep[k]) {
                right     = result[k];
                has_right = true;
                break;
            }
        }

        if (anomaly <= 2) {
            for (size_t k = i; k < j; k++) {
                if (!has_left) {
                    result[k] = right;
                } else if (!has_right) {
                    result[k] = left;
                } else {
                    result[k] = (k - (i - 1)) <= (j - k) ? left : right;
                }
            }
        } else if (has_left && has_right) {
            const double step = (right - left) / (double) (anomaly + 1);
            for (size_t k = i; k < j; k++) {
                result[k] = left + step * (double) (k - i + 1);
            }
        } else if (has_left) {
            for (size_t k = i; k < j; k++) {
                result[k] = left;
            }
        } else if (has_right) {
            for (size_t k = i; k < j; k++) {
                result[k] = right;
            }
        }

        i = j;
    }

    for (size_t k = 0; k < n; k++) {
        times[k] = (int64_t) result[k];
    }
}

//
// aligner
//

struct server_asr_aligner::impl {
    server_asr_aligner_config config;

    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
    mtmd_context *  mctx  = nullptr;

    const llama_vocab * vocab = nullptr;

    uint32_t timestamp_token_id     = 0;
    float    timestamp_segment_ms   = 0.0f;
    int32_t  n_cls_out              = 0;
    int32_t  sample_rate            = 16000;
    int32_t  n_pos_per_embd         = 1;

    std::vector<ggml_backend_dev_t> devices; // keeps the device list alive

    std::mutex mutex; // one alignment at a time, the context is shared

    bool init(const server_asr_aligner_config & cfg, server_asr_error & err);
    void terminate();

    bool align(const server_asr_align_request & request, server_asr_align_result & result,
               const std::atomic<bool> & cancel, server_asr_error & err);
};

bool server_asr_aligner::impl::init(const server_asr_aligner_config & cfg, server_asr_error & err) {
    config = cfg;

    llama_model_params mparams = llama_model_default_params();
    if (config.device != nullptr) {
        devices = { config.device, nullptr };
        mparams.devices      = devices.data();
        mparams.n_gpu_layers = 99;
    } else {
        mparams.n_gpu_layers = 0;
    }

    model = llama_model_load_from_file(config.model_path.c_str(), mparams);
    if (model == nullptr) {
        err = { "internal_error", "failed to load the aligner model", "" };
        return false;
    }

    timestamp_token_id   = llama_model_timestamp_token_id(model);
    timestamp_segment_ms = llama_model_timestamp_segment_time_ms(model);
    n_cls_out            = (int32_t) llama_model_n_cls_out(model);
    vocab                = llama_model_get_vocab(model);

    if (timestamp_token_id == 0 || timestamp_segment_ms <= 0.0f || n_cls_out <= 1) {
        err = { "internal_error", "the aligner model is missing its timestamp metadata", "" };
        return false;
    }

    // the timestamp marker must be one special token with the advertised id
    {
        const llama_tokens tokens = common_tokenize(vocab, "<timestamp>", false, true);
        if (tokens.size() != 1 || (uint32_t) tokens[0] != timestamp_token_id) {
            err = { "internal_error", "the aligner tokenizer has no single <timestamp> token", "" };
            return false;
        }
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) config.n_ctx;
    // one batch carries the whole alignment, audio tokens and text together
    cparams.n_batch         = (uint32_t) std::max(config.n_batch, config.n_ctx);
    cparams.n_threads       = config.n_threads;
    cparams.n_threads_batch = config.n_threads;
    cparams.embeddings      = true;
    cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    cparams.no_perf         = true;

    ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        err = { "internal_error", "failed to create the aligner context", "" };
        return false;
    }

    mtmd_context_params mparams_mmproj = mtmd_context_params_default();
    mparams_mmproj.use_gpu       = config.mmproj_device != nullptr;
    mparams_mmproj.device        = config.mmproj_device;
    mparams_mmproj.n_threads     = config.n_threads;
    mparams_mmproj.print_timings = false;
    mparams_mmproj.warmup        = false;

    mctx = mtmd_init_from_file(config.mmproj_path.c_str(), model, mparams_mmproj);
    if (mctx == nullptr) {
        err = { "internal_error", "failed to load the aligner mmproj", "" };
        return false;
    }

    if (!mtmd_support_audio(mctx)) {
        err = { "internal_error", "the aligner mmproj has no audio encoder", "" };
        return false;
    }

    sample_rate    = mtmd_get_audio_sample_rate(mctx);
    n_pos_per_embd = mtmd_decode_use_mrope(mctx) ? 4 : 1;

    if (sample_rate <= 0) {
        err = { "internal_error", "the aligner mmproj reports no audio sample rate", "" };
        return false;
    }

    return true;
}

void server_asr_aligner::impl::terminate() {
    if (mctx != nullptr) {
        mtmd_free(mctx);
        mctx = nullptr;
    }
    if (ctx != nullptr) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model != nullptr) {
        llama_model_free(model);
        model = nullptr;
    }
    vocab = nullptr;
}

bool server_asr_aligner::impl::align(const server_asr_align_request & request, server_asr_align_result & result,
                                     const std::atomic<bool> & cancel, server_asr_error & err) {
    result = {};
    err    = {};

    if (request.samples == nullptr || request.n_samples == 0) {
        err = { "invalid_audio", "no audio samples to align", "" };
        return false;
    }

    const int64_t audio_ms = (int64_t) (request.n_samples * 1000 / (size_t) sample_rate);
    // the official aligner limit, enforced here because the chunker must respect it too
    if (audio_ms > 180000) {
        err = { "aligner_audio_too_long", "the aligner accepts at most 180 seconds of audio", "" };
        return false;
    }

    const std::vector<std::string> text_items = server_asr_align_tokenize(request.text, request.language);
    if (text_items.empty()) {
        return true; // nothing to align
    }

    // the reference prompt: one audio marker, then each item followed by two timestamps
    std::string text = mtmd_get_marker(mctx);
    for (const auto & item : text_items) {
        text += item;
        text += "<timestamp><timestamp>";
    }

    std::lock_guard<std::mutex> lock(mutex);

    llama_memory_clear(llama_get_memory(ctx), true);

    mtmd_bitmap * bitmap = mtmd_bitmap_init_from_audio(request.n_samples, request.samples);
    if (bitmap == nullptr) {
        err = { "internal_error", "failed to build the aligner audio bitmap", "" };
        return false;
    }

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text mtext = { text.c_str(), text.size(), false, true };

    const mtmd_bitmap * bitmaps[] = { bitmap };
    const int32_t tokenize_ret = mtmd_tokenize(mctx, chunks, &mtext, bitmaps, 1);
    mtmd_bitmap_free(bitmap);

    if (tokenize_ret != 0) {
        mtmd_input_chunks_free(chunks);
        err = { "internal_error", "failed to tokenize the aligner input", "" };
        return false;
    }

    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    std::vector<int32_t> timestamp_indices;
    int64_t n_tokens = 0;
    llama_pos pos    = 0;

    bool ok = true;

    for (size_t i = 0; ok && i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);

        switch (mtmd_input_chunk_get_type(chunk)) {
            case MTMD_INPUT_CHUNK_TYPE_TEXT:
                {
                    size_t n_chunk_tokens = 0;
                    const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_chunk_tokens);

                    for (size_t k = 0; k < n_chunk_tokens; k++) {
                        const int32_t idx = llama_batch_ext_add_token(batch, 0, tokens[k]);
                        if (idx < 0) {
                            ok = false;
                            break;
                        }

                        const llama_pos p = pos++;
                        llama_batch_ext_set_pos(batch, idx, &p);

                        if ((uint32_t) tokens[k] == timestamp_token_id) {
                            llama_batch_ext_set_output_embd(batch, idx, true);
                            timestamp_indices.push_back(idx);
                        }
                        n_tokens++;
                    }
                }
                break;

            case MTMD_INPUT_CHUNK_TYPE_AUDIO:
                {
                    if (cancel.load()) {
                        err = { "cancelled", "the alignment was cancelled", "" };
                        ok  = false;
                        break;
                    }

                    if (mtmd_encode_chunk(mctx, chunk) != 0) {
                        err = { "internal_error", "failed to encode the aligner audio", "" };
                        ok  = false;
                        break;
                    }

                    const float * embd   = mtmd_get_output_embd(mctx);
                    const int32_t n_rows = (int32_t) mtmd_input_chunk_get_n_tokens(chunk);
                    const int32_t n_embd = llama_model_n_embd_inp(model);

                    for (int32_t r = 0; ok && r < n_rows; r++) {
                        const llama_embd e = { embd + (size_t) r * n_embd, 1, (size_t) n_embd };
                        const int32_t idx = llama_batch_ext_add_embd(batch, 0, e);
                        if (idx < 0) {
                            ok = false;
                            break;
                        }

                        llama_pos p[4] = { pos, pos, pos, pos };
                        llama_batch_ext_set_pos(batch, idx, p);
                        pos++;
                        n_tokens++;
                    }
                }
                break;

            default:
                // no other chunk type is produced for audio-only alignment
                break;
        }
    }

    if (ok) {
        const int32_t process_ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        if (process_ret != 0) {
            err = { "internal_error", "the aligner forward pass failed", "" };
            ok  = false;
        }
    }

    std::vector<int64_t> times;
    if (ok) {
        times.reserve(timestamp_indices.size());

        for (const int32_t idx : timestamp_indices) {
            const float * row = llama_get_embeddings_ith(ctx, idx);
            if (row == nullptr) {
                ok = false;
                break;
            }

            int32_t best = 0;
            for (int32_t c = 1; c < n_cls_out; c++) {
                if (row[c] > row[best]) {
                    best = c;
                }
            }

            times.push_back((int64_t) ((float) best * timestamp_segment_ms));
        }
    }

    llama_batch_ext_free(batch);
    mtmd_input_chunks_free(chunks);

    if (!ok) {
        if (err.empty()) {
            err = { "internal_error", "the aligner prompt does not fit its context", "" };
        }
        return false;
    }

    if (times.size() < 2 * text_items.size()) {
        err = { "internal_error", "the aligner returned fewer timestamps than items", "" };
        return false;
    }

    server_asr_align_fix_timestamps(times);

    result.items.reserve(text_items.size());
    for (size_t k = 0; k < text_items.size(); k++) {
        server_asr_item item;
        item.text     = text_items[k];
        item.start_ms = times[2 * k];
        item.end_ms   = times[2 * k + 1];
        result.items.push_back(item);
    }

    server_asr_sort_and_clamp_items(result.items, audio_ms);

    result.usage.has_aligner              = true;
    result.usage.aligner_input_tokens     = n_tokens;
    result.usage.aligner_timestamp_tokens = (int64_t) timestamp_indices.size();
    result.usage.aligner_total_tokens     = n_tokens + (int64_t) timestamp_indices.size();
    result.usage.aligner_chunks           = 1;

    return true;
}

server_asr_aligner::server_asr_aligner() : pimpl(std::make_unique<impl>()) {}

server_asr_aligner::~server_asr_aligner() {
    pimpl->terminate();
}

bool server_asr_aligner::init(const server_asr_aligner_config & config, server_asr_error & err) {
    return pimpl->init(config, err);
}

bool server_asr_aligner::available() const {
    return pimpl->mctx != nullptr && pimpl->ctx != nullptr && pimpl->model != nullptr;
}

void server_asr_aligner::terminate() {
    pimpl->terminate();
}

bool server_asr_aligner::align(const server_asr_align_request & request, server_asr_align_result & result,
                               const std::atomic<bool> & cancel, server_asr_error & err) {
    if (!available()) {
        err = { "timestamps_not_available", "no aligner model is loaded", "timestamps" };
        return false;
    }

    return pimpl->align(request, result, cancel, err);
}
