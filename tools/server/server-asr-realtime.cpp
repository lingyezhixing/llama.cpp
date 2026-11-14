#include "server-asr-realtime.h"

#include "server-asr-engine.h"
#include "server-common.h"

#include "mtmd-helper.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace {

constexpr int32_t MAX_CONTEXT_TOKENS = 1024;

bool send_json(server_http_context::ws_conn & conn, const json & value) {
    return conn.send(safe_json_to_str(value));
}

void send_error(server_http_context::ws_conn & conn, const std::string & code, const std::string & message) {
    send_json(conn, json {
        { "type",    "error" },
        { "code",    code    },
        { "message", message },
        { "fatal",   true    },
    });
}

bool parse_object(const std::string & text, json & out) {
    try {
        out = json::parse(text);
    } catch (const std::exception &) {
        return false;
    }
    return out.is_object();
}

std::string make_session_id() {
    static std::atomic<uint64_t> counter{0};

    char suffix[32];
    std::snprintf(suffix, sizeof(suffix), "%08" PRIx64, counter.fetch_add(1));

    return std::string("asr_rt_") + suffix;
}

void send_update(server_http_context::ws_conn & conn, const server_asr_realtime_update & update) {
    if (!update.deltas.empty()) {
        send_json(conn, json { {"type", "transcript.delta"}, {"text", update.deltas} });
    }
    if (update.language_changed) {
        send_json(conn, json {
            { "type",          "transcript.language" },
            { "language",      update.language       },
            { "language_code", update.language_code  },
        });
    }
}

bool valid_stability(const std::string & stability) {
    return stability == "low_latency" || stability == "balanced" || stability == "high_accuracy";
}

} // namespace

//
// fake runner
//

bool server_asr_fake_realtime_runner::start(const server_asr_realtime_config & config, server_asr_error & err) {
    this->config = config;
    pushed   = false;
    finished = false;
    audio_ms = 0;
    steps    = 0;
    err      = {};
    return true;
}

bool server_asr_fake_realtime_runner::push(const float * samples_16k, size_t n_samples,
                                           server_asr_realtime_update & update, server_asr_error & err) {
    GGML_UNUSED(samples_16k);
    err = {};

    audio_ms += (int64_t) (n_samples * 1000 / 16000);
    steps++;

    update = {};
    if (!pushed) {
        pushed      = true;
        update.deltas = "hello";
    }

    return true;
}

bool server_asr_fake_realtime_runner::finish(server_asr_realtime_update & update, server_asr_realtime_stats & stats,
                                             server_asr_error & err) {
    err = {};

    if (finished) {
        return true;
    }
    finished = true;

    update = {};
    update.deltas = " world";

    stats.processing_ms          = 1;
    stats.usage.audio_ms         = audio_ms;
    stats.usage.decode_steps     = steps;
    stats.usage.output_tokens    = steps + 2;
    stats.usage.input_tokens     = steps * 10;
    stats.usage.total_tokens     = stats.usage.input_tokens + stats.usage.output_tokens;
    stats.usage.committed_output_tokens = 2;

    return true;
}

void server_asr_fake_realtime_runner::cancel() {
    finished = true;
}

//
// protocol
//

void server_asr_ws_session(const server_http_req & req, server_http_context::ws_conn & conn, server_asr_engine * engine) {
    GGML_UNUSED(req);

    if (engine == nullptr || !engine->supported()) {
        send_error(conn, "asr_not_supported", "the loaded model is not Qwen3-ASR");
        conn.close();
        return;
    }

    server_asr_realtime_runner * runner = engine->realtime_runner();
    if (runner == nullptr) {
        send_error(conn, "asr_not_implemented", "the realtime ASR pipeline is not wired yet");
        conn.close();
        return;
    }

    if (!engine->try_begin_realtime()) {
        send_error(conn, "realtime_busy", "another realtime session is active");
        conn.close();
        return;
    }

    struct realtime_guard {
        server_asr_engine * engine;
        ~realtime_guard() { engine->end_realtime(); }
    } guard{engine};

    server_asr_ws_protocol(conn, runner, engine->capabilities(), engine->vocab());
}

void server_asr_ws_protocol(server_http_context::ws_conn & conn,
                            server_asr_realtime_runner * runner,
                            const server_asr_capabilities & caps,
                            const llama_vocab * vocab) {
    GGML_UNUSED(caps);

    const auto t_start = std::chrono::steady_clock::now();

    std::string message;
    if (conn.read(message) != 1) {
        send_error(conn, "invalid_state", "the first frame must be session.start");
        conn.close();
        return;
    }

    json start;
    if (!parse_object(message, start) || start.value("type", std::string("")) != "session.start") {
        send_error(conn, "invalid_state", "the first frame must be session.start");
        conn.close();
        return;
    }

    server_asr_realtime_config config;
    config.language   = start.value("language", std::string(""));
    config.context    = start.value("context", std::string(""));
    config.stability  = start.value("stability", std::string("balanced"));
    config.sample_rate = start.value("sample_rate", 16000);

    if (!valid_stability(config.stability)) {
        send_error(conn, "invalid_state", "stability must be low_latency, balanced, or high_accuracy");
        conn.close();
        return;
    }

    if (config.sample_rate < 8000 || config.sample_rate > 48000) {
        send_error(conn, "invalid_audio_format", "sample_rate must be between 8000 and 48000");
        conn.close();
        return;
    }

    if (!config.language.empty() && server_asr_normalize_language(config.language).empty()) {
        send_error(conn, "unsupported_language", "the language is not supported by this model");
        conn.close();
        return;
    }

    if (!config.context.empty() &&
            server_asr_count_context_tokens(vocab, config.context) > MAX_CONTEXT_TOKENS) {
        send_error(conn, "context_too_long", "context must be at most 1024 tokens");
        conn.close();
        return;
    }

    server_asr_error err;
    if (!runner->start(config, err)) {
        send_error(conn, err.code.empty() ? "internal_error" : err.code,
                       err.message.empty() ? "the session could not start" : err.message);
        conn.close();
        return;
    }

    const std::string session_id = make_session_id();

    send_json(conn, json {
        { "type",        "session.started" },
        { "session_id",  session_id        },
        { "sample_rate", config.sample_rate },
        { "stability",   config.stability  },
    });

    std::unique_ptr<mtmd_helper_resampler, decltype(&mtmd_helper_resampler_free)> resampler(
            mtmd_helper_resampler_init(config.sample_rate, 16000), &mtmd_helper_resampler_free);
    if (!resampler) {
        send_error(conn, "internal_error", "the resampler could not be created");
        conn.close();
        return;
    }

    int64_t audio_ms = 0;
    bool    cancelled = false;
    bool    finishing = false;

    while (true) {
        const int kind = conn.read(message);
        if (kind == 0) {
            cancelled = true;
            break;
        }

        if (kind == 1) {
            json command;
            if (!parse_object(message, command)) {
                send_error(conn, "invalid_state", "expected a JSON message");
                cancelled = true;
                break;
            }

            const std::string type = command.value("type", std::string(""));
            if (type == "session.finish") {
                finishing = true;
                break;
            }
            if (type == "session.cancel") {
                cancelled = true;
                break;
            }

            send_error(conn, "invalid_state", "expected session.finish or session.cancel");
            cancelled = true;
            break;
        }

        // binary: PCM16 little endian, mono, at the negotiated rate
        if (message.size() % 2 != 0) {
            send_error(conn, "invalid_audio_format", "expected PCM16 mono");
            cancelled = true;
            break;
        }

        std::vector<float> samples(message.size() / 2);
        for (size_t i = 0; i < samples.size(); i++) {
            int16_t value = 0;
            std::memcpy(&value, message.data() + i * 2, sizeof(value));
            samples[i] = (float) value / 32768.0f;
        }

        const float * resampled = nullptr;
        const int64_t n_out = mtmd_helper_resampler_process(resampler.get(), samples.data(), samples.size(), &resampled);
        if (n_out < 0 || (n_out > 0 && resampled == nullptr)) {
            send_error(conn, "internal_error", "the audio could not be resampled");
            cancelled = true;
            break;
        }

        audio_ms += (int64_t) (n_out * 1000 / 16000);

        server_asr_realtime_update update;
        if (!runner->push(resampled, (size_t) n_out, update, err)) {
            send_error(conn, err.code.empty() ? "internal_error" : err.code,
                           err.message.empty() ? "the audio could not be decoded" : err.message);
            cancelled = true;
            break;
        }

        send_update(conn, update);
    }

    const int64_t processing_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start).count();

    if (finishing && !cancelled) {
        server_asr_realtime_update update;
        server_asr_realtime_stats  stats;

        if (!runner->finish(update, stats, err)) {
            send_error(conn, err.code.empty() ? "internal_error" : err.code,
                           err.message.empty() ? "the session could not finish" : err.message);
            conn.close();
            return;
        }

        send_update(conn, update);

        stats.usage.audio_ms = audio_ms > 0 ? audio_ms : stats.usage.audio_ms;

        json finished = json {
            { "type",          "session.finished"  },
            { "audio_ms",      stats.usage.audio_ms },
            { "processing_ms", processing_ms      },
            { "usage",         server_asr_usage_json(stats.usage) },
        };
        if (!server_asr_normalize_language(config.language).empty()) {
            finished["language"]      = server_asr_normalize_language(config.language);
            finished["language_code"] = server_asr_normalize_language(config.language, nullptr);
        }

        send_json(conn, finished);
        conn.close();
        return;
    }

    runner->cancel();

    send_json(conn, json {
        { "type",          "session.cancelled" },
        { "audio_ms",      audio_ms            },
        { "processing_ms", processing_ms       },
    });
    conn.close();
}
//
// stable commit
//

int32_t server_asr_realtime_holdback_tokens(const std::string & stability) {
    if (stability == "low_latency") {
        return 5;
    }
    if (stability == "high_accuracy") {
        return 12;
    }
    return 8; // balanced
}

bool server_asr_stable_commit(server_asr_commit_state & state, size_t & emit_begin, size_t & emit_end) {
    const std::vector<llama_token> & raw  = state.segment_raw_tokens;
    const std::vector<llama_token> & prev = state.prev_segment_raw_tokens;

    size_t lcp = 0;
    const size_t n = std::min(raw.size(), prev.size());
    while (lcp < n && raw[lcp] == prev[lcp]) {
        lcp++;
    }

    const size_t holdback = (size_t) std::max(0, state.holdback_tokens);
    const size_t safe = raw.size() > holdback ? std::min(lcp, raw.size() - holdback) : 0;

    const size_t base = state.segment_base_tokens.size();
    const size_t begin = base + (size_t) state.emitted_in_segment;

    if (safe <= begin) {
        return false;
    }

    emit_begin = begin;
    emit_end   = safe;
    return true;
}

size_t server_asr_utf8_holdback(const std::string & text, std::string & complete) {
    size_t end = text.size();

    // walk back over continuation bytes (10xxxxxx); at most three of them
    size_t back = 0;
    while (back < 3 && end > 0 && ((unsigned char) text[end - 1] & 0xC0) == 0x80) {
        end--;
        back++;
    }

    if (end == 0) {
        complete = text;
        return 0;
    }

    const unsigned char lead = (unsigned char) text[end - 1];
    size_t expected = 1;
    if ((lead & 0x80) == 0x00) {
        expected = 1;
    } else if ((lead & 0xE0) == 0xC0) {
        expected = 2;
    } else if ((lead & 0xF0) == 0xE0) {
        expected = 3;
    } else if ((lead & 0xF8) == 0xF0) {
        expected = 4;
    } else {
        // invalid lead byte, pass it through rather than stalling forever
        complete = text;
        return 0;
    }

    const size_t have = back + 1; // lead plus the continuation bytes already seen
    if (have >= expected) {
        complete = text;
        return 0;
    }

    complete = text.substr(0, end - 1);
    return text.size() - complete.size();
}
