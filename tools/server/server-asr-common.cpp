#include "server-asr-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>

// the languages Qwen3-ASR supports, in the order /props reports them
struct server_asr_language {
    const char * name;
    const char * code;
};

static const server_asr_language SERVER_ASR_LANGUAGES[] = {
    { "Chinese",    "zh"  },
    { "English",    "en"  },
    { "Cantonese",  "yue" },
    { "Arabic",     "ar"  },
    { "German",     "de"  },
    { "French",     "fr"  },
    { "Spanish",    "es"  },
    { "Portuguese", "pt"  },
    { "Indonesian", "id"  },
    { "Italian",    "it"  },
    { "Korean",     "ko"  },
    { "Russian",    "ru"  },
    { "Thai",       "th"  },
    { "Vietnamese", "vi"  },
    { "Japanese",   "ja"  },
    { "Turkish",    "tr"  },
    { "Hindi",      "hi"  },
    { "Malay",      "ms"  },
    { "Dutch",      "nl"  },
    { "Swedish",    "sv"  },
    { "Danish",     "da"  },
    { "Finnish",    "fi"  },
    { "Polish",     "pl"  },
    { "Czech",      "cs"  },
    { "Filipino",   "fil" },
    { "Persian",    "fa"  },
    { "Greek",      "el"  },
    { "Romanian",   "ro"  },
    { "Hungarian",  "hu"  },
    { "Macedonian", "mk"  },
};

static std::string lower_ascii(std::string input) {
    for (char & c : input) {
        if (c >= 'A' && c <= 'Z') {
            c = (char) (c - 'A' + 'a');
        }
    }
    return input;
}

static std::string trim_copy(const std::string & input) {
    size_t begin = 0;
    size_t end   = input.size();

    while (begin < end && (unsigned char) input[begin] <= ' ') {
        begin++;
    }
    while (end > begin && (unsigned char) input[end - 1] <= ' ') {
        end--;
    }

    return input.substr(begin, end - begin);
}

void server_asr_usage::add(const server_asr_usage & other) {
    input_tokens  += other.input_tokens;
    output_tokens += other.output_tokens;
    total_tokens  += other.total_tokens;
    cached_tokens += other.cached_tokens;
    audio_ms      += other.audio_ms;
    chunks        += other.chunks;

    has_aligner              = has_aligner || other.has_aligner;
    aligner_input_tokens     += other.aligner_input_tokens;
    aligner_timestamp_tokens += other.aligner_timestamp_tokens;
    aligner_total_tokens     += other.aligner_total_tokens;
    aligner_chunks           += other.aligner_chunks;

    decode_steps            += other.decode_steps;
    committed_output_tokens += other.committed_output_tokens;
}

std::string server_asr_status_name(server_asr_job_status status) {
    switch (status) {
        case server_asr_job_status::QUEUED:     return "queued";
        case server_asr_job_status::RUNNING:    return "running";
        case server_asr_job_status::PAUSED:     return "paused";
        case server_asr_job_status::SUCCEEDED:  return "succeeded";
        case server_asr_job_status::FAILED:     return "failed";
        case server_asr_job_status::CANCELLED:  return "cancelled";
        case server_asr_job_status::CANCELLING: return "cancelling";
    }

    return "queued";
}

std::string server_asr_normalize_language(const std::string & input, std::string * code) {
    if (code) {
        code->clear();
    }

    const std::string key = lower_ascii(trim_copy(input));
    if (key.empty()) {
        return "";
    }

    // the official ISO code of Filipino is "fil", "tl" is the common alternative
    const std::string lookup = key == "tl" ? "fil" : key;

    for (const auto & lang : SERVER_ASR_LANGUAGES) {
        if (lookup == lang.code || lookup == lower_ascii(lang.name)) {
            if (code) {
                *code = lang.code;
            }
            return lang.name;
        }
    }

    return "";
}

server_asr_parse_result server_asr_parse_output(const std::string & raw, const std::string & forced_language) {
    server_asr_parse_result out;

    if (!forced_language.empty()) {
        // a forced language suppresses the metadata, so the whole output is text
        out.language = server_asr_normalize_language(forced_language, &out.language_code);
        if (out.language.empty()) {
            out.language = forced_language;
        }
        out.text = raw;
        return out;
    }

    static const std::string marker = "<asr_text>";
    static const std::string prefix = "language ";

    const std::string head = trim_copy(raw);
    const size_t pos = head.find(marker);

    if (pos == std::string::npos) {
        // no metadata boundary: the whole output is text, even when it starts
        // with "language " (a truncated model output must not be dropped)
        out.text = raw;
        return out;
    }

    const std::string metadata = head.substr(0, pos);
    if (metadata.empty()) {
        out.text = head.substr(pos + marker.size());
        return out;
    }

    if (metadata.size() < prefix.size() || lower_ascii(metadata.substr(0, prefix.size())) != prefix) {
        // the marker is part of the text, not a metadata boundary
        out.text = raw;
        return out;
    }

    const std::string language = trim_copy(metadata.substr(prefix.size()));
    out.language = server_asr_normalize_language(language, &out.language_code);
    out.text     = head.substr(pos + marker.size());
    return out;
}

void server_asr_sort_and_clamp_items(std::vector<server_asr_item> & items, int64_t audio_ms) {
    if (audio_ms < 0) {
        audio_ms = 0;
    }

    for (auto & item : items) {
        item.start_ms = std::min(std::max(item.start_ms, (int64_t) 0), audio_ms);
        item.end_ms   = std::min(std::max(item.end_ms,   (int64_t) 0), audio_ms);
        if (item.end_ms < item.start_ms) {
            item.end_ms = item.start_ms;
        }
    }

    std::stable_sort(items.begin(), items.end(), [](const server_asr_item & a, const server_asr_item & b) {
        return a.start_ms < b.start_ms;
    });
}

int32_t server_asr_count_context_tokens(const llama_vocab * vocab, const std::string & context) {
    if (context.empty()) {
        return 0;
    }

    if (vocab == nullptr) {
        // no vocab yet: keep the chunk budget on the safe side
        return (int32_t) ((context.size() + 2) / 3);
    }

    const std::vector<llama_token> tokens = common_tokenize(vocab, context, false, true);
    return (int32_t) tokens.size();
}

static std::string format_rfc3339(std::time_t t) {
    std::tm tm_utc = {};
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
            tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday,
            tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec);
    return buf;
}

std::string server_asr_rfc3339_after_ms(int64_t offset_ms) {
    return format_rfc3339(std::time(nullptr) + (std::time_t) (offset_ms / 1000));
}

std::string server_asr_now_rfc3339() {
    return format_rfc3339(std::time(nullptr));
}

std::vector<std::string> server_asr_supported_languages() {
    std::vector<std::string> out;
    for (const auto & lang : SERVER_ASR_LANGUAGES) {
        out.emplace_back(lang.name);
    }
    return out;
}

json server_asr_capabilities_json(const server_asr_capabilities & caps) {
    json out = json::object();

    out["offline"]       = caps.offline;
    out["realtime"]      = caps.realtime;
    out["timestamps"]    = caps.timestamps;
    out["audio_formats"] = caps.audio_formats;
    out["languages"]     = caps.languages;

    return out;
}

json server_asr_error_json(const server_asr_error & err) {
    json out = json::object();

    out["code"]    = err.code;
    out["message"] = err.message;
    if (!err.param.empty()) {
        out["param"] = err.param;
    }

    return out;
}

json server_asr_usage_json(const server_asr_usage & usage) {
    json out = json::object();

    out["input_tokens"]  = usage.input_tokens;
    out["output_tokens"] = usage.output_tokens;
    out["total_tokens"]  = usage.total_tokens;
    out["cached_tokens"] = usage.cached_tokens;
    out["audio_ms"]      = usage.audio_ms;
    out["chunks"]        = usage.chunks;

    if (usage.has_aligner) {
        json aligner = json::object();
        aligner["input_tokens"]     = usage.aligner_input_tokens;
        aligner["timestamp_tokens"] = usage.aligner_timestamp_tokens;
        aligner["total_tokens"]     = usage.aligner_total_tokens;
        aligner["chunks"]           = usage.aligner_chunks;
        out["aligner"] = aligner;
    }

    // realtime counters, offline jobs leave them at zero
    if (usage.decode_steps > 0) {
        out["decode_steps"]            = usage.decode_steps;
        out["committed_output_tokens"] = usage.committed_output_tokens;
    }

    return out;
}

json server_asr_result_json(const server_asr_result & result, bool include_items) {
    json out = json::object();

    out["text"]          = result.text;
    out["language"]      = result.language;
    out["language_code"] = result.language_code;

    if (include_items) {
        json items = json::array();
        for (const auto & item : result.items) {
            items.push_back({
                {"text",     item.text},
                {"start_ms", item.start_ms},
                {"end_ms",   item.end_ms},
            });
        }
        out["items"] = items;
    }

    return out;
}

json server_asr_job_json(const server_asr_job_view & view) {
    json out = json::object();

    out["id"]     = view.id;
    out["object"] = "asr.transcription";
    out["status"] = server_asr_status_name(view.status);

    if (view.status == server_asr_job_status::PAUSED && !view.paused_reason.empty()) {
        out["paused_reason"] = view.paused_reason;
    }

    json timing = json::object();

    if (!view.timing.created_at.empty()) {
        timing["created_at"] = view.timing.created_at;
    }
    if (!view.timing.started_at.empty()) {
        timing["started_at"]    = view.timing.started_at;
        timing["processing_ms"] = view.timing.processing_ms;
    }
    if (!view.timing.finished_at.empty()) {
        timing["finished_at"] = view.timing.finished_at;
    }
    if (!view.timing.paused_at.empty()) {
        timing["paused_at"] = view.timing.paused_at;
    }

    // the eta is only meaningful while running
    if (view.status == server_asr_job_status::RUNNING && view.timing.eta_ms > 0 && !view.timing.estimated_end_at.empty()) {
        timing["eta_ms"]           = view.timing.eta_ms;
        timing["estimated_end_at"] = view.timing.estimated_end_at;
    }

    out["timing"] = timing;

    double percent = 0.0;
    if (view.audio_total_ms > 0) {
        percent = 100.0 * (double) view.audio_processed_ms / (double) view.audio_total_ms;
        percent = std::round(percent * 10.0) / 10.0;
        percent = std::min(std::max(percent, 0.0), 100.0);
    }

    json progress = json::object();
    progress["audio_processed_ms"] = view.audio_processed_ms;
    progress["audio_total_ms"]     = view.audio_total_ms;
    progress["percent"]            = percent;
    out["progress"] = progress;

    if (view.has_usage) {
        out["usage"] = server_asr_usage_json(view.usage);
    }
    if (view.has_result) {
        out["result"] = server_asr_result_json(view.result, view.include_items);
    }
    if (!view.error.empty()) {
        out["error"] = server_asr_error_json(view.error);
    }

    return out;
}
