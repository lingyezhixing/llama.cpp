// Shared types and helpers for the Qwen3-ASR endpoints (/v1/asr/*).

#pragma once

#include "server-common.h"

#include <cstdint>
#include <string>
#include <vector>

enum class server_asr_job_status {
    QUEUED,
    RUNNING,
    PAUSED,
    SUCCEEDED,
    FAILED,
    CANCELLED,
    CANCELLING,
};

struct server_asr_error {
    std::string code;
    std::string message;
    std::string param;

    bool empty() const {
        return code.empty();
    }
};

struct server_asr_item {
    std::string text;
    int64_t     start_ms = 0;
    int64_t     end_ms   = 0;
};

struct server_asr_usage {
    int64_t input_tokens  = 0;
    int64_t output_tokens = 0;
    int64_t total_tokens  = 0;
    int64_t cached_tokens = 0;
    int64_t audio_ms      = 0;
    int64_t chunks        = 0;

    bool    has_aligner              = false;
    int64_t aligner_input_tokens     = 0;
    int64_t aligner_timestamp_tokens = 0;
    int64_t aligner_total_tokens     = 0;
    int64_t aligner_chunks           = 0;

    // realtime only: cumulative decode work, and the tokens actually sent to the client
    int64_t decode_steps            = 0;
    int64_t committed_output_tokens = 0;

    void add(const server_asr_usage & other);
};

struct server_asr_timing {
    std::string created_at;
    std::string started_at;
    std::string finished_at;
    std::string paused_at;

    int64_t     processing_ms = 0;
    int64_t     eta_ms        = 0;
    std::string estimated_end_at;
};

struct server_asr_result {
    std::string                  text;
    std::string                  language;
    std::string                  language_code;
    std::vector<server_asr_item> items;
};

// what the loaded model can do, reported by /props as the "asr" object
struct server_asr_capabilities {
    bool offline    = false;
    bool realtime   = false;
    bool timestamps = false;
    std::vector<std::string> audio_formats;
    std::vector<std::string> languages;
};

struct server_asr_job_view {
    std::string           id;
    server_asr_job_status status = server_asr_job_status::QUEUED;
    std::string           paused_reason;

    server_asr_timing timing;
    int64_t           audio_processed_ms = 0;
    int64_t           audio_total_ms     = 0;

    bool             has_usage = false;
    server_asr_usage usage;

    bool              has_result    = false;
    bool              include_items = false; // set when the job asked for timestamps
    server_asr_result result;

    server_asr_error error;
};

std::string server_asr_status_name(server_asr_job_status status);

// maps a canonical Qwen language name or an ISO 639 code to the canonical name
// gives an empty string for an unknown language, and clears code on failure
std::string server_asr_normalize_language(const std::string & input, std::string * code = nullptr);

struct server_asr_parse_result {
    std::string language;
    std::string language_code;
    std::string text;
};

// strips the "language X<asr_text>" metadata prefix from a raw model output
// a forced language means the prompt suppresses the metadata, so raw is the text
server_asr_parse_result server_asr_parse_output(const std::string & raw, const std::string & forced_language = "");

// sorts by start time and clamps every item into [0, audio_ms]
void server_asr_sort_and_clamp_items(std::vector<server_asr_item> & items, int64_t audio_ms);

// token count of the free-form context field, 0 when empty
int32_t server_asr_count_context_tokens(const llama_vocab * vocab, const std::string & context);

// wall clock, e.g. 2026-11-13T16:00:00Z
std::string server_asr_now_rfc3339();

// wall clock shifted by a duration, used for optimistic end estimates
std::string server_asr_rfc3339_after_ms(int64_t offset_ms);

// the 30 languages Qwen3-ASR supports, in the order /props reports them
std::vector<std::string> server_asr_supported_languages();

json server_asr_error_json(const server_asr_error & err);
json server_asr_capabilities_json(const server_asr_capabilities & caps);
json server_asr_usage_json(const server_asr_usage & usage);
json server_asr_result_json(const server_asr_result & result, bool include_items);
json server_asr_job_json(const server_asr_job_view & view);
