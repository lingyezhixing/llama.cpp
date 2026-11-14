// Realtime ASR over WebSocket: the per-connection state machine and the runner
// interface it drives.

#pragma once

#include "server-asr-common.h"
#include "server-http.h"

#include <atomic>
#include <cstdint>
#include <string>

struct server_asr_engine;

struct server_asr_realtime_config {
    std::string language;
    std::string context;
    std::string stability   = "balanced";
    int32_t     sample_rate = 16000;
};

struct server_asr_realtime_update {
    std::string deltas;
    bool        language_changed = false;
    std::string language;
    std::string language_code;
};

struct server_asr_realtime_stats {
    int64_t          processing_ms = 0;
    server_asr_usage usage;
};

class server_asr_realtime_runner {
public:
    virtual ~server_asr_realtime_runner() = default;

    virtual bool start(const server_asr_realtime_config & config, server_asr_error & err) = 0;

    // audio at the model rate; deltas must be append-only
    virtual bool push(const float * samples_16k, size_t n_samples, server_asr_realtime_update & update,
                      server_asr_error & err) = 0;

    virtual bool finish(server_asr_realtime_update & update, server_asr_realtime_stats & stats,
                        server_asr_error & err) = 0;

    virtual void cancel() = 0;
};

// stability preset to holdback tokens; unknown values fall back to balanced
int32_t server_asr_realtime_holdback_tokens(const std::string & stability);

// state of the append-only commit pointer for one segment
struct server_asr_commit_state {
    std::vector<llama_token> segment_base_tokens;      // carried into this segment
    std::vector<llama_token> prompt_prefix_tokens;     // fixed prefix of the next decode
    std::vector<llama_token> segment_raw_tokens;       // prefix + generated text
    std::vector<llama_token> prev_segment_raw_tokens;  // raw tokens of the previous decode

    int64_t committed_tokens   = 0; // global count already sent to the client
    int64_t emitted_in_segment = 0;
    int32_t holdback_tokens    = 8;
};

// token span of segment_raw_tokens that may be emitted now, [begin, end)
// false when nothing new is safe, which is the common case early in a segment
bool server_asr_stable_commit(server_asr_commit_state & state, size_t & emit_begin, size_t & emit_end);

// segment length before a context reset, by stability preset
int64_t server_asr_realtime_segment_max_ms(const std::string & stability);

// finds a quiet run at least min_silence_ms long inside the search range
// cut_ms is the end of that run, i.e. where the next segment starts
bool server_asr_find_silence(const float * samples, size_t n_samples, int sample_rate,
                             int64_t search_start_ms, int64_t search_end_ms,
                             int64_t min_silence_ms, int64_t & cut_ms);

// the lowest-energy 100 ms window inside the range, used when no silence exists
int64_t server_asr_lowest_energy_window(const float * samples, size_t n_samples, int sample_rate,
                                        int64_t search_start_ms, int64_t search_end_ms);

// drops the duplication at a forced cut: the longest suffix of committed that
// the continuation repeats at its start
void server_asr_boundary_dedupe(const std::vector<llama_token> & committed,
                                std::vector<llama_token> & continuation);

// splits off trailing bytes that do not form a complete UTF-8 code point
// returns the number of held bytes; they must be sent with the next span
size_t server_asr_utf8_holdback(const std::string & text, std::string & complete);

// the re-decode runner: accumulated audio, stable commit, append-only deltas
std::unique_ptr<server_asr_realtime_runner> server_asr_make_realtime_runner(server_asr_engine * engine);

// deterministic runner for protocol tests: "hello" on the first push, " world" on finish
class server_asr_fake_realtime_runner : public server_asr_realtime_runner {
public:
    bool start(const server_asr_realtime_config & config, server_asr_error & err) override;
    bool push(const float * samples_16k, size_t n_samples, server_asr_realtime_update & update,
              server_asr_error & err) override;
    bool finish(server_asr_realtime_update & update, server_asr_realtime_stats & stats,
                server_asr_error & err) override;
    void cancel() override;

private:
    server_asr_realtime_config config;
    bool    pushed   = false;
    bool    finished = false;
    int64_t audio_ms = 0;
    int64_t steps    = 0;
};

// the state machine alone: no engine, no global session flag, so it can be
// driven by a mock connection in tests
void server_asr_ws_protocol(server_http_context::ws_conn & conn,
                            server_asr_realtime_runner * runner,
                            const server_asr_capabilities & caps,
                            const llama_vocab * vocab);

// one session; returns when the connection ends
void server_asr_ws_session(const server_http_req & req, server_http_context::ws_conn & conn, server_asr_engine * engine);
