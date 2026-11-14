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
