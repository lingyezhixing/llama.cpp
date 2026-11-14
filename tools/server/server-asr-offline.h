// Offline runners for /v1/asr/transcriptions.

#pragma once

#include "server-asr-job.h"

#include <memory>

struct server_asr_engine;


struct server_asr_chunk_range {
    int64_t start_ms = 0;
    int64_t end_ms   = 0;
};

// splits audio into chunks that fit both contexts, cutting at the quietest
// 100 ms window near each target boundary
void server_asr_compute_chunk_ranges(
        int64_t audio_total_ms,
        int32_t n_ctx_seq,
        int32_t context_tokens,
        bool    timestamps,
        int32_t aligner_ctx,
        const std::vector<float> & energy_per_100ms,
        std::vector<server_asr_chunk_range> & out,
        server_asr_error & err);

// the chunk size the two contexts allow, in milliseconds, 0 when too small
int64_t server_asr_max_chunk_ms(int32_t n_ctx_seq, int32_t context_tokens, bool timestamps, int32_t aligner_ctx);


// the real runner: decodes the file, plans chunks, and transcribes them with the
// engine's dedicated ASR context
std::unique_ptr<server_asr_offline_runner> server_asr_make_offline_runner(server_asr_engine * engine, int32_t aligner_ctx);

// deterministic runner for API tests: three fixed chunks and no model
class server_asr_fake_runner : public server_asr_offline_runner {
public:
    bool probe(const std::string & audio_path, int64_t & audio_total_ms, server_asr_error & err) override;

    server_asr_run_status run(
            const server_asr_job_input & input,
            server_asr_run_state & state,
            const std::atomic<bool> & cancel,
            const std::function<bool()> & pause_requested,
            const std::function<void(int64_t, const server_asr_usage &)> & progress,
            server_asr_error & err) override;
};
