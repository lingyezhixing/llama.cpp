// Offline runners for /v1/asr/transcriptions.

#pragma once

#include "server-asr-job.h"

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
