// Async offline ASR jobs: submission, one worker thread, realtime preemption,
// progress and ETA, retention, and cancellation.

#pragma once

#include "server-asr-common.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>

struct server_asr_job_input {
    std::string audio_path; // owned by the job until a terminal state
    bool        timestamps = false;
    std::string language;
    std::string context;
};

enum class server_asr_run_status {
    DONE,
    PAUSED,    // realtime took the ASR context, the chunk is discarded
    CANCELLED,
    FAILED,
};

struct server_asr_run_state {
    size_t            next_chunk         = 0;
    int64_t           audio_processed_ms = 0;
    server_asr_result result;
    server_asr_usage  usage;
};

// the pipeline a job runs, so the API can be tested without a model
class server_asr_offline_runner {
public:
    virtual ~server_asr_offline_runner() = default;

    // duration of the audio, decoded without transcribing it
    virtual bool probe(const std::string & audio_path, int64_t & audio_total_ms, server_asr_error & err) = 0;

    // processes chunks from state.next_chunk onward, calling progress after each
    // one; returns DONE when the audio is covered, PAUSED at a chunk boundary,
    // CANCELLED, or FAILED
    virtual server_asr_run_status run(
            const server_asr_job_input & input,
            server_asr_run_state & state,
            const std::atomic<bool> & cancel,
            const std::function<bool()> & pause_requested,
            const std::function<void(int64_t audio_processed_ms, const server_asr_usage &)> & progress,
            server_asr_error & err) = 0;
};

enum class server_asr_cancel_result {
    NOT_FOUND,
    CANCELLED,  // queued or paused: dropped from the queue
    CANCELLING, // running: the worker stops at the next safe point
    DELETED,    // already terminal: the record is removed
};

class server_asr_job_manager {
public:
    explicit server_asr_job_manager(
            server_asr_offline_runner * runner,
            size_t  queue_capacity = 8,
            int64_t retention_ms   = 24 * 60 * 60 * 1000);
    ~server_asr_job_manager();

    // probes the audio, then queues it; the job owns audio_path from here on
    std::string submit(server_asr_job_input && input, server_asr_error & err);

    bool get(const std::string & id, server_asr_job_view & out, server_asr_error & err) const;

    server_asr_cancel_result cancel(const std::string & id, server_asr_error & err);

    void set_realtime_active(bool value);
    bool realtime_active() const;

    void terminate();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
