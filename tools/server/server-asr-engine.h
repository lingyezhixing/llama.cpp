// Qwen3-ASR capability detection and forced aligner ownership.

#pragma once

#include "server-asr-align.h"
#include "server-asr-common.h"
#include "server-asr-job.h"
#include "server-asr-realtime.h"

#include <memory>
#include <mutex>
#include <string>

struct server_asr_config {
    llama_model *       model = nullptr; // borrowed, owned by the server
    mtmd_context *      mctx  = nullptr; // borrowed, the server ASR mmproj context
    const llama_vocab * vocab = nullptr;

    int32_t n_ctx     = 0; // per sequence, from n_ctx_slot()
    int32_t n_batch   = 0;
    int32_t n_threads = 4;

    ggml_backend_dev_t device        = nullptr; // null means CPU
    ggml_backend_dev_t mmproj_device = nullptr;

    std::string aligner_model_path;
    std::string aligner_mmproj_path;
    int32_t     aligner_ctx = 0; // 0 = default
};

class server_asr_engine {
public:
    server_asr_engine();
    ~server_asr_engine();

    // detects Qwen3-ASR and loads the aligner when it is configured
    // a model that is not Qwen3-ASR is not an error, supported() stays false
    bool init(const server_asr_config & config, server_asr_error & err);
    bool supported() const;
    const server_asr_capabilities & capabilities() const;

    server_asr_aligner * aligner() const;

    // null when no offline pipeline is wired; the fake runner is used in tests
    server_asr_job_manager * jobs() const;

    const llama_vocab * vocab() const;

    llama_model *  model() const;
    mtmd_context * mctx()  const;

    // creates the dedicated ASR context on first use, null on failure
    llama_context * ensure_context(server_asr_error & err);

    // serializes use of the dedicated context between offline and realtime;
    // realtime priority comes from the realtime flag, not from this mutex
    std::mutex & context_mutex();

    // null until the realtime pipeline exists
    server_asr_realtime_runner * realtime_runner() const;

    // one session at a time; also pauses the offline worker
    bool try_begin_realtime();
    void end_realtime();

    void terminate();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
