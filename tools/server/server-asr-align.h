// Non-autoregressive Qwen3-ForcedAligner: text segmentation, timestamp classes,
// and the single forward pass that maps text items to milliseconds.

#pragma once

#include "server-asr-common.h"

#include "llama.h"
#include "mtmd-helper.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct server_asr_aligner_config {
    std::string model_path;
    std::string mmproj_path;

    int32_t n_ctx     = 8192;
    int32_t n_batch   = 2048;
    int32_t n_threads = 4;

    ggml_backend_dev_t device        = nullptr; // text model, null means CPU
    ggml_backend_dev_t mmproj_device = nullptr; // audio encoder, null means CPU

    mtmd_helper_init_opt mtmd_opt = mtmd_helper_init_opt_default();
};

struct server_asr_align_request {
    const float * samples   = nullptr;
    size_t        n_samples = 0; // mono float32 at the aligner sample rate
    std::string   text;
    std::string   language;
};

struct server_asr_align_result {
    std::vector<server_asr_item> items;
    server_asr_usage             usage; // aligner counters only
};

class server_asr_aligner {
public:
    server_asr_aligner();
    ~server_asr_aligner();

    bool init(const server_asr_aligner_config & config, server_asr_error & err);
    bool available() const;
    void terminate();

    // one alignment call for at most 180 seconds of audio, thread-safe
    bool align(const server_asr_align_request & request,
               server_asr_align_result & result,
               const std::atomic<bool> & cancel,
               server_asr_error & err);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

// splits text into the items the aligner brackets with timestamps
// Latin runs stay whole, CJK characters become single items, punctuation is dropped
std::vector<std::string> server_asr_align_tokenize(const std::string & text, const std::string & language);

// repairs a timestamp vector: keeps the longest non-decreasing run and fills the
// gaps from their neighbours, then interpolates longer anomalies
void server_asr_align_fix_timestamps(std::vector<int64_t> & times);
