// Streaming audio decode for offline ASR: mono float32 at the model sample rate,
// random access by millisecond, and a 100 ms energy profile.

#pragma once

#include "server-asr-common.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class server_asr_audio {
public:
    server_asr_audio();
    ~server_asr_audio();

    server_asr_audio(const server_asr_audio &) = delete;
    server_asr_audio & operator=(const server_asr_audio &) = delete;

    // WAV, MP3 and FLAC; decodes to mono at sample_rate
    bool open(const std::string & path, int32_t sample_rate, server_asr_error & err);
    void close();

    int64_t total_ms() const;
    int32_t sample_rate() const;

    // RMS per 100 ms window, one streaming pass from the start
    bool compute_energy(std::vector<float> & out, server_asr_error & err);

    // reads [start_ms, end_ms) at the model sample rate
    bool read(int64_t start_ms, int64_t end_ms, std::vector<float> & out, server_asr_error & err);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
