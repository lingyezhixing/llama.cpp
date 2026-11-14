#include "server-asr-audio.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// miniaudio pulls in windows.h, which would define min/max macros
#if defined(_WIN32)
#   define WIN32_LEAN_AND_MEAN
#   define NOMINMAX
#endif

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

static constexpr int64_t ENERGY_WINDOW_MS = 100;
static constexpr size_t  READ_BLOCK_FRAMES = 4096;

struct server_asr_audio::impl {
    ma_decoder decoder;
    bool       is_open     = false;
    int32_t    sample_rate = 16000;
    int64_t    total_frames = 0;

    bool read_frames(float * out, size_t n_frames, size_t & n_read, server_asr_error & err) {
        ma_uint64 got = 0;
        const ma_result res = ma_decoder_read_pcm_frames(&decoder, out, n_frames, &got);
        n_read = (size_t) got;

        if (res != MA_SUCCESS && res != MA_AT_END) {
            err = { "invalid_audio", "the audio cannot be decoded", "file" };
            return false;
        }
        if (res == MA_AT_END) {
            n_read = std::min(n_read, (size_t) got);
        }
        return true;
    }
};

server_asr_audio::server_asr_audio() : pimpl(std::make_unique<impl>()) {}

server_asr_audio::~server_asr_audio() {
    close();
}

void server_asr_audio::close() {
    if (pimpl->is_open) {
        ma_decoder_uninit(&pimpl->decoder);
        pimpl->is_open = false;
    }
    pimpl->total_frames = 0;
}

bool server_asr_audio::open(const std::string & path, int32_t sample_rate, server_asr_error & err) {
    err = {};
    close();

    if (path.empty()) {
        err = { "invalid_audio", "no audio path", "file" };
        return false;
    }

    ma_decoder_config config = ma_decoder_config_init(ma_format_f32, 1, (ma_uint32) sample_rate);
    if (ma_decoder_init_file(path.c_str(), &config, &pimpl->decoder) != MA_SUCCESS) {
        err = { "invalid_audio", "the audio cannot be decoded", "file" };
        return false;
    }

    pimpl->is_open      = true;
    pimpl->sample_rate  = sample_rate;

    ma_uint64 frames = 0;
    if (ma_decoder_get_length_in_pcm_frames(&pimpl->decoder, &frames) != MA_SUCCESS) {
        frames = 0;
    }
    pimpl->total_frames = (int64_t) frames;

    return true;
}

int64_t server_asr_audio::total_ms() const {
    if (pimpl->sample_rate <= 0) {
        return 0;
    }
    return pimpl->total_frames * 1000 / pimpl->sample_rate;
}

int32_t server_asr_audio::sample_rate() const {
    return pimpl->sample_rate;
}

bool server_asr_audio::compute_energy(std::vector<float> & out, server_asr_error & err) {
    out.clear();
    err = {};

    if (!pimpl->is_open) {
        err = { "invalid_audio", "the audio is not open", "file" };
        return false;
    }

    if (ma_decoder_seek_to_pcm_frame(&pimpl->decoder, 0) != MA_SUCCESS) {
        err = { "invalid_audio", "the audio cannot be rewound", "file" };
        return false;
    }

    const size_t window_frames = (size_t) (pimpl->sample_rate * ENERGY_WINDOW_MS / 1000);

    std::vector<float> buffer(READ_BLOCK_FRAMES);

    double sum_sq = 0.0;
    size_t in_window = 0;

    for (;;) {
        size_t n_read = 0;
        if (!pimpl->read_frames(buffer.data(), buffer.size(), n_read, err)) {
            return false;
        }
        if (n_read == 0) {
            break;
        }

        for (size_t i = 0; i < n_read; i++) {
            sum_sq += (double) buffer[i] * buffer[i];
            if (++in_window == window_frames) {
                out.push_back((float) std::sqrt(sum_sq / (double) window_frames));
                sum_sq    = 0.0;
                in_window = 0;
            }
        }
    }

    if (in_window > 0) {
        out.push_back((float) std::sqrt(sum_sq / (double) in_window));
    }

    if (ma_decoder_seek_to_pcm_frame(&pimpl->decoder, 0) != MA_SUCCESS) {
        err = { "invalid_audio", "the audio cannot be rewound", "file" };
        return false;
    }

    return true;
}

bool server_asr_audio::read(int64_t start_ms, int64_t end_ms, std::vector<float> & out, server_asr_error & err) {
    out.clear();
    err = {};

    if (!pimpl->is_open) {
        err = { "invalid_audio", "the audio is not open", "file" };
        return false;
    }

    const int64_t start_frame = start_ms * pimpl->sample_rate / 1000;
    const int64_t end_frame   = end_ms   * pimpl->sample_rate / 1000;

    if (start_frame < 0 || end_frame <= start_frame) {
        err = { "invalid_audio", "the chunk range is empty", "" };
        return false;
    }

    if (ma_decoder_seek_to_pcm_frame(&pimpl->decoder, (ma_uint64) start_frame) != MA_SUCCESS) {
        err = { "invalid_audio", "the chunk cannot be seeked", "" };
        return false;
    }

    out.resize((size_t) (end_frame - start_frame));

    size_t filled = 0;
    while (filled < out.size()) {
        size_t n_read = 0;
        if (!pimpl->read_frames(out.data() + filled, out.size() - filled, n_read, err)) {
            out.clear();
            return false;
        }
        if (n_read == 0) {
            break;
        }
        filled += n_read;
    }

    out.resize(filled);

    if (out.empty()) {
        err = { "invalid_audio", "the chunk decoded to nothing", "" };
        return false;
    }

    return true;
}
