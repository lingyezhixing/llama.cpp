#include "server-asr-offline.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>

// 180 s of audio in 60 s chunks, so progress can be checked between them
static constexpr int64_t FAKE_CHUNK_MS  = 60000;
static constexpr int64_t FAKE_TOTAL_MS  = 180000;
static constexpr int     FAKE_CHUNKS    = 3;
// keeps a job in RUNNING long enough for a test to observe or cancel it
static constexpr int     FAKE_STEP_MS   = 150;

// tests widen the window with LLAMA_SERVER_ASR_FAKE_STEP_MS
static int fake_step_ms() {
    if (const char * env = std::getenv("LLAMA_SERVER_ASR_FAKE_STEP_MS"); env != nullptr) {
        const int value = std::atoi(env);
        if (value > 0) {
            return value;
        }
    }
    return FAKE_STEP_MS;
}

bool server_asr_fake_runner::probe(const std::string & audio_path, int64_t & audio_total_ms, server_asr_error & err) {
    std::error_code ec;
    if (audio_path.empty() || !std::filesystem::exists(audio_path, ec)) {
        err = { "invalid_audio", "the audio file is missing", "file" };
        return false;
    }

    audio_total_ms = FAKE_TOTAL_MS;
    return true;
}

server_asr_run_status server_asr_fake_runner::run(
        const server_asr_job_input & input,
        server_asr_run_state & state,
        const std::atomic<bool> & cancel,
        const std::function<bool()> & pause_requested,
        const std::function<void(int64_t, const server_asr_usage &)> & progress,
        server_asr_error & err) {
    while (state.next_chunk < FAKE_CHUNKS) {
        if (cancel.load()) {
            return server_asr_run_status::CANCELLED;
        }
        if (pause_requested()) {
            return server_asr_run_status::PAUSED;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(fake_step_ms()));

        state.next_chunk++;
        state.audio_processed_ms = std::min<int64_t>(FAKE_TOTAL_MS, (int64_t) state.next_chunk * FAKE_CHUNK_MS);

        state.usage.audio_ms      = state.audio_processed_ms;
        state.usage.chunks        = (int64_t) state.next_chunk;
        state.usage.input_tokens += 100;
        state.usage.output_tokens += 10;
        state.usage.total_tokens  = state.usage.input_tokens + state.usage.output_tokens;

        state.result.text          = "fake transcription";
        state.result.language      = input.language.empty() ? "English" : input.language;
        state.result.language_code = input.language.empty() ? "en" : "";

        if (input.timestamps) {
            state.result.items.clear();
            for (int c = 0; c < (int) state.next_chunk; c++) {
                state.result.items.push_back({ "fake", c * FAKE_CHUNK_MS, (c + 1) * FAKE_CHUNK_MS });
            }
        }

        progress(state.audio_processed_ms, state.usage);
    }

    err = {};
    return server_asr_run_status::DONE;
}
