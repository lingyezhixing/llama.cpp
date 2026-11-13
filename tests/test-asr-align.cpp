// Tests the Qwen3-ForcedAligner helper: text segmentation, the timestamp repair,
// and (gated) one real forward pass on CPU.

#include "server-asr-align.h"

#include "testing.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void test_tokenize(testing & t) {
    std::vector<std::string> tokens = server_asr_align_tokenize("你好 world", "Chinese");
    t.assert_equal("chinese count", (size_t) 3, tokens.size());
    if (tokens.size() == 3) {
        t.assert_equal("chinese 1", std::string("你"),    tokens[0]);
        t.assert_equal("chinese 2", std::string("好"),    tokens[1]);
        t.assert_equal("chinese 3", std::string("world"), tokens[2]);
    }

    tokens = server_asr_align_tokenize("Hello, world!", "English");
    t.assert_equal("english count", (size_t) 2, tokens.size());
    if (tokens.size() == 2) {
        t.assert_equal("english 1", std::string("Hello"), tokens[0]);
        t.assert_equal("english 2", std::string("world"), tokens[1]);
    }

    // punctuation is dropped, the apostrophe is kept
    tokens = server_asr_align_tokenize("don't stop.", "English");
    t.assert_equal("apostrophe count", (size_t) 2, tokens.size());
    if (tokens.size() == 2) {
        t.assert_equal("apostrophe", std::string("don't"), tokens[0]);
    }

    // Latin runs inside a CJK sentence stay whole
    tokens = server_asr_align_tokenize("你好世界GPT4", "Chinese");
    t.assert_equal("mixed count", (size_t) 5, tokens.size());
    if (tokens.size() == 5) {
        t.assert_equal("mixed latin", std::string("GPT4"), tokens[4]);
    }

    // the nagisa / soynlp approximation: one item per kana or hangul syllable
    tokens = server_asr_align_tokenize("こんにちは", "Japanese");
    t.assert_equal("japanese count", (size_t) 5, tokens.size());

    tokens = server_asr_align_tokenize("안녕 world", "Korean");
    t.assert_equal("korean count", (size_t) 3, tokens.size());

    t.assert_true("empty text",     server_asr_align_tokenize("", "Chinese").empty());
    t.assert_true("punctuation only", server_asr_align_tokenize("...", "English").empty());
    t.assert_true("spaces only",    server_asr_align_tokenize("   ", "English").empty());
}

static void test_fix_timestamps(testing & t) {
    std::vector<int64_t> times = {0, 0, 5, 3, 10, 10};
    server_asr_align_fix_timestamps(times);
    t.assert_equal("count", (size_t) 6, times.size());
    t.assert_true ("non decreasing", std::is_sorted(times.begin(), times.end()));
    t.assert_equal("first", (int64_t) 0, times[0]);
    t.assert_equal("last",  (int64_t) 10, times[5]);

    // a longer anomaly run is interpolated between its neighbours
    std::vector<int64_t> jump = {0, 100, 200, 10, 20, 30, 400};
    server_asr_align_fix_timestamps(jump);
    t.assert_true("jump non decreasing", std::is_sorted(jump.begin(), jump.end()));

    std::vector<int64_t> empty;
    server_asr_align_fix_timestamps(empty);
    t.assert_true("empty stays empty", empty.empty());

    std::vector<int64_t> single = {42};
    server_asr_align_fix_timestamps(single);
    t.assert_equal("single", (int64_t) 42, single[0]);
}

// minimal RIFF reader for the PCM16 mono 16 kHz fixture
static bool load_wav_16k_mono(const char * path, std::vector<float> & samples) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    char riff[4] = {0};
    char wave[4] = {0};
    uint32_t size = 0;

    in.read(riff, 4);
    in.read(reinterpret_cast<char *>(&size), 4);
    in.read(wave, 4);
    if (std::memcmp(riff, "RIFF", 4) != 0 || std::memcmp(wave, "WAVE", 4) != 0) {
        return false;
    }

    uint16_t channels = 0;
    uint32_t rate     = 0;
    uint16_t bits     = 0;
    bool     have_fmt = false;

    while (in) {
        char     id[4] = {0};
        uint32_t len   = 0;

        in.read(id, 4);
        in.read(reinterpret_cast<char *>(&len), 4);
        if (!in) {
            break;
        }

        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint16_t format    = 0;
            uint32_t byte_rate = 0;
            uint16_t block     = 0;

            in.read(reinterpret_cast<char *>(&format), 2);
            in.read(reinterpret_cast<char *>(&channels), 2);
            in.read(reinterpret_cast<char *>(&rate), 4);
            in.read(reinterpret_cast<char *>(&byte_rate), 4);
            in.read(reinterpret_cast<char *>(&block), 2);
            in.read(reinterpret_cast<char *>(&bits), 2);

            if (format != 1 || len > 16) {
                if (format != 1) {
                    return false;
                }
                in.seekg(len - 16, std::ios::cur);
            }
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0) {
            if (!have_fmt || channels != 1 || rate != 16000 || bits != 16) {
                return false;
            }

            std::vector<int16_t> pcm(len / 2);
            in.read(reinterpret_cast<char *>(pcm.data()), (std::streamsize) pcm.size() * 2);
            if (!in) {
                return false;
            }

            samples.resize(pcm.size());
            for (size_t i = 0; i < pcm.size(); i++) {
                samples[i] = (float) pcm[i] / 32768.0f;
            }
            return true;
        } else {
            in.seekg(len + (len % 2), std::ios::cur);
        }
    }

    return false;
}

static void test_align_forward(testing & t) {
    const char * model_path  = std::getenv("QWEN3_ASR_ALIGNER_MODEL");
    const char * mmproj_path = std::getenv("QWEN3_ASR_ALIGNER_MMPROJ");
    const char * audio_path  = std::getenv("QWEN3_ASR_ALIGN_AUDIO");

    if (model_path == nullptr || mmproj_path == nullptr || audio_path == nullptr) {
        t.skip("set QWEN3_ASR_ALIGNER_MODEL, QWEN3_ASR_ALIGNER_MMPROJ and QWEN3_ASR_ALIGN_AUDIO");
        return;
    }

    std::vector<float> samples;
    if (!t.assert_true("the fixture is PCM16 mono 16 kHz", load_wav_16k_mono(audio_path, samples))) {
        return;
    }

    const std::string text = "the quick brown fox jumps over the lazy dog";
    const std::vector<std::string> words = server_asr_align_tokenize(text, "English");

    server_asr_aligner_config config;
    config.model_path  = model_path;
    config.mmproj_path = mmproj_path;
    config.n_ctx       = 8192;
    config.n_batch     = 8192;
    config.n_threads   = 4;
    // no device on purpose: this test must run on the CPU

    server_asr_aligner aligner;
    server_asr_error err;
    if (!t.assert_true("aligner initializes", aligner.init(config, err))) {
        return;
    }
    t.assert_true("aligner available", aligner.available());

    server_asr_align_request request;
    request.samples   = samples.data();
    request.n_samples = samples.size();
    request.text      = text;
    request.language  = "English";

    const std::atomic<bool> cancel{false};
    server_asr_align_result result;

    if (!t.assert_true("aligned", aligner.align(request, result, cancel, err))) {
        return;
    }

    const int64_t audio_ms = (int64_t) (samples.size() * 1000 / 16000);

    t.assert_equal("item count", words.size(), result.items.size());
    if (result.items.size() != words.size()) {
        return;
    }

    bool sorted = true;
    bool inside = true;
    bool named  = true;

    for (size_t i = 0; i < result.items.size(); i++) {
        const auto & item = result.items[i];
        sorted = sorted && (i == 0 || result.items[i - 1].start_ms <= item.start_ms);
        inside = inside && item.start_ms >= 0 && item.end_ms <= audio_ms && item.end_ms >= item.start_ms;
        named  = named && item.text == words[i];
    }

    t.assert_true ("starts are sorted", sorted);
    t.assert_true ("items are inside the audio", inside);
    t.assert_true ("item text matches the input", named);
    t.assert_true ("end after start", result.items.back().end_ms >= result.items.front().start_ms);
    t.assert_true ("aligner usage", result.usage.has_aligner && result.usage.aligner_input_tokens > 0 &&
                                    result.usage.aligner_timestamp_tokens > 0);

    // a second call must reuse the context cleanly
    server_asr_align_result again;
    if (t.assert_true("aligned twice", aligner.align(request, again, cancel, err))) {
        t.assert_equal("repeat count", result.items.size(), again.items.size());
        t.assert_equal("repeat text", result.items.front().text, again.items.front().text);
    }
}

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    if (verbose) {
        t.verbose = std::string(verbose) == "1";
    }

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            std::fprintf(stderr, "%s", text);
        }
    }, nullptr);

    t.test("tokenize",       test_tokenize);
    t.test("fix_timestamps", test_fix_timestamps);
    t.test("align_forward",  test_align_forward);

    return t.summary();
}
