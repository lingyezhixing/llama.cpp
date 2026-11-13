// Checks that the loader exposes the Qwen3-ASR timestamp classifier.
// Gated: set QWEN3_ASR_MODEL and QWEN3_ASR_ALIGNER_MODEL to GGUF paths.

#include "llama.h"

#include "testing.h"

#include <cmath>
#include <cstdlib>
#include <string>

static llama_model * load_local(const char * path) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0; // loader only, no tensors are uploaded

    return llama_model_load_from_file(path, params);
}

static void test_asr_aligner(testing & t) {
    const char * path = std::getenv("QWEN3_ASR_ALIGNER_MODEL");
    if (path == nullptr || path[0] == '\0') {
        t.skip("QWEN3_ASR_ALIGNER_MODEL is not set");
        return;
    }

    llama_model * model = load_local(path);
    if (!t.assert_true("aligner loads", model != nullptr)) {
        return;
    }

    t.assert_equal("n_cls_out",     (uint32_t) 5000, llama_model_n_cls_out(model));
    t.assert_equal("n_embd_out",    5000,            llama_model_n_embd_out(model));
    t.assert_equal("timestamp token", (uint32_t) 151705, llama_model_timestamp_token_id(model));
    t.assert_true ("segment time",  std::fabs(llama_model_timestamp_segment_time_ms(model) - 80.0f) < 1e-6f);

    llama_model_free(model);
}

static void test_asr_text_model(testing & t) {
    const char * path = std::getenv("QWEN3_ASR_MODEL");
    if (path == nullptr || path[0] == '\0') {
        t.skip("QWEN3_ASR_MODEL is not set");
        return;
    }

    llama_model * model = load_local(path);
    if (!t.assert_true("asr model loads", model != nullptr)) {
        return;
    }

    t.assert_equal("n_cls_out",     (uint32_t) 1, llama_model_n_cls_out(model));
    t.assert_equal("n_embd_out",    llama_model_n_embd(model), llama_model_n_embd_out(model));
    t.assert_equal("timestamp token", (uint32_t) 0, llama_model_timestamp_token_id(model));
    t.assert_true ("segment time",  llama_model_timestamp_segment_time_ms(model) == 0.0f);

    llama_model_free(model);
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

    llama_backend_init();

    t.test("asr_aligner",    test_asr_aligner);
    t.test("asr_text_model", test_asr_text_model);

    llama_backend_free();

    return t.summary();
}
