#include "server-asr-engine.h"

#include "server-asr-offline.h"

#include <cstdlib>
#include <cstring>

static bool detect_qwen3_asr(llama_model * model, mtmd_context * mctx) {
    if (model == nullptr || mctx == nullptr) {
        return false;
    }

    char arch[64] = {0};
    if (llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch)) < 0) {
        return false;
    }
    if (std::strcmp(arch, "qwen3vl") != 0) {
        return false;
    }

    // qwen3a audio without a vision tower: the ASR layout we know
    // note: mtmd.h exposes no projector type query, keep this check narrow
    if (!mtmd_support_audio(mctx) || mtmd_support_vision(mctx)) {
        return false;
    }

    return true;
}

struct server_asr_engine::impl {
    server_asr_config     config;
    server_asr_capabilities caps;
    bool supported = false;

    const llama_vocab * vocab = nullptr;

    std::unique_ptr<server_asr_aligner>    aligner;
    std::unique_ptr<server_asr_fake_runner> fake_runner; // test mode only
    std::unique_ptr<server_asr_job_manager> jobs;
};

server_asr_engine::server_asr_engine() : pimpl(std::make_unique<impl>()) {}

server_asr_engine::~server_asr_engine() {
    terminate();
}

bool server_asr_engine::init(const server_asr_config & config, server_asr_error & err) {
    pimpl->config = config;

    if (!detect_qwen3_asr(config.model, config.mctx)) {
        return true; // not an ASR model, the routes report asr_not_supported
    }

    const bool has_aligner_model  = !config.aligner_model_path.empty();
    const bool has_aligner_mmproj = !config.aligner_mmproj_path.empty();

    if (has_aligner_model != has_aligner_mmproj) {
        err = { "internal_error", "--asr-aligner-model and --asr-aligner-mmproj must be provided together", "" };
        return false;
    }
    if (!has_aligner_model && config.aligner_ctx != 0) {
        err = { "internal_error", "--asr-aligner-ctx requires an aligner model", "" };
        return false;
    }

    pimpl->supported = true;
    pimpl->vocab     = config.vocab;

    pimpl->caps.offline       = true;
    pimpl->caps.realtime      = true;
    pimpl->caps.audio_formats = { "wav", "mp3", "flac" };
    pimpl->caps.languages     = server_asr_supported_languages();

    if (has_aligner_model) {
        server_asr_aligner_config aligner_config;

        aligner_config.model_path  = config.aligner_model_path;
        aligner_config.mmproj_path = config.aligner_mmproj_path;
        aligner_config.n_ctx       = config.aligner_ctx > 0 ? config.aligner_ctx : 8192;
        aligner_config.n_batch     = aligner_config.n_ctx;
        aligner_config.n_threads   = config.n_threads;
        aligner_config.device        = config.device;
        aligner_config.mmproj_device = config.mmproj_device;

        pimpl->aligner = std::make_unique<server_asr_aligner>();
        if (!pimpl->aligner->init(aligner_config, err)) {
            pimpl->aligner.reset();
            return false;
        }
    }

    pimpl->caps.timestamps = pimpl->aligner != nullptr;

    // the fake runner keeps the API testable without a model; it is never a
    // documented flag, only this environment variable turns it on
    const char * fake = std::getenv("LLAMA_SERVER_ASR_FAKE_RUNNER");
    if (fake != nullptr && std::strcmp(fake, "1") == 0) {
        pimpl->fake_runner = std::make_unique<server_asr_fake_runner>();
        pimpl->jobs        = std::make_unique<server_asr_job_manager>(pimpl->fake_runner.get());
    }

    return true;
}

bool server_asr_engine::supported() const {
    return pimpl->supported;
}

const server_asr_capabilities & server_asr_engine::capabilities() const {
    return pimpl->caps;
}

server_asr_aligner * server_asr_engine::aligner() const {
    return pimpl->aligner.get();
}

server_asr_job_manager * server_asr_engine::jobs() const {
    return pimpl->jobs.get();
}

const llama_vocab * server_asr_engine::vocab() const {
    return pimpl->vocab;
}

void server_asr_engine::terminate() {
    // the job manager joins its worker; the aligner and model pointers must
    // outlive it
    pimpl->jobs.reset();
    pimpl->fake_runner.reset();
    pimpl->aligner.reset();
    pimpl->supported = false;
    pimpl->caps = {};
}
