#include "server-asr-engine.h"

#include "server-asr-offline.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

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

    // dedicated context, isolated from the chat slots; created on first use
    std::unique_ptr<llama_context, void (*)(llama_context *)> ctx{nullptr, llama_free};
    std::mutex ctx_mutex;

    std::unique_ptr<server_asr_aligner>    aligner;
    std::unique_ptr<server_asr_fake_runner> fake_runner;    // test mode only
    std::unique_ptr<server_asr_offline_runner> runner;      // the real pipeline
    std::unique_ptr<server_asr_job_manager> jobs;

    std::unique_ptr<server_asr_fake_realtime_runner> fake_realtime; // test mode only
    std::unique_ptr<server_asr_realtime_runner>      realtime;      // Task 11
    std::atomic<bool> realtime_active{false};

    int32_t aligner_ctx = 0; // chunk cap when timestamps are requested
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

        pimpl->aligner_ctx = aligner_config.n_ctx;
    }

    pimpl->caps.timestamps = pimpl->aligner != nullptr;

    // the fake runner keeps the API testable without a model; it is never a
    // documented flag, only this environment variable turns it on
    const char * fake = std::getenv("LLAMA_SERVER_ASR_FAKE_RUNNER");
    if (fake != nullptr && std::strcmp(fake, "1") == 0) {
        pimpl->fake_runner   = std::make_unique<server_asr_fake_runner>();
        pimpl->fake_realtime = std::make_unique<server_asr_fake_realtime_runner>();
        pimpl->jobs          = std::make_unique<server_asr_job_manager>(pimpl->fake_runner.get());
    } else {
        pimpl->runner = server_asr_make_offline_runner(this, pimpl->aligner_ctx);
        pimpl->jobs   = std::make_unique<server_asr_job_manager>(pimpl->runner.get());
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

server_asr_realtime_runner * server_asr_engine::realtime_runner() const {
    if (pimpl->realtime) {
        return pimpl->realtime.get();
    }
    return pimpl->fake_realtime.get();
}

bool server_asr_engine::try_begin_realtime() {
    bool expected = false;
    if (!pimpl->realtime_active.compare_exchange_strong(expected, true)) {
        return false;
    }

    if (pimpl->jobs) {
        pimpl->jobs->set_realtime_active(true);
    }

    return true;
}

void server_asr_engine::end_realtime() {
    pimpl->realtime_active.store(false);

    if (pimpl->jobs) {
        pimpl->jobs->set_realtime_active(false);
    }
}

const llama_vocab * server_asr_engine::vocab() const {
    return pimpl->vocab;
}

llama_model * server_asr_engine::model() const {
    return pimpl->config.model;
}

mtmd_context * server_asr_engine::mctx() const {
    return pimpl->config.mctx;
}

llama_context * server_asr_engine::ensure_context(server_asr_error & err) {
    std::lock_guard<std::mutex> lock(pimpl->ctx_mutex);

    if (pimpl->ctx) {
        return pimpl->ctx.get();
    }

    if (pimpl->config.model == nullptr) {
        err = { "internal_error", "the ASR model is not loaded", "" };
        return nullptr;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) pimpl->config.n_ctx;
    cparams.n_batch         = (uint32_t) std::max(pimpl->config.n_batch, 2048);
    cparams.n_seq_max       = 1;
    cparams.n_threads       = pimpl->config.n_threads;
    cparams.n_threads_batch = pimpl->config.n_threads;
    cparams.embeddings      = false;
    cparams.no_perf         = true;

    llama_context * ctx = llama_init_from_model(pimpl->config.model, cparams);
    if (ctx == nullptr) {
        err = { "internal_error", "failed to create the ASR context", "" };
        return nullptr;
    }

    pimpl->ctx.reset(ctx);

    return pimpl->ctx.get();
}

void server_asr_engine::terminate() {
    // the job manager joins its worker; the aligner, context and model pointers
    // must outlive it
    pimpl->jobs.reset();
    pimpl->runner.reset();
    pimpl->fake_runner.reset();
    pimpl->realtime.reset();
    pimpl->fake_realtime.reset();
    pimpl->realtime_active.store(false);
    pimpl->aligner.reset();

    {
        std::lock_guard<std::mutex> lock(pimpl->ctx_mutex);
        pimpl->ctx.reset();
    }

    pimpl->supported = false;
    pimpl->caps = {};
}
