#include "server-asr-api.h"

#include "server-asr-engine.h"
#include "server-asr-realtime.h"
#include "server-asr-upload.h"
#include "server-common.h"

#include <cstdlib>
#include <filesystem>

static server_http_res_ptr make_error_res(int status, const server_asr_error & err) {
    auto res = std::make_unique<server_http_res>();

    res->status = status;
    res->data   = safe_json_to_str(json { {"error", server_asr_error_json(err)} });

    return res;
}

static server_http_res_ptr make_job_res(int status, const server_asr_job_view & view) {
    auto res = std::make_unique<server_http_res>();

    res->status = status;
    res->data   = safe_json_to_str(server_asr_job_json(view));

    return res;
}

static server_asr_error err_not_supported() {
    return { "asr_not_supported", "the loaded model is not Qwen3-ASR", "" };
}

static server_asr_error err_not_implemented() {
    return { "asr_not_implemented", "the offline ASR pipeline is not wired yet", "" };
}

static bool engine_ready(const server_asr_engine * engine) {
    return engine != nullptr && engine->supported();
}

// temp storage for uploads; LLAMA_SERVER_ASR_TMPDIR overrides the default
static std::string asr_temp_dir() {
    if (const char * env = std::getenv("LLAMA_SERVER_ASR_TMPDIR"); env != nullptr && env[0] != '\0') {
        return env;
    }

    std::error_code ec;
    return (std::filesystem::temp_directory_path(ec) / "llama-asr").string();
}

static int status_for_error(const server_asr_error & err) {
    if (err.code == "queue_full")     return 429;
    if (err.code == "storage_full")   return 507;
    if (err.code == "job_not_found")  return 404;
    return 400;
}

server_http_res_ptr server_asr_post_transcription(const server_http_stream_req & req, server_asr_engine * engine) {
    if (!engine_ready(engine)) {
        return make_error_res(501, err_not_supported());
    }

    const server_asr_capabilities & caps = engine->capabilities();

    server_asr_error err;
    std::unique_ptr<server_asr_upload> upload = server_asr_parse_upload(req, asr_temp_dir(), err);
    if (!upload) {
        return make_error_res(status_for_error(err), err);
    }

    if (upload->timestamps && !caps.timestamps) {
        return make_error_res(400, { "timestamps_not_available", "timestamps requested but no aligner model is loaded", "timestamps" });
    }

    if (!upload->language.empty()) {
        std::string code;
        if (server_asr_normalize_language(upload->language, &code).empty()) {
            return make_error_res(400, { "unsupported_language", "the language is not supported by this model", "language" });
        }
    }

    if (!upload->context.empty()) {
        const int32_t n_context = server_asr_count_context_tokens(engine->vocab(), upload->context);
        if (n_context > 1024) {
            return make_error_res(400, { "context_too_long", "context must be at most 1024 tokens", "context" });
        }
    }

    server_asr_job_manager * jobs = engine->jobs();
    if (jobs == nullptr) {
        return make_error_res(501, err_not_implemented());
    }

    server_asr_job_input input;
    input.audio_path = upload->release_temp_path();
    input.timestamps = upload->timestamps;
    input.language   = upload->language;
    input.context    = upload->context;

    const std::string id = jobs->submit(std::move(input), err);
    if (id.empty()) {
        // the job never took ownership, drop the upload
        std::error_code ec;
        std::filesystem::remove(input.audio_path, ec);

        auto res = make_error_res(status_for_error(err), err);
        if (err.code == "queue_full") {
            res->headers["Retry-After"] = "5";
        }
        return res;
    }

    server_asr_job_view view;
    if (!jobs->get(id, view, err)) {
        return make_error_res(500, err);
    }

    auto res = make_job_res(202, view);
    res->headers["Location"]    = "/v1/asr/transcriptions/" + id;
    res->headers["Retry-After"] = "2";

    return res;
}

server_http_res_ptr server_asr_get_transcription(const server_http_req & req, server_asr_engine * engine) {
    if (!engine_ready(engine)) {
        return make_error_res(501, err_not_supported());
    }

    server_asr_job_manager * jobs = engine->jobs();
    if (jobs == nullptr) {
        return make_error_res(404, { "job_not_found", "unknown transcription id", "id" });
    }

    server_asr_job_view view;
    server_asr_error err;
    if (!jobs->get(req.get_param("id"), view, err)) {
        return make_error_res(404, err);
    }

    return make_job_res(200, view);
}

server_http_res_ptr server_asr_delete_transcription(const server_http_req & req, server_asr_engine * engine) {
    if (!engine_ready(engine)) {
        return make_error_res(501, err_not_supported());
    }

    server_asr_job_manager * jobs = engine->jobs();
    if (jobs == nullptr) {
        return make_error_res(404, { "job_not_found", "unknown transcription id", "id" });
    }

    const std::string id = req.get_param("id");

    server_asr_error err;
    switch (jobs->cancel(id, err)) {
        case server_asr_cancel_result::NOT_FOUND:
            return make_error_res(404, err);

        case server_asr_cancel_result::CANCELLED:
        case server_asr_cancel_result::CANCELLING:
            {
                server_asr_job_view view;
                if (!jobs->get(id, view, err)) {
                    return make_error_res(500, err);
                }
                const int status = view.status == server_asr_job_status::CANCELLING ? 202 : 200;
                return make_job_res(status, view);
            }

        case server_asr_cancel_result::DELETED:
            {
                auto res = std::make_unique<server_http_res>();
                res->status = 204;
                return res;
            }
    }

    return make_error_res(500, { "internal_error", "unexpected cancel result", "" });
}

void server_asr_ws_realtime(const server_http_req & req, server_http_context::ws_conn & conn, server_asr_engine * engine) {
    server_asr_ws_session(req, conn, engine);
}

server_http_res_ptr server_asr_router_unsupported(const server_http_stream_req & req) {
    GGML_UNUSED(req);

    return make_error_res(501, { "asr_not_supported", "the router does not support /v1/asr", "" });
}

void server_asr_router_unsupported_ws(const server_http_req & req, server_http_context::ws_conn & conn) {
    GGML_UNUSED(req);

    conn.send(safe_json_to_str(json {
        { "type",  "error" },
        { "code",  "asr_not_supported" },
        { "message", "the router does not support /v1/asr" },
        { "fatal", true  },
    }));
    conn.close();
}
