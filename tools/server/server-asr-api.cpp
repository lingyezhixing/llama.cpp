#include "server-asr-api.h"

#include "server-common.h"

static server_http_res_ptr make_error_res(int status, const server_asr_error & err) {
    auto res = std::make_unique<server_http_res>();

    res->status = status;
    res->data   = safe_json_to_str(json { {"error", server_asr_error_json(err)} });

    return res;
}

static server_asr_error err_not_supported() {
    return { "asr_not_supported", "the loaded model is not Qwen3-ASR", "" };
}

static server_asr_error err_not_implemented() {
    return { "asr_not_implemented", "the offline ASR pipeline is not wired yet", "" };
}

server_http_res_ptr server_asr_post_transcription(const server_http_stream_req & req, const server_asr_capabilities & caps) {
    GGML_UNUSED(req);

    if (!caps.offline) {
        return make_error_res(501, err_not_supported());
    }

    return make_error_res(501, err_not_implemented());
}

server_http_res_ptr server_asr_get_transcription(const server_http_req & req, const server_asr_capabilities & caps) {
    GGML_UNUSED(req);

    if (!caps.offline) {
        return make_error_res(501, err_not_supported());
    }

    return make_error_res(404, { "job_not_found", "unknown transcription id", "" });
}

server_http_res_ptr server_asr_delete_transcription(const server_http_req & req, const server_asr_capabilities & caps) {
    GGML_UNUSED(req);

    if (!caps.offline) {
        return make_error_res(501, err_not_supported());
    }

    return make_error_res(404, { "job_not_found", "unknown transcription id", "" });
}

void server_asr_ws_realtime(const server_http_req & req, server_http_context::ws_conn & conn, const server_asr_capabilities & caps) {
    GGML_UNUSED(req);

    const std::string code = caps.realtime ? "asr_not_implemented" : "asr_not_supported";
    const std::string text = caps.realtime
        ? "the realtime ASR pipeline is not wired yet"
        : "the loaded model is not Qwen3-ASR";

    conn.send(safe_json_to_str(json {
        { "type",  "error" },
        { "code",  code   },
        { "message", text },
        { "fatal", true  },
    }));
    conn.close();
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
