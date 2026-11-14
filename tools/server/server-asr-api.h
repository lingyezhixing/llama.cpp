// HTTP and WebSocket handlers for /v1/asr/*.

#pragma once

#include "server-asr-common.h"
#include "server-http.h"

#include <memory>

struct server_asr_engine;

// POST /v1/asr/transcriptions, the body is streamed, see server-asr-upload.h
server_http_res_ptr server_asr_post_transcription(const server_http_stream_req & req, server_asr_engine * engine);

// GET and DELETE /v1/asr/transcriptions/:id
server_http_res_ptr server_asr_get_transcription(const server_http_req & req, server_asr_engine * engine);
server_http_res_ptr server_asr_delete_transcription(const server_http_req & req, server_asr_engine * engine);

// WS /v1/asr/realtime
void server_asr_ws_realtime(const server_http_req & req, server_http_context::ws_conn & conn, server_asr_engine * engine);

// router mode: the split process cannot proxy an ASR upload or stream
server_http_res_ptr server_asr_router_unsupported(const server_http_stream_req & req);
void server_asr_router_unsupported_ws(const server_http_req & req, server_http_context::ws_conn & conn);
