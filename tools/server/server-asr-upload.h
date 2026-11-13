// Streams a /v1/asr/transcriptions upload to a temp file, one part at a time.

#pragma once

#include "server-asr-common.h"
#include "server-http.h"

#include <cstdint>
#include <memory>
#include <string>

struct server_asr_upload {
    std::string temp_path;
    std::string filename;     // client supplied, informational only
    std::string content_type; // client supplied, informational only
    uint64_t    size = 0;     // bytes written

    bool        timestamps = false;
    std::string language;
    std::string context;

    // removes the temp file unless it was released
    ~server_asr_upload();

    // hands the temp file to the caller, which then owns its lifetime
    std::string release_temp_path();
};

// reads the multipart body into temp_dir and collects the small text fields
// gives nullptr and fills err on failure; no temp file survives a failure
std::unique_ptr<server_asr_upload> server_asr_parse_upload(
        const server_http_stream_req & req,
        const std::string & temp_dir,
        server_asr_error & err);
