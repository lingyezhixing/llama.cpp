#include "server-asr-upload.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

// text fields are small metadata; audio is limited only by disk space
static constexpr size_t MAX_FIELD_BYTES = 1024 * 1024;

static bool equals_case_insensitive(const std::string & a, const char * b) {
    const size_t n = std::strlen(b);
    if (a.size() != n) {
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        char ca = a[i];
        char cb = b[i];
        if (ca >= 'A' && ca <= 'Z') {
            ca = (char) (ca - 'A' + 'a');
        }
        if (cb >= 'A' && cb <= 'Z') {
            cb = (char) (cb - 'A' + 'a');
        }
        if (ca != cb) {
            return false;
        }
    }
    return true;
}

static std::string get_header(const std::map<std::string, std::string> & headers, const char * key) {
    for (const auto & [name, value] : headers) {
        if (equals_case_insensitive(name, key)) {
            return value;
        }
    }
    return "";
}

// a generated name, never the client filename
static std::string make_temp_path(const std::string & temp_dir) {
    static std::atomic<uint64_t> counter{0};

    const uint64_t stamp = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    const uint64_t id = counter.fetch_add(1);

    char name[64];
    std::snprintf(name, sizeof(name), "asr_%016" PRIx64 "_%04" PRIx64 ".audio", stamp, id);

    return (std::filesystem::path(temp_dir) / name).string();
}

server_asr_upload::~server_asr_upload() {
    if (temp_path.empty()) {
        return;
    }

    std::error_code ec;
    std::filesystem::remove(temp_path, ec);
}

std::string server_asr_upload::release_temp_path() {
    std::string out = temp_path;
    temp_path.clear();
    return out;
}

std::unique_ptr<server_asr_upload> server_asr_parse_upload(
        const server_http_stream_req & req,
        const std::string & temp_dir,
        server_asr_error & err) {
    err = {};

    const std::string content_type = get_header(req.req.headers, "Content-Type");
    if (content_type.find("multipart/form-data") == std::string::npos) {
        err.code    = "invalid_multipart";
        err.message = "expected a multipart/form-data body";
        return nullptr;
    }
    if (!req.read_multipart) {
        err.code    = "invalid_multipart";
        err.message = "the request body cannot be read";
        return nullptr;
    }

    std::error_code ec;
    std::filesystem::create_directories(temp_dir, ec);
    if (ec) {
        err.code    = "storage_full";
        err.message = "cannot create the ASR temp directory";
        return nullptr;
    }

    auto upload = std::make_unique<server_asr_upload>();

    std::ofstream file;
    std::string   field;        // part currently streaming
    std::string   field_data;   // buffered text field
    bool          have_file = false;

    auto commit_field = [&]() -> bool {
        if (field.empty()) {
            return true;
        }
        if (field == "timestamps") {
            if (field_data == "true") {
                upload->timestamps = true;
            } else if (field_data == "false") {
                upload->timestamps = false;
            } else {
                err = { "invalid_timestamps", "timestamps must be true or false", "timestamps" };
                return false;
            }
        } else if (field == "language") {
            upload->language = field_data;
        } else if (field == "context") {
            upload->context = field_data;
        }
        // unknown fields are ignored
        return true;
    };

    auto on_part = [&](const server_http_multipart_part & part) -> bool {
        if (!commit_field()) {
            return false;
        }

        field = part.field;
        field_data.clear();

        if (field != "file") {
            return true;
        }
        if (have_file) {
            err = { "multiple_files", "only one file part is supported", "file" };
            return false;
        }

        have_file = true;
        upload->filename     = part.filename;
        upload->content_type = part.content_type;
        upload->temp_path    = make_temp_path(temp_dir);

        file.open(upload->temp_path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) {
            err = { "storage_full", "cannot create the upload temp file", "file" };
            return false;
        }
        return true;
    };

    auto on_data = [&](const char * data, size_t len) -> bool {
        if (field == "file") {
            file.write(data, (std::streamsize) len);
            if (!file) {
                err = { "storage_full", "cannot write the upload temp file", "file" };
                return false;
            }
            upload->size += len;
            return true;
        }

        if (field_data.size() + len > MAX_FIELD_BYTES) {
            err = { "field_too_large", "a text field is larger than 1 MiB", field };
            return false;
        }
        field_data.append(data, len);
        return true;
    };

    bool ok = req.read_multipart(on_part, on_data);

    if (file.is_open()) {
        file.close();
    }

    if (ok && !commit_field()) {
        ok = false;
    }

    if (!ok) {
        if (err.empty()) {
            err = { "invalid_multipart", "the multipart body is malformed" };
        }
        return nullptr;
    }

    if (!have_file) {
        err = { "file_required", "a file part is required", "file" };
        return nullptr;
    }

    return upload;
}
