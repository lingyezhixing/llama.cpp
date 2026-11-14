// Tests the Qwen3-ASR shared helpers: language handling, output parsing, item
// clamping, and the JSON shapes of the /v1/asr responses. No model required.

#include "server-asr-common.h"
#include "server-asr-upload.h"

#include "testing.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

static void test_normalize_language(testing & t) {
    std::string code = "sentinel";

    t.assert_equal("iso code",     std::string("English"), server_asr_normalize_language("en", &code));
    t.assert_equal("iso code out", std::string("en"),      code);
    t.assert_equal("uppercase",    std::string("English"), server_asr_normalize_language("EN"));
    t.assert_equal("canonical",    std::string("Chinese"), server_asr_normalize_language("Chinese", &code));
    t.assert_equal("canonical out",std::string("zh"),      code);
    t.assert_equal("lowercase",    std::string("Chinese"), server_asr_normalize_language("chinese"));
    t.assert_equal("three letter", std::string("Cantonese"), server_asr_normalize_language("yue"));
    t.assert_equal("padded",       std::string("Japanese"), server_asr_normalize_language("  ja  "));

    t.assert_equal("unknown",      std::string(""), server_asr_normalize_language("klingon", &code));
    t.assert_equal("unknown code", std::string(""), code);
    t.assert_equal("empty",        std::string(""), server_asr_normalize_language(""));
}

static void test_parse_output(testing & t) {
    server_asr_parse_result parsed = server_asr_parse_output("language English<asr_text>Hello");
    t.assert_equal("language", std::string("English"), parsed.language);
    t.assert_equal("code",     std::string("en"),      parsed.language_code);
    t.assert_equal("text",     std::string("Hello"),   parsed.text);

    parsed = server_asr_parse_output("language None<asr_text>");
    t.assert_equal("none language", std::string(""), parsed.language);
    t.assert_equal("none text",     std::string(""), parsed.text);

    parsed = server_asr_parse_output("Hello");
    t.assert_equal("plain language", std::string(""),      parsed.language);
    t.assert_equal("plain text",     std::string("Hello"), parsed.text);

    parsed = server_asr_parse_output("Hello there", "Chinese");
    t.assert_equal("forced language", std::string("Chinese"),     parsed.language);
    t.assert_equal("forced code",     std::string("zh"),          parsed.language_code);
    t.assert_equal("forced text",     std::string("Hello there"), parsed.text);

    // metadata only, no marker yet: nothing is safe to commit
    parsed = server_asr_parse_output("language Eng");
    t.assert_equal("partial language", std::string(""), parsed.language);
    t.assert_equal("partial text",     std::string(""), parsed.text);

    // an unknown language still leaves the text
    parsed = server_asr_parse_output("language Klingon<asr_text>hi");
    t.assert_equal("unknown language", std::string(""),   parsed.language);
    t.assert_equal("unknown text",     std::string("hi"), parsed.text);

    // a marker inside plain text is not metadata
    parsed = server_asr_parse_output("say <asr_text> now");
    t.assert_equal("marker language", std::string(""),                  parsed.language);
    t.assert_equal("marker text",     std::string("say <asr_text> now"), parsed.text);

    // only the first marker is the metadata boundary
    parsed = server_asr_parse_output("language Chinese<asr_text>a<asr_text>b");
    t.assert_equal("first marker language", std::string("Chinese"),     parsed.language);
    t.assert_equal("first marker text",     std::string("a<asr_text>b"), parsed.text);
}

static void test_sort_and_clamp_items(testing & t) {
    std::vector<server_asr_item> items = {
        {"C", 500, 700},
        {"A", 120, 380},
        {"B", -50, -10},
        {"D", 900, 2000},
    };
    server_asr_sort_and_clamp_items(items, 1000);

    t.assert_equal("count",       (size_t) 4,        items.size());
    t.assert_equal("first text",  std::string("B"),  items[0].text);
    t.assert_equal("first start", (int) 0,           (int) items[0].start_ms);
    t.assert_equal("first end",   (int) 0,           (int) items[0].end_ms);
    t.assert_equal("second text", std::string("A"),  items[1].text);
    t.assert_equal("third text",  std::string("C"),  items[2].text);
    t.assert_equal("clamped end", (int) 1000,        (int) items[3].end_ms);
    t.assert_equal("clamped start",(int) 900,        (int) items[3].start_ms);

    std::vector<server_asr_item> inverted = { {"X", 400, 100} };
    server_asr_sort_and_clamp_items(inverted, 1000);
    t.assert_equal("inverted start", (int) 400, (int) inverted[0].start_ms);
    t.assert_equal("inverted end",   (int) 400, (int) inverted[0].end_ms);

    std::vector<server_asr_item> empty;
    server_asr_sort_and_clamp_items(empty, 1000);
    t.assert_true("empty stays empty", empty.empty());
}

static void test_job_json(testing & t) {
    server_asr_job_view view;
    view.id = "asr_test";
    view.timing.created_at = "2026-11-13T16:00:00Z";
    view.audio_total_ms = 180000;

    json job = server_asr_job_json(view);
    t.assert_equal("object", std::string("asr.transcription"), job.at("object").get<std::string>());
    t.assert_equal("id",     std::string("asr_test"),          job.at("id").get<std::string>());
    t.assert_equal("status", std::string("queued"),            job.at("status").get<std::string>());
    t.assert_true ("created_at",       job.at("timing").contains("created_at"));
    t.assert_true ("no started_at",    !job.at("timing").contains("started_at"));
    t.assert_true ("no processing_ms", !job.at("timing").contains("processing_ms"));
    t.assert_true ("no usage",         !job.contains("usage"));
    t.assert_true ("no result",        !job.contains("result"));
    t.assert_true ("no error",         !job.contains("error"));
    t.assert_equal("percent", 0.0, job.at("progress").at("percent").get<double>());

    // running: progress, usage and an eta
    view.status = server_asr_job_status::RUNNING;
    view.timing.started_at = "2026-11-13T16:00:03Z";
    view.timing.processing_ms = 12000;
    view.timing.eta_ms = 21000;
    view.timing.estimated_end_at = "2026-11-13T16:00:36Z";
    view.audio_processed_ms = 65000;
    view.has_usage = true;
    view.usage.input_tokens = 1300;

    job = server_asr_job_json(view);
    t.assert_equal("running status",      std::string("running"), job.at("status").get<std::string>());
    t.assert_equal("running percent",     36.1, job.at("progress").at("percent").get<double>());
    t.assert_equal("running eta",         21000, job.at("timing").at("eta_ms").get<int>());
    t.assert_true ("running end at",      job.at("timing").contains("estimated_end_at"));
    t.assert_equal("running processing",  12000, job.at("timing").at("processing_ms").get<int>());
    t.assert_true ("running usage",       job.contains("usage"));
    t.assert_equal("running input token", 1300, job.at("usage").at("input_tokens").get<int>());

    // paused: no eta, a paused_at, a reason
    view.status = server_asr_job_status::PAUSED;
    view.paused_reason = "realtime_priority";
    view.timing.paused_at = "2026-11-13T16:00:15Z";
    view.timing.eta_ms = 0;
    view.timing.estimated_end_at.clear();

    job = server_asr_job_json(view);
    t.assert_equal("paused status",  std::string("paused"), job.at("status").get<std::string>());
    t.assert_equal("paused reason",  std::string("realtime_priority"), job.at("paused_reason").get<std::string>());
    t.assert_true ("paused at",      job.at("timing").contains("paused_at"));
    t.assert_true ("no paused eta",  !job.at("timing").contains("eta_ms"));

    // succeeded: a result, no eta
    view.status = server_asr_job_status::SUCCEEDED;
    view.timing.paused_at.clear();
    view.timing.finished_at = "2026-11-13T16:00:36Z";
    view.has_result = true;
    view.include_items = true;
    view.result.text = "Hello world.";
    view.result.language = "English";
    view.result.language_code = "en";
    view.result.items = { {"Hello", 120, 380}, {"world", 380, 700} };

    job = server_asr_job_json(view);
    t.assert_equal("succeeded status", std::string("succeeded"), job.at("status").get<std::string>());
    t.assert_true ("finished at",      job.at("timing").contains("finished_at"));
    t.assert_true ("no succeeded eta", !job.at("timing").contains("eta_ms"));
    t.assert_equal("result text",      std::string("Hello world."), job.at("result").at("text").get<std::string>());
    t.assert_equal("result items",     (size_t) 2, job.at("result").at("items").size());
    t.assert_equal("item start",       120, job.at("result").at("items").at(0).at("start_ms").get<int>());

    view.include_items = false;
    job = server_asr_job_json(view);
    t.assert_true("items are optional", !job.at("result").contains("items"));

    // failed: an error object
    view.status = server_asr_job_status::FAILED;
    view.has_result = false;
    view.error.code = "output_limit_reached";
    view.error.message = "chunk hit the generation cap twice";

    job = server_asr_job_json(view);
    t.assert_equal("failed status", std::string("failed"), job.at("status").get<std::string>());
    t.assert_equal("failed code",   std::string("output_limit_reached"), job.at("error").at("code").get<std::string>());
}

static void test_usage_json(testing & t) {
    server_asr_usage usage;
    usage.input_tokens = 3655;
    usage.output_tokens = 441;
    usage.total_tokens = 4096;
    usage.cached_tokens = 0;
    usage.audio_ms = 180000;
    usage.chunks = 2;

    json out = server_asr_usage_json(usage);
    t.assert_equal("input",  (int) 3655,   out.at("input_tokens").get<int>());
    t.assert_equal("output", (int) 441,    out.at("output_tokens").get<int>());
    t.assert_equal("total",  (int) 4096,   out.at("total_tokens").get<int>());
    t.assert_equal("audio",  (int) 180000, out.at("audio_ms").get<int>());
    t.assert_equal("chunks", (int) 2,      out.at("chunks").get<int>());
    t.assert_true ("no aligner",      !out.contains("aligner"));
    t.assert_true ("no decode steps", !out.contains("decode_steps"));

    usage.has_aligner = true;
    usage.aligner_input_tokens = 700;
    usage.aligner_timestamp_tokens = 64;
    usage.aligner_total_tokens = 764;
    usage.aligner_chunks = 2;

    out = server_asr_usage_json(usage);
    t.assert_true ("aligner",         out.contains("aligner"));
    t.assert_equal("aligner input",   (int) 700, out.at("aligner").at("input_tokens").get<int>());
    t.assert_equal("aligner stamps",  (int) 64,  out.at("aligner").at("timestamp_tokens").get<int>());
    t.assert_equal("aligner total",   (int) 764, out.at("aligner").at("total_tokens").get<int>());
    t.assert_equal("aligner chunks",  (int) 2,   out.at("aligner").at("chunks").get<int>());

    // realtime usage carries the cumulative counters
    usage.decode_steps = 60;
    usage.committed_output_tokens = 2100;
    out = server_asr_usage_json(usage);
    t.assert_equal("decode steps",    60,   out.at("decode_steps").get<int>());
    t.assert_equal("committed",       2100, out.at("committed_output_tokens").get<int>());

    server_asr_usage total;
    total.add(usage);
    total.add(usage);
    t.assert_equal("sum input",     (int) 7310,  total.input_tokens);
    t.assert_equal("sum output",    (int) 882,   total.output_tokens);
    t.assert_equal("sum total",     (int) 8192,  total.total_tokens);
    t.assert_equal("sum audio",     (int) 360000,total.audio_ms);
    t.assert_equal("sum chunks",    (int) 4,     total.chunks);
    t.assert_equal("sum aligner",   (int) 1400,  total.aligner_input_tokens);
    t.assert_equal("sum steps",     (int) 120,   total.decode_steps);
    t.assert_true ("sum has aligner", total.has_aligner);
}

static void test_error_json(testing & t) {
    server_asr_error err;
    err.code = "timestamps_not_available";
    err.message = "timestamps requested but no aligner model is loaded";

    json out = server_asr_error_json(err);
    t.assert_equal("code",    std::string("timestamps_not_available"), out.at("code").get<std::string>());
    t.assert_equal("message", std::string("timestamps requested but no aligner model is loaded"), out.at("message").get<std::string>());
    t.assert_true ("no param", !out.contains("param"));
    t.assert_true ("non empty", !err.empty());

    err.param = "timestamps";
    out = server_asr_error_json(err);
    t.assert_equal("param", std::string("timestamps"), out.at("param").get<std::string>());

    t.assert_true("empty", server_asr_error().empty());
}

static void test_result_json(testing & t) {
    server_asr_result result;
    result.text = "Hello";
    result.language = "English";
    result.language_code = "en";
    result.items = { {"Hello", 0, 100} };

    json out = server_asr_result_json(result, false);
    t.assert_equal("text", std::string("Hello"), out.at("text").get<std::string>());
    t.assert_true("no items", !out.contains("items"));

    out = server_asr_result_json(result, true);
    t.assert_equal("items", (size_t) 1, out.at("items").size());
    t.assert_equal("item text", std::string("Hello"), out.at("items").at(0).at("text").get<std::string>());
    t.assert_equal("item end",  100, out.at("items").at(0).at("end_ms").get<int>());
}

static void test_now_rfc3339(testing & t) {
    const std::string now = server_asr_now_rfc3339();
    const std::regex pattern("^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$");
    t.assert_equal("length", (size_t) 20, now.size());
    t.assert_true ("format", std::regex_match(now, pattern));
}

static void test_count_context_tokens(testing & t) {
    t.assert_equal("empty", 0, server_asr_count_context_tokens(nullptr, ""));
}


//
// upload parsing
//

struct fake_part {
    std::string field;
    std::string filename;
    std::string content_type;
    std::string data;
};

static server_http_multipart_reader make_test_reader(const std::vector<fake_part> & parts) {
    return [parts](
            const std::function<bool(const server_http_multipart_part &)> & on_part,
            const std::function<bool(const char *, size_t)> & on_data) {
        for (const auto & p : parts) {
            const server_http_multipart_part part{ p.field, p.filename, p.content_type };
            if (!on_part(part)) {
                return false;
            }
            if (!p.data.empty() && !on_data(p.data.data(), p.data.size())) {
                return false;
            }
        }
        return true;
    };
}

static server_http_stream_req make_test_req(const std::vector<fake_part> & parts) {
    static const std::function<bool()> never_stop = [] { return false; };

    server_http_req req{
        {},
        { { "Content-Type", "multipart/form-data; boundary=test" } },
        "/v1/asr/transcriptions",
        "",
        "",
        {},
        never_stop,
    };

    return server_http_stream_req{
        req,
        nullptr,
        make_test_reader(parts),
    };
}

static std::filesystem::path make_test_dir(const std::string & name) {
    std::error_code ec;
    const auto dir = std::filesystem::temp_directory_path(ec) / "llama-test-asr" / name;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    return dir;
}

static size_t count_files(const std::filesystem::path & dir) {
    std::error_code ec;
    size_t n = 0;
    for (const auto & entry : std::filesystem::directory_iterator(dir, ec)) {
        if (entry.is_regular_file()) {
            n++;
        }
    }
    return n;
}

static std::string read_test_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

static void test_upload_parse(testing & t) {
    const auto dir = make_test_dir("parse");

    const std::vector<fake_part> parts = {
        {"timestamps", "", "", "true"},
        {"language",   "", "", "Chinese"},
        {"context",    "", "", "meeting notes"},
        {"file", "test.wav", "audio/wav", "RIFFtestdata"},
    };

    server_asr_error err;
    auto upload = server_asr_parse_upload(make_test_req(parts), dir.string(), err);

    t.assert_true("parsed", upload != nullptr);
    t.assert_true("no error", err.empty());
    if (!upload) {
        return;
    }

    t.assert_true ("timestamps",   upload->timestamps);
    t.assert_equal("language",     std::string("Chinese"),       upload->language);
    t.assert_equal("context",      std::string("meeting notes"), upload->context);
    t.assert_equal("filename",     std::string("test.wav"),      upload->filename);
    t.assert_equal("size",         (size_t) 12,                  (size_t) upload->size);
    t.assert_equal("bytes on disk",std::string("RIFFtestdata"),  read_test_file(upload->temp_path));
    t.assert_true ("temp path ignores the client name", upload->temp_path.find("test.wav") == std::string::npos);
    t.assert_equal("one temp file",(size_t) 1, count_files(dir));

    const std::string temp_path = upload->temp_path;
    upload.reset();
    t.assert_true("destructor removes the temp file", !std::filesystem::exists(temp_path));
    t.assert_equal("no files left", (size_t) 0, count_files(dir));

    // release hands the file to the caller
    upload = server_asr_parse_upload(make_test_req(parts), dir.string(), err);
    t.assert_true("parsed again", upload != nullptr);
    if (upload) {
        const std::string released = upload->release_temp_path();
        upload.reset();
        t.assert_true("released file survives", std::filesystem::exists(released));
        std::filesystem::remove(released);
    }
}

static void test_upload_errors(testing & t) {
    const auto dir = make_test_dir("errors");
    server_asr_error err;

    // file last, then file first
    const std::vector<fake_part> reversed = {
        {"file", "a.wav", "audio/wav", "abcd"},
        {"timestamps", "", "", "true"},
    };
    auto upload = server_asr_parse_upload(make_test_req(reversed), dir.string(), err);
    t.assert_true("file first works", upload != nullptr);
    if (upload) {
        t.assert_true("file first timestamps", upload->timestamps);
        t.assert_equal("file first size", (size_t) 4, (size_t) upload->size);
    }
    upload.reset();

    // no file part
    err = {};
    upload = server_asr_parse_upload(make_test_req({{"timestamps", "", "", "true"}}), dir.string(), err);
    t.assert_true ("missing file rejected", upload == nullptr);
    t.assert_equal("missing file code",     std::string("file_required"), err.code);
    t.assert_equal("missing file param",    std::string("file"),          err.param);
    t.assert_equal("missing file leaves nothing", (size_t) 0, count_files(dir));

    // two file parts
    err = {};
    upload = server_asr_parse_upload(make_test_req({
        {"file", "a.wav", "audio/wav", "aaaa"},
        {"file", "b.wav", "audio/wav", "bbbb"},
    }), dir.string(), err);
    t.assert_true ("two files rejected", upload == nullptr);
    t.assert_equal("two files code",     std::string("multiple_files"), err.code);
    t.assert_equal("two files leave nothing", (size_t) 0, count_files(dir));

    // oversized text field
    err = {};
    upload = server_asr_parse_upload(make_test_req({
        {"file",    "a.wav", "audio/wav", "aaaa"},
        {"context", "", "", std::string(1024 * 1024 + 1, 'x')},
    }), dir.string(), err);
    t.assert_true ("large field rejected", upload == nullptr);
    t.assert_equal("large field code",     std::string("field_too_large"), err.code);
    t.assert_equal("large field param",    std::string("context"),         err.param);
    t.assert_equal("large field leaves nothing", (size_t) 0, count_files(dir));

    // bad boolean
    err = {};
    upload = server_asr_parse_upload(make_test_req({
        {"file",       "a.wav", "audio/wav", "aaaa"},
        {"timestamps", "", "", "yes"},
    }), dir.string(), err);
    t.assert_true ("bad timestamps rejected", upload == nullptr);
    t.assert_equal("bad timestamps code",     std::string("invalid_timestamps"), err.code);

    // a reader that fails midway
    err = {};
    server_http_stream_req broken = make_test_req({{"file", "a.wav", "audio/wav", "aaaa"}});
    broken.read_multipart = [](const std::function<bool(const server_http_multipart_part &)> & on_part,
                               const std::function<bool(const char *, size_t)> & on_data) {
        on_part({"file", "a.wav", "audio/wav"});
        on_data("aa", 2);
        return false;
    };
    upload = server_asr_parse_upload(broken, dir.string(), err);
    t.assert_true ("broken stream rejected", upload == nullptr);
    t.assert_equal("broken stream code",     std::string("invalid_multipart"), err.code);
    t.assert_equal("broken stream leaves nothing", (size_t) 0, count_files(dir));

    // a non multipart request
    err = {};
    server_http_stream_req raw = make_test_req({});
    raw.req.headers = { { "Content-Type", "application/octet-stream" } };
    upload = server_asr_parse_upload(raw, dir.string(), err);
    t.assert_true ("raw body rejected", upload == nullptr);
    t.assert_equal("raw body code",     std::string("invalid_multipart"), err.code);
}


//
// offline job manager
//

#include "server-asr-job.h"
#include "server-asr-offline.h"

#include <chrono>
#include <fstream>
#include <thread>

// configurable runner so the manager can be tested without a model
namespace {

struct test_runner : server_asr_offline_runner {
    int  chunks    = 3;
    int  step_ms   = 5;
    bool fail      = false;
    bool probe_bad = false;

    std::atomic<bool> gate{false};   // blocks the first chunk until opened
    std::atomic<int>  runs{0};

    bool probe(const std::string & audio_path, int64_t & audio_total_ms, server_asr_error & err) override {
        if (probe_bad) {
            err = { "invalid_audio", "the audio cannot be decoded", "file" };
            return false;
        }
        if (audio_path.empty()) {
            err = { "invalid_audio", "no audio path", "file" };
            return false;
        }
        audio_total_ms = chunks * 100000;
        return true;
    }

    server_asr_run_status run(
            const server_asr_job_input & input,
            server_asr_run_state & state,
            const std::atomic<bool> & cancel,
            const std::function<bool()> & pause_requested,
            const std::function<void(int64_t, const server_asr_usage &)> & progress,
            server_asr_error & err) override {
        runs++;
        if (!gate.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        while (state.next_chunk < (size_t) chunks) {
            if (cancel.load()) {
                return server_asr_run_status::CANCELLED;
            }
            if (pause_requested()) {
                return server_asr_run_status::PAUSED;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));

            state.next_chunk++;
            state.audio_processed_ms = (int64_t) state.next_chunk * 100000;
            state.usage.audio_ms      = state.audio_processed_ms;
            state.usage.chunks        = (int64_t) state.next_chunk;
            state.usage.input_tokens += 10;
            state.usage.output_tokens += 2;
            state.usage.total_tokens  = state.usage.input_tokens + state.usage.output_tokens;
            state.result.text          = "test " + input.language;
            state.result.language      = input.language;
            state.result.language_code = "en";

            progress(state.audio_processed_ms, state.usage);

            if (fail && state.next_chunk == (size_t) chunks) {
                err = { "output_limit_reached", "the chunk hit the generation cap twice", "" };
                return server_asr_run_status::FAILED;
            }
        }

        return server_asr_run_status::DONE;
    }
};

static std::string make_test_audio(const std::string & name) {
    const auto dir = std::filesystem::temp_directory_path() / "llama-test-asr-jobs";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / name;
    std::ofstream(path, std::ios::binary) << "audio";
    return path.string();
}

static bool wait_for_status(server_asr_job_manager & mgr, const std::string & id,
                            server_asr_job_status want, int timeout_ms = 5000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        server_asr_job_view view;
        server_asr_error err;
        if (mgr.get(id, view, err) && view.status == want) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

} // namespace

static void test_job_happy_path(testing & t) {
    test_runner runner;
    runner.gate.store(true);

    server_asr_job_manager mgr(&runner);

    server_asr_job_input input;
    input.audio_path = make_test_audio("happy.wav");
    input.language   = "Chinese";

    server_asr_error err;
    const std::string id = mgr.submit(std::move(input), err);
    t.assert_true("submitted", !id.empty());
    t.assert_true("id prefix", id.rfind("asr_", 0) == 0);

    t.assert_true("reaches success", wait_for_status(mgr, id, server_asr_job_status::SUCCEEDED));

    server_asr_job_view view;
    t.assert_true("get works", mgr.get(id, view, err));

    t.assert_equal("object",     std::string("asr.transcription"), server_asr_job_json(view).at("object").get<std::string>());
    t.assert_equal("text",       std::string("test Chinese"), view.result.text);
    t.assert_equal("audio done", (int) 300000, (int) view.audio_processed_ms);
    t.assert_equal("total",      (int) 300000, (int) view.audio_total_ms);
    t.assert_true ("usage",      view.has_usage && view.usage.chunks == 3);
    t.assert_true ("started",    !view.timing.started_at.empty());
    t.assert_true ("finished",   !view.timing.finished_at.empty());
    t.assert_equal("no eta",     (int) 0, (int) view.timing.eta_ms);

    // a finished job is retained until it is deleted
    t.assert_equal("delete terminal", (int) server_asr_cancel_result::DELETED,
                   (int) mgr.cancel(id, err));
    t.assert_true("gone", !mgr.get(id, view, err));
    t.assert_equal("not found code", std::string("job_not_found"), err.code);
}

static void test_job_pause_and_resume(testing & t) {
    test_runner runner;
    runner.step_ms = 30;

    server_asr_job_manager mgr(&runner);

    server_asr_job_input input;
    input.audio_path = make_test_audio("pause.wav");

    server_asr_error err;
    const std::string id = mgr.submit(std::move(input), err);
    t.assert_true("submitted", !id.empty());

    t.assert_true("running", wait_for_status(mgr, id, server_asr_job_status::RUNNING));

    // realtime takes the context: the job must pause at the next chunk boundary
    mgr.set_realtime_active(true);
    t.assert_true("paused", wait_for_status(mgr, id, server_asr_job_status::PAUSED));

    server_asr_job_view view;
    mgr.get(id, view, err);
    t.assert_equal("paused reason", std::string("realtime_priority"), view.paused_reason);
    t.assert_true ("paused at",     !view.timing.paused_at.empty());
    t.assert_equal("no eta while paused", (int) 0, (int) view.timing.eta_ms);

    // an ETA needs a running job and at least one completed chunk
    mgr.set_realtime_active(false);
    t.assert_true("running again", wait_for_status(mgr, id, server_asr_job_status::RUNNING));
    t.assert_true("finishes",      wait_for_status(mgr, id, server_asr_job_status::SUCCEEDED));

    mgr.get(id, view, err);
    t.assert_equal("resumed audio", (int) 300000, (int) view.audio_processed_ms);
    t.assert_true ("paused_at cleared", view.timing.paused_at.empty());
}

static void test_job_cancel(testing & t) {
    test_runner runner;
    runner.step_ms = 30;

    server_asr_job_manager mgr(&runner);

    // queued: realtime keeps the worker away from the queue
    mgr.set_realtime_active(true);

    server_asr_job_input a;
    a.audio_path = make_test_audio("cancel_a.wav");

    server_asr_error err;
    const std::string id_a = mgr.submit(std::move(a), err);
    t.assert_true("queued job", wait_for_status(mgr, id_a, server_asr_job_status::QUEUED));

    t.assert_equal("cancel queued", (int) server_asr_cancel_result::CANCELLED,
                   (int) mgr.cancel(id_a, err));

    server_asr_job_view view;
    t.assert_true("still recorded", mgr.get(id_a, view, err));
    t.assert_equal("cancelled", std::string("cancelled"), server_asr_status_name(view.status));

    // running: the worker stops at the next safe point
    mgr.set_realtime_active(false);

    server_asr_job_input b;
    b.audio_path = make_test_audio("cancel_b.wav");
    const std::string id_b = mgr.submit(std::move(b), err);
    t.assert_true("running job", wait_for_status(mgr, id_b, server_asr_job_status::RUNNING));

    t.assert_equal("cancel running", (int) server_asr_cancel_result::CANCELLING,
                   (int) mgr.cancel(id_b, err));
    t.assert_true("cancelled eventually", wait_for_status(mgr, id_b, server_asr_job_status::CANCELLED));
}

static void test_job_errors(testing & t) {
    test_runner runner;
    runner.step_ms = 5;

    server_asr_job_manager mgr(&runner, 1); // one job at a time, capacity one

    server_asr_error err;

    // probe failure is reported before the job exists
    runner.probe_bad = true;
    server_asr_job_input bad;
    bad.audio_path = make_test_audio("bad.wav");
    t.assert_true("probe failure", mgr.submit(std::move(bad), err).empty());
    t.assert_equal("probe code", std::string("invalid_audio"), err.code);

    runner.probe_bad = false;

    // the runner failing is stored on the job
    runner.fail = true;
    server_asr_job_input failing;
    failing.audio_path = make_test_audio("fail.wav");
    const std::string id = mgr.submit(std::move(failing), err);
    t.assert_true("failed", wait_for_status(mgr, id, server_asr_job_status::FAILED));

    server_asr_job_view view;
    mgr.get(id, view, err);
    t.assert_equal("failed code", std::string("output_limit_reached"), view.error.code);

    runner.fail = false;
}

static void test_job_queue_full(testing & t) {
    test_runner runner;
    runner.step_ms = 200;

    server_asr_job_manager mgr(&runner, 1);

    server_asr_job_input a;
    a.audio_path = make_test_audio("full_a.wav");

    server_asr_error err;
    const std::string id = mgr.submit(std::move(a), err);
    t.assert_true("first job", !id.empty());
    t.assert_true("running", wait_for_status(mgr, id, server_asr_job_status::RUNNING));

    server_asr_job_input b;
    b.audio_path = make_test_audio("full_b.wav");
    t.assert_true("queue full", mgr.submit(std::move(b), err).empty());
    t.assert_equal("queue code", std::string("queue_full"), err.code);

    // stop the worker quickly, the second upload is the caller's to remove
    mgr.terminate();
    std::error_code ec;
    std::filesystem::remove(make_test_audio("full_b.wav"), ec);
}


//
// offline chunk planning
//

static void test_chunk_limits(testing & t) {
    // 4096 tokens, no timestamps: (4096 * 0.85 - 64) / 20 seconds
    const int64_t c4096 = server_asr_max_chunk_ms(4096, 0, false, 0);
    t.assert_true("4096 is about 170 s", c4096 >= 170000 && c4096 <= 171000);

    const int64_t c8192 = server_asr_max_chunk_ms(8192, 0, false, 0);
    t.assert_true("8192 is about 345 s", c8192 >= 344000 && c8192 <= 345000);
    t.assert_true("larger context is larger", c8192 > c4096);

    // the official aligner limit wins at 8192
    t.assert_equal("timestamps cap at 180 s", (int) 180000,
                   (int) server_asr_max_chunk_ms(8192, 0, true, 8192));

    // a 4096 aligner context limits the chunk before the 180 s cap does
    const int64_t ts4096 = server_asr_max_chunk_ms(8192, 0, true, 4096);
    t.assert_true("aligner 4096 is about 110 s", ts4096 >= 110000 && ts4096 <= 111000);

    // context tokens come out of the budget
    const int64_t with_ctx = server_asr_max_chunk_ms(4096, 1024, false, 0);
    t.assert_true("context shrinks the chunk", with_ctx < c4096 && with_ctx > 100000);

    t.assert_equal("budget below the overhead", (int) 0, (int) server_asr_max_chunk_ms(64, 0, false, 0));
    // 128 tokens still allow a couple of seconds, the caller rejects under one
    t.assert_true("tiny budget stays small", server_asr_max_chunk_ms(128, 0, false, 0) < 3000);
}

static void test_chunk_ranges(testing & t) {
    const int64_t total_ms = 400000;

    std::vector<float> energy(total_ms / 100, 1.0f);
    // a quiet window at 170 s, where the first cut should land
    for (size_t i = 1700; i < 1710; i++) {
        energy[i] = 0.01f;
    }

    std::vector<server_asr_chunk_range> ranges;
    server_asr_error err;

    server_asr_compute_chunk_ranges(total_ms, 4096, 0, false, 0, energy, ranges, err);

    t.assert_true("no error", err.empty());
    t.assert_true("several chunks", ranges.size() >= 2);
    t.assert_equal("first cut at the quiet window", (int) 170000, (int) ranges.front().end_ms);

    // ranges are contiguous and cover the whole audio
    t.assert_equal("starts at zero", (int) 0, (int) ranges.front().start_ms);
    t.assert_equal("ends at the duration", (int) total_ms, (int) ranges.back().end_ms);

    bool contiguous = true;
    bool non_empty  = true;
    for (size_t i = 0; i < ranges.size(); i++) {
        non_empty = non_empty && ranges[i].end_ms > ranges[i].start_ms;
        if (i > 0) {
            contiguous = contiguous && ranges[i].start_ms == ranges[i - 1].end_ms;
        }
    }
    t.assert_true("contiguous", contiguous);
    t.assert_true("no empty chunk", non_empty);

    // an impossible context is rejected
    std::vector<server_asr_chunk_range> bad;
    err = {};
    server_asr_compute_chunk_ranges(total_ms, 64, 0, false, 0, energy, bad, err);
    t.assert_equal("too small", std::string("context_too_small"), err.code);
    t.assert_true("no ranges on error", bad.empty());
}


//
// realtime protocol
//

#include "server-asr-realtime.h"

#include <deque>

namespace {

struct mock_ws_conn : server_http_context::ws_conn {
    std::deque<std::pair<int, std::string>> incoming;
    std::vector<std::string> sent;
    bool closed = false;

    void push_text(const std::string & text) { incoming.emplace_back(1, text); }
    void push_binary(const std::string & data) { incoming.emplace_back(2, data); }

    int read(std::string & msg) override {
        if (incoming.empty()) {
            return 0;
        }
        msg = incoming.front().second;
        const int kind = incoming.front().first;
        incoming.pop_front();
        return kind;
    }
    bool send(const std::string & msg) override { sent.push_back(msg); return true; }
    bool send_binary(const std::string & msg) override { sent.push_back(msg); return true; }
    bool is_open() const override { return !closed; }
    void close() override { closed = true; }

    std::string joined_deltas() const {
        std::string out;
        for (const auto & msg : sent) {
            json j = json::parse(msg);
            if (j.value("type", std::string("")) == "transcript.delta") {
                out += j.value("text", std::string(""));
            }
        }
        return out;
    }

    std::string type_at(size_t i) const {
        if (i >= sent.size()) {
            return "";
        }
        return json::parse(sent[i]).value("type", std::string(""));
    }
};

struct pcm16_runner {
    static std::string frame(int16_t value, size_t n) {
        std::string out;
        for (size_t i = 0; i < n; i++) {
            out.push_back((char) (value & 0xFF));
            out.push_back((char) ((value >> 8) & 0xFF));
        }
        return out;
    }
};

void run_protocol(mock_ws_conn & conn) {
    server_asr_fake_realtime_runner runner;
    server_asr_capabilities caps;
    caps.realtime = true;
    server_asr_ws_protocol(conn, &runner, caps, nullptr);
}

} // namespace

static void test_realtime_happy_path(testing & t) {
    mock_ws_conn conn;
    conn.push_text(R"({"type":"session.start","language":"English"})");
    conn.push_binary(pcm16_runner::frame(1000, 1600));
    conn.push_binary(pcm16_runner::frame(1000, 1600));
    conn.push_text(R"({"type":"session.finish"})");

    run_protocol(conn);

    t.assert_equal("started",      std::string("session.started"),  conn.type_at(0));
    t.assert_equal("delta first",  std::string("transcript.delta"), conn.type_at(1));
    t.assert_equal("delta second", std::string("transcript.delta"), conn.type_at(2));
    t.assert_equal("finished",     std::string("session.finished"), conn.type_at(3));

    // append-only: the joined deltas are exactly the committed text
    t.assert_equal("joined", std::string("hello world"), conn.joined_deltas());
    t.assert_true ("closed", conn.closed);

    const json finished = json::parse(conn.sent[3]);
    t.assert_equal("audio_ms", 200, finished.at("audio_ms").get<int>());
    t.assert_true("usage", finished.at("usage").at("decode_steps").get<int>() >= 2);
}

static void test_realtime_errors(testing & t) {
    // audio before session.start
    {
        mock_ws_conn conn;
        conn.push_binary(pcm16_runner::frame(0, 10));
        run_protocol(conn);
        t.assert_equal("audio first", std::string("error"), conn.type_at(0));
        t.assert_equal("audio first code",
                std::string("invalid_state"),
                json::parse(conn.sent[0]).at("code").get<std::string>());
    }

    // a second session.start
    {
        mock_ws_conn conn;
        conn.push_text(R"({"type":"session.start"})");
        conn.push_text(R"({"type":"session.start"})");
        run_protocol(conn);
        t.assert_equal("started once", std::string("session.started"), conn.type_at(0));
        t.assert_equal("second start", std::string("error"),           conn.type_at(1));
    }

    // a bad sample rate
    {
        mock_ws_conn conn;
        conn.push_text(R"({"type":"session.start","sample_rate":4000})");
        run_protocol(conn);
        t.assert_equal("sample rate", std::string("invalid_audio_format"),
                json::parse(conn.sent[0]).at("code").get<std::string>());
    }

    // an unknown language
    {
        mock_ws_conn conn;
        conn.push_text(R"({"type":"session.start","language":"Klingon"})");
        run_protocol(conn);
        t.assert_equal("language", std::string("unsupported_language"),
                json::parse(conn.sent[0]).at("code").get<std::string>());
    }

    // cancel
    {
        mock_ws_conn conn;
        conn.push_text(R"({"type":"session.start"})");
        conn.push_binary(pcm16_runner::frame(0, 160));
        conn.push_text(R"({"type":"session.cancel"})");
        run_protocol(conn);
        t.assert_equal("cancelled", std::string("session.cancelled"), conn.type_at(conn.sent.size() - 1));
    }

    // an odd byte count is not PCM16
    {
        mock_ws_conn conn;
        conn.push_text(R"({"type":"session.start"})");
        conn.push_binary(std::string(3, 'x'));
        run_protocol(conn);
        t.assert_equal("odd bytes", std::string("invalid_audio_format"),
                json::parse(conn.sent[1]).at("code").get<std::string>());
    }
}


//
// realtime stable commit
//

using T = llama_token;

static std::vector<T> toks(std::initializer_list<int> ids) {
    std::vector<T> out;
    for (int id : ids) {
        out.push_back((T) id);
    }
    return out;
}

static void test_stable_commit(testing & t) {
    t.assert_equal("low latency",   5,  server_asr_realtime_holdback_tokens("low_latency"));
    t.assert_equal("balanced",      8,  server_asr_realtime_holdback_tokens("balanced"));
    t.assert_equal("high accuracy", 12, server_asr_realtime_holdback_tokens("high_accuracy"));
    t.assert_equal("unknown",       8,  server_asr_realtime_holdback_tokens(""));

    // prefix + ABC against the same previous decode, holdback 5: nothing new is safe
    {
        server_asr_commit_state st;
        st.segment_base_tokens      = toks({1, 2});
        st.segment_raw_tokens       = toks({1, 2, 10, 11, 12});
        st.prev_segment_raw_tokens  = toks({1, 2, 10, 11, 12});
        st.holdback_tokens          = 5;

        size_t begin = 0, end = 0;
        t.assert_true("no commit inside holdback", !server_asr_stable_commit(st, begin, end));
    }

    // a divergence stops the commit at the longest common prefix
    {
        server_asr_commit_state st;
        st.segment_raw_tokens      = toks({1, 2, 10, 11, 12, 13});
        st.prev_segment_raw_tokens = toks({1, 2, 10, 11, 90, 91});
        st.holdback_tokens         = 2;

        size_t begin = 0, end = 0;
        t.assert_true("commit through the LCP", server_asr_stable_commit(st, begin, end));
        t.assert_equal("begin", (size_t) 0, begin);
        t.assert_equal("end",   (size_t) 4, end);

        st.emitted_in_segment = 4;
        t.assert_true("pointer does not move again", !server_asr_stable_commit(st, begin, end));
    }

    // emitted tokens stay inside the fixed prefix of the next decode
    {
        server_asr_commit_state st;
        st.segment_raw_tokens      = toks({10, 11, 12, 13, 14, 15});
        st.prev_segment_raw_tokens = toks({10, 11, 12, 13, 14, 15});
        st.holdback_tokens         = 2;

        size_t begin = 0, end = 0;
        t.assert_true("safe", server_asr_stable_commit(st, begin, end));
        t.assert_true("inside the prefix", end <= st.segment_raw_tokens.size() - (size_t) st.holdback_tokens);

        // the next decode uses raw[0 : len - holdback] as its prefix
        st.prompt_prefix_tokens.assign(st.segment_raw_tokens.begin(),
                                       st.segment_raw_tokens.end() - st.holdback_tokens);
        t.assert_equal("prefix size", (size_t) 4, st.prompt_prefix_tokens.size());
    }
}

static void test_utf8_holdback(testing & t) {
    std::string complete;

    t.assert_equal("ascii", (size_t) 0, server_asr_utf8_holdback("hello", complete));
    t.assert_equal("ascii text", std::string("hello"), complete);

    // a single Chinese code point (3 bytes) is complete
    t.assert_equal("cjk complete", (size_t) 0, server_asr_utf8_holdback("\xE4\xBD\xA0", complete));
    t.assert_equal("cjk text", std::string("\xE4\xBD\xA0"), complete);

    // two of the three bytes: nothing is safe to send yet
    t.assert_equal("cjk partial", (size_t) 2, server_asr_utf8_holdback("\xE4\xBD", complete));
    t.assert_true ("cjk partial empty", complete.empty());

    // a complete code point followed by a partial one
    t.assert_equal("mixed partial", (size_t) 2, server_asr_utf8_holdback("\xE4\xBD\xA0\xE4\xBD", complete));
    t.assert_equal("mixed complete", std::string("\xE4\xBD\xA0"), complete);
}


//
// segment resets
//

static void test_reset_helpers(testing & t) {
    t.assert_equal("low latency",   30000, (int) server_asr_realtime_segment_max_ms("low_latency"));
    t.assert_equal("balanced",      45000, (int) server_asr_realtime_segment_max_ms("balanced"));
    t.assert_equal("high accuracy", 60000, (int) server_asr_realtime_segment_max_ms("high_accuracy"));

    // 5 s of speech with a 500 ms silence in the middle
    const int32_t rate = 16000;
    std::vector<float> samples((size_t) rate * 5, 0.2f);
    const size_t silence_from = (size_t) rate * 2;
    const size_t silence_to   = (size_t) rate * 2 + (size_t) rate / 2;
    for (size_t i = silence_from; i < silence_to; i++) {
        samples[i] = 0.0f;
    }

    int64_t cut = 0;
    const bool found = server_asr_find_silence(samples.data(), samples.size(), rate, 0, 5000, 300, cut);
    t.assert_true("silence found", found);
    t.assert_true("cut inside the silence", cut >= 2000 && cut <= 2510);

    // no silence in pure speech
    std::vector<float> tone((size_t) rate * 2, 0.3f);
    int64_t unused = 0;
    t.assert_true("no silence", !server_asr_find_silence(tone.data(), tone.size(), rate, 0, 2000, 300, unused));

    // the quietest 100 ms window is the silence itself
    const int64_t low = server_asr_lowest_energy_window(samples.data(), samples.size(), rate, 0, 5000);
    t.assert_true("low energy inside the silence", low >= 1900 && low <= 2500);

    // boundary dedupe drops a repeated suffix/prefix pair
    std::vector<llama_token> committed  = toks({1, 2, 3, 4, 5});
    std::vector<llama_token> continuation = toks({4, 5, 6, 7});
    server_asr_boundary_dedupe(committed, continuation);
    t.assert_equal("dedupe size", (size_t) 2, continuation.size());
    t.assert_equal("dedupe first", 6, (int) continuation[0]);

    // no overlap leaves the continuation alone
    std::vector<llama_token> other = toks({7, 8});
    server_asr_boundary_dedupe(committed, other);
    t.assert_equal("no overlap size", (size_t) 2, other.size());
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

    t.test("normalize_language",    test_normalize_language);
    t.test("parse_output",          test_parse_output);
    t.test("sort_and_clamp_items",  test_sort_and_clamp_items);
    t.test("job_json",              test_job_json);
    t.test("usage_json",            test_usage_json);
    t.test("error_json",            test_error_json);
    t.test("result_json",           test_result_json);
    t.test("now_rfc3339",           test_now_rfc3339);
    t.test("count_context_tokens",  test_count_context_tokens);
    t.test("upload_parse",          test_upload_parse);
    t.test("upload_errors",         test_upload_errors);
    t.test("job_happy_path",        test_job_happy_path);
    t.test("job_pause_and_resume",  test_job_pause_and_resume);
    t.test("job_cancel",            test_job_cancel);
    t.test("job_errors",            test_job_errors);
    t.test("job_queue_full",        test_job_queue_full);
    t.test("chunk_limits",          test_chunk_limits);
    t.test("chunk_ranges",          test_chunk_ranges);
    t.test("realtime_happy_path",   test_realtime_happy_path);
    t.test("realtime_errors",       test_realtime_errors);
    t.test("stable_commit",         test_stable_commit);
    t.test("utf8_holdback",         test_utf8_holdback);
    t.test("reset_helpers",         test_reset_helpers);

    return t.summary();
}
