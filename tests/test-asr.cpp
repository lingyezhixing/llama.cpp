// Tests the Qwen3-ASR shared helpers: language handling, output parsing, item
// clamping, and the JSON shapes of the /v1/asr responses. No model required.

#include "server-asr-common.h"

#include "testing.h"

#include <cstdlib>
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

    return t.summary();
}
