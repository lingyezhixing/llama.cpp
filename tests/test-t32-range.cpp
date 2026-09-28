// Harness for the range state API
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr, "usage: %s -m model.gguf [common args] --mode <h2d|correctness|range-bench> [--n N] [--chunk C]\n", prog);
}

static int fill_context(llama_context * ctx, int n_tokens, int n_ubatch = 512) {
    llama_batch batch = llama_batch_init(n_ubatch, 0, 1);

    for (int i = 0; i < n_tokens; ) {
        batch.n_tokens = 0;

        const int n = std::min(n_ubatch, n_tokens - i);

        for (int j = 0; j < n; ++j) {
            common_batch_add(batch, 10 + (i + j) % 1000, i + j, { 0 }, j == n - 1);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at token %d\n", i);
            llama_batch_free(batch);
            return 1;
        }

        i += n;
    }

    llama_batch_free(batch);
    return 0;
}

static int run_h2d(llama_context * ctx, int n_tokens) {
    const int64_t t_fill0 = ggml_time_us();

    if (fill_context(ctx, n_tokens) != 0) {
        return 1;
    }

    const double t_fill = (ggml_time_us() - t_fill0) / 1e6;

    const size_t size = llama_state_seq_get_size_ext(ctx, 0, 0);
    if (size == 0) {
        fprintf(stderr, "failed to get state size\n");
        return 1;
    }

    std::vector<uint8_t> data(size);

    const int64_t t_d2h0 = ggml_time_us();
    const size_t n_d2h = llama_state_seq_get_data_ext(ctx, data.data(), data.size(), 0, 0);
    const double t_d2h = (ggml_time_us() - t_d2h0) / 1e6;

    llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);

    const int64_t t_h2d0 = ggml_time_us();
    const size_t n_h2d = llama_state_seq_set_data_ext(ctx, data.data(), data.size(), 1, 0);
    const double t_h2d = (ggml_time_us() - t_h2d0) / 1e6;

    const double mib = size / 1024.0 / 1024.0;

    fprintf(stderr, "[t32-h2d] fill: %d tokens in %.1f s (%.1f t/s)\n", n_tokens, t_fill, n_tokens / t_fill);
    fprintf(stderr, "[t32-h2d] size: %.1f MiB\n", mib);
    fprintf(stderr, "[t32-h2d] D2H:  %.2f s (%.2f GiB/s) [%zu bytes]\n", t_d2h, mib / 1024.0 / t_d2h, n_d2h);
    fprintf(stderr, "[t32-h2d] H2D:  %.2f s (%.2f GiB/s) [%zu bytes]\n", t_h2d, mib / 1024.0 / t_h2d, n_h2d);

    return (n_d2h == size && n_h2d == size) ? 0 : 1;
}

static int n_fail = 0;

static void check(bool cond, const char * what) {
    fprintf(stderr, "[t32] %-72s %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        n_fail++;
    }
}

static std::vector<llama_token> generate(llama_context * ctx, llama_seq_id seq_id, llama_token first, llama_pos pos0, int n, std::vector<std::vector<float>> * logits_out = nullptr) {
    std::vector<llama_token> res;

    llama_sampler * smpl = llama_sampler_init_greedy();
    llama_batch batch = llama_batch_init(1, 0, 1);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));

    llama_token cur = first;
    llama_pos   pos = pos0;

    for (int i = 0; i < n; ++i) {
        batch.n_tokens = 0;
        common_batch_add(batch, cur, pos++, { seq_id }, true);

        if (llama_decode(ctx, batch) != 0) {
            res.clear();
            break;
        }

        if (logits_out) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            logits_out->emplace_back(lg, lg + n_vocab);
        }

        cur = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, cur);
        res.push_back(cur);
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    return res;
}

static float logits_max_diff(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
    float res = 0.0f;

    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        for (size_t j = 0; j < a[i].size() && j < b[i].size(); ++j) {
            res = std::max(res, std::fabs(a[i][j] - b[i][j]));
        }
    }

    return res;
}

static void check_skip(const char * what) {
    fprintf(stderr, "[t32] %-72s SKIP (unified: cell layout differs)\n", what);
}

// exact_layout: false when several sequences share one stream (unified), where the same KV content
// can land in different cells and flash-attention reduction order alone can flip a near-tie token
static int run_correctness(llama_model * model, llama_context * ctx, bool exact_layout) {
    const int L = 64;

    if (fill_context(ctx, L, L) != 0) {
        return 1;
    }

    const bool hybrid = llama_model_is_hybrid(model);

    const size_t size_full  = llama_state_seq_get_size_ext(ctx, 0, 0);
    const size_t size_part  = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const size_t size_range = llama_state_seq_get_size_range_ext(ctx, 0, 0, L, 0);

    fprintf(stderr, "[t32] hybrid=%d full=%zu partial=%zu range=%zu\n", hybrid, size_full, size_part, size_range);

    check(size_full > 0, "full state size > 0");
    check(size_range > 0, "range state size > 0");

    if (hybrid) {
        check(size_range == size_full - size_part + 8, "hybrid: range(0,L) size == full - partial + 8");
    } else {
        check(size_range == size_full, "attn: range(0,L) size == full size");

        std::vector<uint8_t> data_full(size_full);
        std::vector<uint8_t> data_range(size_range);

        const size_t n_full  = llama_state_seq_get_data_ext(ctx, data_full.data(), data_full.size(), 0, 0);
        const size_t n_range = llama_state_seq_get_data_range_ext(ctx, data_range.data(), data_range.size(), 0, 0, L, 0);

        check(n_full == size_full && n_range == size_range, "payload reads returned the expected sizes");
        check(n_range == n_full && memcmp(data_full.data(), data_range.data(), size_full) == 0, "attn: range(0,L) payload == full payload");
    }

    const size_t size_half = llama_state_seq_get_size_range_ext(ctx, 0, 0, L/2, 0);
    check(size_half > 8 && size_half < size_range, "range(0,L/2) size is between header and full range");

    // chunked restore must match a full restore
    const int mid = L / 2;

    const size_t size_c0 = llama_state_seq_get_size_range_ext(ctx, 0, 0,   mid, 0);
    const size_t size_c1 = llama_state_seq_get_size_range_ext(ctx, 0, mid, L,   0);

    std::vector<uint8_t> data_c0(size_c0);
    std::vector<uint8_t> data_c1(size_c1);

    check(llama_state_seq_get_data_range_ext(ctx, data_c0.data(), size_c0, 0, 0,   mid, 0) == size_c0, "read chunk 0");
    check(llama_state_seq_get_data_range_ext(ctx, data_c1.data(), size_c1, 0, mid, L,   0) == size_c1, "read chunk 1");

    std::vector<uint8_t> data_part;
    if (hybrid) {
        data_part.resize(size_part);
        check(llama_state_seq_get_data_ext(ctx, data_part.data(), data_part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size_part, "read recurrent state");
    }

    std::vector<uint8_t> data_full(size_full);
    check(llama_state_seq_get_data_ext(ctx, data_full.data(), data_full.size(), 0, 0) == size_full, "read full state");

    // baseline: seq 0 has not been restored, generate before touching the other sequences
    const llama_token first = 42;
    std::vector<std::vector<float>> lg0, lg1, lg2;
    const auto gen0 = generate(ctx, 0, first, L, 8, &lg0);

    check(!gen0.empty(), "baseline generation (seq 0, no restore)");

    if (hybrid) {
        check(llama_state_seq_set_data_ext(ctx, data_part.data(), data_part.size(), 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size_part, "restore recurrent state into seq 1");
    }

    check(llama_state_seq_set_data_range_ext(ctx, data_c0.data(), size_c0, 1, false, 0) == size_c0, "restore chunk 0 (append=false)");
    check(llama_state_seq_set_data_range_ext(ctx, data_c1.data(), size_c1, 1, true,  0) == size_c1, "restore chunk 1 (append=true)");

    check(llama_state_seq_set_data_ext(ctx, data_full.data(), data_full.size(), 2, 0) == size_full, "restore full state into seq 2");

    const auto gen1 = generate(ctx, 1, first, L, 8, &lg1);
    const auto gen2 = generate(ctx, 2, first, L, 8, &lg2);

    fprintf(stderr, "[t32] logit max|diff|: seq1-seq2=%.6f seq1-seq0=%.6f seq2-seq0=%.6f\n",
            logits_max_diff(lg1, lg2), logits_max_diff(lg1, lg0), logits_max_diff(lg2, lg0));

    if (exact_layout) {
        check(!gen1.empty() && gen1 == gen0, "chunked restore generates the same tokens as no restore");
        check(!gen2.empty() && gen2 == gen0, "full restore generates the same tokens as no restore");
        check(!gen1.empty() && gen1 == gen2, "chunked restore generates the same tokens as a full restore");
    } else {
        check_skip("chunked restore generates the same tokens as no restore");
        check_skip("full restore generates the same tokens as no restore");
        check_skip("chunked restore generates the same tokens as a full restore");
    }

    const size_t n_overlap = llama_state_seq_set_data_range_ext(ctx, data_c0.data(), size_c0, 1, true, 0);
    check(n_overlap == 0, "append with overlapping positions is rejected");

    // gen1/gen2 already decoded past L, so continue both: seq 1 went through the rejected append, seq 2 did not
    const llama_token next = gen1.empty() ? first : gen1.back();

    const auto gen1b = generate(ctx, 1, next, L + 8, 8);
    const auto gen2b = generate(ctx, 2, next, L + 8, 8);

    if (exact_layout) {
        check(!gen1b.empty() && gen1b == gen2b, "rejected append left the existing state intact");
    } else {
        check_skip("rejected append left the existing state intact");
    }

    // a truncated blob must fail after cell allocation and leave no cells behind
    llama_memory_seq_rm(llama_get_memory(ctx), 2, -1, -1);

    check(llama_state_seq_set_data_range_ext(ctx, data_c0.data(), size_c0, 2, false, 0) == size_c0, "rebuild chunk 0 on seq 2");

    std::vector<uint8_t> data_c1_trunc = data_c1;
    data_c1_trunc.resize(data_c1_trunc.size() - 16);

    check(llama_state_seq_set_data_range_ext(ctx, data_c1_trunc.data(), data_c1_trunc.size(), 2, true, 0) == 0, "truncated append blob is rejected");

    check(llama_state_seq_set_data_range_ext(ctx, data_c1.data(), size_c1, 2, true, 0) == size_c1, "destination is clean after the failed append");

    // the range API does not carry the recurrent state, so put seq 2 back on par with seq 1 before the comparison
    if (hybrid) {
        check(llama_state_seq_set_data_ext(ctx, data_part.data(), data_part.size(), 2, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size_part, "restore recurrent state into seq 2 after the failed append");
    }

    const auto gen2c = generate(ctx, 2, first, L, 8);
    if (exact_layout) {
        check(!gen2c.empty() && gen2c == gen1, "destination still generates like seq 1 after the failed append");
    } else {
        check_skip("destination still generates like seq 1 after the failed append");
    }

    check(llama_state_seq_get_size_range_ext(ctx, 0, L, L, 0) == 0, "range with p1 <= p0 is rejected");
    check(llama_state_seq_get_size_range_ext(ctx, 0, -1, L, 0) == 0, "range with p0 < 0 is rejected");
    check(llama_state_seq_get_size_range_ext(ctx, 0, 0, L, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == 0, "range with unsupported flags is rejected");

    return n_fail == 0 ? 0 : 1;
}

// n_parallel == 1: only one sequence exists, so compare restore paths against a no-restore baseline
static int run_correctness_1seq(llama_model * model, llama_context * ctx) {
    const int L = 64;

    if (fill_context(ctx, L, L) != 0) {
        return 1;
    }

    const bool hybrid = llama_model_is_hybrid(model);

    const size_t size_full  = llama_state_seq_get_size_ext(ctx, 0, 0);
    const size_t size_part  = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const size_t size_range = llama_state_seq_get_size_range_ext(ctx, 0, 0, L, 0);

    fprintf(stderr, "[t32-1seq] hybrid=%d full=%zu partial=%zu range=%zu\n", hybrid, size_full, size_part, size_range);

    check(size_full > 0, "full state size > 0");
    check(size_range > 0, "range state size > 0");

    if (hybrid) {
        check(size_range == size_full - size_part + 8, "hybrid: range(0,L) size == full - partial + 8");
    } else {
        check(size_range == size_full, "attn: range(0,L) size == full size");

        std::vector<uint8_t> data_full_ref(size_full);
        std::vector<uint8_t> data_range_ref(size_range);

        const size_t n_full  = llama_state_seq_get_data_ext(ctx, data_full_ref.data(), data_full_ref.size(), 0, 0);
        const size_t n_range = llama_state_seq_get_data_range_ext(ctx, data_range_ref.data(), data_range_ref.size(), 0, 0, L, 0);

        check(n_full == size_full && n_range == size_range, "payload reads returned the expected sizes");
        check(n_range == n_full && memcmp(data_full_ref.data(), data_range_ref.data(), size_full) == 0, "attn: range(0,L) payload == full payload");
    }

    const int mid = L / 2;

    const size_t size_c0 = llama_state_seq_get_size_range_ext(ctx, 0, 0,   mid, 0);
    const size_t size_c1 = llama_state_seq_get_size_range_ext(ctx, 0, mid, L,   0);

    std::vector<uint8_t> data_c0(size_c0);
    std::vector<uint8_t> data_c1(size_c1);

    check(llama_state_seq_get_data_range_ext(ctx, data_c0.data(), size_c0, 0, 0,   mid, 0) == size_c0, "read chunk 0");
    check(llama_state_seq_get_data_range_ext(ctx, data_c1.data(), size_c1, 0, mid, L,   0) == size_c1, "read chunk 1");

    std::vector<uint8_t> data_full(size_full);
    check(llama_state_seq_get_data_ext(ctx, data_full.data(), data_full.size(), 0, 0) == size_full, "read full state");

    std::vector<uint8_t> data_part;
    if (hybrid) {
        data_part.resize(size_part);
        check(llama_state_seq_get_data_ext(ctx, data_part.data(), data_part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size_part, "read recurrent state");
    }

    // baseline: no restore at all
    const llama_token first = 42;
    std::vector<std::vector<float>> lg0, lg_chunked, lg_full;
    const auto gen0 = generate(ctx, 0, first, L, 8, &lg0);

    check(!gen0.empty(), "baseline generation");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    if (hybrid) {
        check(llama_state_seq_set_data_ext(ctx, data_part.data(), data_part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size_part, "restore recurrent state (chunked)");
    }

    check(llama_state_seq_set_data_range_ext(ctx, data_c0.data(), size_c0, 0, false, 0) == size_c0, "restore chunk 0 (append=false)");
    check(llama_state_seq_set_data_range_ext(ctx, data_c1.data(), size_c1, 0, true,  0) == size_c1, "restore chunk 1 (append=true)");

    const auto gen_chunked = generate(ctx, 0, first, L, 8, &lg_chunked);

    check(!gen_chunked.empty() && gen_chunked == gen0, "chunked restore generates the same tokens as no restore");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    check(llama_state_seq_set_data_ext(ctx, data_full.data(), data_full.size(), 0, 0) == size_full, "restore full state");

    const auto gen_full = generate(ctx, 0, first, L, 8, &lg_full);

    fprintf(stderr, "[t32-1seq] logit max|diff|: chunked-base=%.6f full-base=%.6f\n",
            logits_max_diff(lg_chunked, lg0), logits_max_diff(lg_full, lg0));

    check(!gen_full.empty() && gen_full == gen0, "full restore generates the same tokens as no restore");
    check(gen_full == gen_chunked, "full restore and chunked restore agree");

    check(llama_state_seq_get_size_range_ext(ctx, 0, L, L, 0) == 0, "range with p1 <= p0 is rejected");
    check(llama_state_seq_get_size_range_ext(ctx, 0, -1, L, 0) == 0, "range with p0 < 0 is rejected");
    check(llama_state_seq_get_size_range_ext(ctx, 0, 0, L, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == 0, "range with unsupported flags is rejected");

    return n_fail == 0 ? 0 : 1;
}

static int run_range_bench(llama_context * ctx, int n_tokens, int chunk) {
    const int64_t t_fill0 = ggml_time_us();

    if (fill_context(ctx, n_tokens) != 0) {
        return 1;
    }

    const double t_fill = (ggml_time_us() - t_fill0) / 1e6;

    const int n_chunks = (n_tokens + chunk - 1) / chunk;

    std::vector<std::vector<uint8_t>> data(n_chunks);

    size_t total = 0;

    const int64_t t_write0 = ggml_time_us();
    for (int c = 0; c < n_chunks; ++c) {
        const int p0 = c * chunk;
        const int p1 = std::min(n_tokens, p0 + chunk);

        const size_t size = llama_state_seq_get_size_range_ext(ctx, 0, p0, p1, 0);
        data[c].resize(size);

        if (llama_state_seq_get_data_range_ext(ctx, data[c].data(), size, 0, p0, p1, 0) != size) {
            fprintf(stderr, "chunk %d write failed\n", c);
            return 1;
        }

        total += size;
    }
    const double t_write = (ggml_time_us() - t_write0) / 1e6;

    llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);

    const int64_t t_read0 = ggml_time_us();
    for (int c = 0; c < n_chunks; ++c) {
        if (llama_state_seq_set_data_range_ext(ctx, data[c].data(), data[c].size(), 1, c > 0, 0) != data[c].size()) {
            fprintf(stderr, "chunk %d read failed\n", c);
            return 1;
        }
    }
    const double t_read = (ggml_time_us() - t_read0) / 1e6;

    const double mib = total / 1024.0 / 1024.0;

    fprintf(stderr, "[t32-bench] n=%d chunk=%d chunks=%d fill=%.1fs total=%.1f MiB\n", n_tokens, chunk, n_chunks, t_fill, mib);
    fprintf(stderr, "[t32-bench] write: %.2f s (%.1f MiB/s, %.2f ms/chunk)\n", t_write, mib / t_write, 1e3 * t_write / n_chunks);
    fprintf(stderr, "[t32-bench] read:  %.2f s (%.1f MiB/s, %.2f ms/chunk)\n", t_read,  mib / t_read,  1e3 * t_read  / n_chunks);

    return 0;
}

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 1234;
    params.n_parallel = 3;
    params.n_ctx = 512;

    std::string mode = "correctness";
    int n_tokens = 32000;
    int chunk    = 512;
    bool kv_unified = false;

    // extract our own options before handing the rest to the common arg parser
    std::vector<char *> args;
    args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            continue;
        }
        if (arg == "--n" && i + 1 < argc) {
            n_tokens = atoi(argv[++i]);
            continue;
        }
        if (arg == "--chunk" && i + 1 < argc) {
            chunk = atoi(argv[++i]);
            continue;
        }
        // the common parser only exposes -kvu to some example types, so handle it here
        if (arg == "-kvu" || arg == "--kv-unified") {
            kv_unified = true;
            continue;
        }

        args.push_back(argv[i]);
    }

    if (kv_unified) {
        params.kv_unified = true;
    }

    common_init();

    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    if (mode == "h2d") {
        return run_h2d(ctx, n_tokens);
    }
    if (mode == "correctness") {
        if (params.n_parallel == 1) {
            return run_correctness_1seq(model, ctx);
        }

        return run_correctness(model, ctx, !(params.kv_unified && params.n_parallel > 1));
    }
    if (mode == "range-bench") {
        return run_range_bench(ctx, n_tokens, chunk);
    }

    print_usage(argv[0]);
    return 1;
}
