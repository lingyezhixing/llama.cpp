// Harness for the KV tree module
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "server-kv-tree.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool cond, const char * what) {
    fprintf(stderr, "[t32-tree] %-72s %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        n_fail++;
    }
}

static void check_eq(long long got, long long want, const char * what) {
    fprintf(stderr, "[t32-tree] %-72s %s (got %lld, want %lld)\n", what, got == want ? "PASS" : "FAIL", got, want);
    if (got != want) {
        n_fail++;
    }
}

// fake io: payload bytes are a deterministic function of the position, so that
// identical prefixes produce identical blobs (models "attention KV is a function of the prefix")
struct kv_tree_io_fake : public kv_tree_io {
    llama_pos max_pos = -1;
    int64_t set_range_bytes = 0;
    int64_t set_range_calls = 0;
    int64_t set_partial_calls = 0;
    std::vector<std::pair<llama_pos, llama_pos>> ranges;
    std::vector<std::pair<llama_pos, llama_pos>> trims;

    static std::vector<uint8_t> pattern(llama_pos p0, size_t n) {
        std::vector<uint8_t> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = (uint8_t) (p0 * 31 + i * 7);
        }
        return v;
    }

    bool get_range(llama_pos p0, llama_pos p1, std::vector<uint8_t> & out) override {
        ranges.emplace_back(p0, p1);
        out = pattern(p0, (size_t) (p1 - p0) * 8);
        return true;
    }
    bool set_range(const uint8_t * data, size_t size, bool append) override {
        (void) data;
        (void) append;
        set_range_calls++;
        set_range_bytes += (int64_t) size;
        return true;
    }
    bool get_partial(std::vector<uint8_t> & out) override {
        out = pattern(0, 64);
        return true;
    }
    bool set_partial(const uint8_t * data, size_t size) override {
        (void) data;
        (void) size;
        set_partial_calls++;
        return true;
    }
    bool seq_rm(llama_pos p0, llama_pos p1) override {
        trims.emplace_back(p0, p1);
        if (p0 == -1) {
            max_pos = -1;
        } else if (p1 == -1) {
            max_pos = p0 - 1;
        }
        return true;
    }
    llama_pos pos_max() override {
        return max_pos;
    }
};

static std::vector<llama_token> make_tokens(int n, int salt) {
    std::vector<llama_token> t(n);
    for (int i = 0; i < n; ++i) {
        t[i] = 10 + (i * 7 + salt) % 1000;
    }
    return t;
}

static void run_logic_evict() {
    fprintf(stderr, "[t32-tree] logic: eviction\n");

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 20 * 1024;   // 5 chunks worth of fake payloads
    cfg.disk_dir    = "";          // no disk tier: eviction only

    kv_tree tree(cfg);

    const auto tok_a = make_tokens(1536, 0);
    const auto tok_b = make_tokens(1536, 1);

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_a, {}, {}), "evict: park A");
        check_eq(tree.stats().blocks_ram, 3, "evict: A has 3 blocks");
    }

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_b, {}, {}), "evict: park B forces eviction");
        check(tree.stats().evicted_blocks > 0, "evict: blocks were evicted");
        check_eq(tree.stats().park_refused, 0, "evict: park still succeeded");
    }

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        const auto r = tree.restore(io, nullptr, tok_a, {});
        check_eq(r.C, -1, "evict: A degrades to a full prefill (visible)");
    }

    {
        kv_tree_config cfg2 = cfg;
        cfg2.ram_limit = 4 * 1024;

        kv_tree tree2(cfg2);
        kv_tree_io_fake io;
        io.max_pos = 1535;

        check(!tree2.park(io, nullptr, tok_a, {}, {}), "evict: park refused with a tiny budget");
        check_eq(tree2.stats().park_refused, 1, "evict: refusal counted");
        check_eq(tree2.stats().evict_refused, 1, "evict: refusal is visible");
        check_eq(tree2.stats().blocks_ram, 0, "evict: nothing was left behind");
    }
}

static void run_logic_drop() {
    fprintf(stderr, "[t32-tree] logic: drop_seq\n");

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 1 << 20;
    cfg.disk_dir    = "";

    kv_tree tree(cfg);

    const auto tok   = make_tokens(1536, 0);
    auto       tok_x = tok;
    for (int i = 0; i < 512; ++i) {
        tok_x.push_back((llama_token) (10 + (i * 7 + 99) % 1000));
    }
    auto tok_mid   = std::vector<llama_token>(tok.begin(), tok.begin() + 1400);
    auto tok_other = make_tokens(1536, 1);

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok, {}, {}), "drop: park A");
    }

    check(!tree.drop_seq(tok_other, {}),                "drop: unrelated tokens are a no-op");
    check(tree.drop_seq(tok_mid, {}),                   "drop: mid-tail tokens drop the sequence");
    check_eq(tree.stats().evicted_seqs, 1,              "drop: sequence was dropped");
    check_eq(tree.stats().blocks_ram, 0,                "drop: its blocks were released");
    check(!tree.drop_seq(tok, {}),                      "drop: second drop is a no-op");

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok, {}, {}), "drop: re-park A");
    }
    check(tree.drop_seq(tok_x, {}),                     "drop: superstring drops the sequence");
    check_eq(tree.stats().evicted_seqs, 2,              "drop: sequence was dropped again");
}

static void run_logic_stream() {
    fprintf(stderr, "[t32-tree] logic: streaming restore\n");

    const auto dir = std::filesystem::temp_directory_path() / "t32-tree-logic-stream";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 4096;      // fits one fake block (4096 B), nothing more
    cfg.disk_limit  = 1 << 20;
    cfg.disk_dir    = dir.string();

    kv_tree tree(cfg);

    const auto tok = make_tokens(1536, 0);

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok, {}, {}),                      "stream: park with spill");
        check_eq(tree.stats().blocks_disk, 3,                           "stream: all blocks spilled to disk");
        check((size_t) tree.stats().bytes_ram <= cfg.ram_limit,         "stream: park respects the ram budget");
    }

    {
        kv_tree_io_fake io;
        const auto r = tree.restore(io, nullptr, tok, {});
        check_eq(r.C, 1536,                                             "stream: restore hits the tip");
        check_eq(r.heal, -1,                                            "stream: no heal needed for an exact tip");
        check_eq(io.set_range_bytes, 3 * 4096,                          "stream: all blocks were written to the io");
        check_eq(io.set_partial_calls, 1,                               "stream: the anchor was written");
        check((size_t) tree.stats().bytes_ram <= cfg.ram_limit,         "stream: restore respects the ram budget");
        check_eq(tree.stats().bytes_load, 3 * 4096,                     "stream: bytes_load counts payloads moved from disk");
        check_eq(tree.stats().blocks_disk, 3,                           "stream: streamed blocks stay on disk");
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

static void run_logic_capture() {
    fprintf(stderr, "[t32-tree] logic: capture inside a stored chunk\n");

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 1 << 20;
    cfg.disk_dir    = "";

    kv_tree tree(cfg);

    const auto tok = make_tokens(1536, 0);

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok, {}, {}), "capture: park");
        check_eq(tree.stats().anchors_added, 1, "capture: tip anchor only");
    }

    // the io state ends at the capture point, so the tokens are a truncated prefix
    const std::vector<llama_token> pre(tok.begin(), tok.begin() + 1400);

    {
        kv_tree_io_fake io;
        io.max_pos = 1399;
        check(tree.capture_anchor(io, nullptr, pre, {}, 1400), "capture: heal inside the tail block");
        check_eq(tree.stats().anchors_added, 2, "capture: the anchor was stored");
    }

    {
        // request forks after the anchor: the restore must use it
        auto fork = std::vector<llama_token>(tok.begin(), tok.begin() + 1400);
        for (int i = 0; i < 100; ++i) {
            fork.push_back((llama_token) (900 + i));
        }

        kv_tree_io_fake io;
        const auto r = tree.restore(io, nullptr, fork, {});
        check_eq(r.C, 1400, "capture: fork restores at the stored anchor");
        check_eq(r.heal, -1, "capture: no heal needed after the restore");
        check_eq(io.set_partial_calls, 1, "capture: the anchor state was written back");
    }

    {
        // no stored block covers this position
        auto ext = tok;
        for (int i = 0; i < 164; ++i) {
            ext.push_back((llama_token) (800 + i));
        }

        kv_tree_io_fake io;
        io.max_pos = 1699;
        check(!tree.capture_anchor(io, nullptr, ext, {}, 1700), "capture: refused with no stored block");
        check_eq(tree.stats().anchors_added, 2, "capture: nothing stored on refusal");
    }
}

static void run_logic_fixes() {
    fprintf(stderr, "[t32-tree] logic: fixes\n");

    // disk_errors must count one error per failed write, not two
    {
        const auto file = std::filesystem::temp_directory_path() / "t32-tree-diskerr";
        std::error_code ec;
        std::filesystem::remove(file, ec);
        {
            std::FILE * f = fopen(file.string().c_str(), "wb");
            if (f != nullptr) {
                fputc('x', f);
                fclose(f);
            }
        }

        kv_tree_config cfg;
        cfg.ram_limit  = 0;           // force demotion
        cfg.disk_limit = 1ull << 20;
        cfg.disk_dir   = file.string(); // a file, not a directory: every write fails

        kv_tree tree(cfg);

        kv_tree_io_fake io;
        io.max_pos = 1023;
        check(!tree.park(io, nullptr, make_tokens(1024, 0), {}, {}), "fixes: park refused when the disk tier fails");
        check_eq(tree.stats().disk_errors, 2, "fixes: one disk error per failed write (block + anchor)");

        std::filesystem::remove(file, ec);
    }

    // anchor spacing must be scoped to the chain, not global
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 4096;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);

        const auto tok_a = make_tokens(512, 0);
        const auto tok_b = make_tokens(2048, 1);

        {
            kv_tree_io_fake io;
            io.max_pos = 511;
            check(tree.park(io, nullptr, tok_a, {}, {}), "fixes: park chain A (tip anchor at 512)");
        }

        {
            kv_tree_anchor_in ck;
            ck.tok      = 1024;
            ck.data_tgt = kv_tree_io_fake::pattern(0, 64);

            kv_tree_io_fake io;
            io.max_pos = 2047;
            check(tree.park(io, nullptr, tok_b, {}, { ck }), "fixes: park chain B with a candidate at 1024");
        }

        // chain B now has anchors at 1024 and 2048; chain A has one at 512
        check_eq(tree.stats().anchors_added, 3, "fixes: the cross-chain anchor does not suppress the candidate");
    }

    // capture spacing must be chain-scoped too, and same-chain spacing must still apply
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 4096;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);

        const auto tok_a = make_tokens(512, 0);
        const auto tok_b = make_tokens(2048, 1);

        {
            kv_tree_io_fake io;
            io.max_pos = 511;
            check(tree.park(io, nullptr, tok_a, {}, {}), "fixes: park chain A");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 2047;
            check(tree.park(io, nullptr, tok_b, {}, {}), "fixes: park chain B");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 1535;
            check(tree.capture_anchor(io, nullptr, tok_b, {}, 1536), "fixes: cross-chain anchor does not block the capture");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 2559;
            check(!tree.capture_anchor(io, nullptr, tok_b, {}, 2560), "fixes: same-chain spacing still refuses");
        }
    }

    // check that stats_line exposes the park/anchor/disk counters
    {
        kv_tree_config cfg;
        cfg.ram_limit = 1ull << 20;

        kv_tree tree(cfg);

        kv_tree_io_fake io;
        io.max_pos = 1023;
        check(tree.park(io, nullptr, make_tokens(1024, 0), {}, {}), "fixes: park for the stats line");

        const std::string s = tree.stats_line();
        check(s.find("parks=1") != std::string::npos, "fixes: stats line has parks");
        check(s.find("anchors=1") != std::string::npos, "fixes: stats line has anchors");
        check(s.find("disk_err=0") != std::string::npos, "fixes: stats line has disk_err");
    }
}

static void run_logic_fork() {
    fprintf(stderr, "[t32-tree] logic: fork spacing and miss heal\n");

    // fork anchors use fork_step, guesses keep anchor_step
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 32768;
        cfg.fork_step   = 8192;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);
        const auto tok = make_tokens(10240, 0);

        {
            kv_tree_io_fake io;
            io.max_pos = 10239;
            check(tree.park(io, nullptr, tok, {}, {}), "fork: park the chain");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 1023;
            check(tree.capture_anchor(io, nullptr, tok, {}, 1024), "fork: the first capture is free");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 5119;
            check(!tree.capture_anchor(io, nullptr, tok, {}, 5120), "fork: within fork_step is refused");
        }
        check_eq(tree.stats().anchors_skipped_step, 1, "fork: the spacing skip is counted");
        {
            kv_tree_io_fake io;
            io.max_pos = 9215;
            check(tree.capture_anchor(io, nullptr, tok, {}, 9216), "fork: at fork_step it is stored");
        }
        check_eq(tree.stats().anchors_added, 3, "fork: tip + two fork anchors");
        check(tree.stats_line().find("step_skips=1") != std::string::npos, "fork: stats line has step_skips");

        // re-park a prefix: its tip anchor already exists, so only the skip counter changes
        const std::vector<llama_token> prefix(tok.begin(), tok.begin() + 9216);
        const int64_t step_skips_before = tree.stats().anchors_skipped_step;

        kv_tree_anchor_in ck;
        ck.tok      = 8192;
        ck.data_tgt = kv_tree_io_fake::pattern(0, 64);

        {
            kv_tree_io_fake io;
            io.max_pos = 9215;
            tree.park(io, nullptr, prefix, {}, { ck });
        }

        check_eq(tree.stats().anchors_skipped_step - step_skips_before, 1, "fork: the park candidate skip is counted");
    }

    // guesses never suppress a fork anchor
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 32768;
        cfg.fork_step   = 8192;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);
        const auto tok = make_tokens(10240, 0);

        kv_tree_anchor_in guess;
        guess.tok      = 1024;
        guess.data_tgt = kv_tree_io_fake::pattern(0, 64);

        {
            kv_tree_io_fake io;
            io.max_pos = 10239;
            check(tree.park(io, nullptr, tok, {}, { guess }), "fork: park with a guess at 1024");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 5119;
            check(tree.capture_anchor(io, nullptr, tok, {}, 5120), "fork: the guess does not suppress the fork");
        }
        check_eq(tree.stats().anchors_skipped_step, 0, "fork: no spacing skip against the guess");
        {
            kv_tree_io_fake io;
            io.max_pos = 9215;
            check(!tree.capture_anchor(io, nullptr, tok, {}, 9216), "fork: another fork anchor still suppresses (8192)");
        }
        check_eq(tree.stats().anchors_skipped_step, 1, "fork: the fork-vs-fork skip is counted");
    }

    // a restore miss reports the divergence point
    {
        kv_tree_config cfg;
        cfg.chunk       = 512;
        cfg.anchor_step = 32768;
        cfg.fork_step   = 8192;
        cfg.ram_limit   = 1ull << 20;

        kv_tree tree(cfg);

        const auto tok_a = make_tokens(4096, 0);
        auto tok_b = make_tokens(4096, 1);
        for (int i = 0; i < 2048; ++i) {
            tok_b[i] = tok_a[i];
        }

        {
            kv_tree_io_fake io;
            io.max_pos = 4095;
            check(tree.park(io, nullptr, tok_a, {}, {}), "fork: park A");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = 4095;
            const kv_tree_restore r = tree.restore(io, nullptr, tok_b, {});
            check_eq(r.C, -1, "fork: B misses (no anchor on the shared path)");
            check_eq(r.heal, 2048, "fork: the miss reports the divergence point");
        }
    }
}

static void run_logic_wipe() {
    fprintf(stderr, "[t32-tree] logic: startup wipe\n");

    const auto dir = std::filesystem::temp_directory_path() / "t32-tree-wipe";

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir / "blocks", ec);
    std::filesystem::create_directories(dir / "anchors", ec);

    auto touch = [](const std::filesystem::path & p) {
        std::FILE * f = fopen(p.string().c_str(), "wb");
        if (f != nullptr) {
            fputc('x', f);
            fclose(f);
        }
    };

    touch(dir / "blocks" / "aa.bin");
    touch(dir / "blocks" / "bb.bin.tmp");
    touch(dir / "anchors" / "cc.bin");

    {
        kv_tree_config cfg;
        cfg.disk_dir = dir.string();

        kv_tree tree(cfg);
    }

    check(!std::filesystem::exists(dir / "blocks" / "aa.bin"), "wipe: block file removed");
    check(!std::filesystem::exists(dir / "blocks" / "bb.bin.tmp"), "wipe: tmp file removed");
    check(!std::filesystem::exists(dir / "anchors" / "cc.bin"), "wipe: anchor file removed");

    {
        kv_tree_config cfg;
        kv_tree tree(cfg);
    }

    std::filesystem::remove_all(dir, ec);
}

static void run_logic_media() {
    fprintf(stderr, "[t32-tree] logic: media\n");

    kv_tree_config cfg;
    cfg.chunk       = 512;
    cfg.anchor_step = 512;
    cfg.ram_limit   = 1 << 20;
    cfg.disk_dir    = "";

    // local copy of the position mapping: the harness must not depend on the module internals
    auto pos_at = [](const std::vector<kv_tree_media> & media, int64_t t) {
        int64_t p = t;
        for (const auto & m : media) {
            if (m.idx >= t) {
                break;
            }
            p += m.n_pos - m.n_tok;
        }
        return (llama_pos) p;
    };

    auto add_media = [](std::vector<llama_token> & tokens, std::vector<kv_tree_media> & media,
                        int64_t idx, uint64_t id, int32_t n_tok, int32_t n_pos) {
        kv_tree_media m;
        m.idx   = idx;
        m.id    = id;
        m.n_tok = n_tok;
        m.n_pos = n_pos;
        media.push_back(m);
        for (int32_t i = 0; i < n_tok; ++i) {
            tokens[(size_t) (idx + i)] = LLAMA_TOKEN_NULL;
        }
    };

    // identity, alignment and round trip: one chunk crossing the block boundary at 1024
    {
        kv_tree tree(cfg);

        auto tok = make_tokens(3000, 0);
        std::vector<kv_tree_media> media;
        add_media(tok, media, 1000, 0xAA, 1026, 32);

        const size_t    tail_a  = 2538;   // block starts: 0, 512, chunk end 2026, 2538
        const llama_pos pos_end = pos_at(media, 3000);

        {
            kv_tree_io_fake io;
            io.max_pos = pos_end - 1;
            check(tree.park(io, nullptr, tok, media, {}), "media: park with a straddling chunk");
            check_eq((long long) io.ranges.size(), 4, "media: the chunk does not add a split");
            check_eq(io.ranges[1].second, pos_at(media, 2026), "media: the straddled block ends at the chunk end");
            check_eq(io.ranges[3].second, pos_end, "media: the last range ends at pos_at(L)");

            const llama_pos c0 = 1000;
            const llama_pos c1 = pos_at(media, 2026);
            bool aligned = true;
            for (const auto & r : io.ranges) {
                if (!(r.first <= c0 || r.first >= c1) || !(r.second <= c0 || r.second >= c1)) {
                    aligned = false;
                }
            }
            check(aligned, "media: no range boundary falls inside the chunk span");
        }

        {
            const kv_tree_match m = tree.match(tok, media);
            check_eq((long long) m.n_part, (long long) (3000 - tail_a), "media: match reaches the request end");
            check_eq(m.deep, 3000, "media: match depth reaches the request end");
        }
        {
            auto bad = media;
            bad[0].id = 0xBB;
            const kv_tree_match m = tree.match(tok, bad);
            check(m.deep <= 1000, "media: a different id does not match past the chunk");
        }
        {
            auto bad = media;
            bad[0].n_tok = 1025;
            const kv_tree_match m = tree.match(tok, bad);
            check(m.deep <= 1000, "media: a different n_tok does not match past the chunk");
        }
        {
            auto bad = media;
            bad[0].n_pos = 33;
            const kv_tree_match m = tree.match(tok, bad);
            check(m.deep <= 1000, "media: a different n_pos does not match past the chunk");
        }

        {
            kv_tree_io_fake io;
            const kv_tree_restore r = tree.restore(io, nullptr, tok, media);
            check_eq(r.C, 3000, "media: restore lands on the tip anchor");
            check_eq(r.heal, -1, "media: no heal after a full match");
            check_eq((long long) r.anchors.size(), 1, "media: one anchor on the path");
            check_eq((long long) r.anchors[0].tok, 3000, "media: the anchor carries the token index");
            check_eq(r.anchors[0].pos, 2006, "media: the anchor carries the mapped position");
            check_eq(io.set_range_calls, 4, "media: all blocks were written to the io");
            check_eq(io.max_pos, pos_end - 1, "media: the trim used the mapped position");
            check_eq(tree.stats().tokens_reused, 3000, "media: token reuse is counted in tokens");
        }
    }

    // identity inside the tail block: the token walk passes, the media clamp stops at the chunk
    {
        kv_tree tree(cfg);

        auto tok = make_tokens(2048, 1);
        std::vector<kv_tree_media> media;
        add_media(tok, media, 2000, 0x11, 48, 8);

        kv_tree_io_fake io;
        io.max_pos = 2000;
        check(tree.park(io, nullptr, tok, media, {}), "media: park with a chunk in the tail block");

        {
            const kv_tree_match m = tree.match(tok, media);
            check_eq(m.deep, 2048, "media: the same tail identity matches to the end");
        }
        {
            auto bad = media;
            bad[0].id = 0x22;
            const kv_tree_match m = tree.match(tok, bad);
            check_eq((long long) m.n_part, 2000 - 1536, "media: a different tail id clamps to the chunk start");
            check_eq(m.deep, 2000, "media: the tail clamp is the divergence point");
        }
    }

    // malformed request: the token vector ends inside the chunk, match must clamp to the chunk start
    {
        kv_tree tree(cfg);

        auto tok = make_tokens(2048, 2);
        std::vector<kv_tree_media> media;
        add_media(tok, media, 2000, 0x11, 48, 8);

        kv_tree_io_fake io;
        io.max_pos = 2000;
        check(tree.park(io, nullptr, tok, media, {}), "media: park for the malformed request");

        const std::vector<llama_token> trunc(tok.begin(), tok.begin() + 2024);
        const kv_tree_match m = tree.match(trunc, media);
        check_eq((long long) m.n_part, 2000 - 1536, "media: a truncated request clamps to the chunk start");
        check_eq(m.deep, 2000, "media: the malformed request never verifies into the chunk");
    }

    // the park guard uses the last KV cell, not the last token
    {
        kv_tree tree(cfg);

        auto tok = make_tokens(3000, 3);
        std::vector<kv_tree_media> media;
        add_media(tok, media, 2000, 0x33, 1000, 16);

        {
            kv_tree_io_fake io;
            io.max_pos = 2000;   // all chunk cells carry the chunk start position
            check(tree.park(io, nullptr, tok, media, {}), "media: park guarded by the last KV cell");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = pos_at(media, 3000) - 1;
            check(!tree.park(io, nullptr, tok, media, {}), "media: park refused when the state ends past the chunk");
            check_eq(tree.stats().park_refused, 1, "media: the refusal is counted");
        }
    }

    // interleaved chains: the covering block of one chain is not the nearest position key of the other
    {
        kv_tree tree(cfg);

        const auto tok_b = make_tokens(3000, 7);   // text chain with 512-aligned position keys
        {
            kv_tree_io_fake io;
            io.max_pos = 2999;
            check(tree.park(io, nullptr, tok_b, {}, {}), "media: park the interleaved text chain");
        }

        auto tok_a = make_tokens(3000, 0);
        std::vector<kv_tree_media> media;
        add_media(tok_a, media, 1000, 0xAA, 1026, 32);

        kv_tree_anchor_in ck;
        ck.tok      = 2534;   // maps to pos 1540, past the text chain block that starts at 1536
        ck.data_tgt = kv_tree_io_fake::pattern(0, 64);

        {
            kv_tree_io_fake io;
            io.max_pos = pos_at(media, 3000) - 1;
            check(tree.park(io, nullptr, tok_a, media, { ck }), "media: park the media chain over the text chain");
            check_eq(tree.stats().anchors_added, 3, "media: the interleaved candidate is adopted");
        }

        {
            kv_tree_io_fake io;
            const kv_tree_restore r = tree.restore(io, nullptr, tok_a, media);
            check_eq(r.C, 3000, "media: interleaved restore reaches the tip");
            check_eq((long long) r.anchors.size(), 2, "media: both interleaved anchors are on the path");

            bool found = false;
            for (const auto & a : r.anchors) {
                if (a.tok == 2534 && a.pos == 1540) {
                    found = true;
                }
            }
            check(found, "media: the interleaved candidate anchor is usable");
        }
    }

    // retrieval: capture and restore at a media-ending boundary
    {
        kv_tree tree(cfg);

        auto tok = make_tokens(3000, 4);
        std::vector<kv_tree_media> media;
        add_media(tok, media, 1000, 0xAA, 1026, 32);

        {
            kv_tree_io_fake io;
            io.max_pos = pos_at(media, 3000) - 1;
            check(tree.park(io, nullptr, tok, media, {}), "media: park for the boundary capture");
        }
        {
            // past the 2500-token request but below it in positions: the anchor filter must use tokens
            kv_tree_io_fake io;
            io.max_pos = pos_at(media, 2530) - 1;
            check(tree.capture_anchor(io, nullptr, tok, media, 2530), "media: capture past the chunk");
        }
        {
            kv_tree_io_fake io;
            io.max_pos = pos_at(media, 1000);   // the chunk cells all carry the chunk start position
            check(tree.capture_anchor(io, nullptr, tok, media, 2026), "media: capture at the chunk end");
        }

        const std::vector<llama_token> req(tok.begin(), tok.begin() + 2500);

        kv_tree_io_fake io;
        const kv_tree_restore r = tree.restore(io, nullptr, req, media);

        check_eq(r.C, 2026, "media: restore uses the chunk-end anchor");
        check_eq((long long) r.anchors.size(), 1, "media: the chunk-end anchor is on the path");

        if (r.C >= 0 && !r.anchors.empty()) {
            check_eq((long long) r.anchors[0].tok, 2026, "media: the anchor keeps the token index");
            check_eq(r.anchors[0].pos, pos_at(media, 1000) + 1, "media: the anchor keeps the mapped position");

            bool trimmed = false;
            for (const auto & t : io.trims) {
                if (t.first == pos_at(media, r.C) && t.second == -1) {
                    trimmed = true;
                }
            }
            check(trimmed, "media: the trim used the mapped position");
        }
    }

    // retrieval: drop_seq is media-aware
    {
        kv_tree tree(cfg);

        auto tok = make_tokens(3000, 5);
        std::vector<kv_tree_media> media;
        add_media(tok, media, 1000, 0xAA, 1026, 32);

        {
            kv_tree_io_fake io;
            io.max_pos = pos_at(media, 3000) - 1;
            check(tree.park(io, nullptr, tok, media, {}), "media: park for the drop");
        }

        auto bad = media;
        bad[0].id = 0xBB;

        check(!tree.drop_seq(tok, bad),        "media: a different id does not drop the sequence");
        check(tree.drop_seq(tok, media),       "media: the same input drops the sequence");
        check_eq(tree.stats().evicted_seqs, 1, "media: the sequence was dropped");
        check(!tree.drop_seq(tok, media),      "media: the second drop is a no-op");
    }

    // leave_one must never select an anchor beyond the verified prefix
    {
        kv_tree_config cfg2 = cfg;
        cfg2.chunk       = 128;
        cfg2.anchor_step = 1;

        kv_tree tree(cfg2);

        auto tok = make_tokens(700, 6);

        kv_tree_anchor_in ck1;
        ck1.tok = 650;
        ck1.data_tgt = { 1, 2, 3 };

        kv_tree_anchor_in ck2;
        ck2.tok = 680;
        ck2.data_tgt = { 4, 5, 6 };

        {
            kv_tree_io_fake io;
            io.max_pos = 699;
            check(tree.park(io, nullptr, tok, {}, { ck1, ck2 }), "media: park for the leave_one check");
        }

        auto req = tok;
        req[650] = 99999;   // the request diverges inside the last block

        const kv_tree_match m = tree.match(req, {});
        check_eq(m.deep, 650, "media: leave_one match stops at the divergence");

        kv_tree_io_fake io;
        const kv_tree_restore r = tree.restore(io, nullptr, req, {}, true);
        check_eq(r.C, 650, "media: leave_one restore stays within the verified prefix");
    }
}

static int run_logic() {
    fprintf(stderr, "[t32-tree] mode logic\n");

    run_logic_evict();
    run_logic_drop();
    run_logic_stream();
    run_logic_capture();
    run_logic_fixes();
    run_logic_wipe();
    run_logic_fork();
    run_logic_media();

    kv_tree_config cfg;
    cfg.chunk = 512;
    cfg.ram_limit = 64ull << 20;

    kv_tree tree(cfg);

    const auto tok_a = make_tokens(1536, 0);   // 3 full chunks
    const auto tok_b = make_tokens(1024, 0);   // shares the first 2 chunks with tok_a, then diverges
    auto tok_b2 = tok_b;
    tok_b2.resize(1536);
    for (int i = 1024; i < 1536; ++i) {
        tok_b2[i] = 500 + i % 100;             // same as tok_a in [0, 1024), different tail
    }
    auto tok_c = make_tokens(1024, 0);         // shared prefix only, short tail
    tok_c.resize(1124);
    for (int i = 1024; i < 1124; ++i) {
        tok_c[i] = 700 + i % 50;
    }

    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_a, {}, {}), "park A (3 chunks)");
        check_eq(tree.stats().blocks_ram, 3, "blocks after A");
    }
    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_b2, {}, {}), "park B2 (shares 2 chunks)");
        check_eq(tree.stats().blocks_ram, 4, "blocks after B2 (dedup)");
    }
    {
        kv_tree_io_fake io;
        io.max_pos = 1535;
        check(tree.park(io, nullptr, tok_a, {}, {}), "park A again (identical)");
        check_eq(tree.stats().blocks_ram, 4, "blocks after A re-park (no duplicate)");
    }
    {
        kv_tree_io_fake io;
        io.max_pos = 1123;
        check(tree.park(io, nullptr, tok_c, {}, {}), "park C (partial tail block)");
        check_eq(tree.stats().blocks_ram, 5, "blocks after C (tail is a new block)");
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

static int prefill(llama_context * ctx, llama_seq_id seq, const std::vector<llama_token> & tokens, int from, int ubatch) {
    llama_batch batch = llama_batch_init(ubatch, 0, 1);

    for (int i = from; i < (int) tokens.size(); ) {
        batch.n_tokens = 0;

        const int n = std::min(ubatch, (int) tokens.size() - i);

        for (int j = 0; j < n; ++j) {
            common_batch_add(batch, tokens[i + j], i + j, { seq }, j == n - 1);
        }

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[t32-tree] decode failed at token %d\n", i);
            llama_batch_free(batch);
            return 1;
        }

        i += n;
    }

    llama_batch_free(batch);
    return 0;
}

static std::vector<llama_token> generate(llama_context * ctx, llama_seq_id seq_id, llama_token first, llama_pos pos0, int n) {
    std::vector<llama_token> res;

    llama_sampler * smpl = llama_sampler_init_greedy();
    llama_batch batch = llama_batch_init(1, 0, 1);

    llama_token cur = first;
    llama_pos   pos = pos0;

    for (int i = 0; i < n; ++i) {
        batch.n_tokens = 0;
        common_batch_add(batch, cur, pos++, { seq_id }, true);

        if (llama_decode(ctx, batch) != 0) {
            res.clear();
            break;
        }

        cur = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, cur);
        res.push_back(cur);
    }

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    return res;
}

static kv_tree_anchor_in capture_partial(llama_context * ctx, llama_seq_id seq, llama_pos pos) {
    kv_tree_anchor_in c;
    c.tok = pos;

    const size_t size = llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (size == 0) {
        return c;
    }

    c.data_tgt.resize(size);

    const size_t n = llama_state_seq_get_data_ext(ctx, c.data_tgt.data(), c.data_tgt.size(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (n != size) {
        c.data_tgt.clear();
    }

    return c;
}

static std::vector<llama_token> run_baseline(llama_context * ctx, const std::vector<llama_token> & tokens, llama_token first, int n_gen) {
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    if (prefill(ctx, 0, tokens, 0, 512) != 0) {
        return {};
    }

    return generate(ctx, 0, first, (llama_pos) tokens.size(), n_gen);
}

struct path_result {
    std::vector<llama_token> gen;
    kv_tree_restore res;
};

static path_result run_tree_path(llama_context * ctx, kv_tree & tree, kv_tree_io & io,
                                 const std::vector<llama_token> & tokens, llama_token first, int n_gen) {
    path_result r;

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

    r.res = tree.restore(io, nullptr, tokens, {});

    if (r.res.C >= 0 && prefill(ctx, 0, tokens, (int) r.res.C, 512) != 0) {
        return r;
    }

    r.gen = generate(ctx, 0, first, (llama_pos) tokens.size(), n_gen);
    return r;
}

static int scenario_tip(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] scenario: tip\n");

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tokens = make_tokens(1536, 0);

    if (prefill(ctx, 0, tokens, 0, 512) != 0) {
        return 1;
    }

    check(tree.park(io, nullptr, tokens, {}, {}), "tip: park");
    check_eq(tree.stats().anchors_added, 1, "tip: one tip anchor");

    const auto base = run_baseline(ctx, tokens, 42, 8);
    check(!base.empty(), "tip: baseline generation");

    const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
    check_eq(r.res.C, 1536, "tip: restore point is the tip");
    check_eq(r.res.heal, -1, "tip: no heal needed");
    check(r.gen == base, "tip: tokens match the baseline");

    // the sequence now ends at L + 8, so parking it again must be refused
    check(!tree.park(io, nullptr, tokens, {}, {}), "tip: park refused when the sequence ends past L");
    check_eq(tree.stats().park_refused, 1, "tip: refusal counted");

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

static std::vector<llama_token> make_fork_tokens(int n_shared, int n_tail, int salt) {
    std::vector<llama_token> t = make_tokens(n_shared, 0);
    for (int i = 0; i < n_tail; ++i) {
        t.push_back(10 + (i * 13 + salt) % 1000);
    }
    return t;
}

static int scenario_fork(llama_context * ctx, const kv_tree_config & cfg_in) {
    fprintf(stderr, "[t32-tree] scenario: fork\n");

    kv_tree_config cfg = cfg_in;
    cfg.anchor_step = 512;
    cfg.fork_step   = 512;   // allow the heal capture 512 tokens after the anchor at 512

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tok_a  = make_fork_tokens(1024, 1024, 0);
    const auto tok_b  = make_fork_tokens(1024, 1024, 1);
    const auto tok_a2 = make_fork_tokens(1024, 512,  2);

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, std::vector<llama_token>(tok_a.begin(), tok_a.begin() + 512), 0, 512);

    const kv_tree_anchor_in ck = capture_partial(ctx, 0, 512);

    prefill(ctx, 0, tok_a, 512, 512);
    check(tree.park(io, nullptr, tok_a, {}, { ck }), "fork: park A");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, tok_b, 0, 512);
    check(tree.park(io, nullptr, tok_b, {}, {}), "fork: park B");

    {
        const auto base = run_baseline(ctx, tok_a, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tok_a, 42, 8);
        check_eq(r.res.C, 2048, "fork: A restores at the tip");
        check(r.gen == base, "fork: A tokens match the baseline");
    }
    {
        const auto base = run_baseline(ctx, tok_b, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tok_b, 42, 8);
        check_eq(r.res.C, 2048, "fork: B restores at the tip");
        check(r.gen == base, "fork: B tokens match the baseline");
    }
    {
        const auto base = run_baseline(ctx, tok_a2, 42, 8);

        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);

        const auto r = tree.restore(io, nullptr, tok_a2, {});
        check_eq(r.C, 512, "fork: A' restores at the deepest usable anchor");
        check_eq(r.heal, 1024, "fork: A' asks for a heal at the fork point");

        if (r.C >= 0) {
            prefill(ctx, 0, std::vector<llama_token>(tok_a2.begin(), tok_a2.begin() + 1024), (int) r.C, 512);
        }

        check(tree.capture_anchor(io, nullptr, tok_a2, {}, 1024), "fork: heal capture at the fork point");

        prefill(ctx, 0, tok_a2, 1024, 512);

        const auto gen = generate(ctx, 0, 42, (llama_pos) tok_a2.size(), 8);
        check(gen == base, "fork: A' tokens match the baseline");

        const auto r2 = run_tree_path(ctx, tree, io, tok_a2, 42, 8);
        check_eq(r2.res.C, 1024, "heal: A' restores at the self-healed anchor");
        check_eq(r2.res.heal, -1, "heal: no second heal needed");
        check(r2.gen == base, "heal: tokens still match the baseline");
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

static int scenario_sparsify(llama_context * ctx, const kv_tree_config & cfg_in) {
    fprintf(stderr, "[t32-tree] scenario: sparsify\n");

    kv_tree_config cfg = cfg_in;
    cfg.anchor_step = 1024;

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tokens = make_tokens(2048, 0);

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 512), 0, 512);

    const kv_tree_anchor_in ck1 = capture_partial(ctx, 0, 512);

    prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 1024), 512, 512);

    const kv_tree_anchor_in ck2 = capture_partial(ctx, 0, 1024);

    prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 1536), 1024, 512);

    const kv_tree_anchor_in ck3 = capture_partial(ctx, 0, 1536);

    prefill(ctx, 0, tokens, 1536, 512);

    check(tree.park(io, nullptr, tokens, {}, { ck1, ck2, ck3 }), "sparsify: park");

    {
        const std::vector<llama_token> short1(tokens.begin(), tokens.begin() + 1024);
        const auto base = run_baseline(ctx, short1, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, short1, 42, 8);
        check_eq(r.res.C, 512, "sparsify: [0, 1024) restores at 512");
        check(r.gen == base, "sparsify: [0, 1024) tokens match");
    }
    {
        const std::vector<llama_token> short2(tokens.begin(), tokens.begin() + 1536);
        const auto base = run_baseline(ctx, short2, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, short2, 42, 8);
        check_eq(r.res.C, 1536, "sparsify: [0, 1536) restores at 1536");
        check(r.gen == base, "sparsify: [0, 1536) tokens match");
    }
    {
        const auto base = run_baseline(ctx, tokens, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
        check_eq(r.res.C, 2048, "sparsify: full length restores at the tip");
        check(r.gen == base, "sparsify: full length tokens match");
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

static int scenario_ssd(llama_context * ctx, const kv_tree_config & cfg_in) {
    fprintf(stderr, "[t32-tree] scenario: ssd\n");

    kv_tree_config cfg = cfg_in;
    cfg.ram_limit  = 64 << 10;    // force everything to disk
    cfg.disk_limit = 64 << 20;
    cfg.disk_dir   = (std::filesystem::temp_directory_path() / "t32-tree-test").string();

    std::error_code ec;
    std::filesystem::remove_all(cfg.disk_dir, ec);
    std::filesystem::create_directories(cfg.disk_dir + "/blocks", ec);
    std::filesystem::create_directories(cfg.disk_dir + "/anchors", ec);

    kv_tree_io_llama io(ctx, 0);

    const auto tokens = make_tokens(1536, 0);

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, tokens, 0, 512);

    // size the disk budget to the model: pure-attention PARTIAL_ONLY is the whole KV
    const size_t seq_bytes    = llama_state_seq_get_size_range_ext(ctx, 0, 0, 1536, 0);
    const size_t anchor_bytes = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    cfg.disk_limit = 4 * (seq_bytes + anchor_bytes);

    kv_tree tree(cfg);

    check(tree.park(io, nullptr, tokens, {}, {}), "ssd: park");
    check(tree.stats().blocks_disk > 0, "ssd: blocks demoted to disk");
    check(tree.stats().anchors_disk > 0, "ssd: tip anchor demoted to disk");
    check_eq(tree.stats().blocks_ram, 0, "ssd: no blocks left in ram");

    {
        const auto base = run_baseline(ctx, tokens, 42, 8);
        const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
        check_eq(r.res.C, 1536, "ssd: restore point is the tip");
        check(r.gen == base, "ssd: tokens match the baseline");
    }

    for (const auto & e : std::filesystem::directory_iterator(cfg.disk_dir + "/blocks")) {
        std::filesystem::remove(e.path());
    }

    const int64_t errors_before = tree.stats().disk_errors;

    {
        const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
        check_eq(r.res.C, -1, "ssd: restore misses when block files are gone");
        check(tree.stats().disk_errors > errors_before, "ssd: disk error counted");
    }

    std::filesystem::remove_all(cfg.disk_dir, ec);

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

static int scenario_unaligned(llama_context * ctx, const kv_tree_config & cfg_in) {
    fprintf(stderr, "[t32-tree] scenario: unaligned\n");

    kv_tree_config cfg = cfg_in;
    cfg.anchor_step = 512;

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tokens = make_tokens(1300, 7);   // 2 full chunks + a 276-token tail
    const std::vector<llama_token> tok1200(tokens.begin(), tokens.begin() + 1200);

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 1150), 0, 512);

    const kv_tree_anchor_in ck = capture_partial(ctx, 0, 1150);

    prefill(ctx, 0, tokens, 1150, 512);
    check(tree.park(io, nullptr, tokens, {}, { ck }), "unaligned: park (1300 tokens, anchor at 1150)");

    {
        // matched baseline: the same batch boundaries as the original fill
        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        prefill(ctx, 0, std::vector<llama_token>(tok1200.begin(), tok1200.begin() + 1150), 0, 512);
        prefill(ctx, 0, tok1200, 1150, 512);
        const auto base = generate(ctx, 0, 42, 1200, 8);

        const auto r = run_tree_path(ctx, tree, io, tok1200, 42, 8);
        check_eq(r.res.C, 1150, "unaligned: restore uses the non-aligned anchor");
        check_eq(r.res.heal, 1200, "unaligned: heal points at the request end");
        check(r.gen == base, "unaligned: tokens match the matched baseline");
    }

    {
        // matched baseline for the full length
        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 1150), 0, 512);
        prefill(ctx, 0, tokens, 1150, 512);
        const auto base = generate(ctx, 0, 42, 1300, 8);

        const auto r = run_tree_path(ctx, tree, io, tokens, 42, 8);
        check_eq(r.res.C, 1300, "unaligned: full length restores at the non-aligned tip");
        check_eq(r.res.heal, -1, "unaligned: no heal for the full length");
        check(r.gen == base, "unaligned: full length tokens match the matched baseline");
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

static int scenario_restore_anchors(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] scenario: restore anchors\n");

    const auto tokens = make_tokens(3072, 7);

    // park with a checkpoint candidate at 2048
    {
        kv_tree tree(cfg);
        kv_tree_io_llama io(ctx, 0);

        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        if (prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 2048), 0, 512) != 0) {
            return 1;
        }

        const kv_tree_anchor_in ck = capture_partial(ctx, 0, 2048);
        check(!ck.data_tgt.empty(), "restore-anchors: capture the partial state at 2048");

        if (prefill(ctx, 0, tokens, 2048, 512) != 0) {
            return 1;
        }

        check(tree.park(io, nullptr, tokens, {}, { ck }), "restore-anchors: park with the checkpoint");

        const kv_tree_restore r = tree.restore(io, nullptr, tokens, {});
        check_eq(r.C, 3072, "restore-anchors: restore at the tip");
        check_eq((long long) r.anchors.size(), 2, "restore-anchors: two anchors on the path");

        bool found = false;
        bool sorted = true;
        llama_pos prev = -1;
        for (const auto & a : r.anchors) {
            if (a.pos <= prev) {
                sorted = false;
            }
            prev = a.pos;
            if (a.pos == 2048) {
                found = true;
                check(a.data_tgt == ck.data_tgt, "restore-anchors: the payload matches the parked checkpoint");
            }
        }
        check(found, "restore-anchors: the checkpoint anchor is returned");
        check(sorted, "restore-anchors: anchors are sorted by position");
    }

    // same with the payloads on disk
    {
        kv_tree_config cfg2 = cfg;
        cfg2.disk_dir   = (std::filesystem::temp_directory_path() / "t32-tree-anchors").string();
        cfg2.ram_limit  = 8 * 1024;    // force demotion
        cfg2.disk_limit = 1ull << 30;  // the real 2B KV for 3072 tokens is ~150 MB

        std::error_code ec;
        std::filesystem::remove_all(cfg2.disk_dir, ec);

        kv_tree tree(cfg2);
        kv_tree_io_llama io(ctx, 0);

        llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        if (prefill(ctx, 0, std::vector<llama_token>(tokens.begin(), tokens.begin() + 2048), 0, 512) != 0) {
            return 1;
        }

        const kv_tree_anchor_in ck = capture_partial(ctx, 0, 2048);

        if (prefill(ctx, 0, tokens, 2048, 512) != 0) {
            return 1;
        }

        check(tree.park(io, nullptr, tokens, {}, { ck }), "restore-anchors: park with a small ram budget");
        check(tree.stats().anchors_disk + tree.stats().blocks_disk > 0, "restore-anchors: data went to disk");

        const kv_tree_restore r = tree.restore(io, nullptr, tokens, {});
        check_eq(r.C, 3072, "restore-anchors: restore at the tip (ssd)");

        const int64_t bytes_ram_after_first = tree.stats().bytes_ram;
        tree.restore(io, nullptr, tokens, {});
        check_eq(tree.stats().bytes_ram, bytes_ram_after_first, "restore-anchors: repeated disk restores do not inflate ram accounting");

        bool found = false;
        for (const auto & a : r.anchors) {
            if (a.pos == 2048) {
                found = true;
                check(a.data_tgt == ck.data_tgt, "restore-anchors: the disk payload matches the parked checkpoint");
            }
        }
        check(found, "restore-anchors: the disk anchor is returned");

        std::filesystem::remove_all(cfg2.disk_dir, ec);
    }

    return n_fail == 0 ? 0 : 1;
}

static int scenario_fork_miss(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] scenario: fork miss\n");

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    const auto tok_a = make_tokens(4096, 3);
    auto tok_b = make_tokens(4096, 4);
    for (int i = 0; i < 2048; ++i) {
        tok_b[i] = tok_a[i];
    }

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    if (prefill(ctx, 0, tok_a, 0, 512) != 0) {
        return 1;
    }
    check(tree.park(io, nullptr, tok_a, {}, {}), "fork-miss: park A");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    const kv_tree_restore r1 = tree.restore(io, nullptr, tok_b, {});
    check_eq(r1.C, -1, "fork-miss: the first visit misses");
    check_eq(r1.heal, 2048, "fork-miss: the miss reports the divergence point");

    // the server prefills to the heal position and captures there
    if (prefill(ctx, 0, std::vector<llama_token>(tok_b.begin(), tok_b.begin() + 2048), 0, 512) != 0) {
        return 1;
    }
    check(tree.capture_anchor(io, nullptr, tok_b, {}, 2048), "fork-miss: capture at the divergence point");
    if (prefill(ctx, 0, tok_b, 2048, 512) != 0) {
        return 1;
    }

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    const kv_tree_restore r2 = tree.restore(io, nullptr, tok_b, {});
    check_eq(r2.C, 2048, "fork-miss: the second visit restores at the fork");
    check_eq(r2.heal, -1, "fork-miss: no heal on a full match");

    const auto base = run_baseline(ctx, tok_b, 42, 8);
    check(!base.empty(), "fork-miss: baseline generation");

    const auto r = run_tree_path(ctx, tree, io, tok_b, 42, 8);
    check_eq(r.res.C, 2048, "fork-miss: the tree path restores at the fork");
    check(r.gen == base, "fork-miss: tokens match the baseline");

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

static int run_accept(llama_context * ctx, const kv_tree_config & cfg) {
    fprintf(stderr, "[t32-tree] mode accept\n");

    kv_tree tree(cfg);
    kv_tree_io_llama io(ctx, 0);

    // A-mini: two 4096-token sequences sharing 3072 tokens, six alternating restores
    const auto tok_a = make_fork_tokens(3072, 1024, 0);
    const auto tok_b = make_fork_tokens(3072, 1024, 1);

    prefill(ctx, 0, tok_a, 0, 512);

    const size_t seq_bytes    = llama_state_seq_get_size_range_ext(ctx, 0, 0, 4096, 0);
    const size_t anchor_bytes = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

    check(tree.park(io, nullptr, tok_a, {}, {}), "accept: park A (4096)");

    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    prefill(ctx, 0, tok_b, 0, 512);
    check(tree.park(io, nullptr, tok_b, {}, {}), "accept: park B (4096)");

    const auto base_a = run_baseline(ctx, tok_a, 42, 8);
    const auto base_b = run_baseline(ctx, tok_b, 42, 8);

    for (int round = 0; round < 6; ++round) {
        const auto ra = run_tree_path(ctx, tree, io, tok_a, 42, 8);
        check_eq(ra.res.C, 4096, "accept: A restores at the tip");
        check(ra.gen == base_a, "accept: A tokens match the baseline");

        const auto rb = run_tree_path(ctx, tree, io, tok_b, 42, 8);
        check_eq(rb.res.C, 4096, "accept: B restores at the tip");
        check(rb.gen == base_b, "accept: B tokens match the baseline");
    }

    check_eq(tree.stats().tokens_reused, 12 * 4096, "accept: tokens reused");
    check_eq(tree.stats().blocks_ram + tree.stats().blocks_disk, 10, "accept: shared trunk stored once (6+2+2 blocks)");
    check(tree.stats().bytes_ram + tree.stats().bytes_disk < 2 * ((int64_t) seq_bytes + (int64_t) anchor_bytes), "accept: stored bytes below two full copies (dedup)");

    // B-mini: four 1024-token sessions sharing a 512-token prefix
    {
        kv_tree_config cfg2 = cfg;
        cfg2.ram_limit = 4096ull << 20;

        kv_tree tree2(cfg2);
        kv_tree_io_llama io2(ctx, 0);

        std::vector<std::vector<llama_token>> toks;
        for (int i = 0; i < 4; ++i) {
            toks.push_back(make_fork_tokens(512, 512, 10 + i));
        }

        for (int i = 0; i < 4; ++i) {
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
            prefill(ctx, 0, toks[i], 0, 512);
            check(tree2.park(io2, nullptr, toks[i], {}, {}), "accept: park short session");
        }

        check_eq(tree2.stats().blocks_ram + tree2.stats().blocks_disk, 5, "accept: shared prefix stored once (1+4 blocks)");

        for (int i = 0; i < 4; ++i) {
            const auto base = run_baseline(ctx, toks[i], 42, 8);
            const auto r = run_tree_path(ctx, tree2, io2, toks[i], 42, 8);
            check_eq(r.res.C, 1024, "accept: short session restores at the tip");
            check(r.gen == base, "accept: short session tokens match the baseline");
        }
    }

    tree.dump();

    return n_fail == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    std::string mode = "logic";

    int    chunk       = 512;
    int    anchor_step = 32768;
    int    ram_mib     = 8192;
    int    disk_mib    = 65536;
    std::string disk;

    std::vector<char *> args;
    args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
            continue;
        }
        if (arg == "--chunk" && i + 1 < argc) {
            chunk = atoi(argv[++i]);
            continue;
        }
        if (arg == "--anchor-step" && i + 1 < argc) {
            anchor_step = atoi(argv[++i]);
            continue;
        }
        if (arg == "--ram-mib" && i + 1 < argc) {
            ram_mib = atoi(argv[++i]);
            continue;
        }
        if (arg == "--disk" && i + 1 < argc) {
            disk = argv[++i];
            continue;
        }
        if (arg == "--disk-mib" && i + 1 < argc) {
            disk_mib = atoi(argv[++i]);
            continue;
        }

        args.push_back(argv[i]);
    }

    if (mode == "logic") {
        return run_logic();
    }

    common_params params;
    params.sampling.seed = 1234;
    params.n_parallel = 1;

    common_init();

    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        fprintf(stderr, "usage: %s -m model.gguf [common args] --mode <logic|model|accept> [--chunk N] [--anchor-step N] [--ram-mib N] [--disk DIR] [--disk-mib N]\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();

    if (ctx == nullptr) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    kv_tree_config cfg;
    cfg.chunk       = chunk;
    cfg.anchor_step = anchor_step;
    cfg.ram_limit   = (size_t) ram_mib << 20;
    cfg.disk_limit  = (size_t) disk_mib << 20;
    cfg.disk_dir    = disk;

    if (mode == "model") {
        int ret = 0;
        ret |= scenario_tip(ctx, cfg);
        ret |= scenario_fork(ctx, cfg);
        ret |= scenario_sparsify(ctx, cfg);
        ret |= scenario_ssd(ctx, cfg);
        ret |= scenario_restore_anchors(ctx, cfg);
        ret |= scenario_fork_miss(ctx, cfg);
        ret |= scenario_unaligned(ctx, cfg);
        return ret;
    }

    if (mode == "accept") {
        return run_accept(ctx, cfg);
    }

    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
