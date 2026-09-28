#include "server-kv-tree.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>

#define XXH_INLINE_ALL
#include "hash/xxhash/xxhash.h"

static uint64_t chunk_hash(const llama_token * tok, size_t n, uint64_t seed) {
    return XXH64(tok, n * sizeof(llama_token), seed);
}

static void chain_hashes(const std::vector<llama_token> & tokens, int chunk, std::vector<uint64_t> & h) {
    h.clear();

    uint64_t cur = 0;

    for (size_t a = 0; a < tokens.size(); a += (size_t) chunk) {
        const size_t b = std::min(tokens.size(), a + (size_t) chunk);
        cur = chunk_hash(tokens.data() + a, b - a, cur);
        h.push_back(cur);
    }
}

static const uint32_t T32_ENV_MAGIC_BLOCK  = 0x42323354u;  // "T32B"
static const uint32_t T32_ENV_MAGIC_ANCHOR = 0x41323354u;  // "T32A"
static const uint32_t T32_ENV_VERSION      = 1;

static void put_u32(std::vector<uint8_t> & v, uint32_t x) {
    uint8_t b[4];
    memcpy(b, &x, 4);
    v.insert(v.end(), b, b + 4);
}

static void put_i32(std::vector<uint8_t> & v, int32_t x) {
    put_u32(v, (uint32_t) x);
}

static void put_u64(std::vector<uint8_t> & v, uint64_t x) {
    uint8_t b[8];
    memcpy(b, &x, 8);
    v.insert(v.end(), b, b + 8);
}

struct kv_env {
    uint32_t magic   = 0;
    uint64_t hash    = 0;
    llama_pos p0     = 0;
    llama_pos p1     = 0;
    uint32_t n_tgt   = 0;
    uint32_t n_dft   = 0;
    uint32_t n_spec  = 0;
    const uint8_t * tgt  = nullptr;
    const uint8_t * dft  = nullptr;
    const uint8_t * spec = nullptr;
};

static std::vector<uint8_t> envelope_make(uint32_t magic, uint64_t hash, llama_pos p0, llama_pos p1,
                                          const std::vector<uint8_t> & tgt,
                                          const std::vector<uint8_t> & dft,
                                          const std::vector<uint8_t> & spec) {
    uint64_t ph = 0;
    if (!tgt.empty()) {
        ph = XXH64(tgt.data(), tgt.size(), 0);
    }
    if (!dft.empty()) {
        ph = XXH64(dft.data(), dft.size(), ph);
    }
    if (!spec.empty()) {
        ph = XXH64(spec.data(), spec.size(), ph);
    }

    std::vector<uint8_t> v;
    v.reserve(44 + tgt.size() + dft.size() + spec.size());

    put_u32(v, magic);
    put_u32(v, T32_ENV_VERSION);
    put_u64(v, hash);
    put_i32(v, (int32_t) p0);
    put_i32(v, (int32_t) p1);
    put_u32(v, (uint32_t) tgt.size());
    put_u32(v, (uint32_t) dft.size());
    put_u32(v, (uint32_t) spec.size());
    put_u64(v, ph);
    v.insert(v.end(), tgt.begin(), tgt.end());
    v.insert(v.end(), dft.begin(), dft.end());
    v.insert(v.end(), spec.begin(), spec.end());

    return v;
}

static bool envelope_parse(const std::vector<uint8_t> & buf, kv_env & e) {
    if (buf.size() < 44) {
        return false;
    }

    const uint8_t * p = buf.data();
    size_t off = 0;

    uint32_t version = 0;
    uint64_t ph = 0;

    memcpy(&e.magic,   p + off, 4); off += 4;
    memcpy(&version,   p + off, 4); off += 4;
    memcpy(&e.hash,    p + off, 8); off += 8;
    memcpy(&e.p0,      p + off, 4); off += 4;
    memcpy(&e.p1,      p + off, 4); off += 4;
    memcpy(&e.n_tgt,   p + off, 4); off += 4;
    memcpy(&e.n_dft,   p + off, 4); off += 4;
    memcpy(&e.n_spec,  p + off, 4); off += 4;
    memcpy(&ph,        p + off, 8); off += 8;

    if (version != T32_ENV_VERSION) {
        return false;
    }
    if (off + e.n_tgt + e.n_dft + e.n_spec != buf.size()) {
        return false;
    }

    e.tgt  = p + off; off += e.n_tgt;
    e.dft  = p + off; off += e.n_dft;
    e.spec = p + off;

    uint64_t check = 0;
    if (e.n_tgt > 0) {
        check = XXH64(e.tgt, e.n_tgt, 0);
    }
    if (e.n_dft > 0) {
        check = XXH64(e.dft, e.n_dft, check);
    }
    if (e.n_spec > 0) {
        check = XXH64(e.spec, e.n_spec, check);
    }

    return check == ph;
}

kv_tree_io_llama::kv_tree_io_llama(llama_context * ctx, llama_seq_id seq_id) : ctx(ctx), seq_id(seq_id) {
}

bool kv_tree_io_llama::get_range(llama_pos p0, llama_pos p1, std::vector<uint8_t> & out) {
    const size_t size = llama_state_seq_get_size_range_ext(ctx, seq_id, p0, p1, 0);
    if (size == 0) {
        return false;
    }

    out.resize(size);

    return llama_state_seq_get_data_range_ext(ctx, out.data(), out.size(), seq_id, p0, p1, 0) == size;
}

bool kv_tree_io_llama::set_range(const uint8_t * data, size_t size, bool append) {
    return llama_state_seq_set_data_range_ext(ctx, data, size, seq_id, append, 0) == size;
}

bool kv_tree_io_llama::get_partial(std::vector<uint8_t> & out) {
    const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (size == 0) {
        return false;
    }

    out.resize(size);

    return llama_state_seq_get_data_ext(ctx, out.data(), out.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size;
}

bool kv_tree_io_llama::set_partial(const uint8_t * data, size_t size) {
    return llama_state_seq_set_data_ext(ctx, data, size, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == size;
}

bool kv_tree_io_llama::seq_rm(llama_pos p0, llama_pos p1) {
    return llama_memory_seq_rm(llama_get_memory(ctx), seq_id, p0, p1);
}

llama_pos kv_tree_io_llama::pos_max() {
    return llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id);
}

kv_tree::kv_tree(const kv_tree_config & cfg) : cfg(cfg) {
    if (cfg.chunk <= 0 || cfg.anchor_step < 0 || cfg.fork_step < 0) {
        fprintf(stderr, "[kv-tree] invalid config: chunk = %d, anchor_step = %d, fork_step = %d\n", cfg.chunk, cfg.anchor_step, cfg.fork_step);
        GGML_ABORT("invalid kv tree config");
    }

    if (!cfg.disk_dir.empty()) {
        // the tree is not persistent: drop the leftovers of a previous run
        std::error_code ec;

        const size_t n_stale = std::filesystem::remove_all(cfg.disk_dir + "/blocks", ec) +
                               std::filesystem::remove_all(cfg.disk_dir + "/anchors", ec);

        fprintf(stderr, "[kv-tree] cleared %zu stale files from %s\n", n_stale, cfg.disk_dir.c_str());
    }
}

kv_tree_match kv_tree::match(const std::vector<llama_token> & tokens) const {
    kv_tree_match m;

    std::vector<uint64_t> h;
    chain_hashes(tokens, cfg.chunk, h);

    for (size_t i = 0; i < h.size(); ++i) {
        const auto it = blocks.find(h[i]);
        if (it == blocks.end()) {
            break;
        }

        const kv_tree_block & b = it->second;
        const size_t a = i * (size_t) cfg.chunk;

        if (a + b.tokens.size() > tokens.size()) {
            break;
        }
        if (memcmp(tokens.data() + a, b.tokens.data(), b.tokens.size() * sizeof(llama_token)) != 0) {
            break;
        }

        m.path.push_back(h[i]);
        m.n_full++;
        m.deep = (llama_pos) (a + b.tokens.size());
    }

    // partial match inside the last chunk only when the full chain is verified
    const size_t n_full_req = tokens.size() / (size_t) cfg.chunk;
    const size_t tail_a = n_full_req * (size_t) cfg.chunk;

    if (m.n_full == n_full_req && tail_a < tokens.size()) {
        const auto it = blocks_at.find((llama_pos) tail_a);
        if (it != blocks_at.end()) {
            for (const uint64_t hash : it->second) {
                const kv_tree_block & b = blocks.at(hash);

                const size_t n = std::min(b.tokens.size(), tokens.size() - tail_a);
                size_t k = 0;
                while (k < n && b.tokens[k] == tokens[tail_a + k]) {
                    ++k;
                }

                if (k > m.n_part) {
                    m.n_part = k;
                    m.part_hash = hash;
                }
            }
        }

        m.deep = std::max(m.deep, (llama_pos) (tail_a + m.n_part));
    }

    return m;
}

bool kv_tree::store_anchor(uint64_t blk_hash, llama_pos pos, int kind,
                           std::vector<uint8_t> && data_tgt, std::vector<uint8_t> && data_dft) {
    if (pos <= 0 || blk_hash == 0) {
        return false;
    }

    const auto key = std::make_pair(blk_hash, pos);

    const auto it = anchors.find(key);
    if (it != anchors.end()) {
        it->second.refcount++;
        it->second.heat++;
        it->second.last_used = ++now;
        return true;
    }

    const size_t size = data_tgt.size() + data_dft.size();

    kv_tree_anchor & a = anchors[key];

    a.blk_hash  = blk_hash;
    a.pos       = pos;
    a.kind      = kind;
    a.refcount  = 1;
    a.last_used = ++now;
    a.bytes     = size;
    a.data_tgt  = std::move(data_tgt);
    a.data_dft  = std::move(data_dft);

    st.anchors_ram++;
    st.anchors_added++;
    st.bytes_ram += (int64_t) size;

    return true;
}

static size_t anchor_bytes(const kv_tree_anchor & a) {
    return a.bytes;
}

void kv_tree::remove_anchor(std::map<std::pair<uint64_t, llama_pos>, kv_tree_anchor>::iterator it) {
    kv_tree_anchor & a = it->second;

    if (a.transient) {
        st.bytes_ram -= (int64_t) a.bytes;
        st.anchors_ram--;
    }

    if (a.on_disk) {
        std::error_code ec;
        std::filesystem::remove(a.path, ec);
        st.bytes_disk -= (int64_t) a.bytes;
        st.anchors_disk--;
    } else {
        st.bytes_ram -= (int64_t) a.bytes;
        st.anchors_ram--;
    }

    anchors.erase(it);
}

bool kv_tree::capture_anchor(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens, llama_pos pos) {
    if (pos <= 0) {
        return false;
    }
    if (io_tgt.pos_max() != pos - 1) {
        fprintf(stderr, "[kv-tree] capture refused at %d: sequence end %d\n", pos, io_tgt.pos_max());
        return false;
    }

    std::vector<uint64_t> chain;
    chain_hashes(tokens, cfg.chunk, chain);

    uint64_t blk = containing_block(pos, &chain);
    if (blk == 0) {
        // heal can land inside a stored chunk when the request tail does not line up with it;
        // fall back to a stored block that covers pos and matches the stored tokens
        auto base = blocks_at.lower_bound(pos);
        if (base != blocks_at.begin()) {
            --base;

            for (const uint64_t hash : base->second) {
                const auto bi = blocks.find(hash);
                if (bi == blocks.end()) {
                    continue;
                }

                const kv_tree_block & b = bi->second;
                if (b.pos0 >= pos || pos > b.pos1) {
                    continue;
                }

                const size_t n = (size_t) (pos - b.pos0);
                if (n > b.tokens.size() || (size_t) pos > tokens.size()) {
                    continue;
                }
                if (std::equal(b.tokens.begin(), b.tokens.begin() + n, tokens.begin() + b.pos0)) {
                    blk = hash;
                    break;
                }
            }
        }

        if (blk == 0) {
            fprintf(stderr, "[kv-tree] capture refused at %d: no chain block contains this position\n", pos);
            return false;
        }
    }

    const auto key = std::make_pair(blk, pos);

    const auto it = anchors.find(key);
    if (it != anchors.end()) {
        it->second.heat++;
        it->second.last_used = ++now;
        return true;
    }

    llama_pos prev = -1;
    for (const auto & kv : anchors) {
        if (kv.second.pos >= pos || kv.second.pos <= prev) {
            continue;
        }
        if (kv.second.kind == KV_TREE_ANCHOR_MESSAGE) {
            continue;   // guesses never suppress a fork anchor
        }
        if (std::find(chain.begin(), chain.end(), kv.first.first) != chain.end()) {
            prev = kv.second.pos;
        }
    }

    if (prev >= 0 && pos - prev < cfg.fork_step) {
        fprintf(stderr, "[kv-tree] capture skipped at %d: within fork_step\n", pos);
        st.anchors_skipped++;
        st.anchors_skipped_step++;
        return false;
    }

    std::vector<uint8_t> tgt;
    std::vector<uint8_t> dft;

    if (!io_tgt.get_partial(tgt)) {
        fprintf(stderr, "[kv-tree] capture refused at %d: failed to capture the state\n", pos);
        st.anchors_skipped++;
        return false;
    }
    if (io_dft != nullptr && !io_dft->get_partial(dft)) {
        fprintf(stderr, "[kv-tree] capture refused at %d: failed to capture the draft state\n", pos);
        st.anchors_skipped++;
        return false;
    }

    if (!store_anchor(blk, pos, KV_TREE_ANCHOR_ONDEMAND, std::move(tgt), std::move(dft))) {
        return false;
    }

    if (!enforce_budget() || anchors.count(key) == 0) {
        st.anchors_skipped++;
        return false;
    }

    // no pruning needed: prev is the deepest same-chain anchor below pos, so no anchor can sit between them

    return true;
}

uint64_t kv_tree::containing_block(llama_pos pos, const std::vector<uint64_t> * chain) const {
    if (pos <= 0) {
        return 0;
    }

    auto it = blocks_at.lower_bound(pos);
    if (it == blocks_at.begin()) {
        return 0;
    }
    --it;

    for (const uint64_t hash : it->second) {
        const auto b = blocks.find(hash);
        if (b == blocks.end() || b->second.pos0 >= pos || pos > b->second.pos1) {
            continue;
        }
        if (chain != nullptr && std::find(chain->begin(), chain->end(), hash) == chain->end()) {
            continue;
        }
        return hash;
    }

    return 0;
}

std::string kv_tree::block_path(uint64_t hash) const {
    char buf[32];
    snprintf(buf, sizeof(buf), "%016" PRIx64 ".bin", hash);
    return cfg.disk_dir + "/blocks/" + buf;
}

std::string kv_tree::anchor_path(uint64_t blk_hash, llama_pos pos) const {
    char buf[48];
    snprintf(buf, sizeof(buf), "%016" PRIx64 "_%d.bin", blk_hash, (int) pos);
    return cfg.disk_dir + "/anchors/" + buf;
}

bool kv_tree::write_disk(const std::string & path, const std::vector<uint8_t> & buf) {
    const std::string tmp = path + ".tmp";

    {
        std::error_code ec;   // the tier subdirectories are created on demand
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
        if (ec) {
            fprintf(stderr, "[kv-tree] failed to create the directory for %s: %s\n", tmp.c_str(), ec.message().c_str());
            return false;
        }
    }

    std::FILE * f = fopen(tmp.c_str(), "wb");
    if (f == nullptr) {
        fprintf(stderr, "[kv-tree] failed to open %s for writing\n", tmp.c_str());
        return false;
    }

    const size_t n = fwrite(buf.data(), 1, buf.size(), f);
    const bool ok = fclose(f) == 0 && n == buf.size();

    if (!ok) {
        remove(tmp.c_str());
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(path, ec);
        std::filesystem::rename(tmp, path, ec);
    }
    if (ec) {
        remove(tmp.c_str());
        return false;
    }

    return true;
}

bool kv_tree::read_disk(const std::string & path, std::vector<uint8_t> & out) {
    std::FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }

    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return false;
    }

    out.resize((size_t) size);

    const size_t n = fread(out.data(), 1, out.size(), f);
    fclose(f);

    return n == out.size();
}

bool kv_tree::demote_block(kv_tree_block & b) {
    if (cfg.disk_dir.empty() || b.on_disk || b.data.empty()) {
        return false;
    }
    if (st.bytes_disk + (int64_t) b.data.size() > (int64_t) cfg.disk_limit) {
        return false;
    }

    const std::string path = block_path(b.hash);

    if (!write_disk(path, envelope_make(T32_ENV_MAGIC_BLOCK, b.hash, b.pos0, b.pos1, b.data, {}, {}))) {
        fprintf(stderr, "[kv-tree] failed to write block %016" PRIx64 " to disk\n", b.hash);
        st.disk_errors++;
        return false;
    }

    st.bytes_ram  -= (int64_t) b.data.size();
    st.bytes_disk += (int64_t) b.data.size();
    st.blocks_ram--;
    st.blocks_disk++;
    st.bytes_store += (int64_t) b.data.size();

    b.on_disk = true;
    b.path    = path;
    std::vector<uint8_t>().swap(b.data);

    return true;
}

bool kv_tree::demote_anchor(kv_tree_anchor & a) {
    if (cfg.disk_dir.empty() || a.on_disk || a.data_tgt.empty()) {
        return false;
    }

    const size_t size = anchor_bytes(a);
    if (st.bytes_disk + (int64_t) size > (int64_t) cfg.disk_limit) {
        return false;
    }

    const std::string path = anchor_path(a.blk_hash, a.pos);

    if (!write_disk(path, envelope_make(T32_ENV_MAGIC_ANCHOR, a.blk_hash, a.pos, a.pos, a.data_tgt, a.data_dft, a.data_spec))) {
        fprintf(stderr, "[kv-tree] failed to write anchor %016" PRIx64 "@%d to disk\n", a.blk_hash, a.pos);
        st.disk_errors++;
        return false;
    }

    st.bytes_ram  -= (int64_t) size;
    st.bytes_disk += (int64_t) size;
    st.anchors_ram--;
    st.anchors_disk++;
    st.bytes_store += (int64_t) size;

    a.on_disk = true;
    a.path    = path;
    std::vector<uint8_t>().swap(a.data_tgt);
    std::vector<uint8_t>().swap(a.data_dft);
    std::vector<uint8_t>().swap(a.data_spec);

    return true;
}

bool kv_tree::load_payload(kv_tree_block & b) {
    if (!b.on_disk || b.transient) {
        return true;
    }

    std::vector<uint8_t> buf;
    if (!read_disk(b.path, buf)) {
        fprintf(stderr, "[kv-tree] failed to read block %016" PRIx64 " from disk\n", b.hash);
        st.disk_errors++;
        return false;
    }

    kv_env e;
    if (!envelope_parse(buf, e) || e.magic != T32_ENV_MAGIC_BLOCK || e.hash != b.hash || e.p0 != b.pos0 || e.p1 != b.pos1 || e.n_dft != 0 || e.n_spec != 0) {
        fprintf(stderr, "[kv-tree] block %016" PRIx64 " payload check failed, dropping it\n", b.hash);
        st.disk_errors++;
        return false;
    }

    b.data.assign(e.tgt, e.tgt + e.n_tgt);
    st.bytes_load += (int64_t) b.data.size();
    b.transient = true;

    st.bytes_ram += (int64_t) b.data.size();
    st.blocks_ram++;

    return true;
}

bool kv_tree::load_payload(kv_tree_anchor & a) {
    if (!a.on_disk || a.transient) {
        return true;
    }

    std::vector<uint8_t> buf;
    if (!read_disk(a.path, buf)) {
        fprintf(stderr, "[kv-tree] failed to read anchor %016" PRIx64 "@%d from disk\n", a.blk_hash, a.pos);
        st.disk_errors++;
        return false;
    }

    kv_env e;
    if (!envelope_parse(buf, e) || e.magic != T32_ENV_MAGIC_ANCHOR || e.hash != a.blk_hash || e.p0 != a.pos || e.p1 != a.pos) {
        fprintf(stderr, "[kv-tree] anchor %016" PRIx64 "@%d payload check failed, dropping it\n", a.blk_hash, a.pos);
        st.disk_errors++;
        return false;
    }

    a.data_tgt.assign(e.tgt, e.tgt + e.n_tgt);
    a.data_dft.assign(e.dft, e.dft + e.n_dft);
    a.data_spec.assign(e.spec, e.spec + e.n_spec);
    st.bytes_load += (int64_t) anchor_bytes(a);
    a.transient = true;

    st.bytes_ram += (int64_t) anchor_bytes(a);
    st.anchors_ram++;

    return true;
}

// feed one block payload to the io; keep it resident only when it fits the ram budget,
// otherwise stream it from disk through a scratch buffer
bool kv_tree::set_block_payload(kv_tree_io & io, kv_tree_block & b, bool append, std::vector<uint8_t> & scratch) {
    if (b.on_disk && (size_t) st.bytes_ram + b.bytes > cfg.ram_limit) {
        std::vector<uint8_t> buf;
        if (!read_disk(b.path, buf)) {
            fprintf(stderr, "[kv-tree] failed to read block %016" PRIx64 " from disk\n", b.hash);
            st.disk_errors++;
            return false;
        }

        kv_env e;
        if (!envelope_parse(buf, e) || e.magic != T32_ENV_MAGIC_BLOCK || e.hash != b.hash || e.p0 != b.pos0 || e.p1 != b.pos1 || e.n_dft != 0 || e.n_spec != 0) {
            fprintf(stderr, "[kv-tree] block %016" PRIx64 " payload check failed, dropping it\n", b.hash);
            st.disk_errors++;
            return false;
        }

        scratch.assign(e.tgt, e.tgt + e.n_tgt);
        st.bytes_load += (int64_t) scratch.size();
        return io.set_range(scratch.data(), scratch.size(), append);
    }

    if (!load_payload(b)) {
        return false;
    }

    return io.set_range(b.data.data(), b.data.size(), append);
}

void kv_tree::settle() {
    for (auto & p : blocks) {
        kv_tree_block & b = p.second;
        if (!b.transient) {
            continue;
        }

        b.transient = false;

        if ((size_t) st.bytes_ram <= cfg.ram_limit) {
            std::error_code ec;
            std::filesystem::remove(b.path, ec);

            st.bytes_disk -= (int64_t) b.data.size();
            st.blocks_disk--;
            b.on_disk = false;
            b.path.clear();
        } else {
            st.bytes_ram -= (int64_t) b.data.size();
            st.blocks_ram--;
            std::vector<uint8_t>().swap(b.data);
        }
    }

    for (auto & p : anchors) {
        kv_tree_anchor & a = p.second;
        if (!a.transient) {
            continue;
        }

        a.transient = false;

        const size_t size = anchor_bytes(a);

        if ((size_t) st.bytes_ram <= cfg.ram_limit) {
            std::error_code ec;
            std::filesystem::remove(a.path, ec);

            st.bytes_disk -= (int64_t) size;
            st.anchors_disk--;
            a.on_disk = false;
            a.path.clear();
        } else {
            st.bytes_ram -= (int64_t) size;
            st.anchors_ram--;
            std::vector<uint8_t>().swap(a.data_tgt);
            std::vector<uint8_t>().swap(a.data_dft);
            std::vector<uint8_t>().swap(a.data_spec);
        }
    }
}

bool kv_tree::has_successor(const kv_tree_block & b) const {
    const auto it = blocks_at.find(b.pos1);
    return it != blocks_at.end() && !it->second.empty();
}

void kv_tree::remove_seq(std::unordered_map<uint64_t, kv_tree_seq>::iterator it) {
    kv_tree_seq & s = it->second;

    for (auto & kv : anchors) {
        bool in_chain = false;
        for (const uint64_t hash : s.chain) {
            if (hash == kv.first.first) {
                in_chain = true;
                break;
            }
        }
        if (in_chain && kv.second.refcount > 0) {
            kv.second.refcount--;
        }
    }

    for (auto rit = s.chain.rbegin(); rit != s.chain.rend(); ++rit) {
        const auto b = blocks.find(*rit);
        if (b == blocks.end()) {
            continue;
        }
        if (--b->second.refcount <= 0) {
            remove_block(b);
        }
    }

    seqs.erase(it);
    st.evicted_seqs++;
}

bool kv_tree::drop_seq(const std::vector<llama_token> & tokens) {
    if (tokens.empty()) {
        return false;
    }

    const kv_tree_match m = match(tokens);

    uint64_t tip = 0;
    if (m.n_part > 0) {
        tip = m.part_hash;
    } else if (m.n_full > 0) {
        tip = m.path.back();
    } else {
        return false;
    }

    const auto it = seqs.find(tip);
    if (it == seqs.end()) {
        return false;
    }

    remove_seq(it);
    return true;
}

bool kv_tree::demote_one() {
    auto best = blocks.end();
    int64_t best_score = 0;

    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        const kv_tree_block & b = it->second;
        if (b.on_disk || b.data.empty()) {
            continue;
        }

        int64_t score = b.refcount * 1000000 + b.heat * 1000 + b.last_used;
        if (!has_successor(b)) {
            score -= 100000000;   // leaves go to disk first
        }

        if (best == blocks.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best != blocks.end() && demote_block(best->second)) {
        return true;
    }

    auto best_a = anchors.end();
    int64_t best_a_score = 0;

    for (auto it = anchors.begin(); it != anchors.end(); ++it) {
        const kv_tree_anchor & a = it->second;
        if (a.on_disk || a.data_tgt.empty()) {
            continue;
        }

        const int64_t score = a.refcount * 1000000 + a.heat * 1000 + a.last_used;

        if (best_a == anchors.end() || score < best_a_score) {
            best_a = it;
            best_a_score = score;
        }
    }

    return best_a != anchors.end() && demote_anchor(best_a->second);
}

bool kv_tree::evict_anchor_one() {
    auto best = anchors.end();
    int64_t best_score = 0;

    for (auto it = anchors.begin(); it != anchors.end(); ++it) {
        const kv_tree_anchor & a = it->second;
        if (a.pinned) {
            continue;
        }

        const int64_t score = a.refcount * 1000000 + a.heat * 1000 + a.last_used;

        if (best == anchors.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best == anchors.end()) {
        return false;
    }

    remove_anchor(best);
    st.evicted_anchors++;
    return true;
}

bool kv_tree::evict_block_one() {
    auto best = blocks.end();
    int64_t best_score = 0;

    for (auto it = blocks.begin(); it != blocks.end(); ++it) {
        const kv_tree_block & b = it->second;
        if (b.pinned || b.refcount > 1 || has_successor(b)) {
            continue;
        }

        const int64_t score = b.refcount * 1000000 + b.heat * 1000 + b.last_used;

        if (best == blocks.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best == blocks.end()) {
        return false;
    }

    remove_block(best);
    return true;
}

bool kv_tree::evict_seq_one() {
    auto best = seqs.end();
    int64_t best_score = 0;

    for (auto it = seqs.begin(); it != seqs.end(); ++it) {
        const kv_tree_seq & s = it->second;
        if (s.pinned) {
            continue;
        }

        // only leaf sequences: no other stored sequence extends this one
        bool leaf = true;
        for (const auto & other : seqs) {
            const kv_tree_seq & o = other.second;
            if (o.chain.size() <= s.chain.size()) {
                continue;
            }
            if (std::equal(s.chain.begin(), s.chain.end(), o.chain.begin())) {
                leaf = false;
                break;
            }
        }
        if (!leaf) {
            continue;
        }

        const int64_t score = s.last_used;

        if (best == seqs.end() || score < best_score) {
            best = it;
            best_score = score;
        }
    }

    if (best == seqs.end()) {
        return false;
    }

    remove_seq(best);
    return true;
}

bool kv_tree::enforce_budget() {
    while ((size_t) st.bytes_ram > cfg.ram_limit && demote_one()) {
    }

    // still short: evict in the order anchors -> leaf blocks -> whole leaf sequences
    while ((size_t) st.bytes_ram > cfg.ram_limit) {
        if (!evict_anchor_one() && !evict_block_one() && !evict_seq_one()) {
            break;
        }
    }

    if ((size_t) st.bytes_ram > cfg.ram_limit) {
        fprintf(stderr, "[kv-tree] eviction could not free enough ram (%" PRId64 " > %zu)\n", st.bytes_ram, cfg.ram_limit);
        st.evict_refused++;
        return false;
    }

    return true;
}

void kv_tree::remove_block(std::unordered_map<uint64_t, kv_tree_block>::iterator it) {
    kv_tree_block & b = it->second;

    // anchors whose containing block is gone can never be used again
    for (auto a = anchors.begin(); a != anchors.end(); ) {
        if (a->first.first == b.hash) {
            remove_anchor(a++);
        } else {
            ++a;
        }
    }

    if (b.transient) {
        st.bytes_ram -= (int64_t) b.bytes;
        st.blocks_ram--;
    }

    if (b.on_disk) {
        std::error_code ec;
        std::filesystem::remove(b.path, ec);
        st.bytes_disk -= (int64_t) b.bytes;
        st.blocks_disk--;
    } else {
        st.bytes_ram -= (int64_t) b.bytes;
        st.blocks_ram--;
    }

    auto & v = blocks_at[b.pos0];
    v.erase(std::remove(v.begin(), v.end(), b.hash), v.end());
    if (v.empty()) {
        blocks_at.erase(b.pos0);
    }

    blocks.erase(it);
    st.evicted_blocks++;
}

void kv_tree::park_rollback(const std::vector<std::pair<uint64_t, std::vector<uint8_t>>> & blobs,
                            const kv_tree_match & m,
                            const std::vector<std::pair<uint64_t, llama_pos>> & touched,
                            uint64_t tip) {
    for (const auto & p : blobs) {
        auto it = blocks.find(p.first);
        if (it != blocks.end()) {
            remove_block(it);
        }
    }

    for (const auto & key : touched) {
        auto it = anchors.find(key);
        if (it == anchors.end()) {
            continue;
        }
        if (it->second.refcount <= 1) {
            remove_anchor(it);
        } else {
            it->second.refcount--;
        }
    }

    for (const uint64_t hash : m.path) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.refcount--;
        }
    }
    if (m.n_part > 0) {
        auto it = blocks.find(m.part_hash);
        if (it != blocks.end()) {
            it->second.refcount--;
        }
    }

    seqs.erase(tip);
}

bool kv_tree::park(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens,
                   const std::vector<kv_tree_anchor_in> & checkpoints) {
    st.park_calls++;

    const llama_pos L = (llama_pos) tokens.size();

    if (L <= 0 || io_tgt.pos_max() != L - 1) {
        fprintf(stderr, "[kv-tree] park refused: sequence end %d, expected %d\n", io_tgt.pos_max(), L - 1);
        st.park_refused++;
        return false;
    }

    std::vector<uint64_t> h;
    chain_hashes(tokens, cfg.chunk, h);

    const uint64_t tip = h.back();

    if (seqs.count(tip) != 0) {
        seqs[tip].last_used = ++now;
        if (!checkpoints.empty()) {
            fprintf(stderr, "[kv-tree] park: sequence already stored, %zu checkpoint candidates not adopted\n", checkpoints.size());
            st.anchors_skipped += (int64_t) checkpoints.size();
        }
        st.park_ok++;
        return true;
    }

    const kv_tree_match m = match(tokens);

    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> blobs;
    std::vector<size_t> idxs;

    for (size_t i = m.n_full; i < h.size(); ++i) {
        if (blocks.count(h[i]) != 0) {
            continue;
        }

        const size_t a = i * (size_t) cfg.chunk;
        const size_t b = std::min(tokens.size(), a + (size_t) cfg.chunk);

        std::vector<uint8_t> blob;
        if (!io_tgt.get_range((llama_pos) a, (llama_pos) b, blob)) {
            fprintf(stderr, "[kv-tree] park refused: failed to read range [%zu, %zu)\n", a, b);
            st.park_refused++;
            return false;
        }

        blobs.emplace_back(h[i], std::move(blob));
        idxs.push_back(i);
    }

    // tip anchor: the io state is exactly at L
    std::vector<uint8_t> part_tgt;
    std::vector<uint8_t> part_dft;

    if (!io_tgt.get_partial(part_tgt)) {
        fprintf(stderr, "[kv-tree] park refused: failed to capture the tip state\n");
        st.park_refused++;
        return false;
    }
    if (io_dft != nullptr && !io_dft->get_partial(part_dft)) {
        fprintf(stderr, "[kv-tree] park refused: failed to capture the tip draft state\n");
        st.park_refused++;
        return false;
    }

    for (size_t k = 0; k < blobs.size(); ++k) {
        const size_t i = idxs[k];
        const size_t a = i * (size_t) cfg.chunk;
        const size_t b = std::min(tokens.size(), a + (size_t) cfg.chunk);

        kv_tree_block & nb = blocks[blobs[k].first];

        nb.hash      = blobs[k].first;
        nb.pos0      = (llama_pos) a;
        nb.pos1      = (llama_pos) b;
        nb.refcount  = 1;
        nb.last_used = ++now;
        nb.tokens.assign(tokens.begin() + a, tokens.begin() + b);
        nb.data      = std::move(blobs[k].second);
        nb.bytes     = nb.data.size();

        blocks_at[nb.pos0].push_back(nb.hash);
        st.blocks_ram++;
        st.bytes_ram += (int64_t) nb.data.size();
    }

    for (const uint64_t hash : m.path) {
        kv_tree_block & b = blocks[hash];
        b.refcount++;
        b.heat++;
        b.last_used = ++now;
    }
    if (m.n_part > 0) {
        kv_tree_block & b = blocks[m.part_hash];
        b.refcount++;
        b.heat++;
        b.last_used = ++now;
    }

    if (!store_anchor(tip, L, KV_TREE_ANCHOR_TIP, std::move(part_tgt), std::move(part_dft))) {
        fprintf(stderr, "[kv-tree] park refused: cannot store the tip anchor at %d\n", L);
        st.park_refused++;
        return false;
    }

    // adopt checkpoint candidates: sort by pos, greedy with anchor_step, conflict keeps the earlier one
    std::vector<kv_tree_anchor_in> cand = checkpoints;
    std::sort(cand.begin(), cand.end(), [](const kv_tree_anchor_in & a, const kv_tree_anchor_in & b) {
        return a.pos < b.pos;
    });

    std::vector<std::pair<uint64_t, llama_pos>> touched;
    touched.emplace_back(tip, L);

    llama_pos last_kept = -1;

    for (const kv_tree_anchor_in & c : cand) {
        if (c.pos <= 0 || c.pos > L || c.data_tgt.empty()) {
            st.anchors_skipped++;
            continue;
        }

        const uint64_t blk = containing_block(c.pos, &h);
        if (blk == 0) {
            fprintf(stderr, "[kv-tree] candidate at %d skipped: no containing block on the parked chain\n", c.pos);
            st.anchors_skipped++;
            continue;
        }

        llama_pos prev = -1;
        for (const auto & kv : anchors) {
            if (kv.second.pos >= c.pos || kv.second.pos <= prev) {
                continue;
            }
            if (std::find(h.begin(), h.end(), kv.first.first) != h.end()) {
                prev = kv.second.pos;
            }
        }

        const llama_pos prev_kept = prev > last_kept ? prev : last_kept;

        if (prev_kept >= 0 && c.pos - prev_kept < cfg.anchor_step) {
            st.anchors_skipped++;
            st.anchors_skipped_step++;
            continue;
        }

        if (!store_anchor(blk, c.pos, KV_TREE_ANCHOR_MESSAGE, std::vector<uint8_t>(c.data_tgt), std::vector<uint8_t>(c.data_dft))) {
            fprintf(stderr, "[kv-tree] candidate at %d skipped: anchor store failed\n", c.pos);
            st.anchors_skipped++;
            continue;
        }

        touched.emplace_back(blk, c.pos);
        last_kept = c.pos;
    }

    // anchors that already cover this sequence keep their refcount in sync
    for (auto & kv : anchors) {
        if (kv.second.pos > L) {
            continue;
        }

        bool in_chain = false;
        for (const uint64_t hash : h) {
            if (hash == kv.first.first) {
                in_chain = true;
                break;
            }
        }
        if (!in_chain) {
            continue;
        }

        bool was_touched = false;
        for (const auto & t : touched) {
            if (t.first == kv.first.first && t.second == kv.second.pos) {
                was_touched = true;
                break;
            }
        }
        if (!was_touched) {
            kv.second.refcount++;
            touched.emplace_back(kv.first.first, kv.second.pos);
        }
    }

    kv_tree_seq & s = seqs[tip];
    s.chain     = h;
    s.len       = L;
    s.last_used = ++now;
    s.pinned    = true;

    // pin the new payloads and the new sequence while the budget is enforced, then roll back if it cannot be met
    std::vector<uint64_t> pinned_blocks;
    for (const auto & p : blobs) {
        pinned_blocks.push_back(p.first);
    }
    for (const uint64_t hash : pinned_blocks) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = true;
        }
    }
    for (const auto & key : touched) {
        auto it = anchors.find(key);
        if (it != anchors.end()) {
            it->second.pinned = true;
        }
    }

    const bool fits = enforce_budget();

    for (const uint64_t hash : pinned_blocks) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = false;
        }
    }
    for (const auto & key : touched) {
        auto it = anchors.find(key);
        if (it != anchors.end()) {
            it->second.pinned = false;
        }
    }

    s.pinned = false;

    if (!fits) {
        fprintf(stderr, "[kv-tree] park refused: the budget cannot hold the sequence\n");
        park_rollback(blobs, m, touched, tip);
        st.park_refused++;
        return false;
    }

    st.park_ok++;
    if (cfg.debug) {
        dump();
    }
    return true;
}

kv_tree_restore kv_tree::restore(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens) {
    st.restore_calls++;

    io_tgt.seq_rm(-1, -1);
    if (io_dft != nullptr) {
        io_dft->seq_rm(-1, -1);
    }

    kv_tree_restore res;

    const kv_tree_match m = match(tokens);

    if (m.deep <= 0) {
        st.restore_miss++;
        return res;
    }

    std::vector<uint64_t> cand = m.path;
    if (m.n_part > 0) {
        cand.push_back(m.part_hash);
    }

    llama_pos C = -1;
    uint64_t  c_hash = 0;
    std::vector<std::pair<uint64_t, llama_pos>> path_anchors;

    for (const uint64_t hash : cand) {
        for (auto it = anchors.lower_bound(std::make_pair(hash, 0));
             it != anchors.end() && it->first.first == hash; ++it) {
            if (it->second.pos <= m.deep) {
                path_anchors.emplace_back(hash, it->second.pos);
                if (it->second.pos > C) {
                    C = it->second.pos;
                    c_hash = hash;
                }
            }
        }
    }

    if (C < 0) {
        st.restore_miss++;
        res.heal = m.deep;   // seed a fork anchor while the caller re-prefills
        return res;
    }

    for (const uint64_t hash : cand) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = true;
        }
    }
    for (auto & kv : anchors) {
        if (kv.second.pos <= C && std::find(cand.begin(), cand.end(), kv.first.first) != cand.end()) {
            kv.second.pinned = true;
        }
    }

    bool ok = true;

    std::vector<uint8_t> scratch;

    // blocks covering [0, C): the last one may overshoot and is trimmed below
    for (const uint64_t hash : m.path) {
        kv_tree_block & b = blocks[hash];
        if (b.pos0 >= C) {
            break;
        }
        if (!set_block_payload(io_tgt, b, b.pos0 != 0, scratch)) {
            fprintf(stderr, "[kv-tree] restore failed: cannot load block [%d, %d)\n", b.pos0, b.pos1);
            ok = false;
            break;
        }
    }

    if (ok && m.n_part > 0 && C > blocks[m.part_hash].pos0) {
        kv_tree_block & b = blocks[m.part_hash];
        if (!set_block_payload(io_tgt, b, true, scratch)) {
            fprintf(stderr, "[kv-tree] restore failed: cannot load the partial block [%d, %d)\n", b.pos0, b.pos1);
            ok = false;
        }
    }

    if (ok) {
        io_tgt.seq_rm(C, -1);

        const auto it = anchors.find(std::make_pair(c_hash, C));
        if (it == anchors.end()) {
            ok = false;
        } else {
            kv_tree_anchor & a = it->second;
            if (!load_payload(a) || !io_tgt.set_partial(a.data_tgt.data(), a.data_tgt.size())) {
                fprintf(stderr, "[kv-tree] restore failed: cannot load the state at %d\n", C);
                ok = false;
            } else if (io_dft != nullptr && !a.data_dft.empty()) {
                if (!io_dft->set_partial(a.data_dft.data(), a.data_dft.size())) {
                    fprintf(stderr, "[kv-tree] restore failed: cannot load the draft state at %d\n", C);
                    ok = false;
                }
            }
        }
    }

    std::vector<kv_tree_restore_anchor> out_anchors;
    if (ok) {
        for (const auto & key : path_anchors) {
            kv_tree_anchor & a = anchors.at(key);
            if (!load_payload(a)) {
                continue;   // payload unavailable: fewer rebuilt checkpoints
            }

            kv_tree_restore_anchor ra;
            ra.pos      = a.pos;
            ra.data_tgt = a.data_tgt;
            ra.data_dft = a.data_dft;
            out_anchors.push_back(std::move(ra));
        }
    }

    for (const uint64_t hash : cand) {
        auto it = blocks.find(hash);
        if (it != blocks.end()) {
            it->second.pinned = false;
        }
    }
    for (auto & kv : anchors) {
        if (kv.second.pos <= C && std::find(cand.begin(), cand.end(), kv.first.first) != cand.end()) {
            kv.second.pinned = false;
        }
    }

    settle();

    if (!ok) {
        io_tgt.seq_rm(-1, -1);
        st.restore_miss++;
        return res;
    }

    kv_tree_anchor & ca = anchors.at(std::make_pair(c_hash, C));
    ca.heat++;
    ca.last_used = ++now;

    st.restore_hits++;
    st.tokens_reused += C;

    res.C       = C;
    res.heal    = m.deep > C ? m.deep : -1;
    res.anchors = std::move(out_anchors);

    if (cfg.debug) {
        dump();
    }

    return res;
}

std::string kv_tree::stats_line() const {
    char buf[512];

    snprintf(buf, sizeof(buf),
             "parks=%" PRId64 " ok=%" PRId64 " refused=%" PRId64
             " restore=%" PRId64 " hits=%" PRId64 " miss=%" PRId64
             " anchors=%" PRId64 " skipped=%" PRId64 " step_skips=%" PRId64 " reuse_tok=%" PRId64
             " store=%" PRId64 " load=%" PRId64
             " evicted=%" PRId64 "/%" PRId64 "/%" PRId64 " evict_refused=%" PRId64
             " disk_err=%" PRId64 " ram=%" PRId64 " disk=%" PRId64,
             st.park_calls, st.park_ok, st.park_refused,
             st.restore_calls, st.restore_hits, st.restore_miss,
             st.anchors_added, st.anchors_skipped, st.anchors_skipped_step, st.tokens_reused,
             st.bytes_store, st.bytes_load,
             st.evicted_anchors, st.evicted_blocks, st.evicted_seqs, st.evict_refused,
             st.disk_errors, st.bytes_ram, st.bytes_disk);

    return buf;
}

void kv_tree::dump() const {
    fprintf(stderr, "[kv-tree] blocks: %" PRId64 " ram, %" PRId64 " disk, %" PRId64 " bytes ram, %" PRId64 " bytes disk\n",
            st.blocks_ram, st.blocks_disk, st.bytes_ram, st.bytes_disk);
    fprintf(stderr, "[kv-tree] anchors: %" PRId64 " ram, %" PRId64 " disk, %" PRId64 " added, %" PRId64 " skipped\n",
            st.anchors_ram, st.anchors_disk, st.anchors_added, st.anchors_skipped);
    fprintf(stderr, "[kv-tree] park: %" PRId64 " calls, %" PRId64 " ok, %" PRId64 " refused\n",
            st.park_calls, st.park_ok, st.park_refused);
    fprintf(stderr, "[kv-tree] restore: %" PRId64 " calls, %" PRId64 " hits, %" PRId64 " miss, %" PRId64 " tokens reused\n",
            st.restore_calls, st.restore_hits, st.restore_miss, st.tokens_reused);

    for (const auto & p : blocks) {
        const kv_tree_block & b = p.second;
        fprintf(stderr, "[kv-tree]   block %016" PRIx64 " [%d, %d) ref=%" PRId64 " heat=%" PRId64 " %s %zu bytes\n",
                b.hash, b.pos0, b.pos1, b.refcount, b.heat, b.on_disk ? "disk" : "ram", b.bytes);
    }
    for (const auto & p : anchors) {
        const kv_tree_anchor & a = p.second;
        fprintf(stderr, "[kv-tree]   anchor %016" PRIx64 "@%d kind=%d ref=%" PRId64 " heat=%" PRId64 " %s\n",
                a.blk_hash, a.pos, a.kind, a.refcount, a.heat, a.on_disk ? "disk" : "ram");
    }
    for (const auto & p : seqs) {
        const kv_tree_seq & s = p.second;
        fprintf(stderr, "[kv-tree]   seq tip=%016" PRIx64 " len=%d blocks=%zu\n", p.first, s.len, s.chain.size());
    }
}
