// Content-addressed tree storage for attention KV and recurrent state anchors.
// Pure logic is separated from llama_context I/O via kv_tree_io.

#pragma once

#include "llama.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct kv_tree_config {
    int         chunk       = 512;
    int         anchor_step = 32768;
    int         fork_step   = 8192;
    size_t      ram_limit   = 8ull  << 30;
    size_t      disk_limit  = 64ull << 30;
    std::string disk_dir;
    bool        debug       = false;
};

enum kv_tree_anchor_kind {
    KV_TREE_ANCHOR_TIP      = 0,
    KV_TREE_ANCHOR_MESSAGE  = 1,
    KV_TREE_ANCHOR_ONDEMAND = 2,
};

// anchor candidate captured by the caller (e.g. a server prompt checkpoint)
struct kv_tree_anchor_in {
    int64_t tok = 0;
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
};

struct kv_tree_stats {
    int64_t blocks_ram      = 0;
    int64_t blocks_disk     = 0;
    int64_t anchors_ram     = 0;
    int64_t anchors_disk    = 0;
    int64_t bytes_ram       = 0;
    int64_t bytes_disk      = 0;
    int64_t park_calls      = 0;
    int64_t park_ok         = 0;
    int64_t park_refused    = 0;
    int64_t restore_calls   = 0;
    int64_t restore_hits    = 0;
    int64_t restore_miss    = 0;
    int64_t anchors_added   = 0;
    int64_t anchors_skipped = 0;
    int64_t anchors_skipped_step = 0;
    int64_t tokens_reused   = 0;
    int64_t bytes_store     = 0;
    int64_t bytes_load      = 0;
    int64_t evicted_anchors = 0;
    int64_t evicted_blocks  = 0;
    int64_t evicted_seqs    = 0;
    int64_t evict_refused   = 0;
    int64_t disk_errors     = 0;
};

// I/O seam: the harness uses a fake, the server a llama_context adapter
struct kv_tree_io {
    virtual ~kv_tree_io() = default;

    // attention KV of [p0, p1) of the sequence, as a self-contained blob
    virtual bool get_range(llama_pos p0, llama_pos p1, std::vector<uint8_t> & out) = 0;
    // load a blob produced by get_range; append = keep the existing cells
    virtual bool set_range(const uint8_t * data, size_t size, bool append) = 0;
    // recurrent (PARTIAL_ONLY) state of the sequence at its current end
    virtual bool get_partial(std::vector<uint8_t> & out) = 0;
    virtual bool set_partial(const uint8_t * data, size_t size) = 0;
    // drop attention KV in [p0, p1); p1 = -1 means "to the end", p0 = -1 means "everything"
    virtual bool seq_rm(llama_pos p0, llama_pos p1) = 0;
    // largest position currently in the sequence, -1 when empty
    virtual llama_pos pos_max() = 0;
};

struct kv_tree_io_llama : public kv_tree_io {
    kv_tree_io_llama(llama_context * ctx, llama_seq_id seq_id);

    bool get_range(llama_pos p0, llama_pos p1, std::vector<uint8_t> & out) override;
    bool set_range(const uint8_t * data, size_t size, bool append) override;
    bool get_partial(std::vector<uint8_t> & out) override;
    bool set_partial(const uint8_t * data, size_t size) override;
    bool seq_rm(llama_pos p0, llama_pos p1) override;
    llama_pos pos_max() override;

    llama_context * ctx;
    llama_seq_id    seq_id;
};

// media chunk inside the token stream; for M-RoPE n_pos < n_tok, so token index != position
struct kv_tree_media {
    int64_t  idx   = 0;   // start token index of the chunk
    uint64_t id    = 0;   // opaque identity hash (server: over mtmd_input_chunk_get_id)
    int32_t  n_tok = 0;   // tokens occupied by the chunk
    int32_t  n_pos = 0;   // positions advanced by the chunk
};

// one chunk of attention KV, content-addressed
struct kv_tree_block {
    uint64_t    hash     = 0;   // chained content hash
    int64_t     tok0     = 0;   // start token index
    llama_pos   pos0     = 0;   // [pos0, pos1) covered by this block
    llama_pos   pos1     = 0;
    std::vector<kv_tree_media> media;   // entries with idx in [tok0, tok0 + tokens.size())
    int64_t     refcount = 0;   // stored sequences referencing this block
    int64_t     heat     = 0;
    int64_t     last_used = 0;
    bool        on_disk  = false;
    bool        pinned   = false;
    bool        transient = false;
    std::string path;
    size_t      bytes    = 0;   // payload size, valid in both tiers
    std::vector<llama_token> tokens;
    std::vector<uint8_t>     data;
};

// recurrent state (PARTIAL_ONLY) captured at pos
struct kv_tree_anchor {
    uint64_t    blk_hash = 0;   // hash of the block containing pos
    int64_t     tok      = 0;   // token index of the captured state
    llama_pos   pos      = 0;
    int         kind     = KV_TREE_ANCHOR_MESSAGE;
    int64_t     refcount = 0;
    int64_t     heat     = 0;
    int64_t     last_used = 0;
    bool        on_disk  = false;
    bool        pinned   = false;
    bool        transient = false;
    std::string path;
    size_t      bytes    = 0;   // payload size, valid in both tiers
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
};

struct kv_tree_seq {
    std::vector<uint64_t> chain;   // block hashes in order
    llama_pos   len       = 0;
    int64_t     last_used = 0;
    bool        pinned    = false;
};

// verified prefix of a request against the stored blocks
struct kv_tree_match {
    std::vector<uint64_t> path;    // hashes of fully matched blocks
    size_t      n_part    = 0;     // tokens verified inside the last partial block
    uint64_t    part_hash = 0;     // hash of the partially matched block
    llama_pos   deep      = 0;     // deepest verified token index
};

struct kv_tree_restore_anchor {
    int64_t     tok = 0;
    llama_pos   pos = 0;
    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;
};

struct kv_tree_restore {
    llama_pos C = -1;              // restore point in tokens, -1 = caller must do a full prefill
    llama_pos heal = -1;           // capture an anchor when the prefill crosses this token index
    std::vector<kv_tree_restore_anchor> anchors; // path anchors up to C, ascending
};

class kv_tree {
public:
    explicit kv_tree(const kv_tree_config & cfg);

    // store the sequence tokens[0, L); the io state must end exactly at the last KV cell
    bool park(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens,
              const std::vector<kv_tree_media> & media,
              const std::vector<kv_tree_anchor_in> & checkpoints);

    // load the deepest anchor-covered prefix of tokens into the (cleared) sequence
    kv_tree_restore restore(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens,
                            const std::vector<kv_tree_media> & media, bool leave_one = false);

    // capture the state after tok tokens (the io state must be exactly there) as a fork anchor;
    // tokens is the sequence content, used to attach the anchor to the right chain block
    bool capture_anchor(kv_tree_io & io_tgt, kv_tree_io * io_dft, const std::vector<llama_token> & tokens,
                        const std::vector<kv_tree_media> & media, int64_t tok);

    // release the stored sequence that matches a prefix of tokens; no-op when nothing matches
    bool drop_seq(const std::vector<llama_token> & tokens, const std::vector<kv_tree_media> & media);

    // verified prefix of a request against the stored blocks
    kv_tree_match match(const std::vector<llama_token> & tokens, const std::vector<kv_tree_media> & media) const;

    const kv_tree_stats & stats() const { return st; }

    std::string stats_line() const;

    void dump() const;

private:
    bool store_anchor(uint64_t blk_hash, int64_t tok, llama_pos pos, int kind,
                      std::vector<uint8_t> && data_tgt, std::vector<uint8_t> && data_dft);

    uint64_t containing_block(llama_pos pos, const std::vector<uint64_t> & chain) const;

    void remove_anchor(std::map<std::pair<uint64_t, llama_pos>, kv_tree_anchor>::iterator it);

    bool enforce_budget();

    bool has_successor(const kv_tree_block & b) const;

    bool demote_one();

    bool evict_anchor_one();
    bool evict_block_one();
    bool evict_seq_one();

    void remove_seq(std::unordered_map<uint64_t, kv_tree_seq>::iterator it);

    void park_rollback(const std::vector<std::pair<uint64_t, std::vector<uint8_t>>> & blobs,
                       const kv_tree_match & m,
                       const std::vector<std::pair<uint64_t, llama_pos>> & touched,
                       uint64_t tip);

    void remove_block(std::unordered_map<uint64_t, kv_tree_block>::iterator it);

    bool load_payload(kv_tree_block & b);
    bool load_payload(kv_tree_anchor & a);
    bool set_block_payload(kv_tree_io & io, kv_tree_block & b, bool append, std::vector<uint8_t> & scratch);
    bool demote_block(kv_tree_block & b);
    bool demote_anchor(kv_tree_anchor & a);
    void settle();

    bool write_disk(const std::string & path, const std::vector<uint8_t> & buf);
    bool read_disk(const std::string & path, std::vector<uint8_t> & out);
    std::string block_path(uint64_t hash) const;
    std::string anchor_path(uint64_t blk_hash, llama_pos pos) const;

    kv_tree_config cfg;
    kv_tree_stats  st;

    std::unordered_map<uint64_t, kv_tree_block> blocks;
    std::map<int64_t, std::vector<uint64_t>> blocks_at;            // tok0 -> hashes
    std::map<llama_pos, std::vector<uint64_t>> blocks_by_pos0;    // pos0 -> hashes
    std::map<std::pair<uint64_t, llama_pos>, kv_tree_anchor> anchors;
    std::unordered_map<uint64_t, kv_tree_seq> seqs;                  // tip hash -> sequence

    int64_t now = 0;
};
