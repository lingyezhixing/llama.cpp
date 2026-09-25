#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)
        int32_t   tail = -1;

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

    // ReplaySSM: per-token raw-input records for GDN speculative rollback.
    // One packed tensor per layer, [k | v | g | beta, 2*T_cap, mem_size] F32; the 2*T_cap dimension is
    // double-buffered so a fold reads the previous batch while the current batch writes new records.
    std::vector<ggml_tensor *> rec_l;

    // optional per-layer copy of the state committed by the last pass, used by the fold self-check
    std::vector<ggml_tensor *> rec_diag;

    // device tensor with the fold parameters, I32: [half, p_0, ..., p_{n_seq_max-1}, _, _, check];
    // written directly by the graph input with ggml_backend_tensor_set
    ggml_tensor * rec_fold_t = nullptr;

    // record block widths in floats per token per layer; 0 when the model has no records
    uint32_t rec_floats_k = 0;
    uint32_t rec_floats_v = 0;
    uint32_t rec_floats_g = 0;
    uint32_t rec_floats_b = 0;

    bool rec_enabled() const { return rec_floats_k > 0; }
    uint32_t rec_t_cap() const { return n_rs_seq + 1; }

    // replay state: tracking of the batches that write records, per sequence
    bool     rec_replay = false; // GGML_CUDA_GDN_REPLAY
    bool     rec_check  = false; // GGML_CUDA_GDN_REPLAY_CHECK
    std::vector<uint32_t> rec_half_of;   // the seq's last record half; the next recording batch uses the other
    std::vector<uint8_t>  rec_have_rec;  // the seq's last batch was a recording batch, its records can be replayed
    std::vector<uint32_t> rec_n;         // that batch's token count
    std::vector<int32_t>  rec_pos0;      // that batch's first token position (INT32_MIN = never)
    std::vector<int32_t>  rec_p_pending; // replay count to apply once after a state restore (INT32_MIN = none)
    std::vector<uint8_t>  rec_check_bad; // the seq's diag does not match the committed state yet, skip the fold check

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;
    size_t size_rec_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges_s) const;
    void state_write_replay(llama_io_write_i & io, llama_seq_id seq_id) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
    bool state_read_replay(llama_io_read_i & io, llama_seq_id dest_seq_id);

    void state_clear(llama_seq_id seq_id, uint32_t cell_head, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;
    ggml_tensor * get_rec_l(int32_t il) const;
    ggml_tensor * get_rec_fold_t() const;
    ggml_tensor * get_rec_diag(int32_t il) const;

    // records the current batch replays for batch-local seq i, packed as p | (previous token count << 8);
    // the sequence owning the cell at position i of the ubatch (-1 if unassigned)
    llama_seq_id get_rec_seq(int i) const;
    // the packed count lets the op verify a fold that replayed the whole previous batch
    uint32_t get_rec_p(int i, int32_t pos0) const;
    // true once per ubatch, for the call that owns the fold parameters of that ubatch
    bool rec_fold_take() const;
    void rec_commit(int i, uint32_t n_tokens, int32_t pos0) const;
    bool rec_enabled() const;
    bool rec_check_enabled() const;
    uint32_t rec_t_cap() const;

    // fold parameter block: [_, p_i .., read half_i .., write half_i .., bank_i .., self-check, skip check]
    uint32_t rec_fold_size() const;
    uint32_t rec_fold_check_off() const;
    void rec_fold_fill(std::vector<int32_t> & fold, uint32_t n_seqs, const int32_t * pos0) const;
    void rec_forget(int i) const;

    int32_t s_copy(int i) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    // the fold parameters of this context belong to its current ubatch, and the side effects that
    // produce them run once per ubatch even though the memory appears in several graph inputs
    mutable size_t rec_fold_done = (size_t) -1;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
