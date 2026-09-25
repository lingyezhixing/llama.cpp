#include "gated_delta_net.cuh"
#include "ggml-cuda/common.cuh"

// ReplaySSM record layout passed to the kernel (see ggml_gated_delta_net)
struct gdn_rec_params {
    float * rec;          // [k | v | g | beta, 2*T_cap, banks]
    int32_t * fold;       // [_, p_i .., read half_i .., write half_i .., bank_i .., check, skip_check]
    float * diag;         // state committed by the previous pass, for the fold self-check (may be null)
    int64_t nb1;          // stride over the 2*T_cap dimension, in floats
    int64_t nb2;          // stride over banks, in floats
    int64_t half_stride;  // T_cap * nb1
    int64_t t_cap;
    int     n_fold;       // number of sequences the fold block holds (n_seq_max)
    int64_t k_off;
    int64_t v_off;
    int64_t g_off;
    int64_t b_off;
};

// shared state update: the fold and the main loop must compile the exact same arithmetic
template <bool KDA, int rows_per_lane, int warp_size>
__device__ __forceinline__ void gdn_state_update(
        float (&s_shard)[rows_per_lane], const float (&k_reg)[rows_per_lane],
        const float * v_t, const float * g_t, const float beta_val, const int col, const int lane) {
    float kv_shard = 0.0f;
    float g_val = 0.0f;
    if constexpr (!KDA) {
        g_val = expf(*g_t);
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            kv_shard += s_shard[r] * k_reg[r];
        }
    } else {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
        }
    }
    const float kv_col = warp_reduce_sum<warp_size>(kv_shard);
    const float delta_col = (!KDA ? (v_t[col] - g_val * kv_col) : (v_t[col] - kv_col)) * beta_val;
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r] = (KDA ? expf(g_t[i]) : g_val) * s_shard[r] + k_reg[r] * delta_col;
    }
}

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const gdn_rec_params rec, const int replay) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    float * rec_k = nullptr;
    float * rec_v = nullptr;
    float * rec_g = nullptr;
    float * rec_b = nullptr;

    int p       = 0; // records to replay
    int tn      = 0; // token count of the batch that wrote them
    int r_half  = 0; // record half the fold reads
    int w_half  = 0; // record half this batch writes
    const int skip_check = replay ? rec.fold[1 + 4 * rec.n_fold + 1] : 0;

    if (replay) {
        // records are addressed by the bank of the sequence, not by the cell of the batch
        const int bank = rec.fold[1 + 3 * rec.n_fold + sequence];

        rec_k = rec.rec + rec.k_off + bank * rec.nb2;
        rec_v = rec.rec + rec.v_off + bank * rec.nb2;
        rec_g = rec.rec + rec.g_off + bank * rec.nb2;
        rec_b = rec.rec + rec.b_off + bank * rec.nb2;

        const int fold_pk = rec.fold[1 + sequence];
        p      = fold_pk & 0xff;
        tn     = (fold_pk >> 8) & 0xff;
        w_half = rec.fold[1 + 2 * rec.n_fold + sequence];
        r_half = rec.fold[1 +     rec.n_fold + sequence];

        if ((w_half != 0 && w_half != 1) || (r_half != 0 && r_half != 1) ||
            bank < 0 || bank >= rec.n_fold || p > (int) rec.t_cap) {
            w_half = 0;
            r_half = 0;
            p      = 0;
        }
    }

    if (replay) {
        // fold: replay the accepted prefix of the previous batch from the records and commit the
        // resulting state. The committed plane never holds the state after this batch's tokens.
        const float * rk = rec_k + r_half * rec.half_stride + iq1 * S_v;
        const float * rv = rec_v + r_half * rec.half_stride + h_idx * S_v;
        const float * rg = rec_g + r_half * rec.half_stride + h_idx * (KDA ? S_v : 1);
        const float * rb = rec_b + r_half * rec.half_stride + h_idx;

        for (int t = 0; t < p; t++) {
            const float * k_t = rk + t * rec.nb1;
            const float * v_t = rv + t * rec.nb1;
            const float * g_t = rg + t * rec.nb1;
            const float * b_t = rb + t * rec.nb1;

            const float beta_val = *b_t;

            float k_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] = k_t[r * warp_size + lane];
            }

            gdn_state_update<KDA, rows_per_lane, warp_size>(s_shard, k_reg, v_t, g_t, beta_val, col, lane);
        }

        // when the whole previous batch was accepted, the fold must reproduce the state that pass
        // committed; count the elements that differ (0 expected). A restored state has no committed
        // pass to compare against, so the check is skipped for the batch that carries it
        if (skip_check == 0 && rec.diag != nullptr && p > 0 && p == tn) {
            const float * dg = rec.diag + state_out_offset;
            int mism = 0;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                mism += (s_shard[r] != dg[col * S_v + i]);
            }
            for (int off = warp_size / 2; off > 0; off >>= 1) {
                mism += __shfl_xor_sync(0xffffffff, mism, off);
            }
            if (lane == 0 && mism > 0) {
                atomicAdd(rec.fold + 1 + 4 * rec.n_fold, mism);
            }
        }

        // publish the replayed committed state, the copy op moves it into the cache
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }

    const bool record = replay && n_tokens <= rec.t_cap;

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            gdn_state_update<KDA, rows_per_lane, warp_size>(s_shard, k_reg, v_t, g_t, beta_val, col, lane);

            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            gdn_state_update<KDA, rows_per_lane, warp_size>(s_shard, k_reg, v_t, g_t, beta_val, col, lane);

            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        if (record) {
            // raw inputs of this batch; k is shared inside a GQA group, so sibling blocks store the same values
            float * wk = rec_k + w_half * rec.half_stride + t * rec.nb1 + iq1 * S_v;
            float * wv = rec_v + w_half * rec.half_stride + t * rec.nb1 + h_idx * S_v;
            float * wg = rec_g + w_half * rec.half_stride + t * rec.nb1 + h_idx * (KDA ? S_v : 1);
            float * wb = rec_b + w_half * rec.half_stride + t * rec.nb1 + h_idx;

#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                wk[i] = k_reg[r];
                if constexpr (KDA) {
                    wg[i] = g_t[i];
                }
            }
            if (lane == 0) {
                wv[col] = v_t[col];
                wb[0]   = beta_val;
                if constexpr (!KDA) {
                    wg[0] = *g_t;
                }
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            // Replay mode commits only the folded state, so it writes no per-token snapshots.
            const int target_slot = (int) n_tokens - 1 - t;
            if (!replay && target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if (!keep_rs_t && !replay) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }

    if (replay && !record) {
        // batches without records cannot be rolled back, so they commit the state after the batch;
        // recording batches leave the plane at the replayed committed state
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }

    // remember the state after this pass, so the next pass can check its fold against it
    if (rec.diag != nullptr) {
        float * dg = rec.diag + state_out_offset;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            dg[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const gdn_rec_params & rec, const int replay, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rec, replay);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rec, replay);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rec, replay);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, rec, replay);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    ggml_tensor * src_rec  = dst->src[6];
    ggml_tensor * src_fold = dst->src[7];
    const bool replay = src_rec != nullptr;
    GGML_ASSERT(replay == (src_fold != nullptr));

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    gdn_rec_params rec_params = {};
    if (replay) {
        GGML_ASSERT(K >= 1);
        const int64_t k_w = nek0 * nek1;
        const int64_t v_w = nev0 * nev1;
        const int64_t g_w = kda ? nev0 * nev1 : nev1;
        rec_params.rec         = (float *) src_rec->data;
        rec_params.fold        = (int32_t *) src_fold->data;
        rec_params.diag        = dst->src[8] != nullptr ? (float *) dst->src[8]->data : nullptr;
        rec_params.nb1         = src_rec->nb[1] / sizeof(float);
        rec_params.nb2         = src_rec->nb[2] / sizeof(float);
        rec_params.half_stride = (src_rec->ne[1] / 2) * rec_params.nb1;
        rec_params.t_cap       = src_rec->ne[1] / 2;
        rec_params.n_fold      = (int) ((src_fold->ne[0] - 3) / 4);
        rec_params.k_off       = 0;
        rec_params.v_off       = k_w;
        rec_params.g_off       = k_w + v_w;
        rec_params.b_off       = k_w + v_w + g_w;
    }

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    const int replay_i = replay ? 1 : 0;

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rec_params, replay_i, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rec_params, replay_i, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rec_params, replay_i, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, rec_params, replay_i, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
