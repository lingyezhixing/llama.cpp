import io

p = r'D:\LLM\Backend\src\llama.cpp-my\ggml\src\ggml-cuda\gated_delta_net.cu'
s = io.open(p, encoding='utf-8').read()

kernel = '''
// C=2 state columns per warp (S_v == 128, non-KDA, !keep_rs_t, multi-token only).
// Rationale (measured on V100, T03): the recurrent scan is latency-bound: one warp per column pays the
// memory latency and a 5-step butterfly per token. Handling 2 columns per warp amortizes both over
// independent work: 35% less time at 8 warps/block occupancy; the state stays in fp32 registers
// (8 rows x 2 columns per lane).
// lane (gq, sub): gq = lane/16 selects the column, sub = lane%16 owns rows [sub*8, sub*8+8).
template <bool VEC4>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < 128 ? ggml_cuda_get_physical_warp_size() : 128) * 4 / 2, 2)
gated_delta_net_c2_cuda(const float * q,
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
                        int           K) {
    constexpr int S_v = 128;
    constexpr int C   = 2;
    constexpr int LPC = 32 / C;      // lanes per column
    constexpr int RPC = S_v / LPC;   // rows per lane
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      warp     = threadIdx.y;
    const int      gq       = lane / LPC;
    const int      sub      = lane % LPC;
    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);
    float *       attn_data = dst;
    const int64_t state_in_offset  = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    state += (sequence * H + h_idx) * S_v * S_v;
    curr_state += state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    const int col = (blockIdx.z * blockDim.y + warp) * C + gq;
    float     s[RPC];
    if constexpr (VEC4) {
        const float * st = curr_state + col * S_v + sub * RPC;
#pragma unroll
        for (int r = 0; r < RPC; r += 4) {
            const float4 t4 = *reinterpret_cast<const float4 *>(st + r);
            s[r] = t4.x; s[r + 1] = t4.y; s[r + 2] = t4.z; s[r + 3] = t4.w;
        }
    }
    ggml_cuda_pdl_sync();
    const float * k_base = k + iq3 * sq3 + iq1 * sq1;
    const float * q_base = q + iq3 * sq3 + iq1 * sq1;
    const float * v_base = v + sequence * sv3 + h_idx * sv1;
    const int64_t gb_base = sequence * sb3 + h_idx * sb1;

    for (int t = 0; t < n_tokens; t++) {
        float kk[RPC], qq[RPC];
        if constexpr (VEC4) {
#pragma unroll
            for (int r = 0; r < RPC; r += 4) {
                const float4 t4 = *reinterpret_cast<const float4 *>(k_base + t * sq2 + sub * RPC + r);
                kk[r] = t4.x; kk[r + 1] = t4.y; kk[r + 2] = t4.z; kk[r + 3] = t4.w;
            }
#pragma unroll
            for (int r = 0; r < RPC; r += 4) {
                const float4 t4 = *reinterpret_cast<const float4 *>(q_base + t * sq2 + sub * RPC + r);
                qq[r] = t4.x; qq[r + 1] = t4.y; qq[r + 2] = t4.z; qq[r + 3] = t4.w;
            }
        } else {
            for (int r = 0; r < RPC; r++) { kk[r] = k_base[t * sq2 + sub * RPC + r]; qq[r] = q_base[t * sq2 + sub * RPC + r]; }
        }
        const float g_val = expf(g[gb_base + t * sb2]);
        const float b_val = beta[gb_base + t * sb2];
        const float v_val = v_base[t * sv2 + col];

        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPC; r++) { kv += s[r] * kk[r]; }
#pragma unroll
        for (int o = LPC / 2; o > 0; o >>= 1) { kv += __shfl_xor_sync(0xffffffffu, kv, o); }
        const float delta = (v_val - g_val * kv) * b_val;
        float at = 0.0f;
#pragma unroll
        for (int r = 0; r < RPC; r++) { s[r] = g_val * s[r] + kk[r] * delta; at += s[r] * qq[r]; }
#pragma unroll
        for (int o = LPC / 2; o > 0; o >>= 1) { at += __shfl_xor_sync(0xffffffffu, at, o); }
        if (sub == 0) { attn_data[col] = at * scale; }
        attn_data += S_v * H;
    }

    if constexpr (VEC4) {
        float * st = state + col * S_v + sub * RPC;
#pragma unroll
        for (int r = 0; r < RPC; r += 4) {
            *reinterpret_cast<float4 *>(st + r) = make_float4(s[r], s[r + 1], s[r + 2], s[r + 3]);
        }
    }
}

'''
anchor = 'template <bool KDA, bool keep_rs_t>\nstatic void launch_gated_delta_net('
assert anchor in s
s = s.replace(anchor, kernel + anchor, 1)

old_case = '''        case 128: {
            if (vec4) {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, true>, launch_params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            } else {'''
new_case = '''        case 128: {
            static const bool gdn_c2_disabled = getenv("GGML_CUDA_GDN_C2") != nullptr;
            if (vec4 && !KDA && !keep_rs_t && !gdn_c2_disabled) {
                const dim3 grid_c2(H, n_seqs, (S_v + num_warps*2 - 1) / (num_warps*2));
                const ggml_cuda_kernel_launch_params lp_c2 = ggml_cuda_kernel_launch_params(grid_c2, block_dims, 0, stream);
                ggml_cuda_kernel_launch(gated_delta_net_c2_cuda<true>, lp_c2,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            } else if (vec4) {
                ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t, true>, launch_params,
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                    n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                    sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K);
            } else {'''
assert old_case in s
s = s.replace(old_case, new_case, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('integrated C=2 branch (env GGML_CUDA_GDN_C2 disables)')
