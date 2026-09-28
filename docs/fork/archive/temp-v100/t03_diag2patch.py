import io

p = r'<TEMP>\v100\t03_existing.cu'
s = io.open(p, encoding='utf-8').read()

marker = '// ---------------------------------------------------------------- host'
diag2 = '''
// ---------------------------------------------------------------- fine diagnostic (5 checkpoints, base/prefetch)
template <bool PF>
__global__ void __launch_bounds__((32 < S_v ? 32 : S_v) * NWARPS, 2)
diag2_kernel(const float * q, const float * k, const float * v, const float * g, const float * beta,
             const float * curr_state, float * dst, float * state,
             int64_t H_, int64_t n_tokens, int64_t n_seqs,
             int64_t sq1, int64_t sq2, int64_t sq3, int64_t sv1, int64_t sv2, int64_t sv3,
             int64_t sb1, int64_t sb2, int64_t sb3, float scale, unsigned long long * prof) {
    const uint32_t h_idx = blockIdx.x; const uint32_t sequence = blockIdx.y;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    float * attn_data = dst + (sequence * n_tokens * H_ + h_idx) * S_v;
    state += (sequence * H_ + h_idx) * S_v * S_v;
    curr_state += sequence * H_ * S_v * S_v + h_idx * S_v * S_v + col * S_v;
    float s_shard[4];
    { const float4 s4 = ld4(curr_state + 4*lane); s_shard[0]=s4.x; s_shard[1]=s4.y; s_shard[2]=s4.z; s_shard[3]=s4.w; }
    const float * k_base = k + h_idx * sq1 + sequence * sq3;
    const float * q_base = q + h_idx * sq1 + sequence * sq3;
    const float * v_base = v + sequence * sv3 + h_idx * sv1;
    const int64_t gb_base = sequence * sb3 + h_idx * sb1;
    const bool meas = (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0);
    unsigned long long c_ld=0, c_kuse=0, c_red=0, c_upd=0;

    float4 k4n, q4n; float gn, bn, vn;
    if (PF) {
        k4n = __ldg(reinterpret_cast<const float4 *>(k_base) + lane);
        q4n = __ldg(reinterpret_cast<const float4 *>(q_base) + lane);
        gn = __ldg(g + gb_base); bn = __ldg(beta + gb_base); vn = __ldg(v_base + col);
    }
    for (int t = 0; t < n_tokens; t++) {
        unsigned long long a0 = meas ? clock64() : 0;
        float4 k4, q4; float g_raw, b_val, v_val;
        if (PF) {
            k4 = k4n; q4 = q4n; g_raw = gn; b_val = bn; v_val = vn;
            if (t+1 < n_tokens) {
                k4n = __ldg(reinterpret_cast<const float4 *>(k_base + (t+1)*sq2) + lane);
                q4n = __ldg(reinterpret_cast<const float4 *>(q_base + (t+1)*sq2) + lane);
                gn = __ldg(g + gb_base + (t+1)*sb2); bn = __ldg(beta + gb_base + (t+1)*sb2);
                vn = __ldg(v_base + (t+1)*sv2 + col);
            }
        } else {
            k4 = ld4(k_base + t*sq2 + 4*lane);
            q4 = ld4(q_base + t*sq2 + 4*lane);
            g_raw = g[gb_base + t*sb2]; b_val = beta[gb_base + t*sb2]; v_val = v_base[t*sv2 + col];
        }
        unsigned long long a1 = meas ? clock64() : 0;
        float kv_shard = 0.f;
        kv_shard += s_shard[0]*k4.x; kv_shard += s_shard[1]*k4.y;
        kv_shard += s_shard[2]*k4.z; kv_shard += s_shard[3]*k4.w;
        unsigned long long a2 = meas ? clock64() : 0;
        const float g_val = expf(g_raw);
        const float kv_col = warp_reduce_sum(kv_shard);
        const float delta_col = (v_val - g_val*kv_col)*b_val;
        unsigned long long a3 = meas ? clock64() : 0;
        float attn_partial = 0.f;
        s_shard[0] = g_val*s_shard[0] + k4.x*delta_col; attn_partial += s_shard[0]*q4.x;
        s_shard[1] = g_val*s_shard[1] + k4.y*delta_col; attn_partial += s_shard[1]*q4.y;
        s_shard[2] = g_val*s_shard[2] + k4.z*delta_col; attn_partial += s_shard[2]*q4.z;
        s_shard[3] = g_val*s_shard[3] + k4.w*delta_col; attn_partial += s_shard[3]*q4.w;
        const float attn_col = warp_reduce_sum(attn_partial);
        if (lane == 0) attn_data[col] = attn_col * scale;
        attn_data += S_v * H_;
        unsigned long long a4 = meas ? clock64() : 0;
        if (meas) { c_ld += a1-a0; c_kuse += a2-a1; c_red += a3-a2; c_upd += a4-a3; }
    }
    *reinterpret_cast<float4 *>(state + col*S_v + 4*lane) = make_float4(s_shard[0],s_shard[1],s_shard[2],s_shard[3]);
    if (meas) { prof[0]=c_ld; prof[1]=c_kuse; prof[2]=c_red; prof[3]=c_upd; }
}

'''
assert marker in s
s = s.replace(marker, diag2 + marker, 1)

old = '''        unsigned long long hp[8]; CUDA_CHECK(cudaMemcpy(hp,d_prof,8*8,cudaMemcpyDeviceToHost));
        unsigned long long tt = hp[0]+hp[1]+hp[2]+hp[3];
        printf("per-token cycles (warp0/blk0): load=%llu (%.0f%%) expf=%llu (%.0f%%) kv_reduce=%llu (%.0f%%) upd+attn_reduce+store=%llu (%.0f%%)  total=%llu (%.0f/token)\\n",
               hp[0],100.0*hp[0]/tt, hp[1],100.0*hp[1]/tt, hp[2],100.0*hp[2]/tt, hp[3],100.0*hp[3]/tt, tt, (double)tt/t_tokens);'''
new = '''        unsigned long long hp[8]; CUDA_CHECK(cudaMemcpy(hp,d_prof,8*8,cudaMemcpyDeviceToHost));
        unsigned long long tt = hp[0]+hp[1]+hp[2]+hp[3];
        printf("coarse: load=%llu expf+red=%llu upd+red+st=%llu tot=%llu (%.0f/token)\\n", hp[0],hp[1],hp[2],hp[3],tt,(double)tt/t_tokens);
        for (int pf = 0; pf < 2; pf++) {
            CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemset(d_prof,0,sizeof(unsigned long long)*H*NWARPS*8));
            if (pf) diag2_kernel<true ><<<grid,block>>>(d_q,d_k,d_v,d_g,d_b,d_sin,d_out,d_sout,H,t_tokens,1,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,scale,d_prof);
            else    diag2_kernel<false><<<grid,block>>>(d_q,d_k,d_v,d_g,d_b,d_sin,d_out,d_sout,H,t_tokens,1,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,scale,d_prof);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hp,d_prof,8*8,cudaMemcpyDeviceToHost));
            tt = hp[0]+hp[1]+hp[2]+hp[3];
            printf("fine(%s): loads_issued=%llu (%.0f/tok) use_k4+4fma=%llu (%.0f/tok) expf+reduce1+delta=%llu (%.0f/tok) upd+use_q4+reduce2+store=%llu (%.0f/tok)\\n",
                   pf?"prefetch":"base    ", hp[0],(double)hp[0]/t_tokens, hp[1],(double)hp[1]/t_tokens, hp[2],(double)hp[2]/t_tokens, hp[3],(double)hp[3]/t_tokens);
        }'''
assert old in s
s = s.replace(old, new, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('fine diag added')
