// T03: is the EXISTING gated_delta_net kernel load-latency bound? (implementer, 2026-09-22)
// Copies the existing kernel's non-KDA / use_vec4 path + a software-pipelined (prefetch) variant.
// Layouts identical to the previous harness (and to llama.cpp):
//   q,k : [T][H][D]   v : [H][T][D]   g,b : [H][T]
//   state: M[col][i] = S[i][col] at state[h*D*D + col*D + i]      out: out[(h*T+t)*D + j]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define D 128
#define H 32
#define T 512
#define NWARPS 4
#define S_v 128

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float4 ld4(const float * p) { return *reinterpret_cast<const float4 *>(p); }

// ---------------------------------------------------------------- reference (for correctness)
__global__ void ref_kernel(const float * q, const float * k, const float * v, const float * g, const float * b,
                           const float * s_in, float * out, float * s_out, int t_tokens, float scale) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    float col[D];
    const float * sin_h = s_in + (size_t)h*D*D;
    for (int i = 0; i < D; i++) col[i] = sin_h[j*D + i];
    for (int t = 0; t < t_tokens; t++) {
        const float * q_t = q + (size_t)(t*H + h)*D;
        const float * k_t = k + (size_t)(t*H + h)*D;
        const float * v_t = v + (size_t)(h*t_tokens + t)*D;
        const float gt = expf(g[h*t_tokens + t]);
        const float bt = b[h*t_tokens + t];
        float kv = 0.f, at = 0.f;
        for (int i = 0; i < D; i++) kv += col[i]*k_t[i];
        const float delta = (v_t[j] - gt*kv)*bt;
        for (int i = 0; i < D; i++) { col[i] = gt*col[i] + k_t[i]*delta; at += col[i]*q_t[i]; }
        out[(size_t)(t*H + h)*D + j] = at*scale;
    }
    float * so = s_out + (size_t)h*D*D;
    for (int i = 0; i < D; i++) so[j*D + i] = col[i];
}

// ---------------------------------------------------------------- existing kernel (vec4, non-KDA, no keep_rs_t)
template <bool PREFETCH>
__global__ void __launch_bounds__((32 < S_v ? 32 : S_v) * NWARPS, 2)
existing_kernel(const float * q, const float * k, const float * v, const float * g, const float * beta,
                const float * curr_state, float * dst, float * state,
                int64_t H_, int64_t n_tokens, int64_t n_seqs,
                int64_t sq1, int64_t sq2, int64_t sq3,
                int64_t sv1, int64_t sv2, int64_t sv3,
                int64_t sb1, int64_t sb2, int64_t sb3,
                float scale, int64_t state_slot_stride, unsigned long long * prof) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;
    float * attn_data = dst;
    const int64_t state_in_offset  = sequence * H_ * S_v * S_v + h_idx * S_v * S_v;
    state += (sequence * H_ + h_idx) * S_v * S_v;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H_ + h_idx) * S_v;

    float s_shard[4];
    {
        const float4 s4 = ld4(curr_state + 4*lane);
        s_shard[0] = s4.x; s_shard[1] = s4.y; s_shard[2] = s4.z; s_shard[3] = s4.w;
    }
    unsigned long long t0 = (prof && threadIdx.x == 0 && threadIdx.y == 0) ? clock64() : 0ull;

    const float * k_base = k + h_idx * sq1 + sequence * sq3;
    const float * q_base = q + h_idx * sq1 + sequence * sq3;
    const float * v_base = v + sequence * sv3 + h_idx * sv1;
    const int64_t gb_base = sequence * sb3 + h_idx * sb1;

    float4 k4n, q4n; float gn, bn, vn;
    if (PREFETCH) {
        k4n = __ldg(reinterpret_cast<const float4 *>(k_base + 0*sq2) + lane);
        q4n = __ldg(reinterpret_cast<const float4 *>(q_base + 0*sq2) + lane);
        gn  = __ldg(g + gb_base + 0*sb2);
        bn  = __ldg(beta + gb_base + 0*sb2);
        vn  = __ldg(v_base + 0*sv2 + col);
    }
    for (int t = 0; t < n_tokens; t++) {
        float4 k4, q4; float g_val, b_val, v_val;
        if (PREFETCH) {
            k4 = k4n; q4 = q4n; g_val = expf(gn); b_val = bn; v_val = vn;
            if (t+1 < n_tokens) {
                k4n = __ldg(reinterpret_cast<const float4 *>(k_base + (t+1)*sq2) + lane);
                q4n = __ldg(reinterpret_cast<const float4 *>(q_base + (t+1)*sq2) + lane);
                gn  = __ldg(g + gb_base + (t+1)*sb2);
                bn  = __ldg(beta + gb_base + (t+1)*sb2);
                vn  = __ldg(v_base + (t+1)*sv2 + col);
            }
        } else {
            k4 = ld4(k_base + t*sq2 + 4*lane);
            q4 = ld4(q_base + t*sq2 + 4*lane);
            g_val = expf(g[gb_base + t*sb2]);
            b_val = beta[gb_base + t*sb2];
            v_val = v_base[t*sv2 + col];
        }
        float kv_shard = 0.f;
        kv_shard += s_shard[0]*k4.x; kv_shard += s_shard[1]*k4.y;
        kv_shard += s_shard[2]*k4.z; kv_shard += s_shard[3]*k4.w;
        const float kv_col = warp_reduce_sum(kv_shard);
        const float delta_col = (v_val - g_val*kv_col)*b_val;
        float attn_partial = 0.f;
        s_shard[0] = g_val*s_shard[0] + k4.x*delta_col; attn_partial += s_shard[0]*q4.x;
        s_shard[1] = g_val*s_shard[1] + k4.y*delta_col; attn_partial += s_shard[1]*q4.y;
        s_shard[2] = g_val*s_shard[2] + k4.z*delta_col; attn_partial += s_shard[2]*q4.z;
        s_shard[3] = g_val*s_shard[3] + k4.w*delta_col; attn_partial += s_shard[3]*q4.w;
        const float attn_col = warp_reduce_sum(attn_partial);
        if (lane == 0) attn_data[col] = attn_col * scale;
        attn_data += S_v * H_;
    }
    if (prof && threadIdx.x == 0 && threadIdx.y == 0) prof[blockIdx.x*NWARPS + blockIdx.z] = clock64() - t0;

    const float4 s4o = make_float4(s_shard[0], s_shard[1], s_shard[2], s_shard[3]);
    *reinterpret_cast<float4 *>(state + col*S_v + 4*lane) = s4o;
}


// ---------------------------------------------------------------- diagnostic (per-phase cycles, thread 0 of warp 0)
__global__ void __launch_bounds__((32 < S_v ? 32 : S_v) * NWARPS, 2)
diag_kernel(const float * q, const float * k, const float * v, const float * g, const float * beta,
            const float * curr_state, float * dst, float * state,
            int64_t H_, int64_t n_tokens, int64_t n_seqs,
            int64_t sq1, int64_t sq2, int64_t sq3,
            int64_t sv1, int64_t sv2, int64_t sv3,
            int64_t sb1, int64_t sb2, int64_t sb3,
            float scale, int64_t state_slot_stride, unsigned long long * prof) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;
    float * attn_data = dst;
    const int64_t state_in_offset  = sequence * H_ * S_v * S_v + h_idx * S_v * S_v;
    state += (sequence * H_ + h_idx) * S_v * S_v;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H_ + h_idx) * S_v;
    float s_shard[4];
    { const float4 s4 = ld4(curr_state + 4*lane); s_shard[0]=s4.x; s_shard[1]=s4.y; s_shard[2]=s4.z; s_shard[3]=s4.w; }
    const float * k_base = k + h_idx * sq1 + sequence * sq3;
    const float * q_base = q + h_idx * sq1 + sequence * sq3;
    const float * v_base = v + sequence * sv3 + h_idx * sv1;
    const int64_t gb_base = sequence * sb3 + h_idx * sb1;
    const bool meas = (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0);
    unsigned long long c_load=0, c_kv=0, c_upd=0, c_exp=0;
    for (int t = 0; t < n_tokens; t++) {
        unsigned long long a0 = meas ? clock64() : 0;
        const float4 k4 = ld4(k_base + t*sq2 + 4*lane);
        const float4 q4 = ld4(q_base + t*sq2 + 4*lane);
        const float g_raw = g[gb_base + t*sb2];
        const float b_val = beta[gb_base + t*sb2];
        const float v_val = v_base[t*sv2 + col];
        unsigned long long a1 = meas ? clock64() : 0;
        const float g_val = expf(g_raw);
        unsigned long long a2 = meas ? clock64() : 0;
        float kv_shard = 0.f;
        kv_shard += s_shard[0]*k4.x; kv_shard += s_shard[1]*k4.y;
        kv_shard += s_shard[2]*k4.z; kv_shard += s_shard[3]*k4.w;
        const float kv_col = warp_reduce_sum(kv_shard);
        unsigned long long a3 = meas ? clock64() : 0;
        const float delta_col = (v_val - g_val*kv_col)*b_val;
        float attn_partial = 0.f;
        s_shard[0] = g_val*s_shard[0] + k4.x*delta_col; attn_partial += s_shard[0]*q4.x;
        s_shard[1] = g_val*s_shard[1] + k4.y*delta_col; attn_partial += s_shard[1]*q4.y;
        s_shard[2] = g_val*s_shard[2] + k4.z*delta_col; attn_partial += s_shard[2]*q4.z;
        s_shard[3] = g_val*s_shard[3] + k4.w*delta_col; attn_partial += s_shard[3]*q4.w;
        const float attn_col = warp_reduce_sum(attn_partial);
        if (lane == 0) attn_data[col] = attn_col * scale;
        attn_data += S_v * H_;
        unsigned long long a4 = meas ? clock64() : 0;
        if (meas) { c_load += a1-a0; c_exp += a2-a1; c_kv += a3-a2; c_upd += a4-a3; }
    }
    *reinterpret_cast<float4 *>(state + col*S_v + 4*lane) = make_float4(s_shard[0],s_shard[1],s_shard[2],s_shard[3]);
    if (meas) { prof[0]=c_load; prof[1]=c_exp; prof[2]=c_kv; prof[3]=c_upd; }
}


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

// ---------------------------------------------------------------- host
int main(int argc, char ** argv) {
    printf("T03 existing-kernel bottleneck probe: D=%d H=%d T=%d\n", D, H, T);
    const float scale = 1.0f/sqrtf((float)D);
    const int t_tokens = (argc > 1) ? atoi(argv[1]) : T;
    const size_t sz_qk = (size_t)T*H*D, sz_v = (size_t)H*T*D, sz_gb = (size_t)H*T, sz_s = (size_t)H*D*D;
    float *h_q=(float*)malloc(sz_qk*4), *h_k=(float*)malloc(sz_qk*4), *h_v=(float*)malloc(sz_v*4);
    float *h_g=(float*)malloc(sz_gb*4), *h_b=(float*)malloc(sz_gb*4);
    float *h_sin=(float*)malloc(sz_s*4), *h_sref=(float*)malloc(sz_s*4), *h_schk=(float*)malloc(sz_s*4);
    float *h_oref=(float*)malloc(sz_v*4), *h_ochk=(float*)malloc(sz_v*4);
    srand(1234);
    for (size_t i=0;i<sz_qk;i++){h_q[i]=((rand()/(float)RAND_MAX)-0.5f)*2.f/sqrtf((float)D);h_k[i]=((rand()/(float)RAND_MAX)-0.5f)*2.f/sqrtf((float)D);}
    for (size_t i=0;i<sz_v;i++) h_v[i]=((rand()/(float)RAND_MAX)-0.5f)*2.f;
    for (size_t i=0;i<sz_gb;i++){h_g[i]=-((rand()/(float)RAND_MAX))*0.5f;h_b[i]=0.1f+0.9f*(rand()/(float)RAND_MAX);}
    for (size_t i=0;i<sz_s;i++) h_sin[i]=((rand()/(float)RAND_MAX)-0.5f)*0.2f;

    float *d_q,*d_k,*d_v,*d_g,*d_b,*d_sin,*d_sout,*d_out; unsigned long long *d_prof;
    CUDA_CHECK(cudaMalloc(&d_q,sz_qk*4)); CUDA_CHECK(cudaMalloc(&d_k,sz_qk*4)); CUDA_CHECK(cudaMalloc(&d_v,sz_v*4));
    CUDA_CHECK(cudaMalloc(&d_g,sz_gb*4)); CUDA_CHECK(cudaMalloc(&d_b,sz_gb*4));
    CUDA_CHECK(cudaMalloc(&d_sin,sz_s*4)); CUDA_CHECK(cudaMalloc(&d_sout,sz_s*4)); CUDA_CHECK(cudaMalloc(&d_out,sz_v*4));
    CUDA_CHECK(cudaMalloc(&d_prof, sizeof(unsigned long long)*H*NWARPS*8));
    CUDA_CHECK(cudaMemcpy(d_q,h_q,sz_qk*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_k,h_k,sz_qk*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v,h_v,sz_v*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_g,h_g,sz_gb*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,h_b,sz_gb*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_sin,h_sin,sz_s*4,cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
    ref_kernel<<<H,D>>>(d_q,d_k,d_v,d_g,d_b,d_sout,d_out,d_sout,t_tokens,scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_oref,d_out,sz_v*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sref,d_sout,sz_s*4,cudaMemcpyDeviceToHost));

    const dim3 grid(H,1,S_v/NWARPS), block(32,NWARPS,1);
    const int64_t sq1=D, sq2=(int64_t)H*D, sq3=(int64_t)T*H*D;
    const int64_t sv1=(int64_t)T*D, sv2=D, sv3=(int64_t)H*T*D;
    const int64_t sb1=T, sb2=1, sb3=(int64_t)H*T;

    cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    const int reps = 30;
    auto run = [&](auto kern, const char * nm) {
        CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_prof,0,sizeof(unsigned long long)*H*NWARPS*8));
        kern();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_ochk,d_out,sz_v*4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_schk,d_sout,sz_s*4,cudaMemcpyDeviceToHost));
        double mo=0,mr=0,ms=0,msr=0;
        for (size_t i=0;i<(size_t)H*t_tokens*D;i++){ mo=fmax(mo,fabs(h_ochk[i]-h_oref[i])); mr=fmax(mr,fabs(h_oref[i])); }
        for (size_t i=0;i<sz_s;i++){ ms=fmax(ms,fabs(h_schk[i]-h_sref[i])); msr=fmax(msr,fabs(h_sref[i])); }
        CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i=0;i<reps;i++) kern();
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        float tt; CUDA_CHECK(cudaEventElapsedTime(&tt,e0,e1));
        unsigned long long * hprof = (unsigned long long*)malloc(sizeof(unsigned long long)*H*NWARPS*8);
        CUDA_CHECK(cudaMemcpy(hprof,d_prof,sizeof(unsigned long long)*H*NWARPS*8,cudaMemcpyDeviceToHost));
        unsigned long long mx=0; for (int i=0;i<H*NWARPS;i++) mx = hprof[i]>mx?hprof[i]:mx;
        printf("%-14s : %7.3f ms/layer-ubatch   out_rel=%.2e state_rel=%.2e   token-loop cycles(block0)=%llu (~%llu/token @%d tok)\n",
               nm, tt/reps, mo/(mr+1e-30), ms/(msr+1e-30), mx, mx/(unsigned long long)t_tokens, t_tokens);
        return tt/reps;
    };
    auto launch_existing = [&]{ existing_kernel<false><<<grid,block>>>(d_q,d_k,d_v,d_g,d_b,d_sin,d_out,d_sout,H,t_tokens,1,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,scale,0,d_prof); };
    auto launch_prefetch = [&]{ existing_kernel<true ><<<grid,block>>>(d_q,d_k,d_v,d_g,d_b,d_sin,d_out,d_sout,H,t_tokens,1,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,scale,0,d_prof); };
    float ta = run(launch_existing, "existing");
    float tb = run(launch_prefetch, "prefetch");
    printf("speedup = %.2fx\n", ta/tb);
    {
        CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_prof,0,sizeof(unsigned long long)*H*NWARPS*8));
        diag_kernel<<<grid,block>>>(d_q,d_k,d_v,d_g,d_b,d_sin,d_out,d_sout,H,t_tokens,1,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,scale,0,d_prof);
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long hp[8]; CUDA_CHECK(cudaMemcpy(hp,d_prof,8*8,cudaMemcpyDeviceToHost));
        unsigned long long tt = hp[0]+hp[1]+hp[2]+hp[3];
        printf("coarse: load=%llu expf+red=%llu upd+red+st=%llu tot=%llu (%.0f/token)\n", hp[0],hp[1],hp[2],hp[3],tt,(double)tt/t_tokens);
        for (int pf = 0; pf < 2; pf++) {
            CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemset(d_prof,0,sizeof(unsigned long long)*H*NWARPS*8));
            if (pf) diag2_kernel<true ><<<grid,block>>>(d_q,d_k,d_v,d_g,d_b,d_sin,d_out,d_sout,H,t_tokens,1,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,scale,d_prof);
            else    diag2_kernel<false><<<grid,block>>>(d_q,d_k,d_v,d_g,d_b,d_sin,d_out,d_sout,H,t_tokens,1,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,scale,d_prof);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hp,d_prof,8*8,cudaMemcpyDeviceToHost));
            tt = hp[0]+hp[1]+hp[2]+hp[3];
            printf("fine(%s): loads_issued=%llu (%.0f/tok) use_k4+4fma=%llu (%.0f/tok) expf+reduce1+delta=%llu (%.0f/tok) upd+use_q4+reduce2+store=%llu (%.0f/tok)\n",
                   pf?"prefetch":"base    ", hp[0],(double)hp[0]/t_tokens, hp[1],(double)hp[1]/t_tokens, hp[2],(double)hp[2]/t_tokens, hp[3],(double)hp[3]/t_tokens);
        }
    }
    return 0;
}
