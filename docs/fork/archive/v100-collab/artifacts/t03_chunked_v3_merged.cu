// T03 V3: chunked gated_delta_net - merged phases (4 barriers/chunk instead of ~10), inline OS,
// folded un-normalization. Same math as V2 (validated at 1e-16 in numpy / 5e-7 in CUDA).
// Layouts and semantics identical to the reference kernel (non-KDA path):
//   S_t = a_t S_{t-1} + k_t (beta_t (v_t - a_t S_{t-1}^T k_t))^T ; o_t = scale * S_t^T q_t
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define D 128
#define L 16
#define H 32
#define T 512
#define COLS 32
#define WP (L+1)
#define DP (D+1)

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

// ---------------------------------------------------------------- reference
__global__ void ref_kernel(const float * q, const float * k, const float * v, const float * g, const float * b,
                           const float * s_in, float * out, float * s_out, int t_tokens, float scale) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;
    float col[D];
    const float * sin_h = s_in + (size_t)h*D*D;
    for (int i = 0; i < D; i++) col[i] = sin_h[j*D + i];
    for (int t = 0; t < t_tokens; t++) {
        const float * q_t = q + ((size_t)(t*H + h)*D);
        const float * k_t = k + ((size_t)(t*H + h)*D);
        const float * v_t = v + ((size_t)(h*t_tokens + t)*D);
        const float gt = expf(g[h*t_tokens + t]);
        const float bt = b[h*t_tokens + t];
        float kv = 0.f, at = 0.f;
        for (int i = 0; i < D; i++) kv += col[i]*k_t[i];
        const float delta = (v_t[j] - gt*kv)*bt;
        for (int i = 0; i < D; i++) { col[i] = gt*col[i] + k_t[i]*delta; at += col[i]*q_t[i]; }
        out[(size_t)(h*t_tokens + t)*D + j] = at*scale;
    }
    float * so = s_out + (size_t)h*D*D;
    for (int i = 0; i < D; i++) so[j*D + i] = col[i];
}

// ---------------------------------------------------------------- chunked V3
__global__ void __launch_bounds__(256, 2)
chunked_kernel(const float * q, const float * k, const float * v, const float * g, const float * b,
               const float * s_in, float * out, float * s_out, int t_tokens, float scale, int n_chunks, float * dbg) {
    extern __shared__ float smem[];
    float * sK   = smem;                    // [L][DP]
    float * sKKT = sK   + L*DP;             // [L][L]
    float * sWT  = sKKT + L*L;              // [D][WP]
    float * sUT  = sWT  + D*WP;             // [COLS][WP]
    float * sRT  = sUT  + COLS*WP;          // [L][WP]
    float * sT1  = sRT  + L*WP;             // [L][COLS]
    float * sQK  = sT1  + L*COLS;           // [L][L]
    float * sC   = sQK  + L*L;              // [L][L]
    float * sAc  = sC   + L*L;              // [L]
    float * sBt  = sAc  + L;                // [L]
    float * sM   = sBt  + L;                // [COLS][DP]

    const int h  = blockIdx.x;
    const int j0 = blockIdx.z*COLS;
    const int tid = threadIdx.x;
    constexpr int NPAIR = (L*COLS + 255)/256;   // (t,jj) pairs per thread

    for (int idx = tid; idx < COLS*D; idx += blockDim.x) {
        const int jj = idx / D, ii = idx % D;
        sM[jj*DP + ii] = s_in[(size_t)h*D*D + (j0+jj)*D + ii];
    }
    __syncthreads();

    for (int c = 0; c < n_chunks; c++) {
        const int t0 = c*L;
        if (t0 >= t_tokens) break;
        const int len = min(L, t_tokens - t0);

        // ================= phase 1: Ac/Bt, KKT (from global), load sK =================
        if (tid == 0) {
            float acc = 1.f;
            for (int t = 0; t < L; t++) {
                if (t < len) { acc *= expf(g[h*t_tokens + t0 + t]); sAc[t] = acc; sBt[t] = b[h*t_tokens + t0 + t]/acc; }
                else         { sAc[t] = 1.f; sBt[t] = 0.f; }
            }
        }
        const int t_ld = tid / D, i_ld = tid % D;
        if (tid < L*D) sK[t_ld*DP + i_ld] = (t_ld < len) ? k[(size_t)((t0+t_ld)*H + h)*D + i_ld] : 0.f;
        for (int idx = tid; idx < L*L; idx += blockDim.x) {
            const int t = idx / L, s = idx % L;
            float acc = 0.f;
            if (t >= s && t < len) {
                const float * kt = k + (size_t)((t0+t)*H + h)*D;
                const float * ks = k + (size_t)((t0+s)*H + h)*D;
                for (int i = 0; i < D; i++) acc += kt[i]*ks[i];
            }
            sKKT[t*L + s] = acc;
        }
        __syncthreads();

        // ================= phase 2: forward substitutions (W/U/R) + QK =================
        {
            const int ncols = D + COLS + L;
            for (int col = tid; col < ncols; col += blockDim.x) {
                const int kind = (col < D) ? 0 : (col < D+COLS ? 1 : 2);
                const int li   = (kind == 1) ? (col - D) : (kind == 2 ? col - D - COLS : 0);
                for (int t = 0; t < len; t++) {
                    float rhs;
                    if (kind == 0)      rhs = sK[t*DP + col];
                    else if (kind == 1) rhs = v[(size_t)(h*t_tokens + t0 + t)*D + (j0+li)]/sAc[t];
                    else                rhs = (li < t) ? sKKT[t*L + li] : 0.f;
                    float acc = 0.f;
                    const float * kk = sKKT + t*L;
                    for (int s = 0; s < t; s++) {
                        const float cf = kk[s];
                        const float xv = (kind == 0) ? sWT[col*WP + s] : (kind == 1 ? sUT[li*WP + s] : sRT[li*WP + s]);
                        acc += cf*xv;
                    }
                    const float res = b[h*t_tokens + t0 + t]*(rhs - acc);
                    if (kind == 0) sWT[col*WP + t] = res;
                    else if (kind == 1) sUT[li*WP + t] = res;
                    else sRT[li*WP + t] = res;
                }
            }
        }
        for (int idx = tid; idx < L*L; idx += blockDim.x) {
            const int t = idx / L, s = idx % L;
            float acc = 0.f;
            if (t < len && s < len) {
                const float * q_t = q + (size_t)((t0+t)*H + h)*D;
                const float * sk  = sK + s*DP;
                for (int i = 0; i < D; i++) acc += q_t[i]*sk[i];
            }
            sQK[t*L + s] = acc;
        }
        __syncthreads();

        // ================= phase 3: T1/A (registers) + C =================
        float Areg[NPAIR];
        #pragma unroll
        for (int r = 0; r < NPAIR; r++) {
            const int idx = tid + r*blockDim.x;
            const int t = idx / COLS, jj = idx % COLS;
            float t1 = 0.f, av = 0.f;
            if (idx < L*COLS && t < len) {
                const float * q_t = q + (size_t)((t0+t)*H + h)*D;
                const float * sm  = sM + jj*DP;
                for (int m = 0; m < D; m++) {
                    const float mv = sm[m];
                    t1 += sWT[m*WP + t]*mv;
                    av += q_t[m]*mv;
                }
            }
            if (idx < L*COLS) sT1[t*COLS + jj] = t1;
            Areg[r] = av;
        }
        for (int idx = tid; idx < L*L; idx += blockDim.x) {
            const int t = idx / L, s = idx % L;
            float acc = 0.f;
            if (s <= t && t < len) {
                for (int m = s+1; m <= t; m++) acc += sQK[t*L + m]*sRT[s*WP + m];
            }
            sC[t*L + s] = ((s <= t && t < len) ? sQK[t*L + s] : 0.f) - acc;
        }
        __syncthreads();

        // ================= phase 4: outputs (inline OS) + state update (folded un-normalize) =========
        #pragma unroll
        for (int r = 0; r < NPAIR; r++) {
            const int idx = tid + r*blockDim.x;
            if (idx >= L*COLS) break;
            const int t = idx / COLS, jj = idx % COLS;
            if (t >= len) continue;
            float intra = 0.f, os = 0.f;
            for (int s = 0; s <= t; s++) {
                intra += sC[t*L + s]*sBt[s]*v[(size_t)(h*t_tokens + t0 + s)*D + (j0+jj)];
                os    += sQK[t*L + s]*sT1[s*COLS + jj];
            }
            out[(size_t)(h*t_tokens + t0 + t)*D + (j0+jj)] = scale*sAc[t]*(Areg[r] - os + intra);
        }
        {
            const float acal = (len > 0) ? sAc[len-1] : 1.f;
            for (int idx = tid; idx < COLS*D; idx += blockDim.x) {
                const int jj = idx / D, i = idx % D;
                float acc = 0.f;
                for (int t = 0; t < len; t++) acc += sK[t*DP + i]*(sUT[jj*WP + t] - sT1[t*COLS + jj]);
                sM[jj*DP + i] = (sM[jj*DP + i] + acc)*acal;
            }
        }
        __syncthreads();
    }

    if (dbg && h == 0) {
        for (int idx = tid; idx < COLS*D; idx += blockDim.x) {
            const int jj = idx / D, ii = idx % D;
            dbg[blockIdx.z*COLS*D + idx] = sM[jj*DP + ii];
        }
    }
    for (int idx = tid; idx < COLS*D; idx += blockDim.x) {
        const int jj = idx / D, ii = idx % D;
        s_out[(size_t)h*D*D + (j0+jj)*D + ii] = sM[jj*DP + ii];
    }
}

// ---------------------------------------------------------------- host
int main(int argc, char ** argv) {
    printf("T03 V3 chunked GDN: D=%d L=%d H=%d T=%d COLS=%d\n", D, L, H, T, COLS);
    const float scale = 1.0f/sqrtf((float)D);
    const int t_tokens = (argc > 1) ? atoi(argv[1]) : T;
    const int n_chunks = (t_tokens + L - 1)/L;
    const size_t sz_qk = (size_t)T*H*D, sz_v = (size_t)H*T*D, sz_gb = (size_t)H*T, sz_s = (size_t)H*D*D;
    float *h_q = (float*)malloc(sz_qk*4), *h_k = (float*)malloc(sz_qk*4), *h_v = (float*)malloc(sz_v*4);
    float *h_g = (float*)malloc(sz_gb*4), *h_b = (float*)malloc(sz_gb*4);
    float *h_sin = (float*)malloc(sz_s*4), *h_sref = (float*)malloc(sz_s*4), *h_schk = (float*)malloc(sz_s*4);
    float *h_oref = (float*)malloc(sz_v*4), *h_ochk = (float*)malloc(sz_v*4);
    srand(1234);
    for (size_t i = 0; i < sz_qk; i++) { h_q[i] = ((rand()/(float)RAND_MAX)-0.5f)*2.f/sqrtf((float)D); h_k[i] = ((rand()/(float)RAND_MAX)-0.5f)*2.f/sqrtf((float)D); }
    for (size_t i = 0; i < sz_v; i++)  h_v[i] = ((rand()/(float)RAND_MAX)-0.5f)*2.f;
    for (size_t i = 0; i < sz_gb; i++) { h_g[i] = -((rand()/(float)RAND_MAX))*0.5f; h_b[i] = 0.1f + 0.9f*(rand()/(float)RAND_MAX); }
    for (size_t i = 0; i < sz_s; i++)  h_sin[i] = ((rand()/(float)RAND_MAX)-0.5f)*0.2f;

    float *d_q,*d_k,*d_v,*d_g,*d_b,*d_sin,*d_sout,*d_out;
    CUDA_CHECK(cudaMalloc(&d_q, sz_qk*4)); CUDA_CHECK(cudaMalloc(&d_k, sz_qk*4)); CUDA_CHECK(cudaMalloc(&d_v, sz_v*4));
    CUDA_CHECK(cudaMalloc(&d_g, sz_gb*4)); CUDA_CHECK(cudaMalloc(&d_b, sz_gb*4));
    CUDA_CHECK(cudaMalloc(&d_sin, sz_s*4)); CUDA_CHECK(cudaMalloc(&d_sout, sz_s*4)); CUDA_CHECK(cudaMalloc(&d_out, sz_v*4));
    CUDA_CHECK(cudaMemcpy(d_q,h_q,sz_qk*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_k,h_k,sz_qk*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v,h_v,sz_v*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_g,h_g,sz_gb*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,h_b,sz_gb*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_sin,h_sin,sz_s*4,cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
    ref_kernel<<<H, D>>>(d_q,d_k,d_v,d_g,d_b,d_sout,d_out,d_sout,t_tokens,scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_oref,d_out,sz_v*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sref,d_sout,sz_s*4,cudaMemcpyHostToDevice==0?cudaMemcpyDeviceToHost:cudaMemcpyDeviceToHost));

    const size_t smem_bytes = (L*DP + L*L + D*WP + COLS*WP + L*WP + L*COLS + L*L + L*L + 2*L + COLS*DP)*4;
    CUDA_CHECK(cudaFuncSetAttribute(chunked_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    const size_t dbg_floats = D*D;
    float * d_dbg; CUDA_CHECK(cudaMalloc(&d_dbg, dbg_floats*4)); CUDA_CHECK(cudaMemset(d_dbg, 0, dbg_floats*4));
    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
    chunked_kernel<<<dim3(H,1,D/COLS), 256, smem_bytes>>>(d_q,d_k,d_v,d_g,d_b,d_sout,d_out,d_sout,t_tokens,scale,n_chunks,d_dbg);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_ochk,d_out,sz_v*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_schk,d_sout,sz_s*4,cudaMemcpyDeviceToHost));

    double mo=0, mr=0, ms=0, msr=0;
    const size_t sz_v_cmp = (size_t)H*t_tokens*D;
    for (size_t i = 0; i < sz_v_cmp; i++) { mo = fmax(mo, fabs(h_ochk[i]-h_oref[i])); mr = fmax(mr, fabs(h_oref[i])); }
    for (size_t i = 0; i < sz_s; i++) { ms = fmax(ms, fabs(h_schk[i]-h_sref[i])); msr = fmax(msr, fabs(h_sref[i])); }
    printf("out  max_abs_err=%.3e  max_ref=%.3e  rel=%.3e\n", mo, mr, mo/(mr+1e-30));
    printf("state max_abs_err=%.3e  max_ref=%.3e  rel=%.3e\n", ms, msr, ms/(msr+1e-30));
    printf("smem = %.1f KB\n", smem_bytes/1024.0);
    {
        int occ = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, chunked_kernel, 256, smem_bytes);
        printf("occupancy: %d block/SM -> %d warps/SM\n", occ, occ*8);
    }

    cudaEvent_t e0, e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    const int reps = 20;
    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < reps; i++) ref_kernel<<<H, D>>>(d_q,d_k,d_v,d_g,d_b,d_sout,d_out,d_sout,t_tokens,scale);
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float t_ref; CUDA_CHECK(cudaEventElapsedTime(&t_ref, e0, e1));
    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < reps; i++) chunked_kernel<<<dim3(H,1,D/COLS), 256, smem_bytes>>>(d_q,d_k,d_v,d_g,d_b,d_sout,d_out,d_sout,t_tokens,scale,n_chunks,nullptr);
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float t_chk; CUDA_CHECK(cudaEventElapsedTime(&t_chk, e0, e1));
    printf("time: ref=%.3f ms  chunked=%.3f ms  speedup=%.2fx\n", t_ref/reps, t_chk/reps, t_ref/t_chk);
    const double mac_per_layer_block = (L*L*D/2.0 + (double)(L*(L-1)/2)*(D+COLS+L) + L*L*D + (double)L*COLS*D*2 + (double)L*L*L/2 + L*COLS*L + (double)COLS*D*L) / 1.0;
    printf("est. MAC/chunk/block = %.0f (%.0f FMA/thread), measured %.1f us/chunk -> %.1f cycles/FMA @1500MHz\n",
           mac_per_layer_block, mac_per_layer_block/256.0, t_chk*1000.0/n_chunks, t_chk*1e-3*1.5e9/n_chunks/(mac_per_layer_block/256.0));
    return 0;
}
