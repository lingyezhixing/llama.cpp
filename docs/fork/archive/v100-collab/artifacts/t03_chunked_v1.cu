// T03: chunked gated_delta_net prototype (implementer, 2026-09-22)
// Reference : 1 thread per (head, state column), serial over tokens; math identical to the current
//             kernel's non-KDA path (S_t = a S + k (beta (v - a S^T k))^T ; o_t = scale * S_t^T q_t).
// Chunked V1: grid (H, 1, D/COLS); per chunk (L tokens): KKT / 3 forward substitutions / QK / T1+A /
//             C / outputs / state update. State lives in smem (COLS x D).
// Layouts mimic llama.cpp:
//   q,k : [T][H][D]   q[((t*H)+h)*D + i]
//   v   : [H][T][D]   v[((h*T)+t)*D + j]
//   g,b : [H][T]
//   state: [H][D][D]  M[col][i] = S[i][col]      -> state[h*D*D + j*D + i]
//   out : [H][T][D]   out[((h*T)+t)*D + j]
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define D 128
#define L 32
#define H 32
#define T 512
#define COLS 32                 // state columns per block (grid.z = D/COLS)
#define WP (L+1)                // padded row length for the transposed solve buffers
#define DBSTATEOFF (D*WP + COLS*WP + L*WP + L*L + L*COLS + L*COLS + L*L + 2*L)

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

// ---------------------------------------------------------------- reference
__global__ void ref_kernel(const float * q, const float * k, const float * v, const float * g, const float * b,
                           const float * s_in, float * out, float * s_out, int t_tokens, float scale) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;                                   // state column
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

// ---------------------------------------------------------------- chunked V1
__global__ void __launch_bounds__(256, 1)
chunked_kernel(const float * q, const float * k, const float * v, const float * g, const float * b,
               const float * s_in, float * out, float * s_out, int t_tokens, float scale, int n_chunks, float * dbg) {
    extern __shared__ float smem[];
    float * sK   = smem;                    // [L][D]
    float * sKKT = sK   + L*D;              // [L][L]
    float * sWT  = sKKT + L*L;              // [D][WP]   W transposed (row col=j, col s=t)
    float * sUT  = sWT  + D*WP;             // [COLS][WP] U transposed
    float * sRT  = sUT  + COLS*WP;          // [L][WP]   R transposed (row s, col m)
    float * sT1  = sRT  + L*WP;             // [L][COLS] T1
    float * sQK  = sT1  + L*COLS;           // [L][L]    QK -> later overwritten by C
    float * sOS  = sQK  + L*L;              // [L][COLS] OS = sum_s QK*T1  (then unused)
    float * sAc  = sOS  + L*COLS;           // [L]
    float * sBt  = sAc  + L;                // [L]
    float * sM   = sBt  + L;                // [COLS][D] state

    const int h  = blockIdx.x;
    const int j0 = blockIdx.z*COLS;
    const int tid = threadIdx.x;

    for (int idx = tid; idx < COLS*D; idx += blockDim.x) {
        const int jj = idx / D, ii = idx % D;
        sM[idx] = s_in[(size_t)h*D*D + (j0+jj)*D + ii];
    }
    __syncthreads();

    for (int c = 0; c < n_chunks; c++) {
        const int t0 = c*L;
        if (t0 >= t_tokens) break;
        const int len = min(L, t_tokens - t0);
        // wait: all phases below assume len == L except that sK rows t >= len are zeroed
        for (int t = tid; t < L; t += blockDim.x) {
            if (t < len) { sAc[t] = 0.f; sBt[t] = 0.f; }
            else         { sAc[t] = 1.f; sBt[t] = 0.f; }
        }
        __syncthreads();

        // decay + beta (serial over t is fine: L small)
        if (tid == 0) {
            float acc = 1.f;
            for (int t = 0; t < len; t++) {
                acc *= expf(g[h*t_tokens + t0 + t]);
                sAc[t] = acc;
                sBt[t] = b[h*t_tokens + t0 + t]/acc;
            }
        }
        // load K
        for (int idx = tid; idx < L*D; idx += blockDim.x) {
            const int t = idx / D, i = idx % D;
            sK[idx] = (t < len) ? k[(size_t)((t0+t)*H + h)*D + i] : 0.f;
        }
        __syncthreads();

        // KKT
        for (int idx = tid; idx < L*L; idx += blockDim.x) {
            const int t = idx / L, s = idx % L;
            float acc = 0.f;
            if (t >= s && s < len) { for (int i = 0; i < D; i++) acc += sK[t*D+i]*sK[s*D+i]; }
            sKKT[t*L + s] = acc;
        }
        __syncthreads();

        // solve W (D cols), U (COLS cols), R (L cols) -- no intra-solve sync needed (column-private)
        {
            const int ncols = D + COLS + L;
            for (int col = tid; col < ncols; col += blockDim.x) {
                const int kind = (col < D) ? 0 : (col < D+COLS ? 1 : 2);
                const int li   = (kind == 1) ? (col - D) : (kind == 2 ? col - D - COLS : 0);
                for (int t = 0; t < L; t++) {
                    float rhs = 0.f;
                    if (kind == 0) rhs = sK[t*D + col];
                    else if (kind == 1) rhs = (t < len) ? v[(size_t)(h*t_tokens + t0 + t)*D + (j0+li)]/sAc[t] : 0.f;
                    else rhs = (li < t) ? sKKT[t*L + li] : 0.f;
                    float acc = 0.f;
                    for (int s = 0; s < t; s++) {
                        const float cf = sKKT[t*L + s];
                        const float xv = (kind == 0) ? sWT[col*WP + s] : (kind == 1 ? sUT[li*WP + s] : sRT[li*WP + s]);
                        acc += cf*xv;
                    }
                    const float val = sAc[t]*-1.f;  // unused marker to keep the compiler honest
                    (void)val;
                    const float b_t = (kind == 1) ? (t < len ? b[h*t_tokens + t0 + t] : 0.f) : (t < len ? b[h*t_tokens + t0 + t] : 0.f);
                    const float res = b_t*(rhs - acc);
                    if (kind == 0) sWT[col*WP + t] = res;
                    else if (kind == 1) sUT[li*WP + t] = res;
                    else sRT[li*WP + t] = res;
                }
            }
        }
        __syncthreads();

        // debug dump: W, U, R (chunk 0); Ac/Bt at the tail
        if (dbg && c == 0 && blockIdx.x == 0 && blockIdx.z == 0) {
            const int off0 = D*WP + COLS*WP + L*WP + L*L + L*COLS + L*COLS + L*L;
            for (int t = 0; t < L; t++) { dbg[off0 + t] = sAc[t]; dbg[off0 + L + t] = sBt[t]; }
            for (int idx = tid; idx < D*WP; idx += blockDim.x)    dbg[idx] = sWT[idx];
            for (int idx = tid; idx < COLS*WP; idx += blockDim.x) dbg[D*WP + idx] = sUT[idx];
            for (int idx = tid; idx < L*WP; idx += blockDim.x)    dbg[D*WP + COLS*WP + idx] = sRT[idx];
        }
        // QK
        for (int idx = tid; idx < L*L; idx += blockDim.x) {
            const int t = idx / L, s = idx % L;
            float acc = 0.f;
            if (t < len && s < len) {
                for (int i = 0; i < D; i++) acc += q[(size_t)((t0+t)*H + h)*D + i]*sK[s*D + i];
            }
            sQK[t*L + s] = acc;
        }
        __syncthreads();

        // T1[t][jj] = sum_m W[t][m] M[jj][m] ; A[t][jj] = sum_m q_t[m] M[jj][m]
        float Areg[ (L*COLS + 255)/256 ];   // 4 floats per thread for COLS=32,L=32 (1024/256)
        #pragma unroll
        for (int r = 0; r < (L*COLS+255)/256; r++) {
            const int idx = tid + r*blockDim.x;
            if (idx >= L*COLS) break;
            const int t = idx / COLS, jj = idx % COLS;
            float t1 = 0.f, av = 0.f;
            if (t < len) {
                for (int m = 0; m < D; m++) {
                    const float mv = sM[jj*D + m];
                    t1 += sWT[m*WP + t]*mv;
                    av += q[(size_t)((t0+t)*H + h)*D + m]*mv;
                }
            }
            sT1[t*COLS + jj] = t1;
            Areg[r] = av;
        }
        __syncthreads();

        if (dbg && c == 0 && blockIdx.x == 0 && blockIdx.z == 0) {
            const int off = D*WP + COLS*WP + L*WP;
            for (int idx = tid; idx < L*L; idx += blockDim.x)    dbg[off + idx] = sQK[idx];
            for (int idx = tid; idx < L*COLS; idx += blockDim.x) dbg[off + L*L + idx] = sT1[idx];
        }
        // OS[t][jj] = sum_{s<=t} QK[t][s] T1[s][jj]   (uses QK before it is overwritten by C)
        for (int idx = tid; idx < L*COLS; idx += blockDim.x) {
            const int t = idx / COLS, jj = idx % COLS;
            float acc = 0.f;
            for (int s = 0; s <= t && s < len; s++) acc += sQK[t*L + s]*sT1[s*COLS + jj];
            sOS[idx] = acc;
        }
        __syncthreads();

        // C = QK - tril(QK,0) @ R  (in place over sQK; one thread per row, reads only its own row)
        for (int t = tid; t < L; t += blockDim.x) {
            float row[L];
            for (int s = 0; s < L; s++) row[s] = sQK[t*L + s];
            for (int s = 0; s < L; s++) {
                float acc = 0.f;
                for (int m = s+1; m <= t; m++) acc += row[m]*sRT[s*WP + m];
                sQK[t*L + s] = row[s] - acc;
            }
        }
        __syncthreads();

        if (dbg && c == 0 && blockIdx.x == 0 && blockIdx.z == 0) {
            const int off = D*WP + COLS*WP + L*WP + L*L + L*COLS;
            for (int idx = tid; idx < L*COLS; idx += blockDim.x) dbg[off + idx] = sOS[idx];
            for (int idx = tid; idx < L*L; idx += blockDim.x)    dbg[off + L*COLS + idx] = sQK[idx];  // C
        }
        // outputs + state update
        for (int idx = tid; idx < L*COLS; idx += blockDim.x) {
            const int t = idx / COLS, jj = idx % COLS;
            if (t >= len) continue;
            float intra = 0.f;
            for (int s = 0; s <= t; s++) {
                intra += sQK[t*L + s]*sBt[s]*v[(size_t)(h*t_tokens + t0 + s)*D + (j0+jj)];
            }
            float av = 0.f;
            #pragma unroll
            for (int r = 0; r < (L*COLS+255)/256; r++) if (tid + r*blockDim.x == idx) av = Areg[r];
            out[(size_t)(h*t_tokens + t0 + t)*D + (j0+jj)] = scale*sAc[t]*(av - sOS[idx] + intra);
        }
        // state: M[jj][i] += sum_t K[t][i] * (U[t][jj] - T1[t][jj]); then un-normalize
        for (int idx = tid; idx < COLS*D; idx += blockDim.x) {
            const int jj = idx / D, i = idx % D;
            float acc = 0.f;
            for (int t = 0; t < len; t++) {
                acc += sK[t*D + i]*(sUT[jj*WP + t] - sT1[t*COLS + jj]);
            }
            sM[idx] += acc;
        }
        __syncthreads();
        if (len > 0) {
            const float acal = sAc[len-1];
            for (int idx = tid; idx < COLS*D; idx += blockDim.x) sM[idx] *= acal;
        }
        __syncthreads();
    }

    if (dbg && h == 0) {
        for (int idx = tid; idx < COLS*D; idx += blockDim.x) {
            dbg[DBSTATEOFF + blockIdx.z*COLS*D + idx] = sM[idx];
        }
    }
    for (int idx = tid; idx < COLS*D; idx += blockDim.x) {
        const int jj = idx / D, ii = idx % D;
        s_out[(size_t)h*D*D + (j0+jj)*D + ii] = sM[idx];
    }
}

// ---------------------------------------------------------------- host
int main(int argc, char ** argv) {
    printf("T03 chunked GDN prototype: D=%d L=%d H=%d T=%d COLS=%d\n", D, L, H, T, COLS);
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

    // reference
    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
    ref_kernel<<<H, D>>>(d_q,d_k,d_v,d_g,d_b,d_sout,d_out,d_sout,t_tokens,scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_oref,d_out,sz_v*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sref,d_sout,sz_s*4,cudaMemcpyDeviceToHost));

    // chunked
    const size_t smem_bytes = (L*D + L*L + D*WP + COLS*WP + L*WP + L*COLS + L*L + L*COLS + 2*L + COLS*D)*4;
    CUDA_CHECK(cudaFuncSetAttribute(chunked_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
    const size_t dbg_floats = DBSTATEOFF + (size_t)D*D;
    float * d_dbg; CUDA_CHECK(cudaMalloc(&d_dbg, dbg_floats*4)); CUDA_CHECK(cudaMemset(d_dbg, 0, dbg_floats*4));
    chunked_kernel<<<dim3(H,1,D/COLS), 256, smem_bytes>>>(d_q,d_k,d_v,d_g,d_b,d_sout,d_out,d_sout,t_tokens,scale,n_chunks,d_dbg);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_ochk,d_out,sz_v*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_schk,d_sout,sz_s*4,cudaMemcpyDeviceToHost));

    {
        float * hd_dbg = (float*)malloc(dbg_floats*4);
        CUDA_CHECK(cudaMemcpy(hd_dbg, d_dbg, dbg_floats*4, cudaMemcpyDeviceToHost));
        FILE * f = fopen("t03_dbg.bin", "wb");
        fwrite(h_q, 4, (size_t)T*H*D, f); fwrite(h_k, 4, (size_t)T*H*D, f); fwrite(h_v, 4, (size_t)H*T*D, f);
        fwrite(h_g, 4, sz_gb, f); fwrite(h_b, 4, sz_gb, f); fwrite(h_sin, 4, sz_s, f);
        fwrite(hd_dbg, 4, dbg_floats, f); fclose(f);
        printf("dumped t03_dbg.bin (dbg_floats=%d)\n", (int)dbg_floats);
        free(hd_dbg);
    }
    double mo=0, mr=0, ms=0, msr=0;
    const size_t sz_v_cmp = (size_t)H*t_tokens*D;
    for (size_t i = 0; i < sz_v_cmp; i++) { mo = fmax(mo, fabs(h_ochk[i]-h_oref[i])); mr = fmax(mr, fabs(h_oref[i])); }
    for (size_t i = 0; i < sz_s; i++) { ms = fmax(ms, fabs(h_schk[i]-h_sref[i])); msr = fmax(msr, fabs(h_sref[i])); }
    printf("out  max_abs_err=%.3e  max_ref=%.3e  rel=%.3e\n", mo, mr, mo/(mr+1e-30));
    printf("state max_abs_err=%.3e  max_ref=%.3e  rel=%.3e\n", ms, msr, ms/(msr+1e-30));
    printf("smem = %.1f KB\n", smem_bytes/1024.0);

    // timing
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
    printf("time: ref=%.3f ms  chunked=%.3f ms  speedup=%.2fx  (per layer-ubatch)\n", t_ref/reps, t_chk/reps, t_ref/t_chk);
    return 0;
}
