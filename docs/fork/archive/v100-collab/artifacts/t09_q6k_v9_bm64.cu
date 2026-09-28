// T09-A: quantized-operand GEMM microbenchmark
// weights (A operand) come from real Q6_K data, staged as half-super-blocks and
// unpacked to an fp16 smem panel inside the kernel; activations (B) are fp16 k-blocked.
// BM=64, BN=128, BK=32, 256 threads (8 warps 2x4), 2 blocks/SM.
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace nvcuda;

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float>              frag_c;

#define BM 128
#define BN 128
#define BK 32
#define BKP 40
#define NTHREAD 256
#define WM 2
#define WN 4
#define WMA 4      // 16-row A subtiles per warp
#define WNB 2      // 16-token B subtiles per warp
#define HSB 112    // packed half-super-block bytes per row (64 ql + 32 qh + 8 scales + 2 d + 6 pad)

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// reference dequant from the raw Q6_K layout (independent path, mirrors the
// verified dequantize.cuh logic): one warp per 256-element super-block.
__global__ void dequant_q6k_ref(const uint8_t * __restrict__ raw, half * __restrict__ out, const int M, const int K) {
    const int64_t blk = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int nsb = K / 256;
    const int row = (int)(blk / nsb);
    const int sb  = (int)(blk % nsb);
    const uint8_t * x = raw + ((int64_t)row*nsb + sb)*210;
    const uint8_t * ql_ = x;
    const uint8_t * qh_ = x + 128;
    const int8_t  * sc  = (const int8_t *)(x + 192);
    const half    * d_  = (const half   *)(x + 208);

    const int64_t e0 = 8*lane;
    const float d = __half2float(*d_) * (float)sc[e0 >> 4];
    const uint8_t * ql = ql_ + (e0 & 63) + ((e0 >> 7) << 6);
    const uint8_t * qh = qh_ + (e0 & 31) + ((e0 >> 7) << 5);
    const int  shift = 2*((e0 >> 5) & 3);
    const bool hi    = ((e0 >> 6) & 1) != 0;

    const uint16_t qw[4] = { *(const uint16_t*)(ql+0), *(const uint16_t*)(ql+2), *(const uint16_t*)(ql+4), *(const uint16_t*)(ql+6) };
    const uint16_t hw[4] = { *(const uint16_t*)(qh+0), *(const uint16_t*)(qh+2), *(const uint16_t*)(qh+4), *(const uint16_t*)(qh+6) };
    uint32_t v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t qb = (i & 1) ? (qw[i/2] >> 8) : (qw[i/2] & 0xFF);
        const uint32_t hb = (i & 1) ? (hw[i/2] >> 8) : (hw[i/2] & 0xFF);
        const uint32_t nib = hi ? ((qb >> 4) & 0xF) : (qb & 0xF);
        const int32_t  q6  = (int32_t)(nib | (((hb >> shift) & 3) << 4)) - 32;
        v[i] = __half_as_ushort(__float2half(d * (float)q6));
    }
    uint4 o;
    o.x = v[0] | (v[1] << 16);
    o.y = v[2] | (v[3] << 16);
    o.z = v[4] | (v[5] << 16);
    o.w = v[6] | (v[7] << 16);
    *reinterpret_cast<uint4 *>(out + (int64_t)row*K + sb*256 + e0) = o;
}

__global__ void __launch_bounds__(NTHREAD, 1)
gateA_q6k_kernel(const uint8_t * __restrict__ Q, const half * __restrict__ X, float * __restrict__ C,
                 const int M, const int N, const int K, const int mode) {
    extern __shared__ uint8_t smem_dyn[];
    uint8_t * const sAq0 = smem_dyn;                       // 2 x BM*HSB
    half    * const sAf0 = (half *)(smem_dyn + 2*BM*HSB);  // 2 x BM*BKP
    half    * const sB0  = (half *)(smem_dyn + 2*BM*HSB + 2*BM*BKP*2); // 2 x BN*BKP

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*32 + lane;
    const int wm   = warp % WM;
    const int wn   = warp / WM;
    const int m0   = blockIdx.x * BM;
    const int n0   = blockIdx.y * BN;
    const int nkb  = K / BK;
    const int nh   = nkb / 4;   // half-super-blocks per row

    float4 regB[2];
    float4 regH[4];

    // ---- stage half-super-block h (BM rows x HSB bytes) ----
    auto stage_half = [&](const int h, uint8_t * dst) {
        for (int i = tid; i < BM*7; i += NTHREAD) {
            const int row = i / 7;
            const int ch  = i % 7;
            *reinterpret_cast<float4 *>(dst + row*HSB + ch*16) =
                *reinterpret_cast<const float4 *>(Q + ((int64_t)h*M + m0 + row)*HSB + ch*16);
        }
    };
    // ---- unpack one 32-element k-slice (4 k-iterations per half) ----
    auto unpack_slice = [&](const int kb, const uint8_t * src, half * dst) {
        const int s = kb & 3;                 // slice within the half
        const int q   = tid & 3;
        for (int rep = 0; rep < BM/64; ++rep) {
        const int row = (tid >> 2) + rep*64;
        const int l0  = 32*s + 8*q;
        const uint8_t * base = src + row*HSB;
        const uint8_t * ql = base + (l0 & 63);
        const uint8_t * qh = base + 64 + 8*q;
        const float d = __half2float(*reinterpret_cast<const half *>(base + 104)) *
                        (float)(*reinterpret_cast<const int8_t *>(base + 96 + (2*s + (q >> 1))));
        const bool hi    = s >= 2;
        const int  shift = 2*s;
        const uint8_t qb[8] = { ql[0], ql[1], ql[2], ql[3], ql[4], ql[5], ql[6], ql[7] };
        const uint8_t hb[8] = { qh[0], qh[1], qh[2], qh[3], qh[4], qh[5], qh[6], qh[7] };
        uint32_t v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint32_t nib = hi ? ((qb[i] >> 4) & 0xF) : (qb[i] & 0xF);
            const int32_t  q6  = (int32_t)(nib | (((hb[i] >> shift) & 3) << 4)) - 32;
            v[i] = __half_as_ushort(__float2half(d * (float)q6));
        }
        uint4 o;
        o.x = v[0] | (v[1] << 16);
        o.y = v[2] | (v[3] << 16);
        o.z = v[4] | (v[5] << 16);
        o.w = v[6] | (v[7] << 16);
        *reinterpret_cast<uint4 *>(dst + row*BKP + 8*q) = o;   // slice-local column
        }
    };
    auto stage_B = [&](const int kb, half * dst, const float4 * r) {
        const int row = tid / 4;
        const int part = tid % 4;
        *reinterpret_cast<float4 *>(dst + row*BKP + part*8) = r[0];
        *reinterpret_cast<float4 *>(dst + (row + 64)*BKP + part*8) = r[1];
    };

    stage_half(0, sAq0);
    __syncthreads();
    unpack_slice(0, sAq0, sAf0);
    if (mode & 112) {   // debug: ALL blocks return, only block (0,0) dumps
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            if (mode & 32) {   // raw global Q bytes, then smem staged bytes
                for (int i = tid; i < BM*HSB; i += NTHREAD) {
                    reinterpret_cast<uint8_t *>(C)[i] = Q[((int64_t)0*M + m0)*HSB + i];
                }
                for (int i = tid; i < BM*HSB; i += NTHREAD) {
                    reinterpret_cast<uint8_t *>(C)[BM*HSB + i] = sAq0[i];
                }
            }
            if (mode & 64) {   // smem write test: pattern then dump
                for (int i = tid; i < BM*HSB; i += NTHREAD) {
                    sAq0[i] = (uint8_t)(i & 0xFF);
                }
                __syncthreads();
                for (int i = tid; i < BM*HSB; i += NTHREAD) {
                    reinterpret_cast<uint8_t *>(C)[i] = sAq0[i];
                }
            }
            if (mode & 16) {   // unpacked panel (slice 0)
                __syncthreads();
                for (int i = tid; i < BM*BKP; i += NTHREAD) {
                    C[i] = __half2float(sAf0[i]);
                }
            }
        }
        return;
    }
    {
        const int row = tid / 4;
        const int part = tid % 4;
        const float4 a0 = *reinterpret_cast<const float4 *>(X + (int64_t)0*N*BK + (n0 + row)*BK + part*8);
        const float4 a1 = *reinterpret_cast<const float4 *>(X + (int64_t)0*N*BK + (n0 + row + 64)*BK + part*8);
        *reinterpret_cast<float4 *>(sB0 + row*BKP + part*8) = a0;
        *reinterpret_cast<float4 *>(sB0 + (row + 64)*BKP + part*8) = a1;
    }
    __syncthreads();

    frag_c c[WMA][WNB];
#pragma unroll
    for (int ia = 0; ia < WMA; ++ia) {
#pragma unroll
        for (int ib = 0; ib < WNB; ++ib) {
            wmma::fill_fragment(c[ia][ib], 0.0f);
        }
    }

    for (int kb = 0; kb < nkb; ++kb) {
        const int stage = kb & 1;
        const uint8_t * sAq = sAq0 + ((kb >> 2) & 1)*BM*HSB;
        const half * sAf = sAf0 + stage*BM*BKP;
        const half * sB  = sB0  + stage*BN*BKP;

        if (mode & 128) {   // dump panels at kb==1
            if (kb == 1) {
                if (blockIdx.x == 0 && blockIdx.y == 0) {
                    for (int i = tid; i < BM*BKP; i += NTHREAD) C[i] = __half2float(sAf0[1*BM*BKP + i]);
                    for (int i = tid; i < BN*BKP; i += NTHREAD) C[BM*BKP + i] = __half2float(sB0[1*BN*BKP + i]);
                }
                return;
            }
        }

        // start loading the half for k-iterations kb+2.. (issued early to cover latency)
        if ((kb & 3) == 2 && kb + 2 < nkb) {
            const int h2 = (kb + 2) >> 2;
#pragma unroll
            for (int rep = 0; rep < 4; ++rep) {
                const int i = tid + rep*NTHREAD;
                if (i < BM*7) {
                    const int row = i / 7;
                    const int ch  = i % 7;
                    regH[rep] = *reinterpret_cast<const float4 *>(Q + ((int64_t)h2*M + m0 + row)*HSB + ch*16);
                }
            }
        }
        // prefetch next activation slice
        if (kb + 1 < nkb && !(mode & 2)) {
            const int row = tid / 4;
            const int part = tid % 4;
            regB[0] = *reinterpret_cast<const float4 *>(X + (int64_t)(kb + 1)*N*BK + (n0 + row)*BK + part*8);
            regB[1] = *reinterpret_cast<const float4 *>(X + (int64_t)(kb + 1)*N*BK + (n0 + row + 64)*BK + part*8);
        }

        // mma
        if (!(mode & 4)) {
#pragma unroll
            for (int ks = 0; ks < BK/16; ++ks) {
                frag_a a[WMA];
                frag_b b[WNB];
#pragma unroll
                for (int ia = 0; ia < WMA; ++ia) {
                    wmma::load_matrix_sync(a[ia], sAf + (wm*(BM/WM) + ia*16)*BKP + ks*16, BKP);
                }
#pragma unroll
                for (int ib = 0; ib < WNB; ++ib) {
                    wmma::load_matrix_sync(b[ib], sB + (wn*(BN/WN) + ib*16)*BKP + ks*16, BKP);
                }
#pragma unroll
                for (int ia = 0; ia < WMA; ++ia) {
#pragma unroll
                    for (int ib = 0; ib < WNB; ++ib) {
                        wmma::mma_sync(c[ia][ib], a[ia], b[ib], c[ia][ib]);
                    }
                }
            }
        }

        // commit the next half-super-block (loaded at kb-1) before unpacking slice kb+1
        if ((kb & 3) == 3 && kb + 1 < nkb) {
            uint8_t * dst = sAq0 + (((kb + 1) >> 2) & 1)*BM*HSB;
#pragma unroll
            for (int rep = 0; rep < 4; ++rep) {
                const int i = tid + rep*NTHREAD;
                if (i < BM*7) {
                    const int row = i / 7;
                    const int ch  = i % 7;
                    *reinterpret_cast<float4 *>(dst + row*HSB + ch*16) = regH[rep];
                }
            }
            __syncthreads();
        }
        // unpack next slice + commit next activations
        if (kb + 1 < nkb) {
            const int h1 = (kb + 1) >> 2;
            if (!(mode & 8)) {
                unpack_slice(kb + 1, sAq0 + ((h1) & 1)*BM*HSB, sAf0 + ((kb + 1) & 1)*BM*BKP);
            }
            stage_B(kb + 1, sB0 + ((kb + 1) & 1)*BN*BKP, regB);
        }
        __syncthreads();
    }

    const int cm = m0 + wm*(BM/WM);
    const int cn = n0 + wn*(BN/WN);
#pragma unroll
    for (int ia = 0; ia < WMA; ++ia) {
#pragma unroll
        for (int ib = 0; ib < WNB; ++ib) {
            wmma::store_matrix_sync(C + (int64_t)(cm + ia*16)*N + (cn + ib*16), c[ia][ib], N, wmma::mem_row_major);
        }
    }
}

int main(int argc, char ** argv) {
    const int M = 17408, N = 512, K = argc > 3 ? atoi(argv[3]) : 5120;
    const int reps = argc > 1 ? atoi(argv[1]) : 30;
    const int mode = argc > 2 ? atoi(argv[2]) : 0;

    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s  M=%d N=%d K=%d reps=%d mode=%d\n", prop.name, M, N, K, reps, mode);

    // host: load packed Q6_K and raw Q6_K
    const size_t qbytes = (size_t)M * (K/256) * 210;
    uint8_t * hQ = (uint8_t *) malloc(qbytes);
    FILE * f = fopen("q6k_gate.raw", "rb");
    if (!f) { printf("cannot open q6k_gate.raw\n"); return 1; }
    if (fread(hQ, 1, qbytes, f) != qbytes) { printf("short read raw\n"); return 1; }
    fclose(f);

    const size_t pbytes = (size_t)(K/128) * M * 112;
    uint8_t * hP = (uint8_t *) malloc(pbytes);
    f = fopen("q6k_gate_packed112.raw", "rb");
    if (!f) { printf("cannot open q6k_gate_packed112.raw\n"); return 1; }
    if (fread(hP, 1, pbytes, f) != pbytes) { printf("short read packed\n"); return 1; }
    fclose(f);

    // fp16 activations (random) in row-major and in k-blocked layout
    half * hW = (half *) malloc((size_t)M*K*sizeof(half));
    half * hX = (half *) malloc((size_t)N*K*sizeof(half));
    half * hXb = (half *) malloc((size_t)N*K*sizeof(half));
    srand(7);
    for (size_t i = 0; i < (size_t)K*N; ++i) hX[i] = __float2half(((rand() % 2001) - 1000) / 1000.0f);
    for (int kb = 0; kb < K/BK; ++kb) {
        for (int r = 0; r < N; ++r) {
            for (int k = 0; k < BK; ++k) {
                hXb[((size_t)kb*N + r)*BK + k] = hX[(size_t)r*K + kb*BK + k];
            }
        }
    }

    uint8_t * dQ, * dQp; half * dX, * dXb, * dW, * dWb; float * dC, * dCref;
    CHECK(cudaMalloc(&dQ, qbytes));
    CHECK(cudaMalloc(&dQp, pbytes));
    CHECK(cudaMalloc(&dW, (size_t)M*K*sizeof(half)));     // row-major dequantized weights (reference)
    CHECK(cudaMalloc(&dWb, (size_t)M*K*sizeof(half)));
    CHECK(cudaMalloc(&dX, (size_t)N*K*sizeof(half)));
    CHECK(cudaMalloc(&dXb, (size_t)N*K*sizeof(half)));
    CHECK(cudaMalloc(&dC, (size_t)M*N*sizeof(float)));
    CHECK(cudaMalloc(&dCref, (size_t)M*N*sizeof(float)));
    CHECK(cudaMemcpy(dQ, hQ, qbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dQp, hP, pbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dX, hX, (size_t)N*K*sizeof(half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dXb, hXb, (size_t)N*K*sizeof(half), cudaMemcpyHostToDevice));

    // reference dequant + reference k-blocked weights copy (not used by kernel, kept for cuBLAS)
    dequant_q6k_ref<<<(int)((int64_t)M*(K/256)), 32>>>(dQ, dW, M, K);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    {
        half * hWcheck = (half *) malloc((size_t)M*K*sizeof(half));
        CHECK(cudaMemcpy(hWcheck, dW, (size_t)M*K*sizeof(half), cudaMemcpyDeviceToHost));
        double sw = 0; int nzw = 0; float mn = 1e30f, mx = -1e30f;
        for (size_t i = 0; i < (size_t)M*K; ++i) { const float v = __half2float(hWcheck[i]); sw += fabs(v); if (v != 0) nzw++; if (v < mn) mn = v; if (v > mx) mx = v; }
        printf("ref dequant weights: |W|=%.6e nonzero=%d min=%.6e max=%.6e  W[0][0..3]=%.4e %.4e %.4e %.4e\n",
               sw, nzw, mn, mx, __half2float(hWcheck[0]), __half2float(hWcheck[1]), __half2float(hWcheck[2]), __half2float(hWcheck[3]));
        free(hWcheck);
    }
    {
        // build the k-blocked fp16 weights from the dequantized reference (sanity: what the old pipeline would read)
        CHECK(cudaMemcpy(hW, dW, (size_t)M*K*sizeof(half), cudaMemcpyDeviceToHost));
        half * hWb = (half *) malloc((size_t)M*K*sizeof(half));
        for (int kbi = 0; kbi < K/BK; ++kbi) {
            for (int r = 0; r < M; ++r) {
                for (int k = 0; k < BK; ++k) {
                    hWb[((size_t)kbi*M + r)*BK + k] = hW[(size_t)r*K + kbi*BK + k];
                }
            }
        }
        CHECK(cudaMemcpy(dWb, hWb, (size_t)M*K*sizeof(half), cudaMemcpyHostToDevice));
        free(hWb);
    }

    const double flops = 2.0*M*N*K;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
    float ms_cublas = 0.0f;
    {
        const float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                     dX, CUDA_R_16F, K, dW, CUDA_R_16F, K, &beta,
                     dCref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
        for (int i = 0; i < 1; ++i) {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                         dX, CUDA_R_16F, K, dW, CUDA_R_16F, K, &beta,
                         dCref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaEventRecord(t0);
        for (int i = 0; i < reps; ++i) {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                         dX, CUDA_R_16F, K, dW, CUDA_R_16F, K, &beta,
                         dCref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_cublas, t0, t1);
        ms_cublas /= reps;
        printf("cuBLAS fp16 (ref weights): %.3f ms  %.2f TFLOPS\n", ms_cublas, flops/(ms_cublas*1e-3)/1e12);
    }

    const int smem_bytes = 2*BM*HSB + 2*BM*BKP*2 + 2*BN*BKP*2;
    CHECK(cudaFuncSetAttribute(gateA_q6k_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    printf("smem per block = %d bytes (%.1f KB)\n", smem_bytes, smem_bytes/1024.0);

    dim3 grid(M/BM, N/BN);
    dim3 block(32, NTHREAD/32);
    float ms_mine = 0.0f;
    {
        gateA_q6k_kernel<<<grid, block, smem_bytes>>>(dQp, dXb, dC, M, N, K, mode);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < 1; ++i) {
            gateA_q6k_kernel<<<grid, block, smem_bytes>>>(dQp, dXb, dC, M, N, K, mode);
        }
        cudaEventRecord(t0);
        for (int i = 0; i < reps; ++i) {
            gateA_q6k_kernel<<<grid, block, smem_bytes>>>(dQp, dXb, dC, M, N, K, mode);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_mine, t0, t1);
        ms_mine /= reps;
        CHECK(cudaGetLastError());
        printf("q6k kernel : %.3f ms  %.2f TFLOPS\n", ms_mine, flops/(ms_mine*1e-3)/1e12);
    }

    {
        const size_t n = (size_t)M*N;
        float * hC = (float *) malloc(n*sizeof(float));
        float * hR = (float *) malloc(n*sizeof(float));
        CHECK(cudaMemcpy(hC, dC, n*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hR, dCref, n*sizeof(float), cudaMemcpyDeviceToHost));
        double sum_ref = 0.0, sum_dif = 0.0, max_abs = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double r = hR[i], d = hC[i];
            sum_ref += r*r;
            sum_dif += (d-r)*(d-r);
            if (fabs(d-r) > max_abs) max_abs = fabs(d-r);
        }
        printf("correctness vs (dequant+cuBLAS): NMSE=%.3e  max_abs=%.4e  sum_ref=%.4e  sum_dif=%.4e\n", sum_dif/sum_ref, max_abs, sum_ref, sum_dif);
        double sc = 0, sr = 0; int nz = 0, nzr = 0;
        for (size_t i = 0; i < n; ++i) { sc += fabs(hC[i]); sr += fabs(hR[i]); if (hC[i] != 0) nz++; if (hR[i] != 0) nzr++; }
        printf("  |C|=%.6e (%d nonzero)   |Cref|=%.6e (%d nonzero)\n", sc, nz, sr, nzr);
        if (mode & (16|32|64|128)) {
            int bad = 0;
            double sref2 = 0, sdif2 = 0;
            for (int row = 0; row < BM; ++row) {
                for (int k = 0; k < 32; ++k) {
                    const float got = hC[row*BKP + k];
                    const float exp = __half2float(hW[(size_t)(0 + row)*K + ((mode & 128) ? 32 : 0) + k]);
                    sref2 += (double)exp*exp; sdif2 += (double)(got-exp)*(got-exp);
                    if (fabsf(got-exp) > 1e-4f && bad < 8) { printf("  panel[%d][%d] got %.5f exp %.5f\n", row, k, got, exp); bad++; }
                }
            }
            printf("unpacked panel check (block 0, slice 0): NMSE=%.3e bad=%d/2048\n", sdif2/sref2, bad);
            { FILE * fd = fopen("dump_panel.bin", "wb"); if (fd) { fwrite(hC, 1, 40960, fd); fclose(fd); printf("dumped 40KB\n"); } }
        }
        printf("  samples: C[0][0]=%.6e Cref[0][0]=%.6e  C[100][7]=%.6e Cref[100][7]=%.6e\n", hC[0], hR[0], hC[100*N+7], hR[100*N+7]);
        free(hC); free(hR);
    }

    free(hQ); free(hP); free(hX); free(hXb);
    cublasDestroy(h);
    cudaFree(dQ); cudaFree(dQp); cudaFree(dW); cudaFree(dWb); cudaFree(dX); cudaFree(dXb); cudaFree(dC); cudaFree(dCref);
    return 0;
}
