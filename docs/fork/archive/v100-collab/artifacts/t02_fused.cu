// T02 Stage 1: fused Q6_K dequant + fp16 tensor-core GEMM microbenchmark (sm70, V100)
// Shape: C[M=512, N=17408] += X[M,K=5120] (f16) * W[N,K]^T (Q6_K), C is f32.
// mode bits: 1=dequant B tile, 2=mma, 8=load A tile (diagnostics)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

namespace wmma = nvcuda::wmma;

#define QK_K 256
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; half d; } block_q6_K;
static_assert(sizeof(block_q6_K) == 210, "q6_K block size");

__device__ __forceinline__ void dequant_q6_K_8(half * __restrict__ out, const block_q6_K * __restrict__ x, const int lane) {
    const int64_t e0 = 8*lane;
    const float d = __half2float(x->d) * (float)x->scales[e0 >> 4];
    const uint8_t * ql = x->ql + (e0 & 63) + ((e0 >> 7) << 6);
    const uint8_t * qh = x->qh + (e0 & 31) + ((e0 >> 7) << 5);
    const int  shift = 2*((e0 >> 5) & 3);
    const bool hi    = ((e0 >> 6) & 1) != 0;
    const uint16_t qw[4] = { *(const uint16_t *)(ql+0), *(const uint16_t *)(ql+2), *(const uint16_t *)(ql+4), *(const uint16_t *)(ql+6) };
    const uint16_t hw[4] = { *(const uint16_t *)(qh+0), *(const uint16_t *)(qh+2), *(const uint16_t *)(qh+4), *(const uint16_t *)(qh+6) };
    half v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t qb = (i & 1) ? (qw[i/2] >> 8) : (qw[i/2] & 0xFF);
        const uint32_t hb = (i & 1) ? (hw[i/2] >> 8) : (hw[i/2] & 0xFF);
        const uint32_t nib = hi ? ((qb >> 4) & 0xF) : (qb & 0xF);
        const int32_t  q6  = (int32_t)(nib | (((hb >> shift) & 3) << 4)) - 32;
        v[i] = __float2half(d * (float)q6);
    }
    uint4 out4;
    out4.x = *(uint16_t*)&v[0] | (*(uint16_t*)&v[1] << 16);
    out4.y = *(uint16_t*)&v[2] | (*(uint16_t*)&v[3] << 16);
    out4.z = *(uint16_t*)&v[4] | (*(uint16_t*)&v[5] << 16);
    out4.w = *(uint16_t*)&v[6] | (*(uint16_t*)&v[7] << 16);
    *(uint4 *)out = out4;
}

__global__ void dequant_ref_kernel(const block_q6_K * __restrict__ w, half * __restrict__ out, const int64_t nblk) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= nblk) return;
    dequant_q6_K_8(out + warp*QK_K, w + warp, lane);
}

#define BM 64
#define BN 32
#define BK 256
#define NTHREAD 128
#define BKP (BK+8)
#define SMEM_A (BM*BKP*2)
#define SMEM_B (BN*BKP*2)

__global__ void __launch_bounds__(NTHREAD, 2)
fused_q6k_gemm_kernel(const block_q6_K * __restrict__ W, const half * __restrict__ X, float * __restrict__ C,
                      const int M, const int N, const int K, const int mode) {
    extern __shared__ half smem[];
    half * sA = smem;            // [BM][BK]
    half * sB = smem + BM*BKP;    // [BN][BKP] (padded)

    const int m0 = blockIdx.y * BM;
    const int n0 = blockIdx.x * BN;
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int64_t nblk_per_row = K / QK_K;

    const int wm = (warp & 1) * 32;
    const int wn = (warp >> 1) * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2];
    wmma::fill_fragment(c[0], 0.f);
    wmma::fill_fragment(c[1], 0.f);

    for (int kb = 0; kb < K/BK; kb++) {
        if (mode & 8) {
            const int arow = m0 + (threadIdx.x >> 1);
            const int acol = (threadIdx.x & 1) * 128;
            const half * src = X + arow*(int64_t)K + kb*BK + acol;
            float4 * dst = (float4 *)(sA + (threadIdx.x >> 1)*BKP + acol);
#pragma unroll
            for (int i = 0; i < 16; i++) dst[i] = ((const float4 *)src)[i];
        }
        if (mode & 1) {
            for (int r = 0; r < BN/4; r++) {
                const int row = r*4 + warp;
                const block_q6_K * src = W + (int64_t)(n0 + row)*nblk_per_row + kb;
                dequant_q6_K_8(sB + row*BKP, src, lane);
            }
        }
        __syncthreads();

        if (mode & 2) {
#pragma unroll
            for (int kk = 0; kk < BK/16; kk++) {
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                    wmma::load_matrix_sync(a, sA + (wm + i*16)*BKP + kk*16, BKP);
                    wmma::load_matrix_sync(b, sB + (wn + 0)*BKP + kk*16, BKP);
                    wmma::mma_sync(c[i], a, b, c[i]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
        wmma::store_matrix_sync(C + (m0 + wm + i*16)*(int64_t)N + (n0 + wn), c[i], N, wmma::mem_row_major);
    }
}

int main(int argc, char ** argv) {
    const int M = 512, N = 17408, K = 5120;
    const int iters = argc > 1 ? atoi(argv[1]) : 20;
    cudaSetDevice(0);
    const int64_t nblk_per_row = K / QK_K;
    const size_t w_bytes = (size_t)N * nblk_per_row * sizeof(block_q6_K);

    block_q6_K * dW; half * dWdq; half * dX; float * dC; float * dCref;
    CHECK(cudaMalloc(&dW, w_bytes));
    CHECK(cudaMalloc(&dWdq, (size_t)N*K*2));
    CHECK(cudaMalloc(&dX, (size_t)M*K*2));
    CHECK(cudaMalloc(&dC, (size_t)M*N*4));
    CHECK(cudaMalloc(&dCref, (size_t)M*N*4));

    {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> u8(0, 255);
        std::uniform_real_distribution<float> uf(0.f, 1.f);
        std::vector<block_q6_K> hw((size_t)N*nblk_per_row);
        for (auto & b : hw) {
            for (int i = 0; i < QK_K/2; i++) b.ql[i] = (uint8_t)u8(rng);
            for (int i = 0; i < QK_K/4; i++) b.qh[i] = (uint8_t)u8(rng);
            for (int i = 0; i < QK_K/16; i++) b.scales[i] = (int8_t)(u8(rng)-128);
            b.d = __float2half(0.01f * uf(rng) + 0.001f);
        }
        CHECK(cudaMemcpy(dW, hw.data(), w_bytes, cudaMemcpyHostToDevice));
        std::vector<half> hx((size_t)M*K);
        for (auto & v : hx) v = __float2half(uf(rng)*2.f - 1.f);
        CHECK(cudaMemcpy(dX, hx.data(), (size_t)M*K*2, cudaMemcpyHostToDevice));
    }

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const double flop = 2.0*M*N*(double)K;

    const int64_t nblk = (int64_t)N*nblk_per_row;
    const int dq_threads = 256;
    const int dq_blocks = (int)((nblk*32 + dq_threads - 1)/dq_threads);
    dequant_ref_kernel<<<dq_blocks, dq_threads>>>(dW, dWdq, nblk);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) dequant_ref_kernel<<<dq_blocks, dq_threads>>>(dW, dWdq, nblk);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_dq; cudaEventElapsedTime(&ms_dq, e0, e1); ms_dq /= iters;
    printf("dequant kernel            : %8.3f ms  (%.1f GB/s traffic-based)\n", ms_dq, (w_bytes + (size_t)N*K*2)/ms_dq/1e6);

    cublasHandle_t h; cublasCreate(&h);
    void * ws; CHECK(cudaMalloc(&ws, 4u*1024*1024));
    cublasSetWorkspace(h, ws, 4u*1024*1024);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    float alpha = 1.f, beta = 0.f;
    for (int i = 0; i < 3; i++) cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dWdq, CUDA_R_16F, K, dX, CUDA_R_16F, K, &beta, dCref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK(cudaDeviceSynchronize());
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dWdq, CUDA_R_16F, K, dX, CUDA_R_16F, K, &beta, dCref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms_blas; cudaEventElapsedTime(&ms_blas, e0, e1); ms_blas /= iters;
    printf("cuBLAS (f16 weights)      : %8.3f ms  %7.1f TFLOPS\n", ms_blas, flop/ms_blas/1e9);
    printf("BASELINE total (dq+blas)  : %8.3f ms  effective %7.1f TFLOPS\n", ms_dq+ms_blas, flop/(ms_dq+ms_blas)/1e9);

    dim3 grid((N + BN - 1)/BN, (M + BM - 1)/BM);
    const int smem = SMEM_A + SMEM_B;
    CHECK(cudaFuncSetAttribute(fused_q6k_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    // diagnostics: mode sweep
    for (int mode : {1, 2, 8, 9, 10, 11, 15}) {
        fused_q6k_gemm_kernel<<<grid, NTHREAD, smem>>>(dW, dX, dC, M, N, K, mode);
        CHECK(cudaDeviceSynchronize());
        cudaEventRecord(e0);
        for (int i = 0; i < 5; i++) fused_q6k_gemm_kernel<<<grid, NTHREAD, smem>>>(dW, dX, dC, M, N, K, mode);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= 5;
        printf("mode %2d: %8.3f ms  %7.1f TFLOPS\n", mode, ms, flop/ms/1e9);
    }
    // full run timing
    {
        const int mode = 15;
        cudaMemset(dC, 0, (size_t)M*N*4);
        fused_q6k_gemm_kernel<<<grid, NTHREAD, smem>>>(dW, dX, dC, M, N, K, mode);
        CHECK(cudaDeviceSynchronize());
        cudaEventRecord(e0);
        for (int i = 0; i < iters; i++) fused_q6k_gemm_kernel<<<grid, NTHREAD, smem>>>(dW, dX, dC, M, N, K, mode);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms_f; cudaEventElapsedTime(&ms_f, e0, e1); ms_f /= iters;
        printf("FUSED kernel (full)       : %8.3f ms  %7.1f TFLOPS  (%.0f%% of 125)\n", ms_f, flop/ms_f/1e9, 100.0*flop/ms_f/1e9/125.0);
        printf("FUSED vs baseline         : %+.1f%%\n", 100.0*((ms_dq+ms_blas)/ms_f - 1.0));
    }
    {
        std::vector<float> hc((size_t)M*N), hr((size_t)M*N);
        CHECK(cudaMemcpy(hc.data(), dC, (size_t)M*N*4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hr.data(), dCref, (size_t)M*N*4, cudaMemcpyDeviceToHost));
        double max_abs = 0, max_rel = 0, sum_abs = 0, sum_ref = 0;
        for (size_t i = 0; i < hc.size(); i++) {
            const double a = hc[i], b = hr[i];
            if (a != a || b != b) { printf("NaN at %zu: fused=%f ref=%f\n", i, a, b); break; }
            max_abs = fmax(max_abs, fabs(a-b));
            max_rel = fmax(max_rel, fabs(a-b)/(fabs(b)+1e-3));
            sum_abs += fabs(a-b); sum_ref += fabs(b);
        }
        printf("correctness: max_abs=%.4f  max_rel=%.4f  sum_rel=%.2e\n", max_abs, max_rel, sum_abs/(sum_ref+1e-9));
    }
    return 0;
}
