// T02 Gate A v2: fp16 MMA pipeline using nvcuda::wmma 16x16x16 fragments (k-extent 16 per load)
// C[M,N] = W[M,K] * X[N,K]^T, fp16 inputs, fp32 accum.
// A operand = weights [M,K], B operand = activations [N,K] (col-major B == [n][k] storage).
// BM=128 BN=128 BK=32, 2 stages, 256 threads (8 warps, 2x4 over the tile).
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace nvcuda;

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float>              frag_c;

#define BM 64
#define BN 128
#define BK 32
#define BKP (BK + 8)
#define NTHREAD 256
#define WM 2
#define WN 4
#define WMA 2   // 16-row A subtiles per warp (64 rows / 16)
#define WNB 2   // 16-token B subtiles per warp (32 tokens / 16)

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void __launch_bounds__(NTHREAD, 2)
gateA_kernel(const half * __restrict__ W, const half * __restrict__ X, float * __restrict__ C,
             const int M, const int N, const int K, const int mode) {
    __shared__ half sA[2][BM*BKP];
    __shared__ half sB[2][BN*BKP];

    const int lane  = threadIdx.x;
    const int warp  = threadIdx.y;
    const int tid   = warp*32 + lane;
    const int wm    = warp % WM;
    const int wn    = warp / WM;
    // grid.x = token blocks (fastest) so consecutive blocks share the weight panel in L2
    const int m0    = blockIdx.y * BM;
    const int n0    = blockIdx.x * BN;
    const int nkb   = K / BK;

    float4 regA[2];
    float4 regB[2];

    {
        const int row = tid / 4;
        const int part = tid % 4;
        regA[0] = *reinterpret_cast<const float4 *>(W + ((int64_t)(0*M + m0 + row)*BK) + part*8);
        regA[1] = *reinterpret_cast<const float4 *>(W + ((int64_t)(0*M + m0 + row + 64)*BK) + part*8);
        regB[0] = *reinterpret_cast<const float4 *>(X + ((int64_t)(0*N + n0 + row)*BK) + part*8);
        regB[1] = *reinterpret_cast<const float4 *>(X + ((int64_t)(0*N + n0 + row + 64)*BK) + part*8);
        *reinterpret_cast<float4 *>(sA[0] + row*BKP + part*8) = regA[0];
        *reinterpret_cast<float4 *>(sA[0] + (row + 64)*BKP + part*8) = regA[1];
        *reinterpret_cast<float4 *>(sB[0] + row*BKP + part*8) = regB[0];
        *reinterpret_cast<float4 *>(sB[0] + (row + 64)*BKP + part*8) = regB[1];
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
        const bool have_next = kb + 1 < nkb;

        if (have_next) {
            const int row = tid / 4;
            const int part = tid % 4;
            const int64_t koff = (int64_t)(kb + 1)*BK + part*8;
            regA[0] = *reinterpret_cast<const float4 *>(W + (int64_t)(kb + 1)*M*BK + (int64_t)(m0 + row)*BK + part*8);
            regA[1] = *reinterpret_cast<const float4 *>(W + (int64_t)(kb + 1)*M*BK + (int64_t)(m0 + row + 64)*BK + part*8);
            regB[0] = *reinterpret_cast<const float4 *>(X + (int64_t)(kb + 1)*N*BK + (int64_t)(n0 + row)*BK + part*8);
            regB[1] = *reinterpret_cast<const float4 *>(X + (int64_t)(kb + 1)*N*BK + (int64_t)(n0 + row + 64)*BK + part*8);
        }

        if (!(mode & 2)) {
#pragma unroll
            for (int ks = 0; ks < BK/16; ++ks) {
                frag_a a[WMA];
                frag_b b[WNB];
#pragma unroll
                for (int ia = 0; ia < WMA; ++ia) {
                    wmma::load_matrix_sync(a[ia], sA[stage] + (wm*(BM/WM) + ia*16)*BKP + ks*16, BKP);
                }
#pragma unroll
                for (int ib = 0; ib < WNB; ++ib) {
                    wmma::load_matrix_sync(b[ib], sB[stage] + (wn*(BN/WN) + ib*16)*BKP + ks*16, BKP);
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

        if (have_next) {
            const int row = tid / 4;
            const int part = tid % 4;
            const int ns = stage ^ 1;
            *reinterpret_cast<float4 *>(sA[ns] + row*BKP + part*8) = regA[0];
            *reinterpret_cast<float4 *>(sA[ns] + (row + 64)*BKP + part*8) = regA[1];
            *reinterpret_cast<float4 *>(sB[ns] + row*BKP + part*8) = regB[0];
            *reinterpret_cast<float4 *>(sB[ns] + (row + 64)*BKP + part*8) = regB[1];
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
    const int M = argc > 1 ? atoi(argv[1]) : 17408;
    const int N = argc > 2 ? atoi(argv[2]) : 512;
    const int K = argc > 3 ? atoi(argv[3]) : 5120;
    const int reps = argc > 4 ? atoi(argv[4]) : 50;
    const int mode = argc > 5 ? atoi(argv[5]) : 0;
    const int device = argc > 6 ? atoi(argv[6]) : 0;

    CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    printf("device: %s  mode=%d  M=%d N=%d K=%d reps=%d\n", prop.name, mode, M, N, K, reps);

    half * dW, * dX, * dWb, * dXb;
    float * dC, * dCref;
    CHECK(cudaMalloc(&dW, (size_t)M*K*sizeof(half)));
    CHECK(cudaMalloc(&dX, (size_t)N*K*sizeof(half)));
    CHECK(cudaMalloc(&dWb, (size_t)M*K*sizeof(half)));
    CHECK(cudaMalloc(&dXb, (size_t)N*K*sizeof(half)));
    CHECK(cudaMalloc(&dC, (size_t)M*N*sizeof(float)));
    CHECK(cudaMalloc(&dCref, (size_t)M*N*sizeof(float)));

    half * hW = (half *) malloc((size_t)M*K*sizeof(half));
    half * hX = (half *) malloc((size_t)N*K*sizeof(half));
    {
        srand(1234);
        for (size_t i = 0; i < (size_t)M*K; ++i) hW[i] = __float2half(((rand() % 2001) - 1000) / 1000.0f);
        for (size_t i = 0; i < (size_t)N*K; ++i) hX[i] = __float2half(((rand() % 2001) - 1000) / 1000.0f);
        CHECK(cudaMemcpy(dW, hW, (size_t)M*K*sizeof(half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dX, hX, (size_t)N*K*sizeof(half), cudaMemcpyHostToDevice));

        // blocked layout: [kb][row][BK] so a (row, k-block) panel is contiguous
        half * hWb = (half *) malloc((size_t)M*K*sizeof(half));
        half * hXb = (half *) malloc((size_t)N*K*sizeof(half));
        for (int kb = 0; kb < K/BK; ++kb) {
            for (int r = 0; r < M; ++r) {
                for (int k = 0; k < BK; ++k) {
                    hWb[((size_t)kb*M + r)*BK + k] = hW[(size_t)r*K + kb*BK + k];
                }
            }
            for (int r = 0; r < N; ++r) {
                for (int k = 0; k < BK; ++k) {
                    hXb[((size_t)kb*N + r)*BK + k] = hX[(size_t)r*K + kb*BK + k];
                }
            }
        }
        CHECK(cudaMemcpy(dWb, hWb, (size_t)M*K*sizeof(half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dXb, hXb, (size_t)N*K*sizeof(half), cudaMemcpyHostToDevice));
        free(hWb); free(hXb);
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
        for (int i = 0; i < 3; ++i) {
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
        printf("cuBLAS fp16: %.3f ms  %.2f TFLOPS\n", ms_cublas, flops/(ms_cublas*1e-3)/1e12);
    }

    const dim3 grid((N + BN - 1)/BN, (M + BM - 1)/BM);
    const dim3 block(32, NTHREAD/32);
    float ms_mine = 0.0f;
    {
        gateA_kernel<<<grid, block>>>(dWb, dXb, dC, M, N, K, mode);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < 3; ++i) {
            gateA_kernel<<<grid, block>>>(dWb, dXb, dC, M, N, K, mode);
        }
        cudaEventRecord(t0);
        for (int i = 0; i < reps; ++i) {
            gateA_kernel<<<grid, block>>>(dWb, dXb, dC, M, N, K, mode);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_mine, t0, t1);
        ms_mine /= reps;
        CHECK(cudaGetLastError());
        printf("mine       : %.3f ms  %.2f TFLOPS\n", ms_mine, flops/(ms_mine*1e-3)/1e12);
    }

    {
        const size_t n = (size_t)M*N;
        float * hC = (float *) malloc(n*sizeof(float));
        float * hR = (float *) malloc(n*sizeof(float));
        CHECK(cudaMemcpy(hC, dC, n*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hR, dCref, n*sizeof(float), cudaMemcpyDeviceToHost));
        double sum_ref = 0.0, sum_dif = 0.0, max_abs = 0.0, max_rel = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double r = hR[i], d = hC[i];
            sum_ref += r*r;
            sum_dif += (d-r)*(d-r);
            if (fabs(d-r) > max_abs) max_abs = fabs(d-r);
            if (fabs(r) > 1.0) {
                const double rel = fabs(d-r)/fabs(r);
                if (rel > max_rel) max_rel = rel;
            }
        }
        printf("correctness vs cuBLAS: NMSE=%.3e  max_abs=%.4e  max_rel(|ref|>1)=%.3e\n", sum_dif/sum_ref, max_abs, max_rel);
        free(hC); free(hR);
    }

    free(hW); free(hX);
    cublasDestroy(h);
    cudaFree(dW); cudaFree(dX); cudaFree(dWb); cudaFree(dXb); cudaFree(dC); cudaFree(dCref);
    return 0;
}
