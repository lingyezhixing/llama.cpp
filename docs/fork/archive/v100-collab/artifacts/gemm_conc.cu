// Decisive test: can a non-TC kernel run concurrently with a saturating cuBLAS GEMM on V100?
// Mimics the real per-layer mix: 4x cuBLAS GEMM (M=2048, FFN shapes) vs 1x bandwidth kernel + 1x fp32-fma kernel
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s at %d\n", cudaGetErrorString(e), __LINE__); return 1; } } while(0)

// bandwidth kernel: read n float4, write n float4 (mimics dequant traffic)
__global__ void bw_kernel(const float4 * __restrict__ in, float4 * __restrict__ out, const long long n) {
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x*blockDim.x;
    for (; i < n; i += stride) {
        float4 v = in[i];
        v.x += 1.f; v.y += 1.f; v.z += 1.f; v.w += 1.f;
        out[i] = v;
    }
}

// fp32 fma kernel with serial dependency chain (mimics GDN-style non-TC compute), modest grid
__global__ void fma_kernel(const float * __restrict__ in, float * __restrict__ out, const long long n, const int iters) {
    long long i = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x*blockDim.x;
    for (; i < n; i += stride) {
        float a = in[i], b = in[i] + 1.f;
        #pragma unroll 4
        for (int k = 0; k < iters; ++k) { a = fmaf(a, 1.0000001f, b); }
        out[i] = a;
    }
}

static float bench_stream(cudaStream_t s, int iters, cublasHandle_t h, void * A, void * X, void * C,
                          const float4 * bw_in, float4 * bw_out, long long bw_n,
                          const float * fma_in, float * fma_out, long long fma_n, int fma_iters,
                          bool do_gemm, bool do_bw, bool do_fma, int grid_scale) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cublasSetStream(h, s);
    float alpha = 1.f, beta = 0.f;
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; i++) {
        if (do_gemm) {
            for (int g = 0; g < 4; g++)
                cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 17408, 2048, 5120, &alpha, A, CUDA_R_16F, 5120,
                             X, CUDA_R_16F, 5120, &beta, C, CUDA_R_32F, 17408, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        if (do_bw) bw_kernel<<<4096*grid_scale/4, 256, 0, s>>>(bw_in, bw_out, bw_n);
        if (do_fma) fma_kernel<<<160, 256, 0, s>>>(fma_in, fma_out, fma_n, fma_iters);
    }
    cudaEventRecord(e1, s); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

int main(int argc, char ** argv) {
    cudaSetDevice(0);
    const int M = 2048, K = 5120, N = 17408;
    half  * dA; CHECK(cudaMalloc(&dA, (size_t)K*N*2));
    half  * dX; CHECK(cudaMalloc(&dX, (size_t)K*M*2));
    float * dC; CHECK(cudaMalloc(&dC, (size_t)N*M*4));
    // dequant-equivalent traffic per layer: ~105MB read + ~260MB write -> use 47M float4 (150MB) in + out
    const long long bw_n = 47LL*1024*1024/4;
    float4 * bw_in; float4 * bw_out;
    CHECK(cudaMalloc(&bw_in,  bw_n*16));
    CHECK(cudaMalloc(&bw_out, bw_n*16));
    const long long fma_n = 10LL*1024*1024;  // 10M threads-worth of fma work
    float * fma_in; float * fma_out;
    CHECK(cudaMalloc(&fma_in,  fma_n*4));
    CHECK(cudaMalloc(&fma_out, fma_n*4));
    CHECK(cudaMemset(dA, 0, (size_t)K*N*2)); CHECK(cudaMemset(dX, 0, (size_t)K*M*2));
    CHECK(cudaMemset(bw_in, 0, bw_n*16)); CHECK(cudaMemset(fma_in, 0, fma_n*4));

    cublasHandle_t h; cublasCreate(&h);
    cudaStream_t sA, sB, sBhi, sBlo;
    cudaStreamCreate(&sA);
    cudaStreamCreate(&sB);
    int lo, hi; cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStreamCreateWithPriority(&sBhi, cudaStreamNonBlocking, hi);
    cudaStreamCreateWithPriority(&sBlo, cudaStreamNonBlocking, lo);
    printf("stream priority range: low=%d high=%d\n", lo, hi);

    const int iters = 10;
    const int fma_iters = 2048;

    // warmup
    bench_stream(sA, 1, h, dA, dX, dC, bw_in, bw_out, bw_n, fma_in, fma_out, fma_n, fma_iters, true, false, false, 4);
    bench_stream(sB, 1, h, dA, dX, dC, bw_in, bw_out, bw_n, fma_in, fma_out, fma_n, fma_iters, false, true, true, 4);
    cudaDeviceSynchronize();

    float t_gemm = bench_stream(sA, iters, h, dA, dX, dC, bw_in, bw_out, bw_n, fma_in, fma_out, fma_n, fma_iters, true, false, false, 4);
    float t_bw   = bench_stream(sB, iters, h, dA, dX, dC, bw_in, bw_out, bw_n, fma_in, fma_out, fma_n, fma_iters, false, true, false, 4);
    float t_fma  = bench_stream(sB, iters, h, dA, dX, dC, bw_in, bw_out, bw_n, fma_in, fma_out, fma_n, fma_iters, false, false, true, 4);
    float t_bw_fma = bench_stream(sB, iters, h, dA, dX, dC, bw_in, bw_out, bw_n, fma_in, fma_out, fma_n, fma_iters, false, true, true, 4);
    printf("alone:  gemm=%.2fms  bw=%.2fms  fma=%.2fms  bw+fma=%.2fms  (sum gemm+bw+fma=%.2f)\n",
           t_gemm, t_bw, t_fma, t_bw_fma, t_gemm+t_bw+t_fma);

    // concurrent: GEMM on sA, others on sB (default priority), then hi, then lo
    for (int mode = 0; mode < 3; mode++) {
        cudaStream_t sBx = mode == 0 ? sB : (mode == 1 ? sBhi : sBlo);
        cudaEvent_t evA, evB, ev0, ev1;
        cudaEventCreate(&evA); cudaEventCreate(&evB); cudaEventCreate(&ev0); cudaEventCreate(&ev1);
        cudaEventRecord(ev0);
        // launch A on sA
        cublasSetStream(h, sA);
        float alpha = 1.f, beta = 0.f;
        cudaEventRecord(evA, sA);
        // record start marker on sA
        for (int i = 0; i < iters; i++) {
            for (int g = 0; g < 4; g++)
                cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 17408, 2048, 5120, &alpha, dA, CUDA_R_16F, 5120,
                             dX, CUDA_R_16F, 5120, &beta, dC, CUDA_R_32F, 17408, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            bw_kernel<<<4096, 256, 0, sBx>>>(bw_in, bw_out, bw_n);
            fma_kernel<<<160, 256, 0, sBx>>>(fma_in, fma_out, fma_n, fma_iters);
        }
        cudaEventRecord(ev1, sA);
        cudaEventSynchronize(ev1);
        float ms; cudaEventElapsedTime(&ms, ev0, ev1);
        const char * tag = mode == 0 ? "default" : (mode == 1 ? "B=HIGH" : "B=LOW");
        printf("concurrent (sB %s): total=%.2fms   ideal(no overlap)=%.2f   perfect-overlap=%.2f   -> overlap_ratio=%.2f\n",
               tag, ms, t_gemm + t_bw_fma, t_gemm, (t_gemm + t_bw_fma)/ms);
        cudaEventDestroy(evA); cudaEventDestroy(evB); cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    }
    CHECK(cudaGetLastError());
    return 0;
}
