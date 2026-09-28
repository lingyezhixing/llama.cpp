// Does the dequant-write before each GEMM (as in the real model) explain the model-side
// 78 TF vs standalone 85 TF gap? Modes:
//   0 : GEMM only, back-to-back
//   1 : write A (178MB, like dequant output) then GEMM, time GEMM only
//   2 : write a DIFFERENT 178MB buffer D then GEMM (same DRAM writeback traffic, A stays clean)
//   3 : write A then a short dummy kernel (~0.3ms) then GEMM
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("err %d %s\n", __LINE__, cudaGetErrorString(e)); return 1;} } while (0)

__global__ void spin_kernel(float * p, int n, int reps) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = p[i];
    for (int r = 0; r < reps; r++) v = fmaf(v, 1.000001f, 0.000001f);
    p[i] = v;
}

int main(int argc, char ** argv) {
    const int m = 17408, n = 512, k = 5120;
    const int iters = argc > 1 ? atoi(argv[1]) : 300;
    const int mode  = argc > 2 ? atoi(argv[2]) : 0;
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    void * ws; CHECK(cudaMalloc(&ws, 4u*1024*1024));
    cublasSetWorkspace(h, ws, 4u*1024*1024);
    half * A, * B; float * C, * D;
    const size_t abytes = (size_t)k*m*2;
    CHECK(cudaMalloc(&A, abytes)); CHECK(cudaMalloc(&B, (size_t)k*n*2)); CHECK(cudaMalloc(&C, (size_t)m*n*4));
    CHECK(cudaMalloc(&D, abytes));
    CHECK(cudaMemset(A, 1, abytes)); CHECK(cudaMemset(B, 1, (size_t)k*n*2)); CHECK(cudaMemset(D, 1, abytes));
    float a = 1.f, b = 0.f;
    auto gemm = [&]() {
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a, A, CUDA_R_16F, k, B, CUDA_R_16F, k, &b, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    for (int i = 0; i < 5; i++) gemm();
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<double> ts;
    for (int i = 0; i < iters; i++) {
        if (mode == 1 || mode == 3) CHECK(cudaMemsetAsync(A, 1, abytes));
        if (mode == 2)              CHECK(cudaMemsetAsync(D, 1, abytes));
        if (mode == 3) spin_kernel<<<80*2048/256, 256>>>(D, 80*2048, 2);
        cudaEventRecord(e0);
        gemm();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        ts.push_back(ms);
    }
    std::vector<double> s = ts; std::sort(s.begin(), s.end());
    double avg_first = 0, avg_last = 0;
    for (int i = 0; i < 20; i++) avg_first += ts[i];
    for (int i = 0; i < 20; i++) avg_last += ts[iters-1-i];
    printf("mode %d, iters %d: first20=%.4f ms  last20=%.4f ms  median=%.4f ms  min=%.4f  (%.2f TF median, %.2f TF best)\n",
           mode, iters, avg_first/20, avg_last/20, s[iters/2], s[0], 2.0*m*n*k/s[iters/2]/1e9, 2.0*m*n*k/s[0]/1e9);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(D); cudaFree(ws); cublasDestroy(h);
    return 0;
}
