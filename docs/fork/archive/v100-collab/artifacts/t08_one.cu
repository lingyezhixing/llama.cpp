// minimal: only the llama.cpp-exact gate/up GEMM, repeated
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("err %d %s\n", __LINE__, cudaGetErrorString(e)); return 1;} } while (0)
int main(int argc, char ** argv) {
    const int m = 17408, n = 512, k = 5120;
    const int iters = argc > 1 ? atoi(argv[1]) : 50;
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    void * ws; CHECK(cudaMalloc(&ws, 4u*1024*1024));
    cublasSetWorkspace(h, ws, 4u*1024*1024);
    half * A, * B; float * C;
    CHECK(cudaMalloc(&A, (size_t)k*m*2)); CHECK(cudaMalloc(&B, (size_t)k*n*2)); CHECK(cudaMalloc(&C, (size_t)m*n*4));
    CHECK(cudaMemset(A, 1, (size_t)k*m*2)); CHECK(cudaMemset(B, 1, (size_t)k*n*2));
    float a = 1.f, b = 0.f;
    for (int i = 0; i < 3; i++) cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a, A, CUDA_R_16F, k, B, CUDA_R_16F, k, &b, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a, A, CUDA_R_16F, k, B, CUDA_R_16F, k, &b, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= iters;
    printf("gate/up llama.cpp-exact: %.4f ms  %.2f TFLOPS\n", ms, 2.0*m*n*k/ms/1e9);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ws); cublasDestroy(h);
    return 0;
}
