#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>

static void bench(const char * name, cublasHandle_t h, cublasOperation_t opA,
                  const void * A, cudaDataType_t tA, int lda,
                  const void * B, cudaDataType_t tB, int ldb,
                  void * C, cudaDataType_t tC, int ldc,
                  int m, int n, int k, cublasComputeType_t ct) {
    float alpha = 1.f, beta = 0.f;
    for (int i = 0; i < 3; i++) {
        cublasGemmEx(h, opA, CUBLAS_OP_N, m, n, k, &alpha, A, tA, lda, B, tB, ldb, &beta, C, tC, ldc, ct, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int iters = 10;
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) {
        cublasGemmEx(h, opA, CUBLAS_OP_N, m, n, k, &alpha, A, tA, lda, B, tB, ldb, &beta, C, tC, ldc, ct, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    double tf = 2.0*m*(double)n*k*iters/(ms/1000.0)/1e12;
    printf("%-34s m=%5d n=%6d k=%5d : %8.3f ms/call  %7.1f TFLOPS\n", name, m, n, k, ms/iters, tf);
}

int main() {
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);

    const int K = 5120, N = 17408;
    half  * dW;   cudaMalloc(&dW, (size_t)K*N*2);
    half  * dW2;  cudaMalloc(&dW2, (size_t)17408*5120*2);
    half  * dWlm; cudaMalloc(&dWlm, (size_t)K*248320*2);
    const int MAXM = 4096;
    half  * dX;   cudaMalloc(&dX, (size_t)17408*MAXM*2);
    float * dC;   cudaMalloc(&dC, (size_t)248320*MAXM*4);
    cudaMemset(dW, 0, (size_t)K*N*2); cudaMemset(dW2, 0, (size_t)17408*5120*2);
    cudaMemset(dWlm, 0, (size_t)K*248320*2);
    cudaMemset(dX, 0, (size_t)17408*MAXM*2);

    printf("V100 sm70 cuBLAS sweep, real Qwen3.8-27B shapes, fp16 in / fp32 out (llama.cpp cublas path)\n");
    for (int M : {512, 1024, 2048, 4096}) {
        bench("FFN gate/up  W[5120,17408]", h, CUBLAS_OP_T, dW, CUDA_R_16F, K, dX, CUDA_R_16F, K, dC, CUDA_R_32F, N, N, M, K, CUBLAS_COMPUTE_32F);
    }
    for (int M : {512, 1024, 2048, 4096}) {
        bench("FFN down     W[17408,5120]", h, CUBLAS_OP_T, dW2, CUDA_R_16F, 17408, dX, CUDA_R_16F, 17408, dC, CUDA_R_32F, 5120, 5120, M, 17408, CUBLAS_COMPUTE_32F);
    }
    for (int M : {512, 1024, 2048}) {
        bench("lm_head      W[5120,248320]", h, CUBLAS_OP_T, dWlm, CUDA_R_16F, K, dX, CUDA_R_16F, K, dC, CUDA_R_32F, 248320, 248320, M, K, CUBLAS_COMPUTE_32F);
    }
    for (int M : {512, 2048}) {
        bench("fp16out gate/up", h, CUBLAS_OP_T, dW, CUDA_R_16F, K, dX, CUDA_R_16F, K, (half*)dC, CUDA_R_16F, N, N, M, K, CUBLAS_COMPUTE_32F);
    }
    cudaError_t err = cudaGetLastError();
    printf("last err: %s\n", cudaGetErrorString(err));
    return 0;
}
