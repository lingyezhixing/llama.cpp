// T01 addendum: ABAB test - is the 256MB workspace really better than 4MB for the down / gate-up shapes?
// Alternates ws4 / ws256 five times each to control for clock/state drift.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %d %s\n", __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

struct shp { const char * n; int Nout; int Nt; int K; };

static float run(cublasHandle_t h, cublasGemmAlgo_t algo, const half*A, const half*B, float*C,
                 int m, int n, int k, int iters, double flop) {
    float a = 1.f, b = 0.f;
    for (int i = 0; i < 3; i++) cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a, A, CUDA_R_16F, k, B, CUDA_R_16F, k, &b, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, algo);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a, A, CUDA_R_16F, k, B, CUDA_R_16F, k, &b, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, algo);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / iters;
}

int main() {
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);
    void * ws; CHECK(cudaMalloc(&ws, 256u*1024*1024));

    shp shapes[] = {
        { "ffn_down      ", 5120,  512, 17408 },
        { "ffn_gate/up   ", 17408, 512, 5120  },
        { "attn_o/ssm_out", 5120,  512, 6144  },
    };
    const int ROUNDS = 5;
    for (auto & s : shapes) {
        const int m = s.Nout, n = s.Nt, k = s.K;
        const double flop = 2.0*m*n*(double)k;
        half * A, * B; float * C;
        CHECK(cudaMalloc(&A, (size_t)k*m*2)); CHECK(cudaMalloc(&B, (size_t)k*n*2)); CHECK(cudaMalloc(&C, (size_t)m*n*4));
        CHECK(cudaMemset(A,0,(size_t)k*m*2)); CHECK(cudaMemset(B,0,(size_t)k*n*2));
        double sum4 = 0, sum256 = 0, sumLt = 0;
        printf("%s : tokens=%d out=%d K=%d\n", s.n, n, m, k);
        for (int r = 0; r < ROUNDS; r++) {
            cublasSetWorkspace(h, ws, 4u*1024*1024);
            float t4 = run(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, A, B, C, m, n, k, 30, flop);
            cublasSetWorkspace(h, ws, 256u*1024*1024);
            float t256 = run(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, A, B, C, m, n, k, 30, flop);
            float t112 = run(h, CUBLAS_GEMM_ALGO12_TENSOR_OP, A, B, C, m, n, k, 30, flop);
            sum4 += t4; sum256 += t256; sumLt += t112;
            printf("  r%d: ws4=%7.3fms(%5.1fTF)  ws256=%7.3fms(%5.1fTF)  algo112=%7.3fms(%5.1fTF)\n",
                   r, t4, flop/t4/1e9, t256, flop/t256/1e9, t112, flop/t112/1e9);
        }
        printf("  AVG: ws4=%5.1f TF | ws256=%5.1f TF (%+.1f%%) | algo112=%5.1f TF (%+.1f%%)\n\n",
               flop/(sum4/ROUNDS)/1e9, flop/(sum256/ROUNDS)/1e9,
               100.0*((sum4/ROUNDS)/(sum256/ROUNDS)-1.0), flop/(sumLt/ROUNDS)/1e9,
               100.0*((sum4/ROUNDS)/(sumLt/ROUNDS)-1.0));
        cublasSetWorkspace(h, nullptr, 0);
        CHECK(cudaFree(A)); CHECK(cudaFree(B)); CHECK(cudaFree(C));
    }
    cudaFree(ws); cublasDestroy(h);
    return 0;
}
