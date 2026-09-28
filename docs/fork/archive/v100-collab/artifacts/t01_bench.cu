// T01: cuBLAS / cuBLASLt efficiency check for Qwen3.8-27B GEMM shapes on V100 (sm70)
// Convention: weight W is [K, Nout] row-major, act X is [K, Nt] row-major.
//   cuBLAS call: CUBLAS_OP_T x CUBLAS_OP_N, m = Nout, n = Nt, k = K, lda=K, ldb=K, ldc=Nout
//   i.e. C[Nout, Nt] = W^T [Nout,K] * X[K,Nt]
// Reports TFLOPS = 2*m*n*k / t.  Also prints the (analysis-friendly) math view: (tokens, out_dim, K).
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)
#define CUBLAS_OK(x) do { cublasStatus_t st = (x); if (st != CUBLAS_STATUS_SUCCESS) { return st; } } while (0)

struct shape_t { const char * name; int Nout; int Nt; int K; };

static shape_t g_shapes[] = {
    { "ffn_gate/up ub512",   17408,  512,  5120 },
    { "ffn_gate/up ub2048",  17408, 2048,  5120 },
    { "ffn_down ub512",       5120,  512, 17408 },
    { "attn_qkv/ssm ub512",  10240,  512,  5120 },
    { "lm_head ub512",      248320,  512,  5120 },
    { "ref 4096^3",           4096, 4096,  4096 },
    { "attn_q/ssm_gate",      6144,  512,  5120 },
    { "attn_o/ssm_out",       5120,  512,  6144 },
    { "attn_kv",              1024,  512,  5120 },
};
static const int N_SHAPES = sizeof(g_shapes)/sizeof(g_shapes[0]);

static float bench_gemmex(cublasHandle_t h, cublasGemmAlgo_t algo, const half * A, const half * B,
                          void * C, cudaDataType_t ctype, cublasComputeType_t ct,
                          const void * alpha, const void * beta, int m, int n, int k,
                          int iters, cublasStatus_t * st_out) {
    cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
        alpha, A, CUDA_R_16F, k, B, CUDA_R_16F, k, beta, C, ctype, m, ct, algo);
    if (st_out) { *st_out = st; }
    if (st != CUBLAS_STATUS_SUCCESS) return -1.f;
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++) {
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
            alpha, A, CUDA_R_16F, k, B, CUDA_R_16F, k, beta, C, ctype, m, ct, algo);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / iters;
}

int main(int argc, char ** argv) {
    int only_shape = argc > 1 ? atoi(argv[1]) : -1;
    const bool ws_sweep_mode = argc > 1 && std::string(argv[1]) == "ws";
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);
    if (ws_sweep_mode) {
        // find minimum workspace that unlocks the better algo (shapes: down, attn_kv, attn_o, attn_q)
        void * wsbuf = nullptr; CHECK(cudaMalloc(&wsbuf, 256u*1024*1024));
        const int idx[4] = {2, 8, 7, 6};
        for (int ii = 0; ii < 4; ii++) {
            const shape_t & s = g_shapes[idx[ii]];
            const int m = s.Nout, n = s.Nt, k = s.K;
            const double flop = 2.0*m*n*(double)k;
            half * A, * B; float * C;
            CHECK(cudaMalloc(&A, (size_t)k*m*2)); CHECK(cudaMalloc(&B, (size_t)k*n*2)); CHECK(cudaMalloc(&C, (size_t)m*n*4));
            CHECK(cudaMemset(A,0,(size_t)k*m*2)); CHECK(cudaMemset(B,0,(size_t)k*n*2));
            printf("%s : tokens=%d out=%d K=%d\n", s.name, n, m, k);
            const int ws_mb[8] = {0, 4, 8, 16, 32, 64, 128, 256};
            for (int w = 0; w < 8; w++) {
                cublasSetWorkspace(h, ws_mb[w] ? wsbuf : nullptr, (size_t)ws_mb[w]*1024*1024);
                float a = 1.f, b = 0.f; cublasStatus_t st;
                float ms = bench_gemmex(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, A, B, C, CUDA_R_32F, CUBLAS_COMPUTE_32F,
                                        &a, &b, m, n, k, 20, &st);
                printf("   ws=%4d MB : %8.3f ms  %7.1f TFLOPS\n", ws_mb[w], ms, flop/ms/1e9);
            }
            cublasSetWorkspace(h, nullptr, 0);
            CHECK(cudaFree(A)); CHECK(cudaFree(B)); CHECK(cudaFree(C));
        }
        cudaFree(wsbuf); cublasDestroy(h); return 0;
    }


    int cublas_ver = 0; cublasGetVersion(h, &cublas_ver);
    int rt_ver = 0; cudaRuntimeGetVersion(&rt_ver);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("cuBLAS version = %d, CUDA runtime = %d, device = %s (cc %d.%d, %d SMs, %.0f MHz)\n",
           cublas_ver, rt_ver, prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.clockRate/1000.0);
    printf("NOTE: math view is (tokens, out_dim, K); cuBLAS call is m=out_dim, n=tokens, k=K\n\n");

    const size_t WS_BYTES = 256u*1024*1024;
    void * ws = nullptr; CHECK(cudaMalloc(&ws, WS_BYTES));

    for (int si = 0; si < N_SHAPES; si++) {
        if (only_shape >= 0 && si != only_shape) continue;
        const shape_t & s = g_shapes[si];
        const int m = s.Nout, n = s.Nt, k = s.K;
        const double flop = 2.0*m*n*(double)k;

        half * A = nullptr; half * B = nullptr; float * Cf = nullptr; half * Ch = nullptr;
        CHECK(cudaMalloc(&A, (size_t)k*m*2));
        CHECK(cudaMalloc(&B, (size_t)k*n*2));
        CHECK(cudaMalloc(&Cf, (size_t)m*n*4));
        CHECK(cudaMalloc(&Ch, (size_t)m*n*2));
        CHECK(cudaMemset(A, 0, (size_t)k*m*2));
        CHECK(cudaMemset(B, 0, (size_t)k*n*2));

        const int iters = (double)flop > 2e11 ? 10 : 30;
        printf("================================================================================\n");
        printf("%s : tokens=%d out_dim=%d K=%d  (%.3f GFLOP)\n", s.name, n, m, k, flop/1e9);

        // --- 1) mirror of llama.cpp default on Volta: C=fp32, compute=32F, DEFAULT_TENSOR_OP, no workspace set
        {
            float a = 1.f, b = 0.f;
            cublasStatus_t st;
            float ms = bench_gemmex(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, A, B, Cf, CUDA_R_32F, CUBLAS_COMPUTE_32F,
                                    &a, &b, m, n, k, iters, &st);
            printf("  gemmEx  C=fp32 compute=32F DEFAULT_TO   : %8.3f ms  %7.1f TFLOPS\n", ms, flop/ms/1e9);
            // with explicit workspace
            cublasSetWorkspace(h, ws, WS_BYTES);
            ms = bench_gemmex(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, A, B, Cf, CUDA_R_32F, CUBLAS_COMPUTE_32F,
                              &a, &b, m, n, k, iters, &st);
            printf("  gemmEx  C=fp32 compute=32F +ws256MB    : %8.3f ms  %7.1f TFLOPS\n", ms, flop/ms/1e9);
            cublasSetWorkspace(h, nullptr, 0);
        }
        // --- 2) C=fp16, compute=16F (numerics change, ceiling reference only)
        {
            __half a = __float2half(1.f), b = __float2half(0.f);
            cublasStatus_t st;
            float ms = bench_gemmex(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, A, B, Ch, CUDA_R_16F, CUBLAS_COMPUTE_16F,
                                    &a, &b, m, n, k, iters, &st);
            printf("  gemmEx  C=fp16 compute=16F DEFAULT_TO   : %8.3f ms  %7.1f TFLOPS\n", ms, flop/ms/1e9);
        }
        // --- 3) fp32 accumulate with reduced precision allowed (32F_FAST_16F) - ceiling reference
        {
            float a = 1.f, b = 0.f;
            cublasStatus_t st;
            float ms = bench_gemmex(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, A, B, Cf, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_16F,
                                    &a, &b, m, n, k, iters, &st);
            printf("  gemmEx  C=fp32 compute=32F_FAST_16F    : %8.3f ms  %7.1f TFLOPS\n", ms, flop/ms/1e9);
        }
        // --- 4) algo sweep (tensor-op algos) with C=fp32/32F
        {
            float a = 1.f, b = 0.f;
            float best_ms = 1e9f; int best_algo = -1;
            for (int algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP; algo <= CUBLAS_GEMM_ALGO15_TENSOR_OP; algo++) {
                cublasStatus_t st;
                float ms = bench_gemmex(h, (cublasGemmAlgo_t)algo, A, B, Cf, CUDA_R_32F, CUBLAS_COMPUTE_32F,
                                        &a, &b, m, n, k, iters, &st);
                if (st != CUBLAS_STATUS_SUCCESS) continue;
                printf("    algo %3d : %8.3f ms  %7.1f TFLOPS\n", algo, ms, flop/ms/1e9);
                if (ms < best_ms) { best_ms = ms; best_algo = algo; }
            }
            printf("  best algo = %d : %7.1f TFLOPS (default=%d)\n", best_algo, flop/best_ms/1e9, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        // --- 5) cublasLt heuristic sweep
        {
            cublasLtHandle_t lt; cublasLtCreate(&lt);
            for (int out_f16 = 0; out_f16 < 2; out_f16++) {
                cudaDataType_t ctype = out_f16 ? CUDA_R_16F : CUDA_R_32F;
                cublasComputeType_t ct = out_f16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
                cublasLtMatmulDesc_t op; cublasLtMatmulDescCreate(&op, ct, ctype);
                cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
                cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
                cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
                cublasLtMatrixLayout_t Ad, Bd, Cd;
                cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16F, k, m, k);   // col-major, opA=T -> [k, m]
                cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16F, k, n, k);
                cublasLtMatrixLayoutCreate(&Cd, ctype,     m, n, m);
                cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
                cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS_BYTES, sizeof(WS_BYTES));
                cublasLtMatmulHeuristicResult_t res[16]; int nres = 0;
                cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(lt, op, Ad, Bd, Cd, Cd, pref, 16, res, &nres);
                printf("  cublasLt C=%s heuristic returned %d algos (st=%d):\n", out_f16?"fp16":"fp32", nres, (int)hst);
                if (hst == CUBLAS_STATUS_SUCCESS) {
                    for (int i = 0; i < nres; i++) {
                        int algo_id = -1, tile = -1, splitK = -1, stages = -1, swizzle = -1;
                        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), nullptr);
                        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr);
                        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK, sizeof(splitK), nullptr);
                        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), nullptr);
                        cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), nullptr);
                        const void * ap = out_f16 ? (const void*)nullptr : (const void*)nullptr;
                        (void)ap;
                        float ms;
                        if (out_f16) { __half a = __float2half(1.f), b = __float2half(0.f);
                            cublasLtMatmul(lt, op, &a, A, Ad, B, Bd, &b, Ch, Cd, Ch, Cd, &res[i].algo, ws, WS_BYTES, 0);
                            CHECK(cudaDeviceSynchronize());
                            cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
                            for (int it = 0; it < iters; it++)
                                cublasLtMatmul(lt, op, &a, A, Ad, B, Bd, &b, Ch, Cd, Ch, Cd, &res[i].algo, ws, WS_BYTES, 0);
                            cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1); ms/=iters;
                        } else { float a = 1.f, b = 0.f;
                            cublasLtMatmul(lt, op, &a, A, Ad, B, Bd, &b, Cf, Cd, Cf, Cd, &res[i].algo, ws, WS_BYTES, 0);
                            CHECK(cudaDeviceSynchronize());
                            cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
                            for (int it = 0; it < iters; it++)
                                cublasLtMatmul(lt, op, &a, A, Ad, B, Bd, &b, Cf, Cd, Cf, Cd, &res[i].algo, ws, WS_BYTES, 0);
                            cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms,e0,e1); ms/=iters;
                        }
                        printf("    #%2d algo=%3d tile=%2d splitK=%2d stages=%d swizzle=%d : %8.3f ms %7.1f TFLOPS\n",
                               i, algo_id, tile, splitK, stages, swizzle, ms, flop/ms/1e9);
                    }
                }
                cublasLtMatmulPreferenceDestroy(pref);
                cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
                cublasLtMatmulDescDestroy(op);
            }
            cublasLtDestroy(lt);
        }
        CHECK(cudaFree(A)); CHECK(cudaFree(B)); CHECK(cudaFree(Cf)); CHECK(cudaFree(Ch));
        printf("\n");
    }
    cudaFree(ws);
    cublasDestroy(h);
    printf("done\n");
    return 0;
}
