// T08 Step 1: authoritative ABAB arbitration of gemmEx DEFAULT_TENSOR_OP vs cublasLt heuristic.
// Orientation = llama.cpp exact on Volta:
//   cublasGemmEx(OP_T, OP_N, m=out_dim, n=tokens, k, alpha, A=W f16 lda=k, B=X f16 ldb=k,
//                beta, C f32 ldc=m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
// Variants measured interleaved (5 rounds, rotated order):
//   def_tf32   : math=TF32_TENSOR_OP + 4MB workspace on handle  (what llama.cpp/common sets)
//   def_tensor : math=TENSOR_OP,  no workspace                  (what the T02 Gate A harness set)
//   def_dflt   : math=DEFAULT,    no workspace                  (what the T01 bench harness set)
//   lt_best    : cublasLt heuristic candidates (workspace passed per call), correctness checked
// Plus a transposed-orientation diagnostic for gate/up (T02 harness called m=tokens/n=out).
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %d %s\n", __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

struct shape_t { const char * name; int m; int n; int k; double model_ms; };

static const double MODEL_TOTAL_MS = 507.8;

static shape_t g_shapes[] = {
    { "ffn_gate/up   (17408,512,5120)", 17408, 512,  5120, 148.99 },
    { "ffn_down      ( 5120,512,17408)",  5120, 512, 17408,  63.40 },
    { "ssm_inproj    (10240,512,5120)", 10240, 512,  5120,  29.23 },
    { "ssm_out/o_proj( 5120,512, 6144)",  5120, 512,  6144,  23.00 },
    { "ssm_qkv       ( 6144,512,5120)",  6144, 512,  5120,  22.71 },
    { "attn_qkv      (12288,512,5120)", 12288, 512,  5120,  13.07 },
    { "attn_k/v      ( 1024,512,5120)",  1024, 512,  5120,   3.01 },
    { "ssm_ba        (   48,512,5120)",    48, 512,  5120,   1.97 },
};
static const int N_SHAPES = sizeof(g_shapes)/sizeof(g_shapes[0]);
static const int ROUNDS = 5;
static const size_t WS_BYTES = 4u*1024*1024;

// ---- gemmEx with a given handle state -------------------------------------------------
static double ms_gemmex(cublasHandle_t h, cublasGemmAlgo_t algo, const half * A, const half * B, float * C,
                        int m, int n, int k, int iters) {
    float a = 1.f, b = 0.f;
    for (int i = 0; i < 3; i++)
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a, A, CUDA_R_16F, k, B, CUDA_R_16F, k, &b, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, algo);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++)
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a, A, CUDA_R_16F, k, B, CUDA_R_16F, k, &b, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, algo);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / iters;
}

struct lt_ctx {
    cublasLtMatmulDesc_t opd = nullptr;
    cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr;
};

static double ms_lt(cublasLtHandle_t lt, const lt_ctx & c, const half * A, const half * B, float * C,
                    void * ws, const cublasLtMatmulAlgo_t * algo, int iters) {
    float a = 1.f, b = 0.f;
    for (int i = 0; i < 3; i++)
        cublasLtMatmul(lt, c.opd, &a, A, c.Ad, B, c.Bd, &b, C, c.Cd, C, c.Cd, algo, ws, WS_BYTES, 0);
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; i++)
        cublasLtMatmul(lt, c.opd, &a, A, c.Ad, B, c.Bd, &b, C, c.Cd, C, c.Cd, algo, ws, WS_BYTES, 0);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms / iters;
}

static double nmse_vs(const float * C, const float * Cref, size_t n_el) {
    std::vector<float> a(n_el), b(n_el);
    CHECK(cudaMemcpy(a.data(), C,    n_el*4, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(b.data(), Cref, n_el*4, cudaMemcpyDeviceToHost));
    double sd = 0, sr = 0;
    for (size_t i = 0; i < n_el; i++) { double d = (double)a[i]-b[i]; sd += d*d; sr += (double)b[i]*b[i]; }
    return sr > 0 ? sd/sr : -1.0;
}

int main() {
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);
    void * ws = nullptr; CHECK(cudaMalloc(&ws, WS_BYTES));
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    int rt = 0; cudaRuntimeGetVersion(&rt);
    printf("T08 Step 1: llama.cpp-exact orientation (m=out_dim, n=tokens), CUDA runtime %d\n", rt);
    printf("variants: def_tf32 (llama.cpp) | def_tensor (T02 harness) | def_dflt (T01 harness) | lt_best\n");
    printf("%d interleaved rounds, median reported\n\n", ROUNDS);

    double w_lt = 0.0, w_tensor = 0.0, w_dflt = 0.0;
    for (int si = 0; si < N_SHAPES; si++) {
        const shape_t & s = g_shapes[si];
        const int m = s.m, n = s.n, k = s.k;
        const double flop = 2.0*m*n*(double)k;
        const int iters = flop > 5e10 ? 30 : 100;
        half * A, * B; float * C, * Cref;
        CHECK(cudaMalloc(&A, (size_t)k*m*2));
        CHECK(cudaMalloc(&B, (size_t)k*n*2));
        CHECK(cudaMalloc(&C, (size_t)m*n*4));
        CHECK(cudaMalloc(&Cref, (size_t)m*n*4));
        {
            std::vector<half> a((size_t)k*m), b((size_t)k*n);
            for (size_t i = 0; i < a.size(); i++) a[i] = __float2half((float)((int)(i % 21) - 10) * 0.25f);
            for (size_t i = 0; i < b.size(); i++) b[i] = __float2half((float)((int)(i % 13) - 6)  * 0.125f);
            CHECK(cudaMemcpy(A, a.data(), a.size()*2, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(B, b.data(), b.size()*2, cudaMemcpyHostToDevice));
        }
        // reference with llama.cpp-exact state
        cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
        cublasSetWorkspace(h, ws, WS_BYTES);
        {
            float a = 1.f, b = 0.f;
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a, A, CUDA_R_16F, k, B, CUDA_R_16F, k, &b, Cref, CUDA_R_32F, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            CHECK(cudaDeviceSynchronize());
        }

        printf("=== %s m=%d n=%d k=%d flop=%.1f GF model_ms=%.2f\n", s.name, m, n, k, flop/1e9, s.model_ms);

        // Lt descriptors + heuristic scan with validity checks
        lt_ctx lc;
        cublasLtMatmulDescCreate(&lc.opd, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(lc.opd, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
        cublasLtMatmulDescSetAttribute(lc.opd, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
        cublasLtMatrixLayoutCreate(&lc.Ad, CUDA_R_16F, k, m, k);
        cublasLtMatrixLayoutCreate(&lc.Bd, CUDA_R_16F, k, n, k);
        cublasLtMatrixLayoutCreate(&lc.Cd, CUDA_R_32F, m, n, m);

        std::vector<cublasLtMatmulAlgo_t> lt_cand;
        std::vector<int> lt_id, lt_tile, lt_split;
        std::vector<double> lt_ms;
        {
            cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS_BYTES, sizeof(WS_BYTES));
            cublasLtMatmulHeuristicResult_t res[32]; int nres = 0;
            cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(lt, lc.opd, lc.Ad, lc.Bd, lc.Cd, lc.Cd, pref, 32, res, &nres);
            if (hst == CUBLAS_STATUS_SUCCESS) {
                for (int i = 0; i < nres; i++) {
                    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
                    int id=-1, tile=-1, sp=-1, stg=-1;
                    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_ID, &id, sizeof(id), nullptr);
                    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile), nullptr);
                    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sp, sizeof(sp), nullptr);
                    cublasLtMatmulAlgoConfigGetAttribute(&res[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stg, sizeof(stg), nullptr);
                    // validity: single call + NMSE check
                    CHECK(cudaMemset(C, 0, (size_t)m*n*4));
                    float a = 1.f, b = 0.f;
                    cublasStatus_t st = cublasLtMatmul(lt, lc.opd, &a, A, lc.Ad, B, lc.Bd, &b, C, lc.Cd, C, lc.Cd, &res[i].algo, ws, WS_BYTES, 0);
                    CHECK(cudaDeviceSynchronize());
                    double nm = (st == CUBLAS_STATUS_SUCCESS) ? nmse_vs(C, Cref, (size_t)m*n) : -1.0;
                    double t = ms_lt(lt, lc, A, B, C, ws, &res[i].algo, 10);
                    const bool ok = (nm >= 0.0 && nm < 1e-6) && t > 0.01;
                    printf("    lt#%d algo=%4d tile=%2d splitK=%2d stages=%2d : %8.3f ms %6.1f TF  NMSE=%.2e %s\n",
                           i, id, tile, sp, stg, t, flop/t/1e9, nm, ok ? "" : "<-- INVALID, skipped");
                    if (!ok) continue;
                    lt_cand.push_back(res[i].algo); lt_id.push_back(id); lt_tile.push_back(tile); lt_split.push_back(sp);
                    lt_ms.push_back(t);
                }
            }
            cublasLtMatmulPreferenceDestroy(pref);
        }
        int lt_best = -1;
        for (size_t i = 0; i < lt_ms.size(); i++) if (lt_best < 0 || lt_ms[i] < lt_ms[lt_best]) lt_best = (int)i;
        if (lt_best >= 0)
            printf("    -> lt best: algo=%d tile=%d splitK=%d  %.3f ms (%.1f TF)\n",
                   lt_id[lt_best], lt_tile[lt_best], lt_split[lt_best], lt_ms[lt_best], flop/lt_ms[lt_best]/1e9);

        // interleaved rounds over 4 variants, rotated order
        std::vector<double> r_tf32, r_tensor, r_dflt, r_lt;
        for (int r = 0; r < ROUNDS; r++) {
            double t_tf32 = 0, t_tensor = 0, t_dflt = 0, t_lt = 0;
            for (int slot = 0; slot < 4; slot++) {
                const int v = (slot + r) % 4;
                if (v == 0) { cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH); cublasSetWorkspace(h, ws, WS_BYTES);
                              t_tf32 = ms_gemmex(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, A, B, C, m, n, k, iters); }
                if (v == 1) { cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);      cublasSetWorkspace(h, nullptr, 0);
                              t_tensor = ms_gemmex(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, A, B, C, m, n, k, iters); }
                if (v == 2) { cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);        cublasSetWorkspace(h, nullptr, 0);
                              t_dflt = ms_gemmex(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, A, B, C, m, n, k, iters); }
                if (v == 3) { t_lt = lt_best >= 0 ? ms_lt(lt, lc, A, B, C, ws, &lt_cand[lt_best], iters) : 0; }
            }
            r_tf32.push_back(t_tf32); r_tensor.push_back(t_tensor); r_dflt.push_back(t_dflt); r_lt.push_back(t_lt);
            printf("    r%d: tf32=%7.3f (%5.1f)  tensor=%7.3f (%5.1f)  dflt=%7.3f (%5.1f)  lt=%7.3f (%5.1f)\n",
                   r, t_tf32, flop/t_tf32/1e9, t_tensor, flop/t_tensor/1e9, t_dflt, flop/t_dflt/1e9,
                   t_lt, (t_lt > 0 ? flop/t_lt/1e9 : 0.0));
        }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; };
        const double md_tf32 = med(r_tf32), md_tensor = med(r_tensor), md_dflt = med(r_dflt), md_lt = med(r_lt);
        const double g_lt = (md_tf32/md_lt - 1.0)*100.0;
        const double g_tensor = (md_tf32/md_tensor - 1.0)*100.0;
        const double g_dflt = (md_tf32/md_dflt - 1.0)*100.0;
        printf("  MEDIAN tf32=%.3fms(%.1fTF)  tensor=%.3fms(%.1fTF, vs tf32 %+.1f%%)  dflt=%.3fms(%.1fTF, %+.1f%%)  lt=%.3fms(%.1fTF, %+.1f%%)\n\n",
               md_tf32, flop/md_tf32/1e9, md_tensor, flop/md_tensor/1e9, g_tensor,
               md_dflt, flop/md_dflt/1e9, g_dflt, md_lt, flop/md_lt/1e9, g_lt);
        w_lt += s.model_ms * g_lt; w_tensor += s.model_ms * g_tensor; w_dflt += s.model_ms * g_dflt;

        CHECK(cudaFree(A)); CHECK(cudaFree(B)); CHECK(cudaFree(C)); CHECK(cudaFree(Cref));
        cublasLtMatrixLayoutDestroy(lc.Ad); cublasLtMatrixLayoutDestroy(lc.Bd); cublasLtMatrixLayoutDestroy(lc.Cd);
        cublasLtMatmulDescDestroy(lc.opd);
    }

    // ---- transposed-orientation diagnostic (what the T02 Gate A harness measured) ----
    {
        const int m = 17408, n = 512, k = 5120;
        const double flop = 2.0*m*n*(double)k;
        half * W, * Xt; float * C, * Cref;
        CHECK(cudaMalloc(&W,  (size_t)k*m*2));
        CHECK(cudaMalloc(&Xt, (size_t)k*n*2));
        CHECK(cudaMalloc(&C,  (size_t)m*n*4));
        CHECK(cudaMalloc(&Cref, (size_t)m*n*4));
        CHECK(cudaMemset(W, 1, (size_t)k*m*2));
        CHECK(cudaMemset(Xt, 1, (size_t)k*n*2));
        printf("=== DIAGNOSTIC: transposed orientation (T02 harness): GemmEx(m=512 tokens, n=17408 out, A=X, B=W)\n");
        cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH); cublasSetWorkspace(h, nullptr, 0);
        double t_def = ms_gemmex(h, CUBLAS_GEMM_DEFAULT_TENSOR_OP, Xt, W, Cref, n, m, k, 30);
        printf("    gemmEx default (T02 harness state) : %.3f ms  %.1f TF\n", t_def, flop/t_def/1e9);
        double best_t = 1e9; int best_a = -1;
        for (int a = CUBLAS_GEMM_ALGO8_TENSOR_OP; a <= CUBLAS_GEMM_ALGO15_TENSOR_OP; a++) {
            double t = ms_gemmex(h, (cublasGemmAlgo_t)a, Xt, W, C, n, m, k, 30);
            if (t < best_t) { best_t = t; best_a = a; }
        }
        printf("    gemmEx ALGO8..15 best (algo %d)   : %.3f ms  %.1f TF\n", best_a, best_t, flop/best_t/1e9);
        // Lt in the same transposed orientation
        {
            cublasLtMatmulDesc_t opd; cublasLtMatmulDescCreate(&opd, CUBLAS_COMPUTE_32F, CUDA_R_32F);
            cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(opd, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
            cublasLtMatmulDescSetAttribute(opd, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
            cublasLtMatrixLayout_t Ad, Bd, Cd;
            cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16F, k, n, k);
            cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16F, k, m, k);
            cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, n, m, n);
            cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
            cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS_BYTES, sizeof(WS_BYTES));
            cublasLtMatmulHeuristicResult_t res[32]; int nres = 0;
            if (cublasLtMatmulAlgoGetHeuristic(lt, opd, Ad, Bd, Cd, Cd, pref, 32, res, &nres) == CUBLAS_STATUS_SUCCESS) {
                double bl = 1e9; int bi = -1;
                for (int i = 0; i < nres; i++) {
                    if (res[i].state != CUBLAS_STATUS_SUCCESS) continue;
                    CHECK(cudaMemset(C, 0, (size_t)m*n*4));
                    float a = 1.f, b = 0.f;
                    cublasStatus_t st = cublasLtMatmul(lt, opd, &a, Xt, Ad, W, Bd, &b, C, Cd, C, Cd, &res[i].algo, ws, WS_BYTES, 0);
                    CHECK(cudaDeviceSynchronize());
                    double nm = (st == CUBLAS_STATUS_SUCCESS) ? nmse_vs(C, Cref, (size_t)m*n) : -1.0;
                    double t = ms_lt(lt, *(new lt_ctx{opd, Ad, Bd, Cd}), Xt, W, C, ws, &res[i].algo, 10);
                    if (nm >= 0 && nm < 1e-6 && t < bl) { bl = t; bi = i; }
                }
                if (bi >= 0) printf("    cublasLt best (transposed)         : %.3f ms  %.1f TF\n", bl, flop/bl/1e9);
            }
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(Ad); cublasLtMatrixLayoutDestroy(Bd); cublasLtMatrixLayoutDestroy(Cd);
            cublasLtMatmulDescDestroy(opd);
        }
        CHECK(cudaFree(W)); CHECK(cudaFree(Xt)); CHECK(cudaFree(C)); CHECK(cudaFree(Cref));
    }

    printf("\n==================== WEIGHTED SUMMARY ====================\n");
    printf("GEMM (model, pp512) = 305.4 ms of %.1f ms kernel time\n", MODEL_TOTAL_MS);
    printf("llama.cpp-exact (def_tf32) baseline. gains vs it:\n");
    printf("  cublasLt best-valid   : %+.2f %% of GEMM  -> %+.2f %% of prefill kernel time\n", w_lt/305.4, w_lt/MODEL_TOTAL_MS);
    printf("  math=TENSOR_OP,no-ws  : %+.2f %% of GEMM  -> %+.2f %% of prefill kernel time\n", w_tensor/305.4, w_tensor/MODEL_TOTAL_MS);
    printf("  math=DEFAULT,no-ws    : %+.2f %% of GEMM  -> %+.2f %% of prefill kernel time\n", w_dflt/305.4, w_dflt/MODEL_TOTAL_MS);
    cudaFree(ws); cublasLtDestroy(lt); cublasDestroy(h);
    printf("done\n");
    return 0;
}
