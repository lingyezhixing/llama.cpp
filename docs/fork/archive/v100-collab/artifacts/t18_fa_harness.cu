// T18 Stage 0: standalone harness for the sm70 FA mma kernel flash_attn_ext_f16<256,256,32,2>
// Real shape (ub512, depth32k): n_q=512, n_kv=35072, GQA 24/4, head_dim 256.
// Configs: grid=192 (T11-off control) and grid=384 (production: PB=2 + uniform fixup).
// The kernel is included from the ggml-cuda tree so Stage 1 source edits are picked up directly.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "fattn-mma-f16.cuh"

// standalone stub for the ggml core abort handler (harness only)
#include <cstdarg>
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("ggml_abort: %s:%d: ", file, line);
    vprintf(fmt, ap);
    printf("\n");
    va_end(ap);
    exit(1);
}

#define CUDA_CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while (0)

#ifndef T18_NCOLS1
#define T18_NCOLS1 32
#endif
static constexpr int DKQ = 256, DV = 256, NCOLS1 = T18_NCOLS1, NCOLS2 = 2, NCOLS = NCOLS1*NCOLS2;
static constexpr int NQ = 512, NHEAD_Q = 24, NHEAD_KV = 4, GQA = NHEAD_Q/NHEAD_KV;
static int g_nkv = 35072;   // n_kv, set from argv[2]
static constexpr int NTILES_X = (NQ + NCOLS1 - 1)/NCOLS1;            // 16
static constexpr int NTILES_Z_GQA = (GQA + NCOLS2 - 1)/NCOLS2;       // 3
static constexpr int NTILES_DST = NTILES_X*NTILES_Z_GQA*NHEAD_KV;    // 192

struct Ctx {
    float * q, * o, * o_ref;
    half  * k, * v, * m;
    float2 * fixup;
    int nwarps, nthreads;
    size_t smem;
    uint3 ne01_fd;
    int32_t nb01, nb02; int64_t nb03;
    int32_t nb11, nb12; int64_t nb13;
    int32_t nb21, nb22; int64_t nb23;
    int32_t nb31, nb32; int64_t nb33;
    uint32_t n_head_log2;
    float scale;
};

static void launch_fa(const Ctx & c, int nblocks) {
    dim3 grid(nblocks, 1, 1);
    dim3 block(32, c.nwarps, 1);
    flash_attn_ext_f16<DKQ, DV, NCOLS1, NCOLS2, false, false, false><<<grid, block, c.smem>>>(
        (const char *) c.q, (const char *) c.k, (const char *) c.v, (const char *) c.m, nullptr, nullptr,
        c.o, c.fixup,
        c.scale, 0.0f, 1.0f, 1.0f, c.n_head_log2, 0.0f,
        DKQ, c.ne01_fd, NHEAD_Q, 1, c.nb01, c.nb02, c.nb03,
        DKQ, g_nkv, NHEAD_KV, 1, c.nb11, c.nb12, c.nb13,
        c.nb21, c.nb22, c.nb23,
        NQ, 1, 1, c.nb31, c.nb32, c.nb33);
}

static void launch_fixup_uniform(const Ctx & c, int nblocks) {
    const int bpt = nblocks / NTILES_DST;
    const uint3 fd0 = init_fastdiv_values(NTILES_X*NTILES_Z_GQA*NHEAD_KV);
    const uint3 fd1 = init_fastdiv_values(NTILES_X*NTILES_Z_GQA);
    const uint3 fd2 = init_fastdiv_values(NTILES_X);
    dim3 grid(NTILES_DST, NCOLS1, NCOLS2);
    flash_attn_stream_k_fixup_uniform<DV, NCOLS1, NCOLS2><<<grid, dim3(DV, 1, 1)>>>(
        c.o, c.fixup, NQ, NHEAD_Q, NHEAD_KV, nblocks, GQA, bpt, fd0, fd1, fd2);
}

static double bench(const Ctx & c, int nblocks, bool fixup, int iters) {
    launch_fa(c, nblocks);
    if (fixup) launch_fixup_uniform(c, nblocks);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) {
        launch_fa(c, nblocks);
        if (fixup) launch_fixup_uniform(c, nblocks);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    CUDA_CHECK(cudaGetLastError());
    return ms/iters;
}

int main(int argc, char ** argv) {
    cudaSetDevice(0);
    const int iters = argc > 1 ? atoi(argv[1]) : 30;
    g_nkv = argc > 2 ? atoi(argv[2]) : 35072;
    Ctx c;
    memset(&c, 0, sizeof(c));

    c.nthreads = ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, NCOLS, GGML_CUDA_CC_VOLTA);
    const int nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, NCOLS, GGML_CUDA_CC_VOLTA);
    const int nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2(DKQ, DV, NCOLS, GGML_CUDA_CC_VOLTA);
    const int nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2(DKQ, DV, NCOLS, GGML_CUDA_CC_VOLTA);
    const int nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, NCOLS, GGML_CUDA_CC_VOLTA);
    const bool Q_in_reg      = ggml_cuda_fattn_mma_get_Q_in_reg(DKQ, DV, NCOLS, GGML_CUDA_CC_VOLTA);
    c.nwarps = c.nthreads/32;

    // smem: same formula as ggml_cuda_flash_attn_ext_mma_f16_case (no cp.async -> 1 stage)
    const int stride_tile_K = ggml_cuda_fattn_mma_get_stride_tile(nbatch_K2, false);
    const int stride_tile_V = ggml_cuda_fattn_mma_get_stride_tile(nbatch_V2, false);
    const size_t nbytes_KV_1 = (size_t) nbatch_fa * std::max(stride_tile_K, stride_tile_V) * sizeof(half2);
    const size_t nbytes_Q    = (size_t) NCOLS * (DKQ/2 + 4) * sizeof(half2);
    const size_t nbytes_mask = (size_t) NCOLS1 * (nbatch_fa/2 + 4) * sizeof(half2);
    const size_t nbytes_comb = (size_t) c.nwarps * get_cols_per_warp(GGML_CUDA_CC_VOLTA) * (nbatch_combine + 4) * sizeof(half2);
    c.smem = Q_in_reg ? std::max(nbytes_comb, std::max(nbytes_Q, nbytes_KV_1 + nbytes_mask))
                      : std::max(nbytes_comb, nbytes_Q + nbytes_KV_1 + nbytes_mask);
    printf("config: nthreads=%d nwarps=%d nbatch_fa=%d K2=%d V2=%d combine=%d Q_in_reg=%d\n",
           c.nthreads, c.nwarps, nbatch_fa, nbatch_K2, nbatch_V2, nbatch_combine, (int) Q_in_reg);
    printf("smem = %zu B (comb=%zu Q=%zu KV=%zu mask=%zu)\n", c.smem, nbytes_comb, nbytes_Q, nbytes_KV_1, nbytes_mask);

    CUDA_CHECK(cudaFuncSetAttribute((const void *) flash_attn_ext_f16<DKQ, DV, NCOLS1, NCOLS2, false, false, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, (int) c.smem));

    // strides (f32 Q, f16 K/V/mask), mask layout [n_kv, n_q]
    c.nb01 = DKQ*sizeof(float); c.nb02 = DKQ*NQ*sizeof(float); c.nb03 = (int64_t) DKQ*NQ*NHEAD_Q*sizeof(float);
    c.nb11 = DKQ*sizeof(half);  c.nb12 = DKQ*g_nkv*sizeof(half); c.nb13 = (int64_t) DKQ*g_nkv*NHEAD_KV*sizeof(half);
    c.nb21 = c.nb11; c.nb22 = c.nb12; c.nb23 = c.nb13;
    c.nb31 = g_nkv*sizeof(half);  c.nb32 = (int32_t)((int64_t) g_nkv*NQ*sizeof(half)); c.nb33 = (int64_t) g_nkv*NQ*sizeof(half);
    c.ne01_fd = init_fastdiv_values(NQ);
    c.n_head_log2 = 1u << (uint32_t) floorf(log2f((float) NHEAD_Q));   // 16
    c.scale = 1.0f / sqrtf((float) DKQ);

    const size_t q_sz = (size_t) DKQ*NQ*NHEAD_Q*sizeof(float);
    const size_t k_sz = (size_t) DKQ*g_nkv*NHEAD_KV*sizeof(half);
    const size_t m_sz = (size_t) g_nkv*NQ*sizeof(half);
    const size_t fx_sz = (size_t) 4*NTILES_DST * NCOLS * (2 + DV/2);   // enough for grid up to 4*NTILES_DST (PB=4)
    CUDA_CHECK(cudaMalloc(&c.q, q_sz)); CUDA_CHECK(cudaMalloc(&c.o, q_sz)); CUDA_CHECK(cudaMalloc(&c.o_ref, q_sz));
    CUDA_CHECK(cudaMalloc(&c.k, k_sz)); CUDA_CHECK(cudaMalloc(&c.v, k_sz)); CUDA_CHECK(cudaMalloc(&c.m, m_sz));
    CUDA_CHECK(cudaMalloc(&c.fixup, fx_sz*sizeof(float2)));

    {
        srand(42);
        std::vector<float> hq((size_t) DKQ*NQ*NHEAD_Q);
        for (auto & x : hq) x = (rand()/(float)RAND_MAX - 0.5f);
        CUDA_CHECK(cudaMemcpy(c.q, hq.data(), q_sz, cudaMemcpyHostToDevice));
        std::vector<half> hk((size_t) DKQ*g_nkv*NHEAD_KV);
        for (auto & x : hk) x = __float2half((rand()/(float)RAND_MAX - 0.5f));
        CUDA_CHECK(cudaMemcpy(c.k, hk.data(), k_sz, cudaMemcpyHostToDevice));
        std::vector<half> hv(hk.size());
        for (auto & x : hv) x = __float2half((rand()/(float)RAND_MAX - 0.5f));
        CUDA_CHECK(cudaMemcpy(c.v, hv.data(), k_sz, cudaMemcpyHostToDevice));
        std::vector<half> hm((size_t) g_nkv*NQ);
        for (int j = 0; j < NQ; ++j)
            for (int i = 0; i < g_nkv; ++i)
                hm[(size_t) j*g_nkv + i] = (i <= (g_nkv - NQ) + j) ? __float2half(0.0f) : __float2half(-INFINITY);
        CUDA_CHECK(cudaMemcpy(c.m, hm.data(), m_sz, cudaMemcpyHostToDevice));
    }

    printf("NQ=%d g_nkv=%d ntiles_dst=%d fixup buf=%.1f MB\n", NQ, g_nkv, NTILES_DST, fx_sz*sizeof(float2)/1e6);

    // correctness: production config (grid=384 + fixup) vs control (grid=192)
    const int nb_prod = argc > 3 ? atoi(argv[3]) : 2*NTILES_DST;
    launch_fa(c, NTILES_DST); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.o_ref, c.o, q_sz, cudaMemcpyDeviceToDevice));
    launch_fa(c, nb_prod); launch_fixup_uniform(c, nb_prod); CUDA_CHECK(cudaDeviceSynchronize());
    {
        std::vector<float> a(q_sz/sizeof(float)), b(q_sz/sizeof(float));
        CUDA_CHECK(cudaMemcpy(a.data(), c.o_ref, q_sz, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), c.o,     q_sz, cudaMemcpyDeviceToHost));
        double max_abs = 0.0, max_rel = 0.0, sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = fabs((double)a[i] - (double)b[i]);
            const double r = d/(fabs((double)a[i]) + 1e-6);
            if (d > max_abs) max_abs = d;
            if (r > max_rel) max_rel = r;
            sum += d;
        }
        printf("grid384+fixup vs grid192: max_abs=%.3e max_rel=%.3e mean_abs=%.3e\n", max_abs, max_rel, sum/a.size());
    }

    const int nb_ctl = argc > 4 ? atoi(argv[4]) : NTILES_DST;
    const double t_ctl  = bench(c, nb_ctl,  nb_ctl  % NTILES_DST != 0, iters);
    const double t_prod = bench(c, nb_prod, nb_prod % NTILES_DST != 0, iters);
    printf("grid=%d (control):    %8.3f ms/launch\n", nb_ctl,  t_ctl);
    printf("grid=%d (prod PB=2):  %8.3f ms/launch\n", nb_prod, t_prod);
    printf("ratio prod/ctl = %.4f\n", t_prod/t_ctl);
    return 0;
}

