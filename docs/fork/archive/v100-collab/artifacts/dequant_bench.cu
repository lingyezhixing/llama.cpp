// Q6_K / Q5_K dequant micro-benchmark: old vs vectorized kernels
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define QK_K 256

typedef struct {
    uint8_t  ql[QK_K/2];
    uint8_t  qh[QK_K/4];
    int8_t   scales[QK_K/16];
    uint16_t d;
} block_q6_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[QK_K/8];
    uint8_t  qs[QK_K/2];
} block_q5_K;

// ---------------------------------------------------------------------------
// Q6_K old kernel (verbatim port of ggml-cuda dequantize_q6_K + convert.cu)
static __device__ __forceinline__ void old_q6_K_body(const void * vx, const int64_t ib, half * yy, const int tid) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t ip  = tid/32;
    const int64_t il  = tid - 32*ip;
    const int64_t is  = 8*ip + il/16;

    half * y = yy + 128*ip + il;

    const float d = __half2float(x[ib].d);

    const uint8_t * ql = x[ib].ql + 64*ip + il;
    const uint8_t   qh = x[ib].qh[32*ip + il];
    const int8_t  * sc = x[ib].scales + is;

    y[ 0] = __float2half(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = __float2half(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = __float2half(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = __float2half(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

static __global__ void k_old_q6_K(const void * __restrict__ vx, half * __restrict__ yy) {
    old_q6_K_body(vx, blockIdx.x, yy + (int64_t)blockIdx.x*QK_K, threadIdx.x);
}

// ---------------------------------------------------------------------------
// Q6_K new kernel: one warp per 256-element block, one lane handles 8 consecutive elements
static __device__ __forceinline__ void new_q6_K_body(const void * vx, const int64_t ib, half * __restrict__ yy, const int lane) {
    const block_q6_K * x = (const block_q6_K *) vx + ib;

    const int64_t e0 = 8*lane;                  // first element, multiple of 8
    const float d = __half2float(x->d) * (float)x->scales[e0 >> 4];

    const uint8_t * ql = x->ql + (e0 & 63) + ((e0 >> 7) << 6);   // 8 contiguous bytes
    const uint8_t * qh = x->qh + (e0 & 31) + ((e0 >> 7) << 5);   // 8 contiguous bytes
    const int  shift = 2 * ((e0 >> 5) & 3);
    const bool hi    = ((e0 >> 6) & 1) != 0;

    // block stride is 210 bytes -> only 2-byte alignment is guaranteed
    const uint16_t qw[4] = {
        *(const uint16_t *)(ql + 0), *(const uint16_t *)(ql + 2),
        *(const uint16_t *)(ql + 4), *(const uint16_t *)(ql + 6),
    };
    const uint16_t hw[4] = {
        *(const uint16_t *)(qh + 0), *(const uint16_t *)(qh + 2),
        *(const uint16_t *)(qh + 4), *(const uint16_t *)(qh + 6),
    };

    uint32_t v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t qb = (i & 1) ? (qw[i/2] >> 8) : (qw[i/2] & 0xFF);
        const uint32_t hb = (i & 1) ? (hw[i/2] >> 8) : (hw[i/2] & 0xFF);
        const uint32_t nib = hi ? ((qb >> 4) & 0xF) : (qb & 0xF);
        const int32_t  q6  = (int32_t)(nib | (((hb >> shift) & 3) << 4)) - 32;
        v[i] = __half_as_ushort(__float2half(d * (float)q6));
    }

    uint4 out;
    out.x = v[0] | (v[1] << 16);
    out.y = v[2] | (v[3] << 16);
    out.z = v[4] | (v[5] << 16);
    out.w = v[6] | (v[7] << 16);
    *(uint4 *)(yy + ib*QK_K + e0) = out;
}

static __global__ void k_new_q6_K(const void * __restrict__ vx, half * __restrict__ yy, const int64_t nb) {
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;
    for (int64_t ib = (int64_t)blockIdx.x*nwarp + warp; ib < nb; ib += (int64_t)gridDim.x*nwarp) {
        new_q6_K_body(vx, ib, yy, lane);
    }
}

// variant: one warp handles 2 consecutive blocks (2x ILP per lane)
static __global__ void k_new_q6_K_x2(const void * __restrict__ vx, half * __restrict__ yy, const int64_t nb) {
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;
    const int64_t nb2 = nb/2;
    for (int64_t i = (int64_t)blockIdx.x*nwarp + warp; i < nb2; i += (int64_t)gridDim.x*nwarp) {
        new_q6_K_body(vx, 2*i + 0, yy, lane);
        new_q6_K_body(vx, 2*i + 1, yy, lane);
    }
}

// ---------------------------------------------------------------------------
// Q5_K old kernel (verbatim port)
static __device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t * __restrict__ q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static __device__ __forceinline__ void old_q5_K_body(const void * vx, const int64_t ib, half * yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t il  = tid/16;
    const int64_t ir  = tid%16;
    const int64_t is  = 2*il;

    half * y = yy + 64*il + 2*ir;

    const float dall = __half2float(x[ib].d);
    const float dmin = __half2float(x[ib].dmin);

    const uint8_t * ql = x[ib].qs + 32*il + 2*ir;
    const uint8_t * qh = x[ib].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t hm = 1 << (2*il);
    y[ 0] = __float2half(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = __float2half(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = __float2half(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = __float2half(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}

static __global__ void k_old_q5_K(const void * __restrict__ vx, half * __restrict__ yy) {
    old_q5_K_body(vx, blockIdx.x, yy + (int64_t)blockIdx.x*QK_K, threadIdx.x);
}

// ---------------------------------------------------------------------------
// Q5_K new kernel: one warp per block, one lane handles 8 consecutive elements
// element e (0..255):
//   low 4 bits: qs[e & 63] nibble ((e>>6)&1 ? high : low)
//   high bit  : qh[e & 31] bit ((e>>5)&3)
//   scale/min : group g = e>>5 (8 groups of 32 values, from 6-bit packed scales/mins)
static __device__ __forceinline__ void new_q5_K_body(const void * vx, const int64_t ib, half * __restrict__ yy, const int lane) {
    const block_q5_K * x = (const block_q5_K *) vx + ib;
    const int64_t e0 = 8*lane;
    const int h = (int)(e0 >> 5);                // 0..7 : group of 32 elements
    const float dall = __half2float(x->d);
    const float dmin = __half2float(x->dmin);

    uint8_t sc, m;
    get_scale_min_k4(h, x->scales, sc, m);
    const float d1 = dall * sc;
    const float m1 = dmin * m;

    // group h stores its 32 elements as one byte each: element e (e>>5 == h, e&31 == j)
    // uses qs[32*(h>>1) + j], low nibble when h is even, high nibble when h is odd
    const uint8_t * ql = x->qs + 32*(h >> 1) + (e0 & 31);
    const uint8_t * qh = x->qh + (e0 & 31);
    const bool hi = (h & 1) != 0;

    uint32_t v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t qb = ql[i];
        const uint32_t hbit = (qh[i] >> h) & 1;
        const uint32_t nib = hi ? ((qb >> 4) & 0xF) : (qb & 0xF);
        const float val = d1 * (float)(nib | (hbit << 4)) - m1;
        v[i] = __half_as_ushort(__float2half(val));
    }

    uint4 out;
    out.x = v[0] | (v[1] << 16);
    out.y = v[2] | (v[3] << 16);
    out.z = v[4] | (v[5] << 16);
    out.w = v[6] | (v[7] << 16);
    *(uint4 *)(yy + ib*QK_K + e0) = out;
}

static __global__ void k_new_q5_K(const void * __restrict__ vx, half * __restrict__ yy, const int64_t nb) {
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;
    for (int64_t ib = (int64_t)blockIdx.x*nwarp + warp; ib < nb; ib += (int64_t)gridDim.x*nwarp) {
        new_q5_K_body(vx, ib, yy, lane);
    }
}

// ---------------------------------------------------------------------------
typedef void (*launch_fn)(const void *, half *, int64_t);

static void report(const char * name, launch_fn lf, const void * d_in, half * d_out, int64_t nblocks, double bytes_per_block) {
    lf(d_in, d_out, nblocks);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int iters = 30;
    cudaEventRecord(e0);
    for (int i = 0; i < iters; ++i) lf(d_in, d_out, nblocks);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    const double total_gb = (double)nblocks*bytes_per_block/1e9;
    printf("%-34s %8.3f ms   %6.1f GB/s total traffic\n", name, ms/iters, total_gb/(ms/iters/1000.0));
}

int main(int argc, char ** argv) {
    cudaSetDevice(0);
    const char * path6 = argc > 1 ? argv[1] : "w_Q6_K.bin";
    const char * path5 = argc > 2 ? argv[2] : "w_Q5_K.bin";
    cudaError_t err;

    // ---------------- Q6_K ----------------
    std::vector<uint8_t> h6;
    {
        FILE * f = fopen(path6, "rb");
        if (!f) { printf("cannot open %s\n", path6); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        h6.resize(sz); if (fread(h6.data(), 1, sz, f) != (size_t)sz) return 1;
        fclose(f);
    }
    if (sizeof(block_q6_K) != 210) { printf("BAD q6_K struct size %d\n", (int)sizeof(block_q6_K)); return 1; }
    const int64_t nb6 = h6.size()/sizeof(block_q6_K);
    printf("Q6_K: %.1f MB, %lld blocks\n", h6.size()/1e6, (long long)nb6);

    void * d_in6; half * d_out6; half * d_ref6;
    cudaMalloc(&d_in6, h6.size());
    cudaMalloc(&d_out6, nb6*QK_K*2);
    cudaMalloc(&d_ref6, nb6*QK_K*2);
    cudaMemcpy(d_in6, h6.data(), h6.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_ref6, 0, nb6*QK_K*2);
    cudaMemset(d_out6, 0, nb6*QK_K*2);

    k_old_q6_K<<<(int)nb6, 64>>>(d_in6, d_ref6);
    k_new_q6_K<<<512, 256>>>(d_in6, d_out6, nb6);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("q6 kernel error: %s\n", cudaGetErrorString(err)); return 1; }

    {
        std::vector<half> ref(nb6*QK_K), out(nb6*QK_K);
        cudaMemcpy(ref.data(), d_ref6, ref.size()*2, cudaMemcpyDeviceToHost);
        cudaMemcpy(out.data(), d_out6, out.size()*2, cudaMemcpyDeviceToHost);
        int64_t bad = 0, first_bad = -1;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (__half_as_ushort(ref[i]) != __half_as_ushort(out[i])) { if (first_bad < 0) first_bad = (int64_t)i; ++bad; }
        }
        printf("Q6_K correctness: %lld / %zu mismatches (first at %lld)", (long long)bad, ref.size(), (long long)first_bad);
        if (first_bad >= 0) printf("  ref=%.6f new=%.6f", __half2float(ref[first_bad]), __half2float(out[first_bad]));
        printf("\n");
    }

    report("Q6_K old (64thr/blk)",
           [](const void * a, half * b, int64_t n){ k_old_q6_K<<<(int)n, 64>>>((const void*)a, b); }, d_in6, d_ref6, nb6, 722.0);
    report("Q6_K new (warp/blk gs=512)",
           [](const void * a, half * b, int64_t n){ k_new_q6_K<<<512, 256>>>((const void*)a, b, n); }, d_in6, d_out6, nb6, 722.0);
    report("Q6_K new (warp/blk gs=1024)",
           [](const void * a, half * b, int64_t n){ k_new_q6_K<<<1024, 256>>>((const void*)a, b, n); }, d_in6, d_out6, nb6, 722.0);
    report("Q6_K new (warp/blk gs=2048)",
           [](const void * a, half * b, int64_t n){ k_new_q6_K<<<2048, 256>>>((const void*)a, b, n); }, d_in6, d_out6, nb6, 722.0);
    report("Q6_K new (warp/blk gs=4096)",
           [](const void * a, half * b, int64_t n){ k_new_q6_K<<<4096, 256>>>((const void*)a, b, n); }, d_in6, d_out6, nb6, 722.0);
    report("Q6_K new x2 (warp/2blk gs=2048)",
           [](const void * a, half * b, int64_t n){ k_new_q6_K_x2<<<2048, 256>>>((const void*)a, b, n); }, d_in6, d_out6, nb6, 722.0);
    report("Q6_K new x2 (warp/2blk gs=4096)",
           [](const void * a, half * b, int64_t n){ k_new_q6_K_x2<<<4096, 256>>>((const void*)a, b, n); }, d_in6, d_out6, nb6, 722.0);

    // ---------------- Q5_K ----------------
    std::vector<uint8_t> h5;
    {
        FILE * f = fopen(path5, "rb");
        if (!f) { printf("cannot open %s\n", path5); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        h5.resize(sz); if (fread(h5.data(), 1, sz, f) != (size_t)sz) return 1;
        fclose(f);
    }
    if (sizeof(block_q5_K) != 176) { printf("BAD q5_K struct size %d\n", (int)sizeof(block_q5_K)); return 1; }
    const int64_t nb5 = h5.size()/sizeof(block_q5_K);
    printf("Q5_K: %.1f MB, %lld blocks\n", h5.size()/1e6, (long long)nb5);

    void * d_in5; half * d_out5; half * d_ref5;
    cudaMalloc(&d_in5, h5.size());
    cudaMalloc(&d_out5, nb5*QK_K*2);
    cudaMalloc(&d_ref5, nb5*QK_K*2);
    cudaMemcpy(d_in5, h5.data(), h5.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_ref5, 0, nb5*QK_K*2);
    cudaMemset(d_out5, 0, nb5*QK_K*2);

    k_old_q5_K<<<(int)nb5, 64>>>(d_in5, d_ref5);
    k_new_q5_K<<<512, 256>>>(d_in5, d_out5, nb5);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("q5 kernel error: %s\n", cudaGetErrorString(err)); return 1; }

    {
        std::vector<half> ref(nb5*QK_K), out(nb5*QK_K);
        cudaMemcpy(ref.data(), d_ref5, ref.size()*2, cudaMemcpyDeviceToHost);
        cudaMemcpy(out.data(), d_out5, out.size()*2, cudaMemcpyDeviceToHost);
        int64_t bad = 0, first_bad = -1;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (__half_as_ushort(ref[i]) != __half_as_ushort(out[i])) { if (first_bad < 0) first_bad = (int64_t)i; ++bad; }
        }
        printf("Q5_K correctness: %lld / %zu mismatches (first at %lld)", (long long)bad, ref.size(), (long long)first_bad);
        if (first_bad >= 0) printf("  ref=%.6f new=%.6f", __half2float(ref[first_bad]), __half2float(out[first_bad]));
        printf("\n");
    }

    report("Q5_K old (64thr/blk)",
           [](const void * a, half * b, int64_t n){ k_old_q5_K<<<(int)n, 64>>>((const void*)a, b); }, d_in5, d_ref5, nb5, 688.0);
    report("Q5_K new (warp/blk gs=512)",
           [](const void * a, half * b, int64_t n){ k_new_q5_K<<<512, 256>>>((const void*)a, b, n); }, d_in5, d_out5, nb5, 688.0);
    report("Q5_K new (warp/blk gs=2048)",
           [](const void * a, half * b, int64_t n){ k_new_q5_K<<<2048, 256>>>((const void*)a, b, n); }, d_in5, d_out5, nb5, 688.0);
    return 0;
}
