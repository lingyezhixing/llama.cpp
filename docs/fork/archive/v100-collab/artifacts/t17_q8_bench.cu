// T17: Q8_0 -> f16 dequant micro-benchmark: current upstream kernel vs vectorized candidate
// block_q8_0 = { half d; int8_t qs[32]; } = 34 bytes (32 elements)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define QK8_0 32
#define CUDA_Q8_0_NE_ALIGN 2048
#define WARP_SIZE 32

typedef struct {
    half    d;
    int8_t  qs[QK8_0];
} block_q8_0;

static_assert(sizeof(block_q8_0) == 34, "bad block_q8_0 size");

// ---------------------------------------------------------------------------
// current upstream kernel (verbatim port of dequantize_block_q8_0_f16<false>)
template <bool need_check>
static __global__ void k_old_q8_0(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }
        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }
        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];
        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
}

static void launch_old(const void * vx, half * y, int64_t k, cudaStream_t st) {
    const int num_blocks = (int)((k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN);
    k_old_q8_0<false><<<num_blocks, WARP_SIZE, 0, st>>>(vx, y, k);
}

// ---------------------------------------------------------------------------
// candidate: 1 warp per 256 elements (8 blocks), lane handles 8 consecutive int8 -> uint4 store
static __global__ void k_new_q8_0(const void * __restrict__ vx, half * __restrict__ yy, const int64_t nblk) {
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;
    const block_q8_0 * x = (const block_q8_0 *) vx;

    for (int64_t ib = (int64_t)blockIdx.x*nwarp + warp; ib < nblk; ib += (int64_t)gridDim.x*nwarp) {
        const int  sub = lane >> 2;         // 8 sub-blocks per warp
        const int  off = (lane & 3) * 8;    // 8 consecutive elements inside the sub-block
        const block_q8_0 * b = x + ib*8 + sub;
        const half  d = b->d;

        // 8 consecutive int8 values
        const char2 c0 = *(const char2 *)(b->qs + off);
        const char2 c1 = *(const char2 *)(b->qs + off + 2);
        const char2 c2 = *(const char2 *)(b->qs + off + 4);
        const char2 c3 = *(const char2 *)(b->qs + off + 6);

        const half2 d2 = __half2half2(d);
        const half2 h0 = __hmul2(make_half2(c0.x, c0.y), d2);
        const half2 h1 = __hmul2(make_half2(c1.x, c1.y), d2);
        const half2 h2 = __hmul2(make_half2(c2.x, c2.y), d2);
        const half2 h3 = __hmul2(make_half2(c3.x, c3.y), d2);

        uint4 out;
        out.x = *(const uint32_t *)&h0;
        out.y = *(const uint32_t *)&h1;
        out.z = *(const uint32_t *)&h2;
        out.w = *(const uint32_t *)&h3;
        *(uint4 *)(yy + ib*256 + sub*32 + off) = out;
    }
}

// ---------------------------------------------------------------------------
int main() {
    cudaSetDevice(0);
    const int64_t nblk   = 2*1024*1024;     // 256-element blocks -> 512M elements
    const int64_t nelem  = nblk*256;
    const size_t  in_sz  = (size_t)nblk*272; // 8 x 34 bytes per 256 elements
    printf("Q8_0: %lld elements, %.0f MB in, %.0f MB out\n", (long long)nelem, in_sz/1e6, nelem*2/1e6);

    std::vector<uint8_t> h(in_sz);
    srand(1234);
    for (size_t i = 0; i < in_sz; ++i) h[i] = (uint8_t)(rand() & 0xFF);

    void *d_in; half *d_ref; half *d_out;
    cudaMalloc(&d_in, in_sz);
    cudaMalloc(&d_ref, nelem*2);
    cudaMalloc(&d_out, nelem*2);
    cudaMemcpy(d_in, h.data(), in_sz, cudaMemcpyHostToDevice);

    launch_old(d_in, d_ref, nelem, 0);
    k_new_q8_0<<<1024, 256>>>(d_in, d_out, nblk);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("kernel error: %s\n", cudaGetErrorString(err)); return 1; }

    {
        std::vector<half> ref(nelem), out(nelem);
        cudaMemcpy(ref.data(), d_ref, nelem*2, cudaMemcpyDeviceToHost);
        cudaMemcpy(out.data(), d_out, nelem*2, cudaMemcpyDeviceToHost);
        int64_t bad = 0, first = -1;
        for (int64_t i = 0; i < nelem; ++i) {
            if (__half_as_ushort(ref[i]) != __half_as_ushort(out[i])) { if (first < 0) first = i; ++bad; }
        }
        printf("correctness: %lld / %lld mismatches (first at %lld)\n", (long long)bad, (long long)nelem, (long long)first);
    }

    const double bytes = (double)nelem/32.0*34.0 + (double)nelem*2.0;
    for (int variant = 0; variant < 2; ++variant) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        const int iters = 20;
        launch_old(d_in, d_ref, nelem, 0);
        cudaDeviceSynchronize();
        cudaEventRecord(e0);
        for (int i = 0; i < iters; ++i) {
            if (variant == 0) launch_old(d_in, d_ref, nelem, 0);
            else              k_new_q8_0<<<1024, 256>>>(d_in, d_out, nblk);
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("%-28s %8.3f ms   %6.1f GB/s\n", variant == 0 ? "old (warp/2048, smem)" : "new (warp/256, uint4)", ms/iters, bytes/(ms/iters/1000.0)/1e9);
    }
    return 0;
}
