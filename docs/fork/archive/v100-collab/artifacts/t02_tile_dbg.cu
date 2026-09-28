// minimal isolation test: one warp, 64x32 output tile, K=8
// A = W[64 rows][8 k], B = X[32 tok][8 k], C[64][32] = A * B^T
#include "mma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace ggml_cuda_mma;

typedef tile<32, 4, half2, DATA_LAYOUT_I_MAJOR>          tA;
typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tB;
typedef tile<32, 8, float, DATA_LAYOUT_I_MAJOR>          tC;

__device__ __forceinline__ void load_frag4(half2 * x, const half * __restrict__ p) {
    const uint32_t * q = reinterpret_cast<const uint32_t *>(p);
    x[0] = *reinterpret_cast<const half2 *>(&q[0]);
    x[1] = *reinterpret_cast<const half2 *>(&q[1]);
    x[2] = *reinterpret_cast<const half2 *>(&q[2]);
    x[3] = *reinterpret_cast<const half2 *>(&q[3]);
}

__global__ void tile_test(const half * __restrict__ W, const half * __restrict__ X, float * __restrict__ C,
                          const int K, const int ldc) {
    const int lane = threadIdx.x % 32;
    tA a[2];
    tB b[4];
    tC c[2][4];

#pragma unroll
    for (int ia = 0; ia < 2; ++ia) {
        // rows ia*32 + lane, columns 0..7 of this k-block
        load_frag4(a[ia].x, W + (ia*32 + lane)*K);
    }
#pragma unroll
    for (int ib = 0; ib < 4; ++ib) {
        // rows ib*8 + get_i within tile, columns 0..7
        const int r = ib*8 + (lane/16)*4 + (lane%4);
        load_frag4(b[ib].x, X + r*K);
    }
#pragma unroll
    for (int ia = 0; ia < 2; ++ia) {
#pragma unroll
        for (int ib = 0; ib < 4; ++ib) {
#pragma unroll
            for (int l = 0; l < tC::ne; ++l) {
                c[ia][ib].x[l] = 0.0f;
            }
        }
    }
#pragma unroll
    for (int ia = 0; ia < 2; ++ia) {
#pragma unroll
        for (int ib = 0; ib < 4; ++ib) {
            mma(c[ia][ib], a[ia], b[ib]);
        }
    }
#pragma unroll
    for (int ia = 0; ia < 2; ++ia) {
#pragma unroll
        for (int ib = 0; ib < 4; ++ib) {
#pragma unroll
            for (int l = 0; l < tC::ne; ++l) {
                const int i = ia*32 + tC::get_i(l);
                const int j = ib*8  + tC::get_j(l);
                C[i*ldc + j] = c[ia][ib].x[l];
            }
        }
    }
}

int main() {
    const int M = 64, N = 32, K = 8;
    half * hW = (half *) malloc(M*K*sizeof(half));
    half * hX = (half *) malloc(N*K*sizeof(half));
    float * hC = (float *) malloc(M*N*sizeof(float));
    float * hR = (float *) malloc(M*N*sizeof(float));
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) hW[m*K+k] = __float2half((m*8+k) % 7 - 3);
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) hX[n*K+k] = __float2half((n*5+k*3) % 5 - 2);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
        float s = 0;
        for (int k = 0; k < K; ++k) s += __half2float(hW[m*K+k]) * __half2float(hX[n*K+k]);
        hR[m*N+n] = s;
    }
    half * dW, * dX;
    float * dC;
    cudaMalloc(&dW, M*K*sizeof(half));
    cudaMalloc(&dX, N*K*sizeof(half));
    cudaMalloc(&dC, M*N*sizeof(float));
    cudaMemcpy(dW, hW, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, hX, N*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*N*sizeof(float));
    tile_test<<<1, 32>>>(dW, dX, dC, K, N);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("kernel error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(hC, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    int nbad = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            if (hC[m*N+n] != hR[m*N+n]) {
                if (nbad < 12) printf("  C[%2d][%2d] = %6.1f  expected %6.1f\n", m, n, hC[m*N+n], hR[m*N+n]);
                nbad++;
            }
        }
    }
    printf("mismatched: %d / %d\n", nbad, M*N);
    if (nbad == 0) printf("TILE PRIMITIVES OK\n");
    return 0;
}
