// Gate A step 1: peak throughput of the repo's sm70 mma primitives (registers only)
#include "mma.cuh"
#include <cstdio>
#include <cuda_runtime.h>

using namespace ggml_cuda_mma;

typedef tile<32, 4, half2, DATA_LAYOUT_I_MAJOR>          tA;
typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tB;
typedef tile<32, 8, float, DATA_LAYOUT_I_MAJOR>          tC;

#define NTA 2
#define NTB 4
#define NWARPS 8

__global__ void __launch_bounds__(32*NWARPS) mma_peak(float * __restrict__ out, const int iters) {
    tA A[NTA];
    tB B[NTB];
    tC C[NTA][NTB];
    const int tid = threadIdx.x;

#pragma unroll
    for (int i = 0; i < NTA; ++i) {
#pragma unroll
        for (int l = 0; l < tA::ne; ++l) {
            A[i].x[l] = make_half2((float)(tid + l + i*7)*1e-7f, (float)(tid*l + i + 1)*1e-7f);
        }
    }
#pragma unroll
    for (int i = 0; i < NTB; ++i) {
#pragma unroll
        for (int l = 0; l < tB::ne; ++l) {
            B[i].x[l] = make_half2((float)(tid - l - i)*1e-7f, (float)(tid + l + i + 3)*1e-7f);
        }
    }
#pragma unroll
    for (int ia = 0; ia < NTA; ++ia) {
#pragma unroll
        for (int ib = 0; ib < NTB; ++ib) {
#pragma unroll
            for (int l = 0; l < tC::ne; ++l) {
                C[ia][ib].x[l] = 0.0f;
            }
        }
    }

    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int ia = 0; ia < NTA; ++ia) {
#pragma unroll
            for (int ib = 0; ib < NTB; ++ib) {
                mma(C[ia][ib], A[ia], B[ib]);
            }
        }
    }

    float s = 0.0f;
#pragma unroll
    for (int ia = 0; ia < NTA; ++ia) {
#pragma unroll
        for (int ib = 0; ib < NTB; ++ib) {
#pragma unroll
            for (int l = 0; l < tC::ne; ++l) {
                s += C[ia][ib].x[l];
            }
        }
    }
    if (s == 12345.6789f) {
        out[0] = s;
    }
}

int main(int argc, char ** argv) {
    const int iters  = argc > 1 ? atoi(argv[1]) : 20000;
    const int blocks = argc > 2 ? atoi(argv[2]) : 80;

    float * d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaMemset(d_out, 0, sizeof(float));

    mma_peak<<<blocks, 32*NWARPS>>>(d_out, 10);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    mma_peak<<<blocks, 32*NWARPS>>>(d_out, iters);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    // per warp per iter: NTA*NTB mma() calls, each 32x8x8 MACs
    const double flops = (double)blocks * NWARPS * (double)iters * NTA * NTB * 32.0 * 8.0 * 8.0 * 2.0;
    printf("blocks=%d warps/blk=%d iters=%d  time=%.3f ms  TFLOPS=%.2f\n",
           blocks, NWARPS, iters, ms, flops / (ms*1e-3) / 1e12);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaFree(d_out);
    return 0;
}
