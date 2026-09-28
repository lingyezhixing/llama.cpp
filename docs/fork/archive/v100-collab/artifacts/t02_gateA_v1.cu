// T02 Gate A: fp16 MMA pipeline (reuses the repo's sm70 mma primitives)
// C[M,N] = W[M,K] * X[N,K]^T, fp16 inputs, fp32 accum.
// Weight role: A operand (M = output channels). Activation role: B operand (N = tokens).
// mode bit 2 = skip smem staging (pure mma from smem), for phase analysis.
#include "mma.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace ggml_cuda_mma;

typedef tile<32, 4, half2, DATA_LAYOUT_I_MAJOR>          tA;
typedef tile< 8, 4, half2, DATA_LAYOUT_I_MAJOR_MIRRORED> tB;
typedef tile<32, 8, float, DATA_LAYOUT_I_MAJOR>          tC;

#define BM 128
#define BN 128
#define BK 32
#define BKP (BK + 8)
#define NTHREAD 256
#define WM 2
#define WN 4
#define NTA 2
#define NTB 4

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__device__ __forceinline__ void load_frag4(half2 * x, const half * __restrict__ p) {
    const uint32_t * q = reinterpret_cast<const uint32_t *>(p);
    x[0] = *reinterpret_cast<const half2 *>(&q[0]);
    x[1] = *reinterpret_cast<const half2 *>(&q[1]);
    x[2] = *reinterpret_cast<const half2 *>(&q[2]);
    x[3] = *reinterpret_cast<const half2 *>(&q[3]);
}

__global__ void __launch_bounds__(NTHREAD, 2)
gateA_kernel(const half * __restrict__ W, const half * __restrict__ X, float * __restrict__ C,
             const int M, const int N, const int K, const int mode) {
    __shared__ half sA[2][BM*BKP];
    __shared__ half sB[2][BN*BKP];

    // tile get_i/get_j assume threadIdx.x is the lane -> use a 2D block (32, NWARPS)
    const int lane  = threadIdx.x;
    const int warp  = threadIdx.y;
    const int tid   = warp*32 + lane;
    const int wm    = warp % WM;
    const int wn    = warp / WM;
    const int m0    = blockIdx.x * BM;
    const int n0    = blockIdx.y * BN;
    const int nkb   = K / BK;

    float4 regA[2];
    float4 regB[2];

    // prologue: stage k-block 0
    {
        const int row = tid / 4;
        const int part = tid % 4;
        regA[0] = *reinterpret_cast<const float4 *>(W + (int64_t)(m0 + row)*K + part*8);
        regA[1] = *reinterpret_cast<const float4 *>(W + (int64_t)(m0 + row + 64)*K + part*8);
        regB[0] = *reinterpret_cast<const float4 *>(X + (int64_t)(n0 + row)*K + part*8);
        regB[1] = *reinterpret_cast<const float4 *>(X + (int64_t)(n0 + row + 64)*K + part*8);
        *reinterpret_cast<float4 *>(sA[0] + row*BKP + part*8) = regA[0];
        *reinterpret_cast<float4 *>(sA[0] + (row + 64)*BKP + part*8) = regA[1];
        *reinterpret_cast<float4 *>(sB[0] + row*BKP + part*8) = regB[0];
        *reinterpret_cast<float4 *>(sB[0] + (row + 64)*BKP + part*8) = regB[1];
    }
    __syncthreads();

    tC c[NTA][NTB];
#pragma unroll
    for (int ia = 0; ia < NTA; ++ia) {
#pragma unroll
        for (int ib = 0; ib < NTB; ++ib) {
#pragma unroll
            for (int l = 0; l < tC::ne; ++l) {
                c[ia][ib].x[l] = 0.0f;
            }
        }
    }

    for (int kb = 0; kb < nkb; ++kb) {
        const int stage = kb & 1;
        const int kb_next = kb + 1;
        const bool have_next = kb_next < nkb;

        if ((mode & 16) && kb == 0) {
            // dump the fragments each warp actually reads (ks=0)
            tA ad[NTA];
            tB bd[NTB];
            for (int ia = 0; ia < NTA; ++ia) {
                load_frag4(ad[ia].x, sA[0] + (wm*(BM/WM) + ia*32 + lane)*BKP + 0);
            }
            for (int ib = 0; ib < NTB; ++ib) {
                load_frag4(bd[ib].x, sB[0] + (wn*(BN/WN) + ib*8 + (lane/16)*4 + (lane%4))*BKP + 0);
            }
            float * d = C + warp*(NTA + NTB)*32;
            for (int ia = 0; ia < NTA; ++ia) {
                d[(ia)*32 + lane] = __half2float(ad[ia].x[0].x);
            }
            for (int ib = 0; ib < NTB; ++ib) {
                d[(NTA + ib)*32 + lane] = __half2float(bd[ib].x[0].x);
            }
            __syncthreads();
            return;
        }

        if ((mode & 4) && kb == 0) {
            // dump staged smem (row-major, padded) for inspection
            for (int i = tid; i < BM*BKP; i += NTHREAD) {
                C[i] = __half2float(sA[0][i]);
            }
            for (int i = tid; i < BN*BKP; i += NTHREAD) {
                C[BM*BKP + i] = __half2float(sB[0][i]);
            }
            __syncthreads();
            return;
        }

        // prefetch next k-block into registers
        if (have_next) {
            const int row = tid / 4;
            const int part = tid % 4;
            const int64_t koff = (int64_t)kb_next*BK + part*8;
            regA[0] = *reinterpret_cast<const float4 *>(W + (int64_t)(m0 + row)*K + koff);
            regA[1] = *reinterpret_cast<const float4 *>(W + (int64_t)(m0 + row + 64)*K + koff);
            regB[0] = *reinterpret_cast<const float4 *>(X + (int64_t)(n0 + row)*K + koff);
            regB[1] = *reinterpret_cast<const float4 *>(X + (int64_t)(n0 + row + 64)*K + koff);
        }

        // mma on current stage
        if (!(mode & 2)) {
#pragma unroll
            for (int ks = 0; ks < BK/8; ++ks) {
                tA a[NTA];
                tB b[NTB];
#pragma unroll
                for (int ia = 0; ia < NTA; ++ia) {
                    load_frag4(a[ia].x, sA[stage] + (wm*(BM/WM) + ia*32 + lane)*BKP + ks*8);
                }
#pragma unroll
                for (int ib = 0; ib < NTB; ++ib) {
                    load_frag4(b[ib].x, sB[stage] + (wn*(BN/WN) + ib*8 + (lane/16)*4 + (lane%4))*BKP + ks*8);
                }
#pragma unroll
                for (int ia = 0; ia < NTA; ++ia) {
#pragma unroll
                    for (int ib = 0; ib < NTB; ++ib) {
                        mma(c[ia][ib], a[ia], b[ib]);
                    }
                }
            }
        }

        // commit next stage
        if (have_next) {
            const int row = tid / 4;
            const int part = tid % 4;
            const int ns = stage ^ 1;
            *reinterpret_cast<float4 *>(sA[ns] + row*BKP + part*8) = regA[0];
            *reinterpret_cast<float4 *>(sA[ns] + (row + 64)*BKP + part*8) = regA[1];
            *reinterpret_cast<float4 *>(sB[ns] + row*BKP + part*8) = regB[0];
            *reinterpret_cast<float4 *>(sB[ns] + (row + 64)*BKP + part*8) = regB[1];
        }
        __syncthreads();
    }

    // store C (row-major [M][N])
    const int cm = m0 + wm*(BM/WM);
    const int cn = n0 + wn*(BN/WN);
#pragma unroll
    for (int ia = 0; ia < NTA; ++ia) {
#pragma unroll
        for (int ib = 0; ib < NTB; ++ib) {
#pragma unroll
            for (int l = 0; l < tC::ne; ++l) {
                const int i = cm + ia*32 + tC::get_i(l);
                const int j = cn + ib*8  + tC::get_j(l);
                C[(int64_t)i*N + j] = c[ia][ib].x[l];
            }
        }
    }
}

int main(int argc, char ** argv) {
    const int M = argc > 1 ? atoi(argv[1]) : 17408;
    const int N = argc > 2 ? atoi(argv[2]) : 512;
    const int K = argc > 3 ? atoi(argv[3]) : 5120;
    const int reps = argc > 4 ? atoi(argv[4]) : 50;
    const int mode = argc > 5 ? atoi(argv[5]) : 0;
    const int device = argc > 6 ? atoi(argv[6]) : 0;

    CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device));
    printf("device: %s  mode=%d  M=%d N=%d K=%d reps=%d\n", prop.name, mode, M, N, K, reps);

    half * dW, * dX;
    float * dC, * dCref;
    CHECK(cudaMalloc(&dW, (size_t)M*K*sizeof(half)));
    CHECK(cudaMalloc(&dX, (size_t)N*K*sizeof(half)));
    CHECK(cudaMalloc(&dC, (size_t)M*N*sizeof(float)));
    CHECK(cudaMalloc(&dCref, (size_t)M*N*sizeof(float)));

    half * hW = (half *) malloc((size_t)M*K*sizeof(half));
    half * hX = (half *) malloc((size_t)N*K*sizeof(half));
    {
        srand(1234);
        for (size_t i = 0; i < (size_t)M*K; ++i) hW[i] = __float2half(((rand() % 2001) - 1000) / 1000.0f);
        for (size_t i = 0; i < (size_t)N*K; ++i) hX[i] = __float2half(((rand() % 2001) - 1000) / 1000.0f);
        CHECK(cudaMemcpy(dW, hW, (size_t)M*K*sizeof(half), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dX, hX, (size_t)N*K*sizeof(half), cudaMemcpyHostToDevice));
        if (!(mode & 8)) { free(hW); free(hX); hW = nullptr; hX = nullptr; }
    }

    const double flops = 2.0*M*N*K;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // cuBLAS reference
    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
    float ms_cublas = 0.0f;
    {
        const float alpha = 1.0f, beta = 0.0f;
        // C(MxN) = W(MxK) * X^T ; column-major trick: C^T (N x M) = X (N x K) * W^T (K x M)
        // use cublasGemmEx: C^T(N,M) = X(N,K) * W^T(K,M) with W^T = op(W), W row-major (M,K) = column-major (K,M)
        // so: gemm(op=trans, op=notrans): C^T = A^T * B with A = W row-major? => simplify via cublasSgemm on X*W^T
        // cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n=N, m=M, k=K, ...) computing (C^T)^{NxM} = (X)^{NxK} * (W^T)
        cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                     dX, CUDA_R_16F, K, dW, CUDA_R_16F, K, &beta,
                     dCref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
        for (int i = 0; i < 3; ++i) {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                         dX, CUDA_R_16F, K, dW, CUDA_R_16F, K, &beta,
                         dCref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaEventRecord(t0);
        for (int i = 0; i < reps; ++i) {
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                         dX, CUDA_R_16F, K, dW, CUDA_R_16F, K, &beta,
                         dCref, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_cublas, t0, t1);
        ms_cublas /= reps;
        printf("cuBLAS fp16: %.3f ms  %.2f TFLOPS\n", ms_cublas, flops/(ms_cublas*1e-3)/1e12);
    }

    // mine
    dim3 grid((M + BM - 1)/BM, (N + BN - 1)/BN);
    float ms_mine = 0.0f;
    {
        const dim3 block(32, NTHREAD/32);
        gateA_kernel<<<grid, block>>>(dW, dX, dC, M, N, K, mode);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        for (int i = 0; i < 3; ++i) {
            gateA_kernel<<<grid, block>>>(dW, dX, dC, M, N, K, mode);
        }
        cudaEventRecord(t0);
        for (int i = 0; i < reps; ++i) {
            gateA_kernel<<<grid, block>>>(dW, dX, dC, M, N, K, mode);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms_mine, t0, t1);
        ms_mine /= reps;
        CHECK(cudaGetLastError());
        printf("mine       : %.3f ms  %.2f TFLOPS\n", ms_mine, flops/(ms_mine*1e-3)/1e12);
    }

    // correctness
    {
        const size_t n = (size_t)M*N;
        float * hC = (float *) malloc(n*sizeof(float));
        float * hR = (float *) malloc(n*sizeof(float));
        CHECK(cudaMemcpy(hC, dC, n*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(hR, dCref, n*sizeof(float), cudaMemcpyDeviceToHost));
        double sum_ref = 0.0, sum_dif = 0.0;
        double max_abs = 0.0, max_rel = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double r = hR[i], d = hC[i];
            sum_ref += r*r;
            sum_dif += (d-r)*(d-r);
            if (fabs(d-r) > max_abs) max_abs = fabs(d-r);
            if (fabs(r) > 1.0) {
                const double rel = fabs(d-r)/fabs(r);
                if (rel > max_rel) max_rel = rel;
            }
        }
        printf("correctness vs cuBLAS: NMSE=%.3e  max_abs=%.4e  max_rel(|ref|>1)=%.3e\n", sum_dif/sum_ref, max_abs, max_rel);
        if (mode & 16) {
            int nbad = 0;
            for (int w = 0; w < 8; ++w) {
                const int wm = w % WM, wn = w / WM;
                for (int ia = 0; ia < NTA; ++ia) {
                    for (int lane = 0; lane < 32; ++lane) {
                        const float got = hC[w*(NTA+NTB)*32 + ia*32 + lane];
                        const float exp = __half2float(hW[(size_t)(wm*(BM/WM) + ia*32 + lane)*K + 0]);
                        if (fabsf(got-exp) > 1e-3f) { if (nbad < 8) printf("  warp%d A[%d] lane%d got %.3f exp %.3f\n", w, ia, lane, got, exp); nbad++; }
                    }
                }
                for (int ib = 0; ib < NTB; ++ib) {
                    for (int lane = 0; lane < 32; ++lane) {
                        const float got = hC[w*(NTA+NTB)*32 + (NTA+ib)*32 + lane];
                        const float exp = __half2float(hX[(size_t)(wn*(BN/WN) + ib*8 + (lane/16)*4 + (lane%4))*K + 0]);
                        if (fabsf(got-exp) > 1e-3f) { if (nbad < 8) printf("  warp%d B[%d] lane%d got %.3f exp %.3f\n", w, ib, lane, got, exp); nbad++; }
                    }
                }
            }
            printf("fragment dump (A+B, ks=0): bad=%d / %d\n", nbad, 8*(NTA+NTB)*32);
        }
        if (mode & 4) {
            int nbadA = 0, nbadB = 0;
            for (int r = 0; r < BM; ++r) {
                for (int c = 0; c < BK; ++c) {
                    const float got = hC[r*BKP + c];
                    const float exp = __half2float(hW[(size_t)(0 + r)*K + 0 + c]);
                    if (fabsf(got - exp) > 1e-3f) { if (nbadA < 5) printf("  sA[%d][%d]=%.3f exp %.3f\n", r, c, got, exp); nbadA++; }
                }
            }
            for (int r = 0; r < BN; ++r) {
                for (int c = 0; c < BK; ++c) {
                    const float got = hC[BM*BKP + r*BKP + c];
                    const float exp = __half2float(hX[(size_t)(0 + r)*K + 0 + c]);
                    if (fabsf(got - exp) > 1e-3f) { if (nbadB < 5) printf("  sB[%d][%d]=%.3f exp %.3f\n", r, c, got, exp); nbadB++; }
                }
            }
            printf("staging check (block 0, kb 0): sA bad=%d  sB bad=%d\n", nbadA, nbadB);
        }
        if (mode & 8) {
            int nbad = 0;
            double sref = 0, sdif = 0;
            for (int m = 0; m < M; ++m) {
                for (int n = 0; n < N; ++n) {
                    float ref = 0.0f;
                    for (int k = 0; k < K; ++k) ref += __half2float(hW[(size_t)m*K + k]) * __half2float(hX[(size_t)n*K + k]);
                    const float got = hC[(size_t)m*N + n];
                    sref += (double)ref*ref;
                    sdif += (double)(got-ref)*(got-ref);
                    if (fabs(got-ref) > 1e-2f) {
                        if (nbad < 10) printf("  C[%d][%d] = %.4f expected %.4f\n", m, n, got, ref);
                        nbad++;
                    }
                }
            }
            printf("vs host ref: NMSE=%.3e  mismatched=%d / %d\n", sdif/sref, nbad, M*N);
            printf("map (rows 0..127 step 4, cols 0..127 step 4): '.'=ok '0'=zero 'X'=bad\n");
            for (int m = 0; m < M; m += 4) {
                char line[40];
                int p = 0;
                for (int n = 0; n < N; n += 4) {
                    float ref = 0.0f;
                    for (int k = 0; k < K; ++k) ref += __half2float(hW[(size_t)m*K + k]) * __half2float(hX[(size_t)n*K + k]);
                    const float got = hC[(size_t)m*N + n];
                    line[p++] = (fabs(got-ref) < 1e-2f) ? '.' : (got == 0.0f ? '0' : 'X');
                }
                line[p] = 0;
                printf("%3d %s\n", m, line);
            }
        }
        free(hC); free(hR);
    }

    cublasDestroy(h);
    cudaFree(dW); cudaFree(dX); cudaFree(dC); cudaFree(dCref);
    return 0;
}
