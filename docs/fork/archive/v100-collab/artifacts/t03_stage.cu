// T03 V5: block-cooperative smem double-buffered staging of k/q/v/g/b -> hides the per-token global load
// latency without register pressure. Layout: block = 4 warps = 4 columns of the same head (grid (H,1,32)).
// staging: thread i loads k[t][i], q[t][i], v[t][i] (128 each) + t0 loads g/beta. Consumed from smem by all warps.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define D 128
#define H 48
#define T 512
#define NWARPS 4
#define S_v 128

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s line %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)
__device__ __forceinline__ float4 ld4(const float * p) { return *reinterpret_cast<const float4 *>(p); }

__global__ void ref_kernel(const float * q, const float * k, const float * v, const float * g, const float * b,
                           const float * s_in, float * out, float * s_out, int t_tokens, float scale) {
    const int h = blockIdx.x; const int j = threadIdx.x;
    float col[D];
    const float * sin_h = s_in + (size_t)h*D*D;
    for (int i = 0; i < D; i++) col[i] = sin_h[j*D + i];
    for (int t = 0; t < t_tokens; t++) {
        const float * q_t = q + (size_t)(t*H + h)*D;
        const float * k_t = k + (size_t)(t*H + h)*D;
        const float * v_t = v + (size_t)(h*t_tokens + t)*D;
        const float gt = expf(g[h*t_tokens + t]);
        const float bt = b[h*t_tokens + t];
        float kv = 0.f, at = 0.f;
        for (int i = 0; i < D; i++) kv += col[i]*k_t[i];
        const float delta = (v_t[j] - gt*kv)*bt;
        for (int i = 0; i < D; i++) { col[i] = gt*col[i] + k_t[i]*delta; at += col[i]*q_t[i]; }
        out[(size_t)(t*H + h)*D + j] = at*scale;
    }
    float * so = s_out + (size_t)h*D*D;
    for (int i = 0; i < D; i++) so[j*D + i] = col[i];
}

template <int C, bool STAGE, int MB>
__global__ void __launch_bounds__(32*NWARPS, MB)
variant_kernel(const float * q, const float * k, const float * v, const float * g, const float * beta,
               const float * s_in, float * out, float * s_out,
               int64_t H_, int64_t n_tokens, int64_t sq1, int64_t sq2, int64_t sq3,
               int64_t sv1, int64_t sv2, int64_t sv3, int64_t sb1, int64_t sb2, int64_t sb3, float scale) {
    constexpr int LPC = 32 / C;
    constexpr int RPC = S_v / LPC;
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int gq = lane / LPC, sub = lane % LPC;
    const int h = blockIdx.x;
    const int col = (blockIdx.z*NWARPS + warp)*C + gq;

    extern __shared__ float smem[];
    float * sk = smem;                    // [2][D]
    float * sq = sk + 2*D;                // [2][D]
    float * sv = sq + 2*D;                // [2][D]
    float * sgb = sv + 2*D;               // [2][2] (g, beta)

    float s[RPC];
    {
        const float * st = s_in + (size_t)h*S_v*S_v + col*S_v + sub*RPC;
        #pragma unroll
        for (int r = 0; r < RPC; r += 4) { const float4 t4 = ld4(st + r); s[r]=t4.x; s[r+1]=t4.y; s[r+2]=t4.z; s[r+3]=t4.w; }
    }
    const float * k_base = k + h*sq1;
    const float * q_base = q + h*sq1;
    const float * v_base = v + h*sv1;
    const int64_t gb_base = h*sb1;

    float pk=0.f, pq=0.f, pv=0.f, pg=0.f, pb=0.f;
    if (STAGE) {
        pk = k_base[0*sq2 + lane]; pq = q_base[0*sq2 + lane]; pv = v_base[0*sv2 + lane];
        if (lane == 0) { pg = g[gb_base]; pb = beta[gb_base]; }
    }
    for (int t = 0; t < n_tokens; t++) {
        float kk[RPC], qq[RPC]; float g_val, b_val, v_val;
        if (STAGE) {
            const int cur = t & 1;
            sk[cur*D + lane] = pk; sq[cur*D + lane] = pq; sv[cur*D + lane] = pv;
            if (lane == 0) { sgb[cur*2] = pg; sgb[cur*2+1] = pb; }
            __syncthreads();
            if (t+1 < n_tokens) {
                pk = k_base[(t+1)*sq2 + lane]; pq = q_base[(t+1)*sq2 + lane]; pv = v_base[(t+1)*sv2 + lane];
                if (lane == 0) { pg = g[gb_base + (t+1)*sb2]; pb = beta[gb_base + (t+1)*sb2]; }
            }
            #pragma unroll
            for (int r = 0; r < RPC; r++) { kk[r] = sk[cur*D + sub*RPC + r]; qq[r] = sq[cur*D + sub*RPC + r]; }
            v_val = sv[cur*D + col];
            g_val = expf(sgb[cur*2]); b_val = sgb[cur*2+1];
        } else {
            #pragma unroll
            for (int r = 0; r < RPC; r += 4) { const float4 t4 = ld4(k_base + t*sq2 + sub*RPC + r); kk[r]=t4.x; kk[r+1]=t4.y; kk[r+2]=t4.z; kk[r+3]=t4.w; }
            #pragma unroll
            for (int r = 0; r < RPC; r += 4) { const float4 t4 = ld4(q_base + t*sq2 + sub*RPC + r); qq[r]=t4.x; qq[r+1]=t4.y; qq[r+2]=t4.z; qq[r+3]=t4.w; }
            g_val = expf(g[gb_base + t*sb2]); b_val = beta[gb_base + t*sb2]; v_val = v_base[t*sv2 + col];
        }
        float kv = 0.f;
        #pragma unroll
        for (int r = 0; r < RPC; r++) kv += s[r]*kk[r];
        #pragma unroll
        for (int o = LPC/2; o > 0; o >>= 1) kv += __shfl_xor_sync(0xffffffffu, kv, o);
        const float delta = (v_val - g_val*kv)*b_val;
        float at = 0.f;
        #pragma unroll
        for (int r = 0; r < RPC; r++) { s[r] = g_val*s[r] + kk[r]*delta; at += s[r]*qq[r]; }
        #pragma unroll
        for (int o = LPC/2; o > 0; o >>= 1) at += __shfl_xor_sync(0xffffffffu, at, o);
        if (sub == 0) out[((t*H_ + h)*S_v) + col] = at*scale;
    }
    float * so = s_out + (size_t)h*S_v*S_v + col*S_v + sub*RPC;
    #pragma unroll
    for (int r = 0; r < RPC; r += 4) *reinterpret_cast<float4 *>(so + r) = make_float4(s[r],s[r+1],s[r+2],s[r+3]);
}

template <int C, bool STAGE, int MB>
void run_cfg(const char * nm, float * d_sout, float * d_out, const float * d_sin,
             const float * d_q,const float * d_k,const float * d_v,const float * d_g,const float * d_b,
             const float * h_oref, const float * h_sref, float * h_ochk, float * h_schk,
             const size_t sz_s, const size_t sz_v, int t_tokens, double & t_base) {
    const int gridz = S_v/(NWARPS*C);
    const int64_t sq1=D, sq2=(int64_t)H*D, sq3=(int64_t)T*H*D, sv1=(int64_t)T*D, sv2=D, sv3=(int64_t)H*T*D, sb1=T, sb2=1, sb3=(int64_t)H*T;
    cudaFuncAttributes attr = {}; CUDA_CHECK(cudaFuncGetAttributes(&attr, variant_kernel<C,STAGE,MB>));
    const size_t smem = STAGE ? (2*D*3 + 2*2)*sizeof(float) : 0;
    int occ = 0; CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, variant_kernel<C,STAGE,MB>, 32*NWARPS, smem));
    CUDA_CHECK(cudaFuncSetAttribute(variant_kernel<C,STAGE,MB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
    variant_kernel<C,STAGE,MB><<<dim3(H,1,gridz), dim3(32,NWARPS,1), smem>>>(d_q,d_k,d_v,d_g,d_b,d_sin,d_out,d_sout,H,t_tokens,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,1.0f/sqrtf((float)D));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_ochk,d_out,sz_v*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_schk,d_sout,sz_s*4,cudaMemcpyDeviceToHost));
    double mo=0,mr=0,ms=0,msr=0;
    for (size_t i=0;i<(size_t)H*t_tokens*D;i++){ mo=fmax(mo,fabs(h_ochk[i]-h_oref[i])); mr=fmax(mr,fabs(h_oref[i])); }
    for (size_t i=0;i<sz_s;i++){ ms=fmax(ms,fabs(h_schk[i]-h_sref[i])); msr=fmax(msr,fabs(h_sref[i])); }
    cudaEvent_t e0,e1; CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    const int reps = 30;
    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i=0;i<reps;i++) variant_kernel<C,STAGE,MB><<<dim3(H,1,gridz), dim3(32,NWARPS,1), smem>>>(d_q,d_k,d_v,d_g,d_b,d_sin,d_out,d_sout,H,t_tokens,sq1,sq2,sq3,sv1,sv2,sv3,sb1,sb2,sb3,1.0f/sqrtf((float)D));
    CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
    float tt; CUDA_CHECK(cudaEventElapsedTime(&tt,e0,e1));
    const double ms_t = tt/reps;
    if (t_base <= 0) t_base = ms_t;
    printf("  %-16s : %7.3f ms  (%5.2fx vs first)  regs=%3d  occ=%2d blk (%2d warps)  out_rel=%.2e st_rel=%.2e\n",
           nm, ms_t, t_base/ms_t, attr.numRegs, occ, occ*NWARPS, mo/(mr+1e-30), ms/(msr+1e-30));
}

int main(int argc, char ** argv) {
    printf("T03 V5 smem-staging (H=%d, D=%d, T=%d)\n", H, D, T);
    const int t_tokens = (argc > 1) ? atoi(argv[1]) : T;
    const size_t sz_qk=(size_t)T*H*D, sz_v=(size_t)H*T*D, sz_gb=(size_t)H*T, sz_s=(size_t)H*D*D;
    float *h_q=(float*)malloc(sz_qk*4),*h_k=(float*)malloc(sz_qk*4),*h_v=(float*)malloc(sz_v*4);
    float *h_g=(float*)malloc(sz_gb*4),*h_b=(float*)malloc(sz_gb*4);
    float *h_sin=(float*)malloc(sz_s*4),*h_sref=(float*)malloc(sz_s*4),*h_schk=(float*)malloc(sz_s*4);
    float *h_oref=(float*)malloc(sz_v*4),*h_ochk=(float*)malloc(sz_v*4);
    srand(1234);
    for (size_t i=0;i<sz_qk;i++){h_q[i]=((rand()/(float)RAND_MAX)-0.5f)*2.f/sqrtf((float)D);h_k[i]=((rand()/(float)RAND_MAX)-0.5f)*2.f/sqrtf((float)D);}
    for (size_t i=0;i<sz_v;i++) h_v[i]=((rand()/(float)RAND_MAX)-0.5f)*2.f;
    for (size_t i=0;i<sz_gb;i++){h_g[i]=-((rand()/(float)RAND_MAX))*0.5f;h_b[i]=0.1f+0.9f*(rand()/(float)RAND_MAX);}
    for (size_t i=0;i<sz_s;i++) h_sin[i]=((rand()/(float)RAND_MAX)-0.5f)*0.2f;
    float *d_q,*d_k,*d_v,*d_g,*d_b,*d_sin,*d_sout,*d_out;
    CUDA_CHECK(cudaMalloc(&d_q,sz_qk*4)); CUDA_CHECK(cudaMalloc(&d_k,sz_qk*4)); CUDA_CHECK(cudaMalloc(&d_v,sz_v*4));
    CUDA_CHECK(cudaMalloc(&d_g,sz_gb*4)); CUDA_CHECK(cudaMalloc(&d_b,sz_gb*4));
    CUDA_CHECK(cudaMalloc(&d_sin,sz_s*4)); CUDA_CHECK(cudaMalloc(&d_sout,sz_s*4)); CUDA_CHECK(cudaMalloc(&d_out,sz_v*4));
    CUDA_CHECK(cudaMemcpy(d_q,h_q,sz_qk*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_k,h_k,sz_qk*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v,h_v,sz_v*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_g,h_g,sz_gb*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b,h_b,sz_gb*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(d_sin,h_sin,sz_s*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,sz_s*4,cudaMemcpyDeviceToDevice));
    ref_kernel<<<H,D>>>(d_q,d_k,d_v,d_g,d_b,d_sout,d_out,d_sout,t_tokens,1.0f/sqrtf((float)D));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_oref,d_out,sz_v*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sref,d_sout,sz_s*4,cudaMemcpyDeviceToHost));
    double tb = 0;
    run_cfg<1,false,2>("C=1 base MB=2", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<1,true ,2>("C=1 stage MB=2", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<1,true ,4>("C=1 stage MB=4", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<1,true ,8>("C=1 stage MB=8", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<2,true ,2>("C=2 stage MB=2", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<2,true ,6>("C=2 stage MB=6", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    return 0;
}
