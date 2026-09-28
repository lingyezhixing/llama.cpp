import io

p = r'<TEMP>\v100\t03_variant.cu'
s = io.open(p, encoding='utf-8').read()

# occupancy print + fine-grained phase instrumentation inside variant_kernel (thread 0 of warp 0, block 0)
s = s.replace('''    float s[RPC];
    {''', '''    const bool meas = (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.z == 0);
    unsigned long long c_ld=0, c_kv=0, c_upd=0;
    float s[RPC];
    {''')
s = s.replace('''    for (int t = 0; t < n_tokens; t++) {
        float kk[RPC], qq[RPC]; float g_raw, b_val, v_val;''',
'''    for (int t = 0; t < n_tokens; t++) {
        unsigned long long a0 = meas ? clock64() : 0ull;
        float kk[RPC], qq[RPC]; float g_raw, b_val, v_val;''')
s = s.replace('''        float kv = 0.f;
        #pragma unroll
        for (int r = 0; r < RPC; r++) kv += s[r]*kk[r];''',
'''        unsigned long long a1 = meas ? clock64() : 0ull;
        float kv = 0.f;
        #pragma unroll
        for (int r = 0; r < RPC; r++) kv += s[r]*kk[r];''')
s = s.replace('''        const float g_val = expf(g_raw);
        const float delta = (v_val - g_val*kv)*b_val;''',
'''        const float g_val = expf(g_raw);
        const float delta = (v_val - g_val*kv)*b_val;
        unsigned long long a2 = meas ? clock64() : 0ull;''')
s = s.replace('''        if (sub == 0) out[((t*H_ + h)*S_v) + col] = at*scale;
    }''',
'''        if (sub == 0) out[((t*H_ + h)*S_v) + col] = at*scale;
        if (meas) { unsigned long long a3 = clock64(); c_ld += a1-a0; c_kv += a2-a1; c_upd += a3-a2; }
    }''')
s = s.replace('''    float * so = s_out + (size_t)h*S_v*S_v + col*S_v + sub*RPC;''',
'''    if (meas && blockIdx.x == 0 && blockIdx.z == 0) {
        printf("  [C=%d fine] loads=%llu/tok  kv_local+reduce1+expf=%llu/tok  upd+attn_reduce2+store=%llu/tok\\n",
               C, c_ld/n_tokens, c_kv/n_tokens, c_upd/n_tokens);
    }
    float * so = s_out + (size_t)h*S_v*S_v + col*S_v + sub*RPC;''')

# occupancy print in run_variant
s = s.replace('''    const size_t szs = sz_s;
    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,szs*4,cudaMemcpyDeviceToDevice));''',
'''    const size_t szs = sz_s;
    {
        int occ = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, variant_kernel<C,PF>, 32*NWARPS, 0);
        int nreg = 0; cudaFuncGetAttributes((cudaFuncAttributes*)nullptr, variant_kernel<C,PF>);
        printf("  [C=%d pf=%d] occupancy=%d block/SM (%d warps/SM), grid=(%d,%d,%d)\\n", C, (int)PF, occ, occ*NWARPS, H, 1, S_v/(NWARPS*C));
    }
    CUDA_CHECK(cudaMemcpy(d_sout,d_sin,szs*4,cudaMemcpyDeviceToDevice));''')
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('instrumented')
