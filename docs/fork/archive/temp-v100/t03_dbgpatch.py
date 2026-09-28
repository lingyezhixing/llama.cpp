import io

p = r'D:\LLM\Backend\src\llama.cpp-my\ggml\src\ggml-cuda\gated_delta_net.cu'
s = io.open(p, encoding='utf-8').read()
old = '''            static const bool gdn_c2_disabled = getenv("GGML_CUDA_GDN_C2") != nullptr;
            if (vec4 && !KDA && !keep_rs_t && !gdn_c2_disabled) {'''
new = '''            static const bool gdn_c2_disabled = getenv("GGML_CUDA_GDN_C2") != nullptr;
            if (getenv("GGML_CUDA_GDN_C2_DEBUG") != nullptr) {
                static int dbg_calls = 0;
                if (dbg_calls++ < 3) {
                    cudaFuncAttributes attr = {};
                    cudaError_t ea = cudaFuncGetAttributes(&attr, gated_delta_net_c2_cuda<true>);
                    cudaError_t es = cudaGetLastError();
                    fprintf(stderr, "[c2dbg] attr=%s numRegs=%d maxThreads=%d shmemStatic=%zu; stale_err=%s; H=%d n_tok=%d n_seqs=%d S_v=%d num_warps=%d vec4=%d kda=%d keep_rs=%d\\n",
                            cudaGetErrorString(ea), attr.numRegs, attr.maxThreadsPerBlock, attr.sharedSizeBytes,
                            cudaGetErrorString(es), (int) H, (int) n_tokens, (int) n_seqs, (int) S_v, num_warps, (int) vec4, (int) KDA, (int) keep_rs_t);
                }
            }
            if (vec4 && !KDA && !keep_rs_t && !gdn_c2_disabled) {'''
assert old in s
s = s.replace(old, new, 1)
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('debug added')
