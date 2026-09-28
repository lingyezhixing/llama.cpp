p = r'D:\LLM\Backend\src\llama.cpp-my\tests\test-t32-smoke.cpp'
s = open(p, encoding='utf-8').read()

anchor = 'static bool cmp_case(const char * name, const case_result & a, const case_result & b) {'
assert anchor in s

fork_code = r'''
// item5/6: fork an existing sequence at a prefix boundary (seq_cp + optional partial-state restore)
static std::vector<llama_token> gen_branch(llama_model * model, const common_params & params,
                                           const llama_tokens & P, const llama_tokens & C) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;
    auto ctx = llama_context_ptr{llama_init_from_model(model, cparams)};
    std::vector<llama_token> gen;
    if (!ctx) { return gen; }
    if (!decode_tokens(ctx.get(), P, 0, 0)) { return gen; }
    for (size_t j = 0; j < C.size(); ++j) {
        std::vector<llama_token> arg;
        std::vector<std::vector<std::vector<float>>> lg;
        if (!step_batch(ctx.get(), { C[j] }, {0}, (llama_pos) P.size() + (llama_pos) j, false, arg, lg)) {
            gen.clear();
            return gen;
        }
        gen.push_back(arg[0]);
    }
    return gen;
}

static bool fork_case(llama_model * model, const common_params & params, const llama_tokens & P,
                      const llama_tokens & C0, const llama_tokens & C1, bool ext,
                      std::vector<llama_token> & gen0, std::vector<llama_token> & gen1, std::string & err) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 2;
    auto ctx = llama_context_ptr{llama_init_from_model(model, cparams)};
    if (!ctx) { err = "ctx init"; return false; }
    auto * mem = llama_get_memory(ctx.get());
    const int n_pfx = (int) P.size();
    const int n_pre = ext ? 0 : 16;

    if (!decode_tokens(ctx.get(), P, 0, 0)) { err = "prefill"; return false; }

    std::vector<uint8_t> ckpt;
    if (!ext) {
        const size_t sz = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        ckpt.resize(sz);
        const size_t n = llama_state_seq_get_data_ext(ctx.get(), ckpt.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n != sz) { err = "ckpt save"; return false; }
    }

    gen0.clear();
    for (int j = 0; j < n_pre; ++j) {
        std::vector<llama_token> arg;
        std::vector<std::vector<std::vector<float>>> lg;
        if (!step_batch(ctx.get(), { C0[j] }, {0}, n_pfx + j, false, arg, lg)) { err = "seq0 pre"; return false; }
        gen0.push_back(arg[0]);
    }

    // fork seq0 -> seq1 over the shared prefix [0, n_pfx)
    if (!llama_memory_seq_rm(mem, 1, -1, -1)) { err = "rm dst"; return false; }
    llama_memory_seq_cp(mem, 0, 1, 0, n_pfx);
    if (!ext) {
        const size_t n = llama_state_seq_set_data_ext(ctx.get(), ckpt.data(), ckpt.size(), 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (n != ckpt.size()) { err = "ckpt restore"; return false; }
    }

    gen1.clear();
    for (size_t j = 0; j < C1.size(); ++j) {
        std::vector<llama_token> arg;
        std::vector<std::vector<std::vector<float>>> lg;
        if (!step_batch(ctx.get(), { C1[j] }, {1}, n_pfx + (llama_pos) j, false, arg, lg)) { err = "seq1 dec"; return false; }
        gen1.push_back(arg[0]);
    }
    for (size_t j = n_pre; j < C0.size(); ++j) {
        std::vector<llama_token> arg;
        std::vector<std::vector<std::vector<float>>> lg;
        if (!step_batch(ctx.get(), { C0[j] }, {0}, n_pfx + (llama_pos) j, false, arg, lg)) { err = "seq0 post"; return false; }
        gen0.push_back(arg[0]);
    }
    return true;
}
'''
s = s.replace(anchor, fork_code + '\n' + anchor)

# wire into main: after item3, add item5 (ext) and item6 (ckpt)
old = """    LOG("\\n%s\\n", ok ? "SMOKE PASS" : "SMOKE FAIL");"""
new = """    // item 5/6: fork primitive (extension case and checkpoint case)
    for (int ext = 1; ext >= 0; --ext) {
        auto expect0 = gen_branch(model, params, P, C[0]);
        auto expect1 = gen_branch(model, params, P, C[1]);
        std::vector<llama_token> got0, got1;
        std::string err;
        const bool r = fork_case(model, params, P, C[0], C[1], ext == 1, got0, got1, err);
        bool okf = r && expect0.size() == got0.size() && expect1.size() == got1.size();
        if (okf) {
            for (size_t i = 0; i < expect0.size(); ++i) { if (expect0[i] != got0[i]) { okf = false; } }
            for (size_t i = 0; i < expect1.size(); ++i) { if (expect1[i] != got1[i]) { okf = false; } }
        }
        if (!r) { LOG_ERR("item%s fork: error: %s\\n", ext == 1 ? "5-ext" : "6-ckpt", err.c_str()); }
        LOG("item%s fork (src unpolluted + dst correct): %s\\n", ext == 1 ? "5-ext" : "6-ckpt", okf ? "PASS" : "FAIL");
        ok &= okf;
    }

    LOG("\\n%s\\n", ok ? "SMOKE PASS" : "SMOKE FAIL");"""
assert old in s
s = s.replace(old, new)

open(p, 'w', encoding='utf-8').write(s)
print('patched ok')
