p = r'D:\LLM\Backend\src\llama.cpp-my\tests\test-t32-smoke.cpp'
s = open(p, encoding='utf-8').read()

anchor = """static case_result run_case(llama_model * model, const common_params & params, const llama_tokens & P,
                            const std::vector<llama_tokens> & C, const llama_tokens & D0, const case_cfg & cfg) {"""
assert anchor in s

helper = r'''
// shared attention cells, but a private recurrent state per branch (partial state copied after prefill)
static case_result run_case_shared_private(llama_model * model, const common_params & params, const llama_tokens & P,
                                           const std::vector<llama_tokens> & C, const case_cfg & cfg) {
    case_result res;
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = cfg.n_seq;
    auto ctx = llama_context_ptr{llama_init_from_model(model, cparams)};
    if (!ctx) { res.err = "context init failed"; return res; }
    auto * mem = llama_get_memory(ctx.get());
    const int n_pfx = (int) P.size();
    const int n = (int) cfg.active.size();

    if (!decode_tokens(ctx.get(), P, 0, 0)) { res.err = "prefill failed"; return res; }

    const size_t sz = llama_state_seq_get_size_ext(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    std::vector<uint8_t> state0(sz);
    if (llama_state_seq_get_data_ext(ctx.get(), state0.data(), sz, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != sz) {
        res.err = "state save failed";
        return res;
    }

    for (int k = 1; k < cfg.n_seq; ++k) {
        if (!llama_memory_seq_rm(mem, k, -1, -1)) { res.err = "seq_rm failed"; return res; }
        llama_memory_seq_cp(mem, 0, k, 0, n_pfx);
        if (llama_state_seq_set_data_ext(ctx.get(), state0.data(), sz, k, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != sz) {
            res.err = "state restore failed";
            return res;
        }
    }

    res.gen.assign(n, {});
    res.logits.assign(n, {});
    for (int j = 0; j < cfg.n_steps; ++j) {
        std::vector<llama_token> toks(n);
        for (int k = 0; k < n; ++k) {
            toks[k] = C[cfg.active[k]][j];
        }
        std::vector<llama_token> arg;
        if (!step_batch(ctx.get(), toks, cfg.active, n_pfx + j, j + 1 == cfg.n_steps, arg, res.logits)) {
            res.err = "step decode failed";
            return res;
        }
        for (int k = 0; k < n; ++k) {
            res.gen[k].push_back(arg[k]);
        }
    }
    res.ok = true;
    return res;
}
'''

s = s.replace(anchor, helper + '\n' + anchor)

old = """        // compare both interleaved paths against single-sequence ground truth (batch of 1)
        for (int k = 0; k < 4; ++k) {
            auto base = gen_branch(model, params, P, C[k]);
            bool okr = base.size() == a.gen[k].size();
            bool oks = base.size() == b.gen[k].size();
            bool okq = base.size() == c.gen[k].size();
            size_t dr = 0, ds = 0, dq = 0;
            if (okr) { while (dr < base.size() && base[dr] == a.gen[k][dr]) { dr++; } }
            if (oks) { while (ds < base.size() && base[ds] == b.gen[k][ds]) { ds++; } }
            if (okq) { while (dq < base.size() && base[dq] == c.gen[k][dq]) { dq++; } }
            LOG("item1b branch %d vs single-seq ground truth: ref first-diff=%zu, shared/interleaved=%zu, shared/sequential=%zu (n=%zu)\\n",
                    k, dr, ds, dq, base.size());
            if (dr != base.size() || ds != base.size() || dq != base.size()) {
                ok = false;
            }
        }"""
new = """        // compare all paths against single-sequence ground truth (batch of 1)
        auto d = run_case_shared_private(model, params, P, C, sh);   // shared cells + private recurrent, interleaved
        for (int k = 0; k < 4; ++k) {
            auto base = gen_branch(model, params, P, C[k]);
            bool okr = base.size() == a.gen[k].size();
            bool oks = base.size() == b.gen[k].size();
            bool okq = base.size() == c.gen[k].size();
            bool okd = d.ok && base.size() == d.gen[k].size();
            size_t dr = 0, ds = 0, dq = 0, dd = 0;
            if (okr) { while (dr < base.size() && base[dr] == a.gen[k][dr]) { dr++; } }
            if (oks) { while (ds < base.size() && base[ds] == b.gen[k][ds]) { ds++; } }
            if (okq) { while (dq < base.size() && base[dq] == c.gen[k][dq]) { dq++; } }
            if (okd) { while (dd < base.size() && base[dd] == d.gen[k][dd]) { dd++; } }
            LOG("item1b branch %d vs single-seq GT: ref=%zu shared/il=%zu shared/seq=%zu shared/priv-ckpt=%zu (n=%zu)\\n",
                    k, dr, ds, dq, dd, base.size());
            if (dr != base.size() || ds != base.size() || dq != base.size() || dd != base.size() || !d.ok) {
                ok = false;
                if (!d.ok) { LOG_ERR("  shared/priv-ckpt error: %s\\n", d.err.c_str()); }
            }
        }
        // is the shared-private path identical to the independent-prefill path?
        {
            bool same = d.ok && d.gen.size() == a.gen.size();
            if (same) {
                for (size_t k = 0; k < a.gen.size(); ++k) {
                    if (d.gen[k] != a.gen[k]) { same = false; }
                }
            }
            LOG("item1c shared-private vs ref (both interleaved): %s\\n", same ? "IDENTICAL" : "differs");
        }"""
assert old in s
s = s.replace(old, new)

open(p, 'w', encoding='utf-8').write(s)
print('patched ok')
