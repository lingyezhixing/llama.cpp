p = r'D:\LLM\Backend\src\llama.cpp-my\tests\test-t32-smoke.cpp'
s = open(p, encoding='utf-8').read()

# 1) add cfg flag
old = """    int              rollback    = 0;     // rollback seq0 by N positions after the loop
    int              n_steps2    = 0;     // second phase steps on seq0
};"""
new = """    int              rollback    = 0;     // rollback seq0 by N positions after the loop
    int              n_steps2    = 0;     // second phase steps on seq0
    bool             sequential  = false; // decode one seq at a time (batch of 1) instead of interleaved
};"""
assert old in s
s = s.replace(old, new)

# 2) handle sequential in the run_case loop
old = """    for (int j = 0; j < cfg.n_steps; ++j) {
        std::vector<llama_token> toks(n);
        for (int k = 0; k < n; ++k) {
            toks[k] = C[cfg.active[k]][j];
        }
        std::vector<llama_token> arg;
        if (!step_batch(ctx.get(), toks, cfg.active, n_pfx + j, j + 1 == cfg.n_steps, arg, res.logits)) {
            res.err = "step decode failed at " + std::to_string(j);
            return res;
        }
        for (int k = 0; k < n; ++k) {
            res.gen[k].push_back(arg[k]);
        }
    }"""
new = """    if (cfg.sequential) {
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < cfg.n_steps; ++j) {
                std::vector<llama_token> arg;
                std::vector<std::vector<std::vector<float>>> lg;
                if (!step_batch(ctx.get(), { C[cfg.active[k]][j] }, { cfg.active[k] }, n_pfx + j, false, arg, lg)) {
                    res.err = "seq step decode failed";
                    return res;
                }
                res.gen[k].push_back(arg[0]);
            }
        }
    } else {
        for (int j = 0; j < cfg.n_steps; ++j) {
            std::vector<llama_token> toks(n);
            for (int k = 0; k < n; ++k) {
                toks[k] = C[cfg.active[k]][j];
            }
            std::vector<llama_token> arg;
            if (!step_batch(ctx.get(), toks, cfg.active, n_pfx + j, j + 1 == cfg.n_steps, arg, res.logits)) {
                res.err = "step decode failed at " + std::to_string(j);
                return res;
            }
            for (int k = 0; k < n; ++k) {
                res.gen[k].push_back(arg[k]);
            }
        }
    }"""
assert old in s
s = s.replace(old, new)

# 3) extend item1 with ground-truth (single-seq, batch=1) comparison + shared-sequential case
old = """    // item 1: shared prefix vs independent prefills
    {
        case_cfg ref; ref.share = false; ref.active = {0, 1, 2, 3};
        case_cfg sh ; sh.share  = true;  sh.active  = {0, 1, 2, 3};
        auto a = run_case(model, params, P, C, D0, ref);
        auto b = run_case(model, params, P, C, D0, sh);
        ok &= cmp_case("item1 shared-prefix", a, b);
    }"""
new = """    // item 1: shared prefix vs independent prefills
    {
        case_cfg ref; ref.share = false; ref.active = {0, 1, 2, 3};
        case_cfg sh ; sh.share  = true;  sh.active  = {0, 1, 2, 3};
        case_cfg sq ; sq.share  = true;  sq.active  = {0, 1, 2, 3}; sq.sequential = true;
        auto a = run_case(model, params, P, C, D0, ref);
        auto b = run_case(model, params, P, C, D0, sh);
        auto c = run_case(model, params, P, C, D0, sq);
        ok &= cmp_case("item1 shared-prefix", a, b);

        // compare both interleaved paths against single-sequence ground truth (batch of 1)
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
        }
    }"""
assert old in s
s = s.replace(old, new)

open(p, 'w', encoding='utf-8').write(s)
print('patched ok')
