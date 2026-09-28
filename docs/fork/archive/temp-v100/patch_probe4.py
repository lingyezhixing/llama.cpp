p = r'D:\LLM\Backend\src\llama.cpp-my\tests\test-t32-smoke.cpp'
s = open(p, encoding='utf-8').read()

old = """static std::vector<llama_token> gen_branch(llama_model * model, const common_params & params,
                                           const llama_tokens & P, const llama_tokens & C) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = 1;"""
new = """static std::vector<llama_token> gen_branch(llama_model * model, const common_params & params,
                                           const llama_tokens & P, const llama_tokens & C, int n_seq_max = 1) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = n_seq_max;"""
assert old in s
s = s.replace(old, new)

old = """            LOG("item1b branch %d vs single-seq GT: ref=%zu shared/il=%zu shared/seq=%zu shared/priv-ckpt=%zu (n=%zu)\\n",
                    k, dr, ds, dq, dd, base.size());"""
new = """            // config probe: same single-seq decode but in a context with n_seq_max=4 (only seq0 used)
            auto base4 = gen_branch(model, params, P, C[k], 4);
            size_t d4 = 0;
            if (base4.size() == base.size()) {
                while (d4 < base.size() && base4[d4] == base[d4]) { d4++; }
            } else {
                d4 = 999;
            }
            LOG("item1b branch %d vs single-seq GT: ref=%zu shared/il=%zu shared/seq=%zu shared/priv-ckpt=%zu | n_seq_max=4-solo=%zu (n=%zu)\\n",
                    k, dr, ds, dq, dd, d4, base.size());"""
assert old in s
s = s.replace(old, new)

open(p, 'w', encoding='utf-8').write(s)
print('patched ok')
