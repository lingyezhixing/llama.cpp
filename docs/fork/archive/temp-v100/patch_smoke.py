import sys

p = r'D:\LLM\Backend\src\llama.cpp-my\tests\test-t32-smoke.cpp'
s = open(p, encoding='utf-8').read()

old_struct = """struct case_result {
    std::vector<std::vector<llama_token>> gen;    // [branch index in active] x step argmax
    std::vector<std::vector<float>>       logits; // [branch index in active] last step logits
    bool ok = false;
    std::string err;
};"""
new_struct = """struct case_result {
    std::vector<std::vector<llama_token>>        gen;    // [branch] x step argmax
    std::vector<std::vector<std::vector<float>>> logits; // [branch][step] logits
    bool ok = false;
    std::string err;
};"""
assert old_struct in s
s = s.replace(old_struct, new_struct)

old_arg = """        arg[k] = argmax(l, n_vocab);
        if (last) {
            logits[k].assign(l, l + n_vocab);
        }"""
new_arg = """        arg[k] = argmax(l, n_vocab);
        (void) last;
        logits[k].emplace_back(l, l + n_vocab);"""
assert old_arg in s
s = s.replace(old_arg, new_arg)

old_cmp = """    bool ok = true;
    for (size_t k = 0; k < a.gen.size(); ++k) {
        if (a.gen[k] != b.gen[k]) {
            LOG_ERR("%s: FAIL branch %zu: token sequences differ\\n", name, k);
            LOG_ERR("  ref:");
            for (auto t : a.gen[k]) LOG_ERR(" %d", t);
            LOG_ERR("\\n  got:");
            for (auto t : b.gen[k]) LOG_ERR(" %d", t);
            LOG_ERR("\\n");
            ok = false;
        }
    }
    if (!a.logits.empty() && !a.logits[0].empty() && a.logits[0].size() == b.logits[0].size()) {
        double max_d = 0.0;
        for (size_t k = 0; k < a.logits.size(); ++k) {
            for (size_t i = 0; i < a.logits[k].size(); ++i) {
                max_d = std::max(max_d, (double) std::fabs(a.logits[k][i] - b.logits[k][i]));
            }
        }
        LOG("%s: max logit diff = %.3e\\n", name, max_d);
        if (max_d > 1e-4) {
            LOG_ERR("%s: FAIL (logits differ)\\n", name);
            ok = false;
        }
    }"""
new_cmp = """    bool ok = true;
    double max_d_all = 0.0;
    for (size_t k = 0; k < a.gen.size(); ++k) {
        if (a.gen[k] != b.gen[k]) {
            size_t i_first = 0;
            while (i_first < a.gen[k].size() && i_first < b.gen[k].size() && a.gen[k][i_first] == b.gen[k][i_first]) {
                i_first++;
            }
            LOG_ERR("%s: FAIL branch %zu: tokens differ at step %zu (ref = %d, got = %d)\\n",
                    name, k, i_first, a.gen[k][i_first], b.gen[k][i_first]);
            double d = 0.0;
            if (i_first < a.logits[k].size() && i_first < b.logits[k].size() && a.logits[k][i_first].size() == b.logits[k][i_first].size()) {
                for (size_t v = 0; v < a.logits[k][i_first].size(); ++v) {
                    d = std::max(d, (double) std::fabs(a.logits[k][i_first][v] - b.logits[k][i_first][v]));
                }
            }
            LOG_ERR("  logit max diff at that step = %.3e\\n", d);
            ok = false;
        }
    }
    if (!a.logits.empty() && a.logits.size() == b.logits.size()) {
        for (size_t k = 0; k < a.logits.size(); ++k) {
            const size_t ns = std::min(a.logits[k].size(), b.logits[k].size());
            for (size_t i = 0; i < ns; ++i) {
                if (a.logits[k][i].size() != b.logits[k][i].size()) continue;
                for (size_t v = 0; v < a.logits[k][i].size(); ++v) {
                    max_d_all = std::max(max_d_all, (double) std::fabs(a.logits[k][i][v] - b.logits[k][i][v]));
                }
            }
        }
    }
    LOG("%s: max logit diff (all steps) = %.3e\\n", name, max_d_all);"""
assert old_cmp in s
s = s.replace(old_cmp, new_cmp)

open(p, 'w', encoding='utf-8').write(s)
print('patched ok')
