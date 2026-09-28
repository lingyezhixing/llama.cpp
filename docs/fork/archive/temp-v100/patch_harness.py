import io

p = r'<TEMP>\v100\t18_fa_harness.cu'
s = io.open(p, encoding='utf-8').read()

s = s.replace(
    "static constexpr int NQ = 512, NKV = 35072, NHEAD_Q = 24, NHEAD_KV = 4, GQA = NHEAD_Q/NHEAD_KV;",
    "static constexpr int NQ = 512, NHEAD_Q = 24, NHEAD_KV = 4, GQA = NHEAD_Q/NHEAD_KV;\nstatic int g_nkv = 35072;   // n_kv, set from argv[2]")
assert 'g_nkv' in s

# runtime NKV everywhere except the declaration
s = s.replace('NKV', 'g_nkv')
s = s.replace('static int g_g_nkv = 35072;', 'static int g_nkv = 35072;')

# main: parse argv[2]
s = s.replace('    const int iters = argc > 1 ? atoi(argv[1]) : 30;',
              '    const int iters = argc > 1 ? atoi(argv[1]) : 30;\n    g_nkv = argc > 2 ? atoi(argv[2]) : 35072;')

io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('g_nkv occurrences =', s.count('g_nkv'))
