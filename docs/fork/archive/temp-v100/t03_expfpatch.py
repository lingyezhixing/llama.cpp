import io, re

p = r'<TEMP>\v100\t03_sweep.cu'
s = io.open(p, encoding='utf-8').read()

s = s.replace('template <int C, int MB>\n__global__ void __launch_bounds__(32*NWARPS, MB)',
              'template <int C, int MB, int EXPMODE = 0>\n__global__ void __launch_bounds__(32*NWARPS, MB)')
s = s.replace('        const float g_val = expf(g[gb_base + t*sb2]);',
              '        const float g_raw = g[gb_base + t*sb2];\n        const float g_val = (EXPMODE == 0) ? expf(g_raw) : (EXPMODE == 1 ? __expf(g_raw) : g_raw);')
s = s.replace('template <int C, int MB>\nvoid run_cfg(', 'template <int C, int MB, int EXPMODE = 0>\nvoid run_cfg(')
s = s.replace('cudaFuncGetAttributes(&attr, variant_kernel<C,MB>)', 'cudaFuncGetAttributes(&attr, variant_kernel<C,MB,EXPMODE>)')
s = s.replace('cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, variant_kernel<C,MB>, 32*NWARPS, 0)', 'cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, variant_kernel<C,MB,EXPMODE>, 32*NWARPS, 0)')
s = s.replace('variant_kernel<C,MB><<<', 'variant_kernel<C,MB,EXPMODE><<<')

# replace the run list with an expf-focused list
start = s.index('    double tb = 0;')
end = s.index('    return 0;\n}')
newlist = '''    double tb = 0;
    run_cfg<1, 2, 0>("C=1 expf      MB=2", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<1, 2, 1>("C=1 __expf    MB=2", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<1, 8, 1>("C=1 __expf    MB=8", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<2, 8, 1>("C=2 __expf    MB=8", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<4, 6, 1>("C=4 __expf    MB=6", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<1, 2, 2>("C=1 NO-expf   MB=2 (ablation)", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
    run_cfg<2, 8, 2>("C=2 NO-expf   MB=8 (ablation)", d_sout,d_out,d_sin,d_q,d_k,d_v,d_g,d_b,h_oref,h_sref,h_ochk,h_schk,sz_s,sz_v,t_tokens,tb);
'''
s = s[:start] + newlist + s[end:]
io.open(p, 'w', encoding='utf-8', newline='').write(s)
print('expf ablation patch applied')
