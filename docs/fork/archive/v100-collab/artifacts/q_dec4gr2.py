import sqlite3, os, collections

T = os.path.join(os.environ['TEMP'], 'v100')
c = sqlite3.connect(os.path.join(T, 'dec4gr.sqlite'))
mn = c.execute('select min(start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]

# pick a steady-state token window: use the MMVQ<14,1,1> marker to find token starts
rows = [s for (s,) in c.execute("""select start from CUPTI_ACTIVITY_KIND_KERNEL k
        where (select value from StringIds where id=k.demangledName) like '%mul_mat_vec_q<(ggml_type)14, (int)1, (bool)1%'
        order by start""")]
starts = []
prev = None
for s in rows:
    if prev is None or s - prev > 8_000_000:
        starts.append(s)
    prev = s
print('token starts (ms):', ['%.1f' % ((s-mn)/1e6) for s in starts])

# analyze the 3rd graph token
k = 2 if len(starts) > 3 else 1
w0, w1 = starts[k], starts[k+1]
print('\n=== token window %.1f - %.1f ms (%.2f ms) ===' % ((w0-mn)/1e6, (w1-mn)/1e6, (w1-w0)/1e6))

rows = c.execute("""select (select value from StringIds where id=k.demangledName) nm, count(*), sum(k.end-k.start)
                    from CUPTI_ACTIVITY_KIND_KERNEL k where k.start >= ? and k.start < ? group by nm order by 3 desc""",
                 (w0, w1)).fetchall()
tot = sum(r[2] for r in rows)
print('kernels=%d  total busy=%.2f ms' % (sum(r[1] for r in rows), tot/1e6))
print('%-56s %6s %9s %8s %6s' % ('kernel', 'n', 'total ms', 'avg us', '%busy'))
for nm, cnt, ns in rows:
    print('%-56s %6d %9.2f %8.2f %5.1f%%' % ((nm or '?')[:56], cnt, ns/1e6, ns/1e3/cnt, 100.0*ns/tot))

# aggregate by category
cats = {
    'MMVQ (all)': lambda s: 'mul_mat_vec_q' in s,
    'quantize_q8_1': lambda s: 'quantize_q8_1' in s,
    'rms_norm': lambda s: 'rms_norm' in s,
    'scale_f32': lambda s: 'scale_f32' in s,
    'gated_delta_net': lambda s: 'gated_delta_net' in s,
    'flash_attn': lambda s: 'flash_attn' in s,
    'get_rows': lambda s: 'k_get_rows' in s,
    'set_rows': lambda s: 'set_rows' in s,
    'elementwise(add/cpy/silu/concat/softplus/sigmoid/rope/fwht)': lambda s: any(x in s for x in
        ('k_bin_bcast', 'cpy_scalar', 'unary_gated_op', 'concat_cont', 'rope_', 'fwht_', 'unary_op_kernel')),
}
print('\n--- categories ---')
for cname, f in cats.items():
    cnt = sum(r[1] for r in rows if f(r[0] or ''))
    ns = sum(r[2] for r in rows if f(r[0] or ''))
    print('%-58s n=%5d %8.2f ms  %5.1f%%' % (cname, cnt, ns/1e6, 100.0*ns/tot))
