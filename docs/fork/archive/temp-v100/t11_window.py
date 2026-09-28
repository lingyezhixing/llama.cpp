import sqlite3, os

T = os.path.join(os.environ['TEMP'], 'v100')

def cl(nm):
    if 'cutlass::Kernel' in nm or 'splitKreduce' in nm: return 'GEMM'
    if 'dequantize' in nm: return 'dequant'
    if 'gated_delta_net' in nm: return 'GDN'
    if 'flash_attn_ext_f16' in nm: return 'attention'
    if 'flash_attn_stream_k_fixup' in nm: return 'attn_fixup'
    return 'other'

def window(path, label, w0, w1):
    con = sqlite3.connect(path)
    t0 = con.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    lo, hi = t0 + int(w0*1e6), t0 + int(w1*1e6)
    rows = con.execute("select (select value from StringIds where id=k.demangledName) nm, k.start, k.end "
                       "from CUPTI_ACTIVITY_KIND_KERNEL k where k.start >= ? and k.end <= ?", (lo, hi)).fetchall()
    tot = sum(e - s for _, s, e in rows)
    agg = {}
    for nm, s, e in rows:
        c = cl(nm or '')
        agg[c] = agg.get(c, 0) + (e - s)
    print(f'[{label}] window {w0}-{w1} ms -> {tot/1e6:.1f} ms ({len(rows)} launches)')
    for c, ns in sorted(agg.items(), key=lambda x: -x[1]):
        print(f'   {c:<11} {ns/1e6:>9.1f} ms  {100*ns/tot:>5.1f}%')
    con.close()

window(os.path.join(T, 't10_pp4096_d32k.sqlite'), 'BEFORE (192 CTA, 3 waves)', 47020, 53337)
window(os.path.join(T, 't11_kvsplit_d32k.sqlite'), 'AFTER  (stream-K, 80 CTA 1 wave)', 46380, 52200)
