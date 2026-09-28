import sqlite3, os

def classify(nm):
    if 'cutlass::Kernel' in nm or 'splitKreduce' in nm: return 'GEMM'
    if 'dequantize' in nm: return 'dequant'
    if 'gated_delta_net' in nm: return 'GDN'
    if 'flash_attn' in nm: return 'attention'
    return 'other'

def window(path, label, w0_ms, w1_ms):
    con = sqlite3.connect(path)
    t0 = con.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    lo, hi = t0 + int(w0_ms * 1e6), t0 + int(w1_ms * 1e6)
    rows = con.execute(
        "select (select value from StringIds where id = k.demangledName) nm, k.start, k.end "
        "from CUPTI_ACTIVITY_KIND_KERNEL k where k.start >= ? and k.end <= ?", (lo, hi)).fetchall()
    tot = sum(e - s for _, s, e in rows)
    agg, per_kernel = {}, {}
    for nm, s, e in rows:
        c = classify(nm or '')
        agg[c] = agg.get(c, 0) + (e - s)
        if c == 'other':
            short = (nm or '?').split('<')[0].strip()
            per_kernel[short] = per_kernel.get(short, 0) + (e - s)
    print('=' * 84)
    print(f"[{label}] rel window {w0_ms:.0f}-{w1_ms:.0f} ms -> GPU kernels {tot/1e6:.1f} ms ({len(rows)} launches)")
    print('=' * 84)
    for c, ns in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {c:<10} {ns/1e6:>9.1f} ms  {100*ns/tot:>5.1f}%")
    print('  -- other breakdown --')
    for k, ns in sorted(per_kernel.items(), key=lambda x: -x[1])[:12]:
        print(f"     {k:<34} {ns/1e6:>8.1f} ms  {100*ns/tot:>5.1f}%")
    con.close()

T = os.path.join(os.environ['TEMP'], 'v100')
window(os.path.join(T, 't10_pp4096_d32k.sqlite'), 'pp4096@depth32k measured eval', 47020, 53337)
window(os.path.join(T, 't10_pp32768.sqlite'), 'pp32768 forward#1 (32768 tok)', 41731, 83569)
window(os.path.join(T, 't10_pp4096_d0.sqlite'), 'pp4096 d=0 forward#1', 4400, 8788)
