import sqlite3, os

T = os.path.join(os.environ['TEMP'], 'v100')
con = sqlite3.connect(os.path.join(T, 't10_d32k_ub2048.sqlite'))
t0 = con.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
span = con.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
print(f'profile span = {(span[1]-span[0])/1e6:.1f} ms')

# GEMM launches (400 per ubatch) -> last 800 define the last 2 ubatches (the measured eval)
g = con.execute(
    "select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
    "where (select value from StringIds where id=k.demangledName) like '%cutlass::Kernel%' order by k.start").fetchall()
lo = g[-800][0]
hi = span[1]
print(f'eval window = [{(lo-t0)/1e6:.1f}, {(hi-t0)/1e6:.1f}] ms rel  (last 800 GEMM launches)')

rows = con.execute(
    "select (select value from StringIds where id=k.demangledName) nm, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
    "where k.start >= ? and k.end <= ?", (lo, hi)).fetchall()
tot = sum(e - s for _, s, e in rows)
def cl(nm):
    if 'cutlass::Kernel' in nm or 'splitKreduce' in nm: return 'GEMM'
    if 'dequantize' in nm: return 'dequant'
    if 'gated_delta_net' in nm: return 'GDN'
    if 'flash_attn' in nm: return 'attention'
    return 'other'
agg = {}
for nm, s, e in rows:
    c = cl(nm or '')
    agg[c] = agg.get(c, 0) + (e - s)
print(f'eval GPU kernels = {tot/1e6:.1f} ms ({len(rows)} launches)')
for c, ns in sorted(agg.items(), key=lambda x: -x[1]):
    print(f'  {c:<10} {ns/1e6:>9.1f} ms  {100*ns/tot:>5.1f}%')

fl = 2 * 16 * 4 * 256 * 24 * (2048 * 32768 + 2048 * 2049 // 2 + 2048 * 34816 + 2048 * 2049 // 2)
att = agg.get('attention', 0)
print(f'attention TFLOPS (useful causal) = {fl/1e12:.2f} TFLOP / {att/1e9:.3f} s = {fl/att:.1f} TF/s')
print(f'attention per 4096-token eval = {att/1e6:.0f} ms ; ub512 case was 1899 ms')

# per-kernel-type names for the two attention kernels
rows2 = con.execute(
    "select (select value from StringIds where id=k.demangledName) nm, k.gridX, k.blockX*k.blockY, count(*), sum(k.end-k.start) "
    "from CUPTI_ACTIVITY_KIND_KERNEL k "
    "where (select value from StringIds where id=k.demangledName) like '%flash_attn%' group by 1,2,3").fetchall()
print()
for nm, gx, nthreads, n, ns in rows2:
    print(f'  gridX={gx} thr={nthreads} n={n} tot={ns/1e6:.1f}ms  {nm[:110]}')
con.close()
