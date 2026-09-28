import sqlite3, os

T = os.path.join(os.environ['TEMP'], 'v100')
path = os.path.join(T, 't10_d32k_ub2048.sqlite')
con = sqlite3.connect(path)

print('--- FA kernel configs (ub2048, pp4096@depth32k) ---')
for gx, gy, gz, bx, by, bz, n, ns in con.execute(
        "select k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,count(*),sum(k.end-k.start) "
        "from CUPTI_ACTIVITY_KIND_KERNEL k "
        "where (select value from StringIds where id=k.demangledName) like '%flash_attn%' "
        "group by 1,2,3,4,5,6").fetchall():
    print(f'  grid=({gx},{gy},{gz}) block=({bx},{by},{bz}) n={n} tot={ns/1e6:.1f}ms avg={ns/1e3/n:.1f}us')

print()
print('--- all kernels by class ---')
rows = con.execute(
    "select (select value from StringIds where id=k.demangledName) nm, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k").fetchall()
def cl(nm):
    if 'cutlass::Kernel' in nm or 'splitKreduce' in nm: return 'GEMM'
    if 'dequantize' in nm: return 'dequant'
    if 'gated_delta_net' in nm: return 'GDN'
    if 'flash_attn' in nm: return 'attention'
    return 'other'
tot = sum(e - s for _, s, e in rows)
agg = {}
for nm, s, e in rows:
    c = cl(nm or '')
    agg[c] = agg.get(c, 0) + (e - s)
print(f'  total GPU kernel time = {tot/1e6:.1f} ms over {len(rows)} launches')
for c, ns in sorted(agg.items(), key=lambda x: -x[1]):
    print(f'  {c:<10} {ns/1e6:>9.1f} ms  {100*ns/tot:>5.1f}%')

# FA timeline: group by 16 (16 attn layers per ubatch); with ub2048 there are 4x fewer groups
rows = con.execute(
    "select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
    "where (select value from StringIds where id=k.demangledName) like '%flash_attn%' order by k.start").fetchall()
t0 = con.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
print()
print('--- FA groups (16 = 1 ubatch x 16 layers) ---')
N_HEAD, HEAD_DIM, N_LAYER = 24, 256, 16
for i in range(0, len(rows), 16):
    gp = rows[i:i+16]
    us = sum(e - s for s, e in gp)
    print(f'  grp{i//16:>3} start={(gp[0][0]-t0)/1e6:>8.1f}ms tot={us/1e6:>8.2f}ms avg={us/1e3/16:>9.1f}us')
con.close()
