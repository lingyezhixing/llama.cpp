import sqlite3, os, sys

path = sys.argv[1]
con = sqlite3.connect(path)
t0 = con.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]

print('--- FA / fixup kernel configs ---')
rows = con.execute(
    "select (select value from StringIds where id=k.demangledName) nm, k.gridX, k.gridY, k.gridZ, k.blockX*k.blockY, count(*), sum(k.end-k.start) "
    "from CUPTI_ACTIVITY_KIND_KERNEL k "
    "where (select value from StringIds where id=k.demangledName) like '%flash_attn%' "
    "group by 1,2,3,4,5 order by 7 desc").fetchall()
for nm, gx, gy, gz, thr, n, ns in rows:
    print(f'  grid=({gx},{gy},{gz}) thr={thr} n={n:>5} tot={ns/1e6:>8.2f}ms avg={ns/1e3/n:>8.2f}us  {nm[:88]}')

fa = con.execute(
    "select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k "
    "where (select value from StringIds where id=k.demangledName) like 'void flash_attn_ext_f16%' order by k.start").fetchall()
print(f'\n--- FA launches: {len(fa)} (16 per ubatch) ---')
groups = [fa[i:i+16] for i in range(0, len(fa), 16)]
N_HEAD, HEAD_DIM, N_LAYER, NQ = 24, 256, 16, 512
fl = N_LAYER * 4 * HEAD_DIM * N_HEAD * sum(NQ * (32768 + 512*k) + NQ*(NQ+1)//2 for k in range(8))
print(f'expected eval FLOP = {fl/1e12:.2f} TFLOP')
tail = groups[-8:]
tot = sum(e - s for gp in tail for s, e in gp)
print(f'measured eval (last 8 ubatch groups): {tot/1e6:.1f} ms -> {fl/tot:.1f} TF/s')
for i, gp in enumerate(groups[-10:]):
    us = sum(e - s for s, e in gp)
    print(f'  grp{len(groups)-10+i:>3} start={(gp[0][0]-t0)/1e6:>9.1f}ms tot={us/1e6:>8.2f}ms avg={us/1e3/16:>9.1f}us')

tot_all = con.execute("select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
print(f'\ntotal GPU kernel time = {tot_all/1e6:.1f} ms')
con.close()
