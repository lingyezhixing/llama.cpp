import sqlite3, os, collections
con = sqlite3.connect(os.path.expandvars(r'%TEMP%\v100\t08_arb.sqlite')); cur = con.cursor()
q = """SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.registersPerThread, k.dynamicSharedMemory, k.end-k.start
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
WHERE s.value LIKE '%cutlass%' ORDER BY k.start"""
rows = list(cur.execute(q))
agg = collections.defaultdict(lambda: [0,0.0])
for name,gx,gy,gz,bx,regs,smem,ns in rows:
    key = (name[:70], gx,gy,gz,bx,regs,smem)
    agg[key][0]+=1; agg[key][1]+=ns/1e6
print(f'{"n":>4} {"ms":>9}  grid               blk regs smem   kernel')
for k,v in sorted(agg.items(), key=lambda kv:-kv[1][1]):
    name,gx,gy,gz,bx,regs,smem = k
    print(f'{v[0]:4d} {v[1]:9.2f}  ({gx:6d},{gy:3d},{gz:2d})  {bx:4d} {regs:4d} {smem:6d}  {name}')
print()
print('--- model side (from t08_prof.sqlite) ---')
con2 = sqlite3.connect(os.path.expandvars(r'%TEMP%\v100\t08_prof.sqlite')); cur2 = con2.cursor()
q2 = """SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.registersPerThread, k.dynamicSharedMemory, k.end-k.start
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
WHERE s.value LIKE '%cutlass_70_tensorop%' ORDER BY k.start LIMIT 3"""
for r in cur2.execute(q2):
    print(f'grid=({r[1]},{r[2]},{r[3]}) blk={r[4]} regs={r[5]} smem={r[6]} dur={r[7]/1e6:.3f}ms {r[0][:60]}')
con.close(); con2.close()
