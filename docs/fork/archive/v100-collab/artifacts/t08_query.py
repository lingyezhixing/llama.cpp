import sqlite3, os, collections
db = os.path.expandvars(r'%TEMP%\v100\t08_prof.sqlite')
con = sqlite3.connect(db)
cur = con.cursor()
tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
kern = 'CUPTI_ACTIVITY_KIND_KERNEL'
cols = [r[1] for r in cur.execute(f"PRAGMA table_info({kern})")]
print('kernel cols:', [c for c in cols if c.lower() in ('gridx','gridy','gridz','blockx','blocky','blockz','end','start','demangledname','shortname','stringid','registersperthread','dynamicsharedmemory')][:12])
q = f"""SELECT s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, k.end-k.start AS ns, k.registersPerThread, k.dynamicSharedMemory
FROM {kern} k JOIN StringIds s ON s.id = k.demangledName"""
rows = list(cur.execute(q))
print('total kernels:', len(rows))
agg = collections.defaultdict(lambda: [0,0.0,set()])
for name, gx, gy, gz, bx, by, bz, ns, regs, smem in rows:
    key = (name, gx, gy, gz)
    a = agg[key]
    a[0] += 1
    a[1] += ns/1e6
    a[2].add((bx,by,bz,regs,smem))
tot = sum(a[1] for a in agg.values())
print(f'total kernel time: {tot:.1f} ms')
items = sorted(agg.items(), key=lambda kv: -kv[1][1])
print()
print(f'{"count":>6} {"ms":>9} {"tot%":>6}  grid        block/regs/smem          kernel')
for (name, gx, gy, gz), a in items[:28]:
    short = name.split('(')[0][:74]
    blk = next(iter(a[2]))
    print(f'{a[0]:6d} {a[1]:9.2f} {100*a[1]/tot:5.1f}%  ({gx:6d},{gy:3d},{gz:3d})  b={blk[0]:4d},{blk[1]:3d},{blk[2]:2d} r={blk[3]:4d} s={blk[4]:6d}  {short}')
con.close()
