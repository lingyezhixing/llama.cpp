import sqlite3, os, collections
db = os.path.expandvars(r'%TEMP%\v100\t08_prof.sqlite')
con = sqlite3.connect(db); cur = con.cursor()
q = """SELECT s.value, k.gridX, k.gridY, k.gridZ, k.start, k.end-k.start AS ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
WHERE s.value LIKE '%cutlass_70_tensorop%' ORDER BY k.start"""
rows = list(cur.execute(q))
print('cutlass tensorop launches:', len(rows))
# cluster within group (gx,gy,gz) by duration
groups = collections.defaultdict(list)
for name, gx, gy, gz, st, ns in rows:
    groups[(gx,gy,gz)].append(ns/1e6)
for key in sorted(groups, key=lambda k: -sum(groups[k])):
    ds = sorted(groups[key])
    print(f'grid {key}: n={len(ds):4d} total={sum(ds):8.2f} ms  min={ds[0]:.3f} med={ds[len(ds)//2]:.3f} max={ds[-1]:.3f}')
    # histogram of durations (coarse)
    buckets = collections.Counter(round(d,1) for d in ds)
    print('   dur histogram (0.1ms):', dict(sorted(buckets.items())))
# sequence of first 40 launches of the whole run (for pattern)
print()
print('first 40 launches (gridz,ms):', [(gx,gy,gz,round(ns/1e6,3)) for name,gx,gy,gz,st,ns in rows[:40]])
print()
print('launches 40..80:', [(gx,gy,gz,round(ns/1e6,3)) for name,gx,gy,gz,st,ns in rows[40:80]])
con.close()
