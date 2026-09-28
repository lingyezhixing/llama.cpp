import sqlite3, os, collections
con = sqlite3.connect(os.path.expandvars(r'%TEMP%\v100\t08_one.sqlite')); cur = con.cursor()
q = """SELECT s.value, k.gridX, k.gridY, k.gridZ, k.end-k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"""
rows = list(cur.execute(q))
print('t08_one launches:')
for r in rows: print(f'  grid=({r[1]},{r[2]},{r[3]}) {r[4]/1e6:.3f} ms  {r[0][:60]}')
con.close()
