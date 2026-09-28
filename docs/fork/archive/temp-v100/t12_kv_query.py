import sqlite3, sys, collections

db = sys.argv[1]
pat = sys.argv[2] if len(sys.argv) > 2 else "%fattn%"
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute("""SELECT (select value from StringIds where id=k.demangledName) nm,
                      k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ, (k.end-k.start)/1e6
               FROM CUPTI_ACTIVITY_KIND_KERNEL k
               WHERE nm LIKE ?""", (pat,))
agg = collections.defaultdict(lambda: [0, 0.0])
for name, gx, gy, gz, bx, by, bz, ms in cur:
    key = (name.split("(")[0], gx, gy, gz, bx, by, bz)
    a = agg[key]
    a[0] += 1
    a[1] += ms
print("%-52s %-14s %-10s %8s %10s" % ("kernel", "grid", "block", "n", "total_ms"))
tot = 0.0
for (name, gx, gy, gz, bx, by, bz), (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print("%-52s %-14s %-10s %8d %10.2f" % (name[:52], "(%d,%d,%d)" % (gx, gy, gz), "(%d,%d,%d)" % (bx, by, bz), n, ms))
    if "fixup" not in name:
        tot += ms
print("attention excl fixup: %.2f ms" % tot)
