import sqlite3, sys, collections

db = sys.argv[1]
con = sqlite3.connect(db)
cur = con.cursor()
rows = cur.execute("""SELECT (select value from StringIds where id=k.demangledName) nm, k.gridX, k.gridY, k.gridZ, (k.end-k.start)/1e6
                      FROM CUPTI_ACTIVITY_KIND_KERNEL k
                      WHERE nm LIKE '%flash_attn_ext_f16%' OR nm LIKE '%stream_k_fixup%' OR nm LIKE '%combine_results%'""").fetchall()
agg = collections.defaultdict(lambda: [0, 0.0])
for nm, gx, gy, gz, ms in rows:
    if "flash_attn_ext_f16" in nm:
        short = "fa"
    elif "stream_k_fixup" in nm:
        short = "fixup"
    else:
        short = "combine"
    agg[(short, gx, gy, gz)][0] += 1
    agg[(short, gx, gy, gz)][1] += ms
fa = 0.0
tot = 0.0
for (short, gx, gy, gz), (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print("  %-8s grid=(%d,%d,%d) n=%5d  %8.2f ms" % (short, gx, gy, gz, n, ms))
    tot += ms
    if short == "fa":
        fa += ms
print("  FA kernel total %.2f ms (with fixup/combine: %.2f ms)" % (fa, tot))
