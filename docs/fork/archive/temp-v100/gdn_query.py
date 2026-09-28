import sqlite3, os, sys
c = sqlite3.connect(os.path.join(os.environ['TEMP'], 'v100', sys.argv[1]))
r = c.execute("select (select value from StringIds where id=k.demangledName) nm, count(*), sum(k.end-k.start) "
              "from CUPTI_ACTIVITY_KIND_KERNEL k where nm like '%gated_delta%' group by nm").fetchall()
for nm, n, ns in r:
    print('  GDN %-56s n=%5d tot=%8.1fms  avg=%.0f us/call' % (nm[:56], n, ns/1e6, ns/1e3/n))
tot = c.execute('select sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
print('  total GPU kernel time = %.1f ms' % (tot/1e6))
