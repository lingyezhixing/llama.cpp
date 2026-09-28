import sqlite3, os

T = os.path.join(os.environ['TEMP'], 'v100')
c = sqlite3.connect(os.path.join(T, 'dec8.sqlite'))

# schema probe
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
print('tables with RUNTIME/KERNEL:', [t for t in tabs if 'RUNTIME' in t or 'KERNEL' in t][:6])

kcols = [r[1] for r in c.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
print('kernel cols:', kcols)

# api counts
try:
    rows = c.execute("""select (select value from StringIds where id=r.nameId) nm, count(*), sum(r.end-r.start)
                        from CUPTI_ACTIVITY_KIND_RUNTIME r
                        where nm like '%Graph%' or nm like '%Launch%' group by nm order by 2 desc""").fetchall()
    print('\n--- runtime API ---')
    for nm, n, ns in rows:
        print('  %-34s n=%6d  tot=%8.2f ms' % (nm, n, ns/1e6))
except Exception as e:
    print('runtime err', e)

# kernels by name
rows = c.execute("""select (select value from StringIds where id=k.demangledName) nm, count(*), sum(k.end-k.start)
                    from CUPTI_ACTIVITY_KIND_KERNEL k group by nm order by 3 desc""").fetchall()
tot = sum(r[2] for r in rows)
print('\n--- kernels by name (total %.1f ms) ---' % (tot/1e6))
for nm, n, ns in rows:
    print('  %-52s n=%6d  tot=%8.2f ms  avg=%7.2f us' % ((nm or '?')[:52], n, ns/1e6, ns/1e3/n))
