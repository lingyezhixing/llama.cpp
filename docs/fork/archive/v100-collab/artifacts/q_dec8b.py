import sqlite3, os

T = os.path.join(os.environ['TEMP'], 'v100')
c = sqlite3.connect(os.path.join(T, 'dec8.sqlite'))

n = c.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
ng = c.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId != 0').fetchone()[0]
print('kernel rows = %d, with graphNodeId != 0: %d' % (n, ng))

mn, mx = c.execute('select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
print('kernel time span = %.1f ms' % ((mx-mn)/1e6))

# where do the graph launches sit in time?
rows = c.execute("""select r.start, r.end, (select value from StringIds where id=r.nameId) nm
                    from CUPTI_ACTIVITY_KIND_RUNTIME r
                    where nm like '%GraphLaunch%' order by r.start""").fetchall()
for s, e, nm in rows:
    print('  graphLaunch @ %.2f ms  dur=%.2f ms' % ((s-mn)/1e6, (e-s)/1e6))

# kernels grouped by time buckets of 5ms to see gaps / which tokens were traced
print('\n--- kernel activity per 5ms bucket (count, ms) ---')
rows = c.execute("""select cast((k.start-?)/5000000 as int) b, count(*), sum(k.end-k.start)
                    from CUPTI_ACTIVITY_KIND_KERNEL k group by b order by b""", (mn,)).fetchall()
for b, cnt, ns in rows[:30]:
    print('  bucket %3d (%.0f-%.0f ms): n=%5d  busy=%6.2f ms' % (b, b*5, (b+1)*5, cnt, ns/1e6))

# per-name totals, but only for the LAST few ms (steady-state token)
print('\n--- kernels in last 40 ms of trace ---')
rows = c.execute("""select (select value from StringIds where id=k.demangledName) nm, count(*), sum(k.end-k.start)
                    from CUPTI_ACTIVITY_KIND_KERNEL k where k.start > ? group by nm order by 3 desc""",
                 (mx - 40_000_000,)).fetchall()
tt = sum(r[2] for r in rows)
print('  total = %.1f ms' % (tt/1e6))
for nm, cnt, ns in rows[:40]:
    print('  %-50s n=%5d tot=%7.2f ms avg=%6.2f us' % ((nm or '?')[:50], cnt, ns/1e6, ns/1e3/cnt))
