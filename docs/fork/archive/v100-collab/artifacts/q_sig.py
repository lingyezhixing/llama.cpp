import sqlite3, os

T = os.path.join(os.environ['TEMP'], 'v100')
c = sqlite3.connect(os.path.join(T, 'dec4gr.sqlite'))
mn = c.execute('select min(start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
rows = [s for (s,) in c.execute("""select start from CUPTI_ACTIVITY_KIND_KERNEL k
        where (select value from StringIds where id=k.demangledName) like '%mul_mat_vec_q<(ggml_type)14, (int)1, (bool)1%'
        order by start""")]
starts, prev = [], None
for s in rows:
    if prev is None or s - prev > 8_000_000:
        starts.append(s)
    prev = s
w0, w1 = starts[2], starts[3]

print('=== per-kernel signatures in steady-state token (grid x block) ===')
rows = c.execute("""select (select value from StringIds where id=k.demangledName) nm,
                           k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY,
                           count(*), sum(k.end-k.start)
                    from CUPTI_ACTIVITY_KIND_KERNEL k
                    where k.start >= ? and k.start < ?
                    group by nm, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY
                    order by 8 desc""", (w0, w1)).fetchall()
for nm, gx, gy, gz, bx, by, cnt, ns in rows:
    nm = (nm or '?')
    if 'mul_mat_vec_q' in nm:
        nm = nm[:42]
    print('%-46s grid=(%5d,%3d,%3d) blk=(%4d,%d) n=%4d tot=%7.3f ms avg=%6.2f us' %
          (nm[:46], gx, gy, gz, bx, by, cnt, ns/1e6, ns/1e3/cnt))
