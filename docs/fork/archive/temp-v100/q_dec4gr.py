import sqlite3, os, collections

T = os.path.join(os.environ['TEMP'], 'v100')
c = sqlite3.connect(os.path.join(T, 'dec4gr.sqlite'))

n, ng = c.execute('select count(*), sum(case when graphNodeId != 0 then 1 else 0 end) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
print('kernel rows = %d, graph nodes = %d' % (n, ng))
mn, mx = c.execute('select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
print('span = %.1f ms' % ((mx-mn)/1e6))

print('\n--- busy time per 5ms bucket ---')
for b, cnt, ns in c.execute("""select cast((start-?)/5000000 as int) b, count(*), sum(end-start)
                               from CUPTI_ACTIVITY_KIND_KERNEL group by b order by b""", (mn,)):
    print('  bucket %3d: n=%5d busy=%6.2f ms' % (b, cnt, ns/1e6))

# identify token boundaries: MMVQ of type 14 bool1 (the biggest, ffn gate/up fused) appears 36x per token
print('\n--- token boundary probe: start times of mul_mat_vec_q<14,1,1> ---')
rows = c.execute("""select start from CUPTI_ACTIVITY_KIND_KERNEL k
                    where (select value from StringIds where id=k.demangledName) like '%mul_mat_vec_q<(ggml_type)14, (int)1, (bool)1%'
                    order by start""").fetchall()
prev = None
for (s,) in rows[:40]:
    mark = ''
    if prev is not None and s - prev > 8_000_000:
        mark = '  <-- gap %.1f ms' % ((s-prev)/1e6)
    print('  %.2f ms%s' % ((s-mn)/1e6, mark))
    prev = s
