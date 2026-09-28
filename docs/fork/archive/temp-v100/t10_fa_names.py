import sqlite3, os
T = os.path.join(os.environ['TEMP'], 'v100')
for lbl in ['pp4096_d0', 'pp4096_d32k', 'pp32768']:
    con = sqlite3.connect(os.path.join(T, 't10_' + lbl + '.sqlite'))
    rows = con.execute(
        "select (select value from StringIds where id = k.demangledName) nm, count(*), sum(k.end-k.start) "
        "from CUPTI_ACTIVITY_KIND_KERNEL k "
        "where nm like '%flash_attn%' or nm like '%fattn%' or nm like '%stream_k%' or nm like '%softcap%' "
        "group by nm order by 3 desc").fetchall()
    print('===', lbl, '===')
    for nm, n, ns in rows:
        print(f'  n={n:>5} {ns/1e6:>9.1f}ms  {nm}')
    con.close()
