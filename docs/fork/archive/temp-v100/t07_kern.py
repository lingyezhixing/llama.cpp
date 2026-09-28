import sqlite3, os

T = os.path.join(os.environ['TEMP'], 'v100')
for lbl, path in [('BEFORE (4-file DLL)', 't10_pp4096_d0.sqlite'), ('AFTER (rms_norm vec4)', 't07_rms_vec4_pp4096.sqlite')]:
    con = sqlite3.connect(os.path.join(T, path))
    rows = con.execute(
        "select (select value from StringIds where id=k.demangledName) nm, count(*), sum(k.end-k.start) "
        "from CUPTI_ACTIVITY_KIND_KERNEL k "
        "where (select value from StringIds where id=k.demangledName) like '%rms_norm%' "
        "group by nm order by 3 desc").fetchall()
    tot = sum(r[2] for r in rows)
    print('===', lbl, '===')
    for nm, n, ns in rows:
        print(f'  n={n:>5} tot={ns/1e6:>8.2f}ms avg={ns/1e3/n:>7.2f}us  {nm[:96]}')
    print(f'  TOTAL rms_norm = {tot/1e6:.2f} ms (per 2 forwards) -> {tot/2/1e6:.2f} ms/forward')
    # overlap/other checks
    allk = con.execute(
        "select (select value from StringIds where id=k.demangledName) nm, count(*), sum(k.end-k.start) "
        "from CUPTI_ACTIVITY_KIND_KERNEL k group by nm order by 3 desc limit 6").fetchall()
    print('  top kernels overall:')
    for nm, n, ns in allk:
        print(f'     n={n:>5} {ns/1e6:>9.2f}ms  {nm[:80]}')
    con.close()
