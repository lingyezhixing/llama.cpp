import sqlite3, os, csv

T = os.path.join(os.environ['TEMP'], 'v100')
A = r'D:\LLM\Backend\v100-collab\artifacts'

def dump(lbl, sqlite_name, out):
    con = sqlite3.connect(os.path.join(T, sqlite_name))
    rows = con.execute(
        "select (select value from StringIds where id=k.demangledName) nm, count(*), sum(k.end-k.start) "
        "from CUPTI_ACTIVITY_KIND_KERNEL k group by nm order by 3 desc").fetchall()
    tot = sum(r[2] or 0 for r in rows)
    with open(os.path.join(A, out), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kernel', 'launches', 'total_ms', 'share_percent', 'avg_us'])
        for nm, n, ns in rows:
            w.writerow([nm, n, f'{ns/1e6:.3f}', f'{100*ns/tot:.3f}', f'{ns/1e3/max(n,1):.2f}'])
    print(f'{out}: {len(rows)} kernels, total {tot/1e6:.1f} ms')
    con.close()

dump('pp4096_d0', 't10_pp4096_d0.sqlite', 't10_pp4096_d0_kern_sum.csv')
dump('pp4096_d32k', 't10_pp4096_d32k.sqlite', 't10_pp4096_d32k_kern_sum.csv')
dump('pp32768', 't10_pp32768.sqlite', 't10_pp32768_kern_sum.csv')
dump('ub2048', 't10_d32k_ub2048.sqlite', 't10_d32k_ub2048_kern_sum.csv')
