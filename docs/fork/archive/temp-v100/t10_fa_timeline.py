import sqlite3, sys

def fa_timeline(path, label):
    con = sqlite3.connect(path); cur = con.cursor()
    rows = cur.execute("""
        select k.start, k.end, (select value from StringIds where id = k.demangledName)
        from CUPTI_ACTIVITY_KIND_KERNEL k
        where (select value from StringIds where id = k.demangledName) like '%flash_attn%'
        order by k.start
    """).fetchall()
    t0 = cur.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    print("=" * 78)
    print(f"[{label}] FA launches: {len(rows)}, first kernel t0 offset={t0}")
    print("=" * 78)
    print(f"{'grp':>4} {'start_ms':>10} {'gap_ms':>8} {'n':>3} {'sum_us':>10} {'avg_us':>9}")
    prev_end = None
    for g in range(0, len(rows), 16):
        chunk = rows[g:g+16]
        s = chunk[0][0]; e = chunk[-1][1]
        tot = sum(c[1] - c[0] for c in chunk)
        gap = (s - prev_end) / 1e6 if prev_end else 0.0
        print(f"{g//16:>4} {(s-t0)/1e6:>10.1f} {gap:>8.1f} {len(chunk):>3} {tot/1e3:>10.1f} {tot/1e3/len(chunk):>9.1f}")
        prev_end = e
    con.close()

fa_timeline(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
