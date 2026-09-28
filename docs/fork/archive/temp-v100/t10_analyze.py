import sqlite3, sys, os

def col(cur, tab):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({tab})")]

def kernel_report(path, label):
    con = sqlite3.connect(path)
    cur = con.cursor()
    tabs = [r[0] for r in cur.execute("select name from sqlite_master where type='table'")]
    print("=" * 78)
    print(f"[{label}]  {os.path.basename(path)}")
    print("=" * 78)
    if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tabs:
        print("no CUPTI_ACTIVITY_KIND_KERNEL; tables:", [t for t in tabs if 'KERNEL' in t or 'ACTIVITY' in t][:20])
        return
    c = col(cur, 'CUPTI_ACTIVITY_KIND_KERNEL')
    name_expr = None
    for cand in ('demangledName', 'shortName'):
        if cand in c:
            nid = cand
            name_expr = f"(select value from StringIds where id = k.{nid})"
            break
    if name_expr is None:
        for cand in ('nameId', 'name'):
            if cand in c:
                name_expr = f"(select value from StringIds where id = k.{cand})"
                break
    rows = cur.execute(f"""
        select {name_expr} as nm, count(*) as n, sum(k.end - k.start) as ns
        from CUPTI_ACTIVITY_KIND_KERNEL k
        group by nm order by ns desc
    """).fetchall()
    total = sum(r[2] or 0 for r in rows)
    span = cur.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    print(f"kernel events: {sum(r[1] or 0 for r in rows)}, total GPU kernel time: {total/1e6:.1f} ms")
    print(f"timeline span: {(span[1]-span[0])/1e6:.1f} ms  (first kernel start -> last kernel end)")
    print()
    print(f"{'kernel':<72} {'n':>5} {'tot_ms':>9} {'avg_us':>8} {'share':>7}")
    for nm, n, ns in rows[:32]:
        nm = (nm or '?')
        short = nm if len(nm) <= 71 else nm[:34] + '...' + nm[-34:]
        print(f"{short:<72} {n:>5} {ns/1e6:>9.2f} {ns/1e3/max(n,1):>8.2f} {100*ns/total:>6.1f}%")
    print()
    for key in ('flash_attn', 'fattn', 'softmax', 'attn'):
        sel = [r for r in rows if key in (r[0] or '').lower()]
        if sel:
            n = sum(r[1] for r in sel); ns = sum(r[2] for r in sel)
            print(f"  [group {key}] n={n} total={ns/1e6:.2f} ms  share={100*ns/total:.1f}%")
            for nm, nn, nns in sel[:12]:
                print(f"      {nn:>5} x {nns/1e6:>8.2f} ms  {nm[:150]}")
    con.close()

for arg in sys.argv[1:]:
    label, path = arg.split('=', 1)
    kernel_report(path, label)
