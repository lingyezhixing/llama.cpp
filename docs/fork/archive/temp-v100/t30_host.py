import sqlite3
c=sqlite3.connect('t30_cli128k.sqlite'); cur=c.cursor()
tabs=[r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
def has(t): return t in tabs
print("相关表:", [t for t in tabs if 'RUNTIME' in t or 'MEMCPY' in t or 'SYNC' in t or 'MEMSET' in t or 'NVTX' in t])
w0,w1=279.20e9, 284.65e9
print(f"=== 空档窗口 {w0/1e9:.2f}-{w1/1e9:.2f}s (5.45s) ===")
if has('CUPTI_ACTIVITY_KIND_RUNTIME'):
    q="""SELECT s.value, COUNT(*), SUM(r.end-r.start)/1e6 FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
    WHERE r.start>=? AND r.end<=? GROUP BY s.value ORDER BY COUNT(*) DESC LIMIT 15"""
    print(f"{'n':>7} {'ms(调用总耗时)':>14}  API")
    for n,nc,ms in cur.execute(q,(w0,w1)):
        print(f"{n:7d} {ms:14.0f}  {nc[:95]}")
if has('CUPTI_ACTIVITY_KIND_MEMCPY'):
    q2="""SELECT s.value, COUNT(*), SUM(m.bytes)/1048576 FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN StringIds s ON s.id=m.nameId
    WHERE m.start>=? AND m.end<=? GROUP BY s.value ORDER BY COUNT(*) DESC LIMIT 10"""
    print("--- memcpy 类 ---")
    for n,cnt,mb in cur.execute(q2,(w0,w1)):
        print(f"{cnt:7d} 个  {mb:10.1f} MB  {n[:80]}")
if has('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION'):
    q3="""SELECT s.value, COUNT(*), SUM(y.end-y.start)/1e6 FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION y JOIN StringIds s ON s.id=y.nameId
    WHERE y.start>=? AND y.end<=? GROUP BY s.value ORDER BY COUNT(*) DESC LIMIT 8"""
    print("--- 同步类 ---")
    for n,cnt,ms in cur.execute(q3,(w0,w1)):
        print(f"{cnt:7d} 个  {ms:10.1f} ms  {n[:80]}")
