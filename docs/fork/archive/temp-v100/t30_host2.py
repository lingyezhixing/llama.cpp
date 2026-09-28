import sqlite3
c=sqlite3.connect('t30_cli128k.sqlite'); cur=c.cursor()
w0,w1=279.20e9, 284.65e9
print(f"=== 空档窗口 {w0/1e9:.2f}-{w1/1e9:.2f}s ===")
q="""SELECT s.value, COUNT(*), SUM(r.end-r.start)/1e6 FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
WHERE r.start>=? AND r.end<=? GROUP BY s.value ORDER BY COUNT(*) DESC LIMIT 15"""
print(f"{'n':>7} {'ms':>10}  API")
for name,n,ms in cur.execute(q,(w0,w1)):
    print(f"{n:7d} {ms:10.0f}  {name[:95]}")
q2="""SELECT s.value, COUNT(*), SUM(m.bytes)/1048576.0 FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN StringIds s ON s.id=m.nameId
WHERE m.start>=? AND m.end<=? GROUP BY s.value ORDER BY COUNT(*) DESC LIMIT 12"""
print("--- memcpy ---")
for name,cnt,mb in cur.execute(q2,(w0,w1)):
    print(f"{cnt:7d} 个 {mb:10.1f} MB  {name[:80]}")
q3="""SELECT s.value, COUNT(*), SUM(y.end-y.start)/1e6 FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION y JOIN StringIds s ON s.id=y.nameId
WHERE y.start>=? AND y.end<=? GROUP BY s.value ORDER BY COUNT(*) DESC LIMIT 8"""
print("--- 同步 ---")
for name,cnt,ms in cur.execute(q3,(w0,w1)):
    print(f"{cnt:7d} 个 {ms:10.1f} ms  {name[:80]}")
print()
print("=== 对比: verify 密集窗口 278.85-279.15s ===")
w2,w3=278.85e9,279.15e9
for name,n,ms in cur.execute(q,(w2,w3)):
    print(f"{n:7d} {ms:10.0f}  {name[:95]}")
