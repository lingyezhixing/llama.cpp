import sqlite3
c=sqlite3.connect('t30_cli128k.sqlite'); cur=c.cursor()
print("MEMCPY 表结构:", [r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")])
w0,w1=279.20e9, 284.65e9
tot=cur.execute("SELECT SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?",(w0,w1)).fetchone()[0]
cnt=cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?",(w0,w1)).fetchone()[0]
print(f"该窗口 GPU kernel: {cnt} 个, 合计 {tot:.1f} ms (窗口 5450ms) -> GPU busy {100*tot/5450:.1f}%")
q="""SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
WHERE k.start>=? AND k.end<=? GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 12"""
for name,n,ms in cur.execute(q,(w0,w1)):
    print(f"{n:7d} {ms:10.2f} ms  {name[:95]}")
# memcpy
q2="""SELECT copyKind, COUNT(*), SUM(bytes)/1048576.0 FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=? GROUP BY copyKind ORDER BY COUNT(*) DESC"""
print("--- memcpy by copyKind ---")
for k,n,mb in cur.execute(q2,(w0,w1)):
    print(f"kind={k} {n:6d} 个 {mb:9.2f} MB")
# 最大单次 sync 与分布
q3="""SELECT (end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE start>=? AND end<=? ORDER BY (end-start) DESC LIMIT 5"""
print("--- 最长的 5 次同步 (ms) ---", [round(r[0],1) for r in cur.execute(q3,(w0,w1))])
tot_sync=cur.execute("SELECT SUM(end-start)/1e6, COUNT(*) FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION WHERE start>=? AND end<=?",(w0,w1)).fetchone()
print(f"sync 合计 {tot_sync[0]:.0f} ms / {tot_sync[1]} 次")
