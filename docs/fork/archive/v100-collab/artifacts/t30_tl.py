import sqlite3
c=sqlite3.connect('t30_cli128k.sqlite'); cur=c.cursor()
def tl(pat,label):
    rows=cur.execute("""SELECT k.start,k.end,k.gridX,k.gridY FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE ? ORDER BY k.start""",(pat,)).fetchall()
    if not rows: print(label,"none"); return
    print(f"{label}: n={len(rows)}  first={rows[0][0]/1e9:.2f}s last_end={rows[-1][1]/1e9:.2f}s")
    if len(rows)<60:
        print("   starts:", ", ".join(f"{r[0]/1e9:.2f}" for r in rows))
    else:
        import statistics
        gaps=[(rows[i+1][0]-rows[i][1])/1e6 for i in range(len(rows)-1)]
        big=[(i,round(g,1)) for i,g in enumerate(gaps) if g>200]
        print(f"   gaps>200ms at indices: {big[:10]}")
tl('%flash_attn_tile%','TILE')
tl('%flash_attn_ext_vec%','VEC')
tl('%flash_attn_ext_f16%','MMA')
print()
rows=cur.execute("SELECT MIN(start),MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
print("kernel span", rows[0]/1e9, rows[1]/1e9)
