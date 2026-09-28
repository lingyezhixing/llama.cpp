import sqlite3
c=sqlite3.connect('t30_cli128k.sqlite'); cur=c.cursor()
print("=== TILE 实例 (start, dur_ms, gridX, gridY, blockX) ===")
for r in cur.execute("""SELECT k.start,k.end,k.gridX,k.gridY,k.blockX FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE '%flash_attn_tile%' ORDER BY k.start LIMIT 20"""):
    print(f"  t={r[0]/1e9:.3f}s dur={(r[1]-r[0])/1e6:.2f}ms grid=({r[2]},{r[3]}) blockX={r[4]}")
print("=== VEC 实例 ===")
for r in cur.execute("""SELECT k.start,k.end,k.gridX,k.gridY,k.blockX FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE '%flash_attn_ext_vec%' ORDER BY k.start LIMIT 22"""):
    print(f"  t={r[0]/1e9:.3f}s dur={(r[1]-r[0])/1e6:.2f}ms grid=({r[2]},{r[3]}) blockX={r[4]}")
