import sqlite3
c=sqlite3.connect('t30_cli128k_vec.sqlite'); cur=c.cursor()
print("=== VEC (verify, n_q=4 -> cols_per_block=2) 实例: start, dur_ms, grid ===")
rows=cur.execute("""SELECT k.start,k.end,k.gridX,k.gridY,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
WHERE s.value LIKE '%flash_attn_ext_vec%' ORDER BY k.start""").fetchall()
import collections
by=collections.defaultdict(list)
for st,en,gx,gy,name in rows:
    key=(name.split('<')[1].split('>')[0], gx, gy)
    by[key].append((en-st)/1e6)
for key,v in sorted(by.items(), key=lambda kv:-sum(kv[1])):
    print(f"  {len(v):4d} 个  平均 {sum(v)/len(v):6.2f} ms  总计 {sum(v):8.1f} ms  模板<{key[0]}> grid=({key[1]},{key[2]})")
print()
print("=== TILE 实例 (应为 0) ===")
print(cur.execute("""SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE '%flash_attn_tile%'""").fetchone()[0])
print("=== V 物化 (q8_0->f16) 实例数 ===")
print(cur.execute("""SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE s.value LIKE '%dequantize_block_q8_0_f16%'""").fetchone()[0])
