import sqlite3
c=sqlite3.connect('t30_cli128k_vec.sqlite'); cur=c.cursor()
rows=cur.execute("""SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
WHERE s.value LIKE '%flash_attn_ext_vec%' AND s.value LIKE '%(ggml_type)8%' ORDER BY k.start""").fetchall()
print("VEC(q8_0) 实例数:", len(rows))
# 取最后一个完整 verify 窗口: 17 个实例
w0=rows[-17][0]; w1=rows[-1][1]
print(f"last verify window: {(w1-w0)/1e6:.1f} ms, 17 layers")
q="""SELECT s.value, COUNT(*), SUM(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
WHERE k.start>=? AND k.end<=? GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 12"""
tot=cur.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?",(w0,w1)).fetchone()[0]
print(f"total GPU = {tot/1e6:.1f} ms")
for n,cnt,ns in cur.execute(q,(w0,w1)):
    print(f"{ns/1e6:8.2f} ms n={cnt:5d}  {n[:95]}")
print()
q2="""SELECT CASE WHEN s.value LIKE '%dequantize_block_q8_0_f16%' THEN 'v_mat' WHEN s.value LIKE '%flash_attn_ext_vec%' THEN 'fa_vec'
WHEN s.value LIKE '%mul_mat_vec%' THEN 'mmvq' WHEN s.value LIKE '%dequantize%' THEN 'dequant_other' ELSE 'misc' END cls,
COUNT(*), SUM(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
WHERE k.start>=? AND k.end<=? GROUP BY cls ORDER BY SUM(k.end-k.start) DESC"""
for cls,n,ns in cur.execute(q2,(w0,w1)):
    print(f"{ns/1e6:8.2f} ms n={n:5d}  {cls}")
