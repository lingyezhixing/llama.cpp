import sqlite3
c=sqlite3.connect('t30_cli128k.sqlite'); cur=c.cursor()
w0,w1 = 278.60e9, 279.25e9
q="""SELECT s.value, COUNT(*), SUM(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
WHERE k.start>=? AND k.end<=? GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 22"""
tot=cur.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?",(w0,w1)).fetchone()[0]
print(f"window 278.60-279.25s  total GPU = {tot/1e6:.1f} ms")
for n, cnt, ns in cur.execute(q,(w0,w1)):
    print(f"{ns/1e6:8.2f} ms  n={cnt:5d}  {n[:100]}")
print()
q2="""SELECT
  CASE WHEN s.value LIKE '%flash_attn_tile%' THEN 'fa_tile(verify)'
       WHEN s.value LIKE '%flash_attn_ext_vec%' THEN 'fa_vec'
       WHEN s.value LIKE '%flash_attn%' THEN 'fa_other'
       WHEN s.value LIKE '%mul_mat_vec%' THEN 'mmvq'
       WHEN s.value LIKE '%dequantize%' THEN 'dequant'
       WHEN s.value LIKE '%gated_delta_net%' THEN 'gdn'
       WHEN s.value LIKE '%ssm_conv%' THEN 'ssm_conv'
       WHEN s.value LIKE '%rms_norm%' THEN 'rms_norm'
       ELSE 'misc' END cls, COUNT(*), SUM(k.end-k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
WHERE k.start>=? AND k.end<=? GROUP BY cls ORDER BY SUM(k.end-k.start) DESC"""
for cls,n,ns in cur.execute(q2,(w0,w1)):
    print(f"{ns/1e6:8.2f} ms  n={n:5d}  {cls}")
