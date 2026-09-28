import sqlite3, sys

rep = sys.argv[1]
c = sqlite3.connect(rep)
cur = c.cursor()

# find verify (tile) instances: they mark the decode rounds
rows = cur.execute("""
SELECT k.start, k.end, k.gridX, k.gridY, s.value
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName
WHERE s.value LIKE '%flash_attn_tile%'
ORDER BY k.start
""").fetchall()
print(f"flash_attn_tile instances: {len(rows)}")
if rows:
    print(f"first start={rows[0][0]/1e9:.2f}s last end={rows[-1][1]/1e9:.2f}s gridX={rows[0][2]} gridY={rows[0][3]}")
    per_round = 17
    nrounds = len(rows)//per_round
    print(f"=> rounds (17 per verify): {len(rows)}/{per_round} = {nrounds}")

maxend = cur.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
if not rows:
    sys.exit(0)

# clean window: last 3 verify rounds
w0 = rows[-3*17][0] if len(rows) >= 51 else rows[0][0]
w1 = maxend
print(f"window: {(w1-w0)/1e9:.3f}s (last {min(3,nrounds)} rounds)")

def dump(title, where, params):
    q = f"""
    SELECT s.value, COUNT(*), SUM(k.end-k.start), MAX(k.gridX), MAX(k.gridY)
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName
    {where}
    GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT 25
    """
    print(f"\n=== {title} ===")
    tot = cur.execute(f"SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL k {where}", params).fetchone()[0]
    rows2 = cur.execute(q, params).fetchall()
    print(f"total GPU kernel time: {tot/1e6:.1f} ms")
    print(f"{'ms':>9} {'n':>6} {'gX':>7} {'gY':>6}  name")
    for name, n, ns, gx, gy in rows2:
        print(f"{ns/1e6:9.2f} {n:6d} {gx:7d} {gy:6d}  {name[:105]}")

dump("DECODE WINDOW (last rounds)", "WHERE k.start >= ? AND k.end <= ?", (w0, w1))

# per-round class summary
q = f"""
SELECT
  CASE
    WHEN s.value LIKE '%flash_attn_tile%' THEN 'fa_tile (verify)'
    WHEN s.value LIKE '%flash_attn_ext_vec%' THEN 'fa_vec (decode/draft)'
    WHEN s.value LIKE '%flash_attn_ext_f16%' THEN 'fa_mma'
    WHEN s.value LIKE '%flash_attn_stream_k_fixup%' THEN 'fa_fixup'
    WHEN s.value LIKE '%mul_mat_vec%' THEN 'mmvq'
    WHEN s.value LIKE '%gemm%' THEN 'gemm'
    WHEN s.value LIKE '%dequantize_block_q8_0_f16%' THEN 'v_mat (q8_0->f16)'
    WHEN s.value LIKE '%gated_delta_net%' THEN 'gdn'
    WHEN s.value LIKE '%ssm_conv%' THEN 'ssm_conv'
    WHEN s.value LIKE '%rms_norm%' THEN 'rms_norm'
    WHEN s.value LIKE '%silu%' OR s.value LIKE '%sigmoid%' THEN 'act'
    WHEN s.value LIKE '%cpy%' OR s.value LIKE '%concat%' OR s.value LIKE '%bin_bcast%' OR s.value LIKE '%set_rows%' THEN 'copy/misc'
    ELSE 'other'
  END AS cls, COUNT(*), SUM(k.end-k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.demangledName
WHERE k.start >= ? AND k.end <= ?
GROUP BY cls ORDER BY SUM(k.end-k.start) DESC
"""
print("\n=== DECODE WINDOW by class ===")
for cls, n, ns in cur.execute(q, (w0, w1)):
    print(f"{ns/1e6:9.2f} ms  n={n:6d}  {cls}")
