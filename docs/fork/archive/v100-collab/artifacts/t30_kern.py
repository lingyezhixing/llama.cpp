import sqlite3, sys

rep = sys.argv[1]
win_s = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
topn = int(sys.argv[3]) if len(sys.argv) > 3 else 20

c = sqlite3.connect(rep)
cur = c.cursor()

tabs = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
ktab = [t for t in tabs if t.upper().endswith('KERNEL')]
print("kernel tables:", ktab)
tab = ktab[0]

maxend = cur.execute(f"SELECT MAX(end) FROM {tab}").fetchone()[0]
minstart = cur.execute(f"SELECT MIN(start) FROM {tab}").fetchone()[0]
cur.execute(f"SELECT COUNT(*) FROM {tab}")
total = cur.fetchone()[0]
print(f"total kernels={total} span={(maxend-minstart)/1e9:.1f}s")

def dump(title, where, params=()):
    q = f"""
    SELECT s.value, COUNT(*), SUM(k.end-k.start), MAX(k.gridX), MAX(k.gridY)
    FROM {tab} k JOIN StringIds s ON s.id = k.demangledName
    {where}
    GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT {topn}
    """
    print(f"\n=== {title} ===")
    rows = cur.execute(q, params).fetchall()
    tot = sum(r[2] for r in rows)
    print(f"{'time_ms':>9} {'%':>6} {'n':>7} {'gridX':>6} {'gridY':>6}  name")
    for name, n, ns, gx, gy in rows:
        print(f"{ns/1e6:9.1f} {100.0*ns/max(tot,1):6.1f} {n:7d} {gx:6d} {gy:6d}  {name[:110]}")

dump("FULL RUN", "")
dump(f"LAST {win_s:.0f}s", "WHERE k.end >= ?", (maxend - win_s*1e9,))

print("\n=== FA / V-materialization in last window ===")
q = f"""
SELECT s.value, COUNT(*), SUM(k.end-k.start), MAX(k.gridX)
FROM {tab} k JOIN StringIds s ON s.id = k.demangledName
WHERE k.end >= ? AND (s.value LIKE '%flash_attn%' OR s.value LIKE '%dequantize%' OR s.value LIKE '%cpy%' OR s.value LIKE '%quantize%')
GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC
"""
for name, n, ns, gx in cur.execute(q, (maxend - win_s*1e9,)):
    print(f"{ns/1e6:9.2f} ms  n={n:6d} gridXmax={gx:6d}  {name[:120]}")
