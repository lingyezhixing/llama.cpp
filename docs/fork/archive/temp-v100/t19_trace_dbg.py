import csv
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe"
T = r"<TEMP>\v100"

for name in ("OURS", "STOCK"):
    rep = T + "\\t19_dec128_" + name + ".nsys-rep"
    out = subprocess.run([NSYS, "stats", "--report", "cuda_gpu_trace", "--format", "csv", rep],
                         capture_output=True, text=True, errors="replace").stdout
    rows = []
    for line in out.splitlines():
        if not line[:1].isdigit():
            continue
        try:
            f = next(csv.reader([line]))
        except Exception:
            continue
        if len(f) < 16:
            continue
        rows.append((float(f[0]), float(f[1]), f[-1]))
    rows.sort()
    total_span = (rows[-1][0] + rows[-1][1] - rows[0][0]) / 1e9
    big = [r for r in rows if r[1] > 5e6]
    print("=== %s === rows=%d span=%.1fs big(>5ms)=%d" % (name, len(rows), total_span, len(big)))
    if big:
        print("  first big at %.1fs, last big at %.1fs" % (big[0][0] / 1e9, big[-1][0] / 1e9))
    print("  last 5 kernels:")
    for st, dur, k in rows[-5:]:
        print("    %8.2f s  %8.3f ms  %s" % (st / 1e9, dur / 1e6, k[:60]))
