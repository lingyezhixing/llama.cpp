import csv
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe"
T = r"<TEMP>\v100"


def rows_of(name):
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
    return rows


for name in ("OURS", "STOCK"):
    rows = rows_of(name)
    t0 = rows[0][0]
    # per 10s bucket: kernel count and GPU busy, plus max kernel
    buckets = {}
    for st, dur, k in rows:
        b = int((st - t0) / 10e9)
        e = buckets.setdefault(b, [0, 0.0, 0.0])
        e[0] += 1
        e[1] += dur
        e[2] = max(e[2], dur)
    print("=== %s (span %.1fs) ===" % (name, (rows[-1][0] + rows[-1][1] - t0) / 1e9))
    line = []
    for b in sorted(buckets):
        n, s, mx = buckets[b]
        line.append("%ds:n=%d,busy=%.0fms,max=%.0fms" % (b * 10, n, s / 1e6, mx / 1e6))
    print("  " + "\n  ".join(line[-14:]))
