import csv
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe"
T = r"<TEMP>\v100"


def trace_rows(rep):
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
    return rows


def report(name):
    rows = trace_rows(T + "\\t19_dec128_" + ("STOCKb" if name == "STOCK" else name) + ".nsys-rep")
    if not rows:
        print(name, ": no rows"); return None
    big = [r for r in rows if r[1] > 5e6]
    split = max(r[0] + r[1] for r in big) if big else 0.0
    dec = sorted([r for r in rows if r[0] >= split])
    busy = sum(r[1] for r in dec)
    span = (dec[-1][0] + dec[-1][1]) - dec[0][0]
    gaps = span - busy
    print("=== %s ===" % name)
    print("  decode kernels: %d  GPU busy: %.1f ms  wall span: %.1f ms  gaps: %.1f ms (%.1f%%)" %
          (len(dec), busy / 1e6, span / 1e6, gaps / 1e6, 100.0 * gaps / span))
    agg = {}
    for st, dur, k in dec:
        key = re.sub(r"<.*", "", k)[:70]
        a = agg.setdefault(key, [0, 0.0])
        a[0] += 1
        a[1] += dur
    for k, (n, s) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]:
        print("    %8.2f ms  n=%-7d %s" % (s / 1e6, n, k))
    return busy, span


a = report("OURS")
b = report("STOCK")
if a and b:
    print("decode busy: OURS %.1f vs STOCK %.1f -> %+.2f%%" % (a[0] / 1e6, b[0] / 1e6, 100.0 * (a[0] / b[0] - 1.0)))
    print("decode wall: OURS %.1f vs STOCK %.1f -> %+.2f%%" % (a[1] / 1e6, b[1] / 1e6, 100.0 * (a[1] / b[1] - 1.0)))
