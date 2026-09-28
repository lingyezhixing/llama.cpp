"""T19 final figures: OURS vs STOCK on V100 (ub512).

Data: llama-bench, NEWBASE e6ab7c1a4, same session, per point -r 3 (short) / -r 2 (long).
STOCK  = D:\\LLM\\Backend\\llama.cpp    (ggml-cuda.dll 976E2CABF9EADC7D)
OURS   = D:\\LLM\\Backend\\llama.cpp-my (ggml-cuda.dll 7F1B9B2403438803)

Corrections after the decode probe (2026-09-23, see RESULTS "T19 遗留项排查"):
  tg d32768 : matrix 18.93/21.18 was a measurement-state artifact; retest (alternating, r=2 x2)
              23.02/22.82 vs 22.87/22.87 -> parity. Values below are the retest means.
  tg d131072: retest (r=2 x2, alternating) 10.94/10.58 vs 11.33/11.80 -> OURS slower, magnitude
              unstable (thermal state); kernel-level diff not reproducible (STOCK nsys trace truncated).
              Values below are the retest means, still flagged as unresolved.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# point -> (OURS, STOCK)
PP = {512: (948.48, 872.78), 4096: (929.91, 847.45), 8192: (907.54, 814.28),
      32768: (793.47, 663.29), 131072: (531.52, 392.10)}
TG = {0: (26.61, 26.65), 4096: (25.47, 25.56), 8192: (24.05, 23.75),
      32768: (22.92, 22.87), 131072: (10.76, 11.57)}
TG_NOTE = {32768: "matrix 18.93/21.18 excluded (artifact); retest parity",
           131072: "unresolved (matrix -9.4%, retest -3.4/-10.3%)"}

with open(os.path.join(OUT, "t19_data.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kind", "x", "ours", "stock", "delta_percent", "note"])
    for x, (a, b) in PP.items():
        w.writerow(["pp", x, a, b, round(100.0*(a/b - 1.0), 2), ""])
    for x, (a, b) in TG.items():
        w.writerow(["tg128", x, a, b, round(100.0*(a/b - 1.0), 2), TG_NOTE.get(x, "")])

fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.array(sorted(PP), dtype=float)
o = np.array([PP[k][0] for k in sorted(PP)])
s = np.array([PP[k][1] for k in sorted(PP)])
ax.semilogx(x, o, "o-", label="OURS")
ax.semilogx(x, s, "s--", label="STOCK")
for xi, (a, b) in PP.items():
    ax.annotate("%+.1f%%" % (100.0*(a/b - 1.0)), (xi, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.set_xlabel("prompt tokens (pp)")
ax.set_ylabel("t/s")
ax.set_title("Prefill, ub512 (V100, fa on, ctv q8_0)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "t19_pp.png"), dpi=300)

fig, ax = plt.subplots(figsize=(6.4, 4.2))
x = np.array(sorted(TG), dtype=float)
x[0] = 512.0  # d=0 plotted at 512 for the log axis
o = np.array([TG[k][0] for k in sorted(TG)])
s = np.array([TG[k][1] for k in sorted(TG)])
ax.semilogx(x, o, "o-", label="OURS")
ax.semilogx(x, s, "s--", label="STOCK")
for xi, (a, b) in TG.items():
    ax.annotate("%+.1f%%" % (100.0*(a/b - 1.0)), (max(xi, 512), a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.annotate("unresolved", (131072, TG[131072][0]), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8, color="crimson")
ax.set_xlabel("context depth (d, 512 = d0)")
ax.set_ylabel("t/s (tg128)")
ax.set_title("Decode, ub512 (V100, fa on, ctv q8_0)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "t19_tg.png"), dpi=300)

fig, ax = plt.subplots(figsize=(7.6, 4.2))
labels = ["pp512", "pp4096", "pp8192", "pp32768", "pp131072", "tg d0", "tg d4096", "tg d8192", "tg d32768", "tg d131072"]
deltas = [100.0*(PP[k][0]/PP[k][1] - 1.0) for k in sorted(PP)] + [100.0*(TG[k][0]/TG[k][1] - 1.0) for k in sorted(TG)]
colors = ["tab:blue"]*5 + ["tab:green"]*5
bars = ax.bar(labels, deltas, color=colors)
bars[9].set_hatch("//")
for i, d in enumerate(deltas):
    ax.annotate("%+.1f%%" % d, (i, d), textcoords="offset points", xytext=(0, 3 if d >= 0 else -11), ha="center", fontsize=8)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("OURS vs STOCK (%)")
ax.set_title("Speed delta, ub512 (blue = prefill, green = decode; hatched = unresolved)")
plt.xticks(rotation=30, ha="right")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "t19_delta.png"), dpi=300)

print("updated t19_pp.png / t19_tg.png / t19_delta.png / t19_data.csv")
