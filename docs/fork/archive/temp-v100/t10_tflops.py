import sqlite3, sys

# FLOPs per FA launch group (16 launches = 16 attention layers = 1 ubatch of 512 tokens)
# QK^T + PV, causal: per layer 4 * head_dim * n_head * (n_q*d_prev + n_q*(n_q+1)/2)
N_HEAD, HEAD_DIM, N_LAYER, NQ = 24, 256, 16, 512
def group_flops(d_prev):
    return N_LAYER * 4 * HEAD_DIM * N_HEAD * (NQ * d_prev + NQ * (NQ + 1) // 2)

def run(path, kind):
    con = sqlite3.connect(path); cur = con.cursor()
    rows = cur.execute("""
        select k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k
        where (select value from StringIds where id = k.demangledName) like '%flash_attn%'
        order by k.start
    """).fetchall()
    t0 = cur.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    groups = [rows[i:i+16] for i in range(0, len(rows), 16)]
    print("=" * 96)
    print(f"[{kind}] {path.split(chr(92))[-1]}  groups={len(groups)} (each = 1 ubatch x 16 layers)")
    print("=" * 96)
    print(f"{'grp':>4} {'phase':>6} {'depth':>7} {'start_ms':>10} {'gp_us':>10} {'TFLOP':>8} {'TF/s':>7}")
    for i, gp in enumerate(groups):
        if kind == 'd32k' and i >= 64:
            phase, d_prev = 'eval', 32768 + 512 * (i - 64)
        elif kind == 'd32k':
            phase, d_prev = 'fill', 512 * i
        else:
            phase, d_prev = 'fwd%d' % (i // 64), 512 * (i % 64)
        us = sum(e - s for s, e in gp)
        fl = group_flops(d_prev)
        print(f"{i:>4} {phase:>6} {d_prev+512:>7} {(gp[0][0]-t0)/1e6:>10.1f} {us:>10.1f} {fl/1e12:>8.2f} {fl/us/1e6:>7.2f}")
    con.close()

run(sys.argv[1], sys.argv[2])
