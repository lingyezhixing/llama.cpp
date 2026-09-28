import numpy as np
M, K = 17408, 5120
raw = np.fromfile(r"<TEMP>\v100\q6k_gate.raw", dtype=np.uint8).reshape(M, K//256, 210)
pk  = np.fromfile(r"<TEMP>\v100\q6k_gate_packed112.raw", dtype=np.uint8).reshape(K//128, M, 112)

def truth(row, H, l):
    sb = raw[row, H//2]
    e = 128*(H%2) + l          # element index within the super-block (H is the half index globally)
    ql_ = sb[0:128]; qh_ = sb[128:192]; sc = sb[192:208].view(np.int8)
    d = np.float32(sb[208:210].view(np.float16)[0])
    qb = int(ql_[(e & 63) + ((e >> 7) << 6)])
    hbv = int(qh_[(e & 31) + ((e >> 7) << 5)])
    shift = 2*((e >> 5) & 3)
    hi = ((e >> 6) & 1) != 0
    nib = (qb >> 4) & 0xF if hi else qb & 0xF
    q6 = (nib | (((hbv >> shift) & 3) << 4)) - 32
    return float(d) * float(sc[e >> 4]) * q6

def from_packed(row, H, l):
    base = pk[H, row]                      # 112 bytes: ql[64] qh[32] scales[8] d[2]
    s = l // 32; q = (l % 32) // 8; i = l % 8
    l0 = 32*s + 8*q
    dd = np.float32(base[104:106].view(np.float16)[0])
    ss = np.float32(base[96 + 2*s + (q >> 1)])    # int8 -> but stored as uint8; fix sign
    ss = np.float32(np.int8(base[96 + 2*s + (q >> 1)]))
    qbv = int(base[(l0 & 63) + i])
    hbv = int(base[64 + 8*q + i])
    nib = (qbv >> 4) & 0xF if s >= 2 else qbv & 0xF
    q6 = (nib | (((hbv >> (2*s)) & 3) << 4)) - 32
    return float(dd) * float(ss) * q6

bad = 0; first = []
for row in (0, 1, 12345):
    for H in range(8):
        for l in range(0, 128, 7):
            a = from_packed(row, H, l); b = truth(row, H, l)
            if abs(a-b) > 1e-6:
                bad += 1
                if len(first) < 10: first.append((row, H, l, a, b))
print('formula check (3 rows x 8 halves x 19 samples): bad =', bad)
for f in first: print('   row %d half %d l %d packed %.6f truth %.6f' % f)
