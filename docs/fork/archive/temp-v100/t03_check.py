import numpy as np, os, sys

T = os.path.join(os.environ['TEMP'], 'v100')
D, L, H, NT, COLS, WP = 128, 32, 32, 512, 32, 33
f = open(os.path.join(T, 't03_dbg.bin'), 'rb')
def rd(shape):
    n = int(np.prod(shape))
    a = np.frombuffer(f.read(4*n), dtype=np.float32).reshape(shape).astype(np.float64)
    return a
q = rd((NT, H, D)); k = rd((NT, H, D)); v = rd((H, NT, D))
g = rd((H, NT)); b = rd((H, NT)); s_in = rd((H, D, D))
WT = rd((D, WP))[:, :L]               # WT[m][t] = W[t][m]
UT = rd((COLS, WP))[:, :L]            # UT[jj][t] = U[t][jj]   (col-block 0 only)
RT = rd((L, WP))[:, :L]               # RT[li][m] = R[m][li]
QK = rd((L, L))
T1 = rd((L, COLS))
OS = rd((L, COLS))
C  = rd((L, L))
AcBt = rd((2, L))
Ac, Bt = AcBt[0], AcBt[1]
f.close()

# expected for head 0, chunk 0, col-block 0
h, t0 = 0, 0
K = k[t0:t0+L, h, :]                 # (L,D)
Q = q[t0:t0+L, h, :]
V = v[h, t0:t0+L, :]
bb = b[h, t0:t0+L]
gg = g[h, t0:t0+L]
Ac_e = np.exp(np.cumsum(gg))
Bt_e = bb/Ac_e
KKT = K @ K.T
A_beta = np.linalg.inv(np.eye(L) + np.tril(np.outer(bb, np.ones(L))*KKT, -1)) @ np.diag(bb)
W_e = A_beta @ K
U_e = A_beta @ (np.outer(Bt_e, np.ones(D))*V)
R_e = A_beta @ np.tril(KKT, -1)
S0 = s_in[h].T                       # S_math[i][j] = s_in[h][j][i]  (M[j][i])
T1_e = W_e @ S0                      # (L,D) -> T1[t][j]
QK_e = Q @ K.T
C_e = QK_e - np.tril(QK_e, 0) @ R_e
OS_e = (np.tril(np.ones((L,L))) * QK_e) @ T1_e[:, :COLS]   # OS[t][j] = sum_{s<=t} QK[t][s] T1[s][j]

def cmp(nm, a, b_):
    print(f'  {nm:28s} max_abs_err={np.abs(a-b_).max():.3e}   ref_scale={np.abs(b_).max():.3e}')

print(f'Ac  dump/expected: {Ac[0]:.6f} {Ac_e[0]:.6f} ... {Ac[-1]:.6f} {Ac_e[-1]:.6f}')
print(f'Bt  dump/expected: {Bt[0]:.6f} {Bt_e[0]:.6f}')
print('--- intermediate stages (head 0, chunk 0, COLS block 0) ---')
cmp('W', WT.T, W_e)                    # WT[m][t] -> W[t][m]
cmp('U (cols 0..31)', UT[:COLS].T, U_e[:, :COLS])
cmp('R', RT.T, R_e)                    # RT[li][m] -> R[m][li]
cmp('QK', QK, QK_e)
cmp('T1 (cols 0..31)', T1, T1_e[:, :COLS])
cmp('C', C, C_e)
cmp('OS (cols 0..31)', OS, OS_e[:, :COLS])
