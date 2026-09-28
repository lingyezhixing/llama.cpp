import numpy as np, os

T = os.path.join(os.environ['TEMP'], 'v100')
D, L, H, NT, COLS, WP = 128, 32, 32, 512, 32, 33
DBSTATEOFF = D*WP + COLS*WP + L*WP + L*L + L*COLS + L*COLS + L*L + 2*L
tt = 32                                     # single chunk test

f = open(os.path.join(T, 't03_dbg.bin'), 'rb')
def rd(shape):
    a = np.frombuffer(f.read(4*int(np.prod(shape))), dtype=np.float32).reshape(shape).astype(np.float64)
    return a
q = rd((NT, H, D)); k = rd((NT, H, D)); v = rd((H, NT, D))
g = rd((H, NT)); b = rd((H, NT)); s_in = rd((H, D, D))
WT = rd((D, WP))[:, :L]; UT = rd((COLS, WP))[:, :L]; RT = rd((L, WP))[:, :L]
QK = rd((L, L)); T1 = rd((L, COLS)); OS = rd((L, COLS)); C = rd((L, L)); AcBt = rd((2, L))
Ac, Bt = AcBt[0], AcBt[1]
sdump = rd(((D//COLS)*COLS, D))            # (D, D) in M layout: [j][i]

# expected state after one chunk (formula, from the verified intermediates)
h = 0
K = k[0:L, h, :]; V = v[h, 0:L, :]; bb = b[h, 0:L]; gg = g[h, 0:L]
Ac_e = np.exp(np.cumsum(gg)); Bt_e = bb/Ac_e
KKT = K @ K.T
A_beta = np.linalg.inv(np.eye(L) + np.tril(np.outer(bb, np.ones(L))*KKT, -1)) @ np.diag(bb)
W_e = A_beta @ K; U_e = A_beta @ (np.outer(Bt_e, np.ones(D))*V)
S0 = s_in[h].T                                   # S_math[i][j]
Sn_new = S0 + K.T @ (U_e - W_e @ S0)
S_out_formula = Ac_e[-1]*Sn_new                  # S_math[i][j]
M_formula = S_out_formula.T                      # M[j][i]

# true serial state after tt tokens
S = s_in[h].T.copy()
for t in range(tt):
    a = np.exp(gg[t]); kv = S.T @ K[t]
    S = a*S + np.outer(K[t], (V[t] - a*kv)*bb[t])
M_true = S.T

print('formula vs true (numpy):', np.abs(M_formula - M_true).max())
print('kernel dump vs formula  :', np.abs(sdump - M_formula).max())
print('kernel dump vs true     :', np.abs(sdump - M_true).max())
print('kernel dump vs (Ac*Sn)  hmm: scale of dump =', np.abs(sdump).max(), ' formula =', np.abs(M_formula).max())
# where is the error concentrated?
diff = np.abs(sdump - M_true)
print('err by column block: ', [float(diff[i*COLS:(i+1)*COLS].max()) for i in range(D//COLS)])
print('err by row (first 8):', [float(diff[:, i].max()) for i in range(8)])
f.close()
