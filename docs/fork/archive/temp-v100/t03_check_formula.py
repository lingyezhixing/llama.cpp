import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.environ['TEMP'], 'v100'))

T_ = os.path.join(os.environ['TEMP'], 'v100')
D, L, H, NT, COLS, WP = 128, 32, 32, 512, 32, 33
f = open(os.path.join(T_, 't03_dbg.bin'), 'rb')
def rd(shape):
    return np.frombuffer(f.read(4*int(np.prod(shape))), dtype=np.float32).reshape(shape).astype(np.float64)
q = rd((NT, H, D)); k = rd((NT, H, D)); v = rd((H, NT, D))
g = rd((H, NT)); b = rd((H, NT)); s_in = rd((H, D, D))
f.close()

def serial_norm(K, Q, V, beta, gg, S_in):
    L_, D_ = K.shape
    Ac = np.exp(np.cumsum(gg)); Bt = beta/Ac
    Sn = S_in.copy(); o = np.zeros((L_, D_))
    for t in range(L_):
        kv = Sn.T @ K[t]
        Sn = Sn - beta[t]*np.outer(K[t], kv) + Bt[t]*np.outer(K[t], V[t])
        o[t] = Ac[t]*(Sn.T @ Q[t])
    return o, Ac[-1]*Sn

def chunked(K, Q, V, beta, gg, S_in, scale):
    L_, D_ = K.shape
    Ac = np.exp(np.cumsum(gg)); Bt = beta/Ac
    KKT = K @ K.T
    A_beta = np.linalg.inv(np.eye(L_) + np.tril(np.outer(beta, np.ones(L_))*KKT, -1)) @ np.diag(beta)
    W = A_beta @ K; U = A_beta @ (np.outer(Bt, np.ones(D_))*V)
    Sn_new = S_in + K.T @ (U - W @ S_in)
    QK = Q @ K.T
    Qtil = Q - np.tril(QK, 0) @ W
    C = QK - np.tril(QK, 0) @ A_beta @ np.tril(KKT, -1)
    O = scale*(Ac[:, None]*(Qtil @ S_in + np.tril(C) @ (np.outer(Bt, np.ones(D_))*V)))
    return O, Ac[-1]*Sn_new

def true_serial(K, Q, V, beta, gg, S_in):
    L_, D_ = K.shape
    S = S_in.copy()
    for t in range(L_):
        a = np.exp(gg[t]); kv = S.T @ K[t]
        S = a*S + np.outer(K[t], (V[t] - a*kv)*beta[t])
    return S

h = 0
K = k[:L, h, :]; Q = q[:L, h, :]; V = v[h, :L, :]; bb = b[h, :L]; gg = g[h, :L]
S0 = s_in[h].T
S_true = true_serial(K, Q, V, bb, gg, S0)
_, S_norm = serial_norm(K, Q, V, bb, gg, S0)
_, S_chk = chunked(K, Q, V, bb, gg, S0, 1.0)
print('serial_norm vs true :', np.abs(S_norm - S_true).max())
print('chunked     vs true :', np.abs(S_chk  - S_true).max())
print('scale: true=%.4f chunked=%.4f' % (np.abs(S_true).max(), np.abs(S_chk).max()))
print('beta stats:', bb.min(), bb.max(), ' k dot sample:', (K[0]@K[1]))
# conditioning check on the triangular system
KKT = K @ K.T
M = np.tril(np.outer(bb, np.ones(L))*KKT, -1)
print('cond(I+M) = %.3e' % np.linalg.cond(np.eye(L)+M))
