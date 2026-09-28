import numpy as np, os, importlib.util, sys

T_ = os.path.join(os.environ['TEMP'], 'v100')
spec = importlib.util.spec_from_file_location('m4', os.path.join(T_, 't03_math4.py'))
# t03_math4 runs tests on import; capture its functions by exec of the file with run disabled
src = open(os.path.join(T_, 't03_math4.py'), encoding='utf-8').read()
ns = {}
exec(src.split("rng = np.random.default_rng(23)")[0], ns)   # only the function defs
chunked, serial_norm, full_serial, full_chunked = ns['chunked'], ns['serial_norm'], ns['full_serial'], ns['full_chunked']

D, L, H, NT = 128, 32, 32, 512
f = open(os.path.join(T_, 't03_dbg.bin'), 'rb')
def rd(shape):
    return np.frombuffer(f.read(4*int(np.prod(shape))), dtype=np.float32).reshape(shape).astype(np.float64)
q = rd((NT, H, D)); k = rd((NT, H, D)); v = rd((H, NT, D)); g = rd((H, NT)); b = rd((H, NT)); s_in = rd((H, D, D))
f.close()

h = 0
K = k[:L, h, :]; Q = q[:L, h, :]; V = v[h, :L, :]; bb = b[h, :L]; gg = g[h, :L]; S0 = s_in[h].T
o_n, S_n = serial_norm(K, Q, V, bb, gg, S0)
o_c, S_c = chunked(K, Q, V, bb, gg, S0, 1.0)
print('[orig module] chunked vs serial_norm: state', np.abs(S_c - S_n).max(), ' out', np.abs(o_c*1.0 - o_n).max())

# same but with synthetic data from the original test distribution
rng = np.random.default_rng(7)
Ls, Ds = 32, 128
Ks = rng.normal(0, 1/np.sqrt(Ds), (Ls, Ds)); Qs = rng.normal(0, 1/np.sqrt(Ds), (Ls, Ds)); Vs = rng.normal(0, 1, (Ls, Ds))
bs = rng.uniform(0.1, 1.0, Ls); gs = -rng.uniform(0.0, 0.5, Ls); S0s = rng.normal(0, 0.1, (Ds, Ds))
o1, S1 = serial_norm(Ks, Qs, Vs, bs, gs, S0s)
o2, S2 = chunked(Ks, Qs, Vs, bs, gs, S0s, 1.0)
print('[synthetic ] chunked vs serial_norm: state', np.abs(S2 - S1).max(), ' out', np.abs(o2 - o1).max())

# diagnose the state formula alone on harness data
KKT = K @ K.T
Ac = np.exp(np.cumsum(gg)); Bt = bb/Ac
A_beta = np.linalg.inv(np.eye(L) + np.tril(np.outer(bb, np.ones(L))*KKT, -1)) @ np.diag(bb)
W = A_beta @ K; U = A_beta @ (np.outer(Bt, np.ones(D))*V)
Sn_formula = S0 + K.T @ (U - W @ S0)
Sn_true = S_n/Ac[-1]
print('[state only] formula vs serial_norm/Ac: ', np.abs(Sn_formula - Sn_true).max(), ' scale', np.abs(Sn_true).max())
print('   Sn_formula scale', np.abs(Sn_formula).max(), ' U scale', np.abs(U).max(), ' W scale', np.abs(W).max())
print('   Ac[-1]', Ac[-1], 'Bt max', Bt.max())
# where does it break? compare per-token prefix formula for the first few tokens
for Ltest in [1, 2, 4]:
    Kp = K[:Ltest]; Vp = V[:Ltest]; bb2 = bb[:Ltest]; gg2 = gg[:Ltest]
    Ac2 = np.exp(np.cumsum(gg2)); Bt2 = bb2/Ac2
    KKT2 = Kp@Kp.T
    Ab2 = np.linalg.inv(np.eye(Ltest) + np.tril(np.outer(bb2, np.ones(Ltest))*KKT2, -1)) @ np.diag(bb2)
    W2 = Ab2@Kp; U2 = Ab2@(np.outer(Bt2, np.ones(D))*Vp)
    Snf = S0 + Kp.T@(U2 - W2@S0)
    Snt = S0.copy()
    for t in range(Ltest):
        kv = Snt.T@Kp[t]; Snt = Snt - bb2[t]*np.outer(Kp[t], kv) + Bt2[t]*np.outer(Kp[t], Vp[t])
    print(f'   L={Ltest}: formula vs true prefix = {np.abs(Snf-Snt).max():.3e}')
