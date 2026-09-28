import numpy as np
rng = np.random.default_rng(11)
L, D = 4, 8

K = rng.normal(0, 1, (L, D)); V = rng.normal(0, 1, (L, D)); beta = rng.uniform(0.1, 1.0, L)
g = -rng.uniform(0.0, 0.5, L); S0 = rng.normal(0, 0.1, (D, D))
Ac = np.exp(np.cumsum(g)); Bt = beta / Ac

# exact prefix products Q_t = P_t ... P_1  (t = 1..L, Q_0 = I)
Qs = [np.eye(D)]
for t in range(L):
    P = np.eye(D) - beta[t] * np.outer(K[t], K[t])
    Qs.append(P @ Qs[-1])

# ground-truth w_t = Q_t^{-1} k_t
w_true = np.array([np.linalg.solve(Qs[t+1], K[t]) for t in range(L)])   # (L,D)
print('w_true row0:', np.round(w_true[0], 4))

# candidates for W
KKT = K @ K.T
M = np.eye(L) + np.tril(np.outer(beta, np.ones(L)) * KKT, -1)
Atri = np.linalg.inv(M)
W_cand = {
    'A@diag(beta)@K': Atri @ (beta[:, None] * K),
    'A@diag(beta/Ac)@K': Atri @ (Bt[:, None] * K),
    'A@K*(beta)': (Atri @ K) * beta[:, None],
}
for nm, W in W_cand.items():
    err = np.abs(W - w_true).max()
    print(f'  W = {nm:20s}  max_err vs w_true = {err:.3e}')

# identity check: SL == QL (S0 + sum_t Bt_t * w_t (x) v_t)
SL_true = Qs[L] @ S0.copy()
for t in range(L):
    SL_true = SL_true + Bt[t] * np.outer(w_true[t], V[t])   # hmm: check sign/order later
SL_true = Qs[L] @ (S0 + sum(Bt[t]*np.outer(w_true[t], V[t]) for t in range(L)))
# true normalized state via serial recurrence
Sn = S0.copy()
for t in range(L):
    S_prev = Ac[t] * Sn if t > 0 else Sn      # unnormalize back? simpler: run the normalized recurrence directly
Sn = S0.copy()
for t in range(L):
    kv = Sn.T @ K[t]
    delta = (V[t] - kv) * beta[t]             # normalized: a=1
    Sn = Sn + np.outer(K[t], delta)
print('identity err (SL):', np.abs(SL_true - Sn).max())
print()
# also check the full WY product identity used before
print('QL vs I - K^T A beta K:', np.abs(Qs[L] - (np.eye(D) - K.T @ (Atri @ (beta[:,None]*K)))).max())
