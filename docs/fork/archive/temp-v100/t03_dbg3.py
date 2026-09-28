import numpy as np
rng = np.random.default_rng(3)
L, D = 1, 4
K = rng.normal(0, 1, (L, D)); Q = rng.normal(0, 1, (L, D)); V = rng.normal(0, 1, (L, D))
beta = rng.uniform(0.1, 1.0, L); g = -rng.uniform(0.0, 0.5, L); S_in = rng.normal(0, 0.1, (D, D))

# reference
Ac = np.exp(np.cumsum(g)); Bt = beta / Ac
Sn = S_in.copy(); o_ref = np.zeros((L, D))
for t in range(L):
    kv = Sn.T @ K[t]
    Sn = Sn - beta[t]*np.outer(K[t], kv) + Bt[t]*np.outer(K[t], V[t])
    o_ref[t] = Ac[t]*(Sn.T @ Q[t])

# chunked pieces
KKT = K @ K.T
A_beta = np.linalg.inv(np.eye(L) + np.tril(np.outer(beta, np.ones(L))*KKT, -1)) @ np.diag(beta)
W = A_beta @ K
U = A_beta @ ((1.0/Ac)[:, None] * V)
Sn_new = S_in + K.T @ (U - W @ S_in)
QK = Q @ K.T
Qtil = Q - np.tril(QK, 0) @ W
C = QK - np.tril(QK, 0) @ A_beta @ np.triu(KKT, 1)
o_c = Ac[:, None] * (Qtil @ S_in + np.tril(C) @ (Bt[:, None]*V))

print('Ac', Ac, 'Bt', Bt)
print('o_ref ', o_ref)
print('o_c   ', o_c)
print('diff  ', o_c - o_ref)
# manual expected: A_t * [ (q - beta*((q.k))k)^T S_in + Bt*(q.k) v ]
t = 0
q, k, v = Q[t], K[t], V[t]
manual = Ac[t]*((q - beta[t]*(q@k)*k) @ S_in + Bt[t]*(q@k)*v)
print('manual', manual)
print('Qtil row', Qtil[0], ' C', C)
print('state diff', np.abs((Ac[-1]*Sn_new) - Sn).max())
