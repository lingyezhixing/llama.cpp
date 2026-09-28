import numpy as np

def serial_norm_chunk(K, Q, V, beta, g, S_in):
    """Normalized within-chunk recurrence, S_out (unnormalized) + outputs (unnormalized, scale=1)."""
    L, D = K.shape
    Ac = np.exp(np.cumsum(g))
    Sn = np.array(S_in, dtype=np.float64)
    o = np.zeros((L, D))
    for t in range(L):
        kv = Sn.T @ K[t]
        delta = (V[t] - kv) * beta[t]
        Sn = Sn + np.outer(K[t], delta)
        o[t] = Ac[t] * (Sn.T @ Q[t])
    return o, Ac[-1] * Sn, Sn

def candidate(K, Q, V, beta, g):
    L, D = K.shape
    Ac = np.exp(np.cumsum(g)); Bt = beta / Ac
    KKT = K @ K.T
    Atri = np.linalg.inv(np.eye(L) + np.tril(np.outer(beta, np.ones(L)) * KKT, -1))
    W = Atri @ (beta[:, None] * K)
    U = Atri @ (Bt[:, None] * V)
    return Ac, Bt, Atri, W, U

rng = np.random.default_rng(5)
L, D = 6, 12
for trial in range(3):
    K = rng.normal(0, 1, (L, D)); Q = rng.normal(0, 1, (L, D)); V = rng.normal(0, 1, (L, D))
    beta = rng.uniform(0.1, 1.0, L); g = -rng.uniform(0.0, 0.5, L)
    S_in = rng.normal(0, 0.1, (D, D))
    o_ref, S_ref, Sn_ref = serial_norm_chunk(K, Q, V, beta, g, S_in)
    Ac, Bt, Atri, W, U = candidate(K, Q, V, beta, g)

    # state candidates
    Sn_new_A = S_in + K.T @ (U - W @ S_in)
    print(f'trial{trial}: state  S_in + K^T(U - W S_in) err = {np.abs(Sn_new_A - Sn_ref).max():.3e}')

    # output candidates
    mask = np.tril(np.ones((L, L)))
    Qt = Q - (mask * (Q @ K.T)) @ W
    o1 = Ac[:, None] * ((Q @ S_in) + (mask * (Q @ K.T)) @ U)
    o2 = Ac[:, None] * ((Q @ S_in) + (mask * (Qt @ W.T)) @ U)
    o3 = Ac[:, None] * ((Qt @ S_in) + (mask * (Qt @ W.T)) @ U)
    for nm, o in [('o1 mask(QK^T)@U', o1), ('o2 Q@S +(mask Qt W^T)@U', o2), ('o3 Qt@S + ...', o3)]:
        print(f'          output {nm:26s} err = {np.abs(o - o_ref).max():.3e}')
