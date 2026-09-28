import numpy as np

def serial(k, q, v, beta, g, S0, scale):
    T, D = k.shape
    S = np.array(S0, dtype=np.float64)
    o = np.zeros((T, D))
    for t in range(T):
        a = np.exp(g[t])
        kv = S.T @ k[t]
        delta = (v[t] - a * kv) * beta[t]
        S = a * S + np.outer(k[t], delta)
        o[t] = S.T @ q[t]
    return o * scale, S

def chunked(k, q, v, beta, g, S0, scale, L):
    T, D = k.shape
    assert T % L == 0
    S = np.array(S0, dtype=np.float64)     # unnormalized incoming state
    o = np.zeros((T, D))
    for c0 in range(0, T, L):
        sl = slice(c0, c0 + L)
        K = k[sl].astype(np.float64); Q = q[sl].astype(np.float64); V = v[sl].astype(np.float64)
        B = beta[sl].astype(np.float64); G = g[sl].astype(np.float64)
        Ac = np.exp(np.cumsum(G))                      # (L,) within-chunk cumulative decay
        Bt = B / Ac                                    # normalized betas
        KKT = K @ K.T
        M = np.eye(L) + np.tril(np.outer(B, np.ones(L)) * KKT, -1)
        Atri = np.linalg.inv(M)                        # triangular solve (checkpoint 2)
        W = Atri @ (B[:, None] * K)                    # (L,D)
        U = Atri @ (Bt[:, None] * V)                   # (L,D)
        # state:  Sn_out = (I - K^T W)(Sn + W^T U)
        Sn = S
        Sn_out = Sn - K.T @ (W @ Sn) + (W.T - K.T @ (W @ W.T)) @ U
        # outputs
        QK = Q @ K.T
        Qt = Q - np.tril(QK) @ W                       # transformed queries (diag included)
        o[sl] = scale * (Ac[:, None] * (Qt @ Sn + np.tril(Qt @ W.T) @ U))
        S = Ac[-1] * Sn_out
    return o, S

def run(D, L, T, seed, trials=3, verbose=True):
    worst_o = worst_s = 0.0
    for tr in range(trials):
        rng = np.random.default_rng(seed + tr)
        k = rng.normal(0, 1/np.sqrt(D), (T, D)); q = rng.normal(0, 1/np.sqrt(D), (T, D))
        v = rng.normal(0, 1, (T, D))
        beta = rng.uniform(0.1, 1.0, T)
        g = -rng.uniform(0.0, 0.5, T)
        S0 = rng.normal(0, 0.1, (D, D))
        scale = 1.0/np.sqrt(D)
        o_ref, S_ref = serial(k, q, v, beta, g, S0, scale)
        o_c, S_c = chunked(k, q, v, beta, g, S0, scale, L)
        eo = np.abs(o_c - o_ref).max() / (np.abs(o_ref).max() + 1e-30)
        es = np.abs(S_c - S_ref).max() / (np.abs(S_ref).max() + 1e-30)
        worst_o = max(worst_o, eo); worst_s = max(worst_s, es)
        if verbose:
            print(f'  D={D} L={L} T={T} tr{tr}: rel_err o={eo:.3e} S={es:.3e}')
    return worst_o, worst_s

print('=== small ==='); run(16, 8, 32, 100)
print('=== medium ==='); run(64, 32, 128, 200)
print('=== target shape (S_v=128, L=64) ==='); w = run(128, 64, 256, 300, trials=2)
print('worst rel err:', w)
