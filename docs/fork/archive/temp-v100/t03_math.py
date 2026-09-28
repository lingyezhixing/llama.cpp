import numpy as np

rng = np.random.default_rng(1234)

def serial(k, q, v, beta, g, S0, scale):
    """Reference: exactly the current kernel's per-token recurrence.
    S_t = a_t*S_{t-1} + k_t (beta_t (v_t - a_t S_{t-1}^T k_t))^T ;  o_t = scale * S_t^T q_t
    """
    T, D = k.shape
    S = S0.astype(np.float64).copy()
    o = np.zeros((T, D))
    for t in range(T):
        a = np.exp(g[t])
        kv = S.T @ k[t]
        delta = (v[t] - a * kv) * beta[t]
        S = a * S + np.outer(k[t], delta)
        o[t] = S.T @ q[t]
    return o * scale, S

def chunked(k, q, v, beta, g, S0, scale, L, signs):
    """Chunked formulation. signs = (s1, s2) allow flipping candidate conventions."""
    T, D = k.shape
    assert T % L == 0
    s_lower, s_diag = signs
    S = S0.astype(np.float64).copy()
    o = np.zeros((T, D))
    for c0 in range(0, T, L):
        sl = slice(c0, c0 + L)
        K = k[sl].astype(np.float64); Q = q[sl].astype(np.float64); V = v[sl].astype(np.float64)
        B = beta[sl].astype(np.float64); G = g[sl].astype(np.float64)
        A = np.exp(np.cumsum(G))                    # cumulative decay within chunk
        Bt = B / A                                  # normalized betas (decay folded into the v term)
        KKT = K @ K.T                               # (L,L)
        # Tmat: strictly-lower convention
        M = np.tril(np.outer(B, np.ones(L)) * KKT, -1) * s_lower
        Tm = np.linalg.inv(np.eye(L) + M)
        # transformed value/key rows
        U = Tm @ (np.outer(Bt, np.ones(D)) * V)     # (L,D)  normalized values
        W = Tm @ (np.outer(B,  np.ones(D)) * K)     # (L,D)  transformed keys (beta, not Bt)
        # state update (normalized state Sn = S / A):
        Sn = S / A[-1]                              # hmm: state carried in is already unnormalized; normalize by A_end of previous chunk
        # per-chunk: Sn_end = Sn_start + K^T (U - W @ Sn_start)
        Sn_end = Sn + K.T @ (U - W @ Sn)
        # outputs: o_t = scale * A_t * ( q_t^T Sn_t ) with Sn_t = prefix state + intra part
        # intra: (Q K^T masked) @ U  with diagonal handled by s_diag
        attn = Q @ K.T                              # (L,L)
        mask = np.tril(np.ones((L, L)), -1) * s_lower + np.eye(L) * s_diag
        o_intra = (attn * mask) @ U
        o[sl] = scale * (A[:, None] * ((Q @ Sn) + o_intra))
        S = A[-1] * Sn_end
    return o, S

def run(D=16, L=8, T=32, seed=0, trials=3):
    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        k = rng.normal(0, 1, (T, D)); q = rng.normal(0, 1, (T, D)); v = rng.normal(0, 1, (T, D))
        beta = rng.uniform(0.1, 1.0, T)
        g = -rng.uniform(0.0, 0.5, T)
        S0 = rng.normal(0, 0.1, (D, D))
        scale = 1.0 / np.sqrt(D)
        o_ref, S_ref = serial(k, q, v, beta, g, S0, scale)
        for signs in [(1, 1), (-1, 1), (1, 0), (-1, 0)]:
            o_c, S_c = chunked(k, q, v, beta, g, S0, scale, L, signs)
            eo = np.abs(o_c - o_ref).max() / (np.abs(o_ref).max() + 1e-30)
            es = np.abs(S_c - S_ref).max() / (np.abs(S_ref).max() + 1e-30)
            print(f'  trial{trial} signs={signs}: rel_err o={eo:.3e} S={es:.3e}')
        print()

run()
