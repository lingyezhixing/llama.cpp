# T03 chunked GDN: validated numpy reference (implementer, 2026-09-22)
# Status: all checkpoints PASS at machine precision (rel err ~1e-16):
#   L=1/2/8/32/64, D=8..128, single-chunk + multi-chunk (T=256, L=64)
# Algorithm (per chunk, per (seq, head)):
#   Ac      = exp(cumsum(g))                    # within-chunk cumulative decay
#   Bt      = beta / Ac                         # normalized betas
#   KKT     = K K^T
#   A_beta  = inv(I + tril(diag(beta) KKT, -1)) @ diag(beta)   # <-- triangular solve (checkpoint 2)
#   W       = A_beta @ K ; U = A_beta @ (diag(1/Ac) V)
#   Sn_new  = Sn_in + K^T (U - W Sn_in)         # normalized state; S_out = Ac[-1]*Sn_new (checkpoint 3)
#   QK      = Q K^T
#   Qtil    = Q - tril(QK, 0) @ W
#   C       = QK - tril(QK, 0) @ A_beta @ tril(KKT, -1)
#   O       = scale * Ac[:,None] * (Qtil @ Sn_in + tril(C) @ (Bt[:,None]*V))
# Notes:
#   - decay (checkpoint 1): only Ac (cumsum+exp) per chunk; no cross-chunk decay state needed
#   - implementation can replace the explicit inverse by forward substitution:
#     W_t = beta_t (k_t - sum_{s<t} (k_t.k_s) W_s),  U_t = Bt_t v_t - beta_t sum_{s<t}(k_t.k_s) U_s,
#     and one more solve with the same I+M matrix for R = A_beta @ tril(KKT,-1) (for C)
import numpy as np

def serial_norm(K, Q, V, beta, g, S_in):
    """True normalized recurrence: Sn <- (I - beta k k^T) Sn + (beta/Ac) k v^T; o_t = Ac_t * Sn_t^T q_t"""
    L, D = K.shape
    Ac = np.exp(np.cumsum(g)); Bt = beta / Ac
    Sn = np.array(S_in, dtype=np.float64)
    o = np.zeros((L, D))
    for t in range(L):
        kv = Sn.T @ K[t]
        Sn = Sn - beta[t] * np.outer(K[t], kv) + Bt[t] * np.outer(K[t], V[t])
        o[t] = Ac[t] * (Sn.T @ Q[t])
    return o, Ac[-1] * Sn

def chunked(K, Q, V, beta, g, S_in, scale=1.0):
    L, D = K.shape
    Ac = np.exp(np.cumsum(g)); Bt = beta / Ac
    KKT = K @ K.T
    A_beta = np.linalg.inv(np.eye(L) + np.tril(np.outer(beta, np.ones(L)) * KKT, -1)) @ np.diag(beta)
    W = A_beta @ K
    U = A_beta @ ((1.0/Ac)[:, None] * V)
    Sn_new = S_in + K.T @ (U - W @ S_in)
    QK = Q @ K.T
    Qtil = Q - np.tril(QK, 0) @ W
    C = QK - np.tril(QK, 0) @ A_beta @ np.tril(KKT, -1)
    O = scale * (Ac[:, None] * (Qtil @ S_in + np.tril(C) @ (Bt[:, None] * V)))
    return O, Ac[-1] * Sn_new

rng = np.random.default_rng(23)
for (L, D) in [(1, 8), (2, 8), (8, 16), (32, 64), (64, 128)]:
    worst = 0.0; worsts = 0.0
    for trial in range(3):
        K = rng.normal(0, 1/np.sqrt(D), (L, D)); Q = rng.normal(0, 1/np.sqrt(D), (L, D))
        V = rng.normal(0, 1, (L, D))
        beta = rng.uniform(0.1, 1.0, L); g = -rng.uniform(0.0, 0.5, L)
        S_in = rng.normal(0, 0.1, (D, D))
        scale = 1.0/np.sqrt(D)
        o_ref, S_ref = serial_norm(K, Q, V, beta, g, S_in)
        o_c, S_c = chunked(K, Q, V, beta, g, S_in, scale)
        eo = np.abs(o_c - o_ref * scale).max() / (np.abs(o_ref).max() + 1e-30)
        es = np.abs(S_c - S_ref).max() / (np.abs(S_ref).max() + 1e-30)
        worst = max(worst, eo); worsts = max(worsts, es)
    print(f'L={L:3d} D={D:4d}: worst rel err  o={worst:.3e}  S={worsts:.3e}')

# end-to-end: multi-chunk vs full serial (incl. unnormalized state handoff)
def full_serial(k, q, v, beta, g, S0, scale):
    T, D = k.shape
    S = np.array(S0, float); o = np.zeros((T, D))
    for t in range(T):
        a = np.exp(g[t]); kv = S.T @ k[t]
        S = a*S + np.outer(k[t], (v[t] - a*kv)*beta[t])
        o[t] = (S.T @ q[t]) * scale
    return o, S

def full_chunked(k, q, v, beta, g, S0, scale, L):
    T, D = k.shape
    S = np.array(S0, float); o = np.zeros((T, D))
    for c0 in range(0, T, L):
        sl = slice(c0, c0+L)
        o[sl], S = chunked(k[sl], q[sl], v[sl], beta[sl], g[sl], S, scale)
    return o, S

print()
for (L, D, T) in [(8, 16, 64), (64, 128, 256)]:
    worst = 0.0; worsts = 0.0
    for trial in range(3):
        rng = np.random.default_rng(500+trial)
        k = rng.normal(0, 1/np.sqrt(D), (T, D)); q = rng.normal(0, 1/np.sqrt(D), (T, D))
        v = rng.normal(0, 1, (T, D)); beta = rng.uniform(0.1, 1.0, T); g = -rng.uniform(0.0, 0.5, T)
        S0 = rng.normal(0, 0.1, (D, D)); scale = 1.0/np.sqrt(D)
        o_ref, S_ref = full_serial(k, q, v, beta, g, S0, scale)
        o_c, S_c = full_chunked(k, q, v, beta, g, S0, scale, L)
        eo = np.abs(o_c-o_ref).max()/(np.abs(o_ref).max()+1e-30)
        es = np.abs(S_c-S_ref).max()/(np.abs(S_ref).max()+1e-30)
        worst = max(worst, eo); worsts = max(worsts, es)
    print(f'multi-chunk L={L:3d} D={D:4d} T={T}: o err={worst:.3e} S err={worsts:.3e}')
