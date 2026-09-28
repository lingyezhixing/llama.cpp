import numpy as np
rng = np.random.default_rng(11)
L, D = 4, 8
K = rng.normal(0, 1, (L, D)); V = rng.normal(0, 1, (L, D)); beta = rng.uniform(0.1, 1.0, L)
g = -rng.uniform(0.0, 0.5, L); S0 = rng.normal(0, 0.1, (D, D))
Ac = np.exp(np.cumsum(g)); Bt = beta / Ac

P = [np.eye(D) - beta[t]*np.outer(K[t], K[t]) for t in range(L)]
Qs = [np.eye(D)]
for t in range(L):
    Qs.append(P[t] @ Qs[-1])          # Qs[t] = P_t ... P_1

# --- direct expansion of the increment: sum_t (P_L..P_{t+1}) d_t ---
inc_direct = np.zeros((D, D))
for t in range(L):
    prod = np.eye(D)
    for s in range(t+1, L):
        prod = P[s] @ prod            # P_L ... P_{t+1}
    inc_direct += prod @ (Bt[t] * np.outer(K[t], V[t]))

# --- identity side: Q_L @ sum_t Bt_t * (Q_t^{-1} k_t) (x) v_t ---
inc_ident = np.zeros((D, D))
for t in range(L):
    wt = np.linalg.solve(Qs[t+1], K[t])     # Q_t^{-1} k_t   (Q_t = P_t..P_1 = Qs[t+1])
    inc_ident += Bt[t] * np.outer(wt, V[t])
inc_ident = Qs[L] @ inc_ident
print('increment: direct vs identity =', np.abs(inc_direct - inc_ident).max())

# --- closed form candidates for sum_t Bt_t * w_t (x) v_t (call it Sigma) ---
Sigma = np.zeros((D, D))
for t in range(L):
    wt = np.linalg.solve(Qs[t+1], K[t])
    Sigma += Bt[t] * np.outer(wt, V[t])
KKT = K @ K.T
M = np.eye(L) + np.tril(np.outer(beta, np.ones(L)) * KKT, -1)
Atri = np.linalg.inv(M)
cands = {
    'Atri@diag(Bt)@V': Atri @ (Bt[:, None] * V),
    'Atri@diag(beta)@V': Atri @ (beta[:, None] * V),
}
for nm, U in cands.items():
    # hypothesis: Sigma = K^T @ U   hmm
    cand = K.T @ U
    print(f'  Sigma = K^T @ ({nm:16s}) err = {np.abs(cand - Sigma).max():.3e}')
    # hypothesis: Sigma = Wt^T @ U with Wt = the "w matrix" from that form
    Wand = Atri @ (Bt[:, None] * K)
    cand2 = Wand.T @ U
    print(f'  Sigma = ({nm:16s})^T-key @ U err = {np.abs(cand2 - Sigma).max():.3e}')

# --- what IS the correct U for Sigma = K^T U ?  (solve least squares: U = pinv(K^T) Sigma) ---
U_solved = np.linalg.pinv(K.T) @ Sigma
print('U_solved vs Atri@diag(Bt)@V :', np.abs(U_solved - Atri @ (Bt[:, None]*V)).max())
print('U_solved vs Atri@diag(beta)@V:', np.abs(U_solved - Atri @ (beta[:, None]*V)).max())
print('U_solved rows:')
print(np.round(U_solved, 4))
print('Atri@diag(Bt)@V rows:')
print(np.round(Atri @ (Bt[:, None]*V), 4))
