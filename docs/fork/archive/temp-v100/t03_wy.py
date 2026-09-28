import numpy as np
rng = np.random.default_rng(7)
L, D = 4, 8

def check():
    K = rng.normal(0, 1, (L, D))       # row t = k_t^T
    beta = rng.uniform(0.1, 1.0, L)
    # exact product Q = P_L ... P_1 with P_t = I - beta_t k_t k_t^T
    Q = np.eye(D)
    for t in range(L):
        P = np.eye(D) - beta[t] * np.outer(K[t], K[t])
        Q = P @ Q                       # P_L ... P_1 order
    KKT = K @ K.T                       # (L,L), KKT[i,j] = k_i . k_j
    Bd = np.diag(beta)
    # candidates for the WY form  Q = I - K^T A K
    cands = {}
    cands['tril_lower_diagB'] = np.eye(L) + np.tril(Bd @ KKT, -1)
    cands['tril_lower_B@strict'] = np.eye(L) + Bd @ np.tril(KKT, -1)
    cands['tril_upper'] = np.eye(L) + np.triu(Bd @ KKT, 1)
    cands['tril_upper_B@strict'] = np.eye(L) + Bd @ np.triu(KKT, 1)
    for nm, M in cands.items():
        A = np.linalg.inv(M)
        for tag, W in [('A', A), ('A@B', A @ Bd), ('B@A', Bd @ A)]:
            Qc = np.eye(D) - K.T @ W @ K
            err = np.abs(Qc - Q).max()
            print(f'  {nm:22s} W={tag:4s}  max_err={err:.3e}')
check()
print()
# also print the exact Q for reference (first row/col) to see structure
K = rng.normal(0, 1, (L, D)); beta = rng.uniform(0.1, 1.0, L)
Q = np.eye(D)
for t in range(L):
    Q = (np.eye(D) - beta[t]*np.outer(K[t], K[t])) @ Q
print('Q (exact) diagonal-adjacent sample:', np.round(Q[:3,:3], 4))
