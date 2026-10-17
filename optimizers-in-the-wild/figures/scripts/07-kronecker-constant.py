#!/usr/bin/env python3
"""Figure 7.2 of "Optimizers in the Wild" (Chapter 7): how large the constant of the Kronecker
bound is.  For gradients G_1..G_T in R^{m x n} (m = n = 16) of rank r = 4, the tight constant

    c*(T) = lmax( K^{+1/2} Gamma_T K^{+1/2} ),   Gamma_T = sum_t vec(G_t) vec(G_t)^T,
    K = R_T^{1/2} kron L_T^{1/2},  L_T = sum G G^T,  R_T = sum G^T G   (eps = 0, pseudo-inverses),

is the smallest c with Gamma_T <= c K.  The Kronecker bound says c* <= r.  Four families:
  coherent  G_t = c_t G for one fixed rank-r G (theory: c* = r for every T),
  positive  G_t = U diag(s_t) V^T, fixed U, V, entries of s_t iid uniform on (0, 1)
            (theory: c* -> 1 + (r - 1) * 3/4 as T -> infinity; c* = r at T = 1),
  signed    the same with entries of s_t iid standard normal (theory: c* -> 1),
  generic   G_t = X_t Y_t^T with independent Gaussian X_t, Y_t (theory: c* -> 1/sqrt(mn)).
The theory values are the exact limits of Proposition 7.3; the points are the mean over 5 seeds of
c* computed from the mn x mn matrices.  Output: figures/data/07-kronecker-constant.dat.
numpy only; deterministic seeds; about ten seconds on one core.
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
m = n = 16
r = 4
Ts = [2 ** k for k in range(13)]          # 1 .. 4096


def vec(X):
    return X.reshape(-1, order="F")


def pinv_root(M, tol=1e-10):
    w, Q = np.linalg.eigh((M + M.T) / 2)
    keep = w > tol * w.max()
    return (Q[:, keep] * w[keep] ** -0.25) @ Q[:, keep].T      # M^{+1/4}, so K^{+1/2} = R^{+1/4} kron L^{+1/4}


def cstar(Gamma, L, R):
    Kh = np.kron(pinv_root(R), pinv_root(L))
    return np.linalg.eigvalsh(Kh @ Gamma @ Kh).max()


def family_gradient(kind, rng, U, V, G0):
    if kind == "coherent":
        return rng.uniform(0.5, 2.0) * G0
    if kind == "positive":
        return U @ np.diag(rng.uniform(0, 1, r)) @ V.T
    if kind == "signed":
        return U @ np.diag(rng.standard_normal(r)) @ V.T
    return rng.standard_normal((m, r)) @ rng.standard_normal((r, n))


kinds = ["coherent", "positive", "signed", "generic"]
res = {k: np.zeros(len(Ts)) for k in kinds}
seeds = range(5)
for seed in seeds:
    rng = np.random.default_rng(100 + seed)
    U = np.linalg.qr(rng.standard_normal((m, r)))[0]
    V = np.linalg.qr(rng.standard_normal((n, r)))[0]
    G0 = U @ np.diag(rng.uniform(0.5, 2.0, r)) @ V.T
    for kind in kinds:
        Gamma = np.zeros((m * n, m * n)); L = np.zeros((m, m)); R = np.zeros((n, n))
        t = 0
        for i, T in enumerate(Ts):
            while t < T:
                G = family_gradient(kind, rng, U, V, G0)
                g = vec(G)
                Gamma += np.outer(g, g); L += G @ G.T; R += G.T @ G
                t += 1
            res[kind][i] += cstar(Gamma, L, R) / len(seeds)

with open(os.path.join(OUT, "07-kronecker-constant.dat"), "w") as f:
    f.write("T coherent positive signed generic th_coherent th_positive th_signed th_generic\n")
    for i, T in enumerate(Ts):
        f.write(f"{T} " + " ".join(f"{res[k][i]:.5f}" for k in kinds)
                + f" {r:.5f} {1 + (r - 1) * 0.75:.5f} 1.00000 {1 / np.sqrt(m * n):.5f}\n")
for k in kinds:
    print(k, " ".join(f"{v:.3f}" for v in res[k]))
