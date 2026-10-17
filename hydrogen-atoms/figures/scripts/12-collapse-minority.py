"""Chapter 12, figure fig:collapse:minority.

Minority collapse in the norm-constrained unconstrained features model:
K = 4 classes, the first K_A = 2 with R times as many examples as the last
K_B = 2, feature dimension m = 4.  Features are taken collapsed to their class
means (which is optimal, by convexity of the loss in each feature), so the
problem is
  minimize  sum_k (n_k/N) CE(W h_k, k)
  subject to (1/K) sum_k ||w_k||^2 <= E,  (1/K) sum_k ||h_k||^2 <= E.
Solved by SLSQP from 4 random starts (seeds 0 to 3), keeping the best.  Writes
figures/data/12-collapse-minority.dat with the cosine between the two minority
classifiers w_3, w_4 (cosE1, cosE3, cosE10) for E = 1, 3, 10.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

KA, KB, m = 2, 2, 4
K = KA + KB


def solve(R, E, starts=4):
    nk = np.array([R] * KA + [1.0] * KB)
    pk = nk / nk.sum()

    def f(x):
        W, H = x[:K * m].reshape(K, m), x[K * m:].reshape(K, m)
        Z = H @ W.T                                  # row k: logits of class-k mean
        lse = logsumexp(Z, axis=1)
        L = np.sum(pk * (lse - np.diag(Z)))
        G = pk[:, None] * (np.exp(Z - lse[:, None]) - np.eye(K))
        return L, np.concatenate([(G.T @ H).ravel(), (G @ W).ravel()])

    z = np.zeros(K * m)
    cons = [{"type": "ineq", "fun": lambda x: E - np.sum(x[:K * m] ** 2) / K,
             "jac": lambda x: np.concatenate([-2 * x[:K * m] / K, z])},
            {"type": "ineq", "fun": lambda x: E - np.sum(x[K * m:] ** 2) / K,
             "jac": lambda x: np.concatenate([z, -2 * x[K * m:] / K])}]
    best = None
    for s in range(starts):
        x0 = 0.5 * np.random.default_rng(s).standard_normal(2 * K * m)
        r = minimize(f, x0, jac=True, constraints=cons, method="SLSQP",
                     options={"maxiter": 3000, "ftol": 1e-15})
        if best is None or r.fun < best.fun:
            best = r
    W = best.x[:K * m].reshape(K, m)
    return W[2] @ W[3] / np.linalg.norm(W[2]) / np.linalg.norm(W[3])


Rs = np.round(np.logspace(0, 3, 13), 3)
out = os.path.join(os.path.dirname(__file__), "..", "data", "12-collapse-minority.dat")
with open(out, "w") as fh:
    fh.write("R cosE1 cosE3 cosE10\n")
    for R in Rs:
        c = [solve(R, E) for E in (1.0, 3.0, 10.0)]
        print(R, np.round(c, 4))
        fh.write("%.3f %.6f %.6f %.6f\n" % ((R,) + tuple(c)))
