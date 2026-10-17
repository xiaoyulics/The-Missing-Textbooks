"""Chapter 12, figure fig:collapse:phase.

Unconstrained features model with weight decay lambda_W = lambda_H = lambda,
K = 4 classes, n = 5 examples per class, feature dimension m = 6, no bias for
the cross-entropy, an unregularized bias for the square loss.  For each
kappa = K sqrt(n) lambda the regularized objective is minimized by L-BFGS from
three random initializations (seeds 0, 1, 2; entries N(0, 0.5^2)); the best
run is kept.  Writes
  figures/data/12-collapse-phase-theory.dat   kappa, rho and p predicted
  figures/data/12-collapse-phase-sim.dat      kappa, rho and p found
where rho = ||W||_F ||H||_F / (K sqrt n) and p is the mean predicted
correct-class probability (cross-entropy) or correct-class output (square loss).
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

K, n, m = 4, 5, 6
N = K * n
y = np.repeat(np.arange(K), n)
Y = np.eye(K)[y]                                  # N x K one-hot targets


def objective(x, lam, loss):
    W = x[:K * m].reshape(K, m)
    H = x[K * m:K * m + N * m].reshape(N, m)
    b = x[K * m + N * m:]
    Z = H @ W.T + (b if loss == "mse" else 0.0)   # N x K logits
    if loss == "ce":
        lse = logsumexp(Z, axis=1)
        L = np.mean(lse - Z[np.arange(N), y])
        G = (np.exp(Z - lse[:, None]) - Y) / N
    else:
        L = 0.5 * np.sum((Z - Y) ** 2) / N
        G = (Z - Y) / N
    F = L + 0.5 * lam * (np.sum(W ** 2) + np.sum(H ** 2))
    gW = G.T @ H + lam * W
    gH = G @ W + lam * H
    gb = G.sum(0) if loss == "mse" else np.zeros(K)
    return F, np.concatenate([gW.ravel(), gH.ravel(), gb])


def simulate(kappa, loss):
    lam = kappa / (K * np.sqrt(n))
    best = None
    for seed in range(3):
        x0 = 0.5 * np.random.default_rng(seed).standard_normal(K * m + N * m + K)
        r = minimize(objective, x0, args=(lam, loss), jac=True, method="L-BFGS-B",
                     options={"maxiter": 20000, "gtol": 1e-11, "ftol": 1e-15})
        if best is None or r.fun < best.fun:
            best = r
    x = best.x
    W = x[:K * m].reshape(K, m)
    H = x[K * m:K * m + N * m].reshape(N, m)
    b = x[K * m + N * m:]
    rho = np.linalg.norm(W) * np.linalg.norm(H) / (K * np.sqrt(n))
    Z = H @ W.T + (b if loss == "mse" else 0.0)
    if loss == "ce":
        P = np.exp(Z - logsumexp(Z, axis=1)[:, None])
        p = np.mean(P[np.arange(N), y])
    else:
        p = np.mean(Z[np.arange(N), y])
    return rho, p


def theory(kappa):
    if kappa < 1:
        rho_ce = (K - 1) / K * np.log(1 + K * (1 - kappa) / kappa)
        rho_mse = (K - 1) / K * (1 - kappa)
        p = 1 - (K - 1) * kappa / K
    else:
        rho_ce = rho_mse = 0.0
        p = 1.0 / K
    return rho_ce, rho_mse, p


root = os.path.join(os.path.dirname(__file__), "..", "data")
with open(os.path.join(root, "12-collapse-phase-theory.dat"), "w") as f:
    f.write("kappa rhoce rhomse p\n")
    for kappa in np.concatenate([np.linspace(0.04, 0.999, 120), [1.0, 1.4]]):
        f.write("%.5f %.6f %.6f %.6f\n" % ((kappa,) + theory(kappa)))
with open(os.path.join(root, "12-collapse-phase-sim.dat"), "w") as f:
    f.write("kappa rhoce pce rhomse pmse\n")
    for kappa in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.05, 1.2, 1.35]:
        rc, pc = simulate(kappa, "ce")
        rm, pm = simulate(kappa, "mse")
        tc, tm, tp = theory(kappa)
        print("kappa %.2f  CE rho %.5f (%.5f) p %.5f   MSE rho %.5f (%.5f) p %.5f   theory p %.5f"
              % (kappa, rc, tc, pc, rm, tm, pm, tp))
        f.write("%.3f %.6f %.6f %.6f %.6f\n" % (kappa, rc, pc, rm, pm))
