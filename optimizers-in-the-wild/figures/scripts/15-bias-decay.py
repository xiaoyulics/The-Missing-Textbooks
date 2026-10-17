"""Figure 15.3 of Optimizers in the Wild (Chapter 15, key `bias`): where AdamW ends.

Sign descent with decoupled weight decay lambda (AdamW without memory: beta1 = beta2 = 0,
epsilon = 0) is the Frank-Wolfe method on the l-infinity ball of radius B = 1/lambda
(Chapter 13). On the separable dataset of Figure 15.2 (n = 16 examples in R^5, the same seed)
the script computes, for radii B between 1 and 300:
  * the exact end point: a minimizer theta_B of the exponential loss over the ball
    ||theta||_inf <= B (L-BFGS-B on the log of the loss, from two starts; the loss is convex),
    and its normalized l-infinity margin min_i <theta_B, z_i> / ||theta_B||_inf;
  * the lower bound gamma_inf - log(n)/B of the regularization-path proposition;
  * simulation points: AdamW without memory, theta <- (1 - eta_t lambda) theta - eta_t sign(g_t),
    with eta_t lambda = 2/(t+2) (the Frank-Wolfe schedule), run for 20000 steps from zero.
Output: figures/data/15-bias-decay.dat (curves) and figures/data/15-bias-decay-fw.dat (points).
numpy and scipy only; deterministic; about 20 s on a laptop CPU.
"""
import os
import numpy as np
from scipy.optimize import linprog, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

rng = np.random.default_rng(11)          # the dataset of 15-bias-margins.py
d, n = 5, 16
wt = rng.standard_normal(d)
X = rng.standard_normal((400, d))
X /= np.linalg.norm(X, axis=1, keepdims=True)
y = np.sign(X @ wt)
keep = np.abs(X @ wt) / np.linalg.norm(wt) > 0.15
X, y = X[keep][:n], y[keep][:n]
Z = y[:, None] * X

r = linprog(np.r_[np.zeros(d), -1.0], A_ub=np.hstack([-Z, np.ones((n, 1))]), b_ub=np.zeros(n),
            bounds=[(-1, 1)] * d + [(None, None)], method="highs")
gam, ustar = -r.fun, r.x[:d]


def logloss(th):
    m = Z @ th
    mm = m.min()
    w = np.exp(-(m - mm))
    s = w.sum()
    return -mm + np.log(s / n), -(Z.T @ w) / s


def end_point(B):
    best = None
    for x0 in (B * ustar, np.zeros(d)):
        res = minimize(logloss, x0, jac=True, method="L-BFGS-B", bounds=[(-B, B)] * d,
                       options={"ftol": 1e-15, "gtol": 1e-13, "maxiter": 20000})
        if best is None or res.fun < best.fun:
            best = res
    th = best.x
    return np.min(Z @ th) / np.max(np.abs(th))


def adamw_without_memory(B, T=20000):
    lam = 1.0 / B
    th = np.zeros(d)
    for t in range(T):
        m = Z @ th
        w = np.exp(-(m - m.min()))
        g = -(Z.T @ w)                      # the gradient up to a positive factor
        eta = (2.0 / (t + 2)) / lam
        th = (1 - eta * lam) * th - eta * np.sign(g)
    return np.min(Z @ th) / np.max(np.abs(th))


Bs = np.logspace(0, np.log10(300), 60)
with open(os.path.join(DATA, "15-bias-decay.dat"), "w") as f:
    f.write("B lam exact bound gamma\n")
    for B in Bs:
        f.write(f"{B:.5f} {1/B:.6f} {end_point(B):.6f} {gam - np.log(n)/B:.6f} {gam:.6f}\n")
pts = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300]
with open(os.path.join(DATA, "15-bias-decay-fw.dat"), "w") as f:
    f.write("B lam fw exact\n")
    for B in pts:
        a, b = adamw_without_memory(B), end_point(B)
        f.write(f"{B} {1/B:.6f} {a:.6f} {b:.6f}\n")
        print(f"B = {B:5.0f}: end point margin {b:.5f}, AdamW without memory {a:.5f}, "
              f"bound {gam - np.log(n)/B:+.5f}, gamma_inf {gam:.5f}")
