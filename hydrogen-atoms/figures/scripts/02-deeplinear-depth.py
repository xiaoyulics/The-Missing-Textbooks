"""Chapter 2 (deeplinear): depth lengthens the plateau.

One mode of strength s = 1 (Sigma_yx = e1 e1^T in R^{4x4}), depth L in {2, 3, 4}, hidden width 32.
Theory: the half-learning time t_1/2 of the aligned, balanced solution with per-layer scale alpha,
a0 = alpha^L, by quadrature of dt = da / (L a^{2-2/L} (s - a)); for L = 2 in closed form.
Simulations (full-batch gradient descent, step eta = 0.02, time = eta * steps):
  'aligned': balanced aligned initialization with per-layer singular value alpha (random
             orthonormal hidden bases);
  'random':  Gaussian initialization with entries N(0, alpha^2 / fan_in), mean of 3 seeds.
t_1/2 is the first time at which u1^T W v1 reaches s/2.
Writes figures/data/02-deeplinear-depth-theory.dat and 02-deeplinear-depth-sim.dat.
Runs in about a minute.
"""
import os
import numpy as np
from scipy.integrate import quad

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
d, m, s, eta = 4, 32, 1.0, 0.02
Sig = np.zeros((d, d))
Sig[0, 0] = s


def t_half(L, a0):
    if L == 2:
        return np.log(s / a0 - 1) / (2 * s)
    # substitute a = e^u: dt = a^{2/L - 1} du / (L (s - a)), a smooth integrand
    f = lambda u: np.exp(u * (2 / L - 1)) / (L * (s - np.exp(u)))
    return quad(f, np.log(a0), np.log(s / 2), limit=400)[0]


def gd_time(Ws):
    L = len(Ws)
    t = 0.0
    while t < 5000:
        prefix = [np.eye(d)]  # prefix[l] = W_l ... W_1 (prefix[0] = I)
        for M in Ws:
            prefix.append(M @ prefix[-1])
        W = prefix[-1]
        if W[0, 0] >= s / 2:
            return t
        G = W - Sig
        suffix = np.eye(d)  # W_L ... W_{l+1}
        new = [None] * L
        for l in range(L - 1, -1, -1):
            new[l] = Ws[l] - eta * suffix.T @ G @ prefix[l].T
            suffix = suffix @ Ws[l]
        Ws = new
        t += eta
    return np.nan


def aligned(L, alpha, rng):
    dims = [d] + [m] * (L - 1) + [d]
    R = [np.eye(d)[:, :1]] + [np.linalg.qr(rng.standard_normal((m, 1)))[0] for _ in range(L - 1)] + [np.eye(d)[:, :1]]
    return [alpha * R[l + 1] @ R[l].T for l in range(L)]


def random_init(L, alpha, rng):
    dims = [d] + [m] * (L - 1) + [d]
    return [rng.standard_normal((dims[l + 1], dims[l])) * alpha / np.sqrt(dims[l]) for l in range(L)]


grids = {2: np.geomspace(1e-3, 0.5, 8), 3: np.geomspace(0.012, 0.5, 7), 4: np.geomspace(0.04, 0.5, 6)}
th_rows = []
for a in np.geomspace(1e-3, 0.6, 120):
    row = [a]
    for L in (2, 3, 4):
        th = t_half(L, a ** L) if a ** L < s / 2 else np.nan
        row.append(th if th < 2000 else np.nan)
    th_rows.append(row)
np.savetxt(os.path.join(DATA, "02-deeplinear-depth-theory.dat"), np.array(th_rows), fmt="%.6g",
           header="alpha L2 L3 L4", comments="")

sim_rows = []
for L in (2, 3, 4):
    for a in grids[L]:
        rng = np.random.default_rng(0)
        ta = gd_time(aligned(L, a, rng))
        tr = np.mean([gd_time(random_init(L, a, np.random.default_rng(k))) for k in range(3)])
        th = t_half(L, a ** L)
        sim_rows.append([L, a, ta, tr, th])
        print(L, round(a, 4), round(ta, 3), round(tr, 3), round(th, 3))
np.savetxt(os.path.join(DATA, "02-deeplinear-depth-sim.dat"), np.array(sim_rows), fmt="%.6g",
           header="L alpha aligned random theory", comments="")
