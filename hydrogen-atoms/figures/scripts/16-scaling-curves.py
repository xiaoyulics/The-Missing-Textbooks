"""Figure 1 of Chapter 16 (Scaling Laws): learning curves of linear regression on
Gaussian features with power-law spectra.

Model: independent features x_j ~ N(0, lam_j), lam_j = j^{-a}, target coefficients with
lam_j w_j^2 = j^{-(1+b)}, labels y = <w, x> + noise of variance s2; a = 1.5, b = 1.

(a) Least squares on the first N features, P = 1024 features in all, s2 = 0.2. Theory:
    the exact formula E[excess] = T(N) + (s2 + T(N)) N / (D - N - 1), with
    T(N) = sum_{N < j <= P} j^{-(1+b)}; "best" is the minimum over N; "asym" is the
    asymptote (1+b) c^{1/(1+b)} (s2/(bD))^{b/(1+b)} with c = 1/b (infinite P).
(b) Ridge on all P = 4000 features: minimum-norm interpolation without noise, and ridge
    with the ridge parameter that minimizes the deterministic equivalent, s2 = 0.2.
    Theory: the deterministic equivalent (effective regularization kappa).

Writes figures/data/16-scaling-ols-theory.dat, 16-scaling-ols-sim.dat and
16-scaling-ridge-{theory,sim}.dat. Deterministic (seed 1617); about 25 s on one core.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
A, B = 1.5, 1.0
rng = np.random.default_rng(1617)

# ---------------- panel (a): least squares on the first N features ----------------
P, S2 = 1024, 0.2
j = np.arange(1, P + 1)
lam = j ** -A
w = np.sqrt(j ** -(1 + B) / lam)
tail = np.concatenate([np.cumsum((lam * w * w)[::-1])[::-1], [0.0]])  # tail[N] = T(N)
NS = [4, 16, 64, 256]

def excess(N, D):
    if D - N - 1 <= 1:
        return np.nan
    T = tail[N]
    return T + (S2 + T) * N / (D - N - 1)

Dth = np.unique(np.round(np.logspace(np.log10(6), np.log10(2048), 140)).astype(int))
with open(os.path.join(DATA, "16-scaling-ols-theory.dat"), "w") as f:
    f.write("D " + " ".join(f"N{n}" for n in NS) + " best Nbest asym\n")
    for D in Dth:
        vals = [excess(n, D) for n in NS]
        allN = [excess(n, D) for n in range(0, min(D - 2, P + 1))]
        nb = int(np.nanargmin(allN))
        asym = (1 + B) * (1 / B) ** (1 / (1 + B)) * (S2 / (B * D)) ** (B / (1 + B))
        f.write(f"{D} " + " ".join("nan" if np.isnan(v) else f"{v:.6g}" for v in vals)
                + f" {allN[nb]:.6g} {nb} {asym:.6g}\n")

Dsim = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
TRIALS = 150
acc = {(n, D): [] for n in NS for D in Dsim}
for t in range(TRIALS):
    X = rng.standard_normal((Dsim[-1], P)) * np.sqrt(lam)
    y = X @ w + np.sqrt(S2) * rng.standard_normal(Dsim[-1])
    for D in Dsim:
        for n in NS:
            if D < 2 * n + 4:
                continue
            wh = np.linalg.lstsq(X[:D, :n], y[:D], rcond=None)[0]
            acc[(n, D)].append(tail[n] + np.sum(lam[:n] * (wh - w[:n]) ** 2))
with open(os.path.join(DATA, "16-scaling-ols-sim.dat"), "w") as f:
    f.write("D " + " ".join(f"N{n}" for n in NS) + "\n")
    for D in Dsim:
        row = []
        for n in NS:
            v = acc[(n, D)]
            row.append(f"{np.mean(v):.6g}" if v else "nan")
        f.write(f"{D} " + " ".join(row) + "\n")
worst = max(abs(np.mean(acc[k]) - excess(*k)) / (np.std(acc[k]) / np.sqrt(len(acc[k])))
            for k in acc if acc[k])
print("panel (a): largest deviation in standard errors:", round(worst, 2))

# ---------------- panel (b): ridge on all features, deterministic equivalent -------
P2, S2b = 4000, 0.2
j2 = np.arange(1, P2 + 1)
lam2 = j2 ** -A
w2 = np.sqrt(j2 ** -(1 + B) / lam2)
sig2 = lam2 * w2 * w2

def det_equiv(D, rl, s2):
    if rl == 0:
        k = brentq(lambda k: 1 - np.sum(lam2 / (lam2 + k)) / D, 1e-16, 10)
    else:
        k = brentq(lambda k: k - rl - k * np.sum(lam2 / (lam2 + k)) / D, rl, rl + 10)
    g = np.sum(lam2 ** 2 / (lam2 + k) ** 2) / D
    return (k * k * np.sum(sig2 / (lam2 + k) ** 2) + s2 * g) / (1 - g)

grid = np.logspace(-7, 0, 300)
Db = [8, 16, 32, 64, 128, 256, 512, 1024]
Dbt = np.unique(np.round(np.logspace(np.log10(8), np.log10(1024), 100)).astype(int))
lamopt = {}
with open(os.path.join(DATA, "16-scaling-ridge-theory.dat"), "w") as f:
    f.write("D noiseless optimal lamopt\n")
    for D in sorted(set(Dbt) | set(Db)):
        risks = [det_equiv(D, r, S2b) for r in grid]
        i = int(np.argmin(risks)); lamopt[D] = grid[i]
        f.write(f"{D} {det_equiv(D, 0, 0):.6g} {risks[i]:.6g} {grid[i]:.4g}\n")
TR2 = 60
acc0 = {D: [] for D in Db}; acc1 = {D: [] for D in Db}
for t in range(TR2):
    X = rng.standard_normal((Db[-1], P2)) * np.sqrt(lam2)
    f0 = X @ w2
    eps = np.sqrt(S2b) * rng.standard_normal(Db[-1])
    K = X @ X.T
    for D in Db:
        KD = K[:D, :D]
        a0 = np.linalg.solve(KD, f0[:D])
        acc0[D].append(np.sum(lam2 * (X[:D].T @ a0 - w2) ** 2))
        a1 = np.linalg.solve(KD + D * lamopt[D] * np.eye(D), f0[:D] + eps[:D])
        acc1[D].append(np.sum(lam2 * (X[:D].T @ a1 - w2) ** 2))
with open(os.path.join(DATA, "16-scaling-ridge-sim.dat"), "w") as f:
    f.write("D noiseless optimal\n")
    for D in Db:
        f.write(f"{D} {np.mean(acc0[D]):.6g} {np.mean(acc1[D]):.6g}\n")
for D in Db:
    print(D, det_equiv(D, 0, 0), np.mean(acc0[D]), det_equiv(D, lamopt[D], S2b), np.mean(acc1[D]))
