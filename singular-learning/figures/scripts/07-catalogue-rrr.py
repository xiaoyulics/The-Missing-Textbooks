"""07-catalogue-rrr.py -- the learning coefficient of reduced-rank regression, measured (Chapter 7).

Run from the volume root:  python figures/scripts/07-catalogue-rrr.py   (about 70 seconds)

Model y = B A x + noise with x ~ N(0, I_3), noise N(0, I_3), B in R^{3 x H}, A in R^{H x 3}, H = 1, 2, and
the uniform prior on [-1, 1]^d, d = 6H. Truths: rank 0 (C0 = 0) and rank 1 (C0 = (1/2) e1 e1^T).
For each case and n = 10^3, 10^4, 10^5 we draw 12 data sets and estimate lambda by Watanabe's
two-temperature estimator (2013, Corollary 3),
    lambda_hat = (E^{b1}[n L_n] - E^{b2}[n L_n]) / (1/b1 - 1/b2),   b1 = 1/log n,  b2 = 1.5/log n,
as Watanabe computed it in his Section 6.2: draws at b1 only, by random-walk Metropolis
(sk.metropolis: 16 chains per data set, started from the prior, 8000 adaptation steps discarded, 16000
kept, every 40th used), and the mean at b2 by reweighting the same draws with exp(-(b2 - b1) n L_n)
(his eq. (20)). The data enter only through Sxx = sum x x^T, Sxy = sum x y^T and Syy = sum |y|^2.
Predictions: lambda from the Aoyagi--Watanabe formula (3/2, 5/2, 5/2, 7/2 with m = 2, 1, 1, 2), and d/2.
For m = 2 the digests' first-order correction is lambda - (m - 1)/log n; with the exact constant of the
expansion (07-catalogue-formulas.py) the population value is 1.271, 1.342, 1.381 for H = 1, r = 0.
Prints a split-half check (the estimate from the first and the second half of the kept draws) and the
effective sample size of the reweighting. Writes figures/data/07-catalogue-rrr.dat.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

M = N = 3
NS = (1000, 10_000, 100_000)
R, CHAINS = 12, 16
ADAPT, KEEP, THIN = 8000, 16000, 40
CASES = [(1, 0, 1.5, 2), (1, 1, 2.5, 1), (2, 0, 2.5, 1), (2, 1, 3.5, 2)]   # (H, r, lambda, m)


def energy_fn(H, Sxx, Sxy, Syy):
    """n L_n up to the constant (nM/2) log 2pi, for w = (vec B, vec A), per chain (C, d):
    (1/2) (Syy - 2 tr(C Sxy) + tr(C Sxx C^T)) with C = BA."""
    SyxT = np.transpose(Sxy, (0, 2, 1))                              # (C, M, N)

    def energy(w):
        C = w.shape[0]
        Cm = np.matmul(w[:, :M * H].reshape(C, M, H), w[:, M * H:].reshape(C, H, N))
        lin = (Cm * SyxT).sum(axis=(1, 2))
        quad = (np.matmul(Cm, Sxx) * Cm).sum(axis=(1, 2))
        return 0.5 * (Syy - 2.0 * lin + quad)
    return energy


def two_temperatures(e, n):
    """lambda_hat from energies e (draws at b1 = 1/log n), the b2-mean by reweighting; and the
    effective sample size fraction of the weights."""
    b1, b2 = 1.0 / math.log(n), 1.5 / math.log(n)
    w = np.exp(-(b2 - b1) * (e - e.min()))
    w /= w.sum()
    return (e.mean() - float(w @ e)) / (1.0 / b1 - 1.0 / b2), 1.0 / float(np.sum(w * w)) / len(e)


def run_case(H, r, seed):
    d = H * (M + N)
    C0 = np.zeros((M, N))
    if r == 1:
        C0[0, 0] = 0.5
    shape = (len(NS), R, CHAINS)                                      # chain layout
    C = int(np.prod(shape))
    idx = np.arange(C).reshape(shape)
    Sxx = np.empty((C, N, N)); Sxy = np.empty((C, N, M)); Syy = np.empty(C); bet = np.empty(C)
    for ni, n in enumerate(NS):
        for k in range(R):
            g = sk.rng(seed + 1000 * ni + k)
            x = g.standard_normal((n, N))
            y = x @ C0.T + g.standard_normal((n, M))
            sl = idx[ni, k]
            Sxx[sl], Sxy[sl], Syy[sl], bet[sl] = x.T @ x, x.T @ y, float(np.sum(y * y)), 1.0 / math.log(n)
    energy = energy_fn(H, Sxx, Sxy, Syy)

    def logp(w):
        inside = np.all(np.abs(w) <= 1.0, axis=1)
        return np.where(inside, -bet * energy(w), -np.inf)

    x0 = sk.rng(seed + 77).uniform(-1, 1, (C, d))
    draws, acc = sk.metropolis(logp, x0, 0.05, KEEP, seed=seed + 78, adapt=ADAPT, thin=THIN)
    E = np.array([energy(dr) for dr in draws])                      # (kept, C)
    half = len(E) // 2
    out = []
    for ni, n in enumerate(NS):
        lam, ess, split = [], [], []
        for k in range(R):
            e = E[:, idx[ni, k]]
            lk, fk = two_temperatures(e.ravel(), n)
            lam.append(lk); ess.append(fk)
            split.append(two_temperatures(e[:half].ravel(), n)[0] - two_temperatures(e[half:].ravel(), n)[0])
        mu, lo, hi = sk.mean_ci(lam)
        out.append((mu, hi - mu, float(np.std(lam, ddof=1)), float(np.mean(split)),
                    float(np.std(split, ddof=1) / math.sqrt(R)), float(np.mean(ess))))
    return out, acc


def main():
    t0 = time.time()
    results = {}
    for ci, (H, r, lam, m) in enumerate(CASES):
        t = time.time()
        res, acc = run_case(H, r, 75_000 + 10_000 * ci)
        results[(H, r)] = res
        for (n, (mu, half, sd, split, sse, ess)) in zip(NS, res):
            pred = lam - (m - 1) / math.log(n)
            print(f"H = {H}, rank {r}: n = {n:6d}  lambda_hat = {mu:.3f} +- {half:.3f} (sd over data sets {sd:.3f}); "
                  f"lambda = {lam}, m = {m}, lambda - (m-1)/log n = {pred:.3f}, d/2 = {3 * H}; "
                  f"first half - second half {split:+.3f} +- {sse:.3f}; reweighting ESS {ess:.2f}")
        print(f"   acceptance {acc:.2f}, {time.time() - t:.0f} s")
    header = ["n"]
    for (H, r, _, _) in CASES:
        header += [f"H{H}r{r}", f"H{H}r{r}err"]
    rows = []
    for ni, n in enumerate(NS):
        row = [n]
        for (H, r, _, _) in CASES:
            row += [results[(H, r)][ni][0], results[(H, r)][ni][1]]
        rows.append(row)
    sk.write_dat("figures/data/07-catalogue-rrr.dat", header, rows)
    print(f"total time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
