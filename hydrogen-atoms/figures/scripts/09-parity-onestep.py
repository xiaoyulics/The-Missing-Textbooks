"""One gradient step finds the support of a sparse parity (Chapter 9, figure fig:parity:onestep).

Setting: x uniform on {-1,1}^d, y = x_1 x_2 x_3 (k = 3; S = {1,2,3} without loss of generality),
monomial neuron phi(z) = z^k at the symmetric start w_0 = 1 (all ones). The empirical gradient of
the correlation (1/n) sum_s y_s phi(<w, x_s>) at w_0 is
    G_i = (1/n) sum_s y_s k (sum_j x_sj)^(k-1) x_si,
with mean k! on S and 0 off S. Success: the k largest coordinates of G are exactly S.
Simulation: 300 independent data sets for each (d, n), d in {16, 32, 64}, n = a d^2 ln d.
Theory: the central-limit approximation, a Gaussian vector with the exact mean and the exact
covariance of G (computed from the binomial law of sum_j x_j), success probability estimated with
20000 Gaussian draws.
Output: figures/data/09-parity-onestep.dat, columns a, then sim16 th16 sim32 th32 sim64 th64.
Runtime: about 40 seconds.
"""
import os
import math
import numpy as np
from scipy.stats import binom

K, SEED = 3, 20261017
DIMS = (16, 32, 64)
A = np.round(np.arange(0.25, 4.01, 0.25), 2)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "09-parity-onestep.dat")


def moment(d, p, pair=False):
    """E[(sum x)^p] or, with pair=True, E[(sum x)^p x_1 x_2]."""
    if not pair:
        j = np.arange(d + 1)
        return float(np.sum(binom.pmf(j, d, 0.5) * (2.0 * j - d) ** p))
    j = np.arange(d - 1)
    s, pm = 2.0 * j - (d - 2), binom.pmf(j, d - 2, 0.5)
    return float(sum(0.25 * a * b * np.sum(pm * (s + a + b) ** p) for a in (-1, 1) for b in (-1, 1)))


def theory(d, n, rng, draws=20000):
    kf = math.factorial(K)
    m2 = K * K * moment(d, 2 * K - 2)
    c2 = K * K * moment(d, 2 * K - 2, pair=True)
    mu = np.zeros(d)
    mu[:K] = kf
    cov = np.full((d, d), c2)
    np.fill_diagonal(cov, m2)
    cov -= np.outer(mu, mu)
    L = np.linalg.cholesky(cov / n)
    G = mu + rng.standard_normal((draws, d)) @ L.T
    return float(np.mean(G[:, :K].min(1) > G[:, K:].max(1)))


def simulate(d, n, rng, trials=300):
    ok = 0
    for _ in range(trials):
        X = (rng.integers(0, 2, (n, d), dtype=np.int8) * 2 - 1).astype(np.float32)
        y = X[:, :K].prod(1)
        g = X.T @ (y * K * X.sum(1) ** (K - 1)) / n
        ok += g[:K].min() > g[K:].max()
    return ok / trials


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for a in A:
        row = [a]
        for d in DIMS:
            n = int(round(a * d * d * math.log(d)))
            row += [simulate(d, n, rng), theory(d, n, rng)]
        rows.append(row)
    with open(OUT, "w") as f:
        f.write("a " + " ".join(f"sim{d} th{d}" for d in DIMS) + "\n")
        for r in rows:
            f.write(" ".join(f"{v:.4f}" for v in r) + "\n")


if __name__ == "__main__":
    main()
