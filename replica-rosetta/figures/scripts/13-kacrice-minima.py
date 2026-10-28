"""Chapter 13 (kacrice): where the local minima of the pure spherical 3-spin are at finite N.

The exact formula (Theorem thm:kacrice:exact) gives, for every function g,
    E sum_{local minima s} g(H(s)/N) = 2 sqrt(2/p) (p-1)^{N/2} E[ exp(-N a l^2) g(l E_inf / 2) ],
with a = (p-2)/(4p) and l the smallest eigenvalue of a GOE(N) matrix (book normalization, semicircle on
[-2, 2]).  We sample l with the tridiagonal model of Dumitriu and Edelman (diagonal N(0,1), off-diagonal
chi_{N-1}, ..., chi_1 divided by sqrt 2; eigenvalues multiplied by sqrt(2/N)) and a vectorized
Sturm-sequence bisection, and report the mean energy of a local minimum in the expected count,
    E sum_{minima} H/N / E #minima,
its spread, a 95% Monte Carlo interval (1.96 spread / sqrt(effective sample size)), the expected number
of minima, and the share of that number below -E_inf.

Writes figures/data/13-kacrice-minima.dat.  N = 16, ..., 256, 40000 samples each, seed 1313.
About a minute on a laptop.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
P = 3
EINF = 2 * math.sqrt((P - 1) / P)


def goe_lmin(N, M, rng, steps=46):
    """Smallest eigenvalue of M independent GOE(N) matrices: tridiagonal model + Sturm bisection."""
    a = rng.standard_normal((M, N)) * math.sqrt(2.0 / N)
    b2 = rng.chisquare(np.arange(N - 1, 0, -1), size=(M, N - 1)) / 2 * (2.0 / N)   # squared off-diagonal
    lo = np.full(M, -4.0)
    hi = np.full(M, -0.5)
    for _ in range(steps):
        x = 0.5 * (lo + hi)
        q = a[:, 0] - x
        neg = (q < 0).astype(int)
        for i in range(1, N):
            q = np.where(q == 0, 1e-300, q)
            q = a[:, i] - x - b2[:, i - 1] / q
            neg += q < 0
        below = neg >= 1                      # at least one eigenvalue below x
        hi = np.where(below, x, hi)
        lo = np.where(below, lo, x)
    return 0.5 * (lo + hi)


def main():
    t0 = time.time()
    rng = rk.rng(1313)
    # check the sampler against dense GOE matrices
    G = rng.standard_normal((20000, 30, 30)) / math.sqrt(30)
    dense = np.linalg.eigvalsh((G + G.transpose(0, 2, 1)) / math.sqrt(2))[:, 0]
    tri = goe_lmin(30, 20000, rng)
    print(f"check N=30: smallest eigenvalue mean/sd dense {dense.mean():.4f}/{dense.std():.4f},"
          f" tridiagonal {tri.mean():.4f}/{tri.std():.4f}")
    a = (P - 2) / (4 * P)
    rows = []
    for N in (16, 32, 64, 128, 256):
        lm = goe_lmin(N, 40000, rng)
        if np.any(lm < -3.99) or np.any(lm > -0.51):
            raise RuntimeError("bisection bracket too narrow")
        w = np.exp(-N * a * (lm ** 2 - 4))
        u = lm * EINF / 2
        W = w.sum()
        ubar = float(np.sum(w * u) / W)
        sd = math.sqrt(float(np.sum(w * (u - ubar) ** 2) / W))
        ess = W * W / float(np.sum(w * w))
        err = 1.96 * sd / math.sqrt(ess)
        logE = math.log(2 * math.sqrt(2 / P)) + N / 2 * math.log(P - 1) - 4 * N * a + math.log(W / len(w))
        below = float(np.sum(w * (u < -EINF)) / W)
        rows.append([N, ubar, sd, err, math.exp(logE), below])
        print(f"N={N}: mean energy of a local minimum {ubar:.4f} (spread {sd:.4f}, MC 95% +-{err:.4f}, ESS {ess:.0f});"
              f" E #minima {math.exp(logE):.3g}; share below -E_inf {below:.4f};"
              f" (mean + E_inf) sqrt(N) = {(ubar + EINF) * math.sqrt(N):.3f}  ({time.time()-t0:.1f}s)")
    rk.write_dat(os.path.join(OUT, "13-kacrice-minima.dat"),
                 ["N", "umean", "spread", "mcerr", "Eminima", "below"], rows)


if __name__ == "__main__":
    main()
