"""Chapter 6 (Message Passing): what AMP reaches, against PCA and the Bayes-optimal overlap.

Z2 synchronization A = (lam/N) x x^T + W, W ~ GOE(N), N = 2000.  For each lam in a grid above the
BBP threshold, R = 10 matrices (seeds 7000 + 100 j + r): the squared cosine between x and
  - the top eigenvector phi_1 (PCA), limit 1 - 1/lam^2 (Chapter 5, proved);
  - the Bayes-AMP estimate m^T, T = 30, spectral start with the virtual estimate m^{-1} = h^0/lam^2;
    state evolution gives cos^2 -> q_T, and q_T -> q* = gamma*/lam^2, the largest fixed point of
    gamma = lam^2 (1 - mmse(gamma)), which is the Bayes-optimal overlap (Lelarge--Miolane; DAM16).
Also checks numerically that the map gamma -> 1 - mmse(gamma) is concave on (0, 40]; its strict
concavity on [0, inf), which makes the positive fixed point unique for every lam > 1, is proved in
Chapter 4 and in Deshpande--Abbe--Montanari (2016, App. B.2).
Intervals: Student-t 95% over the R matrices.  Writes figures/data/06-amp-lambda-curve.dat and
figures/data/06-amp-lambda-sim.dat.  Run from the volume root (about 45 s).
"""
import math
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk

N, R, T = 2000, 10, 30
LAMS = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.25, 2.5]


def q_of(g):
    return 1.0 - rk.mmse_rademacher(g, n=201) if g > 0 else 0.0


def qstar(lam):
    if lam <= 1.0:
        return 0.0
    g = rk.root(lambda g: lam ** 2 * q_of(g) - g, 1e-9, 4 * lam ** 2 + 5)
    return g / lam ** 2


def main():
    t0 = time.time()
    # curves
    grid = np.concatenate([np.linspace(0.8, 1.0, 5), np.linspace(1.0025, 1.1, 40), np.linspace(1.1, 2.6, 61)[1:]])
    rk.write_dat(os.path.join("figures", "data", "06-amp-lambda-curve.dat"), ["lam", "qstar", "bbp"],
                 [(float(l), qstar(l), rk.bbp_overlap(l)) for l in grid])
    # concavity of q(gamma) (uniqueness of the positive fixed point)
    gg = np.linspace(1e-4, 40, 4001)
    qq = np.array([q_of(g) for g in gg])
    print("max second difference of q on (0,40]: %.3e (negative: concave)" % np.max(qq[2:] - 2 * qq[1:-1] + qq[:-2]))
    rows = []
    for j, lam in enumerate(LAMS):
        amp_c2, pca_c2 = [], []
        for r in range(R):
            gen = rk.rng(7000 + 100 * j + r)
            x = gen.choice([-1.0, 1.0], size=N)
            W = rk.goe(N, gen)
            matvec = lambda v: W @ v + (lam / N) * x * (x @ v)
            op = LinearOperator((N, N), matvec=matvec, dtype=float)
            val, vec = eigsh(op, k=1, which="LA", tol=1e-10)
            phi = vec[:, 0] * (1.0 if vec[:, 0] @ x >= 0 else -1.0)
            pca_c2.append((phi @ x) ** 2 / N)
            h0 = lam * math.sqrt(lam ** 2 - 1) * math.sqrt(N) * phi
            h, mp = h0.copy(), h0 / lam ** 2
            for _ in range(T):
                m = np.tanh(h)
                h, mp = lam * matvec(m) - lam ** 2 * (1 - m @ m / N) * mp, m
            m = np.tanh(h)
            amp_c2.append((m @ x) ** 2 / (N * (m @ m)))
            del W
        a, p = rk.mean_ci(amp_c2), rk.mean_ci(pca_c2)
        # state evolution at t = T from the spectral start
        g = lam ** 2 - 1
        for _ in range(T):
            g = lam ** 2 * q_of(g)
        rows.append((lam, a[0], a[1], p[0], p[1], q_of(g), qstar(lam), rk.bbp_overlap(lam)))
        print("lam=%.2f  AMP cos^2 %.4f +- %.4f (SE q_T %.4f, q* %.4f) | PCA %.4f +- %.4f (1-1/lam^2 %.4f)"
              % (lam, a[0], a[1], q_of(g), qstar(lam), p[0], p[1], rk.bbp_overlap(lam)))
    rk.write_dat(os.path.join("figures", "data", "06-amp-lambda-sim.dat"),
                 ["lam", "amp", "ampci", "pca", "pcaci", "seT", "qstar", "bbp"], rows)
    inside = sum(abs(r[1] - r[5]) <= r[2] for r in rows)
    print("AMP: SE q_T inside the interval at %d of %d values of lam" % (inside, len(rows)))
    inside = sum(abs(r[3] - r[7]) <= r[4] for r in rows)
    print("PCA: 1-1/lam^2 inside the interval at %d of %d values of lam" % (inside, len(rows)))
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
