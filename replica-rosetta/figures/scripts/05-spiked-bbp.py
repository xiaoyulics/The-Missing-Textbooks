"""Chapter 5 (Spiked Matrices): the BBP transition measured at finite N.

Model (OUTLINE.md): A = (lam/N) x x^T + W, W ~ GOE(N) (off-diagonal variance 1/N, diagonal 2/N),
x uniform on {-1,+1}^N (Z2 synchronization), so |x|^2 = N exactly and A = W + lam u u^T, u = x/sqrt(N).
Measured: the top eigenvalue lambda_1(A) and the squared overlap <v_1, u>^2 of a unit top eigenvector,
against the limits of Theorem thm:spiked:bbp: lam + 1/lam and 1 - 1/lam^2 for lam > 1, 2 and 0 otherwise.
Also checked on one instance: the secular equation of Lemma lem:spiked:secular,
1 + theta <u, R_W(mu) u> = 0 at mu = lambda_1(A), with the resolvent R_W(z) = (W - z)^{-1}, and the
overlap formula 1/(theta^2 <u, R_W(mu)^2 u>) of eq:toolkit:overlap.

Output: figures/data/05-spiked-bbp.dat (measurements), figures/data/05-spiked-bbp-curve.dat (limits).
Run from the volume root: python figures/scripts/05-spiked-bbp.py   (one to two minutes)
"""
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

LAMS = [0.0, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]
SIZES = [(500, 40), (2000, 10)]          # (N, repetitions)


def top_pair(A, v0=None):
    """Largest eigenvalue and a unit eigenvector of the symmetric matrix (or operator) A, by Lanczos
    started from v0; the default start is the constant vector, so that every run is deterministic."""
    if v0 is None:
        v0 = np.ones(A.shape[0]) / np.sqrt(A.shape[0])
    val, vec = eigsh(A, k=1, which="LA", tol=1e-9, maxiter=20000, v0=v0)
    return float(val[0]), vec[:, 0]


def spiked_operator(W, u, lam):
    """v -> W v + lam u <u, v>, the matrix W + lam u u^T without forming it."""
    n = W.shape[0]
    return LinearOperator((n, n), matvec=lambda v: W @ v + lam * u * float(u @ v), dtype=float)


def check_secular(N=400, lam=1.5, seed=9):
    """Lemma lem:spiked:secular on one instance, and eigsh against a dense eigendecomposition."""
    gen = rk.rng(seed)
    W = rk.goe(N, gen)
    x = gen.choice([-1.0, 1.0], size=N)
    u = x / np.sqrt(N)
    A = W + lam * np.outer(u, u)
    mu, v = top_pair(A)
    w, V = np.linalg.eigh(A)
    R = np.linalg.inv(W - mu * np.eye(N))      # the resolvent R_W(mu) = (W - mu)^{-1}
    r = float(u @ R @ u)
    ov_formula = 1.0 / (lam ** 2 * float(u @ R @ R @ u))
    print("check (N=%d, lam=%.1f): eigsh %.12f vs eigh %.12f; 1 + theta*<u,R u> = %.2e; "
          "overlap %.10f vs formula %.10f" % (N, lam, mu, w[-1], 1.0 + lam * r, (v @ u) ** 2, ov_formula))


def main():
    t0 = time.time()
    check_secular()
    rows = []
    for N, reps in SIZES:
        ev = {lam: [] for lam in LAMS}
        ov = {lam: [] for lam in LAMS}
        for r in range(reps):
            gen = rk.rng(5000 + 100 * N + r)
            W = rk.goe(N, gen)
            x = gen.choice([-1.0, 1.0], size=N)
            u = x / np.sqrt(N)
            v = None
            for lam in LAMS:                       # warm start: the previous top eigenvector
                mu, v = top_pair(spiked_operator(W, u, lam), v)
                ev[lam].append(mu)
                ov[lam].append(float(v @ u) ** 2)
        for lam in LAMS:
            me, he = rk.mean_ci(ev[lam])
            mo, ho = rk.mean_ci(ov[lam])
            rows.append([N, lam, me, he, mo, ho, rk.bbp_eigenvalue(lam), rk.bbp_overlap(lam)])
            print("N=%5d lam=%.2f  top eigenvalue %.4f +- %.4f (limit %.4f)   overlap^2 %.4f +- %.4f (limit %.4f)"
                  % (N, lam, me, he, rk.bbp_eigenvalue(lam), mo, ho, rk.bbp_overlap(lam)))
        print("   N=%d done after %.1f s" % (N, time.time() - t0))
    rk.write_dat("figures/data/05-spiked-bbp.dat",
                 ["N", "lam", "eig", "eigci", "ov", "ovci", "eiglim", "ovlim"], rows)
    grid = np.linspace(0.0, 3.2, 321)
    rk.write_dat("figures/data/05-spiked-bbp-curve.dat", ["lam", "eig", "ov"],
                 [[float(l), rk.bbp_eigenvalue(l), rk.bbp_overlap(l)] for l in grid])
    # finite-size scaling at the threshold lam = 1 (overlap ~ N^{-1/3}, eigenvalue - 2 ~ N^{-2/3})
    r500 = [r for r in rows if r[0] == 500 and r[1] == 1.0][0]
    r2000 = [r for r in rows if r[0] == 2000 and r[1] == 1.0][0]
    print("at lam = 1: overlap ratio N=500/N=2000 = %.2f (N^{-1/3} scaling predicts 4^{1/3} = %.2f); "
          "eigenvalue - 2: %.4f and %.4f" % (r500[4] / r2000[4], 4 ** (1 / 3), r500[2] - 2, r2000[2] - 2))
    print("total time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
