"""Chapter 8 (Gaussian comparison): the maximal margin kappa_max(G) = max_{|theta|=1} min_mu <g_mu, theta>
of n = alpha d Gaussian patterns, measured exactly for alpha < 2 (the convex regime) and compared with
Gardner's curve kappa*(alpha), the solution of alpha_c(kappa) = alpha, which the capacity theorem proves.

kappa_max is computed by least-distance programming: min |w| s.t. G w >= 1 (Lawson--Hanson, via NNLS),
and kappa_max = 1/|w|.  An instance for which G w >= 1 is infeasible has no positive margin; it is
counted as kappa_max = 0 (a slight upward bias), and the number of such instances is printed.  For each instance one matrix with n_max = 1.75 d rows is drawn and the margin
of each prefix of alpha d rows is computed, alpha = 0.25, 0.5, ..., 1.75.
d = 100, 200, 400 with 200, 100, 40 instances (seed 80900 + d); means with Student-t 95% intervals.
Runtime: about 60 s.  Run from the volume root.
"""
import math
import os
import sys
import time

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

T0 = time.time()
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def max_margin(G):
    """max_{|theta|=1} min_mu <g_mu, theta> when positive (None if G w >= 1 is infeasible)."""
    n, d = G.shape
    E = np.vstack([G.T, np.ones((1, n))])
    f = np.zeros(d + 1)
    f[-1] = 1.0
    u, _ = optimize.nnls(E, f, maxiter=100 * n)
    r = E @ u - f
    if np.linalg.norm(r) < 1e-7:
        return None
    w = -r[:d] / r[d]
    return float(np.min(G @ w) / np.linalg.norm(w))   # equals 1/|w| at the optimum


def kappa_star(a):
    return rk.root(lambda k: rk.gardner_capacity(k) - a, -8.0, 12.0)


ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
TS = {100: 200, 200: 100, 400: 40}
rows = []
for d in TS:
    gen = rk.rng(80900 + d)
    nmax = int(round(1.75 * d))
    M = {a: [] for a in ALPHAS}
    nfail = {a: 0 for a in ALPHAS}   # instances with no positive margin, counted as kappa_max = 0
    t1 = time.time()
    for _ in range(TS[d]):
        G = gen.standard_normal((nmax, d))
        for a in ALPHAS:
            km = max_margin(G[: int(round(a * d))])
            nfail[a] += km is None
            M[a].append(km if km is not None else 0.0)
    for a in ALPHAS:
        m, h = rk.mean_ci(M[a])
        ks = kappa_star(a)
        rows.append([a, d, m, h, ks])
        print(f"d={d} alpha={a:4.2f}: kappa_max = {m:.4f} +- {h:.4f}   Gardner kappa*(alpha) = {ks:.4f}"
              f"   difference {m-ks:+.4f}   (no positive margin: {nfail[a]})")
    print(f"   d={d}: {TS[d]} instances in {time.time()-t1:.1f} s")
for d in TS:
    rk.write_dat(os.path.join(DATA, f"08-gordon-margin-d{d}.dat"), ["alpha", "d", "kappa", "ci", "pred"],
                 [r for r in rows if r[1] == d])
print(f"runtime {time.time()-T0:.1f} s")
