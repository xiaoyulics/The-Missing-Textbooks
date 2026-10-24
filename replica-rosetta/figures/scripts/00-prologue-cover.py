"""Prologue: the number two, measured.

1. Cover's exact probability that n random points in general position in R^d, with random labels,
   are separable by a hyperplane through the origin, for d = 25, 50, 100, 200 (curves), against the
   fraction of separable instances decided by one linear program each at d = 100 (points, with
   Clopper--Pearson intervals).
2. The largest margin of random labeled patterns, kappa_max = max_{|w|=1} min_i <z_i, w>, which is
   the distance from the origin to the convex hull of z_1, ..., z_n, at d = 200, against the inverse
   of Gardner's curve alpha_c(kappa) = alpha. It is computed from the dual problem
   min_{a >= 0} (1/2)|Z^T a|^2 - sum(a): at the optimum |Z^T a|^2 = sum(a) = S and kappa_max = 1/sqrt(S).
   (The margin of the direction Z^T a agrees with 1/sqrt(S) to about 1e-6 on these instances.)
3. 200 further instances at alpha = 1.5 (seed 9), the point whose 20-instance interval misses the curve.

Writes figures/data/00-prologue-*.dat. Runtime about a minute and a half. Run from the volume root:
    python figures/scripts/00-prologue-cover.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import optimize  # noqa: E402

import rosettakit as rk  # noqa: E402

OUT = os.path.join("figures", "data")


def kappa_rs(alpha):
    """The margin kappa with alpha_c(kappa) = alpha (Gardner's curve inverted; for alpha > 2 it is
    negative, and there the replica-symmetric value is only an upper bound on the true margin)."""
    return rk.root(lambda k: rk.gardner_capacity(k) - alpha, -4.0, 6.0)


def max_margin(Z):
    """kappa_max = distance from the origin to conv(rows of Z), by the dual problem (L-BFGS-B)."""
    n = Z.shape[0]
    G = Z @ Z.T

    def f(a):
        Ga = G @ a
        return 0.5 * a @ Ga - a.sum(), Ga - 1.0

    res = optimize.minimize(f, np.full(n, 1.0 / n), jac=True, method="L-BFGS-B",
                            bounds=[(0, None)] * n, options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10})
    S = res.x.sum()
    return 1.0 / math.sqrt(S), res


def main():
    t0 = time.time()
    # 1. Cover's curves and LP simulations
    alphas = np.round(np.arange(1.0, 3.0001, 0.02), 4)
    rows = []
    for a in alphas:
        rows.append([a] + [rk.cover_probability(int(round(a * d)), d) for d in (25, 50, 100, 200)])
    rk.write_dat(os.path.join(OUT, "00-prologue-cover-curves.dat"), ["alpha", "d25", "d50", "d100", "d200"], rows)

    d, trials = 100, 60
    gen = rk.rng(2026)
    sim = []
    for a in np.round(np.arange(1.5, 2.5001, 0.1), 2):
        n = int(round(a * d))
        k = 0
        for _ in range(trials):
            Z = gen.standard_normal((n, d))
            k += rk.separable(Z)
        lo, hi = rk.clopper_pearson(k, trials)
        p = k / trials
        cov = rk.cover_probability(n, d)
        sim.append([a, p, p - lo, hi - p, cov])
        print(f"d={d} alpha={a:.1f}: separable {k}/{trials} = {p:.3f} [{lo:.3f}, {hi:.3f}]; Cover {cov:.4f}"
              f" ({'inside' if lo <= cov <= hi else 'OUTSIDE'} the interval)")
    rk.write_dat(os.path.join(OUT, "00-prologue-cover-sim.dat"), ["alpha", "frac", "lo", "hi", "cover"], sim)

    # 2. The largest margin of random labels against Gardner's curve
    d, reps = 200, 20
    gen = rk.rng(7)
    mar = []
    for a in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75):
        n = int(round(a * d))
        ks = []
        for _ in range(reps):
            Z = gen.standard_normal((n, d))
            k, res = max_margin(Z)
            ks.append(k)
        m, half = rk.mean_ci(ks)
        rs = kappa_rs(a)
        mar.append([a, m, half, rs])
        where = "contains it" if abs(m - rs) <= half else f"misses it by {abs(m - rs) - half:.4f}"
        print(f"d={d} alpha={a:.2f}: kappa_max = {m:.4f} +- {half:.4f}; Gardner inverse {rs:.4f}; the interval {where}")
    rk.write_dat(os.path.join(OUT, "00-prologue-margin.dat"), ["alpha", "kappa", "half", "rs"], mar)
    # 3. The point at alpha = 1.5 misses the curve: a finite-size bias, or a fluctuation of 20 instances?
    gen = rk.rng(9)
    n = int(round(1.5 * d))
    ks = [max_margin(gen.standard_normal((n, d)))[0] for _ in range(200)]
    m, half = rk.mean_ci(ks)
    print(f"d={d} alpha=1.50, 200 further instances (seed 9): kappa_max = {m:.4f} +- {half:.4f}; "
          f"Gardner inverse {kappa_rs(1.5):.4f}")
    curve = [[a, kappa_rs(a)] for a in np.round(np.arange(0.1, 2.0001, 0.05), 3)]
    rk.write_dat(os.path.join(OUT, "00-prologue-margin-curve.dat"), ["alpha", "kappa"], curve)
    curve_neg = [[a, kappa_rs(a)] for a in np.round(np.arange(2.0, 6.0001, 0.1), 3)]
    rk.write_dat(os.path.join(OUT, "00-prologue-margin-curveneg.dat"), ["alpha", "kappa"], curve_neg)
    print(f"Gardner: alpha_c(0) = {rk.gardner_capacity(0):.6f}, alpha_c(0.5) = {rk.gardner_capacity(0.5):.5f}, "
          f"alpha_c(1) = {rk.gardner_capacity(1):.5f}, alpha_c(-1) = {rk.gardner_capacity(-1):.5f}")
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
