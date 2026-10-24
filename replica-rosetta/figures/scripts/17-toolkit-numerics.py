"""Appendix A (toolkit): every number quoted in the appendix, recomputed with the kit.

1. Gauss--Hermite errors of `gauss` (Table A.1): E cos Z, E tanh(2 + sqrt2 Z), E log cosh(1.5 Z + 0.3),
   E|Z| at n = 11, ..., 201 nodes, against closed forms or mpmath at 30 digits; `gauss_split` on E|Z|;
   the weights at 121, 161, 201 nodes; where the kit's H underflows.
2. Damped fixed-point iteration for q = 1 - mmse(lam^2 q), Rademacher prior (lam = 2 and 1.02).
3. Several fixed points: sparse Rademacher prior, rho = 0.05, lam = 0.92; the potential difference;
   the information-theoretic threshold lam_IT.
4. Intervals: coverage of `mean_ci` for exponential replicates; infimum of the Clopper--Pearson coverage.
5. Rice's formula for a random trigonometric polynomial of degree K = 10.
6. Laplace and Varadhan for the Curie--Weiss model at beta = 1.5, h = 0.1.

Writes figures/data/17-toolkit-quadrature.dat. Runtime about half a minute. Run from the volume root:
    python figures/scripts/17-toolkit-numerics.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import optimize, special, stats  # noqa: E402

import rosettakit as rk  # noqa: E402

OUT = os.path.join("figures", "data")


def mp_gauss(f):
    """E f(Z) by mpmath quadrature at 30 digits (reference values)."""
    mp.mp.dps = 30
    return mp.quad(lambda z: f(z) * mp.e ** (-z * z / 2), [-mp.inf, 0, mp.inf]) / mp.sqrt(2 * mp.pi)


def quadrature():
    print("== 1. Gauss--Hermite quadrature ==")
    ref = {"cos": math.exp(-0.5),
           "tanh": float(mp_gauss(lambda z: mp.tanh(2 + mp.sqrt(2) * z))),
           "logcosh": float(mp_gauss(lambda z: mp.log(mp.cosh(1.5 * z + 0.3)))),
           "abs": math.sqrt(2 / math.pi)}
    funcs = {"cos": np.cos, "tanh": lambda z: np.tanh(2 + math.sqrt(2) * z),
             "logcosh": lambda z: np.log(np.cosh(1.5 * z + 0.3)), "abs": np.abs}
    rows = []
    for n in (11, 21, 41, 61, 121, 201):
        errs = [abs(rk.gauss(funcs[k], n) - ref[k]) for k in ("cos", "tanh", "logcosh", "abs")]
        rows.append([n] + errs)
        print(f"n={n:3d}: |error| cos {errs[0]:.1e}, tanh {errs[1]:.1e}, log cosh {errs[2]:.1e}, |z| {errs[3]:.1e}")
    rk.write_dat(os.path.join(OUT, "17-toolkit-quadrature.dat"), ["n", "cos", "tanh", "logcosh", "abs"], rows)
    print(f"gauss_split, E|Z|: |error| {abs(rk.gauss_split(np.abs, 0.0) - ref['abs']):.1e}")
    for n in (121, 161, 201):
        x, w = rk._gh(n)
        print(f"n={n}: |sum w - 1| = {abs(w.sum() - 1):.1e}, weights symmetric: {bool(np.array_equal(w, w[::-1]))}")
    print(f"H(37.5) = {float(rk.H(37.5)):.2e}, H(38) = {float(rk.H(38.0))}, logH(38) = {float(rk.logH(38.0)):.4f}")


def fixed_points():
    print("== 2. Damped iteration, q = 1 - mmse(lam^2 q), Rademacher prior, from q0 = 1, tol = 1e-13 ==")

    def F(q, lam):
        return 1.0 - rk.mmse_rademacher(lam ** 2 * max(q, 0.0))

    for lam in (2.0, 1.02):
        qstar = rk.root(lambda q: F(q, lam) - q, 1e-6, 1.0)
        h = 1e-6
        slope = (F(qstar + h, lam) - F(qstar - h, lam)) / (2 * h)
        q1, it1, _ = rk.fixed_point(lambda q: F(q, lam), 1.0, damping=1.0, tol=1e-13)
        q5, it5, _ = rk.fixed_point(lambda q: F(q, lam), 1.0, damping=0.5, tol=1e-13)
        print(f"lam={lam}: q*={qstar:.10f}, F'(q*)={slope:.3f}; undamped {it1} iterations, error {abs(q1 - qstar):.1e} "
              f"(estimate tol r/(1-r) = {1e-13 * slope / (1 - slope):.1e}); damping 1/2: {it5} iterations "
              f"({it5 / it1:.1f} times as many)")


def several_fixed_points():
    print("== 3. Sparse Rademacher prior, rho = 0.05 ==")
    rho = 0.05

    def F(q, lam):
        return 1.0 - rk.mmse_sparse_rademacher(lam ** 2 * max(q, 0.0), rho, n=201)

    def roots_of(lam, grid):
        G = [F(q, lam) - q for q in grid]
        out = [0.0]
        for a, b, ga, gb in zip(grid[:-1], grid[1:], G[:-1], G[1:]):
            if ga * gb < 0:
                out.append(rk.root(lambda q: F(q, lam) - q, a, b))
        return out

    def dpsi(lam, q1, q2, n=400):
        """Psi(q2) - Psi(q1) = (lam^2/2) int_{q1}^{q2} (q - F(q)) dq (Gauss--Legendre)."""
        xs, ws = np.polynomial.legendre.leggauss(n)
        qs = 0.5 * (q2 - q1) * xs + 0.5 * (q2 + q1)
        return 0.5 * lam ** 2 * 0.5 * (q2 - q1) * float(np.sum(ws * np.array([q - F(q, lam) for q in qs])))

    lam = 0.92
    rts = roots_of(lam, np.concatenate(([1e-6], np.linspace(0.002, 1.0, 120))))
    for r in rts:
        h = 1e-6
        lo, hi = max(r - h, 0.0), r + h
        s = (F(hi, lam) - F(lo, lam)) / (hi - lo)
        print(f"lam={lam}: fixed point q = {r:.5f}, F'(q) = {s:.3f} ({'stable' if abs(s) < 1 else 'unstable'})")
    for q0 in (1.0, 0.3, 0.1, 1e-3):
        q, it, ok = rk.fixed_point(lambda q: F(q, lam), q0, damping=0.5, tol=1e-12)
        print(f"  damped iteration (1/2) from q0 = {q0}: q = {q:.5f} ({it} iterations, converged {ok})")
    print(f"  Psi(q+) - Psi(0) = {dpsi(lam, 0.0, rts[-1]):.4f}")
    grid = np.linspace(0.002, 1.0, 80)
    lam_it = rk.root(lambda l: dpsi(l, 0.0, roots_of(l, grid)[-1]), 0.85, 0.90, tol=1e-9)
    print(f"  lambda_IT (Psi(q+) = Psi(0)) = {lam_it:.5f}")


def intervals():
    print("== 4. Intervals ==")
    gen = rk.rng(2026)
    for R in (5, 10, 30):
        hits = 0
        for _ in range(20000):
            m, h = rk.mean_ci(gen.exponential(1.0, R))
            hits += (m - h <= 1.0 <= m + h)
        print(f"Student-t interval, exponential replicates, R={R}: coverage {hits / 20000:.3f} (20000 repetitions, seed 2026)")
    for n in (10, 50, 200, 1000):
        ends = np.array([rk.clopper_pearson(k, n) for k in range(n + 1)])
        ks = np.arange(n + 1)
        best = 1.0
        # the coverage is unimodal between consecutive end points, so its infimum is a one-sided limit at one of them
        for c in np.unique(ends):
            for th in (c - 1e-10, c + 1e-10):
                if 0 < th < 1:
                    cov = float(np.sum(stats.binom.pmf(ks, n, th) * ((ends[:, 0] <= th) & (th <= ends[:, 1]))))
                    best = min(best, cov)
        print(f"Clopper--Pearson, n={n}: infimum of the coverage {100 * best:.3f}%")


def rice():
    print("== 5. Rice's formula, K = 10 ==")
    K = 10
    print(f"formula 2 sqrt((K+1)(2K+1)/6) = {2 * math.sqrt((K + 1) * (2 * K + 1) / 6):.3f}")
    gen = rk.rng(7)
    counts = []
    for _ in range(20000):
        a, b = gen.standard_normal(K), gen.standard_normal(K)
        # X(t) = sum_k a_k cos kt + b_k sin kt = e^{-iKt} P(e^{it}), P of degree 2K; zeros of X <-> roots of P on |z| = 1
        c = np.zeros(2 * K + 1, dtype=complex)          # c[j] = coefficient of z^j
        c[K + 1:] = (a - 1j * b) / 2
        c[:K][::-1] = (a + 1j * b) / 2
        z = np.roots(c[::-1])
        counts.append(int(np.sum(np.abs(np.abs(z) - 1.0) < 1e-6)))
    m, half = rk.mean_ci(counts)
    print(f"20000 draws (seed 7): mean number of zeros in [0, 2pi) = {m:.3f} +- {half:.3f}")


def curie_weiss():
    print("== 6. Curie--Weiss, beta = 1.5, h = 0.1 ==")
    beta, h = 1.5, 0.1

    def entropy(m):
        p, q = (1 + m) / 2, (1 - m) / 2
        return -(special.xlogy(p, p) + special.xlogy(q, q))

    r1 = optimize.minimize_scalar(lambda m: -(beta * m * m / 2 + h * m + entropy(m)), bounds=(-1 + 1e-12, 1 - 1e-12),
                                  method="bounded", options={"xatol": 1e-12})
    r2 = optimize.minimize_scalar(lambda y: -(-beta * y * y / 2 + math.log(2 * math.cosh(beta * y + h))), bounds=(-3, 3),
                                  method="bounded", options={"xatol": 1e-12})
    print(f"log 2 + sup_m (F - I) = {-r1.fun:.12f}; sup_y (-beta y^2/2 + log 2cosh(beta y + h)) = {-r2.fun:.12f}")
    N = 1000
    k = np.arange(N + 1)
    m = (2 * k - N) / N
    e = special.gammaln(N + 1) - special.gammaln(k + 1) - special.gammaln(N - k + 1) + N * (beta * m * m / 2 + h * m)
    phiN = (e.max() + math.log(np.exp(e - e.max()).sum())) / N
    print(f"N={N}: (1/N) log Z_N = {phiN:.10f}, exceeds the limit by {phiN + r1.fun:.2e}")


def main():
    t0 = time.time()
    quadrature()
    fixed_points()
    several_fixed_points()
    intervals()
    rice()
    curie_weiss()
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
