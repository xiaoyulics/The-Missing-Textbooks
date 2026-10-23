"""Chapter 1 (regular models): the Bayes generalization error n (E[G_n] - S) of three regular models.

Models as in 01-regular-free.py: the regular line and the regular plane (truth 0, prior uniform on [-1,1]^d)
and the mixture (1 - a) N(0,1) + a N(b,1) at the regular truth (1/2, 2) with the prior uniform on
[0, 1] x [-4, 4].

Linear models (R = 2000 data sets for each n = 10^2, 10^2.5, ..., 10^6). The data enter only through
S_xx = sum x_i x_i^T and S_xy = sum x_i y_i, which the script draws from their exact joint law when the
truth is 0: S_xx is Wishart(n, I_d) (Bartlett decomposition S_xx = L L^T) and S_xy = L z with z ~ N(0, I_d).
(A check against data drawn point by point is printed.) The posterior is N(w_hat, S_xx^{-1}) restricted to
[-1,1]^d; a bound for the Gaussian mass outside the box is printed (at most 2e-8 in our runs), and the
script neglects it: the predictive is N(x.w_hat, 1 + x' S_xx^{-1} x) and, for each data set,
    G_n - S = E_x[ (1/2) log(1 + s) - s / (2 (1 + s)) + (x.w_hat)^2 / (2 (1 + s)) ],   s = x' S_xx^{-1} x,
computed by Gauss-Hermite quadrature over x ~ N(0, I_d). Exact mean (the chapter's flat-prior computation):
    E[G_n] - S = (1/2) (psi((n + 1)/2) - psi((n - d + 1)/2)),  which is 1/(n - 1) for d = 2.
Also printed: the plug-in predictive p(y | x, w_hat), whose error is |w_hat|^2 / 2 with mean d / (2 (n - d - 1)),
and its training error, with mean -d / (2n): the gap is Akaike's d/n to first order.

Mixture (R = 200 data sets for each n = 10^2, ..., 10^4): the posterior by the midpoint rule on a 16 x 16
grid in coordinates whitened by the observed information (as in 01-regular-free.py; n (G_n - S) changes by
less than 1e-5 on a 48 x 48 grid), the predictive
p*(x) = sum_k omega_k p(x | w_k), and G_n - S = int q log(q / p*) dx by the trapezoid rule on [-10, 12].

Writes figures/data/01-regular-gen.dat (columns: n; for line, plane and mixture the mean of n (G_n - S) with
the half-widths of its 95% interval; the exact values for the linear models).
Run from the volume root: python figures/scripts/01-regular-gen.py      (one to two minutes)
"""
import importlib.util
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import special, stats  # noqa: E402

import sltkit as sk  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "regular_free", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01-regular-free.py"))
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)        # the mixture machinery (energy, MLE, posterior nodes)

R_LIN = 2000
R_MIX = 200
EXP_LIN = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
EXP_MIX = [2.0, 2.5, 3.0, 3.5, 4.0]
OUT = os.path.join("figures", "data")


def gh_nodes(d, k):
    z, w = np.polynomial.hermite_e.hermegauss(k)
    w = w / w.sum()
    if d == 1:
        return z[:, None], w
    zz = np.stack(np.meshgrid(z, z, indexing="ij"), -1).reshape(-1, 2)
    return zz, np.outer(w, w).ravel()


def sufficient_statistics(n, d, R, gen):
    """Exact draws of (S_xx, S_xy) for y = eps (truth 0), x ~ N(0, I_d): Bartlett decomposition."""
    L = np.zeros((R, d, d))
    for i in range(d):
        L[:, i, i] = np.sqrt(gen.chisquare(n - i, size=R))
        for j in range(i):
            L[:, i, j] = gen.standard_normal(R)
    Sxx = L @ np.transpose(L, (0, 2, 1))
    Sxy = np.einsum("rij,rj->ri", L, gen.standard_normal((R, d)))
    return Sxx, Sxy


def linear_gen_errors(Sxx, Sxy, nodes, wts):
    """Per data set: Bayes G_n - S (Gaussian predictive), plug-in error |w_hat|^2/2, training error of w_hat
    per sample times n (= -S_xy' S_xx^{-1} S_xy / 2), and the Gaussian mass outside [-1,1]^d (a bound)."""
    Sinv = np.linalg.inv(Sxx)
    what = np.einsum("rij,rj->ri", Sinv, Sxy)
    s = np.einsum("ki,rij,kj->rk", nodes, Sinv, nodes)
    m = what @ nodes.T
    bayes = (0.5 * np.log1p(s) - s / (2 * (1 + s)) + m ** 2 / (2 * (1 + s))) @ wts
    plugin = 0.5 * np.sum(what ** 2, axis=1)
    train = -0.5 * np.einsum("ri,ri->r", Sxy, what)
    sd = np.sqrt(np.einsum("rii->ri", Sinv))
    outside = np.sum(2 * stats.norm.sf((1 - np.abs(what)) / sd), axis=1)
    return bayes, plugin, train, outside


def exact_bayes(n, d):
    return 0.5 * (special.digamma((n + 1) / 2) - special.digamma((n - d + 1) / 2))


def mixture_gen_error(xs, grid, qgrid, dx):
    w_hat, H, _ = rf.mix_mle(xs)
    W, wts, _ = rf.mix_posterior_nodes(xs, w_hat, H, N=16)
    ok = wts > 0
    W, wts = W[ok], wts[ok] / np.sum(wts[ok])
    a, b = W[:, 0:1], W[:, 1:2]
    # p(x | a, b) / phi(x) = 1 - a + a exp(b x - b^2/2); q / phi likewise at the truth
    ratio_p = 1 + a * np.expm1(b * grid[None, :] - 0.5 * b * b)
    pstar_over_phi = wts @ ratio_p
    q_over_phi = 1 + rf.A0 * np.expm1(rf.B0 * grid - 0.5 * rf.B0 ** 2)
    integrand = qgrid * np.log(q_over_phi / pstar_over_phi)
    return float(np.sum(0.5 * (integrand[1:] + integrand[:-1])) * dx)


def main():
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)
    gen = sk.rng(20261024)

    # ---- check the sufficient-statistic sampler against data drawn point by point (d = 2, n = 100)
    nodes2, wts2 = gh_nodes(2, 40)
    n = 100
    Sxx, Sxy = sufficient_statistics(n, 2, R_LIN, gen)
    b_ss = n * linear_gen_errors(Sxx, Sxy, nodes2, wts2)[0]
    Sxx_r, Sxy_r = np.zeros((R_LIN, 2, 2)), np.zeros((R_LIN, 2))
    for r in range(R_LIN):
        x, y = sk.REGULAR2.sample([0.0, 0.0], n, seed=500_000 + r)
        Sxx_r[r], Sxy_r[r] = x.T @ x, x.T @ y
    b_rd = n * linear_gen_errors(Sxx_r, Sxy_r, nodes2, wts2)[0]
    m1, l1, h1 = sk.mean_ci(b_ss)
    m2, l2, h2 = sk.mean_ci(b_rd)
    print(f"check, regular plane, n = 100: n(G_n - S) mean {m1:.4f} [{l1:.4f}, {h1:.4f}] (sufficient statistics), "
          f"{m2:.4f} [{l2:.4f}, {h2:.4f}] (data point by point); exact {n * exact_bayes(n, 2):.4f}")

    # ---- the regular line and the regular plane
    rows_lin = {}
    worst_out = 0.0
    for d in (1, 2):
        nodes, wts = gh_nodes(d, 60 if d == 1 else 40)
        for ex in EXP_LIN:
            n = int(round(10 ** ex))
            Sxx, Sxy = sufficient_statistics(n, d, R_LIN, gen)
            bayes, plugin, train, outside = linear_gen_errors(Sxx, Sxy, nodes, wts)
            worst_out = max(worst_out, float(np.max(outside)))
            rows_lin[(d, n)] = (sk.mean_ci(n * bayes), sk.mean_ci(n * plugin), sk.mean_ci(train))
    print(f"largest Gaussian posterior mass outside [-1,1]^d over all linear data sets: {worst_out:.1e}")

    # ---- the mixture
    grid = np.linspace(-10.0, 12.0, 4401)
    dx = grid[1] - grid[0]
    qgrid = rf.q_density(grid)
    rows_mix = {}
    for ex in EXP_MIX:
        n = int(round(10 ** ex))
        vals = [n * mixture_gen_error(rf.mix_sample(n, seed=7_000_000 + 1000 * int(ex * 2) + r), grid, qgrid, dx)
                for r in range(R_MIX)]
        rows_mix[n] = sk.mean_ci(vals)

    # ---- write and print
    rows = []
    for ex in EXP_LIN:
        n = int(round(10 ** ex))
        (b1, bl1, bh1), (p1, _, _), (t1, _, _) = rows_lin[(1, n)]
        (b2, bl2, bh2), (p2, _, _), (t2, _, _) = rows_lin[(2, n)]
        mix = rows_mix.get(n)
        mix_cols = (mix[0], mix[0] - mix[1], mix[2] - mix[0]) if mix else ("nan", "nan", "nan")
        rows.append((n, b1, b1 - bl1, bh1 - b1, n * exact_bayes(n, 1), b2, b2 - bl2, bh2 - b2,
                     n * exact_bayes(n, 2), *mix_cols))
        print(f"n=1e{ex:.1f}: line n(G-S) {b1:.4f} [{bl1:.4f},{bh1:.4f}] exact {n * exact_bayes(n, 1):.4f} | "
              f"plane {b2:.4f} [{bl2:.4f},{bh2:.4f}] exact {n * exact_bayes(n, 2):.4f}"
              + (f" | mixture {mix[0]:.3f} [{mix[1]:.3f},{mix[2]:.3f}]" if mix else ""))
        print(f"          plug-in n E[K(w_hat)]: line {p1:.4f} (exact {n * 1 / (2 * (n - 2)):.4f}), plane {p2:.4f} "
              f"(exact {n * 2 / (2 * (n - 3)):.4f}); n E[L_n(w_hat) - S_n]: line {t1:.4f}, plane {t2:.4f} (exact -d/2)")
    sk.write_dat(os.path.join(OUT, "01-regular-gen.dat"),
                 ["n", "line", "linelo", "linehi", "lineexact", "plane", "planelo", "planehi", "planeexact",
                  "mix", "mixlo", "mixhi"], rows)
    print(f"total time {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
