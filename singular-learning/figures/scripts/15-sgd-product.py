"""Chapter 15 (SGD Is Not Bayes), "By hand" and "Measuring it": where gradient flow stops on the
product model y = abx, against exact posterior draws (Theorem thm:sgd:balance).

The product model y = abx + N(0,1) noise, x ~ N(0,1), uniform prior on [-1,1]^2. The empirical loss
is L_n(a,b) = (kappa/2)(ab - c_hat)^2 + const with kappa = S_xx/n and c_hat = S_xy/S_xx, so both
gradient flow and the posterior depend on the data only through (S_xx, S_xy). At the singular truth
S_xx ~ chi^2_n and S_xy | S_xx ~ N(0, S_xx) exactly, so data sets of size 10^8 are simulated exactly.

Gradient flow. Theorem thm:sgd:balance: with u = a + b, v = a - b and D = a0^2 - b0^2 (conserved),
the flow converges to the point with u v = D, u^2 - v^2 = 4 c_hat and the signs of (u0, v0):
    u^2 = 2 c_hat + sqrt(D^2 + 4 c_hat^2),  v^2 = -2 c_hat + sqrt(D^2 + 4 c_hat^2),
(the origin when the start lies on the line u = 0 for c_hat > 0, or v = 0 for c_hat < 0).
Part 1 checks this closed form against numerical integration of the flow (scipy DOP853).

Posterior. Exact draws (sk.sample_product_posterior); exact distribution function of log|a| from
Theorem thm:singular:scales (i), computed as in 02-singular-scales.py:
    Pi_n(log|a| <= u) = H_Z(u + ell) / H_Z(ell),  H_Z(L) = int_{-inf}^L P(|Z + xi| <= e^w) dw,
with Z = S_xy / sqrt(S_xx), ell = (1/2) log S_xx and xi ~ N(0,1).

Outputs (figures/data):
  15-sgd-portrait.dat        flow lines for the phase portrait (c_hat = 0.25, kappa = 1): id a b
  15-sgd-portrait-ends.dat   their starting and end points
  15-sgd-portrait-arrows.dat one arrow per flow line (point and direction of motion)
  15-sgd-scatter-post.dat    1500 exact posterior draws at n = 10^12 (log10|a|, log10|b|), one data set
  15-sgd-scatter-prior.dat   flow end points from 300 starts uniform on [-1,1]^2, same data set
  15-sgd-scatter-small.dat   flow end points from 300 starts N(0, 10^-12 I), same data set
  15-sgd-cdf.dat             distribution function of log|a|/log n at n = 10^12 on a grid of tau:
                             mean over 200 data sets, with Student-t 95% half-widths, for the
                             posterior draws (2000 per data set) and the flow from the two starts
                             (200 starts per data set); and the exact data-averaged posterior value
Run from the volume root: python figures/scripts/15-sgd-product.py   (about 40 s)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import special, stats
from scipy.integrate import solve_ivp

import sltkit as sk

OUT = os.path.join("figures", "data")
WGRID = np.arange(-60.0, 30.0, 0.002)     # grid in w = log t for H_Z


def flow_limit(a0, b0, c_hat):
    """Limit of gradient flow on (kappa/2)(ab - c_hat)^2 from (a0, b0) (Theorem thm:sgd:balance).
    np.sign(0) = 0 sends the exceptional line to the origin."""
    a0, b0 = np.asarray(a0, dtype=float), np.asarray(b0, dtype=float)
    D = a0 * a0 - b0 * b0
    root = np.sqrt(D * D + 4.0 * c_hat * c_hat)
    u = np.sign(a0 + b0) * np.sqrt(np.maximum(2.0 * c_hat + root, 0.0))
    v = np.sign(a0 - b0) * np.sqrt(np.maximum(-2.0 * c_hat + root, 0.0))
    return (u + v) / 2.0, (u - v) / 2.0


def log_abs_a_limit(a0, b0, c_hat):
    """log|a_inf| = (1/2) log|c_hat| + (1/2) arcsinh(D / (2|c_hat|)), off the exceptional line."""
    D = np.asarray(a0) ** 2 - np.asarray(b0) ** 2
    return 0.5 * math.log(abs(c_hat)) + 0.5 * np.arcsinh(D / (2.0 * abs(c_hat)))


def H_table(z):
    t = np.exp(WGRID)
    g = special.ndtr(t - z) - special.ndtr(-t - z)
    return np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(WGRID))])


def exact_cdf(u, z, ell):
    """Pi_n(log|a| <= u), u <= 0 (Theorem thm:singular:scales (i))."""
    H = H_table(z)
    return np.interp(np.asarray(u) + ell, WGRID, H) / np.interp(ell, WGRID, H)


def product_model(n, Sxx, Sxy):
    m = sk.LinearFeatureModel(lambda w: (w[..., 0] * w[..., 1])[..., None], d=2, k=1, name="y = abx")
    m.n, m.Sxx, m.Sxy, m.Syy = int(n), np.array([[Sxx]]), np.array([Sxy]), float("nan")
    return m


def stats_singular_truth(n, gen):
    Sxx = float(gen.chisquare(n))
    Sxy = float(gen.normal(0.0, math.sqrt(Sxx)))
    return Sxx, Sxy


def part1_check():
    """Closed form against numerical integration, on data sets of size 10^4 at the singular truth."""
    gen = sk.rng(1501)
    worst, worst_cons, count = 0.0, 0.0, 0
    for k in range(6):
        Sxx, Sxy = stats_singular_truth(10_000, gen)
        kappa, c_hat = Sxx / 10_000, Sxy / Sxx
        if abs(c_hat) < 3e-3:            # keep integration times moderate (rate kappa*sqrt(D^2+4c^2))
            continue

        def rhs(t, w):
            r = kappa * (w[0] * w[1] - c_hat)
            return [-r * w[1], -r * w[0]]

        for j in range(6):
            a0, b0 = gen.uniform(-1, 1, 2)
            if j == 5:                    # a small start: plateau, then the balanced point
                a0, b0 = 1e-3 * gen.standard_normal(2)
            T = 60.0 / (kappa * math.sqrt((a0 * a0 - b0 * b0) ** 2 + 4 * c_hat ** 2)) + 40.0 / abs(c_hat)
            sol = solve_ivp(rhs, [0.0, T], [a0, b0], method="DOP853", rtol=1e-11, atol=1e-14)
            aT, bT = sol.y[:, -1]
            ac, bc = flow_limit(a0, b0, c_hat)
            worst = max(worst, abs(aT - ac), abs(bT - bc))
            worst_cons = max(worst_cons, float(np.max(np.abs(sol.y[0] ** 2 - sol.y[1] ** 2 - (a0 * a0 - b0 * b0)))))
            count += 1
    print(f"[check] {count} flows integrated numerically: max |end point - closed form| = {worst:.2e}, "
          f"max drift of a^2 - b^2 = {worst_cons:.2e}")


def part2_byhand():
    """The numbers of Computations 15.1 and 15.2."""
    c_hat, kappa = 0.01, 1.0
    print("[by hand] c_hat = 0.01, kappa = 1")
    for (a0, b0) in [(0.6, 0.3), (0.3, 0.6), (0.6, -0.3), (0.6, 0.6), (0.6, -0.6), (1e-3, 2e-3)]:
        a, b = flow_limit(a0, b0, c_hat)
        D = a0 * a0 - b0 * b0
        sharp = kappa * math.sqrt(D * D + 4 * c_hat * c_hat)
        print(f"   start ({a0:g}, {b0:g}): D = {D:.6g}, end ({float(a):.6f}, {float(b):.6f}), ab = {float(a * b):.6f}, "
              f"top Hessian eigenvalue kappa(a^2+b^2) = {sharp:.5f}")
    c0 = 1e-3 * 2e-3
    print(f"   plateau time from (1e-3, 2e-3): (1/(2 kappa c_hat)) log(c_hat/c0) = {math.log(c_hat / c0) / (2 * kappa * c_hat):.0f}")
    # Computation 15.2 at n = 10^6, |c_hat| = 10^-3: mass beyond |a| = 0.1
    n, Z = 10**6, 1.0
    ell = 0.5 * math.log(n)
    p_post = 1.0 - float(exact_cdf(math.log(0.1), Z, ell))
    gen = sk.rng(1502)
    a0, b0 = gen.uniform(-1, 1, (2, 1_000_000))
    la = log_abs_a_limit(a0, b0, 1e-3)
    p_flow = float(np.mean(la > math.log(0.1)))
    print(f"   n = 10^6, Z = 1 (c_hat = 1e-3): posterior P(|a| > 0.1) = {p_post:.3f}; "
          f"flow from the prior P(|a_inf| > 0.1) = {p_flow:.3f}; flow from a small start: |a_inf| = {math.sqrt(1e-3):.4f}")
    print(f"   prior P(a0^2 - b0^2 > 0) = {np.mean(a0**2 > b0**2):.4f}")


def part3_portrait():
    """Flow lines of the phase portrait: each lies on its hyperbola a^2 - b^2 = D between the start and
    the limit; parametrized by u = a + b, with v = D/u (or along the diagonal when D = 0)."""
    c_hat = 0.25
    starts = [(-0.95, 0.80), (-0.95, 0.35), (-0.95, -0.15), (-0.60, -0.95), (0.10, -0.95), (0.55, -0.95),
              (0.95, -0.80), (0.95, -0.35), (0.95, 0.15), (0.60, 0.95), (-0.10, 0.95), (-0.55, 0.95),
              (0.04, 0.03), (-0.03, -0.04), (0.70, -0.70), (-0.70, 0.70)]
    rows, ends, arrows = [], [], []
    for k, (a0, b0) in enumerate(starts):
        a1, b1 = flow_limit(a0, b0, c_hat)
        a1, b1 = float(a1), float(b1)
        u0, u1 = a0 + b0, a1 + b1
        D = a0 * a0 - b0 * b0
        if abs(u0) < 1e-12:               # the exceptional line a = -b: v shrinks to 0 along the line
            v = np.linspace(a0 - b0, 0.0, 120)
            a, b = v / 2.0, -v / 2.0
        elif abs(D) < 1e-12:              # the diagonal a = b
            u = np.linspace(u0, u1, 120)
            a, b = u / 2.0, u / 2.0
        else:
            u = u0 * (u1 / u0) ** np.linspace(0.0, 1.0, 240)     # geometric in u: even spacing along the arc
            v = D / u
            a, b = (u + v) / 2.0, (u - v) / 2.0
        for ai, bi in zip(a, b):
            rows.append((k, ai, bi))
        ends.append((k, a0, b0, a1, b1))
        # an arrow at the middle of the arc (by arc length), pointing in the direction of motion
        seg = np.hypot(np.diff(a), np.diff(b))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        j = int(np.searchsorted(s, 0.45 * s[-1]))
        j = min(max(j, 1), len(a) - 2)
        da, db = a[j + 1] - a[j - 1], b[j + 1] - b[j - 1]
        h = math.hypot(da, db)
        arrows.append((k, a[j], b[j], 0.06 * da / h, 0.06 * db / h))
    sk.write_dat(os.path.join(OUT, "15-sgd-portrait.dat"), ["id", "a", "b"], rows)
    sk.write_dat(os.path.join(OUT, "15-sgd-portrait-ends.dat"), ["id", "a0", "b0", "a1", "b1"], ends)
    sk.write_dat(os.path.join(OUT, "15-sgd-portrait-arrows.dat"), ["id", "a", "b", "da", "db"], arrows)
    print(f"[portrait] {len(starts)} flow lines for c_hat = {c_hat}")


def part4_scatter_and_cdf():
    n = 10**12
    small = 1e-6                          # scale of the small starts: small^2 << |c_hat| ~ n^-1/2
    logn = math.log(n)
    gen = sk.rng(1503)
    # one data set for the scatter
    Sxx, Sxy = stats_singular_truth(n, gen)
    c_hat = Sxy / Sxx
    print(f"[scatter] n = 1e12: c_hat = {c_hat:.3e}, log10|c_hat| = {math.log10(abs(c_hat)):.3f}")
    draws = sk.sample_product_posterior(product_model(n, Sxx, Sxy), 1500, seed=1504)
    sk.write_dat(os.path.join(OUT, "15-sgd-scatter-post.dat"), ["la", "lb"],
                 np.log10(np.abs(draws)))
    a0, b0 = gen.uniform(-1, 1, (2, 300))
    a, b = flow_limit(a0, b0, c_hat)
    sk.write_dat(os.path.join(OUT, "15-sgd-scatter-prior.dat"), ["la", "lb"],
                 np.column_stack([np.log10(np.abs(a)), np.log10(np.abs(b))]))
    lpa = np.log10(np.abs(a))
    print(f"   flow from the prior: {np.mean(lpa > -1):.3f} with log10|a| > -1, {np.mean(lpa < math.log10(abs(c_hat)) + 1):.3f} "
          f"within a decade of the other end, {np.mean((lpa > -5) & (lpa < -1)):.3f} between")
    s0, t0 = small * gen.standard_normal((2, 300))
    a, b = flow_limit(s0, t0, c_hat)
    sk.write_dat(os.path.join(OUT, "15-sgd-scatter-small.dat"), ["la", "lb"],
                 np.column_stack([np.log10(np.abs(a)), np.log10(np.abs(b))]))
    print(f"   small start: log10|a_inf| from {np.log10(np.abs(a)).min():.3f} to {np.log10(np.abs(a)).max():.3f}")

    # distribution functions over R data sets
    R, M, K = 200, 2000, 200
    taus = np.round(np.arange(-0.65, 0.0501, 0.025), 3)
    F_post, F_prior, F_small, F_exact = [], [], [], []
    mid_post, mid_prior, mid_small = [], [], []
    for r in range(R):
        Sxx, Sxy = stats_singular_truth(n, gen)
        c_hat = Sxy / Sxx
        z, ell = Sxy / math.sqrt(Sxx), 0.5 * math.log(Sxx)
        d = sk.sample_product_posterior(product_model(n, Sxx, Sxy), M, seed=20_000 + r)
        tp = np.log(np.abs(d[:, 0])) / logn
        a0, b0 = gen.uniform(-1, 1, (2, K))
        tq = np.log(np.abs(flow_limit(a0, b0, c_hat)[0])) / logn
        s0, t0 = small * gen.standard_normal((2, K))
        ts = np.log(np.abs(flow_limit(s0, t0, c_hat)[0])) / logn
        F_post.append([np.mean(tp <= t) for t in taus])
        F_prior.append([np.mean(tq <= t) for t in taus])
        F_small.append([np.mean(ts <= t) for t in taus])
        u = np.minimum(taus * logn, 0.0)
        F_exact.append(np.where(taus < 0, exact_cdf(u, z, ell), 1.0))
        band = lambda t: np.mean((t > -0.375) & (t <= -0.125))
        mid_post.append(band(tp)); mid_prior.append(band(tq)); mid_small.append(band(ts))
    F_post, F_prior, F_small, F_exact = map(np.array, (F_post, F_prior, F_small, F_exact))
    tq975 = stats.t.ppf(0.975, R - 1)
    hw = lambda F: tq975 * F.std(axis=0, ddof=1) / math.sqrt(R)
    rows = []
    for j, t in enumerate(taus):
        rows.append((t, F_post[:, j].mean(), hw(F_post)[j], F_prior[:, j].mean(), hw(F_prior)[j],
                     F_small[:, j].mean(), hw(F_small)[j], F_exact[:, j].mean(), min(max(1 + 2 * t, 0.0), 1.0)))
    sk.write_dat(os.path.join(OUT, "15-sgd-cdf.dat"),
                 ["tau", "post", "posthw", "prior", "priorhw", "small", "smallhw", "exact", "unif"], rows)
    print(f"[cdf] n = 1e12, {R} data sets: max |posterior draws - exact| over tau = "
          f"{np.max(np.abs(F_post.mean(0) - F_exact.mean(0))):.4f}")
    for t in (-0.5, -0.375, -0.25, -0.125, 0.0):
        j = int(np.argmin(np.abs(taus - t)))
        print(f"   tau = {t:+.3f}: posterior {F_post[:, j].mean():.3f} (exact {F_exact[:, j].mean():.3f}, limit {max(1 + 2 * t, 0):.3f}),"
              f" flow from prior {F_prior[:, j].mean():.3f}, flow from small start {F_small[:, j].mean():.3f}")
    print(f"   mass of -3/8 < tau <= -1/8: posterior {np.mean(mid_post):.3f}, flow from prior {np.mean(mid_prior):.4f}, "
          f"flow from small start {np.mean(mid_small):.3f}")


if __name__ == "__main__":
    t_start = time.time()
    part1_check()
    part2_byhand()
    part3_portrait()
    part4_scatter_and_cdf()
    print(f"total {time.time() - t_start:.1f} s")
