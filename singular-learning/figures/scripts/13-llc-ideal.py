"""Chapter 13 (llc): the ideal local-learning-coefficient estimator of the ReLU neuron, computed exactly.

Model: y = a relu(b x) + N(0, 1), x ~ N(0, 1), truth a relu(b x) = 0 (pure noise), so that
K(a, b) = a^2 b^2 / 4 and, at both points used here, lambda(w*) = 1/2:
  * w* = (0, 0) with the book's prior box [-1, 1]^2 as the localizer (multiplicity m = 2);
  * w* = (1, 0) with the box [1/2, 3/2] x [-1/2, 1/2], which does not contain the origin (m = 1).
The "ideal" estimator is Lau et al.'s lambda_hat(w*) = n beta (E[L_n(w)] - L_n(w*)), beta = 1/log n, with
the expectation under the exact localized tempered posterior (prop. to exp(-n beta L_n) on the box), no sampler.

How it is computed. On the half-plane b > 0 the mean is (ab) x 1{x > 0}; on b < 0 it is (ab) x 1{x < 0}.
So on each half the energy depends on w only through c = ab (Chapter 3's one-coefficient reduction):
  n [L_n(a, b) - L_n(0, 0)] = S_xx^s c^2 / 2 - S_xy^s c,   s = sign(b),
with S_xx^+ = sum_{x_i > 0} x_i^2, S_xy^+ = sum_{x_i > 0} x_i y_i (and - for x_i < 0). Under the uniform
localizer, c has an explicit density on each half (rho below), and every posterior expectation is a
one-dimensional integral, done here on a graded grid (checked against adaptive quadrature).
The sufficient statistics are drawn exactly: n_+ ~ Bin(n, 1/2), S_xx^+ ~ chi^2(n_+), and given S_xx^+,
S_xy^+ ~ N(0, S_xx^+) (the labels are independent N(0, 1) noise); likewise for the minus side.

A second computation reproduces, without a sampler, the normal-crossing test of the first arXiv version of
Lau et al. (arXiv:2308.12108v1, Section 10.1 and Table 2, row w1 w2 with the uniform prior): y = w1 w2 x
+ N(0, 1/4), x ~ N(0, 1), n = 1000, beta = 1/log n, the Gaussian localizer exp(-|w|^2/2) (gamma = 1) at
w* = 0, and lambda_hat = n beta (E[L_n] - L_n(0)). The energy depends on w only through c = w1 w2, whose
density under N(0, I_2) is K_0(|c|)/pi, so again every posterior average is a one-dimensional integral.

Outputs: figures/data/13-llc-ideal.dat (per n: mean over R data sets with Student-t 95% intervals, the
population skeleton, and the prediction skeleton - 1/(2 log n)), figures/data/13-llc-ideal-curve.dat
(the two predictions on a fine grid of n), figures/data/13-llc-ideal-lau.dat (the normal-crossing test:
mean over R_LAU data sets with its interval, the standard deviation over data sets, and the population
skeleton). Run from the volume root: python figures/scripts/13-llc-ideal.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy import integrate, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

EULER = 0.5772156649015329
R = 200                                   # data sets per sample size
NS = [int(round(10 ** e)) for e in np.arange(2.0, 6.01, 0.5)]
SEED = 1301
R_LAU, N_LAU, SIG2_LAU, SEED_LAU = 2000, 1000, 0.25, 1304   # the normal-crossing test of Lau et al. v1


def rho_origin(c):
    """Density of c = ab on one half (b > 0, or b < 0) for (a, b) uniform on [-1, 1] x (0, 1]:
    (1/2) log(1/|c|) on [-1, 1]."""
    c = np.abs(c)
    with np.errstate(divide="ignore"):
        return np.where((c > 0) & (c <= 1), -0.5 * np.log(c), 0.0)


def rho_off(c):
    """Density of c = ab on the half b > 0 for a ~ U[1/2, 3/2], b ~ U(0, 1/2] (mirror for b < 0):
    2 log 3 for |c| <= 1/4, 2 log(3/(4|c|)) for 1/4 < |c| <= 3/4."""
    c = np.abs(c)
    out = np.where(c <= 0.25, 2 * math.log(3), 2 * np.log(3 / (4 * np.maximum(c, 1e-300))))
    return np.where(c <= 0.75, out, 0.0)


def make_grid(hi):
    """Graded grid on [0, hi]: geometric near 0 (log singularity of rho_origin), uniform elsewhere."""
    g = np.concatenate([[0.0], np.geomspace(1e-14, hi, 6000), np.linspace(0, hi, 20001)])
    return np.unique(g)


GRID_ORIGIN = make_grid(1.0)
GRID_OFF = make_grid(0.75)


def half_moments(Sxx, Sxy, beta, rho, grid, sign):
    """log of int exp(-beta(Sxx c^2/2 - Sxy c)) rho(c) dc over the half's c-range, and the posterior mean of
    the energy beta(Sxx c^2/2 - Sxy c). On the half b > 0 (sign = +1) c ranges over [-hi, hi] for the
    origin box (a takes both signs) and over (0, hi] for the off-origin box; sign = -1 mirrors."""
    c = np.concatenate([-grid[::-1], grid[1:]]) if rho is rho_origin else sign * grid
    if rho is not rho_origin and sign < 0:
        c = c[::-1]
    E = beta * (Sxx * c * c / 2 - Sxy * c)
    lw = -E + np.log(np.maximum(rho(c), 1e-300))
    lw = np.where(rho(c) > 0, lw, -np.inf)
    mx = np.max(lw)
    w = np.exp(lw - mx)
    Z = np.trapezoid(w, c)
    return mx + math.log(Z), float(np.trapezoid(w * E, c) / Z)


def ideal(stats, beta, which):
    rho, grid = (rho_origin, GRID_ORIGIN) if which == "origin" else (rho_off, GRID_OFF)
    Spxx, Spxy, Smxx, Smxy = stats
    lz1, e1 = half_moments(Spxx, Spxy, beta, rho, grid, +1)
    lz2, e2 = half_moments(Smxx, Smxy, beta, rho, grid, -1)
    m = max(lz1, lz2)
    w1, w2 = math.exp(lz1 - m), math.exp(lz2 - m)
    return (w1 * e1 + w2 * e2) / (w1 + w2)


def draw_stats(n, gen):
    npos = gen.binomial(n, 0.5)
    Spxx, Smxx = gen.chisquare(npos), gen.chisquare(n - npos)
    return Spxx, gen.standard_normal() * math.sqrt(Spxx), Smxx, gen.standard_normal() * math.sqrt(Smxx)


def skeleton(n, which):
    """The estimator with the population loss in place of L_n: S_xx^+- = n/2, S_xy^+- = 0."""
    return ideal((n / 2, 0.0, n / 2, 0.0), 1 / math.log(n), which)


GRID_LAU = np.concatenate([-make_grid(1.0)[::-1], make_grid(1.0)[1:]])
LOGRHO_LAU = np.log(np.maximum(special.k0(np.maximum(np.abs(GRID_LAU), 1e-300)) / math.pi, 1e-300))


def lau_estimate(Sxx, Sxy, beta, sig2=SIG2_LAU):
    """Exact lambda_hat = E[beta (Sxx c^2 - 2 Sxy c) / (2 sig2)] under the density prop. to
    K_0(|c|) exp(-beta (Sxx c^2 - 2 Sxy c) / (2 sig2)) of c = w1 w2 (Gaussian localizer gamma = 1 at 0).
    The grid [-1, 1] suffices: beyond |c| = 1/2 the weight is below exp(-beta Sxx / 8 sig2)."""
    c = GRID_LAU
    E = beta * (Sxx * c * c - 2 * Sxy * c) / (2 * sig2)
    lw = LOGRHO_LAU - E
    w = np.exp(lw - lw.max())
    return float(np.trapezoid(w * E, c) / np.trapezoid(w, c))


def lau_check(gen):
    """The normal-crossing test: grid against adaptive quadrature, then the mean over R_LAU data sets."""
    beta = 1 / math.log(N_LAU)
    Sxx = gen.chisquare(N_LAU)
    Sxy = gen.standard_normal() * math.sqrt(SIG2_LAU * Sxx)
    E = lambda c: beta * (Sxx * c * c - 2 * Sxy * c) / (2 * SIG2_LAU)
    pts = sorted({-4.0, 0.0, Sxy / Sxx, 4.0})
    Z = sum(integrate.quad(lambda c: special.k0(abs(c)) * math.exp(-E(c)), u, v, limit=500, epsrel=1e-11)[0]
            for u, v in zip(pts[:-1], pts[1:]))
    N = sum(integrate.quad(lambda c: E(c) * special.k0(abs(c)) * math.exp(-E(c)), u, v, limit=500,
                           epsrel=1e-11)[0] for u, v in zip(pts[:-1], pts[1:]))
    err = abs(N / Z - lau_estimate(Sxx, Sxy, beta))
    vals = []
    for _ in range(R_LAU):
        Sxx = gen.chisquare(N_LAU)
        vals.append(lau_estimate(Sxx, gen.standard_normal() * math.sqrt(SIG2_LAU * Sxx), beta))
    mu, lo, hi = sk.mean_ci(vals)
    skel = lau_estimate(float(N_LAU), 0.0, beta)
    return err, mu, lo, hi, float(np.std(vals, ddof=1)), skel


def check_quadrature(gen):
    """Compare the graded-grid integrals with adaptive quadrature on a few data sets."""
    worst = 0.0
    for n in (100, 10_000, 1_000_000):
        beta = 1 / math.log(n)
        st = draw_stats(n, gen)
        for which in ("origin", "off"):
            rho = rho_origin if which == "origin" else rho_off
            hi = 1.0 if which == "origin" else 0.75
            parts = []
            for Sxx, Sxy, sgn in ((st[0], st[1], 1), (st[2], st[3], -1)):
                lo_c, hi_c = (-hi, hi) if which == "origin" else ((0.0, hi) if sgn > 0 else (-hi, 0.0))
                E = lambda c: beta * (Sxx * c * c / 2 - Sxy * c)
                ch = Sxy / Sxx
                pts = sorted({p for p in (lo_c, hi_c, 0.0, ch, 0.25, -0.25) if lo_c <= p <= hi_c})
                Z = sum(integrate.quad(lambda c: math.exp(-E(c)) * float(rho(np.array(c))), u, v,
                                       limit=500, epsrel=1e-12, epsabs=0)[0] for u, v in zip(pts[:-1], pts[1:]))
                N = sum(integrate.quad(lambda c: E(c) * math.exp(-E(c)) * float(rho(np.array(c))), u, v,
                                       limit=500, epsrel=1e-12, epsabs=0)[0] for u, v in zip(pts[:-1], pts[1:]))
                parts.append((Z, N))
            exact = (parts[0][1] + parts[1][1]) / (parts[0][0] + parts[1][0])
            worst = max(worst, abs(exact - ideal(st, beta, which)))
    return worst


def main():
    t0 = time.perf_counter()
    gen = sk.rng(SEED)
    print("quadrature check: max |grid - adaptive| =", f"{check_quadrature(gen):.2e}")
    # the population formula of the chapter: t E_t[K] = 1/2 - 1/(log t + gamma_E) for the box [-1, 1]^2
    for n in (100, 10_000, 1_000_000):
        t = n / math.log(n)
        print(f"skeleton at the origin, n={n}: grid {skeleton(n, 'origin'):.6f}   "
              f"formula 1/2 - 1/(log t + gamma_E) = {0.5 - 1 / (math.log(t) + EULER):.6f}")
    rows = []
    for n in NS:
        beta = 1 / math.log(n)
        vals = {"origin": [], "off": []}
        for _ in range(R):
            st = draw_stats(n, gen)
            for which in vals:
                vals[which].append(ideal(st, beta, which))
        row = [n, n * beta]
        for which in ("origin", "off"):
            mu, lo, hi = sk.mean_ci(vals[which])
            sk_ = skeleton(n, which)
            row += [mu, mu - lo, hi - mu, sk_, sk_ - beta / 2]
            print(f"n={n:>8}  {which:>6}: mean {mu:.4f} [{lo:.4f}, {hi:.4f}]  sd {np.std(vals[which], ddof=1):.4f}"
                  f"  skeleton {sk_:.4f}  skeleton - beta/2 {sk_ - beta / 2:.4f}")
        rows.append(row)
    sk.write_dat("figures/data/13-llc-ideal.dat",
                 ["n", "t", "o", "oep", "oem", "oskel", "opred", "f", "fep", "fem", "fskel", "fpred"], rows)
    curve = []
    for n in np.geomspace(80, 1.3e6, 60):
        beta = 1 / math.log(n)
        curve.append([n, skeleton(n, "origin") - beta / 2, skeleton(n, "off") - beta / 2,
                      skeleton(n, "origin"), skeleton(n, "off")])
    sk.write_dat("figures/data/13-llc-ideal-curve.dat", ["n", "opred", "fpred", "oskel", "fskel"], curve)
    err, mu, lo, hi, sd, skel = lau_check(sk.rng(SEED_LAU))
    print(f"normal-crossing test (Lau et al. v1): grid vs adaptive {err:.1e}; n={N_LAU}, t={N_LAU / math.log(N_LAU):.1f}:"
          f" mean {mu:.4f} [{lo:.4f}, {hi:.4f}], sd over data sets {sd:.4f}, population skeleton {skel:.4f}"
          f"  (R = {R_LAU}, seed {SEED_LAU})")
    sk.write_dat("figures/data/13-llc-ideal-lau.dat", ["n", "t", "mean", "lo", "hi", "sd", "skel"],
                 [[N_LAU, N_LAU / math.log(N_LAU), mu, lo, hi, sd, skel]])
    print(f"R = {R} data sets per n, seed {SEED}; total time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
