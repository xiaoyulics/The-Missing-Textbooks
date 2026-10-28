"""Chapter 3 (The Free Energy by Hand): every number quoted in the text and the exercises that the two
figure scripts do not print, recomputed.

 1. The exact representation against the kit (free_energy_chain, coefficient_free_energy) and against
    the closed form of Computation 3.1, on simulated data sets (n = 1e2, 1e3, 1e4; seeds 0, 1, 2).
 2. The numbers of Computation 3.1 and of "The cusp by hand" (n = 1e4, S_xx = n, Z_n = 0).
 3. The coverage of the Student-t 95% interval for the mean of -Z^2/2 from 200 draws (40,000 sets,
    seed 1), and the probability of 7 or more misses among 65 intervals.
 4. Proposition 3.7: (z + lambda)^m zeta(z) near z = -lambda for three models, against the formula.
 5. The prior phi(a, b) = |a|/2 on [-1, 1]^2: Monte Carlo P(|ab| <= s) against 2s - s^2 (seed 11).
 6. Exercise answers: the one-sided model y = a^2 x; the Gaussian prior; the cubic cusp; a truth
    near the singular point; the unknown noise variance; the Bayes factor between y = abx and y = ax;
    the two-region model; the fits of A log n - B log log n + C to the measured means (reads
    figures/data/03-coefficient-free.dat, written by 03-coefficient-free.py).
 7. The text after Proposition 3.6: the finite-n prediction of E[R_n] - E chi(Z) for the chain
    y = a1 a2 a3 x at n = 1e10, 1e11, 1e12 (S_xx = n, 40-node Gauss-Hermite in Z), against 0.05.
Writes figures/data/03-coefficient-checks.dat (name value).  Run from the volume root:
    python figures/scripts/03-coefficient-checks.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import integrate, special, stats  # noqa: E402

import sltkit as sk  # noqa: E402

GAMMA = float(np.euler_gamma)
OUT = []


def report(name, value, text=""):
    OUT.append((name, value))
    print(f"{name:28s} {value: .6g}   {text}")


def h(z):
    """h(z) = E log|z + U|, U ~ N(0, 1): integrate over |u| <= 40, split at the singular point u = -z."""
    f = lambda u: math.log(abs(z + u)) * math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)
    pts = sorted({-40.0, 40.0, *([-z] if -40.0 < -z < 40.0 else [])})
    return sum(integrate.quad(f, u, v, limit=400)[0] for u, v in zip(pts[:-1], pts[1:]))


def rho_prod(c):
    a = abs(c); return 0.5 * math.log(1 / a) if 0 < a < 1 else 0.0


def rho_line(c):
    return 0.5 if abs(c) < 1 else 0.0


def rho_cusp(c):
    a = abs(c); return 0.5 * (a ** -0.5 - 1) if 0 < a < 1 else 0.0


def rho_chain(L):
    def f(c):
        a = abs(c)
        return math.log(1 / a) ** (L - 1) / (2 * math.factorial(L - 1)) if 0 < a < 1 else 0.0
    return f


def free(rho, S, Z):
    return sk.coefficient_free_energy(-0.5 * Z * Z, S, rho, -1.0, 1.0, Z / math.sqrt(S))


def main() -> None:
    t0 = time.time()
    zs, wz = np.polynomial.hermite_e.hermegauss(40)
    wz = wz / wz.sum()
    # 1. the exact representation against the kit and the closed form
    worst = 0.0
    for n in (100, 1000, 10000):
        for seed in range(3):
            x, y = sk.PRODUCT.sample([0.0, 0.0], n, seed=seed)
            sk.PRODUCT.fit(x, y)
            S = float(sk.PRODUCT.Sxx[0, 0]); Z = float(sk.PRODUCT.Sxy[0]) / math.sqrt(S)
            nS = float(sk.PRODUCT.energy_of_coefficients(np.array([0.0])))
            kit2 = sk.free_energy_chain(sk.PRODUCT, 2) - nS
            kit3 = sk.free_energy_chain(sk.PRODUCT, 3) - nS
            closed = 0.5 * math.log(S) - math.log(0.5 * math.log(S) - h(Z)) - Z * Z / 2 + 0.5 * math.log(2 / math.pi)
            worst = max(worst, abs(free(rho_prod, S, Z) - kit2), abs(closed - kit2),
                        abs(free(rho_chain(3), S, Z) - kit3),
                        abs(free(rho_cusp, S, Z) - sk.coefficient_free_energy(-Z * Z / 2, S, sk.cusp_density, -1, 1, Z / math.sqrt(S))))
    report("max_disagreement_with_kit", worst, "(representation, closed form, kit; n = 1e2..1e4)")
    # 2. Computation 3.1 and the cusp by hand
    n = 1e4
    report("h0", h(0.0), f"= -(gamma + log 2)/2 = {-(GAMMA + math.log(2)) / 2:.4f}")
    F_prod = 0.5 * math.log(n) - math.log(0.5 * math.log(n) - h(0.0)) + 0.5 * math.log(2 / math.pi)
    report("F_product_n1e4_Z0", F_prod, "(step 5 of Computation 3.1)")
    report("F_product_asymptotic_n1e4", 0.5 * math.log(n) - math.log(math.log(n)) + 0.5 * math.log(8 / math.pi))
    F_plane = math.log(n) + math.log(2 / math.pi)
    report("F_regular_plane_n1e4_Z0", F_plane)
    report("marginal_likelihood_ratio", math.exp(F_plane - F_prod))
    I0 = 2 ** 0.25 * math.gamma(0.25)
    report("cusp_integral_Z0", I0, "= 2^(1/4) Gamma(1/4)")
    report("cusp_constant", -math.log(0.5 * I0))
    report("cusp_F_n1e4_asymptotic", 0.25 * math.log(n) - math.log(0.5 * I0))
    report("cusp_F_n1e4_exact", free(rho_cusp, n, 0.0))
    # 3. coverage of the t interval for -Z^2/2
    gen = sk.rng(1)
    X = -0.5 * gen.standard_normal((40000, 200)) ** 2
    m, s = X.mean(1), X.std(1, ddof=1)
    half = stats.t.ppf(0.975, 199) * s / math.sqrt(200)
    cov = float(np.mean(np.abs(m + 0.5) <= half))
    report("t_interval_coverage", cov)
    report("P_7_or_more_misses_of_65", float(1 - stats.binom.cdf(6, 65, 1 - cov)))
    # 4. Proposition 3.7 near the pole
    closed = {"prod": lambda z: 2 ** -z / (2 * z + 1) ** 2, "chain3": lambda z: 2 ** -z / (2 * z + 1) ** 3,
              "cusp": lambda z: 2 ** -z / ((4 * z + 1) * (2 * z + 1))}
    for name, rho, lam, m_, ell in [("prod", rho_prod, 0.5, 2, 0.5), ("chain3", rho_chain(3), 0.5, 3, 0.25),
                                    ("cusp", rho_cusp, 0.25, 1, 0.5)]:
        zq = -lam + 0.05      # the closed form against quadrature, away from the pole
        zeta = 2 * integrate.quad(lambda c: (c * c / 2) ** zq * rho(c), 0, 1, limit=400)[0]
        z = -lam + 1e-6       # (z + lambda)^m zeta(z) near the pole, from the closed form
        report(f"zeta_check_{name}", (z + lam) ** m_ * closed[name](z),
               f"limit 2^(lam-m)(m-1)!(2 ell) = {2 ** (lam - m_) * math.factorial(m_ - 1) * 2 * ell:.5f};"
               f" closed form vs quadrature at z = -lambda + 0.05: rel. diff {abs(zeta / closed[name](zq) - 1):.1e}")
    # 5. a prior vanishing on one arm
    gen = sk.rng(11)
    N = 4_000_000
    a = np.sqrt(gen.random(N)) * np.where(gen.random(N) < 0.5, -1.0, 1.0)
    b = gen.uniform(-1, 1, N)
    for s in (1e-2, 1e-1, 0.5):
        report(f"armprior_P(|ab|<={s})", float(np.mean(np.abs(a * b) <= s)), f"formula 2s - s^2 = {2 * s - s * s:.5f}")
    # 6. exercise answers
    I_half = lambda z: integrate.quad(lambda t: math.exp(-t * t / 2 + z * t) * t ** -0.5, 0, max(z, 0) + 40, limit=400)[0]
    zs80, wz80 = np.polynomial.hermite_e.hermegauss(80)
    wz80 = wz80 / wz80.sum()
    report("onesided_chi0", -math.log(0.5 * I_half(0.0)))
    report("onesided_Echi", float(np.dot(wz80, [-math.log(0.5 * I_half(z)) for z in zs80])))
    report("gaussian_prior_chi0", 0.5 * math.log(2 * math.pi), f"excess over uniform prior log(pi/2) = {math.log(math.pi / 2):.4f}")
    report("gaussian_prior_normalization", 2 * integrate.quad(lambda c: special.k0(c) / math.pi, 0, np.inf, limit=200)[0])
    report("cubic_chi0", -math.log(0.25 * 2 ** (1 / 6) * math.gamma(1 / 6)))
    # a truth near the singular point: the population free energy from the formula of the exercise
    Fpop = lambda nn, c0: 0.5 * math.log(nn) - math.log(0.5 * math.log(nn) - h(math.sqrt(nn) * c0)) + 0.5 * math.log(2 / math.pi)
    def Fpop_direct(nn, c0):
        s = 1 / math.sqrt(nn)
        f = lambda c: math.exp(-0.5 * nn * (c - c0) ** 2) * rho_prod(c)
        lo, hi = max(-1, c0 - 40 * s), min(1, c0 + 40 * s)
        pts = sorted({lo, hi, *([0.0] if lo < 0 < hi else [])})
        return -math.log(sum(integrate.quad(f, u, v, limit=400, epsrel=1e-12)[0] for u, v in zip(pts[:-1], pts[1:])))
    report("near_c0_0.1_Fpop_n1e4", Fpop(1e4, 0.1), f"(direct quadrature {Fpop_direct(1e4, 0.1):.4f})")
    report("near_c0_0.1_minus_c0_0_n1e4", Fpop(1e4, 0.1) - Fpop(1e4, 0.0))
    for nn in (1e2, 1e3, 1e4):
        e = 0.01
        report(f"near_slope_c0_0.1_n{nn:g}", (Fpop(nn * math.exp(e), 0.1) - Fpop(nn * math.exp(-e), 0.1)) / (2 * e))
    report("near_diff_n1e12_direct", Fpop_direct(1e12, 0.1) - Fpop_direct(1e12, 0.0),
           f"(log log n - 1.527 = {math.log(math.log(1e12)) - 1.5272:.3f})")
    # unknown noise variance: sigma ~ U[1/2, 2], S = n, Z = 0, sum eps^2 = n, product model
    n = 1e5
    def inner(c):
        Q = n + c * c * n
        f = lambda s: math.exp(-n * math.log(s) - Q / (2 * s * s) + n / 2) / 1.5
        sh = math.sqrt(Q / n); w = sh / math.sqrt(2 * n)
        return integrate.quad(f, max(0.5, sh - 30 * w), min(2.0, sh + 30 * w), limit=200, epsrel=1e-10)[0]
    sd = 1 / math.sqrt(n)
    tot = sum(integrate.quad(lambda c: inner(c) * rho_prod(c), u, v, limit=200, epsrel=1e-8)[0]
              for u, v in ((-40 * sd, 0.0), (0.0, 40 * sd)))
    F_known = 0.5 * math.log(n) - math.log(0.5 * math.log(n) - h(0.0)) + 0.5 * math.log(2 / math.pi)
    report("unknown_variance_extra", -math.log(tot) - F_known - 0.5 * math.log(n),
           f"(n = 1e5) against -1/2 log pi + log 3/2 = {-0.5 * math.log(math.pi) + math.log(1.5):.4f}")
    # Bayes factor between y = abx and y = ax
    for e in (4, 8):
        nn = 10.0 ** e
        val = float(np.dot(wz80, [math.log(0.5 * math.log(nn) - h(z)) for z in zs80]))
        report(f"E_logBF_n1e{e}", val, "(y = abx over y = ax)")
    lo, hi = 1.0, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = float(np.dot(wz80, [math.log(0.5 * mid - h(z)) for z in zs80]))
        lo, hi = (mid, hi) if val < 3.0 else (lo, mid)
    report("log10_n_logBF_3", lo / math.log(10))
    # two regions: cusp (lambda 1/4, m 1) against the chain of length 5 (lambda 1/2, m 5)
    for e in (4, 12, 14):
        nn = 10.0 ** e
        d = [free(rho_cusp, nn, z) - free(rho_chain(5), nn, z) for z in zs]
        report(f"regions_EFA-FB_n1e{e}", float(np.dot(wz, d)), "(log posterior odds of the chain's region)")
    # fits of A log n - B log log n + C (weighted by the intervals of the measured means; *_a, *_b are A, B)
    path = os.path.join("figures", "data", "03-coefficient-free.dat")
    if os.path.exists(path):
        dat = np.genfromtxt(path, names=True)
        ns = dat["n"]
        for k, m_ in (("prod", 2), ("chain", 3)):
            y, se = dat[k + "F"], dat[k + "E"] / 1.96
            A = np.vstack([np.log(ns), -np.log(np.log(ns)), np.ones_like(ns)]).T
            W = 1 / se ** 2
            cov = np.linalg.inv(A.T @ (A * W[:, None]))
            coef = cov @ (A.T @ (W * y))
            report(f"fit_{k}_a", coef[0], f"se {math.sqrt(cov[0, 0]):.3f}")
            report(f"fit_{k}_b", coef[1], f"se {math.sqrt(cov[1, 1]):.3f} (m - 1 = {m_ - 1})")
            yy = y + (m_ - 1) * np.log(np.log(ns))
            report(f"fit_{k}_slope_m_known", float(np.polyfit(np.log(ns), yy, 1)[0]))
    # 7. the chain of length three: where its mean remainder comes within 0.05 of its limit
    chi3 = lambda z: 2 * math.log(2) - math.log(0.25 * math.sqrt(2 * math.pi)) - 0.5 * z * z
    for e in (10, 11, 12):
        nn = 10.0 ** e
        gap = [free(rho_chain(3), nn, z) - 0.5 * math.log(nn) + 2 * math.log(math.log(nn)) - chi3(z) for z in zs]
        report(f"chain3_mean_gap_n1e{e}", float(np.dot(wz, gap)),
               f"(first-order term -2 gamma/log n = {-2 * GAMMA / math.log(nn):.4f})")
    sk.write_dat(os.path.join("figures", "data", "03-coefficient-checks.dat"), ["name", "value"], OUT)
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
