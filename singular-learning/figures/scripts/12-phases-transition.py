"""Chapter 12 (phases): a transition between separated regions in the double-well model.

Model  y = a b x1 + q(b) x2 + noise,  q(b) = 2 b^4 - b^2,  x ~ N(0, I_2), unit noise,
prior uniform on [-1, 1]^2; truth y = gamma x2 + noise with gamma = 0.05.
K(a, b) = (ab)^2/2 + (q(b) - gamma)^2/2.
Region A = {|b| <= 1/2}: minimum gamma^2/2 on the whole segment b = 0 (the unit is off; learning
coefficient 1/2). Region B = {|b| >= 1/2}: the two true parameters (0, +-b_B), regular (learning
coefficient 1). The ridge |b| = 1/2 separates them: there K >= (gamma + 1/8)^2/2.

What it computes
  * the constants of Proposition prop:phases:doublewell and the population switch point;
  * for R nested data sets (sufficient statistics S = sum x x^T and W = sum x y), the posterior mass
    of region B by quadrature (closed-form a-integral, trapezoid rule in b on 8001 points, checked by
    doubling), at 25 sample sizes from 10^2 to 10^5; its mean with Student-t 95% intervals;
  * the prediction of the proposition, E[sigma(T_n)] with
        T_n = W' S^{-1} W / 2 - (1/2) log n + kappa_A(W_1/sqrt(n)) - kappa_B,
    by Monte Carlo over 10^5 independent draws of (S, W) per sample size; the data-set-by-data-set
    error of T_n against the quadrature log odds;
  * the regular prediction: sigma of the maximized log-likelihood ratio of the two regions
    (Laplace's approximation region by region, equal penalties), on the same data sets;
  * the population posterior mass (no noise: S = n I, W = n c0);
  * the posterior mass near the ridge (0.4 <= |b| <= 0.6) as a check of separation;
  * exact posterior draws for one data set at three sample sizes.

Writes figures/data/12-phases-transition.dat and 12-phases-draws-n{1,2,3}.dat.
Run from the volume root: python figures/scripts/12-phases-transition.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy import special, stats, optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

OUT = os.path.join("figures", "data")
GAMMA = 0.05
C0 = np.array([0.0, GAMMA])
R = 500
SEED = 1210
NB = 8001


def q(b):
    return 2 * b ** 4 - b ** 2


def qp(b):
    return 8 * b ** 3 - 2 * b


def constants(gamma=GAMMA):
    bB = math.sqrt((1 + math.sqrt(1 + 8 * gamma)) / 4)
    detJ = bB * qp(bB)
    kB = math.log(detJ / math.pi)
    return bB, detJ, kB


_A = np.linspace(-1, 1, 4001)


def kappa_A(z, gamma=GAMMA):
    """kappa_A(z) = -log( (1/4) int_{-1}^{1} sqrt(pi/c(a)) exp(a^2 z^2/(4 c(a))) da ), c = gamma + a^2/2."""
    z = np.atleast_1d(np.asarray(z, dtype=float))[:, None]
    c = gamma + _A[None, :] ** 2 / 2
    f = 0.25 * np.sqrt(np.pi / c) * np.exp(_A[None, :] ** 2 * z ** 2 / (4 * c))
    da = _A[1] - _A[0]
    return -np.log(da * (f.sum(axis=1) - 0.5 * (f[:, 0] + f[:, -1])))


_ZT = np.linspace(0.0, 9.0, 3601)
_KT = kappa_A(_ZT)


def kappa_A_fast(z):
    return np.interp(np.abs(z), _ZT, _KT)


def log_a_integral(A2, A1):
    """log int_{-1}^{1} exp(-A2 a^2 - A1 a) da, A2 >= 0 (arrays); series for tiny A2."""
    small = A2 < 1e-9
    A2s = np.where(small, 1.0, A2)
    mu = -A1 / (2 * A2s)
    sig = 1 / np.sqrt(2 * A2s)
    hi, lo = (1 - mu) / sig, (-1 - mu) / sig
    l_hi, l_lo = special.log_ndtr(hi), special.log_ndtr(lo)
    d1 = l_hi + np.log1p(-np.exp(np.minimum(l_lo - l_hi, 0.0)))
    m_lo, m_hi = special.log_ndtr(-lo), special.log_ndtr(-hi)
    d2 = m_lo + np.log1p(-np.exp(np.minimum(m_hi - m_lo, 0.0)))
    ld = np.where(lo > 0, d2, d1)
    val = 0.5 * np.log(np.pi / A2s) + A1 ** 2 / (4 * A2s) + ld
    return np.where(small, math.log(2) - A2 / 3 + A1 ** 2 / 6, val)


def b_logdensity(S, W, b):
    """log of the unnormalized marginal posterior density of b, minus the constant n L_n(0):
    log( (1/4) int_{-1}^{1} exp(-(n L_n(a, b) - n L_n(0, 0))) da ), for statistics S (R,2,2), W (R,2).
    n L_n(w) - n L_n(0) = g' S g / 2 - g' W with g = (ab, q(b))."""
    qb = q(b)[None, :]
    bb = b[None, :]
    S11, S12, S22 = S[:, 0, 0][:, None], S[:, 0, 1][:, None], S[:, 1, 1][:, None]
    W1, W2 = W[:, 0][:, None], W[:, 1][:, None]
    A2 = 0.5 * S11 * bb * bb
    A1 = bb * (S12 * qb - W1)
    A0 = 0.5 * S22 * qb * qb - W2 * qb
    return math.log(0.25) + log_a_integral(A2, A1) - A0


def trap_logsum(e, b, intervals):
    """log of the trapezoid rule for int exp(e) db over a union of intervals [lo, hi] whose
    endpoints are grid points (each interval gets its own end half-weights)."""
    db = b[1] - b[0]
    m = e.max(axis=1, keepdims=True)
    tot = 0.0
    for lo, hi in intervals:
        i0, i1 = int(round((lo - b[0]) / db)), int(round((hi - b[0]) / db))
        w = np.exp(e[:, i0:i1 + 1] - m)
        tot = tot + db * (w.sum(axis=1) - 0.5 * (w[:, 0] + w[:, -1]))
    with np.errstate(divide="ignore"):         # the ridge band underflows to 0 at the largest n
        return m[:, 0] + np.log(tot)


def region_masses(S, W, nb=NB):
    """log integrals over region A (|b| <= 1/2) and region B (|b| >= 1/2), and the log posterior
    mass of the ridge band 0.4 <= |b| <= 0.6. The grid must contain +-0.4, +-0.5, +-0.6."""
    b = np.linspace(-1, 1, nb)
    e = b_logdensity(S, W, b)
    lA = trap_logsum(e, b, [(-0.5, 0.5)])
    lB = trap_logsum(e, b, [(-1.0, -0.5), (0.5, 1.0)])
    lR = trap_logsum(e, b, [(-0.6, -0.4), (0.4, 0.6)])
    lAll = trap_logsum(e, b, [(-1.0, 1.0)])
    return lA, lB, lR - lAll


def region_maxlik(S, W, nb=4001):
    """Maximized log-likelihood ratio over each region, -min (n L_n(w) - n L_n(0)): a from the
    quadratic in closed form (clipped to [-1, 1]), b on a grid."""
    out = []
    for lo_b, hi_b in ((0.0, 0.5), (0.5, 1.0)):
        b = np.linspace(lo_b, hi_b, nb)
        best = None
        for sgn in (1.0, -1.0):                     # b and -b (a -> -a gives the same function)
            bb = sgn * b[None, :]
            qb = q(b)[None, :]
            S11, S12, S22 = S[:, 0, 0][:, None], S[:, 0, 1][:, None], S[:, 1, 1][:, None]
            W1, W2 = W[:, 0][:, None], W[:, 1][:, None]
            A2 = 0.5 * S11 * bb * bb
            A1 = bb * (S12 * qb - W1)
            A0 = 0.5 * S22 * qb * qb - W2 * qb
            a = np.clip(np.where(A2 > 0, -A1 / (2 * np.where(A2 > 0, A2, 1.0)), 0.0), -1, 1)
            E = A2 * a * a + A1 * a + A0
            cur = -E.min(axis=1)
            best = cur if best is None else np.maximum(best, cur)
        out.append(best)
    return out[0], out[1]


def wishart2(dof, gen, size):
    c1 = np.sqrt(gen.chisquare(dof, size))
    c2 = np.sqrt(gen.chisquare(dof - 1, size))
    z = gen.standard_normal(size)
    A = np.zeros((size, 2, 2))
    A[:, 0, 0], A[:, 1, 0], A[:, 1, 1] = c1, z, c2
    return A @ np.transpose(A, (0, 2, 1))


def draw_stats(n, gen, size):
    S = wishart2(n, gen, size)
    U = np.einsum("rij,rj->ri", np.linalg.cholesky(S), gen.standard_normal((size, 2)))
    return S, S @ C0 + U


def T_stat(S, W, n):
    lr = 0.5 * np.einsum("ri,ri->r", W, np.linalg.solve(S, W[:, :, None])[:, :, 0])
    return lr - 0.5 * math.log(n) + kappa_A_fast(W[:, 0] / math.sqrt(n)) - constants()[2]


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def posterior_draws(S, W, m, gen, nb=40001):
    """m exact draws (up to the b-grid) from the posterior given statistics S (2,2), W (2,)."""
    b = np.linspace(-1, 1, nb)
    e = b_logdensity(S[None], W[None], b)[0]
    p = np.exp(e - e.max())
    cdf = np.cumsum(p)
    cdf /= cdf[-1]
    idx = np.searchsorted(cdf, gen.random(m))
    db = b[1] - b[0]
    bs = np.clip(b[idx] + (gen.random(m) - 0.5) * db, -1, 1)
    A2 = 0.5 * S[0, 0] * bs * bs
    A1 = bs * (S[0, 1] * q(bs) - W[0])
    a = np.empty(m)
    tiny = A2 < 1e-9
    a[tiny] = gen.uniform(-1, 1, tiny.sum())
    mu = -A1[~tiny] / (2 * A2[~tiny])
    sd = 1 / np.sqrt(2 * A2[~tiny])
    a[~tiny] = stats.truncnorm.rvs((-1 - mu) / sd, (1 - mu) / sd, loc=mu, scale=sd, random_state=gen)
    return np.stack([a, bs], axis=1)


def main():
    t0 = time.time()
    bB, detJ, kB = constants()
    Delta = GAMMA ** 2 / 2
    kA0 = float(kappa_A(0.0)[0])
    kA0_closed = -math.log(math.sqrt(2 * math.pi) / 2 * math.asinh(1 / math.sqrt(2 * GAMMA)))
    ridge = 0.5 * (GAMMA + 1 / 8) ** 2
    nbar = optimize.brentq(lambda n: n * Delta - 0.5 * math.log(n) + kA0 - kB, 10, 1e8)
    print(f"gamma={GAMMA}: Delta={Delta}, b_B={bB:.5f}, q'(b_B)={qp(bB):.5f}, |det Dg|={detJ:.5f}")
    print(f"kappa_B=log(|det Dg|/pi)={kB:.5f}; kappa_A(0)={kA0:.5f} (closed form {kA0_closed:.5f})")
    print(f"ridge: K(0,1/2)={ridge:.5f}, barrier above Delta eta={ridge - Delta:.5f}")
    print(f"population switch: n Delta - (1/2) log n + kappa_A(0) - kappa_B = 0 at n = {nbar:.1f}")
    print(f"  leading terms only (n Delta = (1/2) log n): n = "
          f"{optimize.brentq(lambda n: n * Delta - 0.5 * math.log(n), 10, 1e8):.1f}")
    # transversal curvature, local values
    print(f"region A: transversal Hessian a^2 + 2 gamma in [{2*GAMMA}, {1+2*GAMMA}]")

    ns = np.unique(np.round(np.logspace(2, 5, 25)).astype(int))
    gen = sk.rng(SEED)
    S = np.zeros((R, 2, 2))
    U = np.zeros((R, 2))
    prev = 0
    rows = []
    errs_all = []
    frac_rows = []
    for n in ns:
        dS = wishart2(int(n - prev), gen, R)
        U = U + np.einsum("rij,rj->ri", np.linalg.cholesky(dS), gen.standard_normal((R, 2)))
        S = S + dS
        prev = n
        W = S @ C0 + U
        lA, lB, lRidge = region_masses(S, W)
        pB = sigmoid(lB - lA)
        T = T_stat(S, W, n)
        err = (lB - lA) - T
        errs_all.append((n, float(np.mean(err)), float(np.std(err)), float(np.max(np.abs(err)))))
        mlA, mlB = region_maxlik(S, W)
        p_reg = sigmoid(mlB - mlA)
        mu, lo, hi = sk.mean_ci(pB)
        # prediction: independent Monte Carlo of the statistics
        g2 = sk.rng(SEED + 7 + int(n))
        S2, W2 = draw_stats(int(n), g2, 100000)
        T2 = T_stat(S2, W2, n)
        pred = float(np.mean(sigmoid(T2)))
        frac_rows.append((n, float(np.mean(pB >= 0.5)), float(np.mean(T2 > 0))))
        # population posterior mass (no noise)
        Sp = n * np.eye(2)[None]
        Wp = (n * C0)[None]
        pA_, pB_, _ = region_masses(Sp, Wp)
        pop = float(sigmoid(pB_ - pA_)[0])
        popf = float(sigmoid(n * Delta - 0.5 * math.log(n) + kA0 - kB))
        ridge_mass = float(np.mean(np.exp(lRidge)))
        rows.append((n, mu, hi - mu, mu - lo, pred, float(np.mean(p_reg)), pop, popf, ridge_mass))
        print(f"n={n:6d} P(B): measured {mu:.3f} [{lo:.3f},{hi:.3f}]  predicted {pred:.3f}  regular {np.mean(p_reg):.3f}"
              f"  population {pop:.3f} (formula {popf:.3f})  ridge mass {ridge_mass:.1e}"
              f"  |logodds-T|: mean {np.mean(err):+.3f} sd {np.std(err):.3f}")
    sk.write_dat(os.path.join(OUT, "12-phases-transition.dat"),
                 ["n", "pB", "hi", "lo", "pred", "reg", "pop", "popf", "ridge"], rows)
    print("fraction of data sets with P(B) >= 1/2 (measured) and P(T_n > 0) (predicted):")
    for n, fm, fp in frac_rows:
        if n in (316, 1000, 1778, 3162, 5623, 10000):
            print(f"   n={n:6d}: {fm:.3f} vs {fp:.3f}")
    g3 = sk.rng(SEED + 99)
    med = optimize.brentq(lambda n: float(np.mean(T_stat(*draw_stats(int(n), sk.rng(SEED + 99), 100000), n) > 0)) - 0.5,
                          1500, 6000, xtol=5)
    print(f"predicted median switch (P(T_n > 0) = 1/2): n = {med:.0f}")
    errs_all = np.array(errs_all)
    print("log-odds error (quadrature minus T_n): n, mean, sd, max|.|")
    for r in errs_all[::4]:
        print(f"   {int(r[0]):6d} {r[1]:+.4f} {r[2]:.4f} {r[3]:.4f}")

    # grid doubling check
    g4 = sk.rng(SEED + 5)
    worst = 0.0
    for n in (100, 1000, 10000, 100000):
        S2, W2 = draw_stats(n, g4, 40)
        a1, b1, _ = region_masses(S2, W2, nb=NB)
        a2, b2, _ = region_masses(S2, W2, nb=2 * NB - 1)
        worst = max(worst, float(np.max(np.abs(a1 - a2))), float(np.max(np.abs(b1 - b2))))
    print(f"grid check: max change of log region integrals when doubling the b-grid: {worst:.2e}")

    # posterior draws for one data set at three sample sizes
    g5 = sk.rng(SEED + 11)
    S1 = np.zeros((2, 2))
    U1 = np.zeros(2)
    prev = 0
    for k, n in enumerate((300, 3000, 30000)):
        dS = wishart2(n - prev, g5, 1)[0]
        U1 = U1 + np.linalg.cholesky(dS) @ g5.standard_normal(2)
        S1 = S1 + dS
        prev = n
        W1 = S1 @ C0 + U1
        lA, lB, _ = region_masses(S1[None], W1[None])
        d = posterior_draws(S1, W1, 2000, g5)
        sk.write_dat(os.path.join(OUT, f"12-phases-draws-n{k + 1}.dat"), ["a", "b"], d)
        print(f"draws n={n}: P(B)={float(sigmoid(lB - lA)[0]):.4f}, fraction of draws with |b|>=1/2: "
              f"{np.mean(np.abs(d[:, 1]) >= 0.5):.3f}, T_n={float(T_stat(S1[None], W1[None], n)[0]):.3f}")

    # Chen et al. (2023), App. C: n Delta L + Delta lambda log n + Delta c = 0 for the 5-gon -> 6-gon, with
    # L = E||x - f||^2 (their Table 1).  Their eq. (3), p(x|w) ~ exp(-||x - f||^2 / 2), has energy n L / 2:
    # the second line halves the loss term.
    dL, dlam, dc = 0.04819 - 0.06874, 8.5 - 7.0, 6.37767 - 3.62417
    for lab, s in (("energy n L (their App. C)", 1.0), ("energy n L/2 (their eq. (3))", 0.5)):
        f = lambda n: s * n * dL + dlam * math.log(n) + dc
        f0 = lambda n: s * n * dL + dlam * math.log(n)
        print(f"TMS 5->6, {lab}: n_cr with constants {optimize.brentq(f, 50, 1e5):.1f}; "
              f"without constants {optimize.brentq(f0, 50, 1e5):.1f}")
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
