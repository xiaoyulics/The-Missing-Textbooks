"""07-catalogue-validate.py -- validating the two-temperature estimator with Metropolis (Chapter 7).

Run from the volume root:  python figures/scripts/07-catalogue-validate.py   (about 100 seconds)

The estimator (Watanabe 2013, Corollary 3) is
    lambda_hat = (E^{b1}[n L_n] - E^{b2}[n L_n]) / (1/b1 - 1/b2),   b1 = 1/log n,  b2 = 1.5/log n,
computed as in 07-catalogue-rrr.py: random-walk Metropolis at b1 (sk.metropolis, chains started from the
prior; models 1-2: 3000 adaptation steps discarded, 9000 kept, every 10th used; models 3-4: 8000, 16000
and every 40th, the lengths of 07-catalogue-rrr.py) and the b2-mean by reweighting with
exp(-(b2 - b1) n L_n) (Watanabe 2013, eq. (20)). We check it where the tempered means are known exactly:
  1. the regular plane y = a x1 + b x2 (lambda = 1, m = 1): against midpoint quadrature on a grid
     (sk.tempered_mean_grid), on the same data sets;
  2. the product model y = abx at the truth 0 (lambda = 1/2, m = 2): against one-dimensional quadrature
     in c = ab, whose prior density is (1/2) log(1/|c|);
  3. rank-one regression y = b a^T x, a, b in [-1, 1]^3, at the truth 0 and without noise (energy
     n K = (n/2)|a|^2|b|^2): against the exact tempered mean E^b[nK] = (3/2 - 1/(log(nb) + kappa))/b,
     kappa = 0.0397 (exact up to terms of order exp(-nb/2); Chapter 7), and against a Gibbs sampler with
     exact conditionals (1024 chains, 2000 sweeps, the first 400 discarded);
  4. the same model at the rank-one truth C0 = (1/2) e1 e1^T without noise (lambda = 5/2): against the
     population value of the estimator by two-dimensional quadrature (b integrated in closed form, a
     reduced to (a1, a2^2 + a3^2)); and, to locate the source of its excess over 5/2, the same quadrature
     for the uniform prior on the product of the unit balls of R^3 (a in polar coordinates).
Uniform priors on [-1, 1]^d, x ~ N(0, I), unit noise; n = 10^3, 10^4, 10^5; models 1-2: 20 data sets
per n, 16 chains per data set; models 3-4: 1024 chains per n, in 64 groups of 16 for the intervals.
Writes figures/data/07-catalogue-plane.dat, 07-catalogue-product.dat, 07-catalogue-rankone-pop.dat.
"""
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy import integrate, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

NS = (1000, 10_000, 100_000)
R, CHAINS = 20, 16
ADAPT, KEEP, THIN = 3000, 9000, 10
CHAINS_POP = 1024                                  # models 3-4: 64 groups of 16 chains,
ADAPT_POP, KEEP_POP, THIN_POP = 8000, 16000, 40    # with the chain lengths of 07-catalogue-rrr.py
KAPPA = 8 * 0.3021193866750662 / math.pi - math.log(2) - special.digamma(1.5)   # 07-catalogue-formulas.py
KAPPA_PRODUCT = np.euler_gamma + math.log(2)
SIGMA = 0.5                                        # the rank-one truth SIGMA e1 e1^T


def betas(n):
    return 1.0 / math.log(n), 1.5 / math.log(n)


def lam_from_means(E1, E2, n):
    b1, b2 = betas(n)
    return (E1 - E2) / (1.0 / b1 - 1.0 / b2)


def lam_from_draws(e, n):
    """Two-temperature estimate from energies e of draws at b1, the b2-mean by reweighting."""
    b1, b2 = betas(n)
    w = np.exp(-(b2 - b1) * (e - e.min()))
    return lam_from_means(e.mean(), float(w @ e) / w.sum(), n)


def lam_population(lam, m, n, kappa):
    b1, b2 = betas(n)
    return lam_from_means((lam - (m - 1) / (math.log(n * b1) + kappa)) / b1,
                          (lam - (m - 1) / (math.log(n * b2) + kappa)) / b2, n)


def metropolis_energies(energy, d, beta_of_chain, seed, adapt=ADAPT, keep=KEEP, thin=THIN):
    bvec = np.asarray(beta_of_chain, dtype=float)

    def logp(w):
        inside = np.all(np.abs(w) <= 1.0, axis=1)
        return np.where(inside, -bvec * energy(w), -np.inf)

    x0 = sk.rng(seed).uniform(-1, 1, (len(bvec), d))
    draws, acc = sk.metropolis(logp, x0, 0.05, keep, seed=seed + 1, adapt=adapt, thin=thin)
    return np.array([energy(dr) for dr in draws]), acc          # (kept, chains)


# ---------------------------------------------------------------- exact population values, rank-one truth
# g(t) = t E_t[K] under the population posterior prop. to exp(-t K) phi, K = |b a^T - SIGMA e1 e1^T|^2 / 2,
# a, b in R^3; the estimator's population value at n is 3 g(t1) - 2 g(1.5 t1) with t1 = n / log n.
def _gauss_interval(s, mu):
    """int_{-1}^{1} exp(-s (b - mu)^2 / 2) db and int (b - mu)^2 / 2 exp(-s (b - mu)^2 / 2) db."""
    rs, hi, lo = math.sqrt(s), 1 - mu, -1 - mu
    I = math.sqrt(2 * math.pi / s) * (special.ndtr(rs * hi) - special.ndtr(rs * lo))
    D = 0.5 * (I - (hi * math.exp(-s * hi * hi / 2) - lo * math.exp(-s * lo * lo / 2))) / s
    return I, D


def g_box(t, sigma=SIGMA, tol=1e-7):
    """Uniform prior on [-1,1]^6. Given a, K = |a|^2 |b - mu|^2 / 2 + sigma^2 u / (2|a|^2) with
    u = a2^2 + a3^2 and mu = sigma a1 e1 / |a|^2, so the b-integrals are one-dimensional Gaussian integrals
    over [-1, 1]; a enters through (a1, u), u with the density fu of a2^2 + a3^2 for (a2, a3) uniform on
    [-1, 1]^2."""
    def fu(u):
        return math.pi / 4 if u <= 1 else (math.pi - 4 * math.acos(1 / math.sqrt(u))) / 4

    def h(a1, u, which):
        r2 = a1 * a1 + u
        s, c0 = t * r2, 0.5 * sigma * sigma * u / r2
        I1, D1 = _gauss_interval(s, sigma * a1 / r2)
        I0, D0 = _gauss_interval(s, 0.0)
        J = math.exp(-t * c0) * I1 * I0 * I0
        if J <= 0.0:
            return 0.0
        return fu(u) * (J if which == 0 else J * (t * c0 + s * (D1 / I1 + 2 * D0 / I0)))

    def inner(a1, which):
        sc = a1 * a1 / (t * sigma * sigma)
        edges = sorted({0.0, 2.0, *[p for p in (sc, 10 * sc, 100 * sc, 1.0) if 0 < p < 2]})
        return sum(integrate.quad(lambda u: h(a1, u, which), lo, hi, epsabs=0, epsrel=tol, limit=100)[0]
                   for lo, hi in zip(edges[:-1], edges[1:]))
    edges = sorted({0.0, sigma, 1.0})
    Z, W = (sum(integrate.quad(lambda a: inner(a, k), lo, hi, epsabs=0, epsrel=tol, limit=100)[0]
                for lo, hi in zip(edges[:-1], edges[1:])) for k in (0, 1))
    return W / Z


def _ball_B(s, m):
    """E_b exp(-(s/2)|b - m e1|^2) for b uniform on the unit ball of R^3, m >= 0."""
    rs = math.sqrt(s)
    if m < 1e-12:
        return 3 * (math.sqrt(math.pi / 2) * special.erf(rs / math.sqrt(2)) / s ** 1.5 - math.exp(-s / 2) / s)

    def part(sg):
        lo, hi = -sg * m, 1 - sg * m
        return ((math.exp(-s * lo * lo / 2) - math.exp(-s * hi * hi / 2)) / s
                + sg * m * math.sqrt(2 * math.pi / s) * (special.ndtr(rs * hi) - special.ndtr(rs * lo)))
    return 1.5 * (part(1) - part(-1)) / (s * m)


def g_balls(t, sigma=SIGMA, tol=1e-7):
    """Uniform prior on the product of the unit balls of R^3: a in polar coordinates (radius r, c = cosine
    of its angle to e1), b integrated in closed form, the b-part of the energy by a central difference."""
    def h(r, c, which):
        s, m = t * r * r, sigma * c / r
        B = _ball_B(s, m)
        J = 1.5 * r * r * math.exp(-t * sigma * sigma * (1 - c * c) / 2) * B
        if J <= 0.0 or B <= 0.0:
            return 0.0
        if which == 0:
            return J
        Bp, Bm = _ball_B(s * (1 + 1e-5), m), _ball_B(s * (1 - 1e-5), m)
        if Bp <= 0.0 or Bm <= 0.0:
            return 0.0
        return J * (t * sigma * sigma * (1 - c * c) / 2 - (math.log(Bp) - math.log(Bm)) / 2e-5)

    def inner(c, which):
        edges = sorted({0.0, 1.0, *[p for p in (sigma * c, 1 / math.sqrt(t)) if 0 < p < 1]})
        return sum(integrate.quad(lambda r: h(r, c, which), lo, hi, epsabs=0, epsrel=tol, limit=100)[0]
                   for lo, hi in zip(edges[:-1], edges[1:]))
    edges = sorted({0.0, 1.0, *[p for p in (1 - 1 / (t * sigma * sigma), 1 - 10 / (t * sigma * sigma)) if 0 < p < 1]})
    with warnings.catch_warnings():
        # a few inner integrals near c = 0, where the integrand is below 1e-8 of its maximum, cannot reach
        # the relative tolerance; the result agrees with a run at tolerance 1e-12 to four decimals
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        Z, W = (sum(integrate.quad(lambda c: inner(c, k), lo, hi, epsabs=0, epsrel=tol, limit=100)[0]
                    for lo, hi in zip(edges[:-1], edges[1:])) for k in (0, 1))
    return W / Z


def population_value(g, n):
    t1 = n / math.log(n)
    return 3 * g(t1) - 2 * g(1.5 * t1)


def gibbs_energies(n, C, sweeps, burn, seed):
    """Exact-conditional Gibbs sampler for the population posterior of model 3 at b1 = 1/log n: given a,
    the b_j are independent N(0, 1/(b1 n |a|^2)) truncated to [-1, 1], and symmetrically. Each update
    rescales a whole factor, so the sampler moves across the scales of the two branches freely."""
    gen, t = sk.rng(seed), n / math.log(n)

    def draw(other):
        sd = 1 / np.sqrt(t * np.sum(other * other, 1))
        p = special.ndtr(1 / sd)[:, None]
        q = (1 - p) + gen.random((C, 3)) * (2 * p - 1)
        return np.clip(sd[:, None] * special.ndtri(q), -1, 1)
    a, b, E = gen.uniform(-1, 1, (C, 3)), gen.uniform(-1, 1, (C, 3)), []
    for s in range(sweeps):
        b = draw(a)
        a = draw(b)
        if s >= burn:
            E.append(0.5 * n * np.sum(a * a, 1) * np.sum(b * b, 1))
    return np.array(E)


def exact_means(model, stats, n):
    """Exact tempered means of the energy at b1 and b2 for one data set (models 1 and 2)."""
    Sx, Sy, Syy = stats
    out = []
    for bt in betas(n):
        if model == "plane":
            what = np.linalg.solve(Sx, Sy)
            s = 12.0 / math.sqrt(bt * np.linalg.eigvalsh(Sx).min())
            box = [(max(-1.0, what[j] - s), min(1.0, what[j] + s)) for j in range(2)]
            en = lambda w: 0.5 * (Syy - 2 * w @ Sy + np.einsum("ki,ij,kj->k", w, Sx, w))
            out.append(sk.tempered_mean_grid(en, box, 500, beta=bt))
        else:
            sxx, sxy = float(Sx[0, 0]), float(Sy[0])
            chat = sxy / sxx
            s = 1.0 / math.sqrt(bt * sxx)
            lo, hi = max(-1.0, chat - 40 * s), min(1.0, chat + 40 * s)
            cuts = sorted({lo, hi, *([0.0] if lo < 0 < hi else [])})
            f = lambda c: math.exp(-0.5 * bt * sxx * (c - chat) ** 2) * 0.5 * math.log(1 / abs(c))
            num = den = 0.0
            for u, v in zip(cuts[:-1], cuts[1:]):
                den += integrate.quad(f, u, v, limit=400, epsabs=0, epsrel=1e-11)[0]
                num += integrate.quad(lambda c: 0.5 * sxx * (c - chat) ** 2 * f(c), u, v, limit=400,
                                      epsabs=0, epsrel=1e-11)[0]
            out.append(0.5 * (Syy - sxy * sxy / sxx) + num / den)
    return out


def run_regression(model, seed):
    rows = []
    k = 2 if model == "plane" else 1
    for ni, n in enumerate(NS):
        stats = []
        for r in range(R):
            g = sk.rng(seed + 1000 * ni + r)
            x = g.standard_normal((n, k))
            y = g.standard_normal(n)                     # truth 0: y is the noise
            stats.append((x.T @ x, x.T @ y, float(y @ y)))
        C = R * CHAINS
        Sxx = np.repeat(np.array([s[0] for s in stats]), CHAINS, axis=0)
        Sxy = np.repeat(np.array([s[1] for s in stats]), CHAINS, axis=0)
        Syy = np.repeat(np.array([s[2] for s in stats]), CHAINS)
        if model == "plane":
            energy = lambda w: 0.5 * (Syy - 2 * np.einsum("ci,ci->c", w, Sxy) + np.einsum("ci,cij,cj->c", w, Sxx, w))
        else:
            def energy(w):
                c = w[:, 0] * w[:, 1]
                return 0.5 * (Syy - 2 * c * Sxy[:, 0] + c * c * Sxx[:, 0, 0])
        E, acc = metropolis_energies(energy, 2, np.full(C, betas(n)[0]), seed + 500 + ni)
        E = E.reshape(len(E), R, CHAINS)
        lam_mh = np.array([lam_from_draws(E[:, r, :].ravel(), n) for r in range(R)])
        lam_ex = np.array([lam_from_means(*exact_means(model, stats[r], n), n) for r in range(R)])
        mh, ex = sk.mean_ci(lam_mh), sk.mean_ci(lam_ex)
        diff = np.abs(lam_mh - lam_ex)
        pred = lam_population(0.5, 2, n, KAPPA_PRODUCT) if model == "product" else 1.0
        print(f"{model:8s} n = {n:6d}: Metropolis {mh[0]:.3f} +- {mh[2] - mh[0]:.3f}; exact {ex[0]:.3f} +- "
              f"{ex[2] - ex[0]:.3f}; |Metropolis - exact| per data set: mean {diff.mean():.3f}, max {diff.max():.3f}; "
              f"population prediction {pred:.3f}; acceptance {acc:.2f}")
        rows.append((n, mh[0], mh[2] - mh[0], 1.25 * n, ex[0], ex[2] - ex[0], pred))
    return rows


def run_population(seed):
    """Rank-one regression 3 x 3 without noise: energy n K(a, b) with K = |b a^T - C0|_F^2 / 2."""
    rows = {n: [n] for n in NS}
    for truth in ("zero", "rankone"):
        for ni, n in enumerate(NS):
            C0 = np.zeros((3, 3))
            if truth == "rankone":
                C0[0, 0] = SIGMA

            def energy(w):
                D = w[:, :3, None] * w[:, None, 3:] - C0           # b a^T - C0
                return 0.5 * n * np.sum(D * D, axis=(1, 2))
            E, acc = metropolis_energies(energy, 6, np.full(CHAINS_POP, betas(n)[0]), seed + 10 * ni + (5 if truth == "rankone" else 0),
                                         adapt=ADAPT_POP, keep=KEEP_POP, thin=THIN_POP)
            groups = E.reshape(len(E), CHAINS_POP // 16, 16)
            lam = np.array([lam_from_draws(groups[:, gi, :].ravel(), n) for gi in range(CHAINS_POP // 16)])
            est = sk.mean_ci(lam)
            if truth == "zero":
                exact = lam_population(1.5, 2, n, KAPPA)
                G = gibbs_energies(n, CHAINS_POP, 2000, 400, seed + 1000 + ni).reshape(-1, CHAINS_POP // 16, 16)
                gib = sk.mean_ci([lam_from_draws(G[:, gi, :].ravel(), n) for gi in range(CHAINS_POP // 16)])
                print(f"rank-one 3x3, truth 0, no noise, n = {n:6d}: Metropolis {est[0]:.3f} +- {est[2] - est[0]:.3f}; "
                      f"Gibbs {gib[0]:.3f} +- {gib[2] - gib[0]:.3f}; exact {exact:.3f}; acceptance {acc:.2f}")
            else:
                exact = population_value(g_box, n)
                balls = population_value(g_balls, n) if n < 100_000 else float("nan")
                print(f"rank-one 3x3, truth (1/2) e1 e1^T, no noise, n = {n:6d}: Metropolis {est[0]:.3f} +- "
                      f"{est[2] - est[0]:.3f}; exact population value {exact:.3f} (lambda = 5/2); with the prior "
                      f"uniform on the product of balls {balls:.3f}; acceptance {acc:.2f}")
            rows[n] += [est[0], est[2] - est[0], exact]
    return [rows[n] for n in NS]


def main():
    t0 = time.time()
    rows = run_regression("plane", 71_000)
    sk.write_dat("figures/data/07-catalogue-plane.dat", ["n", "mh", "mherr", "nex", "ex", "exerr", "pred"], rows)
    rows = run_regression("product", 72_000)
    sk.write_dat("figures/data/07-catalogue-product.dat", ["n", "mh", "mherr", "nex", "ex", "exerr", "pred"], rows)
    rows = run_population(73_000)
    sk.write_dat("figures/data/07-catalogue-rankone-pop.dat",
                 ["n", "mh", "mherr", "exact", "mh1", "mh1err", "exact1"], rows)
    print(f"total time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
