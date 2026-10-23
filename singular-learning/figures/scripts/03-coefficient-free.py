"""Chapter 3 (The Free Energy by Hand): the free energy of five one-coefficient models.

Models y = g(w) x + N(0, 1) noise, x ~ N(0, 1), the uniform prior on [-1, 1]^d, truth c0 = g(w0) = 0:
    line    y = a x            c = a          rho(c) = 1/2                    lambda 1/2, m 1, d 1
    prod    y = a b x          c = ab         rho(c) = (1/2) log(1/|c|)       lambda 1/2, m 2, d 2
    chain   y = a1 a2 a3 x     c = a1 a2 a3   rho(c) = (1/4) log^2(1/|c|)     lambda 1/2, m 3, d 3
    cusp    y = a b^2 x        c = a b^2      rho(c) = (1/2)(|c|^(-1/2) - 1)  lambda 1/4, m 1, d 2
    cubic   y = a b^3 x        c = a b^3      rho(c) = (1/4)(|c|^(-2/3) - 1)  lambda 1/6, m 1, d 2
For every model the energy depends on w only through c, and (the chapter's exact representation)
    F_n - n S_n = -Z^2/2 - log int exp(-(S/2)(c - Z/sqrt(S))^2) rho(c) dc,
with S = sum x_i^2 and Z = sum x_i y_i / sqrt(S); Z ~ N(0, 1) independently of S ~ chi^2_n. The
integral is computed by adaptive quadrature in c (sltkit.coefficient_free_energy, relative accuracy
about 1e-10), with the densities of the kit (sltkit.product_density, sltkit.cusp_density) coded as
scalar functions for speed (the script checks that they agree).

Measurements: 200 data sets per model at each n = 10^2, 10^2.5, ..., 10^8, independent across
models; seed 1000000 (j + 1) + 1000 k + r for the j-th model, k-th sample size and r-th data set.
For n <= 10^5 the data sets are simulated in full (sltkit.PRODUCT.sample: under the truth c0 = 0
the data of every model are pure noise); for n > 10^5 the sufficient statistics (S, Z) are drawn
from their exact joint law. Intervals: Student-t 95%.

Columns of figures/data/03-coefficient-free.dat, for each model k:
    kF, kE       mean of F_n - n S_n over the data sets, and the half-width of its 95% interval;
    kfin         the finite-n prediction: the exact representation with S replaced by its mean n,
                 averaged over Z ~ N(0, 1) by 40-node Gauss-Hermite quadrature;
    klim         the limit: lambda log n - (m - 1) log log n + E[chi(Z)] (Theorem 3.4);
    kbic         (d/2) log n - 1/2, the mean of BIC = n L_n(w_hat) + (d/2) log n minus n S_n;
    kD, kDE      mean of D_n = R_n - chi(Z_n), R_n = F_n - n S_n - lambda log n + (m-1) log log n,
                 computed data set by data set, and the half-width of its 95% interval;
    kDfin        the finite-n prediction of E[D_n] (= kfin - klim);
    kDone        the first-order term 2 (m - 1) E[kappa(Z)] / log n = -(m - 1) gamma / log n of
                 Proposition 3.6 (lambda = 1/2; zero when m = 1).

Run from the volume root:  python figures/scripts/03-coefficient-free.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import special  # noqa: E402

import sltkit as sk  # noqa: E402

REPS = 200
EXPONENTS = [2.0 + 0.5 * k for k in range(13)]          # 10^2 ... 10^8
FULL_DATA_MAX = 1e5
GAMMA = float(np.euler_gamma)


def _line(c):
    return 0.5 if abs(c) < 1.0 else 0.0


def _prod(c):
    a = abs(c)
    return 0.5 * math.log(1.0 / a) if 0.0 < a < 1.0 else 0.0


def _chain(c):
    a = abs(c)
    return 0.25 * math.log(1.0 / a) ** 2 if 0.0 < a < 1.0 else 0.0


def _cusp(c):
    a = abs(c)
    return 0.5 * (a ** -0.5 - 1.0) if 0.0 < a < 1.0 else 0.0


def _cubic(c):
    a = abs(c)
    return 0.25 * (a ** (-2.0 / 3.0) - 1.0) if 0.0 < a < 1.0 else 0.0


# name: (scalar density, kit density or None, lambda, m, ell (two-sided), d)
MODELS = {
    "line": (_line, lambda c: sk.product_density(c, 1), 0.5, 1, 0.5, 1),
    "prod": (_prod, lambda c: sk.product_density(c, 2), 0.5, 2, 0.5, 2),
    "chain": (_chain, lambda c: sk.product_density(c, 3), 0.5, 3, 0.25, 3),
    "cusp": (_cusp, sk.cusp_density, 0.25, 1, 0.5, 2),
    "cubic": (_cubic, None, 1.0 / 6.0, 1, 0.25, 2),
}


def free_energy(rho, S, Z):
    """F_n - n S_n from the sufficient statistics (the exact representation)."""
    return sk.coefficient_free_energy(-0.5 * Z * Z, S, rho, -1.0, 1.0, Z / math.sqrt(S))


def chi(lam, m, ell, z):
    """The limit law of Theorem 3.4 for a two-sided ell: (m-1) log 2 - log int e^{-t^2/2+zt}|t|^{2lam-1} ell dt,
    where the integral equals ell 2^lam Gamma(lam) M(lam, 1/2, z^2/2) (Kummer's function M)."""
    return (m - 1) * math.log(2.0) - math.log(ell * 2.0 ** lam * math.gamma(lam) * special.hyp1f1(lam, 0.5, 0.5 * z * z))


def gauss_hermite(nodes):
    z, w = np.polynomial.hermite_e.hermegauss(nodes)
    return z, w / w.sum()


def sufficient_statistics(n, seed):
    """(S, Z) of one data set: simulated in full for n <= FULL_DATA_MAX, else from the exact law."""
    if n <= FULL_DATA_MAX:
        x, y = sk.PRODUCT.sample([0.0, 0.0], n, seed=seed)
        S = float(x[:, 0] @ x[:, 0])
        return S, float(x[:, 0] @ y) / math.sqrt(S)
    gen = sk.rng(seed)
    return float(gen.chisquare(n)), float(gen.standard_normal())


def main() -> None:
    t0 = time.time()
    # the scalar densities agree with the kit's
    cs = np.concatenate([-np.logspace(-12, -1e-9, 50), np.logspace(-12, -1e-9, 50)])
    for k, (rho, kit, *_rest) in MODELS.items():
        if kit is not None:
            err = max(abs(rho(c) - float(kit(c))) / max(1.0, abs(rho(c))) for c in cs)
            print(f"density check [{k}]: max relative difference from the kit {err:.1e}")
    zs, wz = gauss_hermite(40)
    zs80, wz80 = gauss_hermite(80)
    Echi = {k: float(np.dot(wz80, [chi(lam, m, ell, z) for z in zs80])) for k, (_, _, lam, m, ell, _) in MODELS.items()}
    for k, v in Echi.items():
        _, _, lam, m, ell, _ = MODELS[k]
        print(f"E chi [{k}] = {v:.4f}; chi(0) = constant of the population free energy = {chi(lam, m, ell, 0.0):.4f}")
    print(f"check: 1/2 log(8/pi) = {0.5 * math.log(8 / math.pi):.4f}, minus 1/2: {0.5 * math.log(8 / math.pi) - 0.5:.4f}")

    header = ["n"]
    for k in MODELS:
        header += [f"{k}F", f"{k}E", f"{k}fin", f"{k}lim", f"{k}bic", f"{k}D", f"{k}DE", f"{k}Dfin", f"{k}Done"]
    rows = []
    means = {k: [] for k in MODELS}
    inside_F = inside_D = total = 0
    for kk, ex in enumerate(EXPONENTS):
        n = int(round(10 ** ex))
        row = [n]
        for j, (k, (rho, _, lam, m, ell, d)) in enumerate(MODELS.items()):
            stats = [sufficient_statistics(n, 1_000_000 * (j + 1) + 1000 * kk + r) for r in range(REPS)]
            base = lam * math.log(n) - (m - 1) * math.log(math.log(n))
            F = np.array([free_energy(rho, S, Z) for S, Z in stats])
            D = F - base - np.array([chi(lam, m, ell, Z) for _, Z in stats])
            mu, lo, hi = sk.mean_ci(F)
            muD, loD, hiD = sk.mean_ci(D)
            fin = float(np.dot(wz, [free_energy(rho, float(n), z) for z in zs]))
            lim = base + Echi[k]
            bic = 0.5 * d * math.log(n) - 0.5
            one = -(m - 1) * GAMMA / math.log(n)
            row += [mu, hi - mu, fin, lim, bic, muD, hiD - muD, fin - lim, one]
            means[k].append(mu)
            inside_F += abs(mu - fin) <= hi - mu
            inside_D += abs(muD - (fin - lim)) <= hiD - muD
            total += 1
            print(f"n=1e{ex:.1f} {k:5s}: F {mu:8.4f} +- {hi - mu:.4f} | fin {fin:8.4f} | lim {lim:8.4f} | BIC {bic:8.4f}"
                  f" || D {muD:+.4f} +- {hiD - muD:.4f} | fin {fin - lim:+.4f} | first order {one:+.4f}")
        rows.append(row)
    os.makedirs(os.path.join("figures", "data"), exist_ok=True)
    sk.write_dat(os.path.join("figures", "data", "03-coefficient-free.dat"), header, rows)

    # summaries quoted in the text
    ns = np.array([r[0] for r in rows], dtype=float)
    for k, (_, _, lam, m, ell, d) in MODELS.items():
        s_all = np.polyfit(np.log(ns), means[k], 1)[0]
        sel = ns <= 1e6
        s_6 = np.polyfit(np.log(ns[sel]), np.array(means[k])[sel], 1)[0]
        print(f"least-squares slope of the means against log n [{k}]: 1e2..1e8 {s_all:.3f}, 1e2..1e6 {s_6:.3f}; "
              f"lambda {lam:.3f}; lambda-(m-1)/log n at n=1e5: {lam - (m - 1) / math.log(1e5):.3f}, "
              f"at n=1e4: {lam - (m - 1) / math.log(1e4):.3f}; d/2 = {d / 2}")
    print(f"finite-n prediction inside the 95% interval: F {inside_F} of {total}, D {inside_D} of {total}")
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
