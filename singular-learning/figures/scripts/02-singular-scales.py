"""Chapter 2 (Singular Models), "Measuring it": the posterior of y = abx at the singular truth
spreads uniformly over scales (Theorem thm:singular:scales).

The product model y = abx + N(0,1) noise, x ~ N(0,1), truth ab = 0, uniform prior on [-1,1]^2.
The data enter the posterior only through S_xx = sum x_i^2 and S_xy = sum x_i y_i, and under the
truth S_xx ~ chi^2_n and S_xy | S_xx ~ N(0, S_xx) exactly; so data sets of every size, up to
n = 10^16, are simulated exactly through these two statistics. Posterior draws are exact: the
coefficient c = ab from its one-dimensional posterior, then log|a| uniform on [log|c|, 0]
(sk.sample_product_posterior, the method derived in the chapter).

Predictions. Theorem thm:singular:scales (i) gives the posterior distribution function of log|a|
in closed form: with Z = S_xy / sqrt(S_xx) and ell = (1/2) log S_xx,
    Pi_n(log|a| <= u) = H_Z(u + ell) / H_Z(ell),   H_Z(L) = E (L - log|Z + xi|)_+ = int_{-inf}^L G_Z(e^w) dw,
where G_Z(t) = P(|Z + xi| <= t) and xi ~ N(0,1). Since Z ~ N(0,1) exactly, the average over data
sets of every posterior quantity is an integral over Z (Gauss-Hermite quadrature, S_xx = n).
The regular comparison is the regular plane y = a x1 + b x2 with the same prior and truth, whose
posterior of a is N(a_hat, v) with a_hat / sqrt(v) ~ N(0,1) and 1/v ~ chi^2_{n-1} (the box
truncates nothing that matters at these n).

Outputs (figures/data):
  02-singular-loglog-n{4,8,12}.dat   1500 exact draws (log10|a|, log10|b|), one data set each
  02-singular-cdf.dat                posterior CDF of log|a|/log n at n = 10^6: measured (200 data
                                     sets x 2000 draws, mean and Student-t 95% interval) for both
                                     models, on a grid of tau
  02-singular-cdfpred.dat            predicted CDFs at n = 10^6 (exact singular, limit, regular)
  02-singular-quartiles.dat          measured quartiles of log|a|/log n against n, both models
  02-singular-quartpred.dat          predicted (data-set averaged) quartiles against n, both models
Run from the volume root: python figures/scripts/02-singular-scales.py   (about 30 s)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import special

import sltkit as sk

OUT = os.path.join("figures", "data")
GAMMA = 0.5772156649015329
PS = (0.25, 0.5, 0.75)
R, M = 200, 2000                      # data sets per n, posterior draws per data set

# grid in w = log t for H_Z(L) = int_{-inf}^L G_Z(e^w) dw; below w = -50 the integrand is < 2 e^w phi(Z)
WGRID = np.arange(-50.0, 30.0, 0.002)


def G(z, w=WGRID):
    t = np.exp(w)
    return special.ndtr(t - z) - special.ndtr(-t - z)


def H_table(z):
    g = G(z)
    return np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(WGRID))])


def exact_cdf(u, z, ell, H=None):
    """Theorem (i): Pi_n(log|a| <= u) for u <= 0."""
    H = H_table(z) if H is None else H
    return np.interp(np.asarray(u) + ell, WGRID, H) / np.interp(ell, WGRID, H)


def exact_quantile(p, z, ell, H=None):
    """The p-quantile of log|a| under the posterior (inverse of exact_cdf)."""
    H = H_table(z) if H is None else H
    return float(np.interp(p * np.interp(ell, WGRID, H), H, WGRID) - ell)


def gauss_log_quantile(p, z, g=None):
    """The p-quantile of log|z + xi|, xi ~ N(0,1): the regular plane's posterior of log|a| - (1/2) log v."""
    return float(np.interp(p, G(z) if g is None else g, WGRID))


def product_model(n, Sxx, Sxy):
    """A fitted y = abx model built from its sufficient statistics (what LinearFeatureModel.fit stores)."""
    m = sk.LinearFeatureModel(lambda w: (w[..., 0] * w[..., 1])[..., None], d=2, k=1, name="y = abx")
    m.n, m.Sxx, m.Sxy, m.Syy = int(min(n, 2**62)), np.array([[Sxx]]), np.array([Sxy]), float("nan")
    return m


def product_stats(n, gen):
    Sxx = gen.chisquare(n)
    return Sxx, math.sqrt(Sxx) * gen.standard_normal()


def regular_draws(n, gen, m):
    """Posterior draws of a for the regular plane at the truth 0: a_hat = sqrt(v) Z', 1/v ~ chi^2_{n-1}."""
    v = 1.0 / gen.chisquare(n - 1)
    a_hat = math.sqrt(v) * gen.standard_normal()
    return a_hat + math.sqrt(v) * gen.standard_normal(m)


ZN, ZW = np.polynomial.hermite_e.hermegauss(80)
ZW = ZW / ZW.sum()
HTAB = [H_table(z) for z in ZN]          # one table per Gauss-Hermite node
GTAB = [G(z) for z in ZN]


def mean_half_log_chi2(k):
    """E[(1/2) log chi^2_k] = (psi(k/2) + log 2) / 2."""
    return 0.5 * (special.digamma(0.5 * k) + math.log(2.0))


def predicted_singular_quartiles(n):
    ell = mean_half_log_chi2(n)                 # E ell_n, ell_n = (1/2) log S_xx
    return [sum(w * exact_quantile(p, z, ell, H) for z, w, H in zip(ZN, ZW, HTAB)) / math.log(n) for p in PS]


def predicted_regular_quartiles(n):
    half_log_v = -mean_half_log_chi2(n - 1)     # E (1/2) log v, 1/v ~ chi^2_{n-1}
    return [(half_log_v + sum(w * gauss_log_quantile(p, z, g) for z, w, g in zip(ZN, ZW, GTAB))) / math.log(n) for p in PS]


def mean_ci_rows(arr):
    out = []
    for col in np.asarray(arr).T:
        mu, lo, hi = sk.mean_ci(col)
        out += [mu, mu - lo]
    return out


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    # ---- 1. draws in logarithmic coordinates (one data set each) ------------------------------
    for k in (4, 8, 12):
        n = 10 ** k
        Sxx, Sxy = product_stats(n, sk.rng(200 + k))
        d = sk.sample_product_posterior(product_model(n, Sxx, Sxy), 1500, seed=300 + k)
        la, lb = np.log10(np.abs(d[:, 0])), np.log10(np.abs(d[:, 1]))
        sk.write_dat(os.path.join(OUT, f"02-singular-loglog-n{k}.dat"), ["la", "lb"], zip(la, lb))
        lc = np.log10(abs(Sxy / Sxx))
        print(f"log-log draws n=1e{k}: log10|c_hat| = {lc:.2f} (-(1/2)log10 n = {-k/2}); "
              f"fraction of draws with |log10|ab| + k/2| < 1: {np.mean(np.abs(la + lb + k / 2) < 1):.3f}; "
              f"median log10|a| = {np.median(la):.2f}")

    # ---- 2. the posterior CDF of log|a|/log n at n = 10^6 ---------------------------------------
    n = 10 ** 6
    taus = np.round(np.arange(-0.70, 0.0001, 0.025), 3)
    gen = sk.rng(2026)
    cdf_s, cdf_r = [], []
    for r in range(R):
        Sxx, Sxy = product_stats(n, gen)
        d = sk.sample_product_posterior(product_model(n, Sxx, Sxy), M, seed=10_000 + r)
        t = np.log(np.abs(d[:, 0])) / math.log(n)
        cdf_s.append([np.mean(t <= x) for x in taus])
        tr = np.log(np.abs(regular_draws(n, gen, M))) / math.log(n)
        cdf_r.append([np.mean(tr <= x) for x in taus])
    cdf_s, cdf_r = np.array(cdf_s), np.array(cdf_r)
    rows = []
    for j, x in enumerate(taus):
        ms, ls, hs = sk.mean_ci(cdf_s[:, j])
        mr, lr, hr = sk.mean_ci(cdf_r[:, j])
        rows.append((x, ms, ms - ls, mr, mr - lr))
    sk.write_dat(os.path.join(OUT, "02-singular-cdf.dat"), ["tau", "sing", "singci", "reg", "regci"], rows)
    fine = np.linspace(-0.7, 0.0, 141)
    ell = 0.5 * math.log(n)
    pred = np.zeros_like(fine)
    for z, w, H in zip(ZN, ZW, HTAB):
        pred += w * exact_cdf(fine * math.log(n), z, ell, H)
    limit = np.clip(1 + 2 * fine, 0, 1)
    reg = 2 * special.ndtr(np.exp((fine + 0.5) * math.log(n)) / math.sqrt(2)) - 1
    sk.write_dat(os.path.join(OUT, "02-singular-cdfpred.dat"), ["tau", "exact", "limit", "reg"], zip(fine, pred, limit, reg))
    j = int(np.argmin(np.abs(taus + 0.25)))
    print(f"\nCDF at n=1e6, tau=-0.25: measured {cdf_s[:, j].mean():.4f} (pred {np.interp(-0.25, fine, pred):.4f}, limit 0.5); "
          f"regular measured {cdf_r[:, j].mean():.4f} (pred {np.interp(-0.25, fine, reg):.4f})")
    dev = np.abs(cdf_s.mean(axis=0) - np.interp(taus, fine, pred))
    half = np.array([sk.mean_ci(cdf_s[:, j])[2] - cdf_s[:, j].mean() for j in range(len(taus))])
    print(f"  singular: max |measured - predicted| = {dev.max():.4f}; intervals covering the prediction: "
          f"{int(np.sum(dev <= half + 1e-12))}/{len(taus)}")
    devr = np.abs(cdf_r.mean(axis=0) - np.interp(taus, fine, reg))
    halfr = np.array([sk.mean_ci(cdf_r[:, j])[2] - cdf_r[:, j].mean() for j in range(len(taus))])
    print(f"  regular:  max |measured - predicted| = {devr.max():.4f}; intervals covering the prediction: "
          f"{int(np.sum(devr <= halfr + 1e-12))}/{len(taus)}")
    print(f"  sup over tau of |exact CDF - limit| at n=1e6 (data-set average): {np.max(np.abs(pred - limit)):.4f}")

    # ---- 3. quartiles of log|a|/log n against n --------------------------------------------------
    exps = (2, 3, 4, 6, 8, 10, 12, 16)
    qrows = []
    print("\nquartiles of log|a|/log n (mean over data sets, half-width of the 95% interval)")
    for k in exps:
        n = 10 ** k
        gen = sk.rng(5000 + k)
        qs, qr, err = [], [], []
        for r in range(R):
            Sxx, Sxy = product_stats(n, gen)
            d = sk.sample_product_posterior(product_model(n, Sxx, Sxy), M, seed=100_000 * k + r)
            t = np.log(np.abs(d[:, 0])) / math.log(n)
            q = np.quantile(t, PS)
            qs.append(q)
            z, ell = Sxy / math.sqrt(Sxx), 0.5 * math.log(Sxx)
            Hz = H_table(z)
            err.append(q - np.array([exact_quantile(p, z, ell, Hz) for p in PS]) / math.log(n))
            qr.append(np.quantile(np.log(np.abs(regular_draws(n, gen, M))) / math.log(n), PS))
        qs, qr, err = np.array(qs), np.array(qr), np.array(err)
        ps, pr = predicted_singular_quartiles(n), predicted_regular_quartiles(n)
        row = [n] + mean_ci_rows(qs) + mean_ci_rows(qr)
        qrows.append(row)
        print(f"  n=1e{k:<2d} y=abx: " + "  ".join(f"{row[1+2*i]:+.4f}+-{row[2+2*i]:.4f}" for i in range(3))
              + "   pred " + " ".join(f"{v:+.4f}" for v in ps)
              + f"   | regular: " + " ".join(f"{row[7+2*i]:+.4f}" for i in range(3))
              + "  pred " + " ".join(f"{v:+.4f}" for v in pr))
        print(f"          sampler check: mean |draw quartile - exact quartile of the same posterior| = "
              f"{np.mean(np.abs(err)):.5f} (Monte Carlo s.e. of one quartile ~ {math.sqrt(0.1875 / M) / 2:.5f})")
    cov_s = cov_r = 0
    for row in qrows:
        ps, pr = predicted_singular_quartiles(row[0]), predicted_regular_quartiles(row[0])
        cov_s += sum(abs(row[1 + 2 * i] - ps[i]) <= row[2 + 2 * i] for i in range(3))
        cov_r += sum(abs(row[7 + 2 * i] - pr[i]) <= row[8 + 2 * i] for i in range(3))
    print(f"  intervals containing the predicted quartile: y=abx {cov_s}/{3 * len(qrows)}, regular plane {cov_r}/{3 * len(qrows)}")
    head = ["n"] + [f"{m}{s}" for m in ("q1", "q2", "q3") for s in ("", "ci")] + [f"{m}{s}" for m in ("r1", "r2", "r3") for s in ("", "ci")]
    sk.write_dat(os.path.join(OUT, "02-singular-quartiles.dat"), head, qrows)
    grid_n = np.logspace(2, 16, 57)
    prow = []
    for n in grid_n:
        ps, pr = predicted_singular_quartiles(n), predicted_regular_quartiles(n)
        prow.append([n] + ps + pr + [-(1 - p) / 2 * (1 + GAMMA / math.log(n)) for p in PS])
    sk.write_dat(os.path.join(OUT, "02-singular-quartpred.dat"),
                 ["n", "p1", "p2", "p3", "g1", "g2", "g3", "a1", "a2", "a3"], prow)
    for n in (1e2, 1e4, 1e6, 1e9, 1e12, 1e16):
        ps = predicted_singular_quartiles(n)
        ap = [-(1 - p) / 2 * (1 + GAMMA / math.log(n)) for p in PS]
        print(f"  predicted quartiles at n={n:.0e}: " + " ".join(f"{v:+.4f}" for v in ps)
              + "   -(1-p)/2 (1 + gamma/log n): " + " ".join(f"{v:+.4f}" for v in ap))
    print(f"\nruntime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
