"""Chapter 11 (Temperature): the learning coefficient from two temperatures, the specific heat, and
WBIC against the free energy, for four running models at the truth w0 = 0.

Models (uniform prior on [-1, 1]^2, x ~ N(0, I), unit noise):
    the regular plane  y = a x1 + b x2      lambda = 1,   m = 1  (d/2 = 1)
    the quartic model  y = a x1 + b^2 x2    lambda = 3/4, m = 1
    the sextic model   y = a x1 + b^3 x2    lambda = 2/3, m = 1
    the product model  y = a b x            lambda = 1/2, m = 2
Every quantity is computed by quadrature from the sufficient statistics S = sum x x^T and
r = sum x y, which are sampled exactly: S ~ Wishart(n, I) (chi-square for one input) and
r | S ~ N(0, S). The energy is n K_n(w) = c^T S c / 2 - c^T r with c = (a, b^k) or c = ab.
For the two-input models the a-integral is a truncated Gaussian integral done in closed form and
the b-integral uses 2 x 400 Gauss-Legendre nodes on a window; for the product model the integral
over c = ab against the density log(1/|c|)/2 uses 2 x 300 Gauss-Legendre nodes after c = W t^3.
Accuracy: agreement with the 2000 x 2000 grid of sltkit to 6 digits at n = 1000, and stability
to 1e-9 under changes of the window and the number of nodes (checked in build/scratch-temperature).

For each model, each n in 10^2..10^8 and R = 200 data sets (seed 11000 + 100 * model + log10 n):
    F    = F_n - n S_n                     (beta = 1)
    W    = WBIC - n S_n = E^{b1}[n K_n]    (b1 = 1 / log n)
    lam  = (E^{b1} - E^{b2}) / (1/b1 - 1/b2),   b2 = 1.5 / log n   (Watanabe 2013, Corollary 3)
    heat = b1^2 Var^{b1}(n K_n)                                  (specific heat estimator)
    z2   = r_2 / sqrt(n)  (the noise in direction x2; r_1/sqrt(n) for the product model)
Predictions: the population values of Proposition prop:temperature:mean, exact:
    product model  b E^b[nK] = 1/2 - 1/(log(n b) + gamma + log 2),
    the other models b E^b[nK] = lambda (up to exponentially small boundary terms);
and Theorem thm:temperature:wbiclinear for the slope of WBIC on z2: -Gamma(3/4)/Gamma(1/4) sqrt(log n / 2)
for the quartic model, 0 for the others (odd symmetry).

Metropolis check: for 40 data sets per model at n = 10^2, 10^4, 10^6 (seeds 21000 + ...), random-walk
Metropolis (sltkit.metropolis) on (a, b) at b1 with 16 chains per data set started from the prior,
2000 adaptation steps toward acceptance 0.3, then 8000 steps thinned by 4; E^{b2} by reweighting the
same draws (Watanabe 2013, eq. (20)). Output: paired differences from quadrature on the same data
sets; for the product model at n = 10^6 also chains four times as long, and the fraction of draws
near the centre of the cross against the exact posterior (sltkit.sample_product_posterior).

Writes figures/data/11-temperature-lambda.dat, 11-temperature-lambda-mcmc.dat,
11-temperature-lambda-wbic.dat and 11-temperature-lambda-curves.dat (prediction curves). Runs in
about a minute.
"""
import math
import os
import sys
import time

import numpy as np
from scipy import special, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk

GAMMA_E = 0.5772156649015329
KAPPA = special.gamma(0.75) / special.gamma(0.25)        # 0.33799
B01, B02 = 1.0, 1.5
R = 200
NS = [10 ** e for e in range(2, 9)]
MODELS = [("regular", 1, 1.0), ("quartic", 2, 0.75), ("sextic", 3, 2.0 / 3.0), ("product", 0, 0.5)]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

GL_X, GL_W = np.polynomial.legendre.leggauss(400)
GL3_X, GL3_W = np.polynomial.legendre.leggauss(300)


# ---------------------------------------------------------------------------------------------
# data: exact sufficient statistics
# ---------------------------------------------------------------------------------------------
def sample_stats(n, k, gen):
    """S = sum x x^T ~ Wishart(n, I_k) by Bartlett's decomposition, r = sum x eps ~ N(0, S) given S."""
    A = np.zeros((k, k))
    for i in range(k):
        A[i, i] = math.sqrt(gen.chisquare(n - i))
        for j in range(i):
            A[i, j] = gen.standard_normal()
    S = A @ A.T
    r = np.linalg.cholesky(S) @ gen.standard_normal(k)
    return S, r


# ---------------------------------------------------------------------------------------------
# quadrature: log Z(beta), E^beta[nK_n], Var^beta[nK_n]
# ---------------------------------------------------------------------------------------------
def _gl(lo, hi):
    return 0.5 * (hi - lo) * GL_X + 0.5 * (hi + lo), 0.5 * (hi - lo) * GL_W


def moments_two_input(S, r, k, beta, nwin=16.0):
    """y = a x1 + b^k x2: exact truncated-Gaussian integral over a in [-1, 1], Gauss-Legendre in b."""
    S11, S12, S22 = S[0, 0], S[0, 1], S[1, 1]
    r1, r2 = r
    B = min(1.0, nwin * (beta * S22) ** (-1.0 / (2 * k)) + 2 * abs(r2 / S22) ** (1.0 / k))
    b1, w1 = _gl(-B, 0.0)
    b2, w2 = _gl(0.0, B)
    b, wb = np.concatenate([b1, b2]), np.concatenate([w1, w2])
    bk = b ** k
    m = (r1 - S12 * bk) / S11                       # q = S11 (a - m)^2 / 2 + Rb
    Rb = 0.5 * S22 * bk * bk - r2 * bk - 0.5 * S11 * m * m
    A = beta * S11
    sA = math.sqrt(A)
    zhi, zlo = sA * (1 - m), sA * (-1 - m)
    dPhi = special.ndtr(zhi) - special.ndtr(zlo)
    ehi, elo = np.exp(-0.5 * zhi ** 2), np.exp(-0.5 * zlo ** 2)
    J0 = math.sqrt(2 * math.pi) * dPhi
    J2 = J0 - zhi * ehi + zlo * elo
    J4 = -(zhi ** 3) * ehi + (zlo ** 3) * elo + 3 * J2
    I0, Iu, Iu2 = J0 / sA, J2 / (A * sA), J4 / (A * A * sA)
    logw = -beta * Rb + np.log(wb) + np.log(I0)
    mx = logw.max()
    wts = np.exp(logw - mx)
    Z = wts.sum()
    Eq_b = Rb + 0.5 * S11 * Iu / I0
    Eq2_b = Rb * Rb + S11 * Rb * Iu / I0 + 0.25 * S11 ** 2 * Iu2 / I0
    Eq = float((wts * Eq_b).sum() / Z)
    Eq2 = float((wts * Eq2_b).sum() / Z)
    return mx + math.log(Z) + math.log(0.25), Eq, Eq2 - Eq * Eq


def moments_product(S, r, beta, nwin=16.0):
    """y = abx through c = ab, whose prior density is log(1/|c|)/2 on [-1, 1]."""
    S, r = float(S[0, 0]), float(r[0])
    W = min(1.0, abs(r / S) + nwin / math.sqrt(beta * S))
    t, wt = 0.5 * (GL3_X + 1), 0.5 * GL3_W
    c = np.concatenate([W * t ** 3, -W * t ** 3])
    w = np.concatenate([wt * 3 * W * t ** 2] * 2)
    q = 0.5 * S * c * c - r * c
    logw = -beta * q + np.log(w * 0.5 * (-np.log(np.abs(c))))
    mx = logw.max()
    wts = np.exp(logw - mx)
    Z = wts.sum()
    Eq = float((wts * q).sum() / Z)
    Eq2 = float((wts * q * q).sum() / Z)
    return mx + math.log(Z), Eq, Eq2 - Eq * Eq


def moments(S, r, k, beta):
    return moments_product(S, r, beta) if k == 0 else moments_two_input(S, r, k, beta)


# ---------------------------------------------------------------------------------------------
# population predictions (Proposition prop:temperature:mean)
# ---------------------------------------------------------------------------------------------
def g_pop(k, lam, s):
    """beta E^beta_pop[n K] as a function of s = n beta (exact up to exponentially small terms)."""
    return 0.5 - 1.0 / (math.log(s) + GAMMA_E + math.log(2.0)) if k == 0 else lam


def heat_pop(k, lam, s):
    if k == 0:
        x = math.log(s) + GAMMA_E + math.log(2.0)
        return 0.5 - 1.0 / x - 1.0 / x ** 2
    return lam


# ---------------------------------------------------------------------------------------------
# Metropolis at b1 with reweighting to b2
# ---------------------------------------------------------------------------------------------
def coef(w, k):
    if k == 0:
        return (w[:, 0] * w[:, 1])[:, None]
    return np.stack([w[:, 0], w[:, 1] ** k], axis=1)


def metropolis_lambda(stats_list, k, beta1, beta2, seed, chains=16, steps=8000, thin=4):
    D = len(stats_list)
    dim = 1 if k == 0 else 2
    Sarr = np.repeat(np.array([s for s, _ in stats_list]).reshape(D, dim, dim), chains, axis=0)
    rarr = np.repeat(np.array([r for _, r in stats_list]).reshape(D, dim), chains, axis=0)

    def energy(w):
        c = coef(w, k)
        return 0.5 * np.einsum("ci,cij,cj->c", c, Sarr, c) - np.einsum("ci,ci->c", c, rarr)

    def logp(w):
        out = -beta1 * energy(w)
        out[np.any(np.abs(w) > 1.0, axis=1)] = -np.inf
        return out

    gen = sk.rng(seed)
    x0 = gen.uniform(-1, 1, size=(D * chains, 2))
    draws, acc = sk.metropolis(logp, x0, 0.1, steps, seed=seed + 1, adapt=2000, thin=thin)
    T, C, _ = draws.shape
    q = np.array([energy(draws[t]) for t in range(T)])          # shape (T, C)
    lam_hat = []
    for j in range(D):
        qj = q[:, j * chains:(j + 1) * chains].ravel()
        E1 = qj.mean()
        lw = -(beta2 - beta1) * qj
        wts = np.exp(lw - lw.max())
        E2 = float((wts * qj).sum() / wts.sum())
        lam_hat.append((E1 - E2) / (1 / beta1 - 1 / beta2))
    # fraction of all kept draws near the centre of the parameter square
    centre = float(np.mean(np.max(np.abs(draws), axis=2) < 0.05))
    return np.array(lam_hat), centre, acc


def main():
    t0 = time.time()
    lam_rows, wbic_rows, mcmc_rows = [], [], []
    for im, (name, k, lam) in enumerate(MODELS):
        dim = 1 if k == 0 else 2
        print(f"== {name}: lambda = {lam:.4f} ==")
        print("      n  lamhat  95% CI            pop.pred  sd(lamhat)  heat    pop.pred   "
              "E[F-WBIC] (95% CI)          slope(WBIC on z2) (95% CI)    pred")
        for n in NS:
            L = math.log(n)
            b1, b2 = B01 / L, B02 / L
            gen = sk.rng(11000 + 100 * im + int(round(math.log10(n))))
            rows = []
            for _ in range(R):
                S, r = sample_stats(n, dim, gen)
                lz, _, _ = moments(S, r, k, 1.0)
                E1, V1 = moments(S, r, k, b1)[1:]
                E2 = moments(S, r, k, b2)[1]
                z2 = (r[0] / math.sqrt(S[0, 0])) if k == 0 else r[1] / math.sqrt(n)
                z1 = 0.0 if k == 0 else r[0] / math.sqrt(n)
                rows.append((-lz, E1, (E1 - E2) / (1 / b1 - 1 / b2), b1 * b1 * V1, z2, z1))
            F, Wb, lh, heat, z2, z1 = np.array(rows).T
            m_l, lo_l, hi_l = sk.mean_ci(lh)
            m_h, lo_h, hi_h = sk.mean_ci(heat)
            lam_pop = (g_pop(k, lam, n * b1) / b1 - g_pop(k, lam, n * b2) / b2) / (1 / b1 - 1 / b2)
            heat_pred = heat_pop(k, lam, n * b1)
            d = F - Wb
            m_d, lo_d, hi_d = sk.mean_ci(d)
            # coefficient of z2 in a least-squares fit of WBIC - nS_n on 1, z2, z2^2 (and z1^2): the
            # squares absorb the O_p(1) terms of the theorem, which are even in the noise to leading
            # order; 95% interval from the heteroscedasticity-consistent (HC3) standard error
            cols = [z2, np.ones_like(z2), z2 ** 2] + ([] if k == 0 else [z1 ** 2])
            X = np.vstack(cols).T
            coefs = np.linalg.lstsq(X, Wb, rcond=None)[0]
            resid = Wb - X @ coefs
            XtXi = np.linalg.inv(X.T @ X)
            hdiag = np.einsum("ij,jk,ik->i", X, XtXi, X)
            meat = (X * (resid / (1 - hdiag))[:, None] ** 2).T @ X
            se = math.sqrt((XtXi @ meat @ XtXi)[0, 0])
            tq = stats.t.ppf(0.975, R - X.shape[1])
            slope = coefs[0]
            slope_pred = -KAPPA * math.sqrt(L / 2) if k == 2 else 0.0
            print(f"{n:8.0e} {m_l:.4f} [{lo_l:.4f},{hi_l:.4f}]  {lam_pop:.4f}   {np.std(lh, ddof=1):.4f}    "
                  f"{m_h:.4f}  {heat_pred:.4f}    {m_d:8.4f} [{lo_d:.4f},{hi_d:.4f}]   "
                  f"{slope:8.4f} [{slope - tq*se:.4f},{slope + tq*se:.4f}]   {slope_pred:8.4f}")
            lam_rows.append((name, n, m_l, m_l - lo_l, hi_l - m_l, lam_pop, np.std(lh, ddof=1), m_h,
                             m_h - lo_h, hi_h - m_h, heat_pred))
            wbic_rows.append((name, n, m_d, m_d - lo_d, hi_d - m_d, np.std(d, ddof=1), slope, tq * se,
                              slope_pred, float(np.mean(Wb - lam * L)), float(np.mean(F - lam * L))))
        print(f"   ({time.time() - t0:.1f}s)")

    # Metropolis validation
    print("\n== Metropolis check: lamhat(MCMC) - lamhat(quadrature) on the same data sets ==")
    for im, (name, k, lam) in enumerate(MODELS):
        dim = 1 if k == 0 else 2
        for n in (10 ** 2, 10 ** 4, 10 ** 6):
            L = math.log(n)
            b1, b2 = B01 / L, B02 / L
            gen = sk.rng(21000 + 100 * im + int(round(math.log10(n))))
            st = [sample_stats(n, dim, gen) for _ in range(40)]
            lq = []
            for S, r in st:
                E1 = moments(S, r, k, b1)[1]
                E2 = moments(S, r, k, b2)[1]
                lq.append((E1 - E2) / (1 / b1 - 1 / b2))
            lq = np.array(lq)
            # Metropolis runs on the 2-D parameter (a, b) for every model, the product model included
            lm, _, acc = metropolis_lambda(st, k, b1, b2, seed=31000 + 100 * im + int(round(math.log10(n))))
            dlt = lm - lq
            md, lo, hi = sk.mean_ci(dlt)
            mm, mlo, mhi = sk.mean_ci(lm)
            print(f"{name:8s} n={n:8.0e}  quad {lq.mean():.4f}  MCMC {mm:.4f} [{mlo:.4f},{mhi:.4f}]  "
                  f"diff {md:+.4f} [{lo:+.4f},{hi:+.4f}]  acceptance {acc:.2f}")
            mcmc_rows.append((name, n, mm, mm - mlo, mhi - mm, float(lq.mean()), md, md - lo, hi - md, acc))
            if k == 0 and n == 10 ** 6:
                # the same data sets with chains four times as long, and the exact posterior
                # fraction of the centre max(|a|, |b|) < 0.05 from the exact sampler of sltkit
                lm4, centre4, acc4 = metropolis_lambda(st, k, b1, b2, seed=41000, steps=32000, thin=16)
                md4, lo4, hi4 = sk.mean_ci(lm4 - lq)
                _, centre1, _ = metropolis_lambda(st, k, b1, b2, seed=31000 + 100 * im + 6)
                exact = []
                for j, (S, r) in enumerate(st):
                    sk.PRODUCT.n, sk.PRODUCT.Sxx, sk.PRODUCT.Sxy, sk.PRODUCT.Syy = n, S, r, 0.0
                    dr = sk.sample_product_posterior(sk.PRODUCT, 20000, seed=51000 + j, beta=b1)
                    exact.append(np.mean(np.max(np.abs(dr), axis=1) < 0.05))
                print(f"{name:8s} n={n:8.0e}  chains x4: MCMC {lm4.mean():.4f}  diff {md4:+.4f} "
                      f"[{lo4:+.4f},{hi4:+.4f}]  acceptance {acc4:.2f}")
                print(f"          fraction of draws with max(|a|,|b|) < 0.05: short chains {centre1:.3f}, "
                      f"long chains {centre4:.3f}, exact {np.mean(exact):.3f}")
        print(f"   ({time.time() - t0:.1f}s)")

    # write data: one file per quantity, with one column group per model
    def wide(rows, cols, path, header_extra=()):
        names = [m[0] for m in MODELS]
        header = ["n"] + [f"{nm}{c}" for nm in names for c in cols]
        table = []
        for n in sorted({r[1] for r in rows}):
            row = [n]
            for nm in names:
                rr = [r for r in rows if r[0] == nm and r[1] == n][0]
                row += list(rr[2:2 + len(cols)])
            table.append(row)
        sk.write_dat(path, header, table)

    wide(lam_rows, ["lam", "lamlo", "lamhi", "lampop", "lamsd", "heat", "heatlo", "heathi", "heatpop"],
         os.path.join(OUT, "11-temperature-lambda.dat"))
    wide(wbic_rows, ["dF", "dFlo", "dFhi", "dFsd", "slope", "slopeci", "slopepred", "Wc", "Fc"],
         os.path.join(OUT, "11-temperature-lambda-wbic.dat"))
    wide(mcmc_rows, ["mc", "mclo", "mchi", "quad", "diff", "difflo", "diffhi", "acc"],
         os.path.join(OUT, "11-temperature-lambda-mcmc.dat"))
    # smooth prediction curves for the figure. quarticexact: the average slope E[d/dz2 WBIC] of the
    # quartic model computed by quadrature with S = n I exactly and z2 averaged by 40-point
    # Gauss-Hermite quadrature (all orders of the expansion, no sampling noise)
    zz, ww = np.polynomial.hermite_e.hermegauss(40)
    ww = ww / ww.sum()
    curve = []
    for e in np.linspace(2, 8, 25):
        n = 10 ** e
        L = math.log(n)
        b1, b2 = B01 / L, B02 / L
        lp = (g_pop(0, 0.5, n * b1) / b1 - g_pop(0, 0.5, n * b2) / b2) / (1 / b1 - 1 / b2)
        Sn = n * np.eye(2)
        h = 1e-4
        f = lambda z: moments_two_input(Sn, np.array([0.0, math.sqrt(n) * z]), 2, b1)[1]
        exact = float(ww @ np.array([(f(z + h) - f(z - h)) / (2 * h) for z in zz]))
        curve.append((n, lp, -KAPPA * math.sqrt(L / 2), exact,
                      -math.log(L) + 0.5 * math.log(8 / math.pi) + 1.0, math.log(2 / math.pi)))
    sk.write_dat(os.path.join(OUT, "11-temperature-lambda-curves.dat"),
                 ["n", "prodlampop", "quarticslope", "quarticexact", "prodFW", "regFW"], curve)
    print("  quartic: leading slope, exact average slope at n = 1e2, 1e4, 1e6, 1e8:",
          [(round(c[2], 4), round(c[3], 4)) for c in curve if abs(math.log10(c[0]) % 2) < 1e-9])
    print("\nnumbers quoted in the text:")
    print("  Gamma(3/4)/Gamma(1/4) =", KAPPA, "; sd prediction of lamhat (quartic) 0.1316/sqrt(log n):",
          [round(KAPPA / math.sqrt(2) / ((B01 ** -0.5 + B02 ** -0.5) * math.sqrt(math.log(n))), 4) for n in NS])
    print("  log(2/pi) =", math.log(2 / math.pi), "; product E[F-WBIC] asymptote at n = 1e2..1e8:",
          [round(-math.log(math.log(n)) + 0.5 * math.log(8 / math.pi) + 1.0, 3) for n in NS])
    print(f"total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
