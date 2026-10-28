"""Chapter 9, Measuring it: the Bayes generalization error n (G_n - S) of four one-coefficient models.

Models y = c x + N(0, 1), x ~ N(0, 1), truth c0 = 0, prior uniform on [-1, 1]^d, coefficient c = g(w):
  line     y = ax        d = 1   density of c: 1/2 on [-1, 1]                 lambda = 1/2, m = 1
  product  y = abx       d = 2   (1/2) log(1/|c|)        (sk.product_density)  lambda = 1/2, m = 2
  cusp     y = ab^2 x    d = 2   (|c|^(-1/2) - 1) / 2    (sk.cusp_density)     lambda = 1/4, m = 1
  cubic    y = ab^3 x    d = 2   (|c|^(-2/3) - 1) / 4                          lambda = 1/6, m = 1

A data set enters the posterior only through sigma = sqrt(S_xx) (kappa_n in the chapter) and
Z = sum x_i eps_i / sigma, which are independent with sigma^2 ~ chi^2_n and Z ~ N(0, 1) (the reduction
lemma of Section 9.3); the script draws REPS = 1000 independent data sets through these two statistics for
each model and each n (seed 900 + 10 j + log10 n for the j-th model). Given (sigma, Z), the posterior of
t = sigma c has density proportional to exp(-(t - Z)^2 / 2) rho(t / sigma), and the generalization error
is computed exactly (to about 1e-9 in n (G_n - S)) by three nested quadratures:
  G_n - S = E_x KL(x),  KL(x) = -E_eps log E_post[exp(t s eps - t^2 s^2 / 2)],  s = x / sigma,
with Gauss-Hermite rules in x (16 nodes, KL is even in x) and eps (16 nodes) and composite Gauss-Legendre
in t after the substitution t = +-u^p that removes the singularity of rho at 0.

Checks: on the regular line the exact value for the flat prior on R is known per data set,
  KL averaged over x of (1/2)[log(1 + x^2/sigma^2) + (1 + Z^2 x^2/sigma^2)/(1 + x^2/sigma^2) - 1],
and in expectation, E[G_n] - S = (1/2)[digamma((n+1)/2) - digamma(n/2)] (Computation 9.2).

Writes figures/data/09-generalization-onecoef.dat (measured means, 95% half-widths, leading terms) and
figures/data/09-generalization-pred.dat (finite-n predictions (1/2) E_Z[tbar(sqrt n, Z)^2]), and prints the
product model's prediction at n = 1e20 and 1e100.
Run from the volume root: python figures/scripts/09-generalization-onecoef.py   (about 50 s)
"""
import math
import os
import sys
import time

import numpy as np
from scipy import special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402


def rho_line(c):
    return np.where(np.abs(c) <= 1.0, 0.5, 0.0)


def rho_cubic(c):
    a = np.abs(c)
    with np.errstate(divide="ignore"):
        return np.where(a < 1.0, 0.25 * (a ** (-2.0 / 3.0) - 1.0), 0.0)


# name: (density of c, power p of the substitution t = +-u^p, lambda, d/2)
MODELS = {
    "line": (rho_line, 1, 0.5, 0.5),
    "product": (lambda c: sk.product_density(c, 2), 4, 0.5, 1.0),
    "cusp": (sk.cusp_density, 2, 0.25, 1.0),
    "cubic": (rho_cubic, 3, 1.0 / 6.0, 1.0),
}
NS = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6]
REPS = 1000

_GL_X, _GL_W = special.roots_legendre(24)
_GL_X, _GL_W = 0.5 * (_GL_X + 1.0), 0.5 * _GL_W
_X, _XW = special.roots_hermitenorm(16)
_XW = _XW / _XW.sum()
GX, GXW = _X[_X > 0], 2.0 * _XW[_X > 0]            # KL(x) is even in x
GE, GEW = special.roots_hermitenorm(16)
GEW = GEW / GEW.sum()


def posterior_nodes(model, sigma, Z, panels=4):
    """Nodes t_j and normalized weights of the posterior of t = sigma c (truth c0 = 0)."""
    rho, p, _, _ = MODELS[model]
    T = min(sigma, abs(Z) + 14.0)                    # |c| <= 1; beyond |Z| + 14 the weight is < e^-98
    edges = np.linspace(0.0, T ** (1.0 / p), panels + 1)
    u = np.concatenate([a + (b - a) * _GL_X for a, b in zip(edges[:-1], edges[1:])])
    wu = np.concatenate([(b - a) * _GL_W for a, b in zip(edges[:-1], edges[1:])])
    ts, lws = [], []
    for side in (1.0, -1.0):
        t = side * u ** p
        with np.errstate(divide="ignore"):
            lws.append(np.log(wu * p * u ** (p - 1) * rho(t / sigma)) - 0.5 * (t - Z) ** 2)
        ts.append(t)
    t, lw = np.concatenate(ts), np.concatenate(lws)
    lw -= special.logsumexp(lw)
    keep = lw > -38.0
    return t[keep], np.exp(lw[keep])


def posterior_mean(model, sigma, Z):
    t, w = posterior_nodes(model, sigma, Z)
    return float(w @ t)


def generalization_error(model, sigma, Z):
    """(G_n - S, leading term tbar^2 / (2 sigma^2), posterior mean tbar) for one data set."""
    t, w = posterior_nodes(model, sigma, Z)
    s = GX / sigma
    A = t[:, None, None] * s[None, :, None] * GE[None, None, :] - 0.5 * (t * t)[:, None, None] * (s * s)[None, :, None]
    kl = -(special.logsumexp(A, axis=0, b=w[:, None, None]) @ GEW)      # KL(x) at the x nodes
    tbar = float(w @ t)
    return float(kl @ GXW), 0.5 * tbar * tbar / sigma ** 2, tbar


def flat_line_exact(sigma, Z):
    """E_x KL for the regular line with the flat prior on R (Computation 9.2, before averaging over Z)."""
    v = GX ** 2 / sigma ** 2
    return float((0.5 * (np.log1p(v) + (1.0 + Z * Z * v) / (1.0 + v) - 1.0)) @ GXW)


def main():
    t0 = time.time()
    out_meas, worst_line = {}, 0.0
    for k, model in enumerate(MODELS):
        for n in NS:
            # REPS independent data sets for each model and each n, drawn through (S_xx, Z)
            gen = sk.rng(900 + 10 * k + int(round(math.log10(n))))
            sig = np.sqrt(gen.chisquare(n, REPS))
            zs = gen.standard_normal(REPS)
            G = np.empty(REPS)
            lead = np.empty(REPS)
            for r in range(REPS):
                G[r], lead[r], _ = generalization_error(model, sig[r], zs[r])
                if model == "line":
                    worst_line = max(worst_line, n * abs(G[r] - flat_line_exact(sig[r], zs[r])))
            mG, lo, hi = sk.mean_ci(n * G)
            mL, _, _ = sk.mean_ci(n * lead)
            out_meas[(model, n)] = (mG, hi - mG, mL, float(np.std(n * G, ddof=1)))
    t1 = time.time()

    # finite-n predictions: (1/2) E_Z[tbar(sqrt n, Z)^2] (leading term of Lemma 9.2, S_xx replaced by n)
    zq, wq = special.roots_hermitenorm(80)
    wq = wq / wq.sum()
    grid = np.unique(np.round(np.logspace(2, 6, 17)).astype(int))
    pred_rows = []
    for n in grid:
        row = [int(n)]
        for model in MODELS:
            if model == "line":
                row.append(0.5 * n * (special.digamma((n + 1) / 2.0) - special.digamma(n / 2.0)))
            else:
                tb = np.array([posterior_mean(model, math.sqrt(n), z) for z in zq])
                row.append(0.5 * float(wq @ tb ** 2))
        row.append(0.5 - 1.0 / math.log(n))          # heuristic lambda - (m - 1)/log n for the product model
        pred_rows.append(row)
    far = {}                                         # the product model far beyond any data set (Section 9.5)
    for e in (20, 100):
        tb = np.array([posterior_mean("product", 10.0 ** (e / 2), z) for z in zq])
        far[e] = (0.5 * float(wq @ tb ** 2), 0.5 - 1.0 / (e * math.log(10.0)))
    t2 = time.time()

    header = ["n"]
    for model in MODELS:
        header += [model, model + "ci", model + "lead"]
    rows = []
    for n in NS:
        row = [n]
        for model in MODELS:
            mG, ci, mL, _ = out_meas[(model, n)]
            row += [mG, ci, mL]
        rows.append(row)
    sk.write_dat("figures/data/09-generalization-onecoef.dat", header, rows)
    sk.write_dat("figures/data/09-generalization-pred.dat", ["n"] + list(MODELS) + ["prodheur"], pred_rows)

    print(f"regular line, uniform prior on [-1,1] (quadrature) vs flat prior (closed form), per data set: "
          f"max |difference| in n (G_n - S) = {worst_line:.1e}")
    print("n (E[G_n] - S): mean over", REPS, "data sets, 95% Student-t half-width, mean of the leading term n tbar^2/(2 S_xx), sd")
    for model, (_, _, lam, hd) in MODELS.items():
        print(f"  {model:8s} (lambda = {lam:.4f}, d/2 = {hd}):")
        for n in NS:
            mG, ci, mL, sd = out_meas[(model, n)]
            print(f"     n = 1e{int(round(math.log10(n)))}: {mG:.4f} +- {ci:.4f}   leading {mL:.4f}   sd {sd:.3f}")
    print("closed form for the regular line, n/2 [psi((n+1)/2) - psi(n/2)]:",
          ", ".join(f"1e{int(round(math.log10(n)))}: {0.5 * n * (special.digamma((n + 1) / 2) - special.digamma(n / 2)):.5f}" for n in NS))
    print("finite-n predictions (1/2) E_Z tbar(sqrt n, Z)^2 and the heuristic 1/2 - 1/log n (product):")
    for row in pred_rows:
        if int(round(math.log10(row[0]) * 4)) % 4 == 0:
            print(f"   n = {row[0]:>8d}: line {row[1]:.4f}  product {row[2]:.4f}  cusp {row[3]:.4f}  cubic {row[4]:.4f}  1/2-1/log n {row[5]:.4f}")
    for e, (p, h) in far.items():
        print(f"   n = 1e{e}: product {p:.6f}  1/2-1/log n {h:.6f}")
    print(f"runtime: measurement {t1 - t0:.1f} s, predictions {t2 - t1:.1f} s")


if __name__ == "__main__":
    main()
