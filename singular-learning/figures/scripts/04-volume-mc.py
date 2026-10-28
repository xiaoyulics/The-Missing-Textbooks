"""04-volume-mc.py -- Chapter 4 (Volume), "Measuring it": Monte Carlo volume fractions.

For five two-parameter models with the uniform prior on [-1, 1]^2 the script estimates the volume
function V(eps) = phi{K < eps} by plain Monte Carlo (sk.volume_fraction, Clopper-Pearson 95%
intervals) at eps = 10^-1, 10^-1.5, ..., 10^-8, and compares it with the exact volumes of the chapter:

    regular plane  K = (a^2 + b^2)/2   V = (pi/2) eps                        lambda = 1,    m = 1
    quartic cup    K = (a^4 + b^4)/2   V = 1.3110 eps^(1/2)                  lambda = 1/2,  m = 1
    product        K = a^2 b^2 / 2     V = s (1 + log(1/s)),  s = sqrt(2 eps) lambda = 1/2,  m = 2
    quartic        K = (a^2 + b^4)/2   V = 1.4699 eps^(3/4)                  lambda = 3/4,  m = 1
    sextic         K = (a^2 + b^6)/2   V = 1.4457 eps^(2/3)                  lambda = 2/3,  m = 1

(all exact for eps <= 1/2). Plain Monte Carlo has relative standard error about (N V)^(-1/2), so
10% accuracy at level eps needs about 100 / V(eps) draws; the numbers of draws below are chosen so
that every singular model keeps at least about 100 draws at eps = 1e-8, which takes 10^8 draws for
the quartic model; the regular plane gets 10^8 draws too, which is not enough below eps ~ 1e-6.

Outputs (figures/data/):
  04-volume-mc-<model>.dat     eps V lo hi ep em count       (rows with count >= 1)
  04-volume-exact.dat          exact V and exact local slopes d log V / d log eps on a fine grid
  04-volume-slope-<model>.dat  local slopes between neighbouring eps with exact conditional
                               Clopper-Pearson 95% intervals (rows with count >= 10)
  04-volume-fits.dat           fitted exponents with parametric-bootstrap 95% intervals
Run from the volume root: python figures/scripts/04-volume-mc.py   (about 60 s)
"""
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

OUT = "figures/data"
G = math.gamma


def sumpow_C(ks):
    """V(eps) = C eps^lam for K = (1/2) sum_j w_j^(2 k_j), uniform prior on [-1,1]^d, eps <= 1/2."""
    lam = sum(1.0 / (2 * k) for k in ks)
    return lam, 2.0 ** lam * np.prod([G(1 + 1.0 / (2 * k)) for k in ks]) / G(1 + lam)


LAM_CUP, C_CUP = sumpow_C((2, 2))
LAM_QUA, C_QUA = sumpow_C((1, 2))
LAM_SEX, C_SEX = sumpow_C((1, 3))


def v_product(eps):
    s = np.sqrt(2.0 * np.asarray(eps, dtype=float))
    return s * (1.0 + np.log(1.0 / s))


MODELS = [
    # key, K, exact V, lambda, m, draws, seed
    ("reg", lambda w: 0.5 * (w[:, 0] ** 2 + w[:, 1] ** 2), lambda e: math.pi / 2 * e, 1.0, 1, 100_000_000, 401),
    ("cup", lambda w: 0.5 * (w[:, 0] ** 4 + w[:, 1] ** 4), lambda e: C_CUP * e ** 0.5, 0.5, 1, 10_000_000, 402),
    ("prod", lambda w: 0.5 * (w[:, 0] * w[:, 1]) ** 2, v_product, 0.5, 2, 10_000_000, 403),
    ("quart", lambda w: 0.5 * (w[:, 0] ** 2 + w[:, 1] ** 4), lambda e: C_QUA * e ** 0.75, 0.75, 1, 100_000_000, 404),
    ("sext", lambda w: 0.5 * (w[:, 0] ** 2 + w[:, 1] ** 6), lambda e: C_SEX * e ** (2 / 3), 2 / 3, 1, 30_000_000, 405),
]
BOX = [(-1.0, 1.0), (-1.0, 1.0)]
EXPS = np.arange(1.0, 8.01, 0.5)
EPS = 10.0 ** -EXPS                      # decreasing: 1e-1, ..., 1e-8
B_BOOT = 2000


def exact_slope(key, e):
    """d log V / d log eps for the exact volumes (eps <= 1/2)."""
    if key == "prod":
        return 0.5 - 1.0 / (2.0 + np.log(1.0 / (2.0 * e)))
    return {"reg": 1.0, "cup": 0.5, "quart": 0.75, "sext": 2 / 3}[key] * np.ones_like(e)


def fit(eps, V, w, m):
    """Weighted least squares of log V - (m-1) log log(1/eps) on log eps; returns the slope."""
    y = np.log(V) - (m - 1) * np.log(np.log(1.0 / eps))
    X = np.vstack([np.log(eps), np.ones_like(eps)]).T
    W = np.sqrt(w)
    coef = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
    return float(coef[0])


def fit_full(eps, V, w, m, start):
    """Weighted nonlinear least squares of log V on lambda log eps + log P(log 1/eps), with P a
    polynomial of degree m - 1 with free coefficients: the exact form of the volume of a monomial
    with a constant prior factor (Step 2 of the proof of prop:volume:monomial). Returns lambda."""
    L = np.log(1.0 / eps)
    W = np.sqrt(w)

    def resid(p):
        poly = sum(p[1 + i] * L ** i for i in range(m))
        return W * (np.log(V) - p[0] * np.log(eps) - np.log(np.maximum(poly, 1e-300)))

    sol = least_squares(resid, start, method="lm", xtol=1e-12, ftol=1e-12)
    return float(sol.x[0]), sol.x


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    gen = sk.rng(499)
    fit_rows = []
    print("eps grid:", " ".join(f"{e:.3g}" for e in EPS))
    for key, K, Vex, lam, m, N, seed in MODELS:
        t1 = time.time()
        rows = sk.volume_fraction(K, BOX, EPS, N, seed)
        counts = rows[:, 4].astype(np.int64)
        V, lo, hi = rows[:, 1], rows[:, 2], rows[:, 3]
        ex = np.array([Vex(e) for e in EPS])
        inside = (counts > 0)
        sk.write_dat(f"{OUT}/04-volume-mc-{key}.dat", ["eps", "V", "lo", "hi", "ep", "em", "count", "exact"],
                     [(EPS[i], V[i], lo[i], hi[i], hi[i] - V[i], V[i] - lo[i], counts[i], ex[i])
                      for i in range(len(EPS)) if inside[i]])
        cover = np.mean((lo <= ex) & (ex <= hi))
        print(f"\n[{key}] N = {N:.0e} draws, seed {seed}, {time.time() - t1:.1f} s; "
              f"exact value inside the 95% interval at {int(round(cover * len(EPS)))} of {len(EPS)} levels")
        for i in range(len(EPS)):
            print(f"   eps={EPS[i]:.3g}  count={counts[i]:>11d}  V={V[i]:.4e}  [{lo[i]:.4e}, {hi[i]:.4e}]  "
                  f"exact={ex[i]:.4e}  ratio={V[i] / ex[i] if V[i] > 0 else float('nan'):.4f}  "
                  f"draws for 10%: {100 * (1 - ex[i]) / ex[i]:.2e}")

        # local slopes with exact conditional Clopper-Pearson intervals: given count c_k at eps_k, the
        # count at eps_{k+1} < eps_k is Binomial(c_k, V(eps_{k+1}) / V(eps_k)), because the sets are nested
        srows = []
        for k in range(len(EPS) - 1):
            ck, ck1 = int(counts[k]), int(counts[k + 1])
            if ck1 < 10:
                continue
            q = ck1 / ck
            qlo, qhi = sk.clopper_pearson(ck1, ck)
            dl = math.log(EPS[k + 1] / EPS[k])
            mid = math.sqrt(EPS[k] * EPS[k + 1])
            s, slo, shi = math.log(q) / dl, math.log(qhi) / dl, math.log(qlo) / dl
            srows.append((mid, s, slo, shi, shi - s, s - slo, exact_slope(key, np.array([mid]))[0],
                          math.log(Vex(EPS[k + 1]) / Vex(EPS[k])) / dl))
        sk.write_dat(f"{OUT}/04-volume-slope-{key}.dat",
                     ["eps", "slope", "lo", "hi", "ep", "em", "exactmid", "exactchord"], srows)
        inside_s = sum(1 for r in srows if r[2] <= r[7] <= r[3])
        print(f"   local slopes: exact chord slope inside the 95% interval for {inside_s} of {len(srows)}")
        for r in srows:
            print(f"     eps~{r[0]:.3g}: slope {r[1]:.4f} [{r[2]:.4f}, {r[3]:.4f}]  exact chord {r[7]:.4f}")

        # fitted exponents over the levels 1e-8 <= eps <= 1e-2 that hold at least 100 draws, with a
        # parametric bootstrap: the counts of the shells between neighbouring levels are multinomial
        sel = (EPS <= 1e-2 + 1e-15) & (counts >= 100)
        e_fit = EPS[sel]
        shells = np.diff(np.concatenate([[0], counts[::-1]]))[::-1]   # shell counts, outermost first
        p_shell = np.append(shells, N - counts[0]) / N
        naive = fit(e_fit, V[sel], counts[sel], 1)
        withm = fit(e_fit, V[sel], counts[sel], m)
        start = np.concatenate([[naive], np.exp(np.polyfit(np.log(e_fit), np.log(V[sel]), 1))[1:], np.zeros(m - 1)])
        full, pfull = fit_full(e_fit, V[sel], counts[sel], m, start)
        boots_n, boots_m, boots_f = [], [], []
        for _ in range(B_BOOT):
            draw = gen.multinomial(N, p_shell)
            c_star = np.cumsum(draw[:-1][::-1])[::-1]
            if np.any(c_star[sel] == 0):
                continue
            v_star = c_star / N
            boots_n.append(fit(e_fit, v_star[sel], c_star[sel], 1))
            boots_m.append(fit(e_fit, v_star[sel], c_star[sel], m))
            boots_f.append(fit_full(e_fit, v_star[sel], c_star[sel], m, pfull)[0] if m > 1 else boots_n[-1])
        qn = np.quantile(boots_n, [0.025, 0.975])
        qm = np.quantile(boots_m, [0.025, 0.975])
        qf = np.quantile(boots_f, [0.025, 0.975])
        print(f"   fit over eps in [{e_fit.min():.3g}, {e_fit.max():.3g}] ({sel.sum()} levels): "
              f"naive slope {naive:.4f} [{qn[0]:.4f}, {qn[1]:.4f}]; with (log 1/eps)^(m-1), m={m}: {withm:.4f} "
              f"[{qm[0]:.4f}, {qm[1]:.4f}]; full form eps^lam poly_(m-1)(log 1/eps): {full:.4f} "
              f"[{qf[0]:.4f}, {qf[1]:.4f}] (coefficients {np.array2string(pfull[1:], precision=4)}); "
              f"lambda = {lam:.4f}  ({len(boots_m)} bootstrap fits)")
        fit_rows.append((key, lam, m, N, e_fit.min(), e_fit.max(), naive, qn[0], qn[1], withm, qm[0], qm[1],
                         full, qf[0], qf[1]))

    sk.write_dat(f"{OUT}/04-volume-fits.dat",
                 ["model", "lambda", "m", "draws", "epsmin", "epsmax", "naive", "naivelo", "naivehi",
                  "withm", "withmlo", "withmhi", "full", "fulllo", "fullhi"], fit_rows)

    # exact curves on a fine grid (eps <= 1/2, where all five formulas are exact)
    grid = 10.0 ** np.linspace(-8.5, -0.5, 161)
    ex_rows = []
    for e in grid:
        ex_rows.append((e, math.pi / 2 * e, C_CUP * e ** 0.5, float(v_product(e)), C_QUA * e ** 0.75,
                        C_SEX * e ** (2 / 3), float(exact_slope("prod", np.array([e]))[0])))
    sk.write_dat(f"{OUT}/04-volume-exact.dat", ["eps", "reg", "cup", "prod", "quart", "sext", "sprod"], ex_rows)
    print("\nconstants: C_cup = %.6f, C_quartic = %.6f, C_sextic = %.6f" % (C_CUP, C_QUA, C_SEX))
    print("product: exact local slope at 1e-2, 1e-4, 1e-8: %.4f %.4f %.4f" %
          tuple(exact_slope("prod", np.array([1e-2, 1e-4, 1e-8]))))
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
