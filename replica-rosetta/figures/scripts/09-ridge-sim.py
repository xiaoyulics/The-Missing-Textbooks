"""Chapter 9 (ridge): simulation at n = 1000 against the limiting risk.

For each load alpha = n/d on a grid from 0.25 to 5 (d = round(n/alpha)), and each repetition:
  X with i.i.d. N(0,1) entries, beta* = r u with u uniform on the sphere, eps ~ N(0, I_n),
  y = X beta* + sigma eps;
  ridge at lam = 0.01 and at lam* = sigma^2/(alpha r^2), and the min-norm interpolator (lam -> 0),
  each computed by a Cholesky solve of the smaller Gram matrix (X'X if d < n, XX' if d > n);
  measured: the excess risk R = |beta_hat - beta*|^2, the training error |y - X beta_hat|^2/n, and the
  resolvent trace chi_n = tr (X'X + n lam* I)^{-1} (the replica susceptibility).
Reports means with Student-t 95% intervals. Seeds: 90000 + 100 k + j for load k, repetition j.
Writes figures/data/09-ridge-sim.dat. Runtime: about one minute (dominated by the Gram matrices).
Run from the volume root: python figures/scripts/09-ridge-sim.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk

R2, S2 = 1.0, 0.25
LAM_SMALL = 0.01
N = 1000
REPS = 8
ALPHAS = [0.25, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def chi_closed(alpha, lam):
    b = alpha * lam + alpha - 1.0
    return (-b + math.sqrt(b * b + 4.0 * alpha * lam)) / (2.0 * alpha * lam)


def risk_closed(alpha, lam, r2=R2, s2=S2):
    c = chi_closed(alpha, lam)
    A = alpha * c * c / (1.0 + c) ** 2
    R = (r2 * (alpha * lam * c) ** 2 + s2 * A) / (1.0 - A)
    return R, c, (R + s2) / (1.0 + c) ** 2


def ridgeless(alpha, r2=R2, s2=S2):
    if alpha > 1:
        return s2 / (alpha - 1.0), s2 * (1.0 - 1.0 / alpha)
    return r2 * (1.0 - alpha) + s2 * alpha / (1.0 - alpha), 0.0


def ridgeless_exact_mean(n, d, r2=R2, s2=S2):
    if n >= d + 2:
        return s2 * d / (n - d - 1.0)
    return r2 * (1.0 - n / d) + s2 * n / (d - n - 1.0)


def one_draw(n, d, lams, gen):
    """One data set; returns, for each lam in lams (0 means the min-norm interpolator), the excess risk
    and the training error, and the resolvent trace tr (X'X + n lams[-1] I)^{-1}."""
    X = gen.standard_normal((n, d))
    u = gen.standard_normal(d)
    beta = math.sqrt(R2) * u / np.linalg.norm(u)
    y = X @ beta + math.sqrt(S2) * gen.standard_normal(n)
    out = []
    chi_n = float("nan")
    if d < n:
        C = X.T @ X
        Xty = X.T @ y
        for lam in lams:
            M = C + n * lam * np.eye(d)
            cf = linalg.cho_factor(M)
            bh = linalg.cho_solve(cf, Xty)
            out.append((float(np.sum((bh - beta) ** 2)), float(np.sum((y - X @ bh) ** 2)) / n))
            if lam == lams[-1]:
                chi_n = float(np.trace(linalg.cho_solve(cf, np.eye(d))))
    else:
        K = X @ X.T
        for lam in lams:
            M = K + n * lam * np.eye(n)
            cf = linalg.cho_factor(M)
            a = linalg.cho_solve(cf, y)
            bh = X.T @ a
            resid = n * lam * a                       # y - X bh = y - K a = n lam a
            out.append((float(np.sum((bh - beta) ** 2)), float(np.sum(resid ** 2)) / n))
            if lam == lams[-1]:
                # nonzero spectra of X'X and XX' agree; the d - n zero eigenvalues of X'X add (d-n)/(n lam)
                chi_n = float(np.trace(linalg.cho_solve(cf, np.eye(n)))) + (d - n) / (n * lam)
    return out, chi_n


def main():
    t0 = time.time()
    print("Chapter 9 simulation: n = %d, %d repetitions per load, r^2 = %g, sigma^2 = %g" % (N, REPS, R2, S2))
    rows = []
    for k, alpha in enumerate(ALPHAS):
        d = int(round(N / alpha))
        a_eff = N / d                                 # the load actually simulated
        ls = S2 / (a_eff * R2)
        lams = [0.0, LAM_SMALL, ls]
        Rs = [[] for _ in lams]
        Ts = [[] for _ in lams]
        chis = []
        for j in range(REPS):
            out, chi_n = one_draw(N, d, lams, rk.rng(90000 + 100 * k + j))
            for i, (R, T) in enumerate(out):
                Rs[i].append(R)
                Ts[i].append(T)
            chis.append(chi_n)
        stats = []
        for i in range(len(lams)):
            mR, hR = rk.mean_ci(Rs[i])
            mT, hT = rk.mean_ci(Ts[i])
            stats += [mR, hR, mT, hT]
        mc, hc = rk.mean_ci(chis)
        th0 = ridgeless(a_eff)
        th1 = risk_closed(a_eff, LAM_SMALL)
        ths = risk_closed(a_eff, ls)
        print("alpha=%.4f d=%4d | ridgeless R %.4f +- %.4f (limit %.4f, exact mean %.4f), T %.4f (limit %.4f)"
              " | lam=0.01 R %.4f +- %.4f (limit %.4f), T %.4f +- %.4f (limit %.4f)"
              " | lam*=%.4f R %.4f +- %.4f (limit %.4f), T %.4f +- %.4f (limit %.4f) | chi_n %.4f +- %.4f (chi %.4f)  [%.0f s]" % (
                  a_eff, d, stats[0], stats[1], th0[0], ridgeless_exact_mean(N, d), stats[2], th0[1],
                  stats[4], stats[5], th1[0], stats[6], stats[7], th1[2],
                  ls, stats[8], stats[9], ths[0], stats[10], stats[11], ths[2], mc, hc, ths[1], time.time() - t0))
        rows.append([a_eff, d] + stats + [mc, hc, ths[1], ridgeless_exact_mean(N, d)])
    rk.write_dat("figures/data/09-ridge-sim.dat",
                 ["alpha", "d", "R0", "R0h", "T0", "T0h", "R1", "R1h", "T1", "T1h", "Rs", "Rsh", "Ts", "Tsh",
                  "chin", "chinh", "chi", "R0fin"], rows)
    print("wrote figures/data/09-ridge-sim.dat (%d rows); runtime %.1f s" % (len(rows), time.time() - t0))


if __name__ == "__main__":
    main()
