"""Chapter 9 (ridge): a universality check. The same simulation as 09-ridge-sim.py, at n = 500, with
design entries that are standard Gaussian or uniform on {-1, +1} (Rademacher), at alpha = 0.5 and 2,
for the min-norm interpolator and the optimal ridge lam* = sigma^2/(alpha r^2); 12 data sets each.
Besides the risk itself, the script reports its conditional mean given the design, in which the
direction of the teacher (uniform on the sphere, independent of X) and the noise are averaged exactly:
  E[R | X] = sigma^2 tr (X'X + n lam* I)^{-1}                    at lam* (Dicker's identity),
  E[R | X] = r^2 (1 - rank/d) + sigma^2 tr (X'X)^+                for the interpolator.
These identities hold for every design, Gaussian or not; they remove the noise of the teacher and of
the labels and leave only the fluctuation of the spectrum of X.
The limits are those of the chapter's theorem (proved for Gaussian designs); for i.i.d. entries, and the
risk averaged over the noise, they are Proposition 2 (alpha > 1, finite fourth moment), Theorem 1
(alpha < 1, finite (4+eta)-th moment) and Corollary 6 (ridge) of Hastie-Montanari-Rosset-Tibshirani
(journal numbering).
Reports means with Student-t 95% intervals. Seeds: 7770 + j for data set j (the same seeds for both
designs). Writes figures/data/09-ridge-universal.dat. Runtime: about 30 seconds.
Run from the volume root: python figures/scripts/09-ridge-universal.py
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
N = 500
REPS = 12


def chi_closed(alpha, lam):
    b = alpha * lam + alpha - 1.0
    return (-b + math.sqrt(b * b + 4.0 * alpha * lam)) / (2.0 * alpha * lam)


def risk_closed(alpha, lam, r2=R2, s2=S2):
    c = chi_closed(alpha, lam)
    nu = alpha * c * c / (1.0 + c) ** 2
    return (r2 * (alpha * lam * c) ** 2 + s2 * nu) / (1.0 - nu)


def ridgeless(alpha, r2=R2, s2=S2):
    return s2 / (alpha - 1.0) if alpha > 1 else r2 * (1.0 - alpha) + s2 * alpha / (1.0 - alpha)


def one_draw(n, d, ls, design, gen):
    """Risks of the interpolator and of ridge at ls, and their conditional means given X."""
    if design == "gauss":
        X = gen.standard_normal((n, d))
    else:
        X = gen.choice([-1.0, 1.0], size=(n, d))
    u = gen.standard_normal(d)
    beta = math.sqrt(R2) * u / np.linalg.norm(u)
    y = X @ beta + math.sqrt(S2) * gen.standard_normal(n)
    out = []
    for lam in (0.0, ls):
        if d < n:
            M = X.T @ X + n * lam * np.eye(d)
            cf = linalg.cho_factor(M)
            bh = linalg.cho_solve(cf, X.T @ y)
            trinv = float(np.trace(linalg.cho_solve(cf, np.eye(d))))
            bias0 = 0.0
        else:
            M = X @ X.T + n * lam * np.eye(n)
            cf = linalg.cho_factor(M)
            bh = X.T @ linalg.cho_solve(cf, y)
            trinv = float(np.trace(linalg.cho_solve(cf, np.eye(n))))
            if lam > 0:
                trinv += (d - n) / (n * lam)      # the d - n zero eigenvalues of X'X
            bias0 = R2 * (1.0 - n / d)            # E |(I - P) beta*|^2 for a uniform teacher
        R = float(np.sum((bh - beta) ** 2))
        cond = S2 * trinv + bias0 if lam == 0.0 else S2 * trinv
        out.append((R, cond))
    return out


def main():
    t0 = time.time()
    rows = []
    print("Universality check: n = %d, %d data sets, r^2 = %g, sigma^2 = %g" % (N, REPS, R2, S2))
    for alpha in [0.5, 2.0]:
        d = int(round(N / alpha))
        ls = S2 / (alpha * R2)
        lim0, lims = ridgeless(alpha), risk_closed(alpha, ls)
        for k, design in enumerate(["gauss", "rademacher"]):
            R0, C0, Rs, Cs = [], [], [], []
            for j in range(REPS):
                (r0, c0), (rs, cs) = one_draw(N, d, ls, design, rk.rng(7770 + j))
                R0.append(r0); C0.append(c0); Rs.append(rs); Cs.append(cs)
            st = [rk.mean_ci(v) for v in (R0, C0, Rs, Cs)]
            print("alpha=%.1f d=%4d %-10s | interpolator R %.4f +- %.4f, E[R|X] %.4f +- %.4f (limit %.4f)"
                  " | lam*=%.4f R %.4f +- %.4f, E[R|X] %.4f +- %.4f (limit %.4f)" % (
                      alpha, d, design, st[0][0], st[0][1], st[1][0], st[1][1], lim0,
                      ls, st[2][0], st[2][1], st[3][0], st[3][1], lims))
            rows.append([alpha, d, k] + [x for s in st for x in s] + [lim0, lims])
    rk.write_dat("figures/data/09-ridge-universal.dat",
                 ["alpha", "d", "rademacher", "R0", "R0h", "C0", "C0h", "Rs", "Rsh", "Cs", "Csh", "R0lim", "Rslim"], rows)
    print("wrote figures/data/09-ridge-universal.dat; runtime %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
