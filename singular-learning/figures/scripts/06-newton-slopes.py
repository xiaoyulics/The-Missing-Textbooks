"""Chapter 6 (Newton Polygons): what a slope of the volume function measures at practical eps.

Three positive even sums, uniform prior on [-1, 1]^2 (so V(eps) = Leb{(a, b) in (0, 1]^2 : K < eps}):
    E1  K = (a^4 + b^6)/2         (the model y = a^2 x1 + b^3 x2)   lambda = 5/12, m = 1, V exactly a power
    E2  K = a^6 + a^2 b^2 + b^6   (a vertex on the diagonal)         lambda = 1/2,  m = 2
    E3  K = a^4 b^6 + a^6 b^4     (a second pole close by, at -1/4)  lambda = 1/5,  m = 1
The local slope s(eps) = d log V / d log eps tends to lambda; Newton's polygon predicts lambda, and the theorem's
asymptotics V ~ C eps^lambda (log 1/eps)^(m-1) predict s = lambda - (m - 1)/log(1/eps).
  * exact local slopes: log V by the quadrature of 06-newton-random.py (logarithmic coordinates), central
    differences in L = log(1/eps) with step 0.25, at log10(1/eps) from 0.5 to 40;
  * Monte Carlo: N = 4 000 000 uniform points of (0, 1]^2 per function (seeds 61, 62, 63); the slope over each
    decade [10^-(k+1), 10^-k], k = 1, ..., 5, is log(n_k / n_(k+1)) / log 10, with the exact 95% interval of the
    binomial proportion n_(k+1)/n_k (Clopper-Pearson); plotted at log10(1/eps) = k + 1/2.
Output: figures/data/06-newton-slopes-curves.dat, figures/data/06-newton-slopes-mc.dat.
Run from the volume root:  python figures/scripts/06-newton-slopes.py   (about half a minute)
"""
import importlib.util
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import sltkit as sk  # noqa: E402

# the quadrature and Newton routines of the companion script
_spec = importlib.util.spec_from_file_location(
    "newton_random", os.path.join(os.path.dirname(os.path.abspath(__file__)), "06-newton-random.py"))
nr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nr)

OUT = os.path.join("figures", "data")
FUNCS = [("E1", {(4, 0): 0.5, (0, 6): 0.5}),
         ("E2", {(6, 0): 1.0, (2, 2): 1.0, (0, 6): 1.0}),
         ("E3", {(4, 6): 1.0, (6, 4): 1.0})]


def main():
    t0 = time.time()
    newt = [nr.newton(Sc) for _, Sc in FUNCS]
    for (name, Sc), (t, m, C, V) in zip(FUNCS, newt):
        print(f"{name}: vertices {V}, t = {t}, lambda = 1/t = {1 / float(t):.6f}, m = {m}, "
              f"V ~ {C:.6f} eps^lambda L^(m-1)")
    decs = np.concatenate([np.arange(0.5, 10.0, 0.25), np.arange(10.0, 40.01, 1.0)])
    rows = []
    h = 0.25
    for d in decs:
        L = d * math.log(10)
        row = [d]
        for (name, Sc) in FUNCS:
            s = -(nr.logV(Sc, L + h) - nr.logV(Sc, L - h)) / (2 * h)
            row.append(s)
        for (t, m, C, V) in newt:
            lam = 1 / float(t)
            row.append(lam - (m - 1) / L)
        rows.append(row)
    sk.write_dat(os.path.join(OUT, "06-newton-slopes-curves.dat"),
                 ["dec", "s1", "s2", "s3", "p1", "p2", "p3"], rows)
    R = np.array(rows)
    for j, (name, _) in enumerate(FUNCS):
        for d in (1.5, 3.5, 5.5, 10.0, 20.0, 40.0):
            i = int(np.argmin(np.abs(R[:, 0] - d)))
            print(f"  {name}: exact local slope at eps = 1e-{R[i, 0]:g}: {R[i, 1 + j]:.4f} "
                  f"(asymptotic prediction {R[i, 4 + j]:.4f})")
    # Monte Carlo slopes over decades
    N = 4_000_000
    mc = []
    ks = [1, 2, 3, 4, 5]
    cols = {}
    for j, (name, Sc) in enumerate(FUNCS):
        g = sk.rng(61 + j)
        a, b = g.random(N), g.random(N)
        Kv = nr.K_eval(Sc, a, b)
        counts = [int(np.count_nonzero(Kv < 10.0 ** (-k))) for k in ks + [ks[-1] + 1]]
        vals = []
        for i, k in enumerate(ks):
            n1, n2 = counts[i], counts[i + 1]
            est = math.log(n1 / n2) / math.log(10)
            lo_r = stats.beta.ppf(0.025, n2, n1 - n2 + 1)
            hi_r = stats.beta.ppf(0.975, n2 + 1, n1 - n2)
            lo, hi = -math.log10(hi_r), -math.log10(lo_r)
            vals.append((est, hi - est, est - lo))
            print(f"  {name}: MC slope over [1e-{k + 1}, 1e-{k}] = {est:.4f} [{lo:.4f}, {hi:.4f}] "
                  f"(counts {n1}, {n2})")
        cols[name] = vals
    for i, k in enumerate(ks):
        row = [k + 0.5]
        for name, _ in FUNCS:
            row.extend(cols[name][i])
        mc.append(row)
    sk.write_dat(os.path.join(OUT, "06-newton-slopes-mc.dat"),
                 ["dec", "m1", "e1p", "e1m", "m2", "e2p", "e2m", "m3", "e3p", "e3m"], mc)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
