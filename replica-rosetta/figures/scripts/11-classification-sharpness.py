"""Chapter 11 (classification): the separability transition sharpens at alpha_MLE(r) as n grows.

Logistic teacher with r = sqrt(5) (Sur and Candes's gamma^2 = 5), predicted alpha_MLE = 3.0714.
For n in {100, 200, 400} and N data sets each (60, 40, 30), the critical dimension d_c of each data set
(the smallest number of leading features for which the data are strictly separable; bisection, one LP
per step, as in 11-classification-threshold.py). The fraction of data sets that are separable at load
alpha = n/d is the fraction with d_c <= d; it is reported on a grid of loads with Clopper--Pearson 95%
intervals. Writes figures/data/11-classification-sharpness.dat. Runtime about a minute.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # shared machine: one BLAS thread per script
os.environ.setdefault("OMP_NUM_THREADS", "1")
import importlib.util
import math
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import rosettakit as rk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "figures", "data")
spec = importlib.util.spec_from_file_location("thr", os.path.join(HERE, "11-classification-threshold.py"))
thr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thr)


def main():
    t0 = time.time()
    r = math.sqrt(5.0)
    h, _ = thr.h_mle(r)
    a_mle = 1.0 / h
    print("r = sqrt(5): alpha_MLE = %.4f" % a_mle)
    gen = rk.rng(11002)
    grid = a_mle * np.array([0.80, 0.86, 0.92, 0.96, 1.00, 1.04, 1.08, 1.14, 1.20, 1.30])
    cols = {}
    for n, N in [(100, 60), (200, 40), (400, 30)]:
        guess = int(round(n * h))
        dc = np.array([thr.critical_dimension(thr.sample_Z(n, n, r, gen), guess) for _ in range(N)])
        ac = n / dc
        res = []
        for a in grid:
            d = n / a                      # separable at load n/d iff d_c <= d
            k = int(np.sum(dc <= d))
            lo, hi = rk.clopper_pearson(k, N)
            res.append((k / N, lo, hi))
        cols[n] = res
        med, lo, hi = thr.median_ci(ac)
        print("n=%d (N=%d): median alpha_c = %.3f [%.3f, %.3f], mean %.3f, sd %.3f   (%.0f s)" % (
            n, N, med, lo, hi, ac.mean(), ac.std(ddof=1), time.time() - t0))
        print("   P(separable) on the grid:", " ".join("%.2f" % p for p, _, _ in res))
    rows = []
    for j, a in enumerate(grid):
        row = [a]
        for n in (100, 200, 400):
            p, lo, hi = cols[n][j]
            row += [p, p - lo, hi - p]
        rows.append(row)
    rk.write_dat(os.path.join(DATA, "11-classification-sharpness.dat"),
                 ["alpha", "p100", "m100", "u100", "p200", "m200", "u200", "p400", "m400", "u400"], rows)
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
