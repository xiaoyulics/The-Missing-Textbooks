"""Chapter 7 (glm): the version space of the Ising teacher, by exact enumeration.

Ising teacher w* uniform on {-1,1}^d (by symmetry fixed to the all-ones vector), patterns
x_mu ~ N(0, I_d), labels y_mu = sgn<x_mu, w*>, mu <= n = round(alpha d).  The version space
V = {w in {-1,1}^d : sgn<x_mu, w> = y_mu for all mu} is found by enumerating the 2^d sign vectors and
filtering them pattern by pattern.  The posterior is uniform on V, so (1/d) log|V| is the posterior
entropy per coordinate (the free entropy of the chapter plus log 2).
Rigorous finite-d bounds (Theorem "version space", proved in the chapter):
    (1 - n/d) log 2  <=  (1/d) E log|V|  <=  (1/d) log E|V| = (1/d) log sum_k C(d,k) p_k^n,
    p_k = 1 - arccos(1 - 2k/d)/pi.
Outputs: figures/data/07-glm-versionspace.dat (alpha, d, n, x = n/d, mean entropy with 95% interval, fraction of
instances with V = {w*} with Clopper--Pearson interval, the two finite-d bounds).
Seeds 50000 + 1000 j + 10 k (j = size index, k = alpha index).  Runtime: about 60 s.
"""
import math
import os
import sys
import time

import numpy as np
from scipy import special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")

SIZES = [(10, 1500), (14, 600), (18, 250)]     # (d, number of instances per alpha)
ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]


def all_vectors(d):
    idx = np.arange(2 ** d, dtype=np.int64)[:, None]
    return (1 - 2 * ((idx >> np.arange(d)) & 1)).astype(np.float32)


def version_space_size(W, X):
    """Number of rows w of W with sgn<x_mu, w> = sgn<x_mu, 1> for all rows x_mu of X."""
    surv = W
    for x in X:
        y = 1.0 if x.sum() > 0 else -1.0
        keep = (surv @ x) * y > 0
        surv = surv[keep]
        if surv.shape[0] == 1:
            break
    return surv.shape[0]


def annealed_bound(d, n):
    k = np.arange(d + 1)
    p = 1 - np.arccos(np.clip(1 - 2 * k / d, -1, 1)) / math.pi
    with np.errstate(divide="ignore"):
        logs = special.gammaln(d + 1) - special.gammaln(k + 1) - special.gammaln(d - k + 1) + n * np.log(p)
    return float(special.logsumexp(logs)) / d


rows = []
for j, (d, reps) in enumerate(SIZES):
    W = all_vectors(d)
    for k, a in enumerate(ALPHAS):
        n = int(round(a * d))
        gen = rk.rng(50000 + 1000 * j + 10 * k)
        ent, single = [], 0
        for r in range(reps):
            X = gen.standard_normal((n, d)).astype(np.float32)
            m = version_space_size(W, X)
            ent.append(math.log(m) / d)
            single += int(m == 1)
        me, he = rk.mean_ci(ent)
        lo, hi = rk.clopper_pearson(single, reps)
        lb = max(0.0, (1 - n / d) * math.log(2))
        ub = annealed_bound(d, n)
        ok = (me + he >= lb - 1e-12) and (me - he <= ub + 1e-12)
        rows.append([a, d, n, n / d, me, he, single / reps, single / reps - lo, hi - single / reps, lb, ub])
        print(f"  d={d:2d} alpha={a:4.2f} n={n:2d}: (1/d)E log|V| = {me:.4f} +- {he:.4f}   P(V={{w*}}) = {single / reps:.3f}"
              f"   bounds [{lb:.4f}, {ub:.4f}] {'ok' if ok else 'VIOLATED'}")
    print(f"   [t = {time.time() - T0:.1f} s]")

rk.write_dat(os.path.join(DATA, "07-glm-versionspace.dat"),
             ["alpha", "d", "n", "x", "ent", "entci", "psingle", "pslo", "pshi", "lower", "upper"], rows)
print(f"runtime {time.time() - T0:.1f} s")
