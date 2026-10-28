"""Chapter 2, figure fig:rem:gibbs(b): the replica moment function varphi(n) = lim N^-1 log E Z_N^n at beta = 2.

Exact finite-N values of (Nn)^-1 log E Z_N^n (Laplace transform for 0 < n < 1, set partitions for
n = 1, 2, 3; 02-rem-laplace.py) at N = 12, 20, 80, 320, against: the limit varphi(n)/n of theorem thm:rem:moments
(equal to the 1RSB formula with Parisi's rule), the continuation of the integer formula,
log 2/n + n beta^2/2, and the replica-symmetric value log 2 + beta^2/2. Sampling estimate at N = 12:
(Nn)^-1 log of the empirical mean of Z^n over 20000 samples (seed 2201), with percentile-bootstrap
95% intervals (400 resamples, seed 2202).
Writes figures/data/02-rem-moments-{curve,exact,sim}.dat. Runtime about 40 s.
"""
import importlib.util
import math
import os
import sys
import time

import numpy as np
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rosettakit as rk  # noqa: E402

spec = importlib.util.spec_from_file_location("rem_laplace", os.path.join(HERE, "02-rem-laplace.py"))
lap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lap)

DATA = os.path.join(HERE, "..", "data")
BC, LOG2 = lap.BETA_C, lap.LOG2
beta = 2.0
m = BC / beta
t0 = time.time()

# curves of the limit and of the two continuations
ns = np.concatenate([np.linspace(0.02, 3.0, 150), [m]])
ns.sort()
rows = [[n, lap.phi_moment(n, beta) / n, LOG2 / n + n * beta * beta / 2, LOG2 + beta * beta / 2] for n in ns]
rk.write_dat(os.path.join(DATA, "02-rem-moments-curve.dat"), ["n", "limit", "integer", "rs"], rows)
print(f"beta = {beta}, m = beta_c/beta = {m:.4f}; limit varphi(n)/n = beta beta_c = {beta * BC:.5f} for n <= m")

# exact finite-N values
nfr = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
rows = []
for n in nfr + [1, 2, 3]:
    row = [n]
    for N in [12, 20, 80, 320]:
        lv = lap.log_EZn_frac(n, beta, N) if n < 1 else lap.log_EZ_int(int(n), beta, N)
        row.append(lv / (N * n))
    row.append(lap.phi_moment(n, beta) / n)
    rows.append(row)
rk.write_dat(os.path.join(DATA, "02-rem-moments-exact.dat"), ["n", "N12", "N20", "N80", "N320", "limit"], rows)
print("exact (Nn)^-1 log E Z_N^n:  N=12, N=20, N=80, N=320 | limit")
for r in rows:
    print(f"  n={r[0]:.2f}: {r[1]:.4f} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} | {r[5]:.4f}")
print(f"exact values done ({time.time() - t0:.0f} s)")

# sampling estimate at N = 12
N, R = 12, 20000
gen = rk.rng(2201)
logZ = np.array([special.logsumexp(beta * math.sqrt(N) * gen.standard_normal(2 ** N)) for _ in range(R)])
boot = rk.rng(2202)
idx = boot.integers(0, R, size=(400, R))
rows = []
for n in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]:
    est = (special.logsumexp(n * logZ) - math.log(R)) / (N * n)
    bs = (special.logsumexp(n * logZ[idx], axis=1) - math.log(R)) / (N * n)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    exact = (lap.log_EZn_frac(n, beta, N) if n < 1 else lap.log_EZ_int(int(n), beta, N)) / (N * n)
    rows.append([n, est, est - lo, hi - est, exact])
rk.write_dat(os.path.join(DATA, "02-rem-moments-sim.dat"), ["n", "est", "lo", "hi", "exact"], rows)
print(f"sampling at N = {N} ({R} samples): estimate [bootstrap 95%] vs exact")
for r in rows:
    print(f"  n={r[0]:.1f}: {r[1]:.4f} [-{r[2]:.4f}, +{r[3]:.4f}]   exact {r[4]:.4f}")
print(f"  typical value N^-1 E log Z_{N} = {lap.E_logZ(beta, N) / N:.4f}; sample mean {logZ.mean() / N:.4f}")
print(f"done in {time.time() - t0:.0f} s")
