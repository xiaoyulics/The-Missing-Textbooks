"""Chapter 2, figure fig:rem:freeentropy: the free entropy of the REM against Derrida's formula.

(a) phi_N(beta) = N^-1 E log Z_N against beta: simulation at N = 20 (150 samples, seed 2001),
    exact finite-N values (Laplace transform, 02-rem-laplace.py) at N = 20, 80, 320, the limit
    phi(beta) (theorem thm:rem:freeentropy), the annealed (replica-symmetric) value log 2 + beta^2/2, and the proved
    upper bound of theorem thm:rem:freeentropy at N = 20.
(b) N (phi(beta) - phi_N(beta)) at beta = 2 against N: exact values for 10 <= N <= 1280, simulation
    for 10 <= N <= 20 (seeds 2002 + N), and the extreme-value asymptote from Bovier--Kurkova--Lowe
    (2002, Thm 1.5): (1/2m)[log(N log 2) + log 4 pi] - C_m, C_m = log Gamma(1-m)/m + gamma (1/m - 1).
Writes figures/data/02-rem-freeentropy-{curve,sim,size,sizesim}.dat. Runtime about 60 s.
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
BC = lap.BETA_C
t0 = time.time()


def log_Z(X32, beta, N):
    """log sum_s exp(beta sqrt(N) X_s) for one sample (float32 exponentials, float64 sums;
    the rounding error in log Z is below 1e-5)."""
    y = np.float32(beta * math.sqrt(N)) * X32
    mx = float(y.max())
    return mx + math.log(float(np.exp(y - np.float32(mx)).sum(dtype=np.float64)))


def upper_bound(beta, N):
    """Theorem thm:rem:freeentropy: phi_N <= log 2 + beta^2/2, and phi_N <= beta beta_c + log(3/2)/(m N) for beta > beta_c."""
    if beta <= BC:
        return math.log(2) + beta * beta / 2
    return beta * BC + math.log(1.5) * beta / (BC * N)


# (a) curves against beta
betas = sorted([round(0.1 * k, 4) for k in range(1, 31)] + [BC])
rows = []
for b in betas:
    rows.append([b, lap.phi_rem(b), math.log(2) + b * b / 2, lap.E_logZ(b, 20) / 20, lap.E_logZ(b, 80) / 80,
                 lap.E_logZ(b, 320) / 320, upper_bound(b, 20)])
rk.write_dat(os.path.join(DATA, "02-rem-freeentropy-curve.dat"),
             ["beta", "limit", "annealed", "exact20", "exact80", "exact320", "bound20"], rows)
print(f"exact curves done ({time.time() - t0:.0f} s)")

# (a) simulation at N = 20
N, R = 20, 150
sim_betas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
gen = rk.rng(2001)
vals = np.zeros((R, len(sim_betas)))
for r in range(R):
    X32 = gen.standard_normal(2 ** N).astype(np.float32)
    for j, b in enumerate(sim_betas):
        vals[r, j] = log_Z(X32, b, N) / N
rows = []
for j, b in enumerate(sim_betas):
    mu, half = rk.mean_ci(vals[:, j])
    rows.append([b, mu, half, lap.E_logZ(b, N) / N, lap.phi_rem(b)])
rk.write_dat(os.path.join(DATA, "02-rem-freeentropy-sim.dat"), ["beta", "mean", "ci", "exact", "limit"], rows)
print("N = 20, 150 samples: simulated phi_N (95% half-width) vs exact finite N and the limit:")
inside = 0
for row in rows:
    ok = abs(row[1] - row[3]) <= row[2]
    inside += ok
    print(f"  beta={row[0]:.2f}: sim {row[1]:.4f} +- {row[2]:.4f}   exact {row[3]:.4f}   limit {row[4]:.4f}   {'in' if ok else 'OUT'}")
print(f"  {inside} of {len(rows)} intervals contain the exact value")
print(f"simulation done ({time.time() - t0:.0f} s)")

# (b) size dependence at beta = 2
beta = 2.0
m = BC / beta
Cm = math.log(special.gamma(1 - m)) / m + lap.EULER * (1 / m - 1)
Ns = [10, 14, 20, 28, 40, 57, 80, 113, 160, 226, 320, 453, 640, 905, 1280]
rows = []
for n_ in Ns:
    gap = n_ * lap.phi_rem(beta) - lap.E_logZ(beta, n_)
    asym = (math.log(n_ * math.log(2)) + math.log(4 * math.pi)) / (2 * m) - Cm
    rows.append([n_, gap, asym])
rk.write_dat(os.path.join(DATA, "02-rem-freeentropy-size.dat"), ["N", "gap", "asym"], rows)
print(f"beta = 2, m = {m:.4f}, C_m = {Cm:.4f}, slope 1/(2m) = {1 / (2 * m):.4f}")
print(f"  proved lower bound on N(phi - phi_N): -log(3/2)/m = {-math.log(1.5) / m:.4f}")
for row in rows:
    print(f"  N={row[0]:5d}: exact N(phi - phi_N) {row[1]:.4f}   asymptote {row[2]:.4f}   difference {row[1] - row[2]:.4f}")

rows = []
j2 = sim_betas.index(2.0)
for n_ in [10, 12, 14, 16, 18, 20]:
    if n_ == 20:
        sample = vals[:, j2] * 20
    else:
        g = rk.rng(2002 + n_)
        reps = 1000 if n_ <= 16 else 400
        sample = np.array([log_Z(g.standard_normal(2 ** n_).astype(np.float32), beta, n_) for _ in range(reps)])
    mu, half = rk.mean_ci(sample)
    rows.append([n_, n_ * lap.phi_rem(beta) - mu, half, len(sample), n_ * lap.phi_rem(beta) - lap.E_logZ(beta, n_)])
rk.write_dat(os.path.join(DATA, "02-rem-freeentropy-sizesim.dat"), ["N", "gap", "ci", "reps", "exact"], rows)
print("simulated N (phi - phi_N) at beta = 2 (95% half-width) vs exact:")
for row in rows:
    print(f"  N={row[0]:2d}: {row[1]:.3f} +- {row[2]:.3f}  ({row[3]} samples)   exact {row[4]:.3f}")
print(f"done in {time.time() - t0:.0f} s")
