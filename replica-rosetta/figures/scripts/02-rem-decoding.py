"""Chapter 2, Section 2.6 (numbers quoted in the text, no figure): decoding 2^N orthogonal codewords
on the Gaussian channel (corollary cor:rem:decoding).

Received scores y_s = lambda sqrt(N) 1{s = s*} + X_s; the posterior is the REM Gibbs measure at
beta = lambda with one planted level.
(a) P(MAP decoding succeeds) = E Phi(lambda sqrt N + X)^(2^N - 1) against lambda, exact for
    N = 12, 40, 160, 640; simulated at N = 12 (2000 transmissions per lambda, seed 2301,
    Clopper--Pearson 95% intervals); the limit is the step at lambda = beta_c (E_b/N_0 = ln 2).
(b) Mutual information per message bit I(s*; y)/N = log 2 + lambda^2 - E log Z(y)/N, exact
    (Laplace transform) for N = 12, 32, 96 (lambda step 0.1); simulated at N = 12 (the same transmissions,
    Student-t 95% intervals); the limit min(lambda^2/2, log 2).
Writes figures/data/02-rem-decoding-{curve,mi,sim}.dat. Runtime about 90 s.
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
t0 = time.time()

lams = [round(0.5 + 0.05 * k, 4) for k in range(0, 31)]
rows = []
for lam in lams:
    rows.append([lam, 1.0 if lam > BC else 0.0] + [lap.map_success(lam, N) for N in (12, 40, 160, 640)])
rk.write_dat(os.path.join(DATA, "02-rem-decoding-curve.dat"), ["lam", "steplimit", "s12", "s40", "s160", "s640"], rows)
print(f"beta_c = {BC:.5f} (E_b/N_0 = ln 2 = {10 * math.log10(LOG2):.3f} dB)")
for r in rows:
    if abs(r[0] - 1.0) < 1e-9 or abs(r[0] - 1.2) < 1e-9 or abs(r[0] - 1.5) < 1e-9:
        print(f"  lambda={r[0]:.2f}: P(success) N=12, 40, 160, 640: " + " ".join(f"{v:.3f}" for v in r[2:]))
rows = []
for lam in [round(0.5 + 0.1 * k, 4) for k in range(0, 16)]:
    rows.append([lam, min(lam * lam / 2, LOG2)] + [lap.mutual_information(lam, N) for N in (12, 32, 96)])
rk.write_dat(os.path.join(DATA, "02-rem-decoding-mi.dat"), ["lam", "milimit", "mi12", "mi32", "mi96"], rows)
for r in rows:
    if abs(r[0] - 1.0) < 1e-9 or abs(r[0] - 1.2) < 1e-9 or abs(r[0] - 1.5) < 1e-9:
        print(f"  lambda={r[0]:.2f}: I/N for N=12, 32, 96: " + " ".join(f"{v:.4f}" for v in r[2:]) + f"   limit {r[1]:.4f}")
print(f"exact curves done ({time.time() - t0:.0f} s)")

N, R = 12, 2000
gen = rk.rng(2301)
rows = []
for lam in [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8]:
    wins, lz = 0, np.zeros(R)
    for r in range(R):
        y = gen.standard_normal(2 ** N)
        y[0] += lam * math.sqrt(N)                      # by symmetry the transmitted codeword is s* = 0
        wins += int(np.argmax(y) == 0)
        lz[r] = special.logsumexp(lam * math.sqrt(N) * y)
    lo, hi = rk.clopper_pearson(wins, R)
    mu, half = rk.mean_ci(LOG2 + lam * lam - lz / N)
    p = wins / R
    rows.append([lam, p, p - lo, hi - p, mu, half, lap.map_success(lam, N), lap.mutual_information(lam, N)])
    print(f"  N=12 lambda={lam:.2f}: success {p:.3f} [{lo:.3f}, {hi:.3f}] (exact {rows[-1][6]:.3f});"
          f"  MI/N {mu:.4f} +- {half:.4f} (exact {rows[-1][7]:.4f})")
cov_s = sum(r[1] - r[2] <= r[6] <= r[1] + r[3] for r in rows)
cov_m = sum(abs(r[4] - r[7]) <= r[5] for r in rows)
print(f"  {cov_s} of {len(rows)} success intervals and {cov_m} of {len(rows)} information intervals contain the exact value")
rk.write_dat(os.path.join(DATA, "02-rem-decoding-sim.dat"), ["lam", "p", "plo", "phi", "mi", "mici", "pexact", "miexact"], rows)
print("check with 40000 transmissions per lambda (seed 2302):")
gen = rk.rng(2302)
for lam in [1.2, 1.3, 1.4]:
    wins = 0
    for r in range(40000):
        y = gen.standard_normal(2 ** N)
        y[0] += lam * math.sqrt(N)
        wins += int(np.argmax(y) == 0)
    lo, hi = rk.clopper_pearson(wins, 40000)
    print(f"  lambda={lam:.1f}: success {wins / 40000:.4f} [{lo:.4f}, {hi:.4f}]   exact {lap.map_success(lam, N):.4f}")
print(f"done in {time.time() - t0:.0f} s")
