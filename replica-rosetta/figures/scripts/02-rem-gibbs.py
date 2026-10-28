"""Chapter 2, figure fig:rem:gibbs(a) and the Poisson--Dirichlet numbers of Section 2.6: condensation.

(a) The participation ratio E Y_N = E sum_s G_N(s)^2 against beta: simulation at N = 20 (150 samples,
    seed 2101), exact finite-N values (Laplace transform, 02-rem-laplace.py) at N = 20, 80, 320, and the
    limit (1 - beta_c/beta)_+ (corollary cor:rem:condensation; the 1RSB prediction P(q) = m delta_0 + (1-m) delta_1).
Poisson--Dirichlet PD(m, 0) sampled by GEM stick-breaking (V_i ~ Beta(1-m, i m), 3000 sticks,
    10^4 samples, seed 2102): E sum W^2 and E (sum W^2)^2 against 1 - m and (1-m)(3-2m)/3; and the
    REM at N = 20 (beta = 1.5, 2, 3): first two moments of Y_N against PD at m_eff = 1 - E Y_N.
Writes figures/data/02-rem-gibbs-{curve,sim,pd}.dat. Runtime about 40 s.
"""
import importlib.util
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rosettakit as rk  # noqa: E402

spec = importlib.util.spec_from_file_location("rem_laplace", os.path.join(HERE, "02-rem-laplace.py"))
lap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lap)

DATA = os.path.join(HERE, "..", "data")
BC = lap.BETA_C
t0 = time.time()


def Y_of(X32, beta, N):
    """participation ratio sum_s G(s)^2 for one sample."""
    y = np.float32(beta * math.sqrt(N)) * X32
    w = np.exp((y - y.max()).astype(np.float64))
    s = w.sum()
    return float(np.dot(w, w) / (s * s))


def pd_moments(m):
    return 1 - m, (1 - m) * (3 - 2 * m) / 3


# (a) exact curves
betas = sorted([round(0.1 * k, 4) for k in range(6, 31)] + [BC])
rows = []
for b in betas:
    rows.append([b, max(0.0, 1 - BC / b), lap.E_Y(b, 20), lap.E_Y(b, 80), lap.E_Y(b, 320)])
rk.write_dat(os.path.join(DATA, "02-rem-gibbs-curve.dat"), ["beta", "limit", "exact20", "exact80", "exact320"], rows)
for r in rows:
    if r[0] in (1.0, 1.5, 2.0, 3.0) or abs(r[0] - BC) < 1e-9:
        print(f"  beta={r[0]:.4f}: E Y_N exact N=20 {r[2]:.4f}  N=80 {r[3]:.4f}  N=320 {r[4]:.4f}   limit {r[1]:.4f}")
print(f"exact curves done ({time.time() - t0:.0f} s)")

# (a) simulation at N = 20
N, R = 20, 150
sim_betas = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
gen = rk.rng(2101)
Ys = np.zeros((R, len(sim_betas)))
for r in range(R):
    X32 = gen.standard_normal(2 ** N).astype(np.float32)
    for j, b in enumerate(sim_betas):
        Ys[r, j] = Y_of(X32, b, N)
rows = []
inside = 0
for j, b in enumerate(sim_betas):
    mu, half = rk.mean_ci(Ys[:, j])
    ex = lap.E_Y(b, N)
    inside += abs(mu - ex) <= half
    rows.append([b, mu, half, ex, max(0.0, 1 - BC / b)])
    print(f"  N=20 beta={b:.2f}: simulated E Y {mu:.4f} +- {half:.4f}   exact {ex:.4f}   limit {rows[-1][4]:.4f}")
print(f"  {inside} of {len(rows)} intervals contain the exact value")
rk.write_dat(os.path.join(DATA, "02-rem-gibbs-sim.dat"), ["beta", "mean", "ci", "exact", "limit"], rows)
print(f"simulation done ({time.time() - t0:.0f} s)")

# Poisson--Dirichlet numbers quoted in the text: PD(m, 0) by GEM stick-breaking, and the REM at N = 20
gen = rk.rng(2102)
K, S = 3000, 10000
table = []
print("PD(m,0) by GEM stick-breaking (3000 sticks, 10^4 samples): mean +- 95% half-width")
for m in [0.25, 0.5, 0.75]:
    ii = np.arange(1, K + 1)
    y1, y2 = np.zeros(S), np.zeros(S)
    for s in range(S):
        V = gen.beta(1 - m, m * ii)
        logrest = np.concatenate([[0.0], np.cumsum(np.log1p(-V))[:-1]])
        W = V * np.exp(logrest)
        q = float(np.dot(W, W))
        y1[s], y2[s] = q, q * q
    (a1, h1), (a2, h2) = rk.mean_ci(y1), rk.mean_ci(y2)
    e1, e2 = pd_moments(m)
    table.append([m, a1, h1, e1, a2, h2, e2])
    print(f"  m={m:.2f}: E sum W^2 = {a1:.4f} +- {h1:.4f} (1-m = {e1:.4f});  E (sum W^2)^2 = {a2:.4f} +- {h2:.4f} ((1-m)(3-2m)/3 = {e2:.4f})")
print("REM at N = 20 (the simulation above): moments of Y_N against PD at m_eff = 1 - E Y_N")
for b in [1.5, 2.0, 3.0]:
    j = sim_betas.index(b)
    (a1, h1), (a2, h2) = rk.mean_ci(Ys[:, j]), rk.mean_ci(Ys[:, j] ** 2)
    meff = 1 - lap.E_Y(b, N)
    e1, e2 = pd_moments(meff)
    table.append([b, a1, h1, e1, a2, h2, e2])
    print(f"  beta={b:.1f} (m={BC / b:.3f}, m_eff={meff:.3f}): E Y = {a1:.4f} +- {h1:.4f} ({e1:.4f});  E Y^2 = {a2:.4f} +- {h2:.4f} (PD at m_eff: {e2:.4f})")
rk.write_dat(os.path.join(DATA, "02-rem-gibbs-pd.dat"), ["param", "m1", "ci1", "pd1", "m2", "ci2", "pd2"], table)
print(f"done in {time.time() - t0:.0f} s")
