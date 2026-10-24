"""Chapter 3 (SK): exact enumeration at small N against the replica predictions and the theorems.

Normalization (OUTLINE.md): H_N(s) = (beta/sqrt N) sum_{i<j} g_ij s_i s_j + h sum_i s_i, here h = 0.
Every average over the disorder is a sample mean over independent draws of the couplings (and of the
fields z of the interpolation), with a Student-t 95% interval.

1. The free entropy phi_N(beta) = N^{-1} E log Z_N at N = 16 for beta = 0.1, ..., 2.0 (120 disorder
   draws, the same draws for every beta; seed 3016). Written to figures/data/03-sk-phi-enum.dat.
2. Guerra's sum rule, phi_RS(q) - phi_N = (beta^2/4) int_0^1 E<(R12 - q)^2>_t dt, which holds exactly at
   every N (Theorem 3.3), measured on both sides at beta = 0.8 (q = 0) and beta = 2 (q = q* = 0.530368)
   for N = 4, ..., 14: the left side from log Z_N (8000 draws at N = 4 down to 1000 at N = 14), the
   right side from the Gibbs measures of the interpolating Hamiltonian at 6 Gauss--Legendre nodes in t
   (2000 draws at N = 4 down to 100 at N = 14). Independent disorder draws for the two sides (seeds
   30000 + 100 N + k and 30050 + 100 N + k). Also printed: N E<R12^2> at beta = 0.8 against 1/(1 - beta^2).
   Written to figures/data/03-sk-sumrule.dat with the physical prediction -(1/4N) log(1 - beta^2) and the
   rigorous bounds of Theorem 3.4 (beta = 0.8) and the rigorous lower bound phi_RS - P_1RSB (beta = 2).

Run from the volume root: python figures/scripts/03-sk-enum.py   (about two minutes on a busy machine)
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

T0 = time.time()
OUT = os.path.join("figures", "data")
LOG2 = math.log(2.0)


def spins(N):
    return rk.all_spins(N).astype(float)


def energies(S, G):
    """X(s) / beta = N^{-1/2} sum_{i<j} G_ij s_i s_j for all configurations (G upper triangular)."""
    N = S.shape[1]
    return np.sum((S @ G) * S, axis=1) / math.sqrt(N)


def draw_couplings(N, gen):
    G = np.zeros((N, N))
    iu = np.triu_indices(N, 1)
    G[iu] = gen.standard_normal(len(iu[0]))
    return G


def logsumexp(x):
    """log sum exp along the last axis (a lean version of scipy's, for speed)."""
    m = np.max(x, axis=-1, keepdims=True)
    return np.squeeze(m, -1) + np.log(np.sum(np.exp(x - m), axis=-1))


def gibbs(S, logw):
    """log Z, magnetizations <s_i> and correlations <s_i s_j> of the Gibbs measure with log-weights logw."""
    lz = logsumexp(logw)
    w = np.exp(logw - lz)
    return lz, S.T @ w, (S * w[:, None]).T @ S


# RS values (from figures/scripts/03-sk-replica.py): beta = 0.8: q = 0, phi_RS = log 2 + beta^2/4;
# beta = 2: q* = 0.530368..., phi_RS(q*) = 1.584786..., phi_1RSB = 1.582610...
def phi_rs_at(beta, q, n=201):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    u = beta * math.sqrt(q) * x
    lc = np.abs(u) + np.log1p(np.exp(-2 * np.abs(u))) - LOG2
    return LOG2 + float(np.dot(w, lc)) + beta ** 2 * (1 - q) ** 2 / 4


def q_rs(beta, n=201):
    if beta <= 1:
        return 0.0
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    return rk.root(lambda q: float(np.dot(w, np.tanh(beta * math.sqrt(q) * x) ** 2)) - q, 1e-3, 1 - 1e-12)


# ---------------------------------------------------------------------------------------------
# 1. phi_16(beta)
# ---------------------------------------------------------------------------------------------
N = 16
S = spins(N)
betas = np.round(np.arange(0.1, 2.0001, 0.1), 4)
gen = rk.rng(3016)
NDRAW16 = 120
L = np.zeros((NDRAW16, len(betas)))
for s in range(NDRAW16):
    E0 = energies(S, draw_couplings(N, gen))
    L[s, :] = logsumexp(np.outer(betas, E0)) / N
rows = []
for j, b in enumerate(betas):
    m, half = rk.mean_ci(L[:, j])
    rows.append((b, m, half))
rk.write_dat(os.path.join(OUT, "03-sk-phi-enum.dat"), ["beta", "phi", "ci"], rows)
print(f"1. phi_16(beta), {NDRAW16} draws (95% intervals); RS and annealed values")
for b, m, half in rows:
    if abs(b - round(b * 2) / 2) < 1e-9 or abs(b - 0.8) < 1e-9:
        q = q_rs(b)
        print(f"   beta={b:.1f}: phi_16 = {m:.4f} +- {half:.4f}   phi_RS = {phi_rs_at(b, q):.4f}   annealed (N=16) = {LOG2 + b * b * (N - 1) / (4 * N):.4f}")
print(f"   ({time.time() - T0:.1f} s)")

# ---------------------------------------------------------------------------------------------
# 2. Guerra's sum rule at beta = 0.8 and beta = 2
# ---------------------------------------------------------------------------------------------
tx, tw = np.polynomial.legendre.leggauss(6)
tx, tw = 0.5 * (tx + 1.0), 0.5 * tw
# draws for the left side (one log-partition function each) and for the right side (6 Gibbs measures)
samples_l = {4: 8000, 6: 8000, 8: 6000, 10: 4000, 12: 2000, 14: 1000}
samples_r = {4: 2000, 6: 1500, 8: 800, 10: 400, 12: 200, 14: 100}
PHI1RSB_2 = 1.582610   # min of the 1RSB functional at beta = 2, h = 0 (03-sk-replica.py)


def sumrule(beta, N, seed):
    q = q_rs(beta)
    phirs = phi_rs_at(beta, q)
    S = spins(N)
    gen_l = rk.rng(seed)
    gen_r = rk.rng(seed + 50)
    lhs = np.empty(samples_l[N])
    for s in range(samples_l[N]):
        lhs[s] = phirs - logsumexp(beta * energies(S, draw_couplings(N, gen_l))) / N
    rhs = np.empty(samples_r[N])
    R2top = np.empty(samples_r[N])
    for s in range(samples_r[N]):
        E0 = energies(S, draw_couplings(N, gen_r))
        Y = S @ gen_r.standard_normal(N)
        acc = 0.0
        for t, w in zip(tx, tw):
            _, m, C = gibbs(S, math.sqrt(t) * beta * E0 + math.sqrt(1 - t) * beta * math.sqrt(q) * Y)
            acc += w * (np.sum(C ** 2) / N ** 2 - 2 * q * np.sum(m ** 2) / N + q * q)
        rhs[s] = beta ** 2 / 4 * acc
        if q == 0.0:
            _, m, C = gibbs(S, beta * E0)
            R2top[s] = np.sum(C ** 2) / N ** 2
    return q, phirs, rk.mean_ci(lhs), rk.mean_ci(rhs), rk.mean_ci(N * R2top)


print("2. Guerra's sum rule: phi_RS(q) - phi_N against (beta^2/4) int E<(R-q)^2>_t dt")
res = {}
for k, beta in enumerate((0.8, 2.0)):
    for N in sorted(samples_l):
        q, phirs, (gm, gh), (rm, rh), (nr, nrh) = sumrule(beta, N, 30000 + 100 * N + k)
        res[(beta, N)] = (gm, gh, rm, rh)
        extra = f"   N E<R^2> = {nr:.3f} +- {nrh:.3f} (1/(1-b^2) = {1 / (1 - beta ** 2):.3f})" if beta < 1 else ""
        print(f"   beta={beta} N={N:2d} ({samples_l[N]}/{samples_r[N]} draws): gap = {gm:.5f} +- {gh:.5f}   "
              f"remainder = {rm:.5f} +- {rh:.5f}   N x gap = {N * gm:.4f}{extra}")
    print(f"   ({time.time() - T0:.1f} s)")

b = 0.8
cb = 0.25 * math.exp(b * b / 2) * math.sqrt(1 - b * b)
qstar2 = q_rs(2.0)
gap_lb_2 = phi_rs_at(2.0, qstar2) - PHI1RSB_2
rows = []
for N in sorted(samples_l):
    g8, gh8, r8, rh8 = res[(0.8, N)]
    g2, gh2, r2, rh2 = res[(2.0, N)]
    pred = -math.log(1 - b * b) / (4 * N)
    lo = b * b / (4 * N)
    up = (LOG2 + b * b / 4) / N + b * math.sqrt(math.log(2 / cb) / N)
    rows.append((N, g8, gh8, r8, rh8, pred, lo, up, g2, gh2, r2, rh2, gap_lb_2))
rk.write_dat(os.path.join(OUT, "03-sk-sumrule.dat"),
             ["N", "gap8", "gci8", "rem8", "rci8", "pred8", "lo8", "up8", "gap2", "gci2", "rem2", "rci2", "lb2"], rows)
print(f"   beta=0.8: c_beta = {cb:.5f}; Theorem 3.4 bound on the gap at N=14: {rows[-1][7]:.4f}; Jensen lower bound {rows[-1][6]:.5f}; prediction {rows[-1][5]:.5f}")
print(f"   beta=2: rigorous lower bound phi_RS(q*) - P_1RSB = {gap_lb_2:.6f} (q* = {qstar2:.6f})")
print(f"done in {time.time() - T0:.1f} s")
