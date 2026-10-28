"""Chapter 4 (nishimori): Z2 synchronization by exact posterior enumeration at N = 6, 8, 10, 12.

Model (book normalization): A = (lam/N) x* x*^T + W, W ~ GOE(N), x* uniform on {-1,+1}^N.
The posterior is the Gibbs measure exp((lam/2) x^T A x) on {-1,+1}^N; the evidence (likelihood ratio
against lam = 0) is Z_N = 2^-N sum_x exp((lam/2) x^T A x - lam^2 N/4), and the free entropy is
phi_N = (1/N) E log Z_N = lam^2/4 - I(x*; A)/N.  For each (N, lam) the script draws M instances,
enumerates all 2^N configurations, and records per instance
    (1/N) log(2^N p(x*|A)),   MMSE sample (1/N^2)|x* x*^T - <x x^T>|_F^2,   <R_{1*}^2>,   <R_{12}^2>,
with R_{1*} = <x, x*>/N and R_{12} = <x^1, x^2>/N for two independent posterior samples.
By Bayes' rule log Z_N = B - log(2^N p(x*|A)) with B = (lam/2) x*^T A x* - lam^2 N/4, and E B = lam^2 N/4
exactly.  So B/N - lam^2/4 is a control variate of known mean zero: the script estimates phi_N by the
mean of (1/N) log Z_N - c (B/N - lam^2/4) with the regression coefficient c fitted on the sample
(c = 0 is the direct estimator, c = 1 is lam^2/4 - (1/N) log(2^N p(x*|A)); the fitted c is near 0 for
small lam, where the direct estimator is best, and near 1 for large lam, where B dominates the variance).
It compares with the replica-symmetric formula max_q F(lam, q) and 1 - q*^2 (04-nishimori-rs.py),
checks Guerra's bound phi_N >= max_q F (a theorem at every N) and the Nishimori identity
E<R_{1*}^2> = E<R_{12}^2>.  Intervals: Student-t 95%.  Seeds 1000 N + k for the k-th value of lam.

Writes figures/data/04-nishimori-exact.dat.  Runtime: about 45 s.
Run from the volume root:  python figures/scripts/04-nishimori-exact.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk

OUT = os.path.join("figures", "data")
SIZES = [6, 8, 10, 12]
LAMS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
M = 1200


def g(gam):
    return 0.0 if gam <= 0 else rk.gauss(lambda z: np.tanh(gam + math.sqrt(gam) * z), 201)


def psi(gam):
    return 0.0 if gam <= 0 else rk.gauss(lambda z: np.log(np.cosh(gam + math.sqrt(gam) * z)), 201) - gam / 2


def rk_t95(n):
    """The Student-t quantile used by rk.mean_ci, to turn a half-width back into a standard error."""
    from scipy import stats
    return float(stats.t.ppf(0.975, n - 1))


def rs(lam):
    """(max_q F(lam, q), q*) for the Rademacher prior."""
    q = 0.0 if lam <= 1 else rk.root(lambda q: g(lam * lam * q) - q, 1e-12, 1.0)
    return psi(lam * lam * q) - lam * lam * q * q / 4, q


def instance(S, lam, gen):
    N = S.shape[1]
    xs = gen.choice([-1.0, 1.0], size=N)
    A = rk.spiked_wigner(xs, lam, gen)
    E = 0.5 * lam * np.einsum("ki,ij,kj->k", S, A, S, optimize=True)
    lse = logsumexp(E)
    logZ = lse - N * math.log(2) - lam * lam * N / 4
    w = np.exp(E - lse)
    B = 0.5 * lam * float(xs @ A @ xs) - lam * lam * N / 4   # log p(A|x*) - log p_W(A), mean lam^2 N/4
    C = (S * w[:, None]).T @ S                       # <x_i x_j>
    r1s = S @ xs / N
    mmse = float(np.sum((np.outer(xs, xs) - C) ** 2)) / N ** 2
    return logZ / N, mmse, float(w @ r1s ** 2), float(np.sum(C ** 2)) / N ** 2, B / N - lam * lam / 4


def cv_mean(a, b):
    """Mean and 95% half-width of a - c b, with c the least-squares coefficient (E b = 0 exactly)."""
    c = float(np.cov(a, b)[0, 1] / np.var(b, ddof=1)) if np.var(b) > 0 else 0.0
    m, h = rk.mean_ci(a - c * b)
    return m, h, c


def main():
    t0 = time.time()
    header = ["lam", "phirs", "mmsers"]
    for N in SIZES:
        header += [f"phi{N}", f"phi{N}ci", f"gap{N}", f"gap{N}ci", f"mmse{N}", f"mmse{N}ci", f"nish{N}", f"nish{N}ci"]
    rows = []
    tvals = []
    diffs = []
    for k, lam in enumerate(LAMS):
        phirs, q = rs(lam)
        row = [lam, phirs, 1 - q * q]
        for N in SIZES:
            S = rk.all_spins(N).astype(float)
            gen = rk.rng(1000 * N + k)
            data = np.array([instance(S, lam, gen) for _ in range(M)])
            phi, phih, c = cv_mean(data[:, 0], data[:, 4])
            gap, gaph = phi - phirs, phih
            direct, directh = rk.mean_ci(data[:, 0])
            mm, mmh = rk.mean_ci(data[:, 1])
            nish, nishh = rk.mean_ci(data[:, 2] - data[:, 3])
            tvals.append(nish / (nishh / rk_t95(M)))
            diffs.append(data[:, 2] - data[:, 3])
            row += [phi, phih, N * gap, N * gaph, mm, mmh, nish, nishh]
            print(f"N={N:2d} lam={lam:4.2f}: phi_N={phi:.4f}+-{phih:.4f} (c={c:.2f}; direct {direct:.4f}+-{directh:.4f})"
                  f"  RS={phirs:.4f}  N*gap={N * gap:.3f}+-{N * gaph:.3f}"
                  f"  MMSE_N={mm:.4f}+-{mmh:.4f} (RS {1 - q * q:.4f})  E<R1*^2>-E<R12^2>={nish:+.4f}+-{nishh:.4f}")
        rows.append(row)
    rk.write_dat(os.path.join(OUT, "04-nishimori-exact.dat"), header, rows)
    tsigned = np.array(tvals)
    tvals = np.abs(tsigned)
    print(f"Nishimori identity: |t| over {tvals.size} tests: max {tvals.max():.2f}, "
          f"{int(np.sum(tvals > 1.96))} above 1.96, {int(np.sum(tvals > 2.58))} above 2.58")
    # The per-instance differences <R1*^2> - <R12^2> are strongly skewed (a few instances on which the
    # posterior is confident and wrong carry large negative values). Recheck the worst cell with two
    # fresh sets of 3000 instances (seeds 77 and 78), and estimate how often a Student statistic as
    # large as the worst one occurs, by resampling the pooled differences recentred to mean zero (the
    # identity is exact), in one cell and for the largest of the forty.
    k = int(np.argmax(tvals))
    lam, N = LAMS[k // len(SIZES)], SIZES[k % len(SIZES)]
    d0 = diffs[k]
    q0 = np.quantile(d0, [0.001, 0.5])
    print(f"  worst cell N={N} lam={lam}: t = {tsigned[k]:+.2f}; median {q0[1]:+.3f}, 0.1% quantile {q0[0]:+.3f}")
    S = rk.all_spins(N).astype(float)
    pooled = [d0]
    for seed in (77, 78):
        gen = rk.rng(seed)
        d = np.array([instance(S, lam, gen) for _ in range(3000)])
        diff = d[:, 2] - d[:, 3]
        pooled.append(diff)
        m, h = rk.mean_ci(diff)
        q = np.quantile(diff, [0.001, 0.5, 0.999])
        print(f"  recheck N={N} lam={lam} seed {seed}: E<R1*^2>-E<R12^2> = {m:+.4f} +- {h:.4f}; "
              f"quantiles 0.1%, 50%, 99.9%: {q[0]:+.3f} {q[1]:+.4f} {q[2]:+.3f}; sd {diff.std(ddof=1):.3f}")
    cen = np.concatenate(pooled)
    cen = cen - cen.mean()
    gen = rk.rng(5)
    ts = []
    for _ in range(100):
        x = cen[gen.integers(0, cen.size, size=(1000, M))]
        ts.append(x.mean(axis=1) / (x.std(axis=1, ddof=1) / math.sqrt(M)))
    ts = np.abs(np.concatenate(ts))
    p1 = float(np.mean(ts >= tvals[k]))
    print(f"  resampled |t| (100000 samples of {M}): P(|t| >= {tvals[k]:.2f}) = {p1:.1e} in one cell, "
          f"{1 - (1 - p1) ** tvals.size:.3f} for the largest of {tvals.size}; "
          f"P(|t| > 1.96) = {np.mean(ts > 1.96):.3f}")
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
