"""Chapter 8 (Gaussian comparison): zero margin. The probability that n Gaussian patterns in R^d
admit a unit vector theta with <g_mu, theta> >= 0 for all mu (equivalently, a.s., > 0), against
  - Cover's (Wendel's) exact formula 2^{1-n} sum_{k<d} C(n-1, k)   [rk.cover_probability],
  - the two-sided bound that the proof of the capacity theorem gives at kappa = 0:
        1 - 2 P(D > 0) <= P(feasible) <= 2 P(D <= 0),   D = |g_+| - |h|,  g in R^n, h in R^d,
    computed exactly: given K = #{g_i > 0} ~ Bin(n, 1/2), |g_+|^2 ~ chi^2_K, so
        P(D <= 0) = sum_k C(n,k) 2^{-n} I_{1/2}(k/2, d/2)      (regularized incomplete beta),
  - the cruder Chebyshev bounds of the theorem: P <= 4/(sqrt(n/2 - 1) - sqrt d)^2 if n/2 > d + 1,
    and 1 - P <= 4/(sqrt(d - 1) - sqrt(n/2))^2 if n/2 < d - 1.
Simulation: for each instance, n_max patterns are drawn once and n* = max{n : the first n are
separable} is found by bisection (separability is monotone in n); separability of a prefix is
decided by Lawson--Hanson least-distance programming, min |w| s.t. G w >= 1, solved by NNLS (a zero
residual certifies infeasibility; checked against the LP feasibility test on a subsample).
d = 25, 50, 100, 200 with 2000, 800, 300, 100 instances (seed 80800 + d); 95% Clopper--Pearson
intervals. By Cover's formula the mean instance capacity is exactly E n* = 2d (the number of fair
coin flips needed for d heads), which the script checks.  The exact curves and the two-sided bound
are also tabulated at d = 400 and 1600.  Runtime: about 60 s.  Run from the volume root.
"""
import math
import os
import sys
import time

import numpy as np
from scipy import optimize, special, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

T0 = time.time()
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def separable(G):
    """True if some w has G w >= 1 (Lawson--Hanson LDP via NNLS)."""
    n, d = G.shape
    E = np.vstack([G.T, np.ones((1, n))])
    f = np.zeros(d + 1)
    f[-1] = 1.0
    u, _ = optimize.nnls(E, f, maxiter=100 * n)
    return float(np.linalg.norm(E @ u - f)) > 1e-7


def lp_separable(G):
    n, d = G.shape
    res = optimize.linprog(np.zeros(d), A_ub=-G, b_ub=-np.ones(n), bounds=[(None, None)] * d, method="highs")
    return res.status == 0


def instance_capacity(G, d):
    """Largest n such that the first n rows of G are separable (n >= d always is, a.s.)."""
    nmax = G.shape[0]
    if separable(G):
        return nmax
    lo, hi = d, nmax          # separable at lo, not at hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if separable(G[:mid]):
            lo = mid
        else:
            hi = mid
    return lo


def p_D_le0(n, d):
    """P(|g_+| <= |h|) for g in R^n, h in R^d standard Gaussian, exactly."""
    k = np.arange(0, n + 1)
    logw = special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1) - n * math.log(2)
    inc = np.ones(n + 1)
    inc[1:] = special.betainc(k[1:] / 2.0, d / 2.0, 0.5)
    return float(np.sum(np.exp(logw) * inc))


DS = [25, 50, 100, 200]
TS = {25: 2000, 50: 800, 100: 300, 200: 100}
caps = {}
for d in DS:
    gen = rk.rng(80800 + d)
    nmax = int(math.ceil((3.4 if d == 25 else 2.8) * d))
    t1 = time.time()
    caps[d] = np.array([instance_capacity(gen.standard_normal((nmax, d)), d) for _ in range(TS[d])])
    exact_mean = sum(rk.cover_probability(n, d) for n in range(1, nmax + 1)) / d
    print(f"d={d}: {TS[d]} instances, n_max={nmax}, mean n*/d = {caps[d].mean()/d:.4f}"
          f" (+- {1.96*caps[d].std(ddof=1)/math.sqrt(TS[d])/d:.4f}; exact E n*/d = {exact_mean:.4f}),"
          f" {time.time()-t1:.1f} s")

# replication at d = 25 with six further seeds (1000 instances each), for the mean instance capacity
rep = []
for s in range(1, 7):
    g_s = rk.rng(s)
    rep += [instance_capacity(g_s.standard_normal((85, 25)), 25) for _ in range(1000)]
rep = np.array(rep)
print(f"replication, d=25, seeds 1-6, 6000 instances: mean n*/d = {rep.mean()/25:.4f}"
      f" (+- {1.96*rep.std(ddof=1)/math.sqrt(rep.size)/25:.4f})")

# sanity check of the NNLS decision against linear programming
gen = rk.rng(80899)
agree = 0
for _ in range(40):
    d = 50
    n = int(gen.integers(80, 121))
    G = gen.standard_normal((n, d))
    agree += separable(G) == lp_separable(G)
print(f"NNLS vs LP separability decisions agree on {agree}/40 instances (d = 50, 80 <= n <= 120)")

# simulated curves, one file per d
for d in DS:
    rows = []
    T = TS[d]
    step = max(1, d // 20)
    for n in range(d, int(math.ceil((3.4 if d == 25 else 2.8) * d)) + 1, step):
        k = int(np.sum(caps[d] >= n))
        lo, hi = rk.clopper_pearson(k, T)
        p = k / T
        rows.append([n / d, p, p - lo, hi - p, rk.cover_probability(n, d)])
    rk.write_dat(os.path.join(DATA, f"08-gordon-cover-d{d}.dat"), ["alpha", "p", "em", "ep", "cover"], rows)

# exact curves on a fine grid: Cover for each d; the CGMT sandwich and the Chebyshev bounds at d = 100
rows = []
for a in np.round(np.arange(1.0, 3.4001, 0.01), 4):
    row = [float(a)]
    for d in DS + [400, 1600]:
        row.append(rk.cover_probability(int(round(a * d)), d))
    d = 100
    n = int(round(a * d))
    pD = p_D_le0(n, d)
    row += [max(0.0, 1 - 2 * (1 - pD)), min(1.0, 2 * pD)]
    lo = 1 - 4 / (math.sqrt(d - 1) - math.sqrt(n / 2)) ** 2 if n / 2 < d - 1 - 1e-12 else 0.0
    hi = 4 / (math.sqrt(n / 2 - 1) - math.sqrt(d)) ** 2 if n / 2 > d + 1 + 1e-12 else 1.0
    row += [max(0.0, lo), min(1.0, hi), 1.0 if a < 2 else (0.5 if a == 2 else 0.0)]
    for d in [400, 1600]:
        pD = p_D_le0(int(round(a * d)), d)
        row += [max(0.0, 1 - 2 * (1 - pD)), min(1.0, 2 * pD)]
    rows.append(row)
rk.write_dat(os.path.join(DATA, "08-gordon-cover-exact.dat"),
             ["alpha", "cover25", "cover50", "cover100", "cover200", "cover400", "cover1600", "low100", "up100",
              "cheblo100", "chebhi100", "step", "low400", "up400", "low1600", "up1600"], rows)

print("\nnumbers quoted in the text (d = 100):")
for a in [1.6, 1.8, 2.0, 2.2, 2.4]:
    n = int(round(a * 100))
    T = TS[100]
    k = int(np.sum(caps[100] >= n))
    lo, hi = rk.clopper_pearson(k, T)
    pD = p_D_le0(n, 100)
    print(f"   alpha={a}: simulated {k/T:.3f} [{lo:.3f}, {hi:.3f}]  Cover {rk.cover_probability(n, 100):.4f}"
          f"  CGMT sandwich [{max(0, 1-2*(1-pD)):.4f}, {min(1, 2*pD):.4f}]  P(D<=0)={pD:.4f}")
for d in DS:
    print(f"   Cover at n = 2d, d={d}: {rk.cover_probability(2*d, d):.6f}")
# the two-sided bound at larger d, where it approaches the step at alpha = 2
for d in [400, 1600]:
    for a in [1.8, 1.9, 2.1, 2.2]:
        n = int(round(a * d))
        pD = p_D_le0(n, d)
        print(f"   d={d}, alpha={a}: bound [{max(0, 1-2*(1-pD)):.4f}, {min(1, 2*pD):.4f}]  Cover {rk.cover_probability(n, d):.4f}")
# the window of the auxiliary problem against Cover's: P(D <= 0) and Cover at d = 100
for a in [1.8, 2.2]:
    n = int(round(a * 100))
    print(f"   window check d=100, alpha={a}: P(D<=0)={p_D_le0(n, 100):.4f} vs normal approx "
          f"{stats.norm.cdf(-(a-2)*10/(4*math.sqrt(9/8))):.4f}; Cover {rk.cover_probability(n, 100):.4f} vs "
          f"{stats.norm.cdf(-(a-2)*10/math.sqrt(2)):.4f}")
print(f"runtime {time.time()-T0:.1f} s")
