"""Chapter 8 (Gaussian comparison): negative margins, alpha > 2. The maximal margin
kappa_max(G) = max_{|theta|=1} min_mu <g_mu, theta> is now negative and the problem is not convex
(it is the maximum-norm point of the polytope {theta : G theta >= -1}: kappa_max = -1/max |theta|).
Two polynomial-time algorithms give lower bounds on kappa_max:
  (LP)  the linear program of Montanari--Zhong--Zhou (2024, Sec. 4.2): maximize <v, theta> subject to
        G theta >= -1 for a uniformly random direction v; the output margin is -1/|theta|.  (Their
        program, max <v, theta> over {G theta >= kappa 1, |theta| <= 1}, returns a unit vector exactly
        when kappa <= -1/|theta|, by rescaling, so this is the same algorithm.)  Their
        Theorem 4.3 proves that it converges to kappa_lin(alpha) = Phi^{-1}(1/alpha).
  (SP)  soft-min ascent on the sphere (Riemannian gradient ascent of -(1/beta) log mean exp(-beta G theta),
        beta = 4, 8, ..., 512) started at v, followed by one linear program that maximizes
        <theta_soft, theta> subject to G theta >= -1 (which never lowers the margin).
They are compared with Gardner's curve kappa*(alpha) (an upper bound on kappa_max, by the capacity
theorem), the lifted Gordon upper bound, and kappa_lin (curves from 08-gordon-numbers.py).
d = 100: alpha = 2.5, 3, 4, 5, 6, 8, 10, 16 instances each, both algorithms on the same instances;
d = 200: alpha = 3, 6, 10, 6 instances each, algorithm SP only. Seeds 81000 + 100 alpha + d.
Means with Student-t 95% intervals.  Runtime: 130 s measured on a heavily loaded machine (less when
idle).  Run from the volume root.
"""
import math
import os
import sys
import time

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

T0 = time.time()
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def lp_direction(G, c):
    """argmax <c, theta> subject to G theta >= -1 (bounded when the patterns are not separable).
    Solved through the dual, min 1'y subject to G'y = -c, y >= 0, which HiGHS solves about twice as
    fast; theta is minus the multiplier of the equality constraints (checked against the primal
    solve to 1e-10 on test instances, build/scratch-gordon/profile_dual.py)."""
    n, d = G.shape
    res = optimize.linprog(np.ones(n), A_eq=G.T, b_eq=-c, bounds=[(0, None)] * n, method="highs")
    return -res.eqlin.marginals if res.status == 0 else None


def softmin_ascent(G, theta0, betas=(4, 8, 16, 32, 64, 128, 256, 512), iters=300, lr=0.05):
    th = theta0 / np.linalg.norm(theta0)
    best_th, best = th.copy(), float(np.min(G @ th))
    for beta in betas:
        for _ in range(iters):
            m = G @ th
            w = -beta * m
            p = np.exp(w - w.max())
            p /= p.sum()
            grad = G.T @ p
            grad -= (grad @ th) * th
            th = th + lr * grad
            th /= np.linalg.norm(th)
            if m.min() > best:
                best, best_th = float(m.min()), th.copy()
        lr *= 0.7
    return best_th


def margin(G, theta):
    return float(np.min(G @ theta) / np.linalg.norm(theta))


def kappa_star(a):
    return rk.root(lambda k: rk.gardner_capacity(k) - a, -8.0, 12.0)


rows = []
for d, alphas, T, algs in [(100, [2.5, 3, 4, 5, 6, 8, 10], 16, ("LP", "SP")), (200, [3, 6, 10], 6, ("SP",))]:
    for a in alphas:
        gen = rk.rng(81000 + int(100 * a) + d)
        res = {alg: [] for alg in algs}
        t1 = time.time()
        for _ in range(T):
            G = gen.standard_normal((int(round(a * d)), d))
            v = gen.standard_normal(d)
            v /= np.linalg.norm(v)
            if "LP" in algs:
                th = lp_direction(G, v)
                res["LP"].append(margin(G, th))
            ths = softmin_ascent(G, v)
            th2 = lp_direction(G, ths)
            res["SP"].append(max(margin(G, ths), margin(G, th2)) if th2 is not None else margin(G, ths))
        ks, kl = kappa_star(a), stats.norm.ppf(1 / a)
        for alg in algs:
            m, h = rk.mean_ci(res[alg])
            rows.append([a, d, alg, m, h, ks, kl])
            print(f"d={d} alpha={a:4.1f} {alg}: margin {m:.4f} +- {h:.4f}   Gardner/Gordon bound {ks:.4f}"
                  f"   LP limit Phi^-1(1/alpha) {kl:.4f}")
        print(f"   ({T} instances, {time.time()-t1:.1f} s)")
for d, alg in [(100, "LP"), (100, "SP"), (200, "SP")]:
    rk.write_dat(os.path.join(DATA, f"08-gordon-negative-{alg.lower()}{d}.dat"),
                 ["alpha", "kappa", "ci", "bound", "lplimit"],
                 [[r[0], r[3], r[4], r[5], r[6]] for r in rows if r[1] == d and r[2] == alg])
print(f"runtime {time.time()-T0:.1f} s")
