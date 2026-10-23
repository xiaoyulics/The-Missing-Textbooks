"""Chapter 15 (SGD Is Not Bayes), "Measuring it": finite steps and gradient noise on the product
model y = abx at its singular truth, against the exact posterior.

Setting: n = 10^4, 400 data sets (x_i, y_i independent N(0,1)), one run of each algorithm per data set,
all started from the same point drawn uniformly from the prior box [-1,1]^2 (seed 1511).
  flow      gradient flow, end point in closed form (Theorem thm:sgd:balance)
  GD        full-batch gradient descent on L_n, step eta = 0.1, 2*10^4 steps
  SGD       minibatch SGD on L_B (minibatches of b = 10 drawn with replacement), eta = 0.1, 5*10^4 steps
  SGLD      the same minibatches and drift, (eps/2) n grad L_B with eps = 2 eta / n = 2e-5, plus
            N(0, eps I) noise, reflected into the prior box: SGLD at beta = 1 without localization
  Langevin  the same with full-batch gradients (the unadjusted Langevin algorithm)
  posterior exact distribution function of log|a| (Theorem thm:singular:scales (i)), per data set.

Prediction for the conserved quantity (Proposition prop:sgd:discrete): every step multiplies
D = a^2 - b^2 by 1 - eta^2 r^2, r = kappa_B (ab - c_B); near the fibre ab = c_hat the minibatch residual
has mean square about v/b, with v = (1/n) sum (x_i y_i)^2 (about 1), so |D_t/D_0| ~ exp(-eta^2 v t / b).
Prediction for collapse (heuristic in the chapter): once D ~ 0, SGD drifts to the origin when
|c_hat| < eta v / (2 kappa b), or when it reached the balanced line on the side opposite to c_hat.

Outputs (figures/data):
  15-sgd-steps-cdf.dat       tau, exact posterior CDF of log|a|/log n (mean over data sets), and the
                             empirical CDFs of flow, SGD, SGLD, Langevin end points with Clopper-Pearson
                             95% intervals (as offsets lo/hi)
  15-sgd-steps-balance.dat   t, median over runs of |D_t / D_0| for SGD, SGLD, GD, and exp(-eta^2 v t / b)
Run from the volume root: python figures/scripts/15-sgd-steps.py   (about 60 s)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import special

import sltkit as sk

OUT = os.path.join("figures", "data")
N, R = 10_000, 400
B, ETA = 10, 0.1
EPS = 2.0 * ETA / N                      # SGLD step with the same drift as SGD
T_SGD, T_GD = 50_000, 20_000
WGRID = np.arange(-60.0, 30.0, 0.002)


def flow_limit(a0, b0, c_hat):
    """Theorem thm:sgd:balance (vectorized over runs; c_hat may be an array)."""
    D = a0 * a0 - b0 * b0
    root = np.sqrt(D * D + 4.0 * c_hat * c_hat)
    u = np.sign(a0 + b0) * np.sqrt(np.maximum(2.0 * c_hat + root, 0.0))
    v = np.sign(a0 - b0) * np.sqrt(np.maximum(-2.0 * c_hat + root, 0.0))
    return (u + v) / 2.0, (u - v) / 2.0


def exact_cdf(u, z, ell):
    t = np.exp(WGRID)
    g = special.ndtr(t - z) - special.ndtr(-t - z)
    H = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(WGRID))])
    return np.interp(np.asarray(u) + ell, WGRID, H) / np.interp(ell, WGRID, H)


def reflect(z):
    z = np.where(z > 1.0, 2.0 - z, z)
    return np.where(z < -1.0, -2.0 - z, z)


def main():
    t_start = time.time()
    gen = sk.rng(1511)
    x = gen.standard_normal((R, N))
    y = gen.standard_normal((R, N))
    xx, xy = x * x, x * y
    Sxx, Sxy = xx.sum(1), xy.sum(1)
    kappa, c_hat = Sxx / N, Sxy / Sxx
    v = (xy ** 2).mean(1) - (Sxy / N) ** 2
    a0, b0 = gen.uniform(-1, 1, (2, R))
    D0 = a0 * a0 - b0 * b0
    logn = math.log(N)

    # gradient flow
    af, _ = flow_limit(a0, b0, c_hat)

    # SGD, SGLD (same minibatches), Langevin and GD with finite step
    rows = np.arange(R)[:, None]
    noise = sk.rng(1512)
    a_s, b_s = a0.copy(), b0.copy()          # SGD
    a_l, b_l = a0.copy(), b0.copy()          # SGLD
    a_f, b_f = a0.copy(), b0.copy()          # Langevin (full batch)
    a_g, b_g = a0.copy(), b0.copy()          # GD, finite step
    checkpoints = sorted({int(round(10 ** e)) for e in np.arange(1.0, math.log10(T_SGD) + 1e-9, 0.25)} | {T_SGD})
    bal = []
    for t in range(1, T_SGD + 1):
        idx = noise.integers(0, N, (R, B))
        kB = xx[rows, idx].mean(1)
        mB = xy[rows, idx].mean(1)
        r = kB * a_s * b_s - mB
        a_s, b_s = a_s - ETA * r * b_s, b_s - ETA * r * a_s
        r = kB * a_l * b_l - mB
        a_l, b_l = (reflect(a_l - ETA * r * b_l + math.sqrt(EPS) * noise.standard_normal(R)),
                    reflect(b_l - ETA * r * a_l + math.sqrt(EPS) * noise.standard_normal(R)))
        r = kappa * a_f * b_f - Sxy / N
        a_f, b_f = (reflect(a_f - ETA * r * b_f + math.sqrt(EPS) * noise.standard_normal(R)),
                    reflect(b_f - ETA * r * a_f + math.sqrt(EPS) * noise.standard_normal(R)))
        if t <= T_GD:
            r = kappa * a_g * b_g - Sxy / N
            a_g, b_g = a_g - ETA * r * b_g, b_g - ETA * r * a_g
        if t in checkpoints:
            # medians of |D_t/D_0|; an exact 0 (a^2 = b^2 in floating point) is written as nan
            med = lambda a, b: float(np.median(np.abs((a * a - b * b) / D0))) or float("nan")
            bal.append((t, med(a_s, b_s), med(a_l, b_l), med(a_g, b_g),
                        math.exp(-ETA ** 2 * float(np.median(v)) * t / B)))
    print(f"[runs] {R} runs per algorithm, {time.time() - t_start:.1f} s")
    sk.write_dat(os.path.join(OUT, "15-sgd-steps-balance.dat"), ["t", "sgd", "sgld", "gd", "pred"], bal)
    for row in bal[::2] + [bal[-1]]:
        print(f"   t = {row[0]:6d}: median |D_t/D_0|  SGD {row[1]:.3e}  SGLD {row[2]:.3e}  GD {row[3]:.4f}  exp(-eta^2 v t/b) {row[4]:.3e}")

    # distribution functions of tau = log|a| / log n
    taus = np.round(np.arange(-1.0, 0.0501, 0.05), 3)
    exact = np.zeros(len(taus))
    for r_ in range(R):
        z, ell = Sxy[r_] / math.sqrt(Sxx[r_]), 0.5 * math.log(Sxx[r_])
        exact += np.where(taus < 0, exact_cdf(np.minimum(taus * logn, 0.0), z, ell), 1.0)
    exact /= R
    ends = {"flow": af, "sgd": a_s, "sgld": a_l, "langevin": a_f}
    cols, data = ["tau", "post"], [taus, exact]
    for name, a in ends.items():
        tau_end = np.log(np.abs(a)) / logn
        k = np.array([np.sum(tau_end <= t) for t in taus])
        p = k / R
        ci = np.array([sk.clopper_pearson(int(kk), R) for kk in k])
        cols += [name, name + "lo", name + "hi"]
        data += [p, p - ci[:, 0], ci[:, 1] - p]
    sk.write_dat(os.path.join(OUT, "15-sgd-steps-cdf.dat"), cols, np.column_stack(data))
    for name, a in ends.items():
        q = np.quantile(np.log(np.abs(a)) / logn, [0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"   {name:9s} deciles/quartiles of log|a|/log n: " + " ".join(f"{qq:+.3f}" for qq in q))
    j = lambda t: int(np.argmin(np.abs(taus - t)))
    print("   exact posterior at tau = -0.5, -0.25, -0.1: " + " ".join(f"{exact[j(t)]:.3f}" for t in (-0.5, -0.25, -0.1)))
    for name, a in ends.items():
        tau_end = np.log(np.abs(a)) / logn
        dev = max(abs(np.mean(tau_end <= t) - exact[j(t)]) for t in taus if -0.75 <= t <= 0.0)
        print(f"   {name:9s} max |CDF - exact posterior CDF| on [-0.75, 0]: {dev:.3f}")

    # collapse of SGD
    c_end = a_s * b_s
    collapsed = np.abs(c_end) < 1e-8
    thr = ETA * v / (2.0 * kappa * B)
    below = np.abs(c_hat) < thr
    print(f"[collapse] SGD runs ending with |ab| < 1e-8: {collapsed.sum()} of {R} ({collapsed.mean():.3f}); "
          f"data sets with |c_hat| < eta v/(2 kappa b): {below.sum()} ({below.mean():.3f})")
    print(f"   collapsed among those below the threshold: {collapsed[below].mean():.3f}; above: {collapsed[~below].mean():.3f}")
    alive = ~collapsed
    same_side = np.sign(c_end[alive]) == np.sign(c_hat[alive])
    print(f"   surviving runs on the side of c_hat: {same_side.mean():.3f}; their |a| / sqrt|c_hat|: median "
          f"{np.median(np.abs(a_s[alive]) / np.sqrt(np.abs(c_hat[alive]))):.3f}")
    print(f"   SGLD runs with |ab| < 1e-8: {np.sum(np.abs(a_l * b_l) < 1e-8)}; GD (finite step) median |D_end/D_0| = "
          f"{np.median(np.abs((a_g * a_g - b_g * b_g) / D0)):.4f}, max |a_GD - a_flow| = {np.max(np.abs(a_g - af)):.3f}")
    print(f"   median v = {np.median(v):.4f}, median kappa = {np.median(kappa):.4f}")
    print(f"total {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
