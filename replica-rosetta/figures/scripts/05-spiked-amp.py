"""Chapter 5 (Spiked Matrices): the hard phase of sparse PCA seen by Bayes-AMP from two starts.

Model: A = (lam/N) x x^T + W, W ~ GOE(N), with exactly k = rho N nonzero entries of x, equal to
+-1/sqrt(rho) with random signs and positions (so |x|^2 = N exactly; with i.i.d. entries |x|^2/N would
fluctuate by about 1/sqrt(rho N) = 8% at N = 3000 and shift the effective lam by as much).
Bayes-AMP in the normalization of Montanari--Venkataramanan (2021):
    x^{t+1} = A f_t(x^t) - b_t f_{t-1}(x^{t-1}),  f_t(y) = lam F(y; g_t),  b_t = mean of f_t'(x^t),
    F(y; g) = E[X | g X + sqrt(g) G = y],  with the data-driven SNR g_{t+1} = |f_t(x^t)|^2 / N.
Initialization independent of W (side information): x^0 = g0 x + sqrt(g0) G, f_{-1} = 0, so that
Bayati--Montanari state evolution applies: g_{t+1} = lam^2 T(g_t), T = 1 - mmse, and the squared cosine
of the estimate F(x^t; g_t) with x tends to T(g_t).  Weak start g0 = 0.01, strong start g0 = 3.

Outputs: figures/data/05-spiked-amp.dat (means, intervals and state evolution after ITERS steps) and
figures/data/05-spiked-amp-runs.dat (every run; lam shifted by -+0.0035 for the two starts, for plotting).
Run from the volume root: python figures/scripts/05-spiked-amp.py   (about one minute)
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

RHO, N, ITERS, SEEDS = 0.05, 3000, 50, 6
LAMS = [0.80, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.04, 1.10]
STARTS = [0.01, 3.0]

ZG = np.linspace(-11.0, 11.0, 4401)
WZ = np.exp(-ZG ** 2 / 2) / math.sqrt(2 * math.pi) * (ZG[1] - ZG[0])


def denoise(y, g, rho=RHO):
    """F(y; g) = E[X | g X + sqrt(g) G = y] and its derivative in y (the posterior variance)."""
    a = 1.0 / math.sqrt(rho)
    lp = math.log(rho / 2) + a * y - g * a * a / 2
    lm = math.log(rho / 2) - a * y - g * a * a / 2
    l0 = math.log(1 - rho) + 0.0 * y
    mx = np.maximum(np.maximum(lp, lm), l0)
    wp, wm, w0 = np.exp(lp - mx), np.exp(lm - mx), np.exp(l0 - mx)
    tot = wp + wm + w0
    m1 = a * (wp - wm) / tot
    return m1, a * a * (wp + wm) / tot - m1 ** 2


def T(g, rho=RHO):
    a = 1.0 / math.sqrt(rho)
    m1, _ = denoise(g * a + math.sqrt(g) * ZG, g, rho)
    return rho * a * float(np.sum(WZ * m1))


def state_evolution(g0, lam, iters=ITERS):
    g = g0
    for _ in range(iters):
        g = lam ** 2 * T(g)
    return g


def amp(W, x, lam, g0, gen, iters=ITERS):
    n = x.shape[0]
    xt = g0 * x + math.sqrt(g0) * gen.standard_normal(n)
    g = g0
    f_prev = np.zeros(n)
    for _ in range(iters):
        m, v = denoise(xt, g)
        ft = lam * m
        wf = (W @ ft.astype(W.dtype)).astype(np.float64)        # W is stored in single precision
        xt_new = wf + (lam / n) * x * float(x @ ft) - lam * float(np.mean(v)) * f_prev
        f_prev, xt = ft, xt_new
        g = max(float(ft @ ft) / n, 1e-12)
    m, _ = denoise(xt, g)
    return float(m @ x) ** 2 / (float(m @ m) * float(x @ x) + 1e-300)


def main():
    t0 = time.time()
    print("check: T(0.7) here %.12f, kit 1 - mmse_sparse_rademacher %.12f" % (T(0.7), 1 - rk.mmse_sparse_rademacher(0.7, RHO)))
    k = int(round(RHO * N))
    res = {(lam, g0): [] for lam in LAMS for g0 in STARTS}
    for s in range(SEEDS):
        gen = rk.rng(700 + s)
        W = rk.goe(N, gen).astype(np.float32)
        x = np.zeros(N)
        x[gen.choice(N, size=k, replace=False)] = gen.choice([-1.0, 1.0], size=k) / math.sqrt(RHO)
        for lam in LAMS:
            for g0 in STARTS:
                res[(lam, g0)].append(amp(W, x, lam, g0, gen))
        print("seed %d done, %.1f s" % (s, time.time() - t0))
    runs = []
    for lam in LAMS:
        for j, g0 in enumerate(STARTS):
            for s, c in enumerate(res[(lam, g0)]):
                runs.append([lam + (-0.0035 if j == 0 else 0.0035), j, s, c])
    rk.write_dat("figures/data/05-spiked-amp-runs.dat", ["lamj", "start", "seed", "cos2"], runs)
    rows = []
    for lam in LAMS:
        row = [lam]
        for g0 in STARTS:
            m, h = rk.mean_ci(res[(lam, g0)])
            se = T(state_evolution(g0, lam))
            esc = sum(c > 0.3 for c in res[(lam, g0)])
            row += [m, h, se]
            print("lam=%.2f g0=%.2f: AMP squared cosine %.3f +- %.3f, SE after %d steps %.4f; runs above 0.3: %d of %d"
                  % (lam, g0, m, h, ITERS, se, esc, SEEDS))
        rows.append(row)
    rk.write_dat("figures/data/05-spiked-amp.dat",
                 ["lam", "weak", "weakci", "weakse", "strong", "strongci", "strongse"], rows)
    print("total time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
