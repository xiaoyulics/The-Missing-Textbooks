"""Chapter 14 (Stages): the LLC estimator at the three k-gons whose learning coefficients the
chapter computes by hand (5-gon 7, 5+-gon 17/2, 6-gon 17/2), as the sample size grows.

Toy model of superposition, r = 2, c = 6, d = 18 (see 14-stages-tms.py for the model and the
per-sample loss 0.5 ||x - f||^2).  For each n in {10^3, 10^4, 10^5, 10^6} and each of 4 data
sets (seed 1440 + 10 j + data-set index), gradient descent on L_n (1500 steps of size 1) from the
exact k-gon finds the empirical critical point next to it; then the estimator of Lau et al.,
lambda_hat = n beta (mean L_n - L_n(w*)) with n beta = n / log n, gamma = 1, step
eps = 0.1 / (n beta), 4 chains of 9000 full-batch SGLD steps (first 2250 discarded).  The 5-gon's
vestigial bias is -3, deep inside its critical set (with a bias near 0 the sampler leaves the
set for the 5+-gon).  Intervals: Student-t 95% over the 4 data sets (chain averages).
At the smallest and the largest n the same chains are also run on the closed-form population
loss, at t = n beta from the exact k-gon: t (mean L - L(w*)), the "population skeleton" of the
estimator (no data noise).  At the smallest n the chains on the data are rerun with three further
sampler seeds (1451-1453), to show how far the means move with the sampler alone.
Writes figures/data/14-stages-calib.dat.
Run from the volume root: python figures/scripts/14-stages-calib.py   (80 to 120 s, depending on load)
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

R, C = 2, 6
NS = [1_000, 10_000, 100_000, 1_000_000]
DATASETS, CHAINS, STEPS, BURN = 4, 4, 9_000, 2_250
GAMMA = 1.0
OUT = os.path.join("figures", "data", "14-stages-calib.dat")
L5, B5 = 1.1704596391, -0.282297283127
L6, B6 = 1.32052633078, -0.618135131868


def kgon(k, l, beta, vest=-3.0, pos=()):
    W = np.zeros((R, C))
    b = np.full(C, float(vest))
    for i in range(k):
        th = 2 * np.pi * i / k
        W[:, i] = [l * np.cos(th), l * np.sin(th)]
        b[i] = beta
    for i in pos:
        b[i] = 1.0 / (2 * C)
    return W, b


POINTS = [kgon(5, L5, B5), kgon(5, L5, B5, pos=(5,)), kgon(6, L6, B6)]
NAMES = ["pent", "pentplus", "hex"]
LAMBDA = [7.0, 8.5, 8.5]


def pop_L_grad(W, b):
    """Population negative log-likelihood L = ell / 2 (up to a constant) and its gradient."""
    c = W.shape[2]
    G = W.transpose(0, 2, 1) @ W
    beta = np.broadcast_to(b[:, :, None], G.shape)
    a = G
    eye = np.eye(c, dtype=bool)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        neg = beta <= 0
        act = neg & (a > 0) & (beta >= -a)
        T = np.where(act, (a + beta) ** 3 / a, 0.0)
        T_a = np.where(act, 3 * (a + beta) ** 2 / a - (a + beta) ** 3 / a ** 2, 0.0)
        T_b = np.where(act, 3 * (a + beta) ** 2 / a, 0.0)
        part = ~neg & (a < -beta)
        full = ~neg & ~part
        T = T + np.where(part, -beta ** 3 / a, 0.0) + np.where(full, a * a + 3 * a * beta + 3 * beta ** 2, 0.0)
        T_a = T_a + np.where(part, beta ** 3 / a ** 2, 0.0) + np.where(full, 2 * a + 3 * beta, 0.0)
        T_b = T_b + np.where(part, -3 * beta ** 2 / a, 0.0) + np.where(full, 3 * a + 6 * beta, 0.0)
        q = (1 - a) ** 2 - 3 * (1 - a) * beta + 3 * beta ** 2
        q_a = -2 * (1 - a) + 3 * beta
        q_b = -3 * (1 - a) + 6 * beta
        D = np.where(act, q + beta ** 3 * (1 / a ** 2 + 1 / a), np.where(neg, 1.0, q))
        D_a = np.where(act, q_a - beta ** 3 * (2 / a ** 3 + 1 / a ** 2), np.where(neg, 0.0, q_a))
        D_b = np.where(act, q_b + 3 * beta ** 2 * (1 / a ** 2 + 1 / a), np.where(neg, 0.0, q_b))
    s = 1.0 / (3 * c) / 2.0
    L = np.where(eye, D, T).sum((1, 2)) * s
    dG = np.where(eye, D_a, T_a) * s
    return L, W @ (dG + dG.transpose(0, 2, 1)), np.where(eye, D_b, T_b).sum(2) * s


class EmpiricalLoss:
    """Exact empirical loss and gradient from prefix sums of sorted magnitudes (see 14-stages-tms.py),
    for several data sets of equal size at once: parameter set s uses data set ds[s]."""

    def __init__(self, data, ds, c=C):
        self.c, self.n = c, len(data[0][1])
        key = np.concatenate([mu + 2.0 * (c * r + feat) for r, (feat, mu) in enumerate(data)])
        order = np.argsort(key)
        self.key = key[order]
        m = np.concatenate([mu for _, mu in data])[order]
        self.P = np.zeros((3, len(m) + 1))
        for p in range(3):
            self.P[p, 1:] = np.cumsum(m ** p)
        grp = 2.0 * np.arange(c * len(data))
        self.start = np.searchsorted(self.key, grp, side="left").reshape(len(data), c)
        self.end = np.searchsorted(self.key, grp + 1.5, side="left").reshape(len(data), c)
        self.T = self.P[:, self.end] - self.P[:, self.start]              # (3, data sets, c)
        self.set_ds(ds)

    def set_ds(self, ds):
        ds = np.asarray(ds)
        self.off = (2.0 * (self.c * ds[:, None] + np.arange(self.c)[None, :]))[:, None, :]   # (S,1,c)
        self.st = self.start[ds][:, None, :]
        self.en = self.end[ds][:, None, :]
        self.Ts = self.T[:, ds][:, :, None, :]                            # (3,S,1,c)

    def __call__(self, W, b):
        c = self.c
        G = W.transpose(0, 2, 1) @ W
        beta = np.broadcast_to(b[:, :, None], G.shape)
        a = G
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = np.where(a != 0, -beta / a, 0.0)
        tau = np.clip(tau, -0.25, 1.25)
        k = np.searchsorted(self.key, (self.off + tau).ravel(), side="right").reshape(a.shape)
        st = np.broadcast_to(self.st, a.shape)
        k = np.clip(k, st, self.en)
        low = self.P[:, k] - self.P[:, st]
        T = self.Ts
        S0, S1, S2 = np.where(a > 0, T - low, np.where(a < 0, low, np.where(beta > 0, T, 0.0)))
        eye = np.eye(c, dtype=bool)[None]
        u = 1 - a
        off = a * a * S2 + 2 * a * beta * S1 + beta * beta * S0
        dia = T[2] - S2 + u * u * S2 - 2 * u * beta * S1 + beta * beta * S0
        da = np.where(eye, -2 * u * S2 + 2 * beta * S1, 2 * a * S2 + 2 * beta * S1)
        dbt = np.where(eye, -2 * u * S1 + 2 * beta * S0, 2 * a * S1 + 2 * beta * S0)
        s = 0.5 / self.n
        dG = da * s
        return np.where(eye, dia, off).sum((1, 2)) * s, W @ (dG + dG.transpose(0, 2, 1)), dbt.sum(2) * s


def langevin(loss, W0, b0, t, rng):
    """Average of t (L(w) - L(w0)) over chains of the localized tempered posterior at w0."""
    eps = 0.1 / t
    K = len(W0)
    Wc, bc = np.repeat(W0, CHAINS, 0), np.repeat(b0, CHAINS, 0)
    W, b = Wc.copy(), bc.copy()
    L0 = loss(Wc, bc)[0]
    acc = np.zeros(K * CHAINS)
    for s in range(STEPS):
        L, dW, db = loss(W, b)
        if s >= BURN:
            acc += L
        W += 0.5 * eps * (-t * dW - GAMMA * (W - Wc)) + np.sqrt(eps) * rng.normal(size=W.shape)
        b += 0.5 * eps * (-t * db - GAMMA * (b - bc)) + np.sqrt(eps) * rng.normal(size=b.shape)
    return (t * (acc / (STEPS - BURN) - L0)).reshape(K, CHAINS).mean(1)


def main():
    t_start = time.perf_counter()
    W0 = np.stack([p[0] for p in POINTS])
    b0 = np.stack([p[1] for p in POINTS])
    rng = sk.rng(1450)
    rows = []
    for j, n in enumerate(NS):
        nbeta = n / np.log(n)
        data = []
        for r in range(DATASETS):
            g = sk.rng(1440 + 10 * j + r)
            data.append((g.integers(0, C, n), g.random(n)))
        # parameter sets ordered (point k, data set r); descend to the empirical critical points
        ds_sets = np.tile(np.arange(DATASETS), 3)
        loss = EmpiricalLoss(data, ds_sets)
        W = np.repeat(W0, DATASETS, 0)
        b = np.repeat(b0, DATASETS, 0)
        for it in range(1500):
            _, dW, db = loss(W, b)
            W -= dW
            b -= db
        loss.set_ds(np.repeat(ds_sets, CHAINS))
        est = langevin(loss, W, b, nbeta, rng).reshape(3, DATASETS)
        # population skeleton at t = n beta (smallest and largest n only), 4 x 4 chains per point
        if n in (NS[0], NS[-1]):
            Wp = np.repeat(W0, DATASETS, 0)
            bp = np.repeat(b0, DATASETS, 0)
            skel = langevin(pop_L_grad, Wp, bp, nbeta, rng).reshape(3, DATASETS)
        else:
            skel = np.full((3, DATASETS), np.nan)
        row = [n, nbeta]
        line = f"n = {n:8d} (n beta = {nbeta:8.1f}):"
        for k in range(3):
            m, lo, hi = sk.mean_ci(est[k])
            ms, los, his = sk.mean_ci(skel[k])
            row += [m, m - lo, hi - m, ms, ms - los, his - ms]
            line += f"  {NAMES[k]} {m:.2f} [{lo:.2f}, {hi:.2f}] (skeleton {ms:.2f} [{los:.2f}, {his:.2f}]; lambda {LAMBDA[k]})"
        rows.append(row)
        print(line + f"   [{time.perf_counter() - t_start:.0f}s]", flush=True)
        print("    per data set (chain averages): " + "  ".join(f"{NAMES[k]} {np.round(est[k], 2).tolist()}" for k in range(3)),
              flush=True)
        if j == 0:
            first = (loss, W, b, nbeta, est.mean(1))
    hdr = ["n", "nbeta"]
    for nm in NAMES:
        hdr += [nm, nm + "lo", nm + "hi", nm + "S", nm + "Slo", nm + "Shi"]
    sk.write_dat(OUT, hdr, rows)
    # sampler-seed check at the smallest n: the same data sets and starting points, three further
    # sampler seeds; the Student-t intervals above cover data sets for one sampler seed only
    loss, W, b, nbeta, means = first
    allm = [means]
    for seed in (1451, 1452, 1453):
        allm.append(langevin(loss, W, b, nbeta, sk.rng(seed)).reshape(3, DATASETS).mean(1))
        print(f"  n = {NS[0]}, sampler seed {seed}: " + "  ".join(f"{NAMES[k]} {allm[-1][k]:.2f}" for k in range(3)), flush=True)
    allm = np.array(allm)
    print(f"  n = {NS[0]}, sampler seeds 1450-1453, range of the means: "
          + "  ".join(f"{NAMES[k]} {allm[:, k].min():.2f} to {allm[:, k].max():.2f}" for k in range(3)))
    print(f"wrote {OUT}; total time {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
