"""Chapter 14 (Stages): stages of SGD training in the toy model of superposition, and the
local learning coefficient estimated along training.

Setting (Chen et al. 2023, Section 5, with the book's conventions):
  model f(x, w) = relu(W^T W x + b), W in R^{2x6}, b in R^6 (d = 18); x = mu e_i with i uniform
  in {1..6} and mu uniform in [0, 1]; per-sample loss 0.5 ||x - f||^2 (the negative log-
  likelihood of p(x | w) ~ exp(-||x - f||^2 / 2), up to a constant).
  SGD: n = 1000 training points, minibatches of 20, step 0.01 on this loss (step 0.005 on the
  squared error ||x - f||^2), 4500 epochs (225,000 steps), started at the standard 4-gon
  (columns e1, e2, -e1, -e2, two zero columns, zero biases) plus N(0, 0.01^2) noise, as in
  Chen et al.  32 runs: data seed 1000 + s, initialization seed 2000 + s, shuffling seed
  3000 + s, s = 0..31.
  LLC: the estimator of Lau et al. at a checkpoint w_t, lambda_hat = n beta (mean L_n - L_n(w_t)),
  with n beta = n / log n = 144.8, localization gamma = 1 (drift -eps gamma (w - w_t) / 2), step
  eps = 0.1 / (n beta), full-batch gradients, 6 chains of 6000 steps of which the first 1500 are
  discarded.  Intervals are Student-t 95% intervals over the chains (sampler variability only:
  the chains share the data set).
  The empirical loss is computed exactly from sorted magnitudes: for output j and input
  feature i the sum of relu(a mu + beta)^2 over the training points of feature i is
  a^2 S2 + 2 a beta S1 + beta^2 S0 over the points where a mu + beta > 0, and these sets are
  upper or lower tails of the sorted magnitudes (prefix sums), so one step costs O(log n).
Selection: the two runs whose population loss drops earliest by more than 0.02 below its value
at step 5000.  Calibration: on the same two data sets, the estimator with the same settings at
the empirical critical points next to the exact 5-gon (sixth bias -3), 5+-gon and 6-gon.
Writes figures/data/14-stages-tms-loss.dat, figures/data/14-stages-tms-llc.dat.
Run from the volume root: python figures/scripts/14-stages-tms.py   (about 100 s)
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
N = 1000
RUNS = 32
EPOCHS = 4500
BATCH = 20
LR = 0.01
EVERY = 2500                     # checkpoint spacing (SGD steps)
NBETA = N / np.log(N)
GAMMA = 1.0
EPS = 0.1 / NBETA
CHAINS, STEPS, BURN = 6, 6000, 1500
OUT_LOSS = os.path.join("figures", "data", "14-stages-tms-loss.dat")
OUT_LLC = os.path.join("figures", "data", "14-stages-tms-llc.dat")
KGON_ELL = {"4": 1 / 9, "4+": 1 / 9 - 1 / 144, "4++": 1 / 9 - 2 / 144, "5": 0.068744, "5+": 0.061799, "6": 0.048192}


# ---------------------------------------------------------------------------------------------
# Population loss (closed form, Chen et al. Lemma 3.1 in the book's derivation) -- for reporting
# ---------------------------------------------------------------------------------------------
def pop_ell(W, b):
    """ell = E||x - f||^2 for parameter batches W (S,2,c), b (S,c)."""
    c = W.shape[2]
    G = W.transpose(0, 2, 1) @ W
    beta = np.broadcast_to(b[:, :, None], G.shape)
    a = G
    eye = np.eye(c, dtype=bool)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        neg = beta <= 0
        act = neg & (a > 0) & (beta >= -a)
        T = np.where(act, (a + beta) ** 3 / a, 0.0)
        part = ~neg & (a < -beta)
        full = ~neg & ~part
        T = T + np.where(part, -beta ** 3 / a, 0.0) + np.where(full, a * a + 3 * a * beta + 3 * beta ** 2, 0.0)
        q = (1 - a) ** 2 - 3 * (1 - a) * beta + 3 * beta ** 2
        D = np.where(act, q + beta ** 3 * (1 / a ** 2 + 1 / a), np.where(neg, 1.0, q))
    return np.where(eye, D, T).sum((1, 2)) / (3 * c)


# ---------------------------------------------------------------------------------------------
# Exact empirical loss and gradient from prefix sums of sorted magnitudes
# ---------------------------------------------------------------------------------------------
class EmpiricalLoss:
    def __init__(self, feat, mu, c=C):
        self.c, self.n = c, len(mu)
        key = mu + 2.0 * feat                      # groups occupy disjoint intervals [2i, 2i+1]
        order = np.argsort(key)
        self.key = key[order]
        m = mu[order]
        self.P = np.zeros((3, self.n + 1))
        for p in range(3):
            self.P[p, 1:] = np.cumsum(m ** p)
        self.start = np.searchsorted(self.key, 2.0 * np.arange(c), side="left")
        self.end = np.searchsorted(self.key, 2.0 * np.arange(c) + 1.5, side="left")
        self.T = self.P[:, self.end] - self.P[:, self.start]

    def __call__(self, W, b):
        c = self.c
        G = W.transpose(0, 2, 1) @ W                # G[j, i]: output j, input i
        beta = np.broadcast_to(b[:, :, None], G.shape)
        a = G
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = np.where(a != 0, -beta / a, 0.0)
        tau = np.clip(tau, -0.25, 1.25)
        k = np.searchsorted(self.key, (2.0 * np.arange(c)[None, None, :] + tau).ravel(), side="right").reshape(a.shape)
        st = np.broadcast_to(self.start[None, None, :], a.shape)
        k = np.clip(k, st, self.end[None, None, :])
        low = self.P[:, k] - self.P[:, st]          # sums over mu <= tau
        T = self.T[:, None, None, :]
        Sx = np.where(a > 0, T - low, np.where(a < 0, low, np.where(beta > 0, T, 0.0)))
        S0, S1, S2 = Sx
        eye = np.eye(c, dtype=bool)[None]
        u = 1 - a
        off = a * a * S2 + 2 * a * beta * S1 + beta * beta * S0
        dia = self.T[2][None, None, :] - S2 + u * u * S2 - 2 * u * beta * S1 + beta * beta * S0
        da = np.where(eye, -2 * u * S2 + 2 * beta * S1, 2 * a * S2 + 2 * beta * S1)
        dbt = np.where(eye, -2 * u * S1 + 2 * beta * S0, 2 * a * S1 + 2 * beta * S0)
        s = 0.5 / self.n
        L = np.where(eye, dia, off).sum((1, 2)) * s
        dG = da * s
        return L, W @ (dG + dG.transpose(0, 2, 1)), dbt.sum(2) * s


def sgld_llc(loss, W0, b0, rng, chains=CHAINS, steps=STEPS, burn=BURN):
    """LLC estimates at each of the parameter sets W0 (K,2,c), b0 (K,c): array (K, chains)."""
    K = len(W0)
    Wc, bc = np.repeat(W0, chains, 0), np.repeat(b0, chains, 0)
    W, b = Wc.copy(), bc.copy()
    L0 = np.repeat(loss(W0, b0)[0], chains)
    acc = np.zeros(K * chains)
    for t in range(steps):
        L, dW, db = loss(W, b)
        if t >= burn:
            acc += L
        W += 0.5 * EPS * (-NBETA * dW - GAMMA * (W - Wc)) + np.sqrt(EPS) * rng.normal(size=W.shape)
        b += 0.5 * EPS * (-NBETA * db - GAMMA * (b - bc)) + np.sqrt(EPS) * rng.normal(size=b.shape)
    return (NBETA * (acc / (steps - burn) - L0)).reshape(K, chains)


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


def main():
    t_start = time.perf_counter()
    seeds = np.arange(RUNS)
    # data -----------------------------------------------------------------------------------
    feats, mus = [], []
    X = np.zeros((RUNS, N, C))
    for s in seeds:
        g = np.random.default_rng(1000 + s)
        i = g.integers(0, C, N)
        mu = g.random(N)
        feats.append(i); mus.append(mu)
        X[s, np.arange(N), i] = mu
    # initialization: the standard 4-gon plus N(0, 0.01^2) noise ------------------------------
    W = np.zeros((RUNS, R, C)); b = np.zeros((RUNS, C))
    for s in seeds:
        g = np.random.default_rng(2000 + s)
        for j in range(4):
            W[s, :, j] = [np.cos(np.pi / 2 * j), np.sin(np.pi / 2 * j)]
        W[s] += 0.01 * g.normal(size=(R, C))
        b[s] = 0.01 * g.normal(size=C)
    shuf = [np.random.default_rng(3000 + s) for s in seeds]
    # SGD ------------------------------------------------------------------------------------
    nb = N // BATCH
    ck_W, ck_b, ck_step = [], [], []
    fine_step, fine_ell = [], []
    ar = np.arange(RUNS)[:, None]
    step = 0
    for ep in range(EPOCHS):
        perm = np.stack([g.permutation(N) for g in shuf])
        Xp = X[ar, perm]
        for kb in range(nb):
            if step % 500 == 0 and step <= 150_000:
                fine_step.append(step); fine_ell.append(pop_ell(W, b))
            if step % EVERY == 0:
                ck_W.append(W.copy()); ck_b.append(b.copy()); ck_step.append(step)
            xb = Xp[:, kb * BATCH:(kb + 1) * BATCH]
            pre = xb @ W.transpose(0, 2, 1) @ W + b[:, None, :]
            mask = pre > 0
            dpre = np.where(mask, np.where(mask, pre, 0.0) - xb, 0.0) * (1.0 / BATCH)
            dG = xb.transpose(0, 2, 1) @ dpre
            W -= LR * (W @ (dG + dG.transpose(0, 2, 1)))
            b -= LR * dpre.sum(1)
            step += 1
    ck_W.append(W.copy()); ck_b.append(b.copy()); ck_step.append(step)
    ck_W, ck_b, ck_step = np.array(ck_W), np.array(ck_b), np.array(ck_step)   # (T,RUNS,...)
    t_train = time.perf_counter() - t_start
    ellt = np.array([pop_ell(ck_W[t], ck_b[t]) for t in range(len(ck_step))])   # (T,RUNS)
    print(f"trained {RUNS} runs x {step} SGD steps in {t_train:.1f}s")
    # transitions ----------------------------------------------------------------------------
    names = list(KGON_ELL)
    vals = np.array(list(KGON_ELL.values()))

    def label(e):
        j = np.argmin(np.abs(vals - e))
        return names[j] if abs(vals[j] - e) < 0.003 else "?"
    i5000 = list(ck_step).index(5000)
    drops = {}
    for s in seeds:
        base = ellt[i5000, s]
        idx = np.where(ellt[:, s] < base - 0.02)[0]
        if len(idx):
            drops[s] = ck_step[idx[0]]
        print(f"  run {s:2d}: loss at step 5000 {base:.5f} ({label(base)}), at the end {ellt[-1, s]:.5f}"
              f" ({label(ellt[-1, s])})" + (f", drop by 0.02 first seen at step {drops[s]}" if s in drops else ""))
    print(f"  {len(drops)} of {RUNS} runs dropped by more than 0.02 within {step} steps")
    sel = sorted(drops, key=lambda s: (drops[s], s))[:2]
    sel = sorted(sel, key=lambda s: drops[s])
    print(f"  selected runs (earliest drops): {sel} at steps {[int(drops[s]) for s in sel]}")
    last = int(min(step, (max(drops[s] for s in sel) // EVERY + 10) * EVERY))
    win = [t for t in range(len(ck_step)) if ck_step[t] <= last]
    # loss file: every 500 steps up to the window end, selected runs
    fine_ell = np.array(fine_ell)
    rows = [[int(st), fine_ell[j, sel[0]], fine_ell[j, sel[1]]] for j, st in enumerate(fine_step) if st <= last]
    sk.write_dat(OUT_LOSS, ["step", "ellA", "ellB"], rows)
    # LLC along the selected runs -----------------------------------------------------------
    rng = sk.rng(1414)
    est = {}
    cal = {}
    for s in sel:
        loss = EmpiricalLoss(feats[s], mus[s])
        t0 = time.perf_counter()
        est[s] = sgld_llc(loss, ck_W[win, s], ck_b[win, s], rng)
        # calibration: empirical critical points next to the exact k-gons
        Wk = np.stack([kgon(5, 1.1704596391, -0.282297283127)[0], kgon(5, 1.1704596391, -0.282297283127, pos=(5,))[0],
                       kgon(6, 1.32052633078, -0.618135131868)[0]])
        bk = np.stack([kgon(5, 1.1704596391, -0.282297283127)[1], kgon(5, 1.1704596391, -0.282297283127, pos=(5,))[1],
                       kgon(6, 1.32052633078, -0.618135131868)[1]])
        for it in range(1500):
            _, dW, db = loss(Wk, bk)
            Wk -= 1.0 * dW; bk -= 1.0 * db
        cal[s] = sgld_llc(loss, Wk, bk, rng)
        print(f"  run {s}: LLC at {len(win)} checkpoints and 3 k-gons in {time.perf_counter() - t0:.1f}s")
    rows = []
    for j, t in enumerate(win):
        row = [int(ck_step[t])]
        for s in sel:
            m, lo, hi = sk.mean_ci(est[s][j])
            row += [m, m - lo, hi - m]
        rows.append(row)
    sk.write_dat(OUT_LLC, ["step", "llcA", "loA", "hiA", "llcB", "loB", "hiB"], rows)
    for s in sel:
        print(f"== run {s}")
        for j, t in enumerate(win):
            m, lo, hi = sk.mean_ci(est[s][j])
            nrm = np.linalg.norm(ck_W[t, s], axis=0)
            print(f"  step {int(ck_step[t]):6d}  ell {ellt[t, s]:.5f} ({label(ellt[t, s]):3s})  lambda_hat {m:5.2f}  [{lo:5.2f}, {hi:5.2f}]"
                  f"   column norms {np.round(nrm, 2)}  biases {np.round(ck_b[t, s], 3)}")
        for name, k in (("5-gon", 0), ("5+-gon", 1), ("6-gon", 2)):
            m, lo, hi = sk.mean_ci(cal[s][k])
            print(f"  calibration at the {name}: {m:.2f} [{lo:.2f}, {hi:.2f}]")
        # plateau averages
        for nm in ("4", "4+", "4++", "5", "5+"):
            js = [j for j, t in enumerate(win) if label(ellt[t, s]) == nm and ck_step[t] >= 5000]
            if js:
                v = est[s][js].mean(1)
                m, lo, hi = sk.mean_ci(v)
                print(f"  plateau {nm:3s}: {len(js)} checkpoints, mean lambda_hat {m:.2f} [{lo:.2f}, {hi:.2f}] (interval over checkpoints)")
    print(f"wrote {OUT_LOSS}, {OUT_LLC}; total time {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
