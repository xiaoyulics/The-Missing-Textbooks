"""Lab 11.1: a grid of isoFLOP slices on the teacher task under the online protocol.

Each run trains a two-layer ReLU MLP of width w (N = w^2 + 11 w + 1 parameters) with Adam for
one pass over D fresh teacher examples (batch 64, so D/64 steps) with a cosine schedule matched
to D: labkit's Config(data="teacher", train_size=D, epochs=1, schedule="cosine"). Compute is
C = 6 N D. The grid is a union of isoFLOP slices: for each of seven budgets K = C/6 = N D from
10^6 to 10^9 (half-decade steps), the widths whose N lies within a factor 6 of a pilot estimate
of the compute-optimal size, with 4096 <= D <= 2^20 (D rounded to a multiple of 64).

Learning rate: in each cell it is chosen among 0.5, 1 and 2 times the rule
min(0.1, 2/w) (4096/D)^0.3 by the test loss of one tuning run (seed 0); three further runs
(seeds 1, 2, 3: initialization, data order and training sample all change) are then trained
with the chosen rate. Only seeds 1-3 enter the analysis. The test loss is the mean squared
error on 16,384 noise-free test inputs common to all runs, plus the noise variance 0.01, so it
estimates the population loss and its irreducible part is exactly E = 0.01.

Writes figures/data/11-fitting-runs.dat (measured runs) and 11-fitting-tuning.dat (tuning runs).
Run from the volume root: python3 figures/scripts/11-fitting-grid.py   (about 3-4 minutes)
"""
import os
import sys
import time

for _v in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import labkit as lk

WIDTHS = [3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 90, 128, 181, 256]
LOG10_BUDGETS = [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]          # K = C/6 = N D
PILOT = dict(E=0.009, AN=0.377, alN=0.673, AD=61.6, alD=0.882)  # rough law from a pilot grid
SEEDS = [1, 2, 3]
NOISE_VAR = 0.01


def n_params(w):
    return w * w + 11 * w + 1                                    # 8w+w, w*w+w, w+1


def lr_rule(w, D):
    return min(0.1, 2.0 / w) * (4096.0 / D) ** 0.3


def cells():
    p = PILOT
    a = p["alD"] / (p["alN"] + p["alD"])
    G = (p["alN"] * p["AN"] / (p["alD"] * p["AD"])) ** (1.0 / (p["alN"] + p["alD"]))
    out = []
    for k in LOG10_BUDGETS:
        K = 10.0 ** k
        n_opt = G * K ** a
        for w in WIDTHS:
            N = n_params(w)
            D = int(round(K / N / 64.0)) * 64
            if 4096 <= D <= 2 ** 20 and n_opt / 6 <= N <= 6 * n_opt:
                out.append((k, w, N, D))
    return out


def main():
    t_start = time.time()
    _, _, Xte, yte = lk.teacher(1, n_test=16384, noise_sd=0.0, split=999_999)
    grid = cells()
    tun = {c: [] for c in ("logK", "width", "N", "D", "mult", "lr", "loss")}
    rec = {c: [] for c in ("logK", "width", "N", "D", "C", "seed", "lr", "loss", "steps", "sec")}
    chosen = []
    for k, w, N, D in grid:
        Xtr, ytr, _, _ = lk.teacher(D, n_test=1, split=0)
        best = None
        for m in (0.5, 1.0, 2.0):
            cfg = lk.Config(data="teacher", train_size=D, epochs=1, schedule="cosine", width=w,
                            optimizer="adam", lr=m * lr_rule(w, D), batch_size=64)
            r = lk.run(cfg, init=0, order=0, data=(Xtr, ytr, Xte, yte))
            loss = r["test_loss"] + NOISE_VAR if np.isfinite(r["test_loss"]) else np.inf
            for c, v in zip(tun, (k, w, N, D, m, cfg.lr, loss)):
                tun[c].append(v)
            if best is None or loss < best[1]:
                best = (m, loss)
        m = best[0]
        chosen.append(m)
        for s in SEEDS:
            Xtr, ytr, _, _ = lk.teacher(D, n_test=1, split=s)
            cfg = lk.Config(data="teacher", train_size=D, epochs=1, schedule="cosine", width=w,
                            optimizer="adam", lr=m * lr_rule(w, D), batch_size=64)
            r = lk.run(cfg, init=s, order=s, data=(Xtr, ytr, Xte, yte))
            assert r["params"] == N
            vals = (k, w, N, D, 6.0 * N * D, s, cfg.lr, r["test_loss"] + NOISE_VAR, r["steps"], r["seconds"])
            for c, v in zip(rec, vals):
                rec[c].append(v)
        losses = rec["loss"][-len(SEEDS):]
        print(f"log10 K={k:.1f} w={w:4d} N={N:6d} D={D:8d} lr x{m:g} "
              f"loss {np.mean(losses):.5f} (sd {np.std(losses, ddof=1):.5f})", flush=True)

    lk.write_dat("figures/data/11-fitting-runs.dat", rec)
    lk.write_dat("figures/data/11-fitting-tuning.dat", tun)
    L = np.array(rec["loss"])
    print(f"\ncells {len(grid)}, measured runs {len(L)}, tuning runs {len(tun['loss'])}")
    print("chosen multipliers: " + ", ".join(f"x{m:g}: {chosen.count(m)}" for m in (0.5, 1.0, 2.0)))
    print(f"N from {min(rec['N'])} to {max(rec['N'])}, D from {min(rec['D'])} to {max(rec['D'])}, "
          f"C from {min(rec['C']):.3g} to {max(rec['C']):.3g}")
    print(f"loss from {L.min():.5f} to {L.max():.5f}")
    # replicate spread: sd of log loss within cells
    lnL = np.log(L).reshape(-1, len(SEEDS))
    sd = lnL.std(axis=1, ddof=1)
    print(f"within-cell sd of log loss: median {np.median(sd):.4f}, pooled {np.sqrt(np.mean(sd**2)):.4f}")
    print(f"total time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
