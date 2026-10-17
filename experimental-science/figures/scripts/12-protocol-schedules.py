"""Lab 12.1 (12-protocol-schedules): the protocol behind the exponent.

A miniature compute-allocation study on the teacher task of labkit (d = 8, irreducible test
loss 0.01), in the online protocol of the book: a run with D examples makes one pass over D
fresh examples. The model family is the two-layer ReLU MLP of labkit, widths 4 to 91; its size
N is the full parameter count w^2 + 11w + 1, and the compute of a run is C = 6 N D. All runs use
Adam with batch size 64 and a learning rate that depends on the width only,
lr(w) = 0.032 (16/w)^(1/2), chosen from a small sweep with matched schedules.

Three protocols produce a loss for every cell (width, D) of one grid:
  C   a separate run for every cell, with a cosine schedule that ends at D (matched);
  K   one run per width with a cosine schedule over 2^15 steps (8 times the longest cell),
      read at the cells as checkpoints: the schedule is unfinished, as in Kaplan et al. (2020);
  KW  as K, with a linear warm-up over the first 393 steps (1.2% of the horizon, the ratio of
      Kaplan et al.'s 3000 warm-up steps to their 250,000 steps).
For each protocol the script builds isoFLOP curves by interpolating the loss of each width in
log D, locates the compute-optimal size N*(C) by a parabola in log N through the five widths
nearest the minimum, fits N* ~ C^a over the compute levels, and gives 95% percentile intervals
for a and for the paired differences between protocols from a bootstrap over seeds.

Run from the volume root:  python3 figures/scripts/12-protocol-schedules.py
Writes figures/data/12-protocol-schedules-*.dat and prints the numbers quoted in the text.
"""
import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk
import numpy as np
import torch
import torch.nn.functional as F

B = 64                                    # batch size (examples per step)
WIDTHS = [4, 6, 8, 11, 16, 23, 32, 45, 64, 91]
TGRID = [32 * 2 ** k for k in range(8)]   # run lengths in steps: 32 ... 4096
HORIZON = 2 ** 15                         # length of the unfinished schedule of K and KW
WARM = 393                                # warm-up steps of KW (1.2% of the horizon)
SEEDS = [0, 1, 2, 3, 4, 5]
CLO, CHI = 1.2e7, 7.7e8                   # range of compute levels (FLOPs)
LEVELS = CLO * 2.0 ** (np.arange(13) / 2)  # 13 levels, a factor sqrt(2) apart
NBOOT = 2000
OUT = "figures/data/12-protocol-schedules"


def n_full(w):
    return w * w + 11 * w + 1             # Linear(8,w) + Linear(w,w) + Linear(w,1), with biases


def lr_of(w):
    return 0.032 * (16.0 / w) ** 0.5


def band(w):
    """The run lengths (steps) of width w needed for the compute levels, plus one grid point
    on each side for interpolation."""
    lo = CLO / (6 * n_full(w) * B)
    hi = CHI / (6 * n_full(w) * B)
    ts = [t for t in TGRID if lo / 2 < t < hi * 2]
    return ts


def train_long(w, seed, Xtr, ytr, Xte, yte, horizon, stop, evals, warmup):
    """One run with a cosine schedule over `horizon` steps, stopped after `stop` steps, with the
    test loss recorded at the steps in `evals`. Same seeds and model as labkit.run."""
    cfg = lk.Config(data="teacher", width=w, optimizer="adam", lr=lr_of(w), batch_size=B,
                    schedule="cosine", warmup=warmup)
    torch.manual_seed(50_000 + seed)
    model = lk.MLP(Xtr.shape[1], 1, w, 2)
    opt = lk.make_optimizer(cfg, model.parameters())
    perm = torch.randperm(Xtr.shape[0], generator=torch.Generator().manual_seed(60_000 + seed))
    out, evs = {}, set(evals)
    for step in range(stop):
        idx = perm[step * B:(step + 1) * B]
        for group in opt.param_groups:
            group["lr"] = cfg.lr * lk.lr_factor(cfg, step, horizon)
        loss = F.mse_loss(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step + 1 in evs:
            with torch.no_grad():
                out[step + 1] = F.mse_loss(model(Xte), yte).item()
    return out


def train_matched(w, seed, Xtr, ytr, Xte, yte, T):
    """Protocol C: labkit's online protocol, one pass over the first B*T examples, cosine
    schedule matched to the run."""
    D = B * T
    cfg = lk.Config(data="teacher", train_size=D, epochs=1, schedule="cosine", width=w,
                    optimizer="adam", lr=lr_of(w), batch_size=B)
    return lk.run(cfg, init=seed, order=seed, data=(Xtr[:D], ytr[:D], Xte, yte))["test_loss"]


# ---------------------------------------------------------------------------------------
# isoFLOP analysis
# ---------------------------------------------------------------------------------------
def iso_points(table, levels, size, csize=None):
    """For each compute level C, the points (log size, log loss) of the widths whose runs bracket
    the run length T = C / (6 n B) at which the width spends C. `table[w]` maps run length to
    mean loss; `size[w]` is the size on the horizontal axis and `csize[w]` (default: the same)
    the size used to count compute. The loss is interpolated linearly in log T."""
    csize = csize or size
    pts = []
    for C in levels:
        xs, ys = [], []
        for w in WIDTHS:
            ts = sorted(table[w])
            T = C / (6 * csize[w] * B)
            if not ts or T < ts[0] or T > ts[-1]:
                continue
            y = np.interp(math.log(T), np.log(ts), np.log([table[w][t] for t in ts]))
            xs.append(math.log(size[w]))
            ys.append(y)
        pts.append((np.array(xs), np.array(ys)))
    return pts


def optimum(xs, ys, full=False):
    """Log of the minimizing size: the vertex of a parabola through the five points nearest the
    smallest loss. None unless the smallest loss is at an interior width and the vertex lies
    within the five points. With full=True, also return the parabola and the window."""
    if len(xs) < 4:
        return None
    k = int(np.argmin(ys))
    if k == 0 or k == len(xs) - 1:
        return None
    lo, hi = max(0, k - 2), min(len(xs), k + 3)
    c2, c1, c0 = np.polyfit(xs[lo:hi], ys[lo:hi], 2)
    if c2 <= 0:
        return None
    v = -c1 / (2 * c2)
    if not xs[lo] <= v <= xs[hi - 1]:
        return None
    return (v, (c2, c1, c0), lo, hi) if full else v


def exponent(table, levels, size, csize=None):
    xo = [optimum(*p) for p in iso_points(table, levels, size, csize)]
    ok = [i for i, v in enumerate(xo) if v is not None]
    if len(ok) < 4:
        return float("nan"), xo
    a, _ = np.polyfit(np.log(levels[ok]), np.array([xo[i] for i in ok]), 1)
    return a, xo


def mean_table(runs, proto, seeds):
    tab = {w: {} for w in WIDTHS}
    for w in WIDTHS:
        for t in band(w):
            tab[w][t] = float(np.mean([runs[(proto, s, w, t)] for s in seeds]))
    return tab


if __name__ == "__main__":
    t0 = time.time()
    _, _, Xte, yte = lk.teacher(16, n_test=8192, split=999)       # one test set for every run
    runs = {}
    steps_done = 0
    for s in SEEDS:
        Xtr, ytr, _, _ = lk.teacher(B * TGRID[-1], n_test=16, split=s)
        for w in WIDTHS:
            ts = band(w)
            for T in ts:
                runs[("C", s, w, T)] = train_matched(w, s, Xtr, ytr, Xte, yte, T)
            for proto, warm in (("K", 0), ("KW", WARM)):
                r = train_long(w, s, Xtr, ytr, Xte, yte, HORIZON, ts[-1], ts, warm)
                for T in ts:
                    runs[(proto, s, w, T)] = r[T]
            steps_done += sum(ts) + 2 * ts[-1]
        print(f"seed {s} done: {time.time() - t0:.0f} s", flush=True)
    # self-check: the long-run loop with a matched horizon reproduces labkit.run
    Xtr, ytr, _, _ = lk.teacher(B * TGRID[-1], n_test=16, split=0)
    chk = train_long(16, 0, Xtr[:B * 256], ytr[:B * 256], Xte, yte, 256, 256, [256], 0)[256]
    assert abs(chk - runs[("C", 0, 16, 256)]) < 1e-6, (chk, runs[("C", 0, 16, 256)])
    print(f"training: {steps_done} steps in {time.time() - t0:.0f} s")

    size = {w: n_full(w) for w in WIDTHS}
    # every run
    rows = sorted(runs)
    pcode = {"C": 0, "K": 1, "KW": 2}
    lk.write_dat(OUT + "-runs.dat", {
        "proto": [pcode[r[0]] for r in rows], "seed": [r[1] for r in rows],
        "width": [r[2] for r in rows], "N": [size[r[2]] for r in rows],
        "steps": [r[3] for r in rows], "D": [B * r[3] for r in rows],
        "loss": [runs[r] for r in rows]})

    # point estimates and bootstrap over seeds (the same resampled seeds for all protocols)
    rng = np.random.default_rng(12)
    est, xo_all, boot, boot_xo = {}, {}, {p: [] for p in pcode}, {p: [] for p in pcode}
    for p in pcode:
        est[p], xo_all[p] = exponent(mean_table(runs, p, SEEDS), LEVELS, size)
    for b in range(NBOOT):
        ss = list(rng.choice(SEEDS, size=len(SEEDS), replace=True))
        for p in pcode:
            a, xo = exponent(mean_table(runs, p, ss), LEVELS, size)
            boot[p].append(a)
            boot_xo[p].append([np.nan if v is None else v for v in xo])
    q = lambda v: (np.nanquantile(v, 0.025), np.nanquantile(v, 0.975))
    print("allocation exponent a (N* ~ C^a), 95% percentile intervals from a bootstrap over seeds:")
    for p in pcode:
        lo, hi = q(boot[p])
        print(f"  {p:2s}: a = {est[p]:.3f}  [{lo:.3f}, {hi:.3f}]  "
              f"({sum(v is not None for v in xo_all[p])} of {len(LEVELS)} levels bracketed)")
    for p1, p2 in (("K", "C"), ("KW", "C"), ("KW", "K")):
        d = np.array(boot[p1]) - np.array(boot[p2])
        lo, hi = q(d)
        print(f"  {p1}-{p2}: {est[p1] - est[p2]:+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"(fraction of resamples > 0: {np.mean(d > 0):.3f})")

    # optima with intervals, fitted lines, and the loss at the optimum
    for p in pcode:
        pts = iso_points(mean_table(runs, p, SEEDS), LEVELS, size)
        xo = xo_all[p]
        bx = np.array(boot_xo[p])
        ok = [i for i, v in enumerate(xo) if v is not None]
        a, b0 = np.polyfit(np.log(LEVELS[ok]), np.array([xo[i] for i in ok]), 1)
        lstar = []
        for i in ok:
            v, (c2, c1, c0), _, _ = optimum(*pts[i], full=True)
            lstar.append(math.exp(c0 + c1 * v + c2 * v * v))
        cols = {"C": LEVELS[ok], "Nstar": [math.exp(xo[i]) for i in ok],
                "lo": [math.exp(np.nanquantile(bx[:, i], 0.025)) for i in ok],
                "hi": [math.exp(np.nanquantile(bx[:, i], 0.975)) for i in ok],
                "fit": np.exp(b0 + a * np.log(LEVELS[ok])), "Lstar": lstar}
        lk.write_dat(OUT + f"-opt-{p}.dat", cols)
        print(f"  {p:2s} optima N*: " + " ".join(f"{c:.1e}:{n:.0f}" for c, n in zip(cols["C"], cols["Nstar"])))
        print(f"     loss at optimum: " + " ".join(f"{v:.4f}" for v in lstar))

    # how far the checkpoints moved the optima: geometric mean over levels of N*_P / N*_C, against
    # the shift 2^a that doubling the area under the schedule predicts (heuristic of Section 12.5)
    for p in ("K", "KW"):
        both = [i for i in range(len(LEVELS)) if xo_all[p][i] is not None and xo_all["C"][i] is not None]
        g = math.exp(np.mean([xo_all[p][i] - xo_all["C"][i] for i in both]))
        bg = []
        for bxp, bxc in zip(boot_xo[p], boot_xo["C"]):
            d = np.array(bxp) - np.array(bxc)
            if np.isfinite(d).sum() >= 4:
                bg.append(math.exp(np.nanmean(d)))
        lo, hi = q(bg)
        print(f"  N*_{p}/N*_C, geometric mean over levels: {g:.2f} [{lo:.2f}, {hi:.2f}];"
              f" 2^a_C = {2 ** est['C']:.2f}")

    # isoFLOP curves of protocols C and K at four levels: per-seed points, means, intervals
    show = [0, 6, 12]
    tabs = {p: {s: mean_table(runs, p, [s]) for s in SEEDS} for p in ("C", "K")}
    for p in ("C", "K"):
        cols = {k: [] for k in ("level", "C", "N", "loss", "hw", "fit")}
        rcols = {k: [] for k in ("level", "N", "loss")}
        mpts = iso_points(mean_table(runs, p, SEEDS), LEVELS, size)
        for j in show:
            C = LEVELS[j]
            per = [iso_points(tabs[p][s], [C], size)[0] for s in SEEDS]
            xs, ys = mpts[j]
            res = optimum(xs, ys, full=True)
            for i, x in enumerate(xs):
                vals = [math.exp(pp[1][i]) for pp in per]
                m, hw = lk.t_interval(vals)
                fit = float("nan")
                if res is not None and res[2] <= i < res[3]:
                    c2, c1, c0 = res[1]
                    fit = math.exp(c0 + c1 * x + c2 * x * x)
                cols["level"].append(j); cols["C"].append(C); cols["N"].append(math.exp(x))
                cols["loss"].append(m); cols["hw"].append(hw); cols["fit"].append(fit)
                for v in vals:
                    rcols["level"].append(j); rcols["N"].append(math.exp(x)); rcols["loss"].append(v)
        lk.write_dat(OUT + f"-iso-{p}.dat", cols)
        lk.write_dat(OUT + f"-isoruns-{p}.dat", rcols)

    # the penalties: ratio of K and KW checkpoints to matched runs. Per cell: geometric mean over
    # seeds. Per run length: geometric mean over the widths, with a t-interval over widths.
    cell = {k: [] for k in ("width", "N", "D", "rK", "rW", "sdK", "sdC")}
    for w in WIDTHS:
        for t in band(w):
            lr_k = [math.log(runs[("K", s, w, t)] / runs[("C", s, w, t)]) for s in SEEDS]
            lr_w = [math.log(runs[("KW", s, w, t)] / runs[("C", s, w, t)]) for s in SEEDS]
            lc = [runs[("C", s, w, t)] for s in SEEDS]
            lkk = [runs[("K", s, w, t)] for s in SEEDS]
            cell["width"].append(w); cell["N"].append(size[w]); cell["D"].append(B * t)
            cell["rK"].append(math.exp(np.mean(lr_k))); cell["rW"].append(math.exp(np.mean(lr_w)))
            cell["sdC"].append(np.std(lc, ddof=1) / np.mean(lc)); cell["sdK"].append(np.std(lkk, ddof=1) / np.mean(lkk))
    lk.write_dat(OUT + "-penalty-cells.dat", cell)
    dd = np.array(cell["D"])
    agg = {k: [] for k in ("D", "rK", "rKlo", "rKhi", "rW", "rWlo", "rWhi", "nw")}
    print("ratio of checkpoint to matched loss by run length (geometric mean over widths, 95% t-interval):")
    for D in sorted(set(dd)):
        m = dd == D
        mk, hk = lk.t_interval(np.log(np.array(cell["rK"])[m]))
        mw, hw_ = lk.t_interval(np.log(np.array(cell["rW"])[m]))
        agg["D"].append(D); agg["nw"].append(int(m.sum()))
        agg["rK"].append(math.exp(mk)); agg["rKlo"].append(math.exp(mk - hk)); agg["rKhi"].append(math.exp(mk + hk))
        agg["rW"].append(math.exp(mw)); agg["rWlo"].append(math.exp(mw - hw_)); agg["rWhi"].append(math.exp(mw + hw_))
        print(f"  D={D:7d} ({int(m.sum())} widths): K/C {math.exp(mk):.3f} [{math.exp(mk-hk):.3f}, {math.exp(mk+hk):.3f}]"
              f"   KW/C {math.exp(mw):.3f} [{math.exp(mw-hw_):.3f}, {math.exp(mw+hw_):.3f}]")
    lk.write_dat(OUT + "-penalty.dat", agg)
    print(f"run-to-run coefficient of variation, median over cells: matched {100*np.median(cell['sdC']):.1f}%,"
          f" checkpoints {100*np.median(cell['sdK']):.1f}%")
    print(f"total time {time.time() - t0:.0f} s")
