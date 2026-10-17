"""Lab 14.1: digits as tokens.

A family of networks of increasing width answers sequences of L test digits; an answer
counts as an exact match only if every digit of the sequence is classified correctly. The
same networks are scored by per-digit accuracy (L = 1), by exact match on L = 3 and L = 10
digits, and by the per-digit log-likelihood of the correct label. A Gumbel curve
(Proposition 14.1 with an irreducible per-digit error E) is fitted to per-digit accuracy
alone and predicts the exact-match curves for L = 3 and L = 10 with no further parameters.

Family: the recipe of Chapter 1 (two hidden ReLU layers, SGD with momentum 0.9, learning
rate 0.05, batch 64, 20 epochs) with widths 1 to 512; split 0 (1,258 training and 539 test
digits); runs s = 0..19 use seed s for initialization and data order.

Exact match on L digits is computed exactly over all 539^L sequences of L test digits (drawn
with replacement): for a given network it is p^L, with p the network's accuracy on the 539
digits. Lab 14.2 measures it instead on finite sets of sequences.

Run from the volume root: python3 figures/scripts/14-emergence-tokens.py  (about 40 s)
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares
import labkit as lk

WIDTHS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512]
SEEDS = range(20)
LS = (1, 3, 10)
OUT = "figures/data/14-emergence-tokens"


def per_item(model, X, y):
    """Per-item correctness (argmax) and log-probability of the correct label."""
    model.eval()
    with torch.no_grad():
        lp = F.log_softmax(model(X), 1)
    model.train()
    correct = (lp.argmax(1) == y).double().numpy()
    loglik = lp[torch.arange(len(y)), y].double().numpy()
    return correct, loglik


def fit_gumbel(N, p):
    """Fit -ln p = E + c N^(-alpha) by least squares on log(-ln p). Returns (E, c, alpha)."""
    lnN, z = np.log(N), np.log(-np.log(p))

    def res(th):
        lnE, lnc, a = th
        return np.logaddexp(lnE, lnc - a * lnN) - z

    best = None
    for a0 in (1.0, 1.5, 2.0, 3.0):
        for lnc0 in (4.0, 8.0, 12.0):
            f = least_squares(res, x0=[math.log(0.03), lnc0, a0])
            if best is None or f.cost < best.cost:
                best = f
    lnE, lnc, a = best.x
    return math.exp(lnE), math.exp(lnc), a, float(np.sqrt(np.mean(best.fun ** 2)))


def em_curve(N, E, c, a, L):
    return np.exp(-L * (E + c * np.power(N, -a)))


def main():
    t0 = time.time()
    data = lk.load(lk.Config())
    Xte, yte = data[2], data[3]
    rows = []
    for w in WIDTHS:
        for s in SEEDS:
            cfg = lk.Config(width=w, optimizer="momentum", lr=0.05, batch_size=64, epochs=20)
            r = lk.run(cfg, init=s, order=s, data=data, return_model=True)
            corr, ll = per_item(r["model"], Xte, yte)
            rows.append(dict(w=w, N=r["params"], seed=s, acc=corr.mean(), ce=-ll.mean()))
    print(f"trained {len(rows)} networks in {time.time() - t0:.1f} s")

    N = np.array([r["N"] for r in rows], float)
    acc = np.array([r["acc"] for r in rows])
    ce = np.array([r["ce"] for r in rows])
    em = {L: acc ** L for L in LS}

    # individual runs
    lk.write_dat(OUT + "-runs.dat", {
        "w": [r["w"] for r in rows], "N": N, "seed": [r["seed"] for r in rows],
        "acc": acc, "em3": em[3], "em10": em[10], "ce": ce,
        "nl1": -np.log(em[1]), "nl3": -np.log(em[3]), "nl10": -np.log(em[10])})

    # means over the 20 runs of each width, with 95% t-intervals
    Ns = np.array([rows[i * len(SEEDS)]["N"] for i in range(len(WIDTHS))], float)
    cols = {"w": WIDTHS, "N": Ns}
    for name, v in [("acc", acc), ("em3", em[3]), ("em10", em[10]), ("ce", ce),
                    ("nl1", -np.log(em[1])), ("nl3", -np.log(em[3])), ("nl10", -np.log(em[10]))]:
        m, h = zip(*[lk.t_interval(v[N == n]) for n in Ns])
        cols[name], cols[name + "_hw"] = np.array(m), np.array(h)
    pbar = cols["acc"]

    # the Gumbel curve with a floor, fitted to per-digit accuracy (L = 1) only
    E, c, a, rms = fit_gumbel(Ns, pbar)
    rng = np.random.default_rng(14)
    boot = []
    for _ in range(1000):
        pb = np.array([rng.choice(acc[N == n], size=len(SEEDS)).mean() for n in Ns])
        boot.append(fit_gumbel(Ns, pb)[:3])
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    print(f"fit to per-digit accuracy: E = {E:.4f} ({lo[0]:.4f}-{hi[0]:.4f}), c = {c:.4g}, "
          f"alpha = {a:.3f} ({lo[2]:.3f}-{hi[2]:.3f}); rms residual on log(-ln p) {rms:.3f}")
    for L in LS:
        mu = (math.log(L * c)) / a                      # location in ln N
        print(f"  L={L:2d}: midpoint N = {math.exp(mu):7.0f} (EM there = {math.exp(-1 - L * E):.3f}, "
              f"ceiling e^(-LE) = {math.exp(-L * E):.3f})")
    print(f"  shift L=1 -> L=10: factor {10 ** (1 / a):.2f} in N; 10-90% width {3.0844 / a:.2f} in ln N "
          f"(factor {math.exp(3.0844 / a):.1f} in N); steepest slope alpha/e = {a / math.e:.3f} per unit ln N")

    for L in LS:
        cols[f"pred{L}"] = em_curve(Ns, E, c, a, L)
    lk.write_dat(OUT + "-mean.dat", cols)

    grid = np.exp(np.linspace(math.log(70), math.log(4e5), 160))
    fit = {"N": grid}
    for L in LS:
        fit[f"em{L}"] = em_curve(grid, E, c, a, L)
        fit[f"nl{L}"] = L * (E + c * grid ** (-a))
    lk.write_dat(OUT + "-fit.dat", fit)

    print("\nwidth      N   acc            EM3            EM10              pred3   pred10   CE")
    for i, w in enumerate(WIDTHS):
        print(f"{w:5d} {int(Ns[i]):6d}  {cols['acc'][i]:.4f}+-{cols['acc_hw'][i]:.4f}  "
              f"{cols['em3'][i]:.4f}+-{cols['em3_hw'][i]:.4f}  {cols['em10'][i]:.2e}+-{cols['em10_hw'][i]:.1e}  "
              f"{cols['pred3'][i]:.4f}  {cols['pred10'][i]:.2e}  {cols['ce'][i]:.3f}+-{cols['ce_hw'][i]:.3f}")
    for L in (3, 10):
        inside = np.abs(cols[f"em{L}"] - cols[f"pred{L}"]) <= cols[f"em{L}_hw"]
        dev = np.abs(cols[f"em{L}"] - cols[f"pred{L}"])
        print(f"L={L}: prediction inside the 95% interval at {inside.sum()}/{len(WIDTHS)} widths; "
              f"max |measured - predicted| = {dev.max():.3f} at width {WIDTHS[int(dev.argmax())]}")
    # what a single-run, factor-of-two ladder of model sizes would show
    print("\nseed 0, widths 1,2,4,8,16,32,64: per-digit acc / EM10 / CE")
    for w in (1, 2, 4, 8, 16, 32, 64):
        r = [x for x in rows if x["w"] == w and x["seed"] == 0][0]
        print(f"  w={w:3d} N={r['N']:6d} acc={r['acc']:.3f} EM10={r['acc'] ** 10:.4f} CE={r['ce']:.3f}")
    # Jensen: mean of p^10 against (mean p)^10
    for i, w in enumerate(WIDTHS[:4]):
        print(f"  w={w}: mean EM10 {cols['em10'][i]:.2e} vs (mean acc)^10 {pbar[i] ** 10:.2e}; "
              f"runs' acc range {acc[N == Ns[i]].min():.3f}-{acc[N == Ns[i]].max():.3f}")
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
