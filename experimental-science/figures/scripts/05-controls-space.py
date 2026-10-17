"""Lab 5.1 (Chapter 5, Controls): the spread of a search space.

A tiny tabular benchmark in the spirit of NAS-Bench-101. The search space is every MLP
for the digits data with

    depth      1, 2, 3, 4 hidden layers
    width      16, 32, 64, 128 units
    activation relu, tanh, gelu
    dropout    0, 0.25
    batchnorm  off, on

(4 x 4 x 3 x 2 x 2 = 192 architectures). The training pipeline is fixed for all of them:
Adam, learning rate 1e-3, batch 64, 30 epochs, constant learning rate. The data are the
digits split 0 of labkit; its 1258 training images are divided once (seed 5) into 958 for
training and 300 for validation, and the 539 test images are used only here, to define the
truth. The true quality of an architecture is its mean test error over six runs with
fresh seeds (init = order = noise = 2000 + s, s = 0..5). The searches of Lab 5.2 never see
these numbers; they look only at validation errors of separate runs.

Writes
  figures/data/05-controls-space.dat       one row per architecture, sorted by mean error
  figures/data/05-controls-space-runs.dat  one row per run (rank, test error)
and prints the numbers quoted in the chapter. Runtime three to four minutes on two threads.
Run from the volume root: python3 figures/scripts/05-controls-space.py
"""
import os
import sys
import time
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk  # noqa: E402
import numpy as np  # noqa: E402

DEPTHS = (1, 2, 3, 4)
WIDTHS = (16, 32, 64, 128)
ACTS = ("relu", "tanh", "gelu")
DROPS = (0.0, 0.25)
BNS = (False, True)
SPACE = list(itertools.product(DEPTHS, WIDTHS, ACTS, DROPS, BNS))
TRUTH_SEEDS = [2000 + s for s in range(6)]
EPOCHS = 30


def data_splits():
    """(search data, truth data): train on 958 images; validate on 300; test on 539."""
    Xtr, ytr, Xte, yte = lk.digits(split=0)
    perm = np.random.default_rng(5).permutation(len(ytr))
    tr, va = perm[:958], perm[958:]
    search = (Xtr[tr], ytr[tr], Xtr[va], ytr[va])
    truth = (Xtr[tr], ytr[tr], Xte, yte)
    return search, truth


def config(arch, epochs=EPOCHS, eval_every=None):
    depth, width, act, drop, bn = arch
    return lk.Config(depth=depth, width=width, activation=act, dropout=drop, batchnorm=bn,
                     optimizer="adam", lr=1e-3, batch_size=64, epochs=epochs,
                     eval_every=eval_every)


def curated(arch):
    """The curated subspace: the widths a practitioner would try first."""
    return arch[1] >= 64


def main():
    t0 = time.time()
    _, truth = data_splits()
    errs = np.zeros((len(SPACE), len(TRUTH_SEEDS)))
    for i, arch in enumerate(SPACE):
        cfg = config(arch)
        for j, s in enumerate(TRUTH_SEEDS):
            errs[i, j] = lk.run(cfg, init=s, order=s, noise=s, data=truth)["test_err"]
    n = errs.shape[1]
    mean, sd = errs.mean(1), errs.std(1, ddof=1)
    hw = np.array([lk.t_interval(e)[1] for e in errs])
    order = np.argsort(mean, kind="stable")
    rank = np.empty(len(SPACE), dtype=int)
    rank[order] = np.arange(1, len(SPACE) + 1)
    cur = np.array([curated(a) for a in SPACE])

    cols = {"rank": rank[order], "id": order,
            "depth": [SPACE[i][0] for i in order], "width": [SPACE[i][1] for i in order],
            "act": [SPACE[i][2] for i in order], "dropout": [SPACE[i][3] for i in order],
            "bn": [int(SPACE[i][4]) for i in order], "curated": cur[order].astype(int),
            "mean": mean[order], "sd": sd[order], "hw": hw[order]}
    for j in range(n):
        cols[f"e{j + 1}"] = errs[order, j]
    # a column that is the mean for curated architectures and nan otherwise (for plotting)
    cols["meancur"] = np.where(cur[order], mean[order], np.nan)
    cols["meanwide"] = np.where(~cur[order], mean[order], np.nan)
    lk.write_dat("figures/data/05-controls-space.dat", cols)
    rr = np.repeat(rank[order], n)
    lk.write_dat("figures/data/05-controls-space-runs.dat",
                 {"rank": rr, "err": errs[order].ravel(),
                  "curated": np.repeat(cur[order].astype(int), n)})
    # the two placebo levels (mean error of a random architecture), as horizontal lines
    lk.write_dat("figures/data/05-controls-space-ref.dat",
                 {"x": [0, len(SPACE) + 1], "full": [mean.mean()] * 2,
                  "curated": [mean[cur].mean()] * 2})

    def summary(mask, name):
        m, e = mean[mask], errs[mask]
        within = (e.var(1, ddof=1)).mean()               # pooled seed variance
        between = max(m.var(ddof=1) - within / n, 0.0)   # one-way random effects (ANOVA)
        best = np.argmin(np.where(mask, mean, np.inf))
        print(f"[{name}] architectures {mask.sum()}: mean of means {100 * m.mean():.2f}%, "
              f"SD of means {100 * m.std(ddof=1):.2f} pp, seed SD {100 * np.sqrt(within):.2f} pp, "
              f"between-architecture SD (corrected) {100 * np.sqrt(between):.2f} pp")
        print(f"[{name}]   best (table) {SPACE[best]}: {100 * mean[best]:.2f}% "
              f"+- {100 * hw[best]:.2f}; worst {100 * m.max():.2f}%; "
              f"headroom mean - best {100 * (m.mean() - mean[best]):.2f} pp")
        q = np.quantile(m, [0.05, 0.25, 0.5, 0.75, 0.95])
        print(f"[{name}]   quantiles of the mean error (5,25,50,75,95%): "
              + ", ".join(f"{100 * x:.2f}" for x in q))
        # how many architectures are statistically indistinguishable from the table best
        from scipy import stats
        tied = 0
        for i in np.where(mask)[0]:
            if i == best:
                continue
            p = stats.ttest_ind(errs[i], errs[best], equal_var=False).pvalue
            tied += p > 0.05
        print(f"[{name}]   architectures not distinguishable from the best by Welch t (p>0.05): "
              f"{tied} of {mask.sum() - 1}")
        return m.mean(), np.sqrt(between), np.sqrt(within)

    print(f"runs: {errs.size} ({len(SPACE)} architectures x {n} seeds), "
          f"{time.time() - t0:.0f} s")
    summary(np.ones(len(SPACE), bool), "full space")
    summary(cur, "curated, width >= 64")
    by = lambda k, vals: [(v, 100 * mean[[a[k] == v for a in SPACE]].mean()) for v in vals]
    print("mean error by width:", ", ".join(f"{v}: {x:.2f}%" for v, x in by(1, WIDTHS)))
    print("mean error by depth:", ", ".join(f"{v}: {x:.2f}%" for v, x in by(0, DEPTHS)))
    print("mean error by activation:", ", ".join(f"{v}: {x:.2f}%" for v, x in by(2, ACTS)))
    print("mean error by batchnorm:", ", ".join(f"{v}: {x:.2f}%" for v, x in by(4, BNS)))
    print("mean error by dropout:", ", ".join(f"{v}: {x:.2f}%" for v, x in by(3, DROPS)))
    textbook = SPACE.index((2, 64, "relu", 0.0, False))
    print(f"textbook MLP (2 x 64, relu): {100 * mean[textbook]:.2f}% +- {100 * hw[textbook]:.2f}, "
          f"rank {rank[textbook]}")


if __name__ == "__main__":
    main()
