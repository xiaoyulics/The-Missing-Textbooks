"""Epilogue: one experiment, run by the book's protocols from question to report.

Question: does label smoothing (epsilon = 0.1) lower the test error of the digits MLP?

The predictions below were written into this file before any of its runs were made; the
script prints them next to what happened.

Stages
  1. pilot: 10 paired seeds per arm on a validation split, to estimate the spread of the
     paired difference and choose the number of seeds;
  2. tuning with equal budgets: the same grid of learning rates and weight decays for both
     arms, 3 seeds per configuration, scored on the validation split only;
  3. confirmation: n paired seeds per arm at each arm's tuned configuration, scored once on
     the test set;
  4. positive control: the tuned baseline trained on half of the training set, which must
     come out visibly worse if the pipeline can detect an effect at all;
  5. a 2x2 factorial check (label smoothing x weight decay) at the baseline's tuned learning
     rate, for an interaction.
Writes figures/data/16-epilogue-*.dat and prints every number quoted in the epilogue.
Runtime: about a minute on a laptop CPU.
"""
import itertools
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk

PREDICTIONS = {
    "effect of label smoothing on test error (points, tuned arms)": "between -0.5 and -0.1 (it helps a little)",
    "half-width of the 95% interval for that effect with ~20 paired seeds": "about 0.15 points",
    "positive control (half the training data)": "worse by at least 1 point",
    "interaction of label smoothing with weight decay": "smaller than 0.2 points in absolute value",
}

BASE = dict(width=64, depth=2, optimizer="momentum", epochs=30, batch_size=64)
LRS = [0.01, 0.02, 0.05, 0.1, 0.2]
WDS = [0.0, 1e-4]
EPS = {"baseline": 0.0, "smoothing": 0.1}


def split_validation(data, n_val=300, seed=0):
    Xtr, ytr, Xte, yte = data
    g = torch.Generator().manual_seed(90_000 + seed)
    perm = torch.randperm(len(ytr), generator=g)
    va, tr = perm[:n_val], perm[n_val:]
    return Xtr[tr], ytr[tr], Xtr[va], ytr[va]


def err(cfg, seed, data):
    return lk.run(cfg, init=seed, order=seed, data=data)["test_err"] * 100   # in points


full = lk.digits(split=0)
val = split_validation(full)
half = (full[0][: len(full[1]) // 2], full[1][: len(full[1]) // 2], full[2], full[3])

# 1. pilot ---------------------------------------------------------------------------------
pilot = {arm: np.array([err(lk.Config(**BASE, lr=0.05, label_smoothing=e), s, val) for s in range(10)])
         for arm, e in EPS.items()}
d_pilot = pilot["smoothing"] - pilot["baseline"]
sd_pilot = d_pilot.std(ddof=1)
delta = 0.25                                          # smallest effect worth detecting, points
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
n_needed = int(np.ceil((z * sd_pilot / delta) ** 2))
n = max(20, n_needed)
print(f"pilot (validation, lr=0.05): baseline {pilot['baseline'].mean():.2f}, smoothing {pilot['smoothing'].mean():.2f}; "
      f"paired difference sd {sd_pilot:.3f} points -> n >= {n_needed} for delta={delta} at power 0.8; using n = {n}")

# 2. tuning with equal budgets -------------------------------------------------------------
grid = list(itertools.product(LRS, WDS))
tune = {}
for arm, e in EPS.items():
    scores = []
    for lr, wd in grid:
        cfg = lk.Config(**BASE, lr=lr, weight_decay=wd, label_smoothing=e)
        scores.append(np.mean([err(cfg, 100 + s, val) for s in range(3)]))
    best = int(np.argmin(scores))
    tune[arm] = dict(lr=grid[best][0], wd=grid[best][1], val=scores[best], scores=scores)
    print(f"tuning {arm}: best lr={grid[best][0]}, wd={grid[best][1]} (validation error {scores[best]:.2f}); "
          f"{len(grid)} configurations x 3 seeds")
lk.write_dat("figures/data/16-epilogue-tuning.dat",
             {"lr": [g[0] for g in grid], "wd": [g[1] for g in grid],
              "baseline": tune["baseline"]["scores"], "smoothing": tune["smoothing"]["scores"]})

# 3. confirmation on the test set, paired seeds ---------------------------------------------
seeds = range(1000, 1000 + n)
conf = {}
for arm, e in EPS.items():
    cfg = lk.Config(**BASE, lr=tune[arm]["lr"], weight_decay=tune[arm]["wd"], label_smoothing=e)
    conf[arm] = np.array([err(cfg, s, full) for s in seeds])
d = conf["smoothing"] - conf["baseline"]
m, h = lk.t_interval(d)
rng = np.random.default_rng(0)
flips = rng.choice([-1, 1], size=(20_000, n))
p_perm = (1 + np.sum(np.abs((flips * d).mean(1)) >= abs(d.mean()))) / (1 + len(flips))
_, blo, bhi = lk.bootstrap_ci(d)
wins = int(np.sum(d < 0)), int(np.sum(d > 0)), int(np.sum(d == 0))
print(f"confirmation (test, n={n} paired seeds): baseline {conf['baseline'].mean():.3f} +- {lk.t_interval(conf['baseline'])[1]:.3f}, "
      f"smoothing {conf['smoothing'].mean():.3f} +- {lk.t_interval(conf['smoothing'])[1]:.3f}")
print(f"  paired difference {m:+.3f} points, 95% t-interval [{m - h:+.3f}, {m + h:+.3f}], bootstrap [{blo:+.3f}, {bhi:+.3f}], "
      f"sign-flip permutation p = {p_perm:.4f}; smoothing better on {wins[0]}, worse on {wins[1]}, tied on {wins[2]} seeds")
print(f"  unpaired half-width would be {stats.t.ppf(0.975, 2 * n - 2) * np.sqrt(conf['baseline'].var(ddof=1) / n + conf['smoothing'].var(ddof=1) / n):.3f} points "
      f"(paired: {h:.3f}); correlation of paired outcomes {np.corrcoef(conf['baseline'], conf['smoothing'])[0, 1]:.2f}")
lk.write_dat("figures/data/16-epilogue-paired.dat",
             {"seed": list(seeds), "baseline": conf["baseline"], "smoothing": conf["smoothing"], "diff": d})

# 4. positive control ----------------------------------------------------------------------
cfg = lk.Config(**BASE, lr=tune["baseline"]["lr"], weight_decay=tune["baseline"]["wd"])
pc = np.array([err(cfg, s, half) for s in seeds])
mpc, hpc = lk.t_interval(pc - conf["baseline"])
print(f"positive control (half the training data): {pc.mean():.3f} vs {conf['baseline'].mean():.3f}; "
      f"paired difference {mpc:+.3f} +- {hpc:.3f} points")

# 5. factorial check -----------------------------------------------------------------------
lr0 = tune["baseline"]["lr"]
cells = {}
for (e_name, e), wd in itertools.product(EPS.items(), [0.0, 1e-3]):
    cfg = lk.Config(**BASE, lr=lr0, weight_decay=wd, label_smoothing=e)
    cells[(e_name, wd)] = np.array([err(cfg, s, full) for s in seeds])
sm = lambda wd: cells[("smoothing", wd)] - cells[("baseline", wd)]
inter = sm(1e-3) - sm(0.0)
mi, hi = lk.t_interval(inter)
main_ls = 0.5 * (sm(0.0) + sm(1e-3))
mm, hm = lk.t_interval(main_ls)
main_wd = 0.5 * ((cells[("baseline", 1e-3)] - cells[("baseline", 0.0)]) + (cells[("smoothing", 1e-3)] - cells[("smoothing", 0.0)]))
mw, hw = lk.t_interval(main_wd)
print(f"factorial at lr={lr0} (weight decay 0 or 1e-3): main effect of smoothing {mm:+.3f} +- {hm:.3f}, "
      f"main effect of weight decay {mw:+.3f} +- {hw:.3f}, interaction (difference of simple effects) {mi:+.3f} +- {hi:.3f} points")
for k, v in cells.items():
    print(f"  cell {k}: {v.mean():.3f} +- {lk.t_interval(v)[1]:.3f}")
lk.write_dat("figures/data/16-epilogue-cells.dat",
             {"cell": [f"{a}-{w:g}" for (a, w) in cells], "mean": [v.mean() for v in cells.values()],
              "hw": [lk.t_interval(v)[1] for v in cells.values()]})

print("\npredictions written before the runs:")
for k, v in PREDICTIONS.items():
    print(f"  {k}: {v}")
