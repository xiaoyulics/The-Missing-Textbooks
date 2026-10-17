"""Epilogue: one experiment, run by the book's protocols from question to report.

Question: does label smoothing (epsilon = 0.1) lower the test error of the digits MLP?

The predictions below were written into this file before any of its runs were made; the
script prints them next to what happened.

Stages
  1. pilot: 10 paired seeds per arm on a validation split, to estimate the spread of the
     paired difference and choose the number of seeds;
  2. tuning with equal budgets: the same grid of learning rates and weight decays for both
     arms, 3 seeds per configuration, scored on the validation split only (ties go to the
     first configuration in grid order: the smaller learning rate, then the smaller weight
     decay);
  3. confirmation: n paired seeds per arm at each arm's tuned configuration, scored once on
     the test set;
  4. positive control: the tuned baseline trained on half of the training set, which must
     come out visibly worse if the pipeline can detect an effect at all;
  5. a 2x2 factorial check (label smoothing x weight decay) at the baseline's tuned learning
     rate, for an interaction.
Added on review, after the first full run (the predictions and stages 1-5 are unchanged):
  2b. the weight-decay axis extended to 1e-3 and 1e-2 for both arms, on the validation split,
      because the baseline's choice sat at the edge of the grid (Protocol 4.1, step 3);
  3b. the confirmation runs scored image by image: the test images on which some run erred,
      and a bootstrap that resamples the test images as well as the seeds (Owen's pigeonhole
      bootstrap), since the seed interval holds the 539 test images fixed;
  3c. label smoothing at the baseline's tuned configuration (the home-field comparison);
  4b. a negative control: the tuned baseline against itself, with fresh seeds 2000-2047.
Writes figures/data/16-epilogue-*.dat and prints every number quoted in the epilogue.
Runtime: about a minute on a laptop CPU.
"""
import itertools
import os
import sys
from dataclasses import astuple

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
WDS_EXT = [0.0, 1e-4, 1e-3, 1e-2]                     # 2b, added on review
EPS = {"baseline": 0.0, "smoothing": 0.1}


def split_validation(data, n_val=300, seed=0):
    Xtr, ytr, Xte, yte = data
    g = torch.Generator().manual_seed(90_000 + seed)
    perm = torch.randperm(len(ytr), generator=g)
    va, tr = perm[:n_val], perm[n_val:]
    return Xtr[tr], ytr[tr], Xtr[va], ytr[va]


_cache = {}


def item_errors(cfg, seed, data, name):
    """0/1 errors of one run on each evaluation image of `data` (named `name` for the cache)."""
    key = (astuple(cfg), seed, name)
    if key not in _cache:
        r = lk.run(cfg, init=seed, order=seed, data=data, return_model=True)
        if not np.isfinite(r["test_loss"]):           # a diverged run: labkit's convention, error 0.9
            wrong = np.full(len(data[3]), 0.9)
        else:
            model = r["model"]
            model.eval()
            with torch.no_grad():
                wrong = (model(data[2]).argmax(1) != data[3]).numpy().astype(float)
        assert abs(wrong.mean() - r["test_err"]) < 1e-5
        _cache[key] = wrong
    return _cache[key]


def err(cfg, seed, data, name):
    return 100 * item_errors(cfg, seed, data, name).mean()      # in points


def config(lr, wd, e):
    return lk.Config(**BASE, lr=lr, weight_decay=wd, label_smoothing=e)


def tune_arm(grid, e):
    scores = [np.mean([err(config(lr, wd, e), 100 + s, val, "val") for s in range(3)]) for lr, wd in grid]
    best = int(np.argmin(scores))                     # first minimum in grid order
    return dict(lr=grid[best][0], wd=grid[best][1], val=scores[best], scores=scores,
                ties=int(np.sum(np.isclose(scores, scores[best]))))


def churn_share(W):
    """Item churn as a share of the seed variance of the error (runs x images), as in
    01-seeds-bestof.py."""
    q = W.mean(0)
    return np.mean(q * (1 - q)) * len(W) / (len(W) - 1) / W.shape[1] / W.mean(1).var(ddof=1)


def corr_ci(a, b):
    """Pearson correlation and a 95% interval from Fisher's z, as a printable string."""
    r = np.corrcoef(a, b)[0, 1]
    zr, s = np.arctanh(r), 1 / np.sqrt(len(a) - 3)
    return f"{r:.2f} ({np.tanh(zr - 1.96 * s):+.2f} to {np.tanh(zr + 1.96 * s):+.2f})"


full = lk.digits(split=0)
val = split_validation(full)
half = (full[0][: len(full[1]) // 2], full[1][: len(full[1]) // 2], full[2], full[3])

# 1. pilot ---------------------------------------------------------------------------------
pilot = {arm: np.array([err(config(0.05, 0.0, e), s, val, "val") for s in range(10)]) for arm, e in EPS.items()}
d_pilot = pilot["smoothing"] - pilot["baseline"]
sd_pilot = d_pilot.std(ddof=1)
delta = 0.25                                          # smallest effect worth detecting, points
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
n_exact = (z * sd_pilot / delta) ** 2
n_needed = int(np.ceil(n_exact))
n = max(20, n_needed)
print(f"pilot (validation, lr=0.05): baseline {pilot['baseline'].mean():.2f}, smoothing {pilot['smoothing'].mean():.2f}; "
      f"paired difference sd {sd_pilot:.3f} points -> (z_.975+z_.8)^2 sd^2/delta^2 = {n_exact:.1f}, "
      f"n >= {n_needed} for delta={delta} at power 0.8; using n = {n}")

# 2. tuning with equal budgets -------------------------------------------------------------
grid = list(itertools.product(LRS, WDS))
tune = {arm: tune_arm(grid, e) for arm, e in EPS.items()}
for arm, t in tune.items():
    print(f"tuning {arm}: best lr={t['lr']}, wd={t['wd']} (validation error {t['val']:.2f}, "
          f"{t['ties']} configuration(s) at that score); {len(grid)} configurations x 3 seeds")

# 2b. (review) the weight-decay axis extended for both arms, validation split only ---------
grid_ext = list(itertools.product(LRS, WDS_EXT))
tune_ext = {arm: tune_arm(grid_ext, e) for arm, e in EPS.items()}
for arm, t in tune_ext.items():
    i_old = grid_ext.index((tune[arm]["lr"], tune[arm]["wd"]))
    i_up = grid_ext.index((tune[arm]["lr"], 1e-3))
    print(f"extended grid {arm}: best lr={t['lr']}, wd={t['wd']} (validation error {t['val']:.2f}, "
          f"{t['ties']} configuration(s) tied); at the first choice {t['scores'][i_old]:.2f}, "
          f"same lr with wd=1e-3 {t['scores'][i_up]:.2f}; {len(grid_ext)} configurations x 3 seeds")
lk.write_dat("figures/data/16-epilogue-tuning.dat",
             {"lr": [g[0] for g in grid_ext], "wd": [g[1] for g in grid_ext],
              "first_grid": [int(g[1] in WDS) for g in grid_ext],
              "baseline": tune_ext["baseline"]["scores"], "smoothing": tune_ext["smoothing"]["scores"]})

# 3. confirmation on the test set, paired seeds ---------------------------------------------
seeds = range(1000, 1000 + n)
cfg_arm = {arm: config(tune[arm]["lr"], tune[arm]["wd"], e) for arm, e in EPS.items()}
E = {arm: np.array([item_errors(cfg_arm[arm], s, full, "full") for s in seeds]) for arm in EPS}
conf = {arm: 100 * E[arm].mean(1) for arm in EPS}
d = conf["smoothing"] - conf["baseline"]
m, h = lk.t_interval(d)
rng = np.random.default_rng(0)
flips = rng.choice([-1, 1], size=(20_000, n))
n_extreme = int(np.sum(np.abs((flips * d).mean(1)) >= abs(d.mean())))
p_perm = (1 + n_extreme) / (1 + len(flips))
_, blo, bhi = lk.bootstrap_ci(d)
wins = int(np.sum(d < 0)), int(np.sum(d > 0)), int(np.sum(d == 0))
print(f"confirmation (test, n={n} paired seeds): baseline {conf['baseline'].mean():.3f} +- {lk.t_interval(conf['baseline'])[1]:.3f}, "
      f"smoothing {conf['smoothing'].mean():.3f} +- {lk.t_interval(conf['smoothing'])[1]:.3f}")
print(f"  paired difference {m:+.3f} points, 95% t-interval [{m - h:+.3f}, {m + h:+.3f}], bootstrap [{blo:+.3f}, {bhi:+.3f}], "
      f"sign-flip permutation p = {p_perm:.2e} ({n_extreme} of {len(flips)} sign flips as extreme); "
      f"smoothing better on {wins[0]}, worse on {wins[1]}, tied on {wins[2]} seeds")
sd_d = d.std(ddof=1)
h20 = stats.t.ppf(0.975, 19) * sd_d / np.sqrt(20)
print(f"  sd of the paired differences {sd_d:.3f} points; projected half-width with 20 seeds {h20:.3f}")
print(f"  unpaired half-width would be {stats.t.ppf(0.975, 2 * n - 2) * np.sqrt(conf['baseline'].var(ddof=1) / n + conf['smoothing'].var(ddof=1) / n):.3f} points "
      f"(paired: {h:.3f}); correlation of paired outcomes {corr_ci(conf['baseline'], conf['smoothing'])}")
lk.write_dat("figures/data/16-epilogue-paired.dat",
             {"seed": list(seeds), "baseline": conf["baseline"], "smoothing": conf["smoothing"], "diff": d})

# 3b. (review) the test images: which ones carry the difference, and an interval that ----
# resamples them as well as the seeds (rows: seeds, columns: images; Owen 2007)
D = E["smoothing"] - E["baseline"]
n_img = D.shape[1]
erred = (E["baseline"].sum(0) > 0) | (E["smoothing"].sum(0) > 0)
always = {arm: int(np.sum(E[arm].mean(0) == 1)) for arm in EPS}
rng = np.random.default_rng(1)
B = 10_000
reps = np.empty(B)
for b in range(B):
    ci = np.bincount(rng.integers(0, n, n), minlength=n)
    cj = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
    reps[b] = 100 * (ci @ D @ cj) / (n * n_img)
plo, phi = np.quantile(reps, [0.025, 0.975])
print(f"  test images: {n_img}; {int(erred.sum())} were misclassified by some run of either arm "
      f"(always by the baseline: {always['baseline']}, always with smoothing: {always['smoothing']}); "
      f"mean errors per run {E['baseline'].sum(1).mean():.1f} (baseline) and {E['smoothing'].sum(1).mean():.1f} (smoothing) images")
print(f"  bootstrap over seeds and test images: [{plo:+.3f}, {phi:+.3f}] (half-width about {(phi - plo) / 2:.2f})")
for arm in EPS:
    p_arm = E[arm].mean()
    shares = [churn_share(E[arm][rng.integers(0, n, n)]) for _ in range(400)]
    print(f"  {arm}: binomial standard error of one run's error {100 * np.sqrt(p_arm * (1 - p_arm) / n_img):.2f} points; "
          f"item churn {100 * churn_share(E[arm]):.0f}% of the seed variance "
          f"(95% bootstrap over runs {100 * np.quantile(shares, 0.025):.0f}-{100 * np.quantile(shares, 0.975):.0f}%)")

# 3c. (review) label smoothing at the baseline's tuned configuration -----------------------
cfg_home = config(tune["baseline"]["lr"], tune["baseline"]["wd"], EPS["smoothing"])
home = np.array([err(cfg_home, s, full, "full") for s in seeds])
mh, hh = lk.t_interval(home - conf["baseline"])
print(f"smoothing at the baseline's configuration (lr={tune['baseline']['lr']}, wd={tune['baseline']['wd']}): "
      f"{home.mean():.3f}; paired difference {mh:+.3f} +- {hh:.3f} points, {100 * (1 - mh / m):.0f}% smaller than at its own")

# 4. positive control ----------------------------------------------------------------------
pc = np.array([err(cfg_arm["baseline"], s, half, "half") for s in seeds])
mpc, hpc = lk.t_interval(pc - conf["baseline"])
print(f"positive control (half the training data): {pc.mean():.3f} vs {conf['baseline'].mean():.3f}; "
      f"paired difference {mpc:+.3f} +- {hpc:.3f} points")

# 4b. (review) negative control: the tuned baseline against itself, fresh seeds ----------
aa = np.array([err(cfg_arm["baseline"], s, full, "full") for s in range(2000, 2000 + n)])
va_, vb_ = aa.var(ddof=1) / n, conf["baseline"].var(ddof=1) / n
nu = (va_ + vb_) ** 2 / (va_ ** 2 / (n - 1) + vb_ ** 2 / (n - 1))
haa = stats.t.ppf(0.975, nu) * np.sqrt(va_ + vb_)
maa = aa.mean() - conf["baseline"].mean()
print(f"negative control (baseline, seeds 2000-{1999 + n}, against the confirmation baseline): {aa.mean():.3f} vs "
      f"{conf['baseline'].mean():.3f}; difference {maa:+.3f} +- {haa:.3f} points (Welch, {nu:.0f} degrees of freedom)")

# 5. factorial check -----------------------------------------------------------------------
lr0 = tune["baseline"]["lr"]
cells = {}
for (e_name, e), wd in itertools.product(EPS.items(), [0.0, 1e-3]):
    cells[(e_name, wd)] = np.array([err(config(lr0, wd, e), s, full, "full") for s in seeds])
sm = lambda wd: cells[("smoothing", wd)] - cells[("baseline", wd)]
inter = sm(1e-3) - sm(0.0)
mi, hi = lk.t_interval(inter)
main_ls = 0.5 * (sm(0.0) + sm(1e-3))
mm, hm = lk.t_interval(main_ls)
main_wd = 0.5 * ((cells[("baseline", 1e-3)] - cells[("baseline", 0.0)]) + (cells[("smoothing", 1e-3)] - cells[("smoothing", 0.0)]))
mw, hw = lk.t_interval(main_wd)
print(f"factorial at lr={lr0} (weight decay 0 or 1e-3): main effect of smoothing {mm:+.3f} +- {hm:.3f}, "
      f"main effect of weight decay {mw:+.3f} +- {hw:.3f}, interaction (difference of simple effects) {mi:+.3f} +- {hi:.3f} points")
for wd in [0.0, 1e-3]:
    print(f"  simple effect of smoothing at wd={wd:g}: {sm(wd).mean():+.3f} +- {lk.t_interval(sm(wd))[1]:.3f}")
for k, v in cells.items():
    print(f"  cell {k}: {v.mean():.3f} +- {lk.t_interval(v)[1]:.3f}")
print(f"  correlation of paired outcomes at a shared learning rate: {corr_ci(cells[('baseline', 0.0)], cells[('smoothing', 0.0)])} "
      f"(wd 0), {corr_ci(cells[('baseline', 1e-3)], cells[('smoothing', 1e-3)])} (wd 1e-3); "
      f"baseline with wd 1e-4 against wd 0: {corr_ci(conf['baseline'], cells[('baseline', 0.0)])}")
lk.write_dat("figures/data/16-epilogue-cells.dat",
             {"level": [0, 1], "wd": [0.0, 1e-3],
              "baseline": [cells[("baseline", w)].mean() for w in (0.0, 1e-3)],
              "baseline_hw": [lk.t_interval(cells[("baseline", w)])[1] for w in (0.0, 1e-3)],
              "smoothing": [cells[("smoothing", w)].mean() for w in (0.0, 1e-3)],
              "smoothing_hw": [lk.t_interval(cells[("smoothing", w)])[1] for w in (0.0, 1e-3)]})

# 2b, continued: the effect at the choices of the extended grid (test runs, same seeds) ----
ext = {arm: np.array([err(config(tune_ext[arm]["lr"], tune_ext[arm]["wd"], e), s, full, "full") for s in seeds])
       for arm, e in EPS.items()}
mx, hx = lk.t_interval(ext["smoothing"] - ext["baseline"])
print(f"effect at the extended grid's choices: baseline {ext['baseline'].mean():.3f}, smoothing {ext['smoothing'].mean():.3f}; "
      f"paired difference {mx:+.3f} +- {hx:.3f} points")

print("\npredictions written before the runs:")
for k, v in PREDICTIONS.items():
    print(f"  {k}: {v}")
