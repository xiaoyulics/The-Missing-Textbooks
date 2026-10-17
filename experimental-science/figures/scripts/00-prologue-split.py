"""Prologue: ten runs of the same method, split into two groups of five.

Ten runs of one configuration of the digits MLP (two hidden layers of 32 units, plain SGD,
20% label noise) differ only in their initialization and data-order seeds, 0 to 9. For each
of the 126 ways to split them into two groups of five, compare the groups by
  (a) Welch's t-test on one outcome per run, its mean test accuracy over the second half
      of training (the run is the experimental unit), and
  (b) Student's t-test on the 30 evaluation points of the second half of every learning
      curve, pooled as if each were an independent observation (sacrificial
      pseudoreplication, in Hurlbert's terms).
Writes figures/data/00-prologue-curves.dat (the learning curves of the most extreme split)
and figures/data/00-prologue-pvalues.dat (sorted p-values of both tests), and prints the
numbers quoted in the prologue. Runtime: a few seconds.
"""
import itertools
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk

cfg = lk.Config(width=32, optimizer="sgd", lr=0.03, epochs=30, batch_size=32, label_noise=0.2, eval_every=20)
data = lk.load(cfg, noise=0)
runs = [lk.run(cfg, init=s, order=s, data=data) for s in range(10)]
steps = np.array([c[0] for c in runs[0]["curve"]])
acc = np.array([[c[2] for c in r["curve"]] for r in runs])          # runs x evaluation points
final = acc[:, -1]
late = steps >= steps[-1] / 2                                       # second half of training
summary = acc[:, late].mean(1)                                      # one number per run

p_run, p_pool, splits = [], [], []
for g in itertools.combinations(range(10), 5):
    if 0 not in g:          # count each unordered split once
        continue
    h = [i for i in range(10) if i not in g]
    p_run.append(stats.ttest_ind(summary[list(g)], summary[h], equal_var=False).pvalue)
    p_pool.append(stats.ttest_ind(acc[list(g)][:, late].ravel(), acc[h][:, late].ravel()).pvalue)
    splits.append((g, h))
p_run, p_pool = np.array(p_run), np.array(p_pool)
print(f"runs: final test accuracy {final.mean():.4f} +- {final.std(ddof=1):.4f} (sd), "
      f"range {final.min():.4f}..{final.max():.4f}; {len(steps)} evaluation points per curve, "
      f"{late.sum()} in the second half")
print(f"splits: {len(p_run)}")
for name, p in [("run-level Welch t-test", p_run), ("pooled curve points", p_pool)]:
    print(f"{name}: share with p<0.05 = {np.mean(p < 0.05):.3f}; share with p<0.001 = {np.mean(p < 0.001):.3f}; "
          f"min p = {p.min():.2e}")
k = int(np.argmin(p_pool))
g, h = splits[k]
print(f"most extreme split by pooled test: groups {g} vs {h}: pooled p = {p_pool[k]:.2e}, run-level p = {p_run[k]:.3f}")
cols = {"step": steps}
for name, idx in (("a", g), ("b", h)):
    m = acc[list(idx)].mean(0)
    cols[f"mean{name}"] = m
    for j, i in enumerate(idx):
        cols[f"{name}{j}"] = acc[i]
lk.write_dat("figures/data/00-prologue-curves.dat", cols)
lk.write_dat("figures/data/00-prologue-pvalues.dat",
             {"rank": np.arange(1, len(p_run) + 1) / len(p_run), "prun": np.sort(p_run), "ppool": np.sort(p_pool)})
