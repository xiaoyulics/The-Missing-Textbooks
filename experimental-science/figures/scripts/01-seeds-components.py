"""Lab 1.2 (Chapter 1, Seeds): where the spread comes from.

The recipe of Lab 1.1 (two-layer MLP on the scikit-learn digits) is run in two designs.

  Crossed design, fixed split 0: every combination of I = 20 initialization seeds and
  J = 20 data-order seeds, one run per cell (400 runs). The ANOVA estimators of the
  chapter's variance-components theorem give the initialization, data-order and
  interaction components. Per-item predictions give the part of the seed variance on this
  fixed test set that is item churn (runs disagreeing on borderline items).

  Nested design: S = 50 train/test splits, R = 8 runs per split with fresh initialization
  and data-order seeds (400 runs). A one-way random-effects analysis gives the split
  component and the within-split (seed) component.

Both are compared with the binomial noise of a test set of m = 539 items.
Intervals: parametric bootstrap under the fitted normal random-effects model (2,000
replicates); for the churn share, a bootstrap over initialization seeds (rows).

Writes
  figures/data/01-seeds-components-grid.dat   400 runs: seeds, error (%), plot positions
  figures/data/01-seeds-components-init.dat   20 initialization seeds: mean and 95% t
  figures/data/01-seeds-components-order.dat  20 data-order seeds: mean and 95% t
Prints every number quoted in the text.
Run from the volume root:  python3 figures/scripts/01-seeds-components.py   (about 1 min)
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from scipy import stats
import labkit as lk

T0 = time.time()
CFG = lk.Config(width=64, depth=2, optimizer="momentum", lr=0.05, epochs=20, batch_size=64)
I = J = 20
S, R = 50, 8
N_POOL = 1797
rng = np.random.default_rng(20261017)


def mean_squares_crossed(Y):
    """ANOVA mean squares of an I x J table with one observation per cell."""
    I_, J_ = Y.shape
    gm, rm, cm = Y.mean(), Y.mean(1), Y.mean(0)
    msa = J_ * ((rm - gm) ** 2).sum() / (I_ - 1)
    msb = I_ * ((cm - gm) ** 2).sum() / (J_ - 1)
    res = Y - rm[:, None] - cm[None, :] + gm
    mse = (res ** 2).sum() / ((I_ - 1) * (J_ - 1))
    return msa, msb, mse


def components_crossed(Y):
    I_, J_ = Y.shape
    msa, msb, mse = mean_squares_crossed(Y)
    return np.array([(msa - mse) / J_, (msb - mse) / I_, mse])


def components_nested(Z):
    S_, R_ = Z.shape
    gm = Z.mean()
    msb = R_ * ((Z.mean(1) - gm) ** 2).sum() / (S_ - 1)
    msw = ((Z - Z.mean(1)[:, None]) ** 2).sum() / (S_ * (R_ - 1))
    return np.array([(msb - msw) / R_, msw])


def sd_interval(boot):
    """95% percentile interval for a standard deviation from bootstrap variance estimates."""
    v = np.sqrt(np.clip(boot, 0, None))
    return np.percentile(v, [2.5, 97.5])


# --------------------------------------------------------------------------------------
# Crossed design on split 0
# --------------------------------------------------------------------------------------
data0 = lk.load(CFG, split=0)
Xte, yte = data0[2], data0[3]
m = len(yte)
Y = np.zeros((I, J))
E = np.zeros((I, J, m), dtype=np.int8)
for i in range(I):
    for j in range(J):
        res = lk.run(CFG, init=i, order=j, data=data0, return_model=True)
        with torch.no_grad():
            E[i, j] = (res["model"](Xte).argmax(1) != yte).numpy()
        Y[i, j] = 100.0 * res["test_err"]
assert np.allclose(100.0 * E.mean(2), Y)
t_crossed = time.time() - T0

msa, msb, mse = mean_squares_crossed(Y)
comp = components_crossed(Y)                     # sigma_a^2, sigma_b^2, sigma_e^2 in %^2
total_var = Y.var(ddof=1)
B = 2000
sa, sb, se = np.sqrt(np.clip(comp, 0, None))
boot = np.empty((B, 3))
for b in range(B):
    Ys = (sa * rng.standard_normal(I)[:, None] + sb * rng.standard_normal(J)[None, :]
          + se * rng.standard_normal((I, J)))
    boot[b] = components_crossed(Ys)
pa = stats.f.sf(msa / mse, I - 1, (I - 1) * (J - 1))
pb = stats.f.sf(msb / mse, J - 1, (I - 1) * (J - 1))

print(f"== crossed design: {I} initialization x {J} data-order seeds, split 0, m = {m} test items "
      f"({t_crossed:.0f} s)")
print(f"mean test error {Y.mean():.3f}%; SD over the 400 runs {math.sqrt(total_var):.3f} points")
print(f"mean squares (%^2): MS_A {msa:.4f}  MS_B {msb:.4f}  MS_E {mse:.4f}")
names = ["initialization", "data order", "interaction"]
for k, nm in enumerate(names):
    lo_, hi_ = sd_interval(boot[:, k])
    print(f"  sigma_{nm:15s} = {math.sqrt(max(comp[k], 0)):.4f} points  (95% [{lo_:.4f}, {hi_:.4f}])"
          f"  variance {comp[k]:+.4f}, share {comp[k]/comp.sum():5.1%}")
print(f"  sum of components {comp.sum():.4f} vs sample variance of the 400 runs {total_var:.4f}")
print(f"  F tests: initialization p = {pa:.2g}; data order p = {pb:.2g}")
print(f"  SD of a run with a new order seed, same init: sqrt(sb^2+se^2) = "
      f"{math.sqrt(max(comp[1],0)+comp[2]):.3f}; new init seed, same order: "
      f"sqrt(sa^2+se^2) = {math.sqrt(comp[0]+comp[2]):.3f}")
# the same table read the other way: SD across orders within each init, averaged (Dodge et al.'s statistic)
print(f"  average over init seeds of the SD across orders: {Y.std(1, ddof=1).mean():.3f}; "
      f"average over order seeds of the SD across inits: {Y.std(0, ddof=1).mean():.3f}")

# item churn on the fixed test set
n_runs = I * J
def churn_parts(Eflat):
    nr = Eflat.shape[0]
    q = Eflat.mean(0)
    p_hat = Eflat.mean(1)
    seed_var = p_hat.var(ddof=1)
    churn = (q * (1 - q)).sum() * nr / (nr - 1) / m ** 2
    return seed_var, churn, q
seed_var, churn, q = churn_parts(E.reshape(n_runs, m).astype(float))
boot_share = []
for b in range(1000):
    rows = rng.integers(0, I, size=I)
    sv, ch, _ = churn_parts(E[rows].reshape(I * J, m).astype(float))
    boot_share.append(ch / sv)
cs_lo, cs_hi = np.percentile(boot_share, [2.5, 97.5])
pbar = q.mean()
binom_var = pbar * (1 - pbar) / m
item_var = binom_var - churn                      # (1/m) Var_x q, estimated
fpc = (N_POOL - m) / (N_POOL - 1)
print(f"== item churn on the fixed test set")
print(f"  items ever misclassified by some run: {(q > 0).sum()} of {m}; by every run: {(q == 1).sum()}; "
      f"by at most half the runs: {((q > 0) & (q <= 0.5)).sum()}")
print(f"  seed variance on this test set {1e4*seed_var:.3f} (x1e-4); churn {1e4*churn:.3f} = "
      f"{churn/seed_var:.0%} of it (95% [{cs_lo:.0%}, {cs_hi:.0%}]); co-movement "
      f"{1e4*(seed_var-churn):.3f}, SD {100*math.sqrt(max(seed_var-churn,0)):.3f} points")
print(f"  binomial SD sqrt(p(1-p)/m) = {100*math.sqrt(binom_var):.4f} points (finite pool "
      f"{100*math.sqrt(binom_var*fpc):.4f}); churn SD "
      f"{100*math.sqrt(churn):.3f}; item-difficulty SD sqrt(Var_x q / m) = {100*math.sqrt(item_var):.3f}; "
      f"with finite-pool factor {fpc:.3f}: {100*math.sqrt(item_var*fpc):.3f}")
# do good initializations stay good on other items? split the test items into two halves
# (the same halves as Lab 1.3) and correlate the 20 initialization-seed means across them
perm = np.random.default_rng(101).permutation(m)
h1, h2 = perm[: (m + 1) // 2], perm[(m + 1) // 2:]
init_h1 = 100 * E[:, :, h1].mean(axis=(1, 2)); init_h2 = 100 * E[:, :, h2].mean(axis=(1, 2))
r_init = np.corrcoef(init_h1, init_h2)[0, 1]
zr, hr = np.arctanh(r_init), 1.96 / math.sqrt(I - 3)
order_h1 = 100 * E[:, :, h1].mean(axis=(0, 2)); order_h2 = 100 * E[:, :, h2].mean(axis=(0, 2))
print(f"  initialization-seed means on two halves of the test set: correlation {r_init:.2f} "
      f"(95% [{np.tanh(zr-hr):.2f}, {np.tanh(zr+hr):.2f}], {I} seeds); SD of the means "
      f"{init_h1.std(ddof=1):.3f} and {init_h2.std(ddof=1):.3f} points")
print(f"  data-order-seed means on the two halves: correlation {np.corrcoef(order_h1, order_h2)[0,1]:.2f}")

# --------------------------------------------------------------------------------------
# Nested design: splits x runs
# --------------------------------------------------------------------------------------
t1 = time.time()
Z = np.zeros((S, R))
for s in range(S):
    d = lk.load(CFG, split=1 + s)
    for r in range(R):
        Z[s, r] = 100.0 * lk.run(CFG, init=1000 + 100 * s + r, order=1000 + 100 * s + r,
                                 split=1 + s, data=d)["test_err"]
compn = components_nested(Z)
ssp, ssw = np.sqrt(np.clip(compn, 0, None))
bootn = np.empty((B, 2))
for b in range(B):
    Zs = ssp * rng.standard_normal(S)[:, None] + ssw * rng.standard_normal((S, R))
    bootn[b] = components_nested(Zs)
gm_s = Z.mean()
msb_n = R * ((Z.mean(1) - gm_s) ** 2).sum() / (S - 1)
msw_n = ((Z - Z.mean(1)[:, None]) ** 2).sum() / (S * (R - 1))
p_split = stats.f.sf(msb_n / msw_n, S - 1, S * (R - 1))
print(f"== nested design: {S} splits x {R} runs ({time.time()-t1:.0f} s)")
print(f"  mean test error {Z.mean():.3f}%, SD over the {S*R} runs {Z.std(ddof=1):.3f} points")
for k, nm in enumerate(["split", "within split"]):
    lo_, hi_ = sd_interval(bootn[:, k])
    print(f"  sigma_{nm:13s} = {math.sqrt(max(compn[k],0)):.3f} points  (95% [{lo_:.3f}, {hi_:.3f}])"
          f"  share {compn[k]/compn.sum():5.1%}")
print(f"  F test for the split: p = {p_split:.2g}")
print(f"  per-split mean error ranges from {Z.mean(1).min():.3f}% to {Z.mean(1).max():.3f}%")
pb2 = Z.mean() / 100
print(f"  binomial SD at the pooled mean: {100*math.sqrt(pb2*(1-pb2)/m):.3f}; with finite-pool factor "
      f"{100*math.sqrt(pb2*(1-pb2)/m*fpc):.3f}")

# --------------------------------------------------------------------------------------
# Data for the figure: runs grouped by initialization seed and by data-order seed
# --------------------------------------------------------------------------------------
row_order = np.argsort(Y.mean(1), kind="stable")
col_order = np.argsort(Y.mean(0), kind="stable")
row_rank = np.empty(I, int); row_rank[row_order] = np.arange(1, I + 1)
col_rank = np.empty(J, int); col_rank[col_order] = np.arange(1, J + 1)
ii, jj = np.meshgrid(np.arange(I), np.arange(J), indexing="ij")
# errors live on a lattice of 100/m %, so many runs of a group coincide: place runs with
# the same error side by side, 0.07 units apart, centred on the group's position
def side_by_side(group_rank, groups, values):
    x = np.empty(values.shape, float)
    for g in np.unique(groups):
        for v in np.unique(values[groups == g]):
            idx = np.flatnonzero((groups == g) & (values == v))
            x[idx] = group_rank[g] + (np.arange(len(idx)) - (len(idx) - 1) / 2) * 0.07
    return x
items = np.rint(Y * m / 100).astype(int)
x_init = side_by_side(row_rank, ii.ravel(), items.ravel())
x_order = side_by_side(col_rank, jj.ravel(), items.ravel())
lk.write_dat("figures/data/01-seeds-components-grid.dat",
             {"init": ii.ravel(), "order": jj.ravel(), "err": Y.ravel(), "xinit": x_init, "xorder": x_order})
print(f"crossed runs range from {Y.min():.3f}% to {Y.max():.3f}%; widest stack of equal errors in an "
      f"initialization group: {max(np.bincount(items[i]).max() for i in range(I))}")
rows = [lk.t_interval(Y[i]) for i in range(I)]
cols = [lk.t_interval(Y[:, j]) for j in range(J)]
lk.write_dat("figures/data/01-seeds-components-init.dat",
             {"rank": row_rank, "seed": np.arange(I), "mean": [r[0] for r in rows], "hw": [r[1] for r in rows]})
lk.write_dat("figures/data/01-seeds-components-order.dat",
             {"rank": col_rank, "seed": np.arange(J), "mean": [c[0] for c in cols], "hw": [c[1] for c in cols]})
print(f"init-seed means range {min(r[0] for r in rows):.3f} to {max(r[0] for r in rows):.3f}; "
      f"order-seed means range {min(c[0] for c in cols):.3f} to {max(c[0] for c in cols):.3f}")
print(f"[{time.time()-T0:.1f} s]")
