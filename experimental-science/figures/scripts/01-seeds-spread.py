"""Lab 1.1 (Chapter 1, Seeds): the spread of a single method.

Two hundred runs of one fixed training recipe (a two-layer MLP on the scikit-learn
digits), each with all four labkit seeds changed: initialization, data order, the
train/test split, and the noise seed (which has nothing to act on here: no label noise,
dropout or augmentation). We record the test error of every run and follow the running
mean with its 95% t-interval as runs accumulate: the real-data version of the ornament at
the foot of each chapter opening.

Writes
  figures/data/01-seeds-spread-runs.dat     one row per run: index, test error (%), stack
                                            height for the dot plot, running mean and its
                                            95% t-interval (lo, hi) after this many runs
Prints every number quoted in the text.
Run from the volume root:  python3 figures/scripts/01-seeds-spread.py   (about 15 s)
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
import labkit as lk

T0 = time.time()
N_RUNS = 200
CFG = lk.Config(width=64, depth=2, optimizer="momentum", lr=0.05, epochs=20, batch_size=64)

err = np.empty(N_RUNS)
m_test = None
for r in range(N_RUNS):
    data = lk.load(CFG, split=r, noise=r)
    m_test = data[3].shape[0]
    res = lk.run(CFG, init=r, order=r, split=r, noise=r, data=data)
    err[r] = 100.0 * res["test_err"]          # percent
n_pool = 1797                                   # size of the digits pool
n_train = n_pool - m_test

# --- running mean and 95% t-interval -------------------------------------------------
run_mean = np.cumsum(err) / np.arange(1, N_RUNS + 1)
lo = np.full(N_RUNS, np.nan); hi = np.full(N_RUNS, np.nan)
for j in range(2, N_RUNS + 1):
    mu, h = lk.t_interval(err[:j])
    lo[j - 1], hi[j - 1] = mu - h, mu + h
# the first run has no interval; plot it at the run itself
lo[0] = hi[0] = err[0]

# --- stack heights for a dot plot (errors live on a lattice of 100/m_test %) ---------
k_items = np.rint(err * m_test / 100.0).astype(int)
stack = np.zeros(N_RUNS, dtype=int)
seen = {}
for r in range(N_RUNS):
    seen[k_items[r]] = seen.get(k_items[r], 0) + 1
    stack[r] = seen[k_items[r]]

lk.write_dat("figures/data/01-seeds-spread-runs.dat",
             {"run": np.arange(1, N_RUNS + 1), "err": err, "items": k_items, "stack": stack,
              "mean": run_mean, "lo": lo, "hi": hi})

# --- numbers for the text ---------------------------------------------------------------
mean, hw = lk.t_interval(err)
sd = err.std(ddof=1)
print(f"recipe: {CFG.width}x{CFG.depth} MLP, {CFG.optimizer} lr={CFG.lr}, {CFG.epochs} epochs, "
      f"batch {CFG.batch_size}; train {n_train} / test {m_test} images; {N_RUNS} runs")
print(f"test error: mean {mean:.3f}% +- {hw:.3f} (95% t), SD {sd:.3f} points, SE {sd/math.sqrt(N_RUNS):.3f}")
b_lo, b_hi = lk.bootstrap_ci(err, seed=1)[1:]
print(f"bootstrap 95% interval for the mean: [{b_lo:.3f}, {b_hi:.3f}]")
q = np.percentile(err, [0, 2.5, 25, 50, 75, 97.5, 100])
print("min, 2.5%, 25%, median, 75%, 97.5%, max:", np.round(q, 3))
print(f"range {err.max()-err.min():.3f} points = {k_items.max()-k_items.min()} test items; "
      f"one item = {100.0/m_test:.3f} points; distinct values: {len(np.unique(k_items))}")
# SD confidence interval (chi-square)
c_lo, c_hi = stats.chi2.ppf([0.975, 0.025], N_RUNS - 1)
print(f"95% interval for the run SD: [{sd*math.sqrt((N_RUNS-1)/c_lo):.3f}, {sd*math.sqrt((N_RUNS-1)/c_hi):.3f}]")
print(f"skewness {stats.skew(err):.3f}, excess kurtosis {stats.kurtosis(err):.3f}")

# binomial noise of a test set of m items, with and without the finite-pool correction
p = mean / 100.0
binom = 100.0 * math.sqrt(p * (1 - p) / m_test)
fpc = math.sqrt((n_pool - m_test) / (n_pool - 1))
print(f"binomial SD sqrt(p(1-p)/m) = {binom:.3f} points; with finite-pool factor "
      f"sqrt((N-m)/(N-1)) = {fpc:.3f}: {binom*fpc:.3f} points")

# two runs of one method: 100 disjoint pairs
d = np.abs(err[0::2] - err[1::2])
frac = np.mean(d > sd)
ci = stats.binomtest(int(np.sum(d > sd)), len(d)).proportion_ci(confidence_level=0.95, method="exact")
pred = 2 * stats.norm.cdf(-1 / math.sqrt(2))
print(f"100 disjoint pairs: fraction with |Y1-Y2| > SD = {frac:.2f} "
      f"(exact 95% CI [{ci.low:.2f}, {ci.high:.2f}]); normal prediction 2*Phi(-1/sqrt2) = {pred:.3f}")
print(f"median |Y1-Y2| over the pairs = {np.median(d):.3f}; normal prediction 0.954*SD = {0.9539*sd:.3f}")

# the best and the worst of the 200, against the normal prediction for the extremes of 200
from scipy import integrate
def e_k(k):
    f = lambda z: z * k * stats.norm.pdf(z) * stats.norm.cdf(z) ** (k - 1)
    return integrate.quad(f, -12, 12, limit=200)[0]
e200 = e_k(N_RUNS)
print(f"best run {err.min():.3f}%, worst {err.max():.3f}%; normal prediction for the expected "
      f"best/worst of 200: mean -+ {e200:.3f} SD = {mean - e200*sd:.3f} / {mean + e200*sd:.3f}")

# running interval at a few checkpoints (the ornament shows up to 16 runs)
for j in (1, 2, 3, 5, 10, 16, 20, 50, 100, 200):
    if j == 1:
        print(f"after {j:3d} run : mean {run_mean[0]:.3f}  (no interval)")
    else:
        print(f"after {j:3d} runs: mean {run_mean[j-1]:.3f}  95% t-interval [{lo[j-1]:.3f}, {hi[j-1]:.3f}]"
              f"  half-width {(hi[j-1]-lo[j-1])/2:.3f}")
print(f"half-width ratio 16 -> 200 runs: {((hi[15]-lo[15])/(hi[199]-lo[199])):.2f} "
      f"(sqrt(200/16) = {math.sqrt(200/16):.2f})")
print(f"sample SD of the first 16 runs {err[:16].std(ddof=1):.3f}; t quantiles {stats.t.ppf(0.975, 15):.3f} (15 df), "
      f"{stats.t.ppf(0.975, 199):.3f} (199 df)")
one_item = 100.0 / m_test
first = next(j for j in range(2, N_RUNS + 1) if (hi[j-1] - lo[j-1]) / 2 < one_item)
print(f"first number of runs with a half-width below one test item ({one_item:.3f} points): {first}")
cover = np.mean([(lo[j-1] <= mean <= hi[j-1]) for j in range(2, N_RUNS + 1)])
print(f"fraction of running intervals (n = 2..200) that contain the final mean: {cover:.2f}")
print(f"[{time.time()-T0:.1f} s]")
