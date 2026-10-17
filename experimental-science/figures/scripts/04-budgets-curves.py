"""Lab 4.2: budget curves for momentum and Adam under two search spaces.

The task, model and training are those of Lab 4.1 (04-budgets-protocols.py): digits with 256
distractor features, 300 training, 300 validation and 1197 test images, a two-layer MLP of
width 64, 100 epochs, batch size 32, constant learning rate; data split 0.

For each of four arms we draw a pool of K = 200 independent random-search trials, one run
(its own seed) per trial, and record validation and test error:

  momentum, 2017 space   lr log-uniform on [1e-3, 1]; momentum 0.9; no weight decay
  Adam, 2017 space       lr log-uniform on [10^-4.5, 10^-1.5]; betas (0.9, 0.999), eps 1e-8
  momentum, 2019 space   lr, 1-momentum and weight decay, as in Lab 4.1
  Adam, 2019 space       lr, eps, 1-beta1, 1-beta2 and weight decay, as in Lab 4.1

The budget curve at k is the expected test error of the trial that a search of k trials would
select (lowest validation error, ties broken by validation loss). It is estimated without bias
by averaging over all k-subsets of the pool (the U-statistic of the chapter), with 95%
percentile intervals from 1000 bootstrap resamples of the pool. A control repeats the four
pools (K = 100 each) on the plain digits, without distractors.

Run from the volume root:  python3 figures/scripts/04-budgets-curves.py
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk
import numpy as np
from scipy.special import gammaln

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "protocols", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04-budgets-protocols.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)          # reuse the task, the model and the search spaces of Lab 4.1

ARMS = ("m17", "a17", "m19", "a19")
NAMES = {"m17": "momentum, 2017 space", "a17": "Adam, 2017 space",
         "m19": "momentum, 2019 space", "a19": "Adam, 2019 space"}
B_BOOT = 1000


def sample(arm, rng):
    if arm == "m17":
        return dict(optimizer="momentum", lr=10 ** rng.uniform(-3, 0))
    if arm == "a17":
        return dict(optimizer="adam", lr=10 ** rng.uniform(-4.5, -1.5))
    return P.sample_2019("momentum" if arm == "m19" else "adam", rng)


def pool(arm, n, data, test, seed0):
    rng = np.random.default_rng(seed0)
    rows = []
    for j in range(n):
        h = sample(arm, rng)
        v, vl, t = P.trial(h, data, test, seed0 * 1000 + j)
        rows.append((v, vl, t))
    return np.array(rows)          # columns: val error, val loss, test error


def subset_weights(n, ks):
    """w[k][i] = C(i, k-1) / C(n, k), i = 0..n-1 (0-based rank from the worst): the probability
    that the trial of rank i is the best in a uniformly random k-subset."""
    i = np.arange(n)
    W = []
    for k in ks:
        logc = np.where(i >= k - 1, gammaln(i + 1) - gammaln(k) - gammaln(i - k + 2), -np.inf)
        logn = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        W.append(np.exp(logc - logn))
    return np.array(W)


def curve(rows, W, which=2):
    """Unbiased estimate, for each k, of E[column `which` of the selected trial] (selection by
    lowest validation error, then lowest validation loss)."""
    order = np.lexsort((-rows[:, 1], -rows[:, 0]))     # worst first ... best last
    return W @ rows[order, which]


def curve_with_replacement(rows, ks):
    """The plug-in estimate of Dodge et al. (2019): expected best of k draws with replacement
    from the empirical distribution of the pool (here for the validation error)."""
    v = np.sort(rows[:, 0])[::-1]                        # worst first
    n = len(v)
    out = []
    for k in ks:
        F = (np.arange(1, n + 1) / n) ** k              # P(best of k draws has rank <= i)
        w = np.diff(np.concatenate([[0.0], F]))
        out.append(float(w @ v))
    return np.array(out)


def boot(rows, W, rng):
    n = len(rows)
    reps = np.empty((B_BOOT, W.shape[0]))
    for b in range(B_BOOT):
        reps[b] = curve(rows[rng.integers(0, n, n)], W)
    return reps


def main():
    t0 = time.time()
    results = {}
    for tag, d_noise, n, kmax in (("main", P.D_NOISE, 200, 50), ("control", 0, 100, 25)):
        data, test = P.make_data(split=0, d_noise=d_noise)
        ks = np.arange(1, kmax + 1)
        W = subset_weights(n, ks)
        cols = {"k": ks}
        pools, reps = {}, {}
        for a, arm in enumerate(ARMS):
            pools[arm] = pool(arm, n, data, test, seed0=(7 if tag == "main" else 8) * 10 + a)
            est = curve(pools[arm], W)
            reps[arm] = boot(pools[arm], W, np.random.default_rng(100 + a))
            cols[arm] = 100 * est
            cols[arm + "lo"] = 100 * np.quantile(reps[arm], 0.025, axis=0)
            cols[arm + "hi"] = 100 * np.quantile(reps[arm], 0.975, axis=0)
            print(f"[{tag}] pool {NAMES[arm]:22s} done ({time.time() - t0:.0f}s); "
                  f"diverged trials: {int((pools[arm][:, 2] > 0.5).sum())} of {n}", flush=True)
        lk.write_dat(f"figures/data/04-budgets-curves-{tag}.dat", cols)
        results[tag] = (ks, cols, pools, reps, W)

    # ---- numbers quoted in the text
    for tag, (ks, cols, pools, reps, W) in results.items():
        print(f"\n[{tag}] expected test error (%) of the selected trial, with 95% bootstrap intervals")
        show = [1, 2, 4, 8, 16, 25] + ([50] if tag == "main" else [])
        for arm in ARMS:
            print(f"  {NAMES[arm]:22s} " + "  ".join(
                f"k={k}: {cols[arm][k - 1]:.2f} [{cols[arm + 'lo'][k - 1]:.2f}, {cols[arm + 'hi'][k - 1]:.2f}]"
                for k in show))
        for x, y in (("a17", "m17"), ("a19", "m19"), ("a19", "a17"), ("m19", "m17")):
            for k in (1, 4, 10, 16) + ((50,) if tag == "main" else (25,)):
                d = 100 * (reps[x][:, k - 1] - reps[y][:, k - 1])
                est = cols[x][k - 1] - cols[y][k - 1]
                print(f"  {NAMES[x]} minus {NAMES[y]} at k={k}: {est:+.2f} "
                      f"[{np.quantile(d, 0.025):+.2f}, {np.quantile(d, 0.975):+.2f}]")
        for x, y in (("a17", "m17"), ("a19", "m19")):
            for k0 in (4, 6, 8):
                dd = np.abs(cols[x] - cols[y])[ks >= k0]
                print(f"  max |{NAMES[x]} - {NAMES[y]}| over k >= {k0}: {dd.max():.2f}")
        # crossings of the point estimates
        for x, y in (("a17", "m17"), ("a19", "m19"), ("a19", "a17"), ("a19", "m17")):
            s = np.sign(cols[x] - cols[y])
            cross = [int(ks[i + 1]) for i in range(len(ks) - 1) if s[i] != s[i + 1] and s[i + 1] != 0]
            print(f"  sign changes of ({NAMES[x]}) - ({NAMES[y]}): at k = {cross}")
        # best validation error: unbiased versus with replacement, and the optimism of validation
        kall = np.arange(2, ks[-1] + 1)
        for arm in ARMS:
            gap = 100 * (curve_with_replacement(pools[arm], kall) - curve(pools[arm], W[kall - 1], which=0))
            print(f"  {NAMES[arm]:22s} with-replacement minus unbiased best validation error, k=2..{ks[-1]}: "
                  f"min {gap.min():+.3f}, max {gap.max():+.3f}")
        kk = np.array([1, 4, 16] + ([50] if tag == "main" else [25]))
        for arm in ARMS:
            Wk = W[kk - 1]
            unb = 100 * curve(pools[arm], Wk, which=0)
            wr = 100 * curve_with_replacement(pools[arm], kk)
            tst = 100 * curve(pools[arm], Wk, which=2)
            print(f"  {NAMES[arm]:22s} k={list(kk)}: best val error unbiased {np.round(unb, 2)}, "
                  f"with replacement {np.round(wr, 2)}; test error of the selected trial {np.round(tst, 2)}")
    print(f"\n{sum(len(p) for r in results.values() for p in r[2].values())} runs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
