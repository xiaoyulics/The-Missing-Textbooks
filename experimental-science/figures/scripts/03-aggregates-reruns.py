"""Lab 3.2 of *Deep Learning as an Experimental Science*, Chapter 3 (Many Tasks, Few Runs).

Re-run a three-runs-per-task "paper" 200 times. The runs come from the bank of Lab 3.1
(03-aggregates-suite.py: 50 runs of each method on each of the twelve tasks), which this script
reads from figures/data/03-aggregates-runs.dat, training it first if the file is missing.

Papers. In each of 200 papers (seed 2026), 3 runs per task of method A and 3 of method B are
drawn without replacement from the 50 runs of the bank, and B - A is computed by the mean, the
median and the IQM. The reference is the same difference computed from all 50 runs.
Null papers: A against A, two disjoint sets of 3 runs per task drawn from A's 50 runs.

Also computed, for every paper:
  - the 95% stratified bootstrap interval (1000 resamples) of each aggregate of each method, and
    whether it contains the 50-run value (coverage), and of the difference B - A (power) and the
    null difference A - A (false claims);
  - Welch's t-test of B against A on each task (3 runs each), and of A against A in the null
    papers; the number of tasks "significant" at level 0.05 without adjustment, with Holm's
    procedure and with the Benjamini-Hochberg procedure.
Finally, with all 50 runs: the winner by each aggregate on each of the 495 subsets of 8 of the
12 tasks, for B against A and for method C (dropout) against A. For the method section the script
also computes the asymptotic efficiencies of the mean, the median and the interquartile mean for
five location families (by numerical integration), and simulates the effective number of tasks
for 13 Gaussian tasks with equal gaps.

Run from the volume root:  python3 figures/scripts/03-aggregates-reruns.py
Writes figures/data/03-aggregates-reruns.dat and -rerunsref.dat and prints the numbers quoted in the
text. Runtime: under a minute once the bank exists.
"""
import importlib.util
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats

import labkit as lk

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("suite", os.path.join(HERE, "03-aggregates-suite.py"))
suite = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite)

N_PAPERS, N_PER_PAPER, N_BOOT = 200, 3, 1000
AGG = {"mean": suite.agg_mean, "median": suite.agg_median, "iqm": suite.agg_iqm}


def draw(S, rng, k):
    """k runs per task drawn without replacement: returns indices of shape (J, k)."""
    J, n = S.shape
    return np.stack([rng.choice(n, size=k, replace=False) for _ in range(J)])


def take(S, idx):
    return S[np.arange(S.shape[0])[:, None], idx]


def holm(p, alpha=0.05):
    """Number of rejections of Holm's step-down procedure."""
    ps = np.sort(p)
    m = len(p)
    for k in range(m):
        if ps[k] > alpha / (m - k):
            return k
    return m


def bh(p, alpha=0.05):
    """Number of rejections of the Benjamini-Hochberg step-up procedure."""
    ps = np.sort(p)
    m = len(p)
    ok = np.nonzero(ps <= alpha * np.arange(1, m + 1) / m)[0]
    return 0 if len(ok) == 0 else int(ok[-1] + 1)


def welch_p(X, Y):
    """Two-sided Welch p-values, task by task (rows of X and Y)."""
    with np.errstate(all="ignore"):
        p = stats.ttest_ind(X, Y, axis=1, equal_var=False).pvalue
    return np.where(np.isfinite(p), p, 1.0)


def theory_checks():
    """Numbers for the method section: asymptotic variances of the mean, the median and the
    interquartile mean for several location families (Proposition on one sample), and a
    simulation of the effective number of tasks for J Gaussian tasks with equal gaps."""
    from scipy import integrate
    from scipy.stats import trim_mean

    class Contaminated:                     # (1 - eps) N(0, 1) + eps N(0, k^2)
        def __init__(self, eps, k):
            self.eps, self.k = eps, k
        def pdf(self, x):
            return (1 - self.eps) * stats.norm.pdf(x) + self.eps * stats.norm.pdf(x, scale=self.k)
        def cdf(self, x):
            return (1 - self.eps) * stats.norm.cdf(x) + self.eps * stats.norm.cdf(x, scale=self.k)
        def ppf(self, q):
            from scipy.optimize import brentq
            return brentq(lambda x: self.cdf(x) - q, -100, 100)
        def var(self):
            return (1 - self.eps) + self.eps * self.k ** 2

    families = [("normal", stats.norm()), ("logistic", stats.logistic()), ("Laplace", stats.laplace()),
                ("t with 3 df", stats.t(3)), ("normal, 5% with SD 10", Contaminated(0.05, 10.0))]
    print("\nOne sample from a symmetric location family: asymptotic variance times N, and the"
          " efficiency relative to the mean (mean variance / variance)")
    for name, d in families:
        q = d.ppf(0.75)
        v_mean = d.var()
        v_med = 1.0 / (4.0 * d.pdf(0.0) ** 2)
        v_iqm = 4.0 * (integrate.quad(lambda y: y * y * d.pdf(y), -q, q)[0] + q * q / 2.0)
        print(f"  {name:22s} mean {v_mean:7.3f}  median {v_med:7.3f}  IQM {v_iqm:7.3f}"
              f"   efficiency: median {v_mean / v_med:5.2f}, IQM {v_mean / v_iqm:5.2f}")

    print("\nEffective number of tasks: J = 13 Gaussian tasks with errors of SD s, means g apart"
          " (100000 draws, seed 3)")
    rng = np.random.default_rng(3)
    J = 13
    E = rng.standard_normal((100_000, J))
    for g in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        X = np.arange(J) * g + E
        rmse_med = np.sqrt(np.mean((np.median(X, 1) - (J // 2) * g) ** 2))
        target = trim_mean(np.arange(J) * g, 0.25)
        rmse_tm = np.sqrt(np.mean((trim_mean(X, 0.25, axis=1) - target) ** 2))
        rmse_mean = np.sqrt(np.mean((X.mean(1) - (J - 1) * g / 2) ** 2))
        print(f"  g/s = {g:3.1f}: RMSE/s mean {rmse_mean:.3f}, median {rmse_med:.3f}, 25%-trimmed mean {rmse_tm:.3f};"
              f" effective numbers {1 / rmse_mean ** 2:5.2f}, {1 / rmse_med ** 2:5.2f}, {1 / rmse_tm ** 2:5.2f}")
    print(f"  separated limits: mean {1 / np.sqrt(J):.3f}, median 1.000, trimmed mean (middle 7) {1 / np.sqrt(7):.3f};"
          f" tied limit for the median sqrt(pi/(2J)) = {np.sqrt(np.pi / (2 * J)):.3f}")


def main():
    t0 = time.time()
    bank = suite.load_bank()
    if bank is None:
        print("no bank of runs found: training it (as in 03-aggregates-suite.py) ...")
        bank = suite.train_bank()
        suite.save_bank(bank)
        bank = suite.load_bank()
    names = bank["methods"]
    SA, SB, SC = (bank["score"][names.index(k)] for k in ("A", "B", "C"))
    J, n = SA.shape
    ref = {k: f(SB) - f(SA) for k, f in AGG.items()}
    full = {k: (f(SA), f(SB)) for k, f in AGG.items()}
    print(f"bank: {J} tasks, {n} runs per task and method")
    print("50-run differences B - A: " + ", ".join(f"{k} {v:+.4f}" for k, v in ref.items()))

    rng = np.random.default_rng(2026)
    d = {k: np.zeros(N_PAPERS) for k in AGG}        # B - A per paper
    d0 = {k: np.zeros(N_PAPERS) for k in AGG}       # A - A' per paper
    aggA = {k: np.zeros(N_PAPERS) for k in AGG}
    aggB = {k: np.zeros(N_PAPERS) for k in AGG}
    cover = {(m, k): 0 for m in "AB" for k in AGG}
    claim = {k: 0 for k in AGG}          # B - A interval excludes 0 on the positive side
    wrongclaim = {k: 0 for k in AGG}     # B - A interval excludes 0 on the negative side
    falseclaim = {k: 0 for k in AGG}     # A - A' interval excludes 0
    tests = {"raw": [], "holm": [], "bh": [], "raw0": [], "holm0": [], "bh0": []}
    brng = np.random.default_rng(99)
    for r in range(N_PAPERS):
        iA, iB = draw(SA, rng, N_PER_PAPER), draw(SB, rng, N_PER_PAPER)
        PA, PB = take(SA, iA), take(SB, iB)
        i6 = draw(SA, rng, 2 * N_PER_PAPER)
        PA1, PA2 = take(SA, i6[:, :N_PER_PAPER]), take(SA, i6[:, N_PER_PAPER:])
        RA = suite.strat_resample(PA, N_BOOT, brng)
        RB = suite.strat_resample(PB, N_BOOT, brng)
        R1 = suite.strat_resample(PA1, N_BOOT, brng)
        R2 = suite.strat_resample(PA2, N_BOOT, brng)
        for k, f in AGG.items():
            aggA[k][r], aggB[k][r] = f(PA), f(PB)
            d[k][r] = aggB[k][r] - aggA[k][r]
            d0[k][r] = f(PA1) - f(PA2)
            bA, bB = f(RA), f(RB)
            for m, b, truth in (("A", bA, full[k][0]), ("B", bB, full[k][1])):
                lo, hi = np.quantile(b, [0.025, 0.975])
                cover[(m, k)] += int(lo <= truth <= hi)
            lo, hi = np.quantile(bB - bA, [0.025, 0.975])
            claim[k] += int(lo > 0)
            wrongclaim[k] += int(hi < 0)
            lo, hi = np.quantile(f(R1) - f(R2), [0.025, 0.975])
            falseclaim[k] += int(lo > 0 or hi < 0)
        p = welch_p(PB, PA)
        tests["raw"].append(int((p <= 0.05).sum())); tests["holm"].append(holm(p)); tests["bh"].append(bh(p))
        p0 = welch_p(PA1, PA2)
        tests["raw0"].append(int((p0 <= 0.05).sum())); tests["holm0"].append(holm(p0)); tests["bh0"].append(bh(p0))

    print(f"\n{N_PAPERS} papers with {N_PER_PAPER} runs per task and method")
    print("aggregate  50-run B-A   papers: mean(B-A)  SD(B-A)   wrong sign   SD(A agg)   null SD(A-A')  null 95% range")
    for k in AGG:
        wrong = int((np.sign(d[k]) != np.sign(ref[k])).sum())
        lo0, hi0 = np.quantile(d0[k], [0.025, 0.975])
        print(f"  {k:7s}  {ref[k]:+.4f}     {d[k].mean():+.4f}    {d[k].std(ddof=1):.4f}    "
              f"{wrong:3d} ({wrong / N_PAPERS:.1%})   {aggA[k].std(ddof=1):.4f}      {d0[k].std(ddof=1):.4f}"
              f"      [{lo0:+.4f}, {hi0:+.4f}]")
    # expected value of the 3-run aggregates against the 50-run values (bias of the sample median)
    for k in AGG:
        print(f"  {k:7s} mean of 3-run values: A {aggA[k].mean():.4f} (50 runs {full[k][0]:.4f}),"
              f" B {aggB[k].mean():.4f} (50 runs {full[k][1]:.4f})")

    # run-noise SD predicted by the effective number of tasks, for method A
    sdA = SA.std(1, ddof=1)
    # finite-population correction of the bank: k of n runs drawn without replacement have a mean
    # with variance (S^2 / k)(1 - k/n), where S^2 is the bank's sample variance (divisor n - 1)
    fpc = (n - N_PER_PAPER) / n
    pred_mean = np.sqrt(np.sum(sdA**2) / N_PER_PAPER * fpc) / J
    order = np.argsort(SA.mean(1))
    mid2 = order[J // 2 - 1:J // 2 + 1]
    pred_median = 0.5 * np.sqrt(np.sum(sdA[mid2]**2) / N_PER_PAPER * fpc)
    mid6 = order[J // 4:3 * J // 4]
    pred_iqm = np.sqrt(np.sum(sdA[mid6]**2) / N_PER_PAPER * fpc) / len(mid6)
    print("\nRun-noise SD of A's aggregates over the papers, observed against the separated-tasks"
          " prediction s*sqrt(sum c_k^2):")
    print(f"  mean   observed {aggA['mean'].std(ddof=1):.4f}  predicted {pred_mean:.4f} (exact for the mean)")
    print(f"  median observed {aggA['median'].std(ddof=1):.4f}  predicted {pred_median:.4f}"
          f" (two middle tasks: {suite.TASKS[mid2[0]][0]}, {suite.TASKS[mid2[1]][0]})")
    print(f"  iqm    observed {aggA['iqm'].std(ddof=1):.4f}  predicted {pred_iqm:.4f} (six middle tasks)")

    print("\nCoverage of nominal 95% stratified bootstrap intervals computed from 3 runs (50-run value as truth):")
    for k in AGG:
        print(f"  {k:7s} A {cover[('A', k)] / N_PAPERS:.1%}   B {cover[('B', k)] / N_PAPERS:.1%}")
    print("Papers whose 95% interval for B - A excludes zero (B better / B worse); null A - A': excludes zero")
    for k in AGG:
        print(f"  {k:7s} B better {claim[k] / N_PAPERS:.1%}   B worse {wrongclaim[k] / N_PAPERS:.1%}"
              f"   null false claims {falseclaim[k] / N_PAPERS:.1%}")

    print("\nPer-task Welch tests at level 0.05 (12 tasks, 3 runs per arm):")
    for lab, key in (("B vs A", ""), ("A vs A (null)", "0")):
        raw, ho, b = (np.array(tests[t + key]) for t in ("raw", "holm", "bh"))
        print(f"  {lab:14s} mean number significant: raw {raw.mean():.2f}, Holm {ho.mean():.2f}, BH {b.mean():.2f};"
              f" papers with at least one: raw {np.mean(raw > 0):.1%}, Holm {np.mean(ho > 0):.1%}, BH {np.mean(b > 0):.1%}")
    print(f"  prediction for the null, independent exact tests: at least one with probability 1-0.95^12 = {1 - 0.95**12:.3f},"
          f" expected number {12 * 0.05:.2f}")

    print("\nTask subsets: winner on each of the 495 subsets of 8 of the 12 tasks (50 runs)")
    for lab, SX in (("B vs A", SB), ("C vs A", SC)):
        fullsign = {k: np.sign(f(SX) - f(SA)) for k, f in AGG.items()}
        flips = {k: 0 for k in AGG}
        nsub = 0
        for sub in itertools.combinations(range(J), 8):
            sub = list(sub)
            nsub += 1
            for k, f in AGG.items():
                flips[k] += int(np.sign(f(SX[sub]) - f(SA[sub])) != fullsign[k])
        print(f"  {lab}: full-suite signs " + ", ".join(f"{k} {int(fullsign[k]):+d}" for k in AGG) +
              "; subsets with the other winner: " + ", ".join(f"{k} {flips[k]} ({flips[k] / nsub:.0%})" for k in AGG))

    theory_checks()

    # data for the strip plot
    jit = np.random.default_rng(5).uniform(-0.3, 0.3, size=(6, N_PAPERS))
    cols = {"rep": np.arange(1, N_PAPERS + 1)}
    for c, k in enumerate(AGG):
        cols["d" + k] = d[k]
        cols["n" + k] = d0[k]
        cols["j" + k] = jit[c]
        cols["jn" + k] = jit[3 + c]
    lk.write_dat("figures/data/03-aggregates-reruns.dat", cols, fmt="{:.5g}")
    lk.write_dat("figures/data/03-aggregates-rerunsref.dat",
                 {"y": [3, 2, 1], "d50": [ref["mean"], ref["median"], ref["iqm"]]}, fmt="{:.5g}")
    print(f"\ntotal runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
