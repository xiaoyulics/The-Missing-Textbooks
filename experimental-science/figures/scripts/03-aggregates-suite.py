"""Lab 3.1 of *Deep Learning as an Experimental Science*, Chapter 3 (Many Tasks, Few Runs).

A miniature of the Atari 100k benchmark: a suite of twelve tasks built from the scikit-learn
digits, two training methods, and fifty runs of each method on each task.

Tasks. Each task changes the digits problem in one way (fewer training examples, label noise,
pixel noise, occluded rows, a test set shifted by one pixel). Every task gets the same budget of
400 optimizer steps with batch 32, as every Atari 100k game gets 100k interactions.

Scores. A run's test accuracy is normalized as
    score = (accuracy - chance) / (reference - chance),
with chance 0.1 and, as reference, the test accuracy of a 1-nearest-neighbor classifier fitted
to the same training set: the miniature's analog of the human-normalized score, with the
reference playing the human. Where the reference is weak (label noise), scores exceed 1.

Methods. A: an MLP with two hidden layers of 64 ReLU units, SGD with momentum 0.9, constant
learning rate 0.05. B: method A with label smoothing 0.1. C: method A with dropout 0.2 (quoted in
the text only). Method m uses the seeds init = order = noise = 1000 m + i, i = 0..49. Each run is
evaluated at the end of training (the "final" protocol) and every 40 steps; "max" is the best of
these ten evaluations, the protocol of reporting the maximum over evaluations during training.

Aggregates. The mean and the median of the task means, the interquartile mean (IQM, the 25%
trimmed mean of all runs pooled over tasks), the optimality gap at 1, and the average probability
of improvement; 95% percentile intervals from the stratified bootstrap (runs resampled with
replacement within each task, 5000 resamples), for the first 3 runs and for all 50 runs.

Run from the volume root:  python3 figures/scripts/03-aggregates-suite.py
Writes figures/data/03-aggregates-runs.dat, -taskruns.dat, -tasks.dat, -agg.dat and prints the
numbers quoted in the text. Runtime: two to four minutes on a laptop CPU with two threads.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace

import numpy as np
import torch
from scipy.stats import trim_mean

import labkit as lk

STEPS, BATCH, EVAL_EVERY = 400, 32, 40
N_RUNS = 50
CHANCE = 0.1

# name, label for figures, task parameters
TASKS = [
    ("clean", "all data", dict()),
    ("n400", "400 examples", dict(train_size=400)),
    ("n150", "150 examples", dict(train_size=150)),
    ("n60", "60 examples", dict(train_size=60)),
    ("noise20", "20\\% label noise", dict(label_noise=0.2)),
    ("noise40", "40\\% label noise", dict(label_noise=0.4)),
    ("noise60", "60\\% label noise", dict(label_noise=0.6)),
    ("n400noise30", "400 ex., 30\\% noise", dict(train_size=400, label_noise=0.3)),
    ("pixel", "pixel noise", dict(pixel_noise=0.3)),
    ("occl3", "3 rows occluded", dict(occlude=3)),
    ("occl5", "5 rows occluded", dict(occlude=5)),
    ("shift", "shifted test set", dict(test_shift=1)),
]
BASE = lk.Config(width=64, depth=2, optimizer="momentum", lr=0.05, momentum=0.9,
                 steps=STEPS, epochs=None, batch_size=BATCH, eval_every=EVAL_EVERY)
METHODS = {"A": BASE,
           "B": replace(BASE, label_smoothing=0.1),
           "C": replace(BASE, dropout=0.2)}
SEED_OFFSET = {"A": 0, "B": 1000, "C": 2000}

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_FILE = os.path.normpath(os.path.join(HERE, "..", "data", "03-aggregates-runs.dat"))


# ---------------------------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------------------------
def _raw_digits():
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.data.astype(np.float32) / 16.0, d.target.astype(np.int64)


def _shift_right(X, dx):
    img = X.reshape(-1, 8, 8)
    out = np.zeros_like(img)
    out[:, :, dx:] = img[:, :, :8 - dx]
    return out.reshape(-1, 64)


def make_task(train_size=None, label_noise=0.0, pixel_noise=0.0, occlude=0, test_shift=0, split=0):
    """Training and test tensors of one task. The split follows labkit.digits (30% test, seed
    10000 + split); the corruption is applied to the raw images, which are then standardized
    with the training-set mean and standard deviation of each pixel."""
    X, y = _raw_digits()
    rng = np.random.default_rng(10_000 + split)
    perm = rng.permutation(len(y))
    n_test = int(round(0.3 * len(y)))
    te, tr = perm[:n_test], perm[n_test:]
    Xtr, ytr, Xte, yte = X[tr].copy(), y[tr].copy(), X[te].copy(), y[te].copy()
    if train_size is not None:
        Xtr, ytr = Xtr[:train_size], ytr[:train_size]
    if label_noise > 0:
        r = np.random.default_rng(123)
        flip = r.random(len(ytr)) < label_noise
        ytr[flip] = r.integers(0, 10, size=flip.sum())
    if occlude:
        for A in (Xtr, Xte):
            A.reshape(-1, 8, 8)[:, 8 - occlude:, :] = 0.0
    if test_shift:
        Xte = _shift_right(Xte, test_shift)
    if pixel_noise > 0:
        r = np.random.default_rng(456)
        Xtr = Xtr + pixel_noise * r.standard_normal(Xtr.shape).astype(np.float32)
        Xte = Xte + pixel_noise * r.standard_normal(Xte.shape).astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    return t(Xtr), t(ytr), t(Xte), t(yte)


def reference_accuracy(data):
    """Test accuracy of the 1-nearest-neighbor classifier fitted to the training set."""
    Xtr, ytr, Xte, yte = data
    return (ytr[torch.cdist(Xte, Xtr).argmin(1)] == yte).float().mean().item()


# ---------------------------------------------------------------------------------------------
# The bank of runs
# ---------------------------------------------------------------------------------------------
def train_bank(n_runs=N_RUNS, verbose=True):
    """Train every method on every task n_runs times. Returns dict with arrays of shape
    (methods, tasks, runs) for final and max accuracy and scores, and the reference accuracies."""
    J = len(TASKS)
    names = list(METHODS)
    acc = np.zeros((len(names), J, n_runs))
    accmax = np.zeros_like(acc)
    ref = np.zeros(J)
    t0 = time.time()
    for j, (tname, _, kw) in enumerate(TASKS):
        data = make_task(**kw)
        ref[j] = reference_accuracy(data)
        for m, mname in enumerate(names):
            for i in range(n_runs):
                s = SEED_OFFSET[mname] + i
                r = lk.run(METHODS[mname], init=s, order=s, noise=s, data=data)
                acc[m, j, i] = r["test_acc"]
                accmax[m, j, i] = max(c[2] for c in r["curve"])
        if verbose:
            print(f"  trained task {tname:12s} (reference {ref[j]:.3f}) {time.time() - t0:6.1f}s", flush=True)
    score = (acc - CHANCE) / (ref[None, :, None] - CHANCE)
    scoremax = (accmax - CHANCE) / (ref[None, :, None] - CHANCE)
    return dict(methods=names, acc=acc, accmax=accmax, ref=ref, score=score, scoremax=scoremax)


def save_bank(bank, path=RUNS_FILE):
    names, J, n = bank["methods"], len(TASKS), bank["score"].shape[2]
    cols = {k: [] for k in ("task", "method", "run", "ref", "acc", "accmax", "score", "scoremax")}
    for m, mname in enumerate(names):
        for j in range(J):
            for i in range(n):
                cols["task"].append(j + 1)
                cols["method"].append(mname)
                cols["run"].append(i)
                cols["ref"].append(bank["ref"][j])
                cols["acc"].append(bank["acc"][m, j, i])
                cols["accmax"].append(bank["accmax"][m, j, i])
                cols["score"].append(bank["score"][m, j, i])
                cols["scoremax"].append(bank["scoremax"][m, j, i])
    lk.write_dat(path, cols, fmt="{:.6g}")


def load_bank(path=RUNS_FILE):
    """Read the runs written by save_bank (None if the file does not exist)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        header = f.readline().split()
        rows = [line.split() for line in f if line.strip()]
    col = {h: [r[k] for r in rows] for k, h in enumerate(header)}
    names = sorted(set(col["method"]))
    J = max(int(t) for t in col["task"])
    n = max(int(r) for r in col["run"]) + 1
    out = {k: np.zeros((len(names), J, n)) for k in ("acc", "accmax", "score", "scoremax")}
    for k in range(len(rows)):
        m, j, i = names.index(col["method"][k]), int(col["task"][k]) - 1, int(col["run"][k])
        for key in out:
            out[key][m, j, i] = float(col[key][k])
    ref = np.zeros(J)
    for k in range(len(rows)):
        ref[int(col["task"][k]) - 1] = float(col["ref"][k])
    out.update(methods=names, ref=ref)
    return out


# ---------------------------------------------------------------------------------------------
# Aggregates and the stratified bootstrap
# ---------------------------------------------------------------------------------------------
def agg_mean(S):
    """S has shape (..., J, n): mean over tasks of the task means."""
    return S.mean(-1).mean(-1)


def agg_median(S):
    return np.median(S.mean(-1), axis=-1)


def agg_iqm(S):
    flat = S.reshape(S.shape[:-2] + (-1,))
    return trim_mean(flat, 0.25, axis=-1)


def agg_optgap(S, gamma=1.0):
    return gamma - np.minimum(S, gamma).mean((-2, -1))


AGGREGATES = {"mean": agg_mean, "median": agg_median, "iqm": agg_iqm, "optgap": agg_optgap}


def prob_improvement(SX, SY):
    """Average over tasks of the Mann-Whitney estimate of P(X > Y) + P(X = Y)/2.
    SX, SY have shape (..., J, n)."""
    gt = (SX[..., :, None] > SY[..., None, :]).mean((-2, -1))
    eq = (SX[..., :, None] == SY[..., None, :]).mean((-2, -1))
    return (gt + 0.5 * eq).mean(-1)


def strat_resample(S, B, rng):
    """B stratified bootstrap resamples of S (shape (J, n)): runs drawn with replacement within
    each task. Returns an array of shape (B, J, n)."""
    J, n = S.shape
    idx = rng.integers(0, n, size=(B, J, n))
    return S[np.arange(J)[None, :, None], idx]


def boot_interval(S, stat, B=5000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    reps = stat(strat_resample(S, B, rng))
    a = (1 - level) / 2
    return float(stat(S)), float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))


def boot_interval_diff(SX, SY, stat, B=5000, seed=0, level=0.95):
    """Interval for stat(X) - stat(Y), resampling the two methods independently."""
    rng = np.random.default_rng(seed)
    reps = stat(strat_resample(SX, B, rng)) - stat(strat_resample(SY, B, rng))
    a = (1 - level) / 2
    return float(stat(SX) - stat(SY)), float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))


def boot_interval_poi(SX, SY, B=1000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    reps = np.array([prob_improvement(strat_resample(SX, 1, rng)[0], strat_resample(SY, 1, rng)[0])
                     for _ in range(B)])
    a = (1 - level) / 2
    return float(prob_improvement(SX, SY)), float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))


# ---------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    if "--reuse" in sys.argv and os.path.exists(RUNS_FILE):
        print("Lab 3.1: reusing the bank in", RUNS_FILE)
        bank = load_bank()
    else:
        print("Lab 3.1: training the bank (3 methods x 12 tasks x 50 runs) ...")
        bank = train_bank()
        save_bank(bank)
    names = bank["methods"]
    SA, SB, SC = (bank["score"][names.index(k)] for k in ("A", "B", "C"))
    SAmax = bank["scoremax"][names.index("A")]
    SBmax = bank["scoremax"][names.index("B")]
    J = len(TASKS)
    train_seconds = time.time() - t0
    print(f"training took {train_seconds:.1f}s")

    # --- per task -----------------------------------------------------------------------------
    order = np.argsort(SA.mean(1))            # tasks sorted by method A's mean score
    print("\nPer task (sorted by A's mean score): reference accuracy; mean score with 95% t-interval, 50 runs;"
          " B - A; run SD of A")
    trows = {k: [] for k in ("pos", "task", "ref", "meanA", "hwA", "meanB", "hwB", "mean3A", "mean3B",
                             "diff", "sdA", "sdB", "gainmax")}
    for pos, j in enumerate(order, start=1):
        mA, hA = lk.t_interval(SA[j])
        mB, hB = lk.t_interval(SB[j])
        trows["pos"].append(pos); trows["task"].append(TASKS[j][0]); trows["ref"].append(bank["ref"][j])
        trows["meanA"].append(mA); trows["hwA"].append(hA); trows["meanB"].append(mB); trows["hwB"].append(hB)
        trows["mean3A"].append(SA[j, :3].mean()); trows["mean3B"].append(SB[j, :3].mean())
        trows["diff"].append(mB - mA); trows["sdA"].append(SA[j].std(ddof=1)); trows["sdB"].append(SB[j].std(ddof=1))
        trows["gainmax"].append(SAmax[j].mean() - mA)
        print(f"  {pos:2d} {TASKS[j][0]:12s} ref {bank['ref'][j]:.3f}  A {mA:.3f}±{hA:.3f}  B {mB:.3f}±{hB:.3f}"
              f"  B-A {mB - mA:+.4f}  sdA {SA[j].std(ddof=1):.4f}  sdB {SB[j].std(ddof=1):.4f}"
              f"  maxgain(A) {SAmax[j].mean() - mA:+.3f}")
    lk.write_dat("figures/data/03-aggregates-tasks.dat", trows, fmt="{:.5g}")

    # runs for the per-task figure: x position with a deterministic jitter
    jit = np.random.default_rng(7).uniform(-0.09, 0.09, size=(2, J, N_RUNS))
    rr = {"xA": [], "scoreA": [], "xB": [], "scoreB": []}
    for pos, j in enumerate(order, start=1):
        for i in range(N_RUNS):
            rr["xA"].append(pos - 0.2 + jit[0, j, i]); rr["scoreA"].append(SA[j, i])
            rr["xB"].append(pos + 0.2 + jit[1, j, i]); rr["scoreB"].append(SB[j, i])
    lk.write_dat("figures/data/03-aggregates-taskruns.dat", rr, fmt="{:.5g}")

    # --- aggregates ---------------------------------------------------------------------------
    print("\nAggregates with 95% stratified bootstrap intervals (5000 resamples)")
    configs = [("A, 3 runs", SA[:, :3]), ("B, 3 runs", SB[:, :3]), ("A, 50 runs", SA),
               ("B, 50 runs", SB), ("A, max protocol, 50 runs", SAmax)]
    agg = {"y": [5, 4, 3, 2, 1]}
    for key in ("mean", "median", "iqm", "optgap"):
        est, em, ep = [], [], []
        for c, (label, S) in enumerate(configs):
            e, lo, hi = boot_interval(S, AGGREGATES[key], seed=11 + c)
            est.append(e); em.append(e - lo); ep.append(hi - e)
            print(f"  {key:7s} {label:26s} {e:.4f}  [{lo:.4f}, {hi:.4f}]  width {hi - lo:.4f}")
        agg[key] = est; agg[key + "em"] = em; agg[key + "ep"] = ep
    lk.write_dat("figures/data/03-aggregates-agg.dat", agg, fmt="{:.5g}")

    print("\nDifferences B - A with 95% stratified bootstrap intervals")
    for n in (3, 50):
        for key in ("mean", "median", "iqm"):
            e, lo, hi = boot_interval_diff(SB[:, :n], SA[:, :n], AGGREGATES[key], seed=21)
            print(f"  {key:7s} n={n:2d}: {e:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
        e, lo, hi = boot_interval_poi(SB[:, :n], SA[:, :n], seed=31)
        print(f"  P(B > A) n={n:2d}: {e:.3f}  [{lo:.3f}, {hi:.3f}]")
    print("\nThe max protocol for A against the final protocol for B (50 runs): (A max) - (B final)")
    for key in ("mean", "median", "iqm"):
        e, lo, hi = boot_interval_diff(SAmax, SB, AGGREGATES[key], seed=41)
        print(f"  {key:7s}: {e:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print("  same protocol for both, max: (B max) - (A max)")
    for key in ("mean", "median", "iqm"):
        e, lo, hi = boot_interval_diff(SBmax, SAmax, AGGREGATES[key], seed=42)
        print(f"  {key:7s}: {e:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    print("\nMethod C (dropout 0.2) against A, 50 runs: the aggregate decides the winner")
    for key in ("mean", "median", "iqm"):
        e, lo, hi = boot_interval_diff(SC, SA, AGGREGATES[key], seed=51)
        print(f"  {key:7s} C - A: {e:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    wins = int(((SC.mean(1) - SA.mean(1)) > 0).sum())
    print(f"  C beats A on {wins} of {J} tasks (task means, 50 runs)")
    print("  per task, in the order of A's mean score: C - A (task means, 50 runs)")
    for pos, j in enumerate(order, start=1):
        print(f"    {pos:2d} {TASKS[j][0]:12s} {SC[j].mean() - SA[j].mean():+.4f}")
    winsB = int(((SB.mean(1) - SA.mean(1)) > 0).sum())
    print(f"  B beats A on {winsB} of {J} tasks (task means, 50 runs)")

    # tasks as the sampling units (Demsar 2006): tests across the twelve tasks, 50-run task means
    from scipy import stats
    print("\nAcross tasks (task means, 50 runs): tasks as the sampling units")
    for lab, SX in (("B - A", SB), ("C - A", SC)):
        dj = SX.mean(1) - SA.mean(1)
        w = stats.wilcoxon(dj, alternative="two-sided")
        print(f"  {lab}: between-task SD of the per-task difference {dj.std(ddof=1):.4f};"
              f" task-level SE of the mean difference {dj.std(ddof=1) / np.sqrt(J):.4f};"
              f" Wilcoxon signed-rank p = {w.pvalue:.4g}; wins {int((dj > 0).sum())}/{J}")
    means3 = np.stack([SA.mean(1), SB.mean(1), SC.mean(1)], axis=1)       # (J, 3)
    ranks = np.array([stats.rankdata(-row) for row in means3])           # rank 1 = best
    fr = stats.friedmanchisquare(SA.mean(1), SB.mean(1), SC.mean(1))
    q05 = 2.343                                                          # Nemenyi, k = 3 (Demsar 2006, Table 5)
    cd = q05 * np.sqrt(3 * 4 / (6 * J))
    print(f"  average ranks A {ranks[:, 0].mean():.2f}, B {ranks[:, 1].mean():.2f}, C {ranks[:, 2].mean():.2f};"
          f" Friedman chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.3g}; Nemenyi critical difference {cd:.3f}")

    # where do the aggregates put their weight? share of the mean's run variance per task
    v = SA.var(1, ddof=1) / N_RUNS
    share = v / v.sum()
    top = np.argsort(share)[::-1][:3]
    print("\nShare of the run-noise variance of A's mean aggregate contributed by each task (top 3):")
    for j in top:
        print(f"  {TASKS[j][0]:12s} {share[j]:.2f}")
    print(f"\ntotal runtime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
