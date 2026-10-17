"""Lab 5.3 (Chapter 5, Controls): a historical control and the pipeline as a co-treatment.

A 2 x 2 factorial on the digits data of Labs 5.1 and 5.2 (958 training images, 539 test
images): two architectures crossed with two training pipelines, plus a placebo arm.

  architectures  the textbook MLP (2 hidden layers of 64 ReLU units, no dropout, no
                 batch normalization) and the architecture "found" in Lab 5.2: the one in the
                 curated space with the lowest mean validation error after full training
                 (read from figures/data/05-controls-search-proxy.dat; run 05-controls-space.py
                 and 05-controls-search.py first)
  pipelines      old: Adam, learning rate 1e-3, constant, 30 epochs (the pipeline of Lab 5.1)
                 new: Adam, learning rate 2e-3, cosine schedule, 60 epochs, label smoothing 0.1
  replicates     10 runs per cell, seeds 3000..3009 (init = order = noise)
  placebo        16 architectures drawn uniformly from the curated space (seed 21), 2 runs
                 each (seeds 3000, 3001), under both pipelines

The historical comparison sets the found architecture under the new pipeline against the
textbook MLP under the old one, as a paper does that copies a baseline number from an older
paper. Writes figures/data/05-controls-pipeline.dat and prints the decomposition quoted in
the chapter. Runtime about one minute on two threads.
Run from the volume root: python3 figures/scripts/05-controls-pipeline.py
"""
import os
import sys
import time
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk  # noqa: E402
import numpy as np  # noqa: E402

SPACE = list(itertools.product((1, 2, 3, 4), (16, 32, 64, 128), ("relu", "tanh", "gelu"),
                               (0.0, 0.25), (False, True)))
PIPES = {"old": dict(lr=1e-3, epochs=30),
         "new": dict(lr=2e-3, epochs=60, schedule="cosine", label_smoothing=0.1)}
SEEDS = list(range(3000, 3010))
TEXTBOOK = (2, 64, "relu", 0.0, False)


def run_arch(arch, pipe, seeds, data):
    depth, width, act, drop, bn = arch
    cfg = lk.Config(depth=depth, width=width, activation=act, dropout=drop, batchnorm=bn,
                    optimizer="adam", batch_size=64, **PIPES[pipe])
    return np.array([lk.run(cfg, init=s, order=s, noise=s, data=data)["test_err"] for s in seeds])


def main():
    t0 = time.time()
    path = os.path.join(lk.DATA_DIR, "05-controls-search-proxy.dat")
    if not os.path.exists(path):
        sys.exit("run 05-controls-space.py and 05-controls-search.py first")
    tab = np.genfromtxt(path, names=True)
    found = SPACE[int(tab["id"][tab["found"] == 1][0])]
    Xtr, ytr, Xte, yte = lk.digits(split=0)
    perm = np.random.default_rng(5).permutation(len(ytr))
    data = (Xtr[perm[:958]], ytr[perm[:958]], Xte, yte)
    curated = [a for a in SPACE if a[1] >= 64]
    rng = np.random.default_rng(21)
    placebo = [curated[i] for i in rng.choice(len(curated), 16, replace=False)]

    cells, rows = {}, {"arm": [], "pipe": [], "mean": [], "hw": []}
    for pipe in PIPES:
        for name, arch in (("textbook", TEXTBOOK), ("found", found)):
            cells[name, pipe] = run_arch(arch, pipe, SEEDS, data)
        # placebo: the mean over two seeds of each of 16 random architectures
        cells["placebo", pipe] = np.array([run_arch(a, pipe, SEEDS[:2], data).mean() for a in placebo])
    for (name, pipe), x in cells.items():
        m, h = lk.t_interval(x)
        rows["arm"].append({"textbook": 1, "found": 2, "placebo": 3}[name])
        rows["pipe"].append({"old": 1, "new": 2}[pipe])
        rows["mean"].append(100 * m)
        rows["hw"].append(100 * h)
        print(f"{name:8s} {pipe}: {100 * m:.2f}% +- {100 * h:.2f} (n = {len(x)})")
    lk.write_dat("figures/data/05-controls-pipeline.dat", rows)

    def diff(a, b):
        """Welch difference of means a - b in points, with a 95% half-width."""
        from scipy import stats
        x, y = cells[a], cells[b]
        se2 = x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y)
        df = se2 ** 2 / ((x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1)
                         + (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1))
        return 100 * (x.mean() - y.mean()), 100 * stats.t.ppf(0.975, df) * np.sqrt(se2)

    print(f"found architecture: {found}; placebo architectures: {len(placebo)}")
    for label, a, b in (("historical comparison: textbook/old - found/new", ("textbook", "old"), ("found", "new")),
                        ("pipeline effect on the textbook MLP: old - new", ("textbook", "old"), ("textbook", "new")),
                        ("pipeline effect on the found architecture", ("found", "old"), ("found", "new")),
                        ("pipeline effect on the placebo arm", ("placebo", "old"), ("placebo", "new")),
                        ("architecture effect, old pipeline: textbook - found", ("textbook", "old"), ("found", "old")),
                        ("architecture effect, new pipeline: textbook - found", ("textbook", "new"), ("found", "new")),
                        ("placebo-adjusted effect, old pipeline: placebo - found", ("placebo", "old"), ("found", "old")),
                        ("placebo-adjusted effect, new pipeline: placebo - found", ("placebo", "new"), ("found", "new"))):
        d, h = diff(a, b)
        print(f"{label}: {d:+.2f} +- {h:.2f} pp")
    inter = (cells["textbook", "old"].mean() - cells["textbook", "new"].mean()) \
        - (cells["found", "old"].mean() - cells["found", "new"].mean())
    print(f"interaction (pipeline effect on textbook minus on found): {100 * inter:+.2f} pp")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
