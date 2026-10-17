"""Figure 2 of Chapter 16 (Scaling Laws): learning curves of a memorizing learner on
Zipf-distributed items.

Items k = 1..K with p_k proportional to k^{-(1+b)} (normalized over K = 10^7 items).
A learner that has memorized every item seen among D independent samples errs with
probability M(D) = sum_k p_k (1 - p_k)^D (exact, curves). Simulation: draw the counts of
D samples (multinomial), and score the learner by the exact mass of unseen items;
20 trials per point (seed 1618). Asymptote: Gamma(b/(1+b)) / ((1+b) zeta(1+b)^{1/(1+b)})
* D^{-b/(1+b)} for the untruncated law.
Panel (b): b = 1, probability that item k has been seen, 1 - (1 - p_k)^D, for
k = 1, 10, 100, 1000, together with M(D).

Writes figures/data/16-scaling-zipf-theory.dat, 16-scaling-zipf-sim.dat and
16-scaling-zipf-items.dat. Deterministic; about 30 s on one core.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.special import gamma, zeta

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
K = 10 ** 7
BS = [0.5, 1.0, 2.0]
rng = np.random.default_rng(1618)
k = np.arange(1, K + 1, dtype=float)
probs = {}
for b in BS:
    p = k ** -(1 + b)
    probs[b] = p / p.sum()

def M(p, D):
    return float(np.sum(p * np.exp(D * np.log1p(-p))))

def const(b):
    return gamma(b / (1 + b)) / ((1 + b) * zeta(1 + b) ** (1 / (1 + b)))

Dth = np.logspace(0, 6, 61)
with open(os.path.join(DATA, "16-scaling-zipf-theory.dat"), "w") as f:
    f.write("D " + " ".join(f"exact{i} asym{i}" for i in range(len(BS))) + "\n")
    for D in Dth:
        row = []
        for b in BS:
            row += [f"{M(probs[b], D):.6g}", f"{const(b) * D ** (-b / (1 + b)):.6g}"]
        f.write(f"{D:.6g} " + " ".join(row) + "\n")

Dsim = [3, 10, 30, 100, 300, 1000, 3000, 10 ** 4, 3 * 10 ** 4, 10 ** 5, 3 * 10 ** 5, 10 ** 6]
TRIALS = 20
with open(os.path.join(DATA, "16-scaling-zipf-sim.dat"), "w") as f:
    f.write("D " + " ".join(f"sim{i}" for i in range(len(BS))) + "\n")
    for D in Dsim:
        row = []
        for b in BS:
            p = probs[b]
            cdf = np.cumsum(p)
            errs = []
            for t in range(TRIALS):
                idx = np.searchsorted(cdf, rng.random(D) * cdf[-1])
                seen = np.zeros(K, dtype=bool); seen[np.minimum(idx, K - 1)] = True
                errs.append(p[~seen].sum())
            row.append(f"{np.mean(errs):.6g}")
            print(b, D, np.mean(errs), M(p, D), np.std(errs) / np.sqrt(TRIALS))
        f.write(f"{D} " + " ".join(row) + "\n")

p = probs[1.0]
with open(os.path.join(DATA, "16-scaling-zipf-items.dat"), "w") as f:
    f.write("D k1 k10 k100 k1000 loss\n")
    for D in np.logspace(0, 7, 141):
        vals = [1 - np.exp(D * np.log1p(-p[i - 1])) for i in (1, 10, 100, 1000)]
        f.write(f"{D:.6g} " + " ".join(f"{v:.6g}" for v in vals) + f" {M(p, D):.6g}\n")
for b in BS:
    print(b, const(b))
