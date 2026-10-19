"""Chapter 12 (Verifying Learning): labeled examples used by the verifier of Protocol 12.1 against
labeled examples a learner needs on its own, as the VC dimension d grows.

The class is the histogram classifiers with d equal bins on [0,1) (VC dimension d, |H| = 2^d). The
distributions: inputs uniform; in bin j the label function has a fraction 1/2 + s_j*gamma of ones
(a fine deterministic pattern, as in 12-pacverify-delegation.py), so that for i.i.d. inputs the
bin counts are multinomial and the labels in bin j are i.i.d. Bernoulli(1/2 + s_j*gamma): the
simulation below samples them directly. The learner is ERM (majority vote per bin, ties to 0). It
fails when its excess loss, 2*gamma*(wrong bins)/d, exceeds eps.

For each d and each gamma in a grid we find the smallest m on a geometric grid (ratio 1.12) at
which the failure rate over 2000 trials is at most delta there and at the next three grid points,
and report the largest over gamma. The verifier's numbers are q = ceil(4 ln(2/delta)/eps) labeled
examples and m = ceil(ln(2^(d+1)/(delta/2))/(2(eps/4)^2)) points labeled by the prover.
Output: figures/data/12-pacverify-labels.dat. Seed 1210. Runs in about half a minute.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

EPS, DELTA = 0.1, 0.1
GAMMAS = (0.12, 0.16, 0.2, 0.25, 0.3, 0.4)
DIMS = (4, 16, 64, 256, 1024)
TRIALS = 2000
GRID = sorted(set(int(round(1.12 ** g)) for g in range(12, 125)))


def failure_rate(gen, d, signs, gamma, m):
    qj = 0.5 + gamma * signs
    n = gen.multinomial(m, np.full(d, 1.0 / d), size=TRIALS)
    n1 = gen.binomial(n, qj)
    out = (n1 > n - n1).astype(np.int8)
    wrong = np.sum(out != (qj > 0.5), axis=1)
    return float(np.mean(2 * gamma * wrong > EPS * d + 1e-9))


def labels_needed(gen, d, signs, gamma):
    for i, m in enumerate(GRID):
        if failure_rate(gen, d, signs, gamma, m) <= DELTA:
            if all(failure_rate(gen, d, signs, gamma, GRID[i + k]) <= DELTA for k in (1, 2, 3)):
                return m
    return None


def main():
    t0 = time.time()
    gen = pk.rng(1210)
    q = math.ceil(4 * math.log(2 / DELTA) / EPS)
    rows = []
    for d in DIMS:
        signs = np.where(gen.random(d) < 0.5, -1, 1)
        need = {g: labels_needed(gen, d, signs, g) for g in GAMMAS}
        worst = max(v for v in need.values() if v is not None)
        m = math.ceil(((d + 1) * math.log(2) + math.log(2 / DELTA)) / (2 * (EPS / 4) ** 2))
        rows.append([d, q, m, worst, worst / d])
        print(f"d = {d:5d}: verifier labels q = {q}, prover labels m = {m}; ERM needs {worst} labels "
              f"({worst/d:.2f} per dimension); by gamma: {need}")
    pk.write_dat("figures/data/12-pacverify-labels.dat", ["d", "q", "m", "learner", "perd"], rows)
    print(f"time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
