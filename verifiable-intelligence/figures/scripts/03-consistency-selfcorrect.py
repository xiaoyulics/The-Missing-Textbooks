"""Chapter 3 (Consistency): self-correction of a function close to linear, at the input that matters.

f(z) = <a, z> mod 2 on F_2^n (n = 16), a hidden linear function. A program g agrees with f except on
a set E with |E| = delta 2^n, and E always contains the input that matters x* (the program's own
answer at x* is wrong). The self-corrector outputs g(x* + r) + g(r) for a uniform r, or the
majority of t = 9 independent such values. Programs:
  adversarial  E inside the half-space {z_i = 0} plus x*, where x*_i = 1, so E and E + x* are
               disjoint: the error of one trial is exactly 2 delta (the bound is attained)
  random       x* plus a uniformly random set: error 2 delta (1 - delta) N/(N-1) in expectation
  memory       not a fixed function: it remembers its previous query and, when the two queries
               add up to x*, answers so that the corrected value is wrong (error 1)
A second experiment compares, for the program that is wrong only at x*, a checker that compares the
program with the truth at 100 random points and the self-correcting checker (majority of 9).
Writes figures/data/03-consistency-selfcorrect.dat and figures/data/03-consistency-selfcorrect-theory.dat
Run from the volume root:  python3 figures/scripts/03-consistency-selfcorrect.py
"""
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk

NBITS = 16
NPTS = 2 ** NBITS
TRIALS = 20_000
T_MAJ = 9


def parity(v):
    """Parity of the bits of each nonnegative integer in v (int64 array)."""
    v = v.copy()
    for s in (32, 16, 8, 4, 2, 1):
        v ^= v >> s
    return v & 1


def make_programs(a, xstar, delta, gen):
    """Truth table of f and corrupted tables g (adversarial, random) with |E| = round(delta N)."""
    z = np.arange(NPTS, dtype=np.int64)
    f = parity(z & a)
    m = int(round(delta * NPTS))
    tables = {}
    # adversarial: x* plus m - 1 points of the half-space z_i = 0 (not 0 itself), x*_i = 1
    i = int(np.flatnonzero([(xstar >> b) & 1 for b in range(NBITS)])[0])
    half = z[((z >> i) & 1) == 0]
    half = half[half != 0]
    E = np.concatenate([[xstar], gen.choice(half, size=max(m - 1, 0), replace=False)]) if m > 0 else np.array([], dtype=np.int64)
    g = f.copy(); g[E] ^= 1
    assert m == 0 or not np.any(np.isin(E ^ xstar, E))      # E and E + x* are disjoint
    tables["adversarial"] = g
    # random: x* plus m - 1 uniformly random other points
    others = z[z != xstar]
    E = np.concatenate([[xstar], gen.choice(others, size=max(m - 1, 0), replace=False)]) if m > 0 else np.array([], dtype=np.int64)
    g = f.copy(); g[E] ^= 1
    tables["random"] = g
    return f, tables, m


def corrected_errors(g, f, xstar, gen, trials, t):
    """Fraction of runs whose corrected value (majority of t trials) differs from f(x*)."""
    r = gen.integers(0, NPTS, size=(trials, t), dtype=np.int64)
    vals = g[xstar ^ r] ^ g[r]                    # in F_2, g(x*+r) - g(r) = g(x*+r) + g(r)
    maj = (vals.sum(axis=1) * 2 > t).astype(np.int64)
    return float(np.mean(maj != f[xstar])), int(np.sum(maj != f[xstar]))


class MemoryProgram:
    """Answers f correctly, except that when the current query and the previous one add up to x*
    it returns the value that makes their sum wrong: a program with state."""
    def __init__(self, f, xstar):
        self.f, self.xstar, self.prev = f, xstar, None

    def __call__(self, q):
        ans = int(self.f[q])
        if self.prev is not None and (self.prev[0] ^ q) == self.xstar:
            ans = self.prev[1] ^ int(self.f[self.xstar]) ^ 1
        self.prev = (q, ans)
        return ans


def main():
    gen = pk.rng(7)
    a = int(gen.integers(1, NPTS))
    xstar = int(gen.integers(1, NPTS))
    deltas = np.round(np.arange(0.0, 0.3001, 0.02), 4)
    rows = []
    print(f"self-correction on F_2^{NBITS}, {TRIALS} runs per point; x* = {xstar}, a = {a}")
    print(" delta   adversarial (2d)    random (2d(1-d)N/(N-1))   maj-9 adv (binomial)   memory")
    for d in deltas:
        f, tables, m = make_programs(a, xstar, d, pk.rng(100 + int(1000 * d)))
        dd = m / NPTS
        e_adv, k_adv = corrected_errors(tables["adversarial"], f, xstar, gen, TRIALS, 1)
        e_rnd, k_rnd = corrected_errors(tables["random"], f, xstar, gen, TRIALS, 1)
        e_maj, k_maj = corrected_errors(tables["adversarial"], f, xstar, gen, TRIALS, T_MAJ)
        prog = MemoryProgram(f, xstar)
        wrong = 0
        for _ in range(2000):
            r = int(gen.integers(0, NPTS))
            wrong += (prog(xstar ^ r) ^ prog(r)) != int(f[xstar])
        e_mem = wrong / 2000
        th_rnd = 2 * dd * (1 - dd) * NPTS / (NPTS - 1)
        th_maj = float(stats.binom.sf((T_MAJ - 1) // 2, T_MAJ, min(2 * dd, 1.0)))
        cis = [pk.clopper_pearson(k, TRIALS) for k in (k_adv, k_rnd, k_maj)]
        print(f" {d:.2f}   {e_adv:.4f} ({2*dd:.4f})     {e_rnd:.4f} ({th_rnd:.4f})          "
              f"{e_maj:.4f} ({th_maj:.4f})       {e_mem:.3f}")
        rows.append((d, e_adv, e_adv - cis[0][0], cis[0][1] - e_adv,
                     e_rnd, e_rnd - cis[1][0], cis[1][1] - e_rnd,
                     e_maj, e_maj - cis[2][0], cis[2][1] - e_maj, e_mem))
    pk.write_dat("figures/data/03-consistency-selfcorrect.dat",
                 ["delta", "adv", "advem", "advep", "rnd", "rndem", "rndep", "maj", "majem", "majep", "mem"], rows)

    grid = np.linspace(0.0, 0.3, 61)
    trows = [(d, 2 * d, 2 * d * (1 - d), float(stats.binom.sf((T_MAJ - 1) // 2, T_MAJ, 2 * d)),
              float(np.exp(-2 * T_MAJ * (0.5 - 2 * d) ** 2)) if d < 0.25 else 1.0) for d in grid]
    pk.write_dat("figures/data/03-consistency-selfcorrect-theory.dat",
                 ["delta", "bound", "rnd", "maj", "hoeff"], trows)

    # the input that matters: a program wrong only at x*
    f, _, _ = make_programs(a, xstar, 0.0, pk.rng(1))
    g = f.copy(); g[xstar] ^= 1
    runs = 20_000
    zs = gen.integers(0, NPTS, size=(runs, 100))
    spot_pass = int(np.sum(np.all(g[zs] == f[zs], axis=1)))
    e, k = corrected_errors(g, f, xstar, gen, runs, T_MAJ)      # wrong majority = checker accepts g(x*)
    print(f"\nprogram wrong only at x*: 100 random spot checks pass in {spot_pass} of {runs} runs; "
          f"the self-correcting checker (majority of {T_MAJ}) accepts its answer at x* in {k} of {runs} runs")

    # reference numbers, not quoted in the chapter: the binomial tail against Hoeffding at delta = 0.1 (2 delta = 0.2)
    for t in range(1, 200, 2):
        if stats.binom.sf((t - 1) // 2, t, 0.2) <= 1e-6:
            print(f"delta = 0.1: smallest odd t with P[Bin(t, 0.2) > t/2] <= 1e-6 is {t} "
                  f"(tail {stats.binom.sf((t - 1) // 2, t, 0.2):.2e}); Hoeffding needs t >= "
                  f"{np.log(1e6) / (2 * 0.3 ** 2):.1f}")
            break
    print(f"delta = 0.1, t = 9: binomial tail {stats.binom.sf(4, 9, 0.2):.4f}, Hoeffding {np.exp(-18 * 0.09):.4f}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nruntime {time.perf_counter() - t0:.1f} s")
