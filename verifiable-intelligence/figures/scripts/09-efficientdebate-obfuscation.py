"""Chapter 9 (Efficient Debate): obfuscated arguments in recursive debate.

A dishonest debater A' defends a false claim with an argument tree of depth D and N = 2^D leaves,
built with Barnes's recipe: a claim P is supported by the two claims "Q => P" and
"(Q => P) => P" for a fresh claim Q that neither debater can evaluate. In recursive debate the
challenger B picks a child at every level and the judge checks the leaf B reaches.

The script
  1. builds such trees as propositional formulas over random truth values and checks that a false
     root has exactly one false leaf, at a uniformly distributed position (Barnes's recipe);
  2. plays the recursive debate with challengers that can afford to check a leaves before
     navigating, and compares their measured win rates with (a + 1)/N (the chapter's proposition on a
     flaw that cannot be found);
  3. checks the interval-halving argument "R has no divisor in [2, isqrt(R)]" on small products
     R = p p' of two primes: the only false leaf is the smaller prime factor (the proposition 'winning
     is factoring'; the chapter writes R for the modulus because N counts the leaves).
Writes figures/data/09-efficientdebate-needle*.dat; prints every number quoted in the chapter.
Run from the volume root: python3 figures/scripts/09-efficientdebate-obfuscation.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402


# ---------------------------------------------------------------- 1. Barnes's recipe, literally
def build_and_evaluate(D, F_value, gen):
    """Return the truth values of the 2^D leaves of the tree for root claim F (given truth value).
    Formulas are represented by their truth values only: a node with value P and fresh Q gets the
    children (Q => P) and ((Q => P) => P)."""
    level = [bool(F_value)]
    for _ in range(D):
        nxt = []
        for P in level:
            Q = bool(gen.random() < 0.5)          # a fresh claim, true with probability 1/2
            c1 = (not Q) or P                     # Q => P
            c2 = (not c1) or P                    # (Q => P) => P
            nxt += [c1, c2]
        level = nxt
    return level


def check_recipe():
    gen = pk.rng(93)
    D, trials = 6, 6400
    N = 2 ** D
    positions = np.zeros(N, dtype=np.int64)
    for _ in range(trials):
        leaves = build_and_evaluate(D, False, gen)
        false = [i for i, v in enumerate(leaves) if not v]
        assert len(false) == 1, "a false root must have exactly one false leaf"
        positions[false[0]] += 1
    for _ in range(500):
        assert all(build_and_evaluate(D, True, gen)), "a true root must have only true leaves"
    chi2 = float(np.sum((positions - trials / N) ** 2 / (trials / N)))
    pval = float(stats.chi2.sf(chi2, N - 1))
    print("recipe, D = %d: %d false roots, each with exactly one false leaf; 500 true roots, all leaves true"
          % (D, trials))
    print("  position of the false leaf: chi-square %.1f on %d degrees of freedom, p-value %.3f" % (chi2, N - 1, pval))


# ---------------------------------------------------------------- 2. recursive debate with a budget
def play(N, budget, gen, n_trials):
    """Each trial: the false leaf F is uniform on [0, N); the challenger checks `budget` distinct
    leaves chosen at random (it learns nothing until it hits F); if it found F it navigates to it,
    otherwise it navigates to an unchecked leaf chosen at random. Returns the number of wins."""
    wins = 0
    for _ in range(n_trials):
        F = int(gen.integers(0, N))
        checked = gen.choice(N, size=budget, replace=False) if budget > 0 else np.empty(0, dtype=np.int64)
        if budget > 0 and np.any(checked == F):
            wins += 1
            continue
        if budget >= N:
            continue
        # navigate to a uniformly random unchecked leaf
        mask = np.ones(N, dtype=bool)
        mask[checked] = False
        target = gen.choice(np.flatnonzero(mask))
        wins += int(target == F)
    return wins


def needle_experiment():
    gen = pk.rng(94)
    n_trials = 3000
    for D in (6, 10):
        N = 2 ** D
        budgets = sorted(set([0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]) & set(range(N)))
        rows, notes = [], []
        for a in budgets:
            k = play(N, a, gen, n_trials)
            lo, hi = pk.clopper_pearson(k, n_trials)
            rows.append((a + 1, k / n_trials, k / n_trials - lo, hi - k / n_trials, (a + 1) / N))
            if lo > (a + 1) / N or hi < (a + 1) / N:
                notes.append("  NOTE: the 95%% interval at a = %d, [%.4f, %.4f], misses (a+1)/N = %.4f"
                             % (a, lo, hi, (a + 1) / N))
        pk.write_dat("figures/data/09-efficientdebate-needle-%d.dat" % N, ["a1", "win", "lo", "hi", "theory"], rows)
        print("\nrecursive debate, N = %d leaves, %d trials per budget" % (N, n_trials))
        for r in rows:
            print("  budget a = %4d: win rate %.4f  (a+1)/N = %.4f" % (r[0] - 1, r[1], r[4]))
        for line in notes:
            print(line)
    n_random = sum(1 for N in (64, 1024) for a in (0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023) if a < N - 1)
    print("\n%d points are random (the points with a = N - 1 are certain wins); with 95%% intervals around"
          " exact values, at least one interval misses with probability at most %.2f" % (n_random, 1 - 0.95 ** n_random))
    # the line (a+1)/N is drawn only for a+1 <= N (nan is skipped by pgfplots)
    line = [(x, x / 64 if x <= 64 * (1 + 1e-9) else float("nan"), x / 1024) for x in np.geomspace(1, 1024, 61)]
    pk.write_dat("figures/data/09-efficientdebate-needle-theory.dat", ["a1", "n64", "n1024"], line)


# ---------------------------------------------------------------- 3. the interval-halving argument
def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13):
        if n % p == 0:
            return n == p
    r = math.isqrt(n)
    f = 17
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


def interval_argument(R):
    """Leaves of A''s bisection argument for 'no divisor in [2, isqrt(R)]': singletons d, each the
    claim 'd does not divide R'. Returns the false leaves and the number of leaves."""
    lo, hi = 2, math.isqrt(R)
    return [d for d in range(lo, hi + 1) if R % d == 0], hi - lo + 1


def check_intervals():
    gen = pk.rng(95)
    bits = 12
    primes = [q for q in range(2 ** (bits - 1), 2 ** bits) if is_prime(q)]
    for _ in range(5):
        p, q = sorted(gen.choice(primes, size=2, replace=False))
        p, q = int(p), int(q)
        false, nleaves = interval_argument(p * q)
        assert false == [p]
        print("R = %d = %d * %d: %d leaves, the only false leaf is d = %d; the halving argument has depth %d"
              % (p * q, p, q, nleaves, false[0], math.ceil(math.log2(nleaves))))


def main():
    t0 = time.time()
    check_recipe()
    needle_experiment()
    print()
    check_intervals()
    print("\nruntime %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
