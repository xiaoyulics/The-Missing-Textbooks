"""01-certificates-lazy.py -- Chapter 1 (Certificates): a checker that reads only k clauses
(the chapter writes q for k, the number of queries).

The full certificate check reads every clause of the formula and accepts an assignment only if it
satisfies all of them. A lazy checker reads k clauses chosen uniformly at random without
replacement and accepts if the assignment satisfies those. We run it against
  - the honest prover on the satisfiable PHP^n_n (a perfect matching; PHP^8_8 has 232 clauses, so
    for k > 232 its checker reads all of them);
  - the best cheating prover on the unsatisfiable PHP^{n+1}_n: pigeons 1..n in holes 1..n and
    pigeon n+1 in no hole, which falsifies exactly one clause;
  - a careless cheating prover on PHP^{n+1}_n: every pigeon in a uniformly random hole, drawn
    afresh in each trial; it falsifies one hole clause per pair of pigeons that share a hole.
Theory: an assignment that falsifies u of the M clauses is accepted with probability
C(M-u, k)/C(M, k); every assignment of an unsatisfiable formula has u >= 1, so every cheating
prover is accepted with probability at most 1 - k/M (exact for the best one). For the careless
prover the distribution of u is computed exactly from the occupancy numbers of n+1 balls in n
bins.

Writes figures/data/01-certificates-lazy.dat and prints the numbers quoted in the chapter.
Run from the volume root:  python3 figures/scripts/01-certificates-lazy.py
"""
import os
import sys
import math
from fractions import Fraction
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

N_HOLES = 8
TRIALS = 2000
SEED = 11
KS = [1, 2, 5, 10, 20, 40, 60, 80, 100, 130, 160, 200, 240, 280, 297]


def php(m, n):
    v = lambda i, j: i * n + j + 1
    cls = [tuple(v(i, j) for j in range(n)) for i in range(m)]
    for j in range(n):
        for a, b in combinations(range(m), 2):
            cls.append((-v(a, j), -v(b, j)))
    return m * n, cls


def satisfied(c, a):
    return any(a[abs(l)] == (l > 0) for l in c)


def lazy_check(clauses, a, k, gen):
    """Read k clauses chosen uniformly without replacement; accept iff a satisfies them."""
    idx = gen.choice(len(clauses), size=k, replace=False)
    return all(satisfied(clauses[i], a) for i in idx)


def accept_prob(M, u, k):
    """Probability that k clauses drawn without replacement avoid all u falsified ones."""
    return math.comb(M - u, k) / math.comb(M, k) if k <= M - u else 0.0


def collision_distribution(balls, bins):
    """Exact distribution of the number of pairs of balls sharing a bin, for a uniformly random
    function from balls to bins, as {u: probability}."""
    dist = {}

    def rec(remaining, bins_left, ways, u):
        # assign the remaining balls to the remaining bins, one bin at a time
        if bins_left == 0:
            if remaining == 0:
                dist[u] = dist.get(u, 0) + ways
            return
        for load in range(remaining + 1):
            rec(remaining - load, bins_left - 1, ways * math.comb(remaining, load), u + load * (load - 1) // 2)
    rec(balls, bins, 1, 0)
    total = bins ** balls
    return {u: Fraction(w, total) for u, w in dist.items()}


def main():
    n = N_HOLES
    N, F = php(n + 1, n)
    Nh, Fh = php(n, n)
    M, Mh = len(F), len(Fh)
    gen = pk.rng(SEED)
    honest = [None] + [False] * Nh
    for i in range(n):
        honest[i * n + i + 1] = True
    assert all(satisfied(c, honest) for c in Fh)
    best = [None] + [False] * N
    for i in range(n):
        best[i * n + i + 1] = True
    u_best = sum(not satisfied(c, best) for c in F)
    assert u_best == 1

    def careless(g):
        a = [None] + [False] * N
        for i in range(n + 1):
            a[i * n + int(g.integers(0, n)) + 1] = True
        return a

    dist = collision_distribution(n + 1, n)
    assert sum(dist.values()) == 1
    mean_u = float(sum(u * p for u, p in dist.items()))
    print(f"PHP^{n+1}_{n}: N = {N} variables, M = {M} clauses; PHP^{n}_{n}: M = {Mh} clauses")
    print(f"best cheater falsifies u = {u_best} clause; careless cheater: E[u] = {mean_u:.4f}, "
          f"P[u = 1] = {float(dist.get(1, 0)):.4f}")
    rows = []
    cover_best = cover_care = care_below = 0
    print("   k  honest(lo,hi)  best: rate (lo,hi) theory  careless: rate (lo,hi) theory")
    for k in KS:
        kh = min(k, Mh)
        h = pk.acceptance(lambda g: lazy_check(Fh, honest, kh, g), TRIALS, seed=SEED + k)
        b = pk.acceptance(lambda g: lazy_check(F, best, k, g), TRIALS, seed=2 * SEED + k)
        c = pk.acceptance(lambda g: lazy_check(F, careless(g), k, g), TRIALS, seed=3 * SEED + k)
        tb = accept_prob(M, 1, k)
        tc = float(sum(p * Fraction(accept_prob(M, u, k)) for u, p in dist.items()))
        for rate, lo, hi in (b, c):               # no measured rate may exceed the proved bound
            assert lo <= tb + 1e-12
        print(f"{k:4d}  {h[0]:.3f} ({h[1]:.3f},{h[2]:.3f})  {b[0]:.4f} ({b[1]:.4f},{b[2]:.4f}) {tb:.4f}"
              f"  {c[0]:.4f} ({c[1]:.4f},{c[2]:.4f}) {tc:.4f}")
        rows.append((k, h[0], h[0] - h[1], h[2] - h[0], b[0], b[0] - b[1], b[2] - b[0], tb,
                     c[0], c[0] - c[1], c[2] - c[0], tc))
        cover_best += b[1] <= tb <= b[2]
        cover_care += c[1] <= tc <= c[2]
        care_below += c[0] <= tb
    print(f"best cheater: interval contains 1 - k/M at {cover_best} of {len(KS)} values of k; careless: "
          f"interval contains the exact curve at {cover_care} of {len(KS)}, rate <= 1 - k/M at {care_below}")
    k_half = next(k for k in range(1, M + 1) if accept_prob(M, 1, k) <= 0.5)
    print(f"k needed to catch the best cheater with probability at least 1/2: {k_half} of {M}")
    print(f"full check (k = M = {M}): best cheater accepted with probability {accept_prob(M, 1, M)}")
    # a dense curve for the theory
    curve = [(k, accept_prob(M, 1, k),
              float(sum(p * Fraction(accept_prob(M, u, k)) for u, p in dist.items())))
             for k in range(0, M + 1)]
    pk.write_dat("figures/data/01-certificates-lazy.dat",
                 ["k", "honest", "honestlo", "honesthi", "best", "bestlo", "besthi", "bestth",
                  "care", "carelo", "carehi", "careth"], rows)
    pk.write_dat("figures/data/01-certificates-lazy-theory.dat", ["k", "bound", "careless"], curve)


if __name__ == "__main__":
    main()
