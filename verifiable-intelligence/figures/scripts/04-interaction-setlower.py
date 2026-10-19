"""04-interaction-setlower.py -- Chapter 4 (Interaction): the Goldwasser-Sipser set lower bound.

Protocol 4.3: the verifier draws h(x) = Ax + c over F_2 (A uniform k x m, c uniform in {0,1}^k)
and y uniform in {0,1}^k, all public; the prover must return some x in S with h(x) = y, with a
certificate that x is in S.  The best prover is accepted exactly when y lies in h(S).  For sets of
controlled size the script measures this acceptance rate against the bounds of the theorem,
     |S|/2^k - |S|(|S|-1)/2^(2k+1)  <=  Pr[accept]  <=  |S|/2^k ,
plotted against x = |S|/2^k (the lower bound is then at least x - x^2/2):
  - random subsets of {0,1}^20 and linear subspaces of {0,1}^20, with k = 10;
    for a subspace of dimension d the exact value is E[2^rank(M)]/2^k, M a uniform k x d matrix;
  - the set S = {(H, tau): H isomorphic to G_0 or G_1, tau an automorphism of H} of the proof
    that graph non-isomorphism is in AM, for one non-isomorphic and one isomorphic pair of graphs
    on 6 vertices (|S| = 2*6! = 1440 or 6! = 720; m = 25 bits; K = 1440, k = 12).
It also prints how many parallel runs the threshold rule of Corollary 4.11 needs.

Run from the volume root:  python3 figures/scripts/04-interaction-setlower.py
Writes figures/data/04-interaction-setlower*.dat.  Deterministic; a few seconds on a laptop CPU.
"""
import itertools
import math
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

OUT = "figures/data"


def accept_rate(xs, m, k, trials, gen, chunk=200):
    """Run the protocol `trials` times on the set xs (uint64 array of m-bit strings) with a fresh
    (A, c, y) each time; the prover answers whenever y has a preimage in S.  Returns the count."""
    xs = np.asarray(xs, dtype=np.uint64)
    hits = 0
    done = 0
    while done < trials:
        T = min(chunk, trials - done)
        A = gen.integers(0, 2 ** m, size=(T, k), dtype=np.uint64)  # rows of A as bit masks
        c = gen.integers(0, 2 ** k, size=T, dtype=np.uint64)
        y = gen.integers(0, 2 ** k, size=T, dtype=np.uint64)
        hx = np.zeros((T, len(xs)), dtype=np.uint64)
        for r in range(k):
            par = np.bitwise_count(A[:, r, None] & xs[None, :]) & np.uint8(1)
            hx |= par.astype(np.uint64) << np.uint64(r)
        hx ^= c[:, None]
        hits += int(np.any(hx == y[:, None], axis=1).sum())
        done += T
    return hits


def subspace_exact(d, k):
    """E[2^rank(M)] / 2^k for M uniform in F_2^{k x d}: add columns one at a time."""
    p = np.zeros(k + 1)
    p[0] = 1.0
    for _ in range(d):
        q = np.zeros(k + 1)
        for r in range(k + 1):
            if p[r] == 0:
                continue
            stay = 2.0 ** (r - k)
            q[r] += p[r] * stay
            if r < k:
                q[r + 1] += p[r] * (1 - stay)
        p = q
    return float(sum(p[r] * 2.0 ** r for r in range(k + 1)) / 2.0 ** k)


def graph_set(n, A0, A1):
    """The set S of the proof that GNI is in AM, encoded as integers: code(H) | index(tau) << e."""
    iu = np.triu_indices(n, 1)
    w = np.int64(1) << np.arange(len(iu[0]), dtype=np.int64)
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)
    index = {tuple(p): i for i, p in enumerate(perms)}
    shift = len(iu[0])

    def codes_of(stack):
        return stack[..., iu[0], iu[1]].astype(np.int64) @ w

    def all_relabel(A):
        return A[perms[:, :, None], perms[:, None, :]]

    S = set()
    for A in (A0, A1):
        rel = all_relabel(A)
        cls_codes, first = np.unique(codes_of(rel), return_index=True)
        for code, f in zip(cls_codes, first):
            H = rel[f]
            auts = np.nonzero(codes_of(all_relabel(H)) == code)[0]
            for a in auts:
                S.add(int(code) | (index[tuple(perms[a])] << shift))
    m = shift + math.ceil(math.log2(len(perms)))
    return np.array(sorted(S), dtype=np.uint64), m


def random_graph(n, gen):
    U = np.triu(gen.random((n, n)) < 0.5, 1)
    return U | U.T


def runs_ok(t, q_yes, q_no, theta, err=1 / 3):
    """Whether Bin(t, q_yes) < theta t and Bin(t, q_no) >= theta t each have probability at most
    err (the threshold rule: accept iff at least theta t runs accept)."""
    need = math.ceil(theta * t - 1e-12)
    miss = stats.binom.cdf(need - 1, t, q_yes)
    fool = stats.binom.sf(need - 1, t, q_no)
    return miss <= err and fool <= err


def runs_needed(q_yes, q_no, theta, err=1 / 3, tmax=2000):
    """Smallest t that works, and the smallest t0 such that every t in [t0, tmax] works (the
    errors are not monotone in t, because the threshold ceil(theta t) moves in integer steps)."""
    good = [runs_ok(t, q_yes, q_no, theta, err) for t in range(1, tmax + 1)]
    if not any(good):
        return None, None
    first = good.index(True) + 1
    from_on = next(t for t in range(tmax, 0, -1) if not good[t - 1]) + 1 if not all(good) else 1
    return first, from_on


def main():
    t0 = time.perf_counter()
    gen = pk.rng(51)
    m, k, trials = 20, 10, 4000
    rows_rand, rows_sub = [], []
    print("random sets and subspaces of {0,1}^%d, k = %d, %d trials each" % (m, k, trials))
    for N in [16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]:
        xs = gen.choice(2 ** m, size=N, replace=False).astype(np.uint64)
        hits = accept_rate(xs, m, k, trials, gen)
        x = N / 2 ** k
        lo, hi = pk.clopper_pearson(hits, trials)
        rate = hits / trials
        bonf = N / 2 ** k - N * (N - 1) / 2 ** (2 * k + 1)
        rows_rand.append([x, rate, hi - rate, rate - lo, bonf, min(1.0, x), 1 - (1 - 2.0 ** -k) ** N])
        print("   random |S|=%4d x=%.4f  accept %.4f [%.4f,%.4f]  lower %.4f  upper %.4f  random fn %.4f"
              % (N, x, rate, lo, hi, bonf, min(1.0, x), 1 - (1 - 2.0 ** -k) ** N))
    for d in range(4, 11):
        basis = []
        while len(basis) < d:  # d linearly independent vectors
            v = int(gen.integers(1, 2 ** m))
            span = {0}
            for b in basis:
                span |= {s ^ b for s in span}
            if v not in span:
                basis.append(v)
        span = [0]
        for b in basis:
            span = span + [s ^ b for s in span]
        xs = np.array(span, dtype=np.uint64)
        hits = accept_rate(xs, m, k, trials, gen)
        x = 2 ** d / 2 ** k
        lo, hi = pk.clopper_pearson(hits, trials)
        rate = hits / trials
        ex = subspace_exact(d, k)
        rows_sub.append([x, rate, hi - rate, rate - lo, ex])
        print("   subspace d=%2d x=%.4f  accept %.4f [%.4f,%.4f]  exact %.4f  lower %.4f"
              % (d, x, rate, lo, hi, ex, 2 ** d / 2 ** k - 2 ** d * (2 ** d - 1) / 2 ** (2 * k + 1)))
    pk.write_dat(os.path.join(OUT, "04-interaction-setlower-random.dat"),
                 ["x", "acc", "accp", "accm", "bonf", "upper", "randfn"], rows_rand)
    pk.write_dat(os.path.join(OUT, "04-interaction-setlower-subspace.dat"),
                 ["x", "acc", "accp", "accm", "exact"], rows_sub)
    # theory curves on a dense grid
    dense = []
    for x in np.linspace(0, 1, 101):
        dense.append([x, x - x * x / 2, min(1.0, x), 1 - math.exp(-x)])
    pk.write_dat(os.path.join(OUT, "04-interaction-setlower-theory.dat"), ["x", "lower", "upper", "randfn"], dense)
    subex = [[2 ** d / 2 ** k, subspace_exact(d, k)] for d in range(0, k + 1)]
    pk.write_dat(os.path.join(OUT, "04-interaction-setlower-subexact.dat"), ["x", "exact"], subex)

    # the set of the proof that GNI is in AM, n = 6
    n = 6
    ggen = pk.rng(52)
    while True:  # a non-isomorphic pair with equal edge counts
        A0, A1 = random_graph(n, ggen), random_graph(n, ggen)
        if A0.sum() != A1.sum():
            continue
        S_non, mm = graph_set(n, A0, A1)
        if len(S_non) == 2 * math.factorial(n):
            break
    q = ggen.permutation(n)
    S_iso, _ = graph_set(n, A0, A0[np.ix_(q, q)])
    K = 2 * math.factorial(n)
    kk = next(j for j in range(1, 40) if 2 ** (j - 2) <= K <= 2 ** (j - 1))
    p = K / 2 ** kk
    gtrials = 4000
    rows_g = []
    rates = {}
    for name, S in (("non-isomorphic", S_non), ("isomorphic", S_iso)):
        hits = accept_rate(S, mm, kk, gtrials, gen)
        lo, hi = pk.clopper_pearson(hits, gtrials)
        rate = hits / gtrials
        x = len(S) / 2 ** kk
        rates[name] = rate
        rows_g.append([x, rate, hi - rate, rate - lo])
        print("GNI set (n=%d, m=%d bits, K=%d, k=%d, p=K/2^k=%.4f): %s |S|=%d x=%.4f accept %.4f [%.4f,%.4f]"
              % (n, mm, K, kk, p, name, len(S), x, rate, lo, hi))
    pk.write_dat(os.path.join(OUT, "04-interaction-setlower-gni.dat"), ["x", "acc", "accp", "accm"], rows_g)
    print("   theorem: honest >= 3p/4 = %.4f (Bonferroni at |S|=K: %.4f); cheating <= p/2 = %.4f"
          % (0.75 * p, p - K * (K - 1) / 2 ** (2 * kk + 1), p / 2))
    theta = 5 * p / 8
    hoeff = math.ceil(math.log(3) / (2 * (p / 8) ** 2))
    hoeff_worst = math.ceil(math.log(3) / (2 * (1 / 32) ** 2))
    print("   threshold 5p/8 = %.4f; Hoeffding: t >= ln 3 / (2 (p/8)^2) = %d runs (p=%.4f); %d for p=1/4"
          % (theta, hoeff, p, hoeff_worst))
    print("   exact binomial with the worst rates the theorem allows (3p/4, p/2): t = %s works; every t >= %s works"
          % runs_needed(0.75 * p, 0.5 * p, theta))
    print("   exact binomial with the measured rates (%.4f, %.4f): t = %s works; every t >= %s works"
          % (rates["non-isomorphic"], rates["isomorphic"], *runs_needed(rates["non-isomorphic"], rates["isomorphic"], theta)))
    print("   verifier's random bits per run: k*m + 2k = %d (A, c, y)" % (kk * mm + 2 * kk))
    print("total time %.1f s" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
