#!/usr/bin/env python3
"""Chapter 10 (rational): the Brier-scored rational proof for #SAT, run against honest, lying and
lazy provers, and against an auditor who tries to tell them apart by their payments.

The protocol (Protocol 10.1): on a CNF formula phi over l variables the prover reports a count y;
the verifier draws one uniformly random assignment r, computes b = phi(r), and pays
    R = 2 - 2 (b - p)^2,   p = y / 2^l,
which is Azar and Micali's shifted Brier score. With q = #phi / 2^l the expected payment is
    E R = 2 - 2 q (1 - q) - 2 (p - q)^2                      (Theorem 10.3 (ii), Theorem 10.5).
(In the code the number of variables is called n.)

Parts (all on one random 3-CNF formula with l = 12 variables and 10 clauses, seed 10):
  A  payments against the report: the protocol is run literally (random assignments, phi evaluated
     clause by clause), 20,000 runs for each of ten reports; the mean payment gets an exact 95%
     interval, the affine image of the Clopper-Pearson interval for the frequency of b = 1.
  B  the m-sampler of Proposition 10.11 (called the k-sampler in the code): it evaluates phi at
     m = 2^j uniform assignments and reports the fraction that satisfy phi. Its loss against the
     honest prover, measured on common coins, is compared with 2 q (1 - q) / m. A run depends on phi
     only through Bernoulli(q) bits, so the script draws the sampler's count from Binomial(m, q) and
     the verifier's bit from Bernoulli(q); it checks this shortcut against literal runs for m <= 16.
  C  the auditor of Proposition 10.13: t runs of the honest prover and of a liar who reports
     #phi + 256; success means that the honest prover's total payment is strictly larger (compared
     exactly, in integers). Paired: both are paid on the same coins; independent: on independent
     coins. 4,000 experiments per t, with Clopper-Pearson intervals, against the exact success
     probabilities (binomial sums); also the runs needed for a lie by one.
  D  costs: the crossover cost per evaluation above which the one-sample prover out-earns the
     honest one, the payment scale that exact honesty needs (Corollary 10.12), and timings of the
     verifier's run and of the honest count by enumeration for l = 12, 16, 20.
  E  linear programs that check Proposition 10.8 and Exercise 10.7 (more samples per run): over all
     proper payment rules on the grid {0, 1/N, ..., 1} with payments in [0, 2] and T samples per run,
     the largest possible smallest loss of a lie by one, compared with Brier's 2 / N^2.

Writes figures/data/10-rational-brier-{payment,payment-honest,curve,sampler,detect,detect-theory}.dat
and prints every number the chapter quotes. Run from the volume root:
    python3 figures/scripts/10-rational-brier.py          (about 3 s on a laptop)
"""
import os
import sys
import time

import numpy as np
from scipy import stats
from scipy.optimize import linprog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

N_VARS, M_CLAUSES = 12, 10
SEED_FORMULA, SEED_A, SEED_B, SEED_C, SEED_D = 10, 1010, 2020, 3030, 4040
RUNS_A = 20000
RUNS_B = 4_000_000
EXPERIMENTS_C = 4000
LIE_C = 256
OUT = "figures/data/10-rational-brier"


# ----------------------------------------------------------------------------- formula and protocol
def random_cnf(n, m, gen):
    """A random 3-CNF formula: m clauses, each with three distinct variables and random signs.
    A literal is a pair (v, s): s = 1 for x_v, s = 0 for not x_v."""
    clauses = []
    for _ in range(m):
        vs = gen.choice(n, size=3, replace=False)
        signs = gen.integers(0, 2, size=3)
        clauses.append(tuple(zip(vs.tolist(), signs.tolist())))
    return clauses


def eval_phi(clauses, X):
    """phi at each row of X (an integer array of shape (T, n) with entries 0/1)."""
    sat = np.ones(X.shape[0], dtype=bool)
    for c in clauses:
        cl = np.zeros(X.shape[0], dtype=bool)
        for v, s in c:
            cl |= X[:, v] == s
        sat &= cl
    return sat


def count_models(clauses, n, chunk=1 << 20):
    """#phi by enumeration of all 2^n assignments, in chunks: the honest prover's work."""
    total = 0
    for start in range(0, 2 ** n, chunk):
        idx = np.arange(start, min(start + chunk, 2 ** n))
        X = (idx[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
        total += int(eval_phi(clauses, X).sum())
    return total


def payment(p, b):
    """The verifier's payment 2 - 2 (b - p)^2 (vectorized in p and b)."""
    return 2.0 - 2.0 * (np.asarray(b, dtype=float) - p) ** 2


def expected_payment(p, q):
    return 2.0 - 2.0 * q * (1.0 - q) - 2.0 * (p - q) ** 2


def verifier_runs(clauses, n, runs, gen):
    """The verifier's coins in `runs` runs: b = phi(r) for uniform r in {0,1}^n."""
    X = gen.integers(0, 2, size=(runs, n))
    return eval_phi(clauses, X)


# ----------------------------------------------------------------------------- part A
def part_a(clauses, count):
    n = N_VARS
    N = 2 ** n
    q = count / N
    reports = sorted(set([j * N // 8 for j in range(9)] + [count]))
    rows = []
    inside = 0
    for i, y in enumerate(reports):
        gen = pk.rng(SEED_A + i)
        b = verifier_runs(clauses, n, RUNS_A, gen)
        p = y / N
        k = int(b.sum())
        lo, hi = pk.clopper_pearson(k, RUNS_A)
        mean = float(payment(p, b).mean())
        # the payment is (2 - 2p^2) + (4p - 2) b: an affine image of the frequency of b = 1
        ends = sorted([2 - 2 * p * p + (4 * p - 2) * lo, 2 - 2 * p * p + (4 * p - 2) * hi])
        exact = expected_payment(p, q)
        inside += int(ends[0] - 1e-12 <= exact <= ends[1] + 1e-12)
        rows.append((y, p, mean, ends[1] - mean, mean - ends[0], exact))
        tag = "honest" if y == count else "report"
        print(f"  A {tag:6s} y={y:5d} p={p:.4f}: mean payment {mean:.4f} "
              f"[{ends[0]:.4f}, {ends[1]:.4f}], exact {exact:.4f}, frequency of b=1 {k / RUNS_A:.4f}")
    pk.write_dat(OUT + "-payment.dat", ["y", "p", "mean", "ep", "em", "exact"],
                 [r for r in rows if r[0] != count])
    pk.write_dat(OUT + "-payment-honest.dat", ["y", "p", "mean", "ep", "em", "exact"],
                 [r for r in rows if r[0] == count])
    grid = np.linspace(0, 1, 201)
    pk.write_dat(OUT + "-curve.dat", ["p", "exact"], [(x, expected_payment(x, q)) for x in grid])
    print(f"  A: {inside} of {len(reports)} intervals contain the exact expected payment")
    print(f"  A: honest maximum 2-2q(1-q) = {expected_payment(q, q):.5f}; "
          f"report 1/2 earns exactly 1.5; lie by one costs 2*4^-n = {2 / N ** 2:.4g}")


# ----------------------------------------------------------------------------- part B
def part_b(clauses, count):
    n = N_VARS
    N = 2 ** n
    q = count / N
    # check the binomial shortcut against literal sampling for small k
    gen = pk.rng(SEED_B - 1)
    for k in (1, 4, 16):
        runs = 20000
        X = gen.integers(0, 2, size=(runs * k, n))
        s_lit = eval_phi(clauses, X).reshape(runs, k).sum(axis=1)
        b_lit = verifier_runs(clauses, n, runs, gen)
        d = payment(q, b_lit) - payment(s_lit / k, b_lit)
        print(f"  B check k={k:2d}: literal mean loss {d.mean():.5f} +- "
              f"{1.96 * d.std() / np.sqrt(runs):.5f} (20,000 runs), theory {2 * q * (1 - q) / k:.5f}")
    rows = []
    gen = pk.rng(SEED_B)
    for j in range(n + 1):
        k = 2 ** j
        tot, tot2 = 0.0, 0.0
        done = 0
        while done < RUNS_B:
            m = min(1_000_000, RUNS_B - done)
            s = gen.binomial(k, q, size=m)
            b = gen.random(m) < q
            d = payment(q, b) - payment(s / k, b)
            tot += float(d.sum())
            tot2 += float((d * d).sum())
            done += m
        mean = tot / RUNS_B
        sd = np.sqrt(max(tot2 / RUNS_B - mean * mean, 0.0))
        half = 1.96 * sd / np.sqrt(RUNS_B)
        theory = 2 * q * (1 - q) / k
        rows.append((k, mean, half, half, theory, 2.0 / N ** 2))
        print(f"  B k={k:5d}: loss {mean:.4e} +- {half:.1e}, theory {theory:.4e}, "
              f"sampler's expected payment {expected_payment(q, q) - theory:.5f}")
    pk.write_dat(OUT + "-sampler.dat", ["k", "loss", "ep", "em", "theory", "onelie"], rows)
    inside = sum(1 for (k, mean, half, _, theory, _) in rows if abs(mean - theory) <= half)
    print(f"  B: {inside} of {len(rows)} intervals contain 2q(1-q)/m; the {N}-sampler's exact loss is "
          f"{(2 * q * (1 - q) / N) / (2.0 / N ** 2):.0f} times that of a lie by one")


# ----------------------------------------------------------------------------- part C
# Payments are compared exactly in integers: N^2 R = 2N^2 - 2y^2 + (4yN - 2N^2) b for a report y.
def scaled(y, N):
    """(constant, slope) of N^2 times the payment for report y, as Python integers."""
    return 2 * N * N - 2 * y * y, 4 * y * N - 2 * N * N


def success_paired(t, c, l, N):
    """P(honest total > liar total) with common coins, for the honest report c and the liar's l.
    Per run the difference of scaled payments is (ch - cl) + (ah - al) b, so the honest prover wins
    iff t (ch - cl) + (ah - al) S > 0, S ~ Bin(t, c/N)."""
    (ch, ah), (cl, al) = scaled(c, N), scaled(l, N)
    K, A = t * (ch - cl), ah - al          # need K + A S > 0
    q = c / N
    if A > 0:                              # S > -K / A
        return float(stats.binom.sf((-K) // A, t, q))
    if A < 0:                              # S < K / (-A)
        return float(stats.binom.cdf(-((-K) // (-A)) - 1 if K % (-A) == 0 else K // (-A), t, q))
    return float(K > 0)


def success_independent(t, c, l, N):
    """P(honest total > liar total) with independent coins, as an exact binomial sum over S1."""
    (ch, ah), (cl, al) = scaled(c, N), scaled(l, N)
    q = c / N
    s1 = np.arange(t + 1, dtype=np.int64)
    w1 = stats.binom.pmf(s1, t, q)
    K = t * (ch - cl) + ah * s1            # need K - al S2 > 0
    if al < 0:                             # S2 > K / al, i.e. S2 > floor(K / al)
        thr = np.floor_divide(K, al)
        p2 = stats.binom.sf(thr, t, q)
    elif al > 0:                           # S2 < K / al
        thr = np.where(K % al == 0, K // al - 1, np.floor_divide(K, al))
        p2 = stats.binom.cdf(thr, t, q)
    else:
        p2 = (K > 0).astype(float)
    return float((w1 * p2).sum())


def first_t(fn, target=0.95, hi=2 ** 40):
    """Smallest t (to within 0.1%) with fn(t) >= target, by doubling and bisection."""
    t = 1
    while fn(t) < target:
        t *= 2
        if t > hi:
            return None
    a, b = t // 2, t
    while b - a > max(1, a // 1000):
        mid = (a + b) // 2
        if fn(mid) >= target:
            b = mid
        else:
            a = mid
    return b


def part_c(count):
    n = N_VARS
    N = 2 ** n
    q = count / N
    lie = count + LIE_C
    pl = lie / N
    delta = pl - q
    (ch, ah), (cl, al) = scaled(count, N), scaled(lie, N)
    gen = pk.rng(SEED_C)
    ts = [2 ** j for j in range(0, 19)]
    rows = []
    for t in ts:
        S = gen.binomial(t, q, size=EXPERIMENTS_C).astype(np.int64)
        win_p = int((t * (ch - cl) + (ah - al) * S > 0).sum())
        S1 = gen.binomial(t, q, size=EXPERIMENTS_C).astype(np.int64)
        S2 = gen.binomial(t, q, size=EXPERIMENTS_C).astype(np.int64)
        win_i = int((t * (ch - cl) + ah * S1 - al * S2 > 0).sum())
        lp, hp = pk.clopper_pearson(win_p, EXPERIMENTS_C)
        li, hi = pk.clopper_pearson(win_i, EXPERIMENTS_C)
        rp, ri = win_p / EXPERIMENTS_C, win_i / EXPERIMENTS_C
        rows.append((t, rp, hp - rp, rp - lp, ri, hi - ri, ri - li))
    pk.write_dat(OUT + "-detect.dat", ["t", "paired", "pep", "pem", "indep", "iep", "iem"], rows)
    trows = []
    exact_at = {}
    # the exact curves at the measured t below 256 (where the lattice of K makes them jagged) and on a
    # finer grid above
    grid = sorted(set([2 ** j for j in range(0, 9)] +
                      [int(round(2 ** (e / 4))) for e in range(32, 4 * 18 + 1)]))
    for t in grid:
        sp = success_paired(t, count, lie, N)
        si = success_independent(t, count, lie, N)
        hb = 1 - np.exp(-t * delta ** 2 / 2)
        trows.append((t, sp, si, hb))
        exact_at[t] = (sp, si)
    pk.write_dat(OUT + "-detect-theory.dat", ["t", "paired", "indep", "hoeff"], trows)
    inside_p = inside_i = 0
    for (t, rp, pep, pem, ri, iep, iem) in rows:
        sp, si = exact_at[t]
        inside_p += int(rp - pem - 1e-12 <= sp <= rp + pep + 1e-12)
        inside_i += int(ri - iem - 1e-12 <= si <= ri + iep + 1e-12)
        hb = 1 - np.exp(-t * delta ** 2 / 2)
        if rp + pep < hb - 1e-12:
            print(f"  C WARNING: paired success at t={t} lies below the Hoeffding bound")
    print(f"  C liar reports #phi+{LIE_C}: p_l = {pl:.4f}, delta = {delta:.4f}")
    print(f"  C {inside_p} of {len(rows)} paired intervals and {inside_i} of {len(rows)} independent "
          f"intervals contain the exact success probability")
    for t in (16, 64, 256, 1024, 4096, 16384):
        sp, si = exact_at[t]
        print(f"  C t={t:6d}: exact success paired {sp:.4f}, independent {si:.4f}")
    tp = first_t(lambda t: success_paired(t, count, lie, N))
    ti = first_t(lambda t: success_independent(t, count, lie, N), hi=2 ** 22)
    th = int(np.ceil(2 * np.log(1 / 0.05) / delta ** 2))
    print(f"  C 95% success: paired after about {tp} runs (Hoeffding guarantees it from {th}), "
          f"independent after about {ti} runs")
    print(f"  C Le Cam: below q(1-q)/(2 delta^2) = {q * (1 - q) / (2 * delta ** 2):.3g} runs every "
          f"auditor errs with probability at least 1/4")
    # a lie by one: the exact paired formula, and the normal approximation for independent coins
    lie1 = count + 1
    d1 = 1 / N
    tp1 = first_t(lambda t: success_paired(t, count, lie1, N), hi=2 ** 34)
    pl1 = lie1 / N
    var_i = 4 * q * (1 - q) * ((1 - 2 * q) ** 2 + (1 - 2 * pl1) ** 2)
    ti1 = (stats.norm.ppf(0.95) * np.sqrt(var_i) / (2 * d1 ** 2)) ** 2
    print(f"  C lie by one: paired 95% after about {tp1:.3g} runs; Le Cam threshold "
          f"{q * (1 - q) / (2 * d1 ** 2):.3g}; independent, normal approximation, about {ti1:.2g} runs")


# ----------------------------------------------------------------------------- part D
def part_d(clauses, count):
    n = N_VARS
    N = 2 ** n
    q = count / N
    cstar = 2 * q * (1 - q) / (N - 1)
    print(f"  D crossover cost per evaluation c* = 2q(1-q)/(2^n-1) = {cstar:.3g} (payments in [0,2])")
    for c in (1e-6, 1e-5, 1e-4, 1e-3):
        honest = expected_payment(q, q) - c * N
        ks = 2 ** np.arange(0, n + 1)
        nets = expected_payment(q, q) - 2 * q * (1 - q) / ks - c * ks
        best = int(ks[np.argmax(nets)])
        print(f"  D c={c:.0e}: honest net {honest:.4f}; best sampler k={best} net {nets.max():.4f}; "
              f"one-sample net {nets[0]:.4f}")
    scale = 4 ** (n - 1) / (2 * q * (1 - q))
    print(f"  D exact honesty against every k-sampler needs payment scale M >= c * {scale:.4g}")
    # timings: one verifier run, and the honest count by enumeration
    gen = pk.rng(SEED_D)
    reps = 20000
    t0 = time.perf_counter()
    for _ in range(reps):
        x = gen.integers(0, 2, size=(1, n))
        b = eval_phi(clauses, x)[0]
        _ = 2.0 - 2.0 * (float(b) - count / N) ** 2
    tv = (time.perf_counter() - t0) / reps
    print(f"  D one verifier run (draw r, evaluate phi, pay): {tv * 1e6:.1f} microseconds")
    for nn in (12, 16, 20):
        g = pk.rng(SEED_D + nn)
        cl = random_cnf(nn, int(round(M_CLAUSES * nn / N_VARS)), g)
        t0 = time.perf_counter()
        cnt = count_models(cl, nn)
        th = time.perf_counter() - t0
        print(f"  D honest count by enumeration, n={nn}, m={len(cl)}: #phi={cnt}, {th:.3f} s")


# ----------------------------------------------------------------------------- part E
def lp_best_min_gap(n, T, B=2.0):
    """Largest t such that some payment rule S(y, s) in [0, B], s ~ Bin(T, y/N) the number of
    satisfying samples, is proper on the grid and makes every lie by one cost at least t."""
    N = 2 ** n
    P = np.arange(N + 1) / N
    nS = (N + 1) * (T + 1)
    nv = nS + 1

    def es(i, qq):
        v = np.zeros(nv)
        v[i * (T + 1):(i + 1) * (T + 1)] = stats.binom.pmf(np.arange(T + 1), T, qq)
        return v
    A, rhs = [], []
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                A.append(es(j, P[i]) - es(i, P[i]))
                rhs.append(0.0)
    for i in range(N):
        for tr, rep in ((i, i + 1), (i + 1, i)):
            v = es(rep, P[tr]) - es(tr, P[tr])
            v[-1] = 1.0
            A.append(v)
            rhs.append(0.0)
    c = np.zeros(nv)
    c[-1] = -1.0
    res = linprog(c, A_ub=np.array(A), b_ub=np.array(rhs),
                  bounds=[(0, B)] * nS + [(None, None)], method="highs")
    return -res.fun


def part_e():
    for n in range(1, 6):
        g = lp_best_min_gap(n, 1)
        print(f"  E T=1, n={n}: best smallest loss of a lie by one {g:.6g}; "
              f"Brier 2/N^2 = {2 / 4 ** n:.6g}")
    n = 4
    for T in (1, 2, 4, 8):
        g = lp_best_min_gap(n, T)
        print(f"  E n={n}, T={T} samples per run: best smallest loss {g:.5g} = "
              f"{g / (2 / 4 ** n):.3f} x Brier's; bound B T^2/N^2 = {2 * T * T / 4 ** n:.4g}")


def main():
    t_start = time.perf_counter()
    clauses = random_cnf(N_VARS, M_CLAUSES, pk.rng(SEED_FORMULA))
    count = count_models(clauses, N_VARS)
    q = count / 2 ** N_VARS
    print(f"formula: n={N_VARS}, m={M_CLAUSES}, #phi={count}, q={q:.5f}, q(1-q)={q * (1 - q):.5f}")
    print("A: payment against the report")
    part_a(clauses, count)
    print("B: samplers")
    part_b(clauses, count)
    print("C: telling a liar from the honest prover")
    part_c(count)
    print("D: costs")
    part_d(clauses, count)
    print("E: linear programs over payment rules")
    part_e()
    print(f"total time {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
