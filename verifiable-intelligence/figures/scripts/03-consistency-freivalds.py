"""Chapter 3 (Consistency): Freivalds's check of a matrix product, run against honest and cheating
programs.

A program claims C = AB for n x n matrices over F_p. The checker draws r uniformly from S^n and
accepts a trial if A(Br) = Cr; a run of t trials accepts if every trial does, and stops at the
first failing trial (as the checker would). Programs:
  honest      C = AB                                                  (accepted with probability 1)
  best        C = AB + one wrong entry: the error has a single nonzero column, which attains the
              bound 1/|S| of the theorem exactly (theory |S|^{-t})
  off-by-one  each inner product skips its last term: C = AB - a b^T with a = A[:,n-1],
              b = B[n-1,:] (a natural bug; accepted only if b.r = 0 mod p, probability ~ 1/p + 2^-n)
Costs: wall-clock time of recomputing AB (float64 BLAS, one thread) against one trial of the check
(three matrix-vector products) on integer matrices small enough for exact float arithmetic.

Writes figures/data/03-consistency-freivalds-rates.dat   (S = {0,1})
       figures/data/03-consistency-freivalds-rates4.dat  (S = {0,1,2,3})
       figures/data/03-consistency-freivalds-costs.dat
Run from the volume root:  python3 figures/scripts/03-consistency-freivalds.py
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread: fair timings, and the laptop is shared

import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk

P = 1_000_003          # a prime; entries < 2^20 keep every product of the check inside int64
N_DIM = 40             # matrix size for the acceptance experiments
RUNS = 100_000         # independent runs of the checker per program
T_MAX = 14             # trials per run (the run stops at the first failing trial)


def passes_before_failure(A, B, C, s_size, t_max, runs, gen):
    """For each run, the number of consecutive trials passed before the first failure (capped at
    t_max). A trial draws r uniform on {0,...,s_size-1}^n and passes if A(Br) = Cr mod P."""
    n = A.shape[0]
    count = np.zeros(runs, dtype=np.int64)
    active = np.arange(runs)
    for _ in range(t_max):
        if active.size == 0:
            break
        R = gen.integers(0, s_size, size=(n, active.size), dtype=np.int64)
        lhs = (A @ ((B @ R) % P)) % P
        rhs = (C @ R) % P
        ok = np.all(lhs == rhs, axis=0)
        count[active[ok]] += 1
        active = active[ok]
    return count


def rates_from_counts(count, t_values):
    rows = []
    for t in t_values:
        k = int(np.sum(count >= t))
        lo, hi = pk.clopper_pearson(k, count.size)
        rows.append((t, k, k / count.size, lo, hi))
    return rows


def main():
    gen = pk.rng(3)
    n = N_DIM
    A = gen.integers(0, P, size=(n, n), dtype=np.int64)
    B = gen.integers(0, P, size=(n, n), dtype=np.int64)
    AB = (A @ B) % P            # entries < 2^20: products < 2^40, sums of 40 terms < 2^46
    i, j = gen.integers(0, n, size=2)
    best = AB.copy()
    best[i, j] = (best[i, j] + 1 + gen.integers(0, P - 1)) % P          # one wrong entry
    offbyone = (AB - np.outer(A[:, n - 1], B[n - 1, :]) % P) % P        # skips the last term
    assert not np.array_equal(best, AB) and not np.array_equal(offbyone, AB)

    ts = np.arange(1, T_MAX + 1)
    out = {}
    for name, C, s_size, seed in (("honest2", AB, 2, 11), ("best2", best, 2, 12),
                                  ("off2", offbyone, 2, 13), ("best4", best, 4, 14)):
        cnt = passes_before_failure(A, B, C, s_size, T_MAX, RUNS, pk.rng(seed))
        out[name] = rates_from_counts(cnt, ts)

    print(f"Freivalds over F_{P}, n = {n}, {RUNS} runs per program, runs stop at the first failure")
    print(" t   honest      best S={0,1}  (2^-t)         best |S|=4  (4^-t)      off-by-one")
    for k in range(T_MAX):
        t = ts[k]
        h, b2, b4, o2 = out["honest2"][k], out["best2"][k], out["best4"][k], out["off2"][k]
        print(f"{t:2d}  {h[2]:.4f}   {b2[2]:.6f} ({2.0**-t:.6f})   {b4[2]:.6f} ({4.0**-t:.6f})   "
              f"{o2[1]} of {RUNS}")

    rows2 = []
    for (t, k, r, lo, hi), h in zip(out["best2"], out["honest2"]):
        if k >= 3:                         # log axis: keep points with at least three acceptances
            rows2.append((t, r, r - lo, hi - r, h[2], 2.0 ** -t))
    pk.write_dat("figures/data/03-consistency-freivalds-rates.dat",
                 ["t", "best", "bestem", "bestep", "honest", "bound"], rows2)
    rows4 = [(t, r, r - lo, hi - r, 4.0 ** -t) for (t, k, r, lo, hi) in out["best4"] if k >= 3]
    pk.write_dat("figures/data/03-consistency-freivalds-rates4.dat",
                 ["t", "best", "bestem", "bestep", "bound"], rows4)

    # costs: recompute AB versus one trial of the check, exact float64 arithmetic
    print("\ncosts (one BLAS thread; best of 5 timings; integer entries 0..9, r in {0,1}^n)")
    print("    n   recompute AB [ms]   one check [ms]   ratio   multiplications n^3 vs 3n^2")
    cost_rows = []
    gcost = pk.rng(5)
    for nn in (250, 500, 1000, 2000):
        A2 = gcost.integers(0, 10, size=(nn, nn)).astype(np.float64)
        B2 = gcost.integers(0, 10, size=(nn, nn)).astype(np.float64)
        C2 = A2 @ B2
        r = gcost.integers(0, 2, size=nn).astype(np.float64)
        tm, tc = [], []
        for _ in range(5):
            t0 = time.perf_counter(); _ = A2 @ B2; tm.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); ok = np.array_equal(A2 @ (B2 @ r), C2 @ r); tc.append(time.perf_counter() - t0)
            assert ok
        tmul, tchk = 1e3 * min(tm), 1e3 * min(tc)
        cost_rows.append((nn, tmul, tchk, tmul / tchk))
        print(f"{nn:5d}   {tmul:14.2f}   {tchk:13.3f}   {tmul / tchk:6.0f}   {nn**3:.2e} vs {3*nn**2:.2e}")
    pk.write_dat("figures/data/03-consistency-freivalds-costs.dat",
                 ["n", "tmul", "tcheck", "ratio"], cost_rows)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nruntime {time.perf_counter() - t0:.1f} s")
