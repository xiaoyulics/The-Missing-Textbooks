"""14-training-monitor.py -- detection of a training run by random chip inspections (Shavit 2023).

Chapter 14 (Proofs of Training) of Verifiable Intelligence. Run from the volume root:

    python3 figures/scripts/14-training-monitor.py

A developer owns C = 1000 chips and runs a training run on C_run = 222 of them for p = 3
monitoring periods of T_m = 10 days (a run of the size of Chinchilla on H100s in 30 days, in
Shavit's accounting). Each chip stores weight snapshots at the times of a Poisson process of rate
f = 0.1 per day. In each period the verifier inspects m distinct chips chosen uniformly at random;
the run is detected if some inspected chip holds a snapshot of the run taken in that period.

Measured detection rates (20,000 simulated runs per m, Clopper-Pearson 95% intervals) are compared
with the exact probability (hypergeometric) and with the bound of Proposition 14.10,
1 - (1 - (C_run/C)(1 - e^{-f T_m}))^{m p}. Also prints Shavit's Table 1 sample counts recomputed.
Writes figures/data/14-training-monitor*.dat.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

OUT = "figures/data"
C, C_RUN, F_RATE, T_M, PERIODS = 1000, 222, 0.1, 10.0, 3
A_SNAP = 1 - math.exp(-F_RATE * T_M)        # a run chip holds a snapshot from a given period


def exact_detect(m):
    """1 - (E[u^K])^p with K hypergeometric: run chips among m distinct inspected chips."""
    u = 1 - A_SNAP
    ks = np.arange(0, m + 1)
    pmf = stats.hypergeom(C, C_RUN, m).pmf(ks)
    miss = float(np.sum(pmf * u ** ks))
    return 1 - miss ** PERIODS


def bound_detect(m):
    q1 = (C_RUN / C) * A_SNAP
    return 1 - (1 - q1) ** (m * PERIODS)


def main():
    os.makedirs(OUT, exist_ok=True)
    gen = pk.rng(1440)
    n = 20_000
    rows = []
    worst_gap = 0.0
    for m in range(1, 13):
        # simulate: run chips among the m inspected, then which of them hold a snapshot, per period
        k = gen.hypergeometric(C_RUN, C - C_RUN, m, size=(n, PERIODS))
        hot = gen.binomial(k, A_SNAP)
        detected = int(np.sum(np.any(hot > 0, axis=1)))
        lo, hi = pk.clopper_pearson(detected, n)
        r = detected / n
        ex, bd = exact_detect(m), bound_detect(m)
        worst_gap = max(worst_gap, ex - bd)
        ok = lo <= ex <= hi
        print(f"m={m:2d}: detected {detected}/{n} = {r:.4f} [{lo:.4f},{hi:.4f}]  exact {ex:.4f} {'ok' if ok else 'OUTSIDE'}  bound {bd:.4f}")
        rows.append([m, r, hi - r, r - lo])
    pk.write_dat(f"{OUT}/14-training-monitor.dat", ["m", "det", "detep", "detem"], rows)
    pk.write_dat(f"{OUT}/14-training-monitor-theory.dat", ["m", "exact", "bound"],
                 [[m, exact_detect(m), bound_detect(m)] for m in range(1, 13)])
    print(f"largest gap exact - bound over m=1..12: {worst_gap:.4f}")
    q1 = (C_RUN / C) * A_SNAP
    m90 = math.ceil(math.log(0.1) / (PERIODS * math.log(1 - q1)))
    print(f"per-inspection probability q = {q1:.4f}; inspections per period for 90%: m = {m90}")

    # Shavit's Table 1 (Chinchilla row): annual samples 365 s / T_m with T = T_m = 30 days, p = 0.9
    H_FLOP, a = 5.76e23, 1e15 * 24 * 3600
    T = 30.0
    for Cc in (1e3, 1e5, 1e7):
        frac = H_FLOP / (a * T * Cc)
        q = frac * (1 - math.exp(-F_RATE * T))
        s_exact = math.log(1 - 0.9) / math.log(1 - q)
        s_approx = math.log(1 / (1 - 0.9)) / q
        print(f"Shavit Table 1, C={Cc:.0e}: chips used H/(aT) = {H_FLOP / (a * T):.1f}; samples/yr exact {365 * s_exact / T:.1f}, "
              f"with log(1-q) ~ -q: {365 * s_approx / T:.1f}")


if __name__ == "__main__":
    main()
