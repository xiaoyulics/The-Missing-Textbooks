#!/usr/bin/env python3
"""Chapter 5 (sumcheck): what the verifier and the honest prover spend in the counting protocol.

Runs the sumcheck protocol for #SAT (Protocol 5.1 on the arithmetization of a random 3-CNF with
l variables and m = 2l clauses, in which every variable occurs in exactly six clauses, so that
every d_j = 6) over F_p with p = 2^31 - 1, for l = 9, 12, 15, 18, 21, and records:
  rounds l; field elements sent by the prover, sum_j (d_j + 1); the honest prover's evaluations of
  the polynomial g, sum_j (d_j + 1) 2^(l-j); the prover's wall-clock time; the verifier's field
  operations (additions, subtractions and multiplications, counted by an instrumented verifier that
  evaluates each message at r_j by Lagrange interpolation with prefix and suffix products, and then
  evaluates g = P_phi at r once); the verifier's wall-clock time (mean of 200 repetitions).
The honest prover is the vectorized one of 05-sumcheck-rates.py; every run must be accepted.

Writes figures/data/05-sumcheck-costs.dat. Run from the volume root (about 2 s on a laptop):
    python3 figures/scripts/05-sumcheck-costs.py
"""
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import protokit as pk  # noqa: E402

_spec = importlib.util.spec_from_file_location("rates", os.path.join(HERE, "05-sumcheck-rates.py"))
rates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rates)

P = pk.PRIME
SIZES = [9, 12, 15, 18, 21]
OCC = 6                            # occurrences of each variable
OUT = "figures/data/05-sumcheck-costs.dat"


class Counter:
    def __init__(self):
        self.ops = 0


def lagrange_at(vals, r, p, inv_den, cnt):
    """Value at r of the polynomial with values vals at 0..d, with O(d) field operations given the
    precomputed inverse denominators: prefix and suffix products of (r - k)."""
    d = len(vals) - 1
    diffs = [(r - k) % p for k in range(d + 1)]
    cnt.ops += d + 1
    pre = [1] * (d + 2)
    for k in range(d + 1):
        pre[k + 1] = pre[k] * diffs[k] % p
    suf = [1] * (d + 2)
    for k in range(d, -1, -1):
        suf[k] = suf[k + 1] * diffs[k] % p
    cnt.ops += 2 * (d + 1)
    tot = 0
    for i in range(d + 1):
        tot = (tot + vals[i] * (pre[i] * suf[i + 1] % p) % p * inv_den[i]) % p
    cnt.ops += 4 * (d + 1)
    return tot


def verify(messages, r, C, clauses, deg, p, inv_dens, cnt):
    """The verifier of Protocol 5.1 on one transcript: returns accept/reject and counts operations."""
    v = C
    for j, vals in enumerate(messages):
        if len(vals) != deg[j] + 1:
            return False
        cnt.ops += 1
        if (vals[0] + vals[1]) % p != v:
            return False
        v = lagrange_at(vals, r[j], p, inv_dens[j], cnt)
    # one evaluation of g = P_phi at r: per literal one subtraction or none, per clause
    # two multiplications, one subtraction and one multiplication into the product
    val = 1
    for c in clauses:
        prod = 1
        for (var, s) in c:
            prod = prod * ((1 - r[var]) % p if s == 1 else r[var]) % p
        val = val * ((1 - prod) % p) % p
        cnt.ops += sum(1 for (_, s) in c if s == 1) + 2 + 1 + 1
    return val == v


def regular_cnf(l, occ, gen):
    """A random 3-CNF with m = occ*l/3 clauses on distinct variables in which every variable
    occurs in exactly occ clauses (random signs)."""
    while True:
        slots = np.repeat(np.arange(l), occ)
        gen.shuffle(slots)
        triples = slots.reshape(-1, 3)
        if all(len(set(t.tolist())) == 3 for t in triples):
            signs = gen.integers(0, 2, size=triples.shape)
            clauses = [tuple(zip(t.tolist(), sg.tolist())) for t, sg in zip(triples, signs)]
            return clauses, [occ] * l


def main():
    t_all = time.time()
    rows = []
    for l in SIZES:
        m = 2 * l
        clauses, deg = regular_cnf(l, OCC, pk.rng(50 + l))
        gen = pk.rng(500 + l)
        r = gen.integers(0, P, size=(1, l)).astype(np.int64)
        # the claimed value C: the true sum, computed by the honest prover's first message
        t0 = time.time()
        messages = []
        for j in range(l):
            S, _ = rates.honest_round(clauses, l, j, deg[j], r[:, :j], P)
            messages.append([int(x) for x in S[0]])
        t_prover = time.time() - t0
        C = (messages[0][0] + messages[0][1]) % P
        inv_dens = [rates.Lagrange(d, P).inv_den for d in deg]
        cnt = Counter()
        ok = verify(messages, [int(x) for x in r[0]], C, clauses, deg, P, inv_dens, cnt)
        assert ok, "the honest prover must be accepted"
        reps = 200
        t0 = time.time()
        for _ in range(reps):
            verify(messages, [int(x) for x in r[0]], C, clauses, deg, P, inv_dens, Counter())
        t_verifier = (time.time() - t0) / reps
        evals = sum((deg[j] + 1) * 2 ** (l - j - 1) for j in range(l))
        comm = sum(d + 1 for d in deg)
        rows.append([l, m, sum(deg), comm, evals, t_prover, cnt.ops, t_verifier * 1e3])
        print(f"l={l:2d} m={m:2d} sum d_j={sum(deg):3d}  prover: {evals:9d} evaluations of g, "
              f"{t_prover:7.3f} s   verifier: {cnt.ops:5d} field ops, {t_verifier*1e3:6.3f} ms   "
              f"prover sends {comm} field elements; count mod p = {C}")
    pk.write_dat(OUT, ["l", "m", "sumd", "comm", "evals", "tprover", "vops", "tverifierms"], rows)
    for a, b in zip(rows, rows[1:]):
        print(f"l {a[0]}->{b[0]}: prover time x{b[5]/a[5]:.1f} (2^{b[0]-a[0]} = {2**(b[0]-a[0])}), "
              f"evaluations x{b[4]/a[4]:.1f}, verifier ops x{b[6]/a[6]:.2f}")
    print(f"total time {time.time()-t_all:.1f} s")


if __name__ == "__main__":
    main()
