#!/usr/bin/env python3
"""Chapter 5 (sumcheck): the counting protocol against honest and lying provers.

Runs the sumcheck protocol (Protocol 5.1) on the arithmetization of a random 3-CNF formula with
l = 10 variables and m = 20 clauses, over F_p for ten primes p, and measures how often the
verifier accepts. Four provers face the same verifier coins in every trial:

  honest          claims the true count K and sends the true polynomials tau_j (s_j below);
  root-planting   claims K + 1; while its current claim is false it sends s_j + c e_j, where e_j has
                  d_j distinct roots (0 and d_j - 1 random points of F_p minus {0, 1}) and c makes the
                  message pass the verifier's check; once its claim is true it plays honestly.
                  This is the optimal liar of Proposition 5.5: exact acceptance 1 - prod(1 - d_j/p);
  careless        claims K + 1; while its claim is false it sends s_j + h_j with h_j uniform among the
                  polynomials of degree <= d_j that pass the check (every d_j >= 2 here):
                  exact acceptance 1 - (1 - 1/p)^l (Exercise 5.2);
  constant-shift  claims K + 1 and adds half the discrepancy to every value: never accepted.

The verifier is the one of Protocol 5.1 with the degree bounds d_j = number of clauses that contain
x_j; it receives each g_j as its values at 0, 1, ..., d_j. Its coins r_1..r_l are drawn before the
run, which is equivalent because they are independent of the messages and the provers below only
ever see r_1..r_{j-1} when they compute g_j.

Writes figures/data/05-sumcheck-rates.dat (measured rates with Clopper-Pearson 95% intervals) and
figures/data/05-sumcheck-rates-theory.dat (the exact curves and the bound on a dense grid of p).
Run from the volume root:  python3 figures/scripts/05-sumcheck-rates.py   (about 40 s on a laptop)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

L, M = 10, 20                      # variables and clauses of the formula
SEED_FORMULA, SEED_COINS, SEED_LIARS = 5, 505, 5005
PRIMES = [67, 103, 163, 251, 419, 643, 1031, 1619, 2579, 4091]   # all 3 mod 4, as in the TQBF run
TRIALS = 20000                     # per prime
BATCH = 500
OUT = "figures/data/05-sumcheck-rates.dat"
OUT_THEORY = "figures/data/05-sumcheck-rates-theory.dat"


# ----------------------------------------------------------------------------- the formula
def random_cnf(l, m, gen):
    """A random 3-CNF: each clause has three distinct variables with random signs. Resample until
    every variable occurs in at least two clauses, so that every d_j >= 2."""
    while True:
        clauses = []
        for _ in range(m):
            vs = gen.choice(l, size=3, replace=False)
            signs = gen.integers(0, 2, size=3)          # 1: the literal x_v, 0: the literal not x_v
            clauses.append(tuple(zip(vs.tolist(), signs.tolist())))
        deg = [sum(1 for c in clauses for (v, _) in c if v == j) for j in range(l)]
        if min(deg) >= 2:
            return clauses, deg


def count_models(clauses, l):
    xs = (np.arange(2 ** l)[:, None] >> (l - 1 - np.arange(l))[None, :]) & 1
    sat = np.ones(2 ** l, dtype=bool)
    for c in clauses:
        cl = np.zeros(2 ** l, dtype=bool)
        for v, s in c:
            cl |= xs[:, v] == s
        sat &= cl
    return int(sat.sum())


def one_minus_literal(x, s, p):
    """1 - lambda for the literal x (s = 1) or not x (s = 0), elementwise mod p."""
    return (1 - x) % p if s == 1 else x % p


def p_phi(clauses, pts, p):
    """The arithmetization P_phi at the points pts (T, l): a product of clause polynomials."""
    out = np.ones(pts.shape[0], dtype=np.int64)
    for c in clauses:
        prod = np.ones(pts.shape[0], dtype=np.int64)
        for v, s in c:
            prod = prod * one_minus_literal(pts[:, v], s, p) % p
        out = out * ((1 - prod) % p) % p
    return out


# ----------------------------------------------------------------------------- the honest prover
def honest_round(clauses, l, j, d, rpre, p):
    """Values s_j(t) for t = 0..d of the honest round-j polynomial
    s_j(X) = sum over b in {0,1}^(l-j-1) of P_phi(r_0..r_{j-1}, X, b), for each trial (0-indexed j).
    rpre has shape (T, j). Returns a (T, d+1) array and the number of field multiplications."""
    T, D, B = rpre.shape[0], d + 1, 2 ** (l - j - 1)
    t = np.arange(D, dtype=np.int64)
    bidx = np.arange(B, dtype=np.int64)

    def val(v):
        if v < j:
            return rpre[:, v].reshape(T, 1, 1)
        if v == j:
            return t.reshape(1, D, 1)
        return ((bidx >> (l - 1 - v)) & 1).reshape(1, 1, B)

    mults = 0
    rest = np.ones((T, 1, B), dtype=np.int64)        # clauses without x_j
    with_j = []
    for c in clauses:
        prod = None
        for v, s in c:
            f = one_minus_literal(val(v), s, p)
            prod = f if prod is None else prod * f % p
        q = (1 - prod) % p
        if any(v == j for v, _ in c):
            with_j.append(q)
        else:
            rest = rest * q % p
            mults += rest.size
    acc = np.broadcast_to(rest, (T, D, B))
    for q in with_j:
        acc = acc * q % p
        mults += acc.size
    return acc.sum(axis=2) % p, mults


# ----------------------------------------------------------------------------- the verifier
class Lagrange:
    """Evaluate a polynomial of degree <= d given by its values at 0..d at points r (vectorized)."""

    def __init__(self, d, p):
        self.d, self.p = d, p
        self.inv_den = []
        for i in range(d + 1):
            den = 1
            for k in range(d + 1):
                if k != i:
                    den = den * (i - k) % p
            self.inv_den.append(pow(den % p, p - 2, p))

    def __call__(self, vals, r):
        d, p = self.d, self.p
        out = np.zeros(r.shape[0], dtype=np.int64)
        for i in range(d + 1):
            num = np.ones(r.shape[0], dtype=np.int64)
            for k in range(d + 1):
                if k != i:
                    num = num * ((r - k) % p) % p
            out = (out + vals[:, i] * (num * self.inv_den[i] % p)) % p
        return out


# ----------------------------------------------------------------------------- the liars
def distinct_roots(n_rows, k, p, gen):
    """n_rows sets of k distinct elements of {2, ..., p-1}."""
    roots = gen.integers(2, p, size=(n_rows, k))
    while k > 1:
        srt = np.sort(roots, axis=1)
        bad = np.any(srt[:, 1:] == srt[:, :-1], axis=1)
        if not bad.any():
            break
        roots[bad] = gen.integers(2, p, size=(int(bad.sum()), k))
    return roots


def planted_message(S, delta, d, p, gen):
    """s_j + c e_j, with e_j(X) = X prod (X - rho) over d - 1 random rho in F_p minus {0,1} and
    c = delta / e_j(1), so that the message sums to the claim at 0 and 1; rows with delta = 0 are
    honest. Returns the values at 0..d."""
    G = S.copy()
    rows = np.nonzero(delta)[0]
    if rows.size == 0:
        return G
    t = np.arange(d + 1, dtype=np.int64)
    e = np.broadcast_to(t, (rows.size, d + 1)).copy()        # the root 0
    if d > 1:
        rho = distinct_roots(rows.size, d - 1, p, gen)
        for k in range(d - 1):
            e = e * ((t[None, :] - rho[:, k:k + 1]) % p) % p
    e1 = e[:, 1]                                             # nonzero: no root equals 1
    inv = np.array([pow(int(a), p - 2, p) for a in e1], dtype=np.int64)
    c = delta[rows] * inv % p
    G[rows] = (S[rows] + c[:, None] * e) % p
    return G


def careless_message(S, delta, d, p, gen):
    """s_j + h_j with h_j uniform among polynomials of degree <= d with h(0) + h(1) = delta."""
    G = S.copy()
    rows = np.nonzero(delta)[0]
    if rows.size == 0:
        return G
    h = gen.integers(0, p, size=(rows.size, d + 1))
    h[:, 1] = (delta[rows] - h[:, 0]) % p
    G[rows] = (S[rows] + h) % p
    return G


def shifted_message(S, delta, p):
    inv2 = pow(2, p - 2, p)
    return (S + (delta * inv2 % p)[:, None]) % p


# ----------------------------------------------------------------------------- one batch of runs
def run_batch(clauses, deg, K, p, T, gen_v, gen_l, lag):
    r = gen_v.integers(0, p, size=(T, L)).astype(np.int64)
    claims = {"honest": K % p, "plant": (K + 1) % p, "careless": (K + 1) % p, "shift": (K + 1) % p}
    v = {k: np.full(T, c, dtype=np.int64) for k, c in claims.items()}
    alive = {k: np.ones(T, dtype=bool) for k in claims}
    for j in range(L):
        d = deg[j]
        S, _ = honest_round(clauses, L, j, d, r[:, :j], p)
        for k in claims:
            delta = (v[k] - S[:, 0] - S[:, 1]) % p
            if k == "honest":
                G = S
            elif k == "plant":
                G = planted_message(S, delta, d, p, gen_l)
            elif k == "careless":
                G = careless_message(S, delta, d, p, gen_l)
            else:
                G = shifted_message(S, delta, p)
            # the verifier: the check g_j(0) + g_j(1) = current claim, then the new claim g_j(r_j)
            alive[k] &= (G[:, 0] + G[:, 1]) % p == v[k]
            v[k] = lag[j](G, r[:, j])
    final = p_phi(clauses, r, p)
    return {k: int(np.sum(alive[k] & (v[k] == final))) for k in claims}


def main():
    t0 = time.time()
    clauses, deg = random_cnf(L, M, pk.rng(SEED_FORMULA))
    K = count_models(clauses, L)
    sd, dmax = sum(deg), max(deg)
    print(f"formula: l={L}, m={M}, models K={K}, degrees d_j={deg}, sum d_j={sd}, max d_j={dmax}")
    gen_v, gen_l = pk.rng(SEED_COINS), pk.rng(SEED_LIARS)
    rows = []
    for p in PRIMES:
        lag = [Lagrange(d, p) for d in deg]
        acc = {"honest": 0, "plant": 0, "careless": 0, "shift": 0}
        for _ in range(TRIALS // BATCH):
            res = run_batch(clauses, deg, K, p, BATCH, gen_v, gen_l, lag)
            for k in acc:
                acc[k] += res[k]
        n = TRIALS
        exact_plant = 1.0 - np.prod([1.0 - d / p for d in deg])
        exact_care = 1.0 - (1.0 - 1.0 / p) ** L
        bound = min(1.0, sd / p)
        out = [p, bound, min(1.0, L * dmax / p), exact_plant, exact_care, acc["honest"] / n]
        for k in ("plant", "careless"):
            rate = acc[k] / n
            lo, hi = pk.clopper_pearson(acc[k], n)
            out += [rate, rate - lo, hi - rate, acc[k]]
        out += [acc["shift"]]
        rows.append(out)
        print(f"p={p:5d}  honest {acc['honest']}/{n}  root-planting {acc['plant']}/{n} "
              f"(rate {acc['plant']/n:.4f}, exact {exact_plant:.4f}, bound {sd/p:.4f})  "
              f"careless {acc['careless']}/{n} (rate {acc['careless']/n:.4f}, exact {exact_care:.4f})  "
              f"constant-shift {acc['shift']}/{n}")
    pk.write_dat(OUT, ["p", "bound", "boundunif", "exactplant", "exactcare", "honest",
                       "plant", "plantem", "plantep", "plantk",
                       "care", "careem", "careep", "carek", "shiftk"], rows)
    grid = np.exp(np.linspace(np.log(55), np.log(5200), 160))
    theory = [[float(q), min(1.0, sd / q), 1.0 - np.prod([1.0 - d / q for d in deg]),
               1.0 - (1.0 - 1.0 / q) ** L] for q in grid]
    pk.write_dat(OUT_THEORY, ["p", "bound", "exactplant", "exactcare"], theory)
    # how close the union bound is to the exact value
    for p in (103, 1031):
        ex = 1.0 - np.prod([1.0 - d / p for d in deg])
        print(f"p={p}: union bound {sd/p:.4f}, exact {ex:.4f}, difference {sd/p-ex:.4f}, "
              f"second-order term (sum d/p)^2/2 = {(sd/p)**2/2:.4f}")
    print(f"uniform-degree verifier (only deg <= {dmax} checked) at p=1031: optimal liar "
          f"{1-(1-dmax/1031)**L:.4f} against {1-np.prod([1-d/1031 for d in deg]):.4f}")
    print(f"total time {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
