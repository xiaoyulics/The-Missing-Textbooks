#!/usr/bin/env python3
"""Chapter 5 (sumcheck): the degree-reduced protocol for TQBF against an honest and a lying prover.

Runs Protocol 5.2 on a random quantified Boolean formula Psi = E x1 A x2 E x3 A x4 E x5 phi, with
phi a random 3-CNF with m = 8 clauses, over F_p for ten primes p (all p = 3 mod 4), and measures
how often the verifier accepts. The operator string is
    Q1 L1 Q2 L1 L2 Q3 L1 L2 L3 Q4 L1 .. L4 Q5 L1 .. L5      (K = n(n+3)/2 = 20 operators),
and the verifier's degree bound in the step of an operator is 1 for a quantifier, d_j (the number
of clauses containing x_j) for L_j in the last block, and 2 for every other L_j: the bound of
Theorem 5.11 on the soundness error is (n^2 + sum_j d_j)/p. Two provers face the same coins:

  honest          claims the true value of Psi and sends the true polynomials tau_k;
  root-planting   claims the opposite value; while its current claim is false it sends a
                  polynomial h_k that passes the verifier's test and agrees with tau_k at exactly
                  delta_k points (the most a different polynomial of degree <= delta_k can);
                  once its claim is true it plays honestly. Exact acceptance 1 - prod(1 - delta_k/p)
                  whenever the planting succeeds, which the script checks and reports.

The honest prover evaluates the polynomials P_k through truth tables: the polynomial in front of
the quantifier Q_i is the multilinear extension of the truth table of the residual formula
Q_{i+1} x_{i+1} ... Q_n x_n phi, so every value it needs is a fold of a table of size at most 2^n.
A brute-force evaluator that applies the operators literally checks this on a few points.

Writes figures/data/05-sumcheck-tqbf.dat and figures/data/05-sumcheck-tqbf-theory.dat.
Run from the volume root:  python3 figures/scripts/05-sumcheck-tqbf.py   (about 30 s on a laptop)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

N_VARS, M = 5, 8
QUANT = "EAEAE"                     # E: exists, A: for all, for x1..x5
SEED_FORMULA, SEED_COINS, SEED_LIAR = 7, 707, 7007
PRIMES = [67, 103, 163, 251, 419, 643, 1031, 1619, 2579, 4091]   # all 3 mod 4
TRIALS = 20000
BATCH = 1000
OUT = "figures/data/05-sumcheck-tqbf.dat"
OUT_THEORY = "figures/data/05-sumcheck-tqbf-theory.dat"


# ----------------------------------------------------------------------------- formula and tables
def random_qbf(n, m, gen):
    while True:
        clauses = []
        for _ in range(m):
            vs = gen.choice(n, size=3, replace=False)
            signs = gen.integers(0, 2, size=3)
            clauses.append(tuple(zip(vs.tolist(), signs.tolist())))
        deg = [sum(1 for c in clauses for (v, _) in c if v == j) for j in range(n)]
        if min(deg) >= 1:
            return clauses, deg


def truth_tables(clauses, n, quant):
    """T[i] = truth table (length 2^i, x1 the most significant bit) of Q_{i+1}..Q_n phi."""
    xs = (np.arange(2 ** n)[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1
    sat = np.ones(2 ** n, dtype=np.int64)
    for c in clauses:
        cl = np.zeros(2 ** n, dtype=bool)
        for v, s in c:
            cl |= xs[:, v] == s
        sat &= cl.astype(np.int64)
    T = [None] * (n + 1)
    T[n] = sat
    for i in range(n - 1, -1, -1):
        pair = T[i + 1].reshape(-1, 2)                    # last variable x_{i+1} is the low bit
        T[i] = pair.min(axis=1) if quant[i] == "A" else pair.max(axis=1)
    return T


def one_minus_literal(x, s, p):
    return (1 - x) % p if s == 1 else x % p


def p_phi(clauses, pts, p):
    """P_phi at points pts (..., n): the product of clause polynomials 1 - prod(1 - literal)."""
    out = np.ones(pts.shape[:-1], dtype=np.int64)
    for c in clauses:
        prod = np.ones(pts.shape[:-1], dtype=np.int64)
        for v, s in c:
            prod = prod * one_minus_literal(pts[..., v], s, p) % p
        out = out * ((1 - prod) % p) % p
    return out


def fold(table, coords, p):
    """Fix the leading coordinates of a multilinear table (T, 2^k) to the values coords (T, c):
    returns (T, 2^(k-c)), the multilinear extension evaluated there."""
    vals = table % p
    for c in range(coords.shape[1]):
        half = vals.shape[1] // 2
        lo, hi = vals[:, :half], vals[:, half:]
        vals = (lo + (hi - lo) % p * coords[:, c:c + 1]) % p
    return vals


def combine(q, u0, u1, a, p):
    """The verifier's test A_k: the value of O_k p at the current point from p's values at 0, 1."""
    if q == "A":
        return u0 * u1 % p
    if q == "E":
        return (1 - (1 - u0) * (1 - u1)) % p
    return ((1 - a) % p * u0 + a * u1) % p            # linearization L_j at a_j = a


# ----------------------------------------------------------------------------- operators
def operator_string(n, quant, deg):
    """List of (kind, var, block, delta): kind in 'A', 'E', 'L'; var 0-indexed; block i means the
    operator sits in the group that follows Q_i (1-indexed); delta is the verifier's degree bound."""
    ops = []
    for i in range(1, n + 1):
        ops.append((quant[i - 1], i - 1, i, 1))
        for j in range(1, i + 1):
            ops.append(("L", j - 1, i, deg[j - 1] if i == n else 2))
    return ops


def honest_values(k, ops, a, T, clauses, n, quant, p):
    """Values at X = 0..delta of the true polynomial t_k(X) = P_k(a with X_var := X), for each
    trial; a has shape (Tr, n). P_k is the polynomial that O_k acts on."""
    kind, var, block, delta = ops[k]
    Tr = a.shape[0]
    xs = np.arange(delta + 1, dtype=np.int64)
    out = np.zeros((Tr, delta + 1), dtype=np.int64)
    if kind in "AE":
        # P_k = M_i, the multilinear extension of T[i] (i = block = var + 1)
        i = block
        tab = np.broadcast_to(T[i], (Tr, 2 ** i))
        two = fold(tab, a[:, :i - 1], p)                 # values at X = 0, 1
        for x in range(delta + 1):
            out[:, x] = (two[:, 0] + (two[:, 1] - two[:, 0]) * x) % p
        return out
    i, j = block, var + 1                              # P_k = L_{j+1} .. L_i R_i
    for x in xs:
        y = a[:, :i].copy()
        y[:, j - 1] = x
        if i < n:
            # R_i(z) = Q_{i+1} applied to M_{i+1}: fold T[i+1] over the first j coordinates
            tab = np.broadcast_to(T[i + 1], (Tr, 2 ** (i + 1)))
            F = fold(tab, y[:, :j], p)                    # (Tr, 2^(i+1-j)) over x_{j+1}..x_{i+1}
            pair = F.reshape(Tr, -1, 2)
            R = combine(quant[i], pair[:, :, 0], pair[:, :, 1], None, p)   # over x_{j+1}..x_i
        else:
            B = 2 ** (n - j)
            bits = (np.arange(B)[:, None] >> (n - j - 1 - np.arange(n - j))[None, :]) & 1
            pts = np.empty((Tr, B, n), dtype=np.int64)
            pts[:, :, :j] = y[:, None, :j]
            pts[:, :, j:] = bits[None, :, :]
            R = p_phi(clauses, pts, p)                    # (Tr, 2^(n-j))
        out[:, x] = fold(R, y[:, j:i], p)[:, 0]
    return out


def brute_P(k, ops, pt, clauses, quant, p, memo=None):
    """P_k at one point, by applying the operators O_{k+1}, ..., O_K literally (memoized on the
    point, which below level k differs from pt only in coordinates set to 0 or 1)."""
    memo = {} if memo is None else memo
    key = (k, tuple(int(x) for x in pt))
    if key in memo:
        return memo[key]
    if k == len(ops):
        val = int(p_phi(clauses, np.array(pt, dtype=np.int64), p))
    else:
        kind, var, _, _ = ops[k]
        lo, hi = list(pt), list(pt)
        lo[var], hi[var] = 0, 1
        u0 = brute_P(k + 1, ops, lo, clauses, quant, p, memo)
        u1 = brute_P(k + 1, ops, hi, clauses, quant, p, memo)
        val = int(combine(kind, u0, u1, pt[var], p))
    memo[key] = val
    return val


# ----------------------------------------------------------------------------- the liar
def inv(x, p):
    return pow(int(x) % p, p - 2, p)


def sqrt_mod(x, p):
    """A square root of x mod p (p = 3 mod 4), or None."""
    x %= p
    r = pow(x, (p + 1) // 4, p)
    return r if r * r % p == x else None


def plant_one(kind, t0, t1, v, a, delta, p, gen):
    """Coefficient c and roots R (|R| = delta) such that h = t + c prod_{rho in R}(X - rho) passes
    the test of the operator, given the true values t0 = t(0), t1 = t(1) and the claim v; None if
    the search fails."""
    if kind == "L":
        # linear test (1-a) h(0) + a h(1) = v: plant 0 (if a != 0) or 1 (if a = 0), plus delta-1
        # random points of F_p minus {0, 1}
        first = 0 if a % p != 0 else 1
        others = []
        while len(others) < delta - 1:
            rho = int(gen.integers(2, p))
            if rho not in others:
                others.append(rho)
        R = [first] + others
        e0 = 1
        for rho in R:
            e0 = e0 * ((0 - rho) % p) % p
        e1 = 1
        for rho in R:
            e1 = e1 * ((1 - rho) % p) % p
        br = ((1 - a) * e0 + a * e1) % p
        if br == 0:
            return None
        c = (v - ((1 - a) * t0 + a * t1)) % p * inv(br, p) % p
        return c, R
    # quantifier steps have delta = 1; for E work with u = 1 - t and target 1 - v
    sign = 1
    u0, u1, w = t0, t1, v
    if kind == "E":
        u0, u1, w, sign = (1 - t0) % p, (1 - t1) % p, (1 - v) % p, -1
    if u0 != 0:                                        # root 0: u0 (u1 + c') = w
        cp = (w * inv(u0, p) - u1) % p
        return (sign * cp) % p, [0]
    if u1 != 0:                                        # root 1: (u0 - c') u1 = w
        cp = (u0 - w * inv(u1, p)) % p
        return (sign * cp) % p, [1]
    for rho in range(2, p):                            # -c'^2 rho (1 - rho) = w
        s = sqrt_mod((-w) * inv(rho * (1 - rho), p), p)
        if s is not None and s != 0:
            return (sign * s) % p, [rho]
    return None


def planted_values(kind, tv, v, a, delta, p, gen):
    """Message values at 0..delta for the root-planting liar; rows whose claim is true get t."""
    out = tv.copy()
    fails = 0
    xs = np.arange(delta + 1)
    for row in range(tv.shape[0]):
        t0, t1 = int(tv[row, 0]), int(tv[row, 1])
        if int(combine(kind, t0, t1, int(a[row]), p)) == int(v[row]):
            continue
        res = plant_one(kind, t0, t1, int(v[row]), int(a[row]), delta, p, gen)
        if res is None:
            fails += 1
            continue
        c, R = res
        e = np.ones(delta + 1, dtype=np.int64)
        for rho in R:
            e = e * ((xs - rho) % p) % p
        out[row] = (tv[row] + c * e) % p
    return out, fails


class Lagrange:
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


# ----------------------------------------------------------------------------- runs
def run_batch(ops, T, clauses, n, quant, truth, p, Tr, gen_v, gen_l, lags):
    r = gen_v.integers(0, p, size=(Tr, len(ops))).astype(np.int64)
    a = {s: np.zeros((Tr, n), dtype=np.int64) for s in ("honest", "plant")}
    v = {"honest": np.full(Tr, truth, dtype=np.int64), "plant": np.full(Tr, 1 - truth, dtype=np.int64)}
    alive = {s: np.ones(Tr, dtype=bool) for s in a}
    fails = 0
    for k, (kind, var, _, delta) in enumerate(ops):
        for s in a:
            tv = honest_values(k, ops, a[s], T, clauses, n, quant, p)
            if s == "honest":
                G = tv
            else:
                G, f = planted_values(kind, tv, v[s], a[s][:, var], delta, p, gen_l)
                fails += f
            test = combine(kind, G[:, 0], G[:, 1], a[s][:, var], p)
            alive[s] &= test == v[s]
            v[s] = lags[k](G, r[:, k])
            a[s][:, var] = r[:, k]
    acc = {}
    for s in a:
        acc[s] = int(np.sum(alive[s] & (p_phi(clauses, a[s], p) == v[s])))
    return acc, fails


def main():
    t0 = time.time()
    clauses, deg = random_qbf(N_VARS, M, pk.rng(SEED_FORMULA))
    T = truth_tables(clauses, N_VARS, QUANT)
    truth = int(T[0][0])
    ops = operator_string(N_VARS, QUANT, deg)
    deltas = [op[3] for op in ops]
    sdelta = sum(deltas)
    print(f"Psi = {QUANT} phi, n={N_VARS}, m={M}, clauses={clauses}")
    print(f"degrees d_j={deg}, truth value of Psi = {truth}, operators K={len(ops)}, "
          f"degree bounds={deltas}, sum={sdelta} = n^2 + sum d_j = {N_VARS**2 + sum(deg)}")
    # check the table-based honest prover against the literal operator definitions
    gen = pk.rng(1)
    p = 101
    for k in range(len(ops)):
        a = gen.integers(0, p, size=(2, N_VARS)).astype(np.int64)
        tv = honest_values(k, ops, a, T, clauses, N_VARS, QUANT, p)
        for row in range(2):
            for x in range(ops[k][3] + 1):
                pt = list(a[row])
                pt[ops[k][1]] = x
                assert tv[row, x] == brute_P(k + 1, ops, pt, clauses, QUANT, p), (k, row, x)
    assert brute_P(0, ops, [0] * N_VARS, clauses, QUANT, p) == truth
    print("table-based prover agrees with the literal operator definitions")
    gen_v, gen_l = pk.rng(SEED_COINS), pk.rng(SEED_LIAR)
    rows, total_fails = [], 0
    for p in PRIMES:
        lags = [Lagrange(d, p) for d in deltas]
        acc = {"honest": 0, "plant": 0}
        for _ in range(TRIALS // BATCH):
            res, f = run_batch(ops, T, clauses, N_VARS, QUANT, truth, p, BATCH, gen_v, gen_l, lags)
            total_fails += f
            for s in acc:
                acc[s] += res[s]
        exact = 1.0 - np.prod([1.0 - d / p for d in deltas])
        rate = acc["plant"] / TRIALS
        lo, hi = pk.clopper_pearson(acc["plant"], TRIALS)
        rows.append([p, min(1.0, sdelta / p), exact, acc["honest"] / TRIALS, rate, rate - lo, hi - rate,
                     acc["plant"]])
        z = (rate - exact) / np.sqrt(exact * (1 - exact) / TRIALS)
        print(f"p={p:5d}  honest {acc['honest']}/{TRIALS}  root-planting {acc['plant']}/{TRIALS} "
              f"(rate {rate:.4f} [{lo:.4f}, {hi:.4f}], exact {exact:.4f}, bound {sdelta/p:.4f}, "
              f"{z:+.1f} standard errors from exact)")
    print(f"planting failures (no admissible h found): {total_fails}")
    pk.write_dat(OUT, ["p", "bound", "exact", "honest", "plant", "plantem", "plantep", "plantk"], rows)
    grid = np.exp(np.linspace(np.log(55), np.log(5200), 160))
    pk.write_dat(OUT_THEORY, ["p", "bound", "exact"],
                 [[float(q), min(1.0, sdelta / q), 1.0 - np.prod([1.0 - d / q for d in deltas])] for q in grid])
    print(f"total time {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
