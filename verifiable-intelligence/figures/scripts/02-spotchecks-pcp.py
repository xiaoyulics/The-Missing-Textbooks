"""The Hadamard PCP for quadratic equations over F_2, run against honest and cheating proofs.

Chapter 2 (Spot Checks), "Running the protocol". A system of M quadratic equations
u^T A_j u = b_j in l unknowns over F_2 is proved satisfiable by a proof (f, g): f should be the
Hadamard codeword x -> u.x of a solution u (2^l bits) and g the codeword Z -> <u u^T, Z> of its
tensor square (2^(l^2) bits). One run of the verifier (Protocol 2.1 of the chapter) makes four
tests with independent coins and reads 12 bits:

  L_f  f(x) + f(y) = f(x + y)                                        (3 queries)
  L_g  g(X) + g(Y) = g(X + Y)                                        (3 queries)
  T    f(r) f(r') = g(r r'^T + Z) + g(Z)                             (4 queries)
  S    g(C + Z') + g(Z') = beta, C = sum_j zeta_j A_j, beta = zeta.b  (2 queries)

The chapter proves perfect completeness and soundness error exactly 3/4. Proofs are oracles
computed on demand: for l = 8 the table g has 2^64 entries. Matrices are l^2-bit integers, entry
(i, j) at bit i*l + j. In the code, n stands for the chapter's l and m for its M.

The script
  1. checks the soundness error exhaustively for l = 2: every one of the 2^20 proofs, for every
     unsatisfiable system in 2 unknowns (up to its set of linearized solutions);
  2. runs the verifier R times against an honest proof (planted instance, l = 8, M = 12) and
     against cheating proofs for an unsatisfiable instance with the same left-hand sides: a valid
     encoding of a non-solution, inconsistent pairs (f, g) = (h_u, h_W) with W in the linearized
     solution set and rank(W - u u^T) = 1 or 2, a corrupted copy of the rank-one pair, and a proof
     patched on the span of the A_j; also against two weakened verifiers (no self-correction;
     no tensor test);
  3. measures acceptance after t independent runs, t = 1..10 (disjoint blocks of runs);
  4. compares with naive spot checks of an N-step chain with one bad step;
  5. prints the costs.

Run from the volume root:  python3 figures/scripts/02-spotchecks-pcp.py   (about 3 seconds)
Writes figures/data/02-spotchecks-pcp-{runs,repeat,naive,bound}.dat.
"""
import itertools
import os
import sys
import time
from fractions import Fraction
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

OUT = "figures/data/02-spotchecks-pcp"
R = 200_000          # runs of the verifier per prover
N_VARS, N_EQ = 8, 12
SEED = 5
ONE = np.uint64(1)


# ------------------------------------------------------------------ bits and matrices
def parity(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.uint64).copy()
    for s in (32, 16, 8, 4, 2, 1):
        v ^= v >> np.uint64(s)
    return (v & ONE).astype(np.uint8)


def ubits(gen, size, nbits):
    """Uniform nbits-bit integers (nbits <= 64) as uint64."""
    hi = gen.integers(0, 2 ** 32, size, dtype=np.uint64)
    lo = gen.integers(0, 2 ** 32, size, dtype=np.uint64)
    v = (hi << np.uint64(32)) | lo
    return v if nbits == 64 else v & np.uint64(2 ** nbits - 1)


def outer(r, rp, n):
    """r (x) r' as n^2-bit integers: bit i*n + j is r_i r'_j."""
    t = np.zeros(np.shape(r), dtype=np.uint64)
    for i in range(n):
        t |= np.where((np.asarray(r, dtype=np.uint64) >> np.uint64(i)) & ONE,
                      np.asarray(rp, dtype=np.uint64) << np.uint64(i * n), np.uint64(0))
    return t


def square(u, n):
    return int(outer(np.array([u], dtype=np.uint64), np.array([u], dtype=np.uint64), n)[0])


def rank2(rows):
    """Rank over F_2 of a list of integers (bit vectors)."""
    basis = []
    for v in rows:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)


def matrix_rows(W, n):
    return [(W >> (i * n)) & (2 ** n - 1) for i in range(n)]


# ------------------------------------------------------------------ instances
class System:
    """m equations <A_k, u (x) u> = b_k in n unknowns; A_k are n^2-bit integers, b an m-bit integer."""

    def __init__(self, n, A, b):
        self.n, self.A, self.b, self.m = n, list(A), int(b), len(A)
        self.A_arr = np.array(self.A, dtype=np.uint64)

    def violated(self, u):
        """Bit k is set if u violates equation k."""
        s = square(u, self.n)
        return sum(((bin(s & a).count("1") + (self.b >> k)) & 1) << k for k, a in enumerate(self.A))

    def solutions(self):
        return [u for u in range(2 ** self.n) if self.violated(u) == 0]

    def in_sol(self, W):
        return all(((bin(W & a).count("1") + (self.b >> k)) & 1) == 0 for k, a in enumerate(self.A))

    def combine(self, zeta):
        """C = sum_k zeta_k A_k and beta = zeta . b, for an array of m-bit integers zeta."""
        C = np.zeros(zeta.shape, dtype=np.uint64)
        for k, a in enumerate(self.A):
            C ^= np.where((zeta >> np.uint64(k)) & ONE, np.uint64(a), np.uint64(0))
        return C, parity(zeta & np.uint64(self.b))


# ------------------------------------------------------------------ proofs (oracles on demand)
class Proof:
    """f = h_u; g = h_W, optionally corrupted at a pseudo-random fraction eps of positions, or
    patched to answer zeta.b at every point sum_k zeta_k A_k of the span of the A_k."""

    def __init__(self, u, W, eps=0.0, salt=0, patch=None):
        self.u, self.W, self.eps, self.salt, self.patch = np.uint64(u), np.uint64(W), eps, salt, patch

    def f(self, x):
        return parity(x & self.u)

    def g(self, z):
        val = parity(z & self.W)
        if self.eps > 0:
            val ^= (splitmix(z ^ np.uint64(self.salt)) < np.uint64(int(self.eps * 2.0 ** 64))).astype(np.uint8)
        if self.patch is not None:
            inside, psi = self.patch.lookup(z)
            val = np.where(inside, psi, val).astype(np.uint8)
        return val


def splitmix(x):
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


class SpanPatch:
    """Membership in span{A_k} and the linear functional sum zeta_k A_k -> zeta . b on it."""

    def __init__(self, system):
        basis = []                       # (vector, pivot bit, combination of equations)
        for k, a in enumerate(system.A):
            v, c = a, 1 << k
            for (bv, piv, bc) in basis:
                if (v >> piv) & 1:
                    v, c = v ^ bv, c ^ bc
            if v:
                basis.append((v, v.bit_length() - 1, c))
        assert len(basis) == system.m, "the A_k should be independent"
        self.basis, self.b = basis, system.b

    def lookup(self, z):
        v = np.asarray(z, dtype=np.uint64).copy()
        comb_ = np.zeros(v.shape, dtype=np.uint64)
        for (bv, piv, bc) in sorted(self.basis, key=lambda t: -t[1]):
            hit = (v >> np.uint64(piv)) & ONE == ONE
            v = np.where(hit, v ^ np.uint64(bv), v)
            comb_ = np.where(hit, comb_ ^ np.uint64(bc), comb_)
        return v == 0, parity(comb_ & np.uint64(self.b))


# ------------------------------------------------------------------ the verifier
def run(proof, system, runs, gen, selfcorrect=True, tensor=True):
    """Accept/reject of `runs` independent runs of Protocol 2.1 (boolean array)."""
    n, n2, m = system.n, system.n ** 2, system.m
    f, g = proof.f, proof.g
    x, y = ubits(gen, runs, n), ubits(gen, runs, n)
    ok = (f(x) ^ f(y) ^ f(x ^ y)) == 0                                     # L_f
    X, Y = ubits(gen, runs, n2), ubits(gen, runs, n2)
    ok &= (g(X) ^ g(Y) ^ g(X ^ Y)) == 0                                    # L_g
    r, rp, z = ubits(gen, runs, n), ubits(gen, runs, n), ubits(gen, runs, n2)
    t = outer(r, rp, n)
    gt = (g(t ^ z) ^ g(z)) if selfcorrect else g(t)
    if tensor:
        ok &= (f(r) & f(rp)) == gt                                          # T
    zeta, zp = ubits(gen, runs, m), ubits(gen, runs, n2)
    C, beta = system.combine(zeta)
    gC = (g(C ^ zp) ^ g(zp)) if selfcorrect else g(C)
    ok &= gC == beta                                                        # S
    return ok


# ------------------------------------------------------------------ 1. exhaustive check, n = 2
def exhaustive_n2():
    n, N1, N2 = 2, 4, 16
    F = np.array([[(f >> x) & 1 for x in range(N1)] for f in range(2 ** N1)], dtype=np.uint8)
    G = np.array([[(g >> z) & 1 for z in range(N2)] for g in range(2 ** N2)], dtype=np.uint8)

    def lin_pass(T, N):
        acc = np.zeros(T.shape[0])
        for a in range(N):
            for b in range(N):
                acc += (T[:, a] ^ T[:, b] ^ T[:, a ^ b]) == 0
        return acc / N ** 2

    PLf, PLg = lin_pass(F, N1), lin_pass(G, N2)
    PT = np.zeros((2 ** N1, 2 ** N2))
    for r in range(N1):
        for rp in range(N1):
            t = int(outer(np.array([r]), np.array([rp]), n)[0])
            lhs = F[:, r] & F[:, rp]
            for z in range(N2):
                PT += lhs[:, None] == (G[:, t ^ z] ^ G[:, z])[None, :]
    PT /= N1 * N1 * N2
    sign = 1.0 - 2.0 * G
    chars = np.array([[(-1) ** bin(W & z).count("1") for z in range(N2)] for W in range(N2)])
    G2 = (sign @ chars.T / N2) ** 2              # squared Fourier coefficients of g
    # S passes with probability 1/2 + 1/2 * (Fourier weight of g on the linearized solutions)

    def span(vs):
        S = {0}
        for v in vs:
            S |= {s ^ v for s in S}
        return frozenset(S)
    subspaces = {span(vs) for k in range(5) for vs in itertools.combinations(range(1, N2), k)}
    affine = {frozenset(c ^ v for v in V) for V in subspaces for c in range(N2)}
    squares = {square(u, n) for u in range(N1)}
    unsat = [S for S in affine if not S & squares]
    base = PLf[:, None] * PLg[None, :] * PT
    best = 0.5 * base.max()                       # an empty linearized solution set
    for S in unsat:
        ind = np.zeros(N2)
        ind[list(S)] = 1
        best = max(best, (base * (0.5 + 0.5 * (G2 @ ind))[None, :]).max())
    # the system u1 = 1, u2 = 1, u1 u2 = 0
    tight = System(2, [0b0001, 0b1000, 0b0010], 0b011)
    S = frozenset(W for W in range(N2) if tight.in_sol(W))
    ind = np.zeros(N2)
    ind[list(S)] = 1
    acc = base * (0.5 + 0.5 * (G2 @ ind))[None, :]
    return len(unsat), Fraction(best).limit_denominator(1 << 20), Fraction(acc.max()).limit_denominator(1 << 20), \
        int(np.sum(acc > acc.max() - 1e-12))


# ------------------------------------------------------------------ instances for the runs
def make_instances(gen):
    n, m = N_VARS, N_EQ
    while True:
        A = [int(v) for v in ubits(gen, m, n * n)]
        ustar = int(gen.integers(1, 2 ** n))
        b = sum((bin(square(ustar, n) & a).count("1") & 1) << k for k, a in enumerate(A))
        sat = System(n, A, b)
        unsat = System(n, A, b ^ 1)          # flip the right-hand side of the first equation
        if rank2(A) == m and len(sat.solutions()) == 1 and not unsat.solutions():
            return ustar, sat, unsat


def find_cheats(system):
    """Inconsistent pairs: (u, W) with W in the linearized solution set and rank(W - u u^T) = 1, 2."""
    n, m = system.n, system.m
    vecs = np.arange(2 ** n, dtype=np.int64)
    bitsv = ((vecs[:, None] >> np.arange(n)) & 1).astype(np.int64)         # 2^n x n
    B = np.zeros((2 ** n, 2 ** n), dtype=np.int64)                          # B[x, y]: bits x^T A_k y
    for k, a in enumerate(system.A):
        M = np.array([[(a >> (i * n + j)) & 1 for j in range(n)] for i in range(n)])
        B |= ((bitsv @ M @ bitsv.T) % 2) << k
    rank1 = rank2_ = None
    for u in range(2 ** n):
        c = system.violated(u)               # needed value of the correction on the equations
        hits = np.argwhere(B == c)
        if rank1 is None and len(hits):
            x, y = map(int, hits[0])
            rank1 = (u, square(u, n) ^ int(outer(np.array([x]), np.array([y]), n)[0]))
        if rank2_ is None:
            for x1 in range(1, 2 ** n):
                for y1 in range(1, 2 ** n):
                    rest = np.argwhere(B == (c ^ int(B[x1, y1])))
                    for x2, y2 in rest[:50]:
                        x2, y2 = int(x2), int(y2)
                        D = int(outer(np.array([x1]), np.array([y1]), n)[0]) ^ int(outer(np.array([x2]), np.array([y2]), n)[0])
                        if rank2(matrix_rows(D, n)) == 2:
                            rank2_ = (u, square(u, n) ^ D)
                            break
                    if rank2_ or y1 > 8:
                        break
                if rank2_ or x1 > 8:
                    break
        if rank1 and rank2_:
            return rank1, rank2_
    raise RuntimeError("no inconsistent pair found")


def main() -> None:
    t0 = time.time()
    # 1. exhaustive soundness check for n = 2
    n_unsat, best, tight, n_opt = exhaustive_n2()
    print(f"[l = 2] exhaustive over all 2^20 proofs and {n_unsat} linearized solution sets of unsatisfiable "
          f"systems (plus the empty one): maximum acceptance = {best}")
    print(f"[l = 2] system u1 = 1, u2 = 1, u1 u2 = 0: maximum acceptance = {tight}, attained by {n_opt} proofs")

    # 2. one run of the verifier against each proof
    gen = pk.rng(SEED)
    ustar, sat, unsat = make_instances(gen)
    n = N_VARS
    (u1, W1), (u2, W2) = find_cheats(unsat)
    r1 = rank2(matrix_rows(W1 ^ square(u1, n), n))
    r2 = rank2(matrix_rows(W2 ^ square(u2, n), n))
    assert unsat.in_sol(W1) and unsat.in_sol(W2) and (r1, r2) == (1, 2)
    print(f"[l = {n}, M = {N_EQ}] planted solution u* = {ustar:08b}; the unsatisfiable variant flips b_1; "
          f"u* violates {bin(unsat.violated(ustar)).count('1')} equation(s) of it")
    print(f"  proof length 2^{n} + 2^{n * n} bits; random bits per run 4l^2 + 4l + M = {4 * n * n + 4 * n + N_EQ}")
    provers = [
        ("honest", "honest proof, satisfiable system", sat, Proof(ustar, square(ustar, n)), 1.0),
        ("nonsol", "valid encoding of a non-solution", unsat, Proof(ustar, square(ustar, n)), 0.5),
        ("rank1", "inconsistent pair, rank 1", unsat, Proof(u1, W1), 0.75),
        ("rank2", "inconsistent pair, rank 2", unsat, Proof(u2, W2), 0.625),
        ("corrupt", "rank-1 pair, 5% of g corrupted", unsat, Proof(u1, W1, eps=0.05, salt=77), None),
        ("patched", "non-solution, g patched on span A", unsat,
         Proof(ustar, square(ustar, n), patch=SpanPatch(unsat)), None),
    ]
    rows = []
    accept_streams = {}
    print("  per-run acceptance: full verifier | without self-correction | without tensor test")
    for i, (key, label, system, proof, exact) in enumerate(provers):
        res = []
        for j, (sc, te) in enumerate([(True, True), (False, True), (True, False)]):
            ok = run(proof, system, R, pk.rng(1000 + 10 * i + j), selfcorrect=sc, tensor=te)
            if j == 0:
                accept_streams[key] = ok
            k = int(ok.sum())
            lo, hi = pk.clopper_pearson(k, R)
            res.append((k / R, lo, hi))
        rows.append([i + 1] + [v for trip in res for v in trip])
        ex = "" if exact is None else f" (exact {exact})"
        print(f"  {label:36s} " + " | ".join(f"{a:.4f} [{lo:.4f}, {hi:.4f}]" for a, lo, hi in res) + ex)
    pk.write_dat(OUT + "-runs.dat", ["prover", "full", "fulllo", "fullhi", "nosc", "nosclo", "noschi",
                                     "notensor", "notensorlo", "notensorhi"], rows)
    worst = max(r[1] for r in rows[1:])
    print(f"  largest measured per-run acceptance of a cheating proof (full verifier): {worst:.4f}; bound 0.75")

    # 3. acceptance after t independent runs (disjoint blocks of the runs above)
    rep_rows = []
    for t in range(1, 11):
        row = [t, 0.75 ** t, 0.5 ** t, 0.625 ** t]
        for key in ("honest", "nonsol", "rank1", "rank2", "corrupt"):
            ok = accept_streams[key]
            blocks = (R // t)
            allacc = ok[: blocks * t].reshape(blocks, t).all(axis=1)
            k = int(allacc.sum())
            lo, hi = pk.clopper_pearson(k, blocks)
            row += [k / blocks, k / blocks - lo, hi - k / blocks]
        rep_rows.append(row)
    hdr = ["t", "bound", "half", "fiveeighths"]
    for key in ("honest", "nonsol", "rank1", "rank2", "corrupt"):
        hdr += [key, key + "lo", key + "hi"]
    pk.write_dat(OUT + "-repeat.dat", hdr, rep_rows)
    print("  acceptance after t runs (rank-1 pair; bound (3/4)^t):",
          ", ".join(f"t={r[0]}: {r[10]:.4f}" for r in rep_rows[::3]))
    viol = [(r[0], key) for r in rep_rows for key, col in (("nonsol", 7), ("rank1", 10), ("rank2", 13), ("corrupt", 16))
            if r[col] - r[col + 1] > 0.75 ** r[0] + 1e-12]
    print(f"  measured rates whose lower 95% limit exceeds (3/4)^t: {viol if viol else 'none'}")

    # 4. naive spot checks of an L-step chain with one bad step, against bits read by the PCP
    gen = pk.rng(SEED + 1)
    trials = 20000
    naive = []
    for L in (1000, 10000):
        for q in (12, 36, 120, 360):
            bad = gen.integers(0, L, trials)
            hits = sum(1 for b in bad if b in set(gen.choice(L, q, replace=False).tolist()))
            lo, hi = pk.clopper_pearson(hits, trials)
            naive.append([L, q, q / L, hits / trials, hits / trials - lo, hi - hits / trials])
    pk.write_dat(OUT + "-naive.dat", ["L", "q", "exact", "measured", "elo", "ehi"], naive)
    for r in naive:
        print(f"  naive chain N = {r[0]:5d}, q = {r[1]:3d} steps: detection {r[3]:.4f} (exact q/N = {r[2]:.4f})")
    curve = [[12 * t, min(1.0, 12 * t / 1000), 12 * t / 10000, 1 - 0.75 ** t] for t in range(1, 101)]
    pk.write_dat(OUT + "-bound.dat", ["q", "naiveA", "naiveB", "pcp"], curve)
    for t in (1, 3, 8, 16, 17):
        q = 12 * t
        exact = 1 - comb(10000 - 1, q) / comb(10000, q)
        print(f"  PCP: t = {t:2d} runs read {q:3d} bits; worst-case detection 1 - (3/4)^t = {1 - 0.75 ** t:.5f}; "
              f"checking {q} random steps of a 10^4-step chain with one bad step: {exact:.4f}")

    # 5. costs of one run
    m = N_EQ
    print(f"  costs (l = {n}, M = {m}): 12 queries; {4 * n * n + 4 * n + m} random bits; "
          f"verifier forms C with at most {m * n * n} bit operations; honest answer to a query: "
          f"parity of {n * n} ANDed bits")
    tt = time.time()
    run(provers[0][3], sat, R, pk.rng(9))
    print(f"  measured: {1e6 * (time.time() - tt) / R:.2f} microseconds per run (vectorized over {R} runs)")
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
