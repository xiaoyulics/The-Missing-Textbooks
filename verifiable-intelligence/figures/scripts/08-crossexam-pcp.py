"""Chapter 8 (Cross-Examination): the Hadamard PCP of Chapter 2 against a prover that sees the
queries, with and without a second, isolated prover.

The verifier of Chapter 2 reads 12 bits of a proof (f, g), f: F_2^l -> F_2 and g: F_2^{l x l} -> F_2,
and accepts every proof of an unsatisfiable system of quadratic equations with probability at
most 3/4; a "rank-one inconsistent pair" (f, g) = (h_u, h_W) with W - u u^T of rank one attains 3/4.
A prover that sees all twelve queries before answering passes with certainty: it answers from the
rank-one pair and, when the tensor test would fail, flips its answer g(Z). Cross-examination
(Protocol 8.1) sends the twelve queries to P1 and one of them, uniformly at random, to P2, who
answers from the fixed pair; the theorem "two provers simulate an oracle" bounds the acceptance by
1 - (1 - 3/4)/12 = 47/48, and the pair above attains it exactly: P1 departs from P2's table only
in its answer g(Z), and only when the tensor test would fail (probability exactly 1/4), so the
pair is caught with probability exactly (1/4)(1/12), whether or not query positions coincide.

Instance: l = 8 unknowns, M = 12 random equations with a planted solution u0; flipping b_1 leaves a
system without a solution, which u0 violates in one equation. Matrices are stored as 64-bit words,
entry (i, i') at bit 8 i + i'; the table g has 2^64 entries and is computed on demand.

Run from the volume root:  python3 figures/scripts/08-crossexam-pcp.py
Writes figures/data/08-crossexam-pcp-*.dat and prints every number the chapter quotes.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import protokit as pk

T0 = time.time()

OUT = "figures/data"
L, M = 8, 12
ROUNDS = 400_000
TS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
U64 = np.uint64


def parity(words):
    return (np.bitwise_count(np.asarray(words, dtype=U64)) & 1).astype(np.int8)


def outer(r, rp):
    """The 64-bit word of the matrix r rp^T for arrays of 8-bit vectors r, rp."""
    r = np.asarray(r, dtype=U64)
    rp = np.asarray(rp, dtype=U64)
    out = np.zeros(np.broadcast(r, rp).shape, dtype=U64)
    for i in range(L):
        bit = (r >> U64(i)) & U64(1)
        out |= (bit * rp) << U64(L * i)
    return out


def quad(A, u):
    """u^T A u for a 64-bit matrix word A and an 8-bit vector u."""
    return int(parity(np.uint64(A) & outer(u, u)))


def setup(gen):
    A = gen.integers(0, 2 ** 63, size=M, dtype=np.int64).astype(U64) * U64(2) + gen.integers(0, 2, size=M).astype(U64)
    u0 = int(gen.integers(1, 2 ** L))
    b_sat = np.array([quad(A[j], u0) for j in range(M)], dtype=np.int8)
    b = b_sat.copy()
    b[0] ^= 1
    # the system with b has no solution: check all 2^8 vectors
    sols = [u for u in range(2 ** L) if all(quad(A[j], u) == b[j] for j in range(M))]
    return A, u0, b_sat, b, sols


def rank_one_pair(A, u0, b):
    """W = u0 u0^T + a c^T with <A_j, W> = b_j for all j and a, c nonzero (search over 2^16 pairs)."""
    e = np.array([b[j] ^ quad(A[j], u0) for j in range(M)], dtype=np.int8)
    a = np.repeat(np.arange(1, 2 ** L), 2 ** L - 1)
    c = np.tile(np.arange(1, 2 ** L), 2 ** L - 1)
    words = outer(a, c)
    ok = np.ones(len(words), dtype=bool)
    for j in range(M):
        ok &= parity(A[j] & words) == e[j]
    idx = int(np.flatnonzero(ok)[0])
    W = int(outer(u0, u0)) ^ int(words[idx])
    return W, int(a[idx]), int(c[idx]), int(ok.sum())


def draw(gen, R, A, b):
    """The coins of R runs and the twelve query positions: (table, position) with table 0 = f, 1 = g."""
    x, y, r, rp = (gen.integers(0, 2 ** L, size=R).astype(U64) for _ in range(4))
    X, Y, Z, Zp = (gen.integers(0, 2 ** 63, size=R, dtype=np.int64).astype(U64) * U64(2)
                   + gen.integers(0, 2, size=R).astype(U64) for _ in range(4))
    zeta = gen.integers(0, 2, size=(R, M)).astype(bool)
    C = np.zeros(R, dtype=U64)
    for j in range(M):
        C ^= np.where(zeta[:, j], A[j], U64(0))
    beta = (zeta.astype(np.int8) @ b.astype(np.int8)) & 1
    tables = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1])
    pos = [x, y, x ^ y, X, Y, X ^ Y, r, rp, outer(r, rp) ^ Z, Z, C ^ Zp, Zp]
    return tables, pos, beta


def oracle_answers(u, W, tables, pos):
    return np.stack([parity((U64(u) if t == 0 else U64(W)) & p) for t, p in zip(tables, pos)], axis=1)


def tests(ans, beta):
    t1 = (ans[:, 0] ^ ans[:, 1]) == ans[:, 2]
    t2 = (ans[:, 3] ^ ans[:, 4]) == ans[:, 5]
    t3 = (ans[:, 6] & ans[:, 7]) == (ans[:, 8] ^ ans[:, 9])
    t4 = (ans[:, 10] ^ ans[:, 11]) == beta
    return t1 & t2 & t3 & t4, t3


def block_rates(acc, expected):
    rows = []
    for t in TS:
        B = len(acc) // t
        if B * expected(t) < 20:
            continue
        k = int(acc[:B * t].reshape(B, t).all(axis=1).sum())
        lo, hi = pk.clopper_pearson(k, B)
        rows.append((t, k / B, hi - k / B, k / B - lo, B))
    return rows


def main():
    gen = pk.rng(8)
    A, u0, b_sat, b, sols = setup(gen)
    assert not sols, "the flipped system must have no solution"
    W, a, c, nsol = rank_one_pair(A, u0, b)
    print(f"== l = {L}, M = {M}: planted u0 = {u0:08b}; after flipping b_1 no solution; "
          f"{nsol} rank-one pairs (a, c) with <A_j, u0u0^T + ac^T> = b_j")

    # honest pair on the satisfiable system (b unflipped), with cross-examination
    tables, pos, beta = draw(pk.rng(41), ROUNDS, A, b_sat)
    honest = oracle_answers(u0, int(outer(u0, u0)), tables, pos)   # P1's and P2's tables agree
    acc_h, _ = tests(honest, beta)
    i = pk.rng(42).integers(0, 12, size=ROUNDS)
    acc_h &= honest[np.arange(ROUNDS), i] == honest[np.arange(ROUNDS), i]
    print(f"  honest pair, cross-examined: per-round acceptance {acc_h.mean():.4f}")

    # cheating provers on the unsatisfiable system
    tables, pos, beta = draw(pk.rng(43), ROUNDS, A, b)
    fixed = oracle_answers(u0, W, tables, pos)            # the rank-one pair as a fixed table
    acc_fixed, t3 = tests(fixed, beta)
    adaptive = fixed.copy()
    adaptive[~t3, 9] ^= 1                                 # flip g(Z) when the tensor test would fail
    acc_adapt, _ = tests(adaptive, beta)
    i = pk.rng(44).integers(0, 12, size=ROUNDS)           # the query sent to P2
    agree = adaptive[np.arange(ROUNDS), i] == fixed[np.arange(ROUNDS), i]
    acc_cross = acc_adapt & agree
    for name, a_, exact in [("fixed rank-one pair, plain verifier", acc_fixed, 3 / 4),
                            ("adaptive prover, plain verifier", acc_adapt, 1.0),
                            ("adaptive P1 + fixed P2, cross-examined", acc_cross, 47 / 48)]:
        lo, hi = pk.clopper_pearson(int(a_.sum()), ROUNDS)
        print(f"  {name:40s}: {a_.mean():.4f} [{lo:.4f}, {hi:.4f}]  exact {exact:.4f}")
    print(f"  tensor test fails on the fixed pair in {1 - t3.mean():.4f} of runs (exact 1/4); "
          f"P1 deviates from P2's table only there, at one of 12 positions")

    curves = {"fixed": lambda t: 0.75 ** t, "adaptive": lambda t: 1.0,
              "cross": lambda t: (47 / 48) ** t, "honest": lambda t: 1.0}
    data = {"fixed": acc_fixed, "adaptive": acc_adapt, "cross": acc_cross, "honest": acc_h}
    for name, f in curves.items():
        rows = block_rates(data[name], f)
        pk.write_dat(f"{OUT}/08-crossexam-pcp-{name}.dat", ["t", "rate", "eplus", "eminus", "blocks"], rows)
        print(f"  {name:8s} after t rounds: " + ", ".join(f"t={r[0]}: {r[1]:.4f}" for r in rows))
        miss = [r[0] for r in rows if not (r[1] - r[3] - 1e-12 <= f(r[0]) <= r[1] + r[2] + 1e-12)]
        print(f"           95% intervals that miss the curve: {miss if miss else 'none'}")
    grid = np.arange(1, 281)
    pk.write_dat(f"{OUT}/08-crossexam-pcp-curves.dat", ["t", "cross", "fixed"],
                 [(int(t), (47 / 48) ** t, 0.75 ** t) for t in grid])
    for eps in (0.01,):
        print(f"== rounds for soundness error {eps}: plain verifier against fixed proofs "
              f"{int(np.ceil(np.log(eps) / np.log(0.75)))}, cross-examination "
              f"{int(np.ceil(np.log(eps) / np.log(47 / 48)))}")
    print(f"done in {time.time() - T0:.1f} s (after start-up)")


if __name__ == "__main__":
    main()
