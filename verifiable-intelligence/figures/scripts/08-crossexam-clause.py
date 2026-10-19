"""Chapter 8 (Cross-Examination): the clause-variable game, run against honest and cheating provers.

The verifier sends prover P1 a uniformly random clause of a 3-CNF formula and prover P2 a uniformly
random one of that clause's three variables. It accepts if P1's three bits satisfy the clause and
P1's bit for the cross-examined variable equals P2's bit. (P1 is told the clause, not only its three
variables; on phi8, whose clauses all have the same variables, this is what lets it adapt.) On a
formula on which the least number of clauses an assignment violates is k >= 1, out of m, the best
isolated pair is accepted with probability exactly 1 - k/(3m) per round (Proposition
"clause-variable game"); a pair that answers from an assignment violating k clauses gets 1 - k/m;
a pair in which P1 may send one bit to P2 gets 1.

Formulas: phi8, the eight clauses on x1, x2, x3 (every assignment violates exactly one clause);
psi7, phi8 without the clause (not x1 or not x2 or not x3), satisfied only by 111; a random
unsatisfiable formula (n = 12 variables, m = 60 clauses, minimum number of violated clauses k
computed exactly over all 2^12 assignments); a random satisfiable formula with several satisfying
assignments, for honest provers that do not coordinate on one of them.

Run from the volume root:  python3 figures/scripts/08-crossexam-clause.py
Writes figures/data/08-crossexam-clause-*.dat and prints every number the chapter quotes.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import protokit as pk

T0 = time.time()

OUT = "figures/data"
ROUNDS = 1_500_000                   # simulated rounds per strategy
TS = [1, 2, 4, 8, 16, 32, 64, 128]   # numbers of sequential rounds in the figure


# ---------------------------------------------------------------- formulas
# A clause is (vars, signs): literal k is x_{vars[k]} if signs[k] = 1 and its negation if
# signs[k] = 0; it is true under sigma exactly when sigma[vars[k]] == signs[k].
def phi8():
    vs = np.array([[0, 1, 2]] * 8)
    sg = np.array([[(c >> 2) & 1, (c >> 1) & 1, c & 1] for c in range(8)])
    return vs, sg


def random_formula(n, m, gen):
    vs = np.array([gen.choice(n, 3, replace=False) for _ in range(m)])
    sg = gen.integers(0, 2, size=(m, 3))
    return vs, sg


def all_assignments(n):
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def violated(vs, sg, sigmas):
    """Boolean matrix (assignments x clauses): clause violated by the assignment."""
    vals = sigmas[:, vs]                       # (A, m, 3)
    return np.all(vals != sg[None, :, :], axis=2)


def satisfies(answers, sg):
    """answers: (R, 3) bits for the clause's variables; sg: (R, 3) signs."""
    return np.any(answers == sg, axis=1)


# ---------------------------------------------------------------- one round, vectorized
def play(vs, sg, p1, p2, gen, rounds, channel=False):
    """p1(j, gen) -> (answers (R,3), message bits (R,)); p2(v, mu) -> bits (R,).
    Returns a boolean array: accepted in each round."""
    m = len(vs)
    j = gen.integers(0, m, size=rounds)
    i = gen.integers(0, 3, size=rounds)
    a, mu = p1(j, gen)
    v = vs[j, i]
    b = p2(v, mu if channel else np.zeros(rounds, dtype=np.int8), gen)
    ok_clause = satisfies(a, sg[j])
    ok_cross = a[np.arange(rounds), i] == b
    return ok_clause & ok_cross


def from_assignment(vs, sigma):
    """P1 answers sigma restricted to the clause; P2 answers sigma."""
    def p1(j, gen):
        return sigma[vs[j]], np.zeros(len(j), dtype=np.int8)

    def p2(v, mu, gen):
        return sigma[v]
    return p1, p2


def best_isolated(vs, sg, sigma):
    """P2 answers sigma; P1 answers sigma, except that on a clause sigma violates it flips the
    clause's first variable (which satisfies the clause)."""
    viol = np.all(sigma[vs] != sg, axis=1)

    def p1(j, gen):
        a = sigma[vs[j]].copy()
        a[viol[j], 0] ^= 1
        return a, np.zeros(len(j), dtype=np.int8)

    def p2(v, mu, gen):
        return sigma[v]
    return p1, p2


def one_bit_channel(vs, sg, sigma):
    """P1 tells P2 one bit mu = [sigma violates the clause] and answers sigma XOR mu on the clause;
    P2 answers sigma(v) XOR mu. Every literal of a clause that sigma violates is false under sigma,
    so all three are true under the complement."""
    viol = np.all(sigma[vs] != sg, axis=1).astype(np.int8)

    def p1(j, gen):
        mu = viol[j]
        return sigma[vs[j]] ^ mu[:, None], mu

    def p2(v, mu, gen):
        return sigma[v] ^ mu
    return p1, p2


def independent_random(vs, sg):
    """P1 answers a uniformly random satisfying assignment of the clause (fresh each round); P2
    answers a uniformly random bit. P2's bit is independent of P1's, so the cross-check passes
    with probability exactly 1/2."""
    local = all_assignments(3)                 # the 8 local assignments

    def p1(j, gen):
        R = len(j)
        a = np.empty((R, 3), dtype=np.int8)
        todo = np.arange(R)
        while len(todo):                       # rejection sampling: 7 of 8 local assignments work
            cand = local[gen.integers(0, 8, size=len(todo))]
            good = satisfies(cand, sg[j[todo]])
            a[todo[good]] = cand[good]
            todo = todo[~good]
        return a, np.zeros(R, dtype=np.int8)

    def p2(v, mu, gen):
        return gen.integers(0, 2, size=len(v)).astype(np.int8)
    return p1, p2


def block_rates(acc, ts, expected):
    """Acceptance after t sequential rounds, measured on disjoint blocks of t rounds; points whose
    expected number of accepted blocks is below 20 are dropped."""
    rows = []
    for t in ts:
        B = len(acc) // t
        if B * expected(t) < 20:
            continue
        blocks = acc[:B * t].reshape(B, t).all(axis=1)
        k = int(blocks.sum())
        lo, hi = pk.clopper_pearson(k, B)
        rate = k / B
        rows.append((t, rate, hi - rate, rate - lo, B))
    return rows


def main():
    # ---------------------------------------------------------------- phi8 and psi7
    vs8, sg8 = phi8()
    A3 = all_assignments(3)
    viol8 = violated(vs8, sg8, A3)
    assert np.all(viol8.sum(axis=1) == 1), "every assignment violates exactly one clause of phi8"
    m8, k8 = 8, 1
    sigma0 = np.zeros(3, dtype=np.int8)                       # violates (x1 or x2 or x3)
    keep = ~np.all(sg8 == 0, axis=1)                          # drop (not x1 or not x2 or not x3)
    vs7, sg7 = vs8[keep], sg8[keep]
    sat7 = ~violated(vs7, sg7, A3).any(axis=1)
    assert sat7.sum() == 1 and np.all(A3[sat7][0] == 1), "psi7 is satisfied by 111 only"
    sigma111 = np.ones(3, dtype=np.int8)

    print("== phi8: m = 8 clauses on 3 variables, k = 1; psi7: phi8 minus (~x1 v ~x2 v ~x3)")
    strategies8 = {
        "best": (best_isolated(vs8, sg8, sigma0), 1 - k8 / (3 * m8), False, 11),
        "fixed": (from_assignment(vs8, sigma0), 1 - k8 / m8, False, 12),
        "random": (independent_random(vs8, sg8), 0.5, False, 13),
        "channel": (one_bit_channel(vs8, sg8, sigma0), 1.0, True, 14),
    }
    acc = {}
    for name, ((p1, p2), exact, ch, seed) in strategies8.items():
        a = play(vs8, sg8, p1, p2, pk.rng(seed), ROUNDS, channel=ch)
        acc[name] = a
        lo, hi = pk.clopper_pearson(int(a.sum()), ROUNDS)
        print(f"  phi8 {name:8s}: per-round acceptance {a.mean():.4f}  [{lo:.4f}, {hi:.4f}]  exact {exact:.4f}")
    p1, p2 = from_assignment(vs7, sigma111)
    a = play(vs7, sg7, p1, p2, pk.rng(15), ROUNDS)
    acc["honest"] = a
    print(f"  psi7 honest  : per-round acceptance {a.mean():.4f} (exact 1)")

    # ---------------------------------------------------------------- a random unsatisfiable formula
    n, m = 12, 60
    An = all_assignments(n)
    for seed in range(100, 200):
        vsr, sgr = random_formula(n, m, pk.rng(seed))
        counts = violated(vsr, sgr, An).sum(axis=1)
        if counts.min() >= 1:
            break
    k = int(counts.min())
    nbest = int((counts == k).sum())
    sigma_r = An[int(np.argmin(counts))]
    print(f"== random formula: n = {n}, m = {m}, seed {seed}: unsatisfiable, k = {k} "
          f"(attained by {nbest} assignments); value 1 - k/(3m) = {1 - k / (3 * m):.5f}, "
          f"fixed assignment 1 - k/m = {1 - k / m:.5f}")
    p1, p2 = best_isolated(vsr, sgr, sigma_r)
    a = play(vsr, sgr, p1, p2, pk.rng(21), ROUNDS)
    acc["rbest"] = a
    lo, hi = pk.clopper_pearson(int(a.sum()), ROUNDS)
    print(f"  random best : per-round acceptance {a.mean():.5f}  [{lo:.5f}, {hi:.5f}]")
    p1, p2 = one_bit_channel(vsr, sgr, sigma_r)
    a = play(vsr, sgr, p1, p2, pk.rng(22), ROUNDS, channel=True)
    print(f"  random one-bit channel: per-round acceptance {a.mean():.5f}")

    # ---------------------------------------------------------------- honest provers that do not coordinate
    ns, ms = 12, 30
    for seed in range(300, 400):
        vss, sgs = random_formula(ns, ms, pk.rng(seed))
        sat = ~violated(vss, sgs, An).any(axis=1)
        if sat.sum() >= 2:
            break
    sols = An[sat]
    gen = pk.rng(31)
    pair = gen.choice(len(sols), 2, replace=False)
    s1, s2 = sols[pair[0]], sols[pair[1]]
    d = (s1[vss] != s2[vss]).sum(axis=1)                       # disagreements inside each clause
    exact_unc = 1 - d.sum() / (3 * ms)

    def p1u(j, g):
        return s1[vss[j]], np.zeros(len(j), dtype=np.int8)

    def p2u(v, mu, g):
        return s2[v]
    a = play(vss, sgs, p1u, p2u, pk.rng(32), ROUNDS)
    lo, hi = pk.clopper_pearson(int(a.sum()), ROUNDS)
    print(f"== satisfiable formula: n = {ns}, m = {ms}, seed {seed}, {int(sat.sum())} satisfying assignments; "
          f"two of them differ in {int((s1 != s2).sum())} variables")
    print(f"  uncoordinated honest pair: per-round acceptance {a.mean():.4f} [{lo:.4f}, {hi:.4f}], "
          f"exact 1 - sum_j d_j/(3m) = {exact_unc:.4f}")

    # ---------------------------------------------------------------- repetition: blocks of t rounds
    curves = {
        "best": lambda t: (1 - 1 / 24) ** t,
        "fixed": lambda t: (7 / 8) ** t,
        "honest": lambda t: 1.0,
        "channel": lambda t: 1.0,
        "rbest": lambda t: (1 - k / (3 * m)) ** t,
    }
    for name, f in curves.items():
        rows = block_rates(acc[name], TS, f)
        pk.write_dat(f"{OUT}/08-crossexam-clause-{name}.dat", ["t", "rate", "eplus", "eminus", "blocks"], rows)
        print(f"  {name:7s} after t rounds: " + ", ".join(f"t={r[0]}: {r[1]:.4f}" for r in rows))
        miss = [(r[0], round(r[1] - r[3], 5), round(r[1] + r[2], 5), round(f(r[0]), 5)) for r in rows
                if not (r[1] - r[3] - 1e-12 <= f(r[0]) <= r[1] + r[2] + 1e-12)]
        print(f"          95% intervals that miss the curve (t, lo, hi, curve): {miss if miss else 'none'}")
    grid = np.arange(1, 141)
    pk.write_dat(f"{OUT}/08-crossexam-clause-curves.dat", ["t", "bound", "fixed", "rbound"],
                 [(int(t), (23 / 24) ** t, (7 / 8) ** t, (1 - k / (3 * m)) ** t) for t in grid])

    # ---------------------------------------------------------------- costs and repetitions
    for eps in (0.01,):
        tb = int(np.ceil(np.log(eps) / np.log(23 / 24)))
        tf = int(np.ceil(np.log(eps) / np.log(7 / 8)))
        tr = int(np.ceil(np.log(eps) / np.log(1 - k / (3 * m))))
        print(f"== rounds for soundness error {eps}: phi8 cross-examination {tb}; phi8 against a fixed "
              f"assignment {tf}; random formula {tr}")
    print(f"== one round on the random formula: {np.log2(m):.2f} + {np.log2(3):.2f} random bits, "
          f"questions of {int(np.ceil(np.log2(m)))} and {int(np.ceil(np.log2(n)))} bits, 4 answer bits; "
          f"reading a whole assignment would take {n} bits")
    print(f"done in {time.time() - T0:.1f} s (after start-up)")


if __name__ == "__main__":
    main()
