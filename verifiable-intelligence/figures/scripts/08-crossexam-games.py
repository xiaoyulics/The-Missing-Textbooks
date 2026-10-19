"""Chapter 8 (Cross-Examination): small two-prover games, their exact values and measured rates.

A one-round game: the verifier draws questions (x, y) from a distribution, sends x to P1 and y to
P2, and accepts the answers (a, b) if a predicate Q(x, y, a, b) holds. Its value is the largest
acceptance probability of two isolated provers; by averaging, deterministic strategies attain it,
so it is computed exactly by searching over P2's deterministic strategies with P1's best response.

Games:
  CHSH       x, y uniform bits; accept iff a XOR b = x AND y.          value 3/4
  CHSH^2     two copies in parallel (questions and answers are pairs). value 5/8 > (3/4)^2
  Fortnow    x, y uniform bits; accept iff (x OR a) != (y OR b).        value 1/2
  Fortnow^2  two copies in parallel.                                     value 3/8 > (1/2)^2
  Variant    as Fortnow, questions uniform on {00, 01, 10}.             value 2/3
  Variant^2  two copies in parallel.                                     value 2/3: no decrease
and the entangled strategy for CHSH: the provers share (|00> + |11>)/sqrt 2 and measure in bases
rotated by angles 0, pi/4 (P1) and pi/8, -pi/8 (P2); the joint distribution of answers is computed
from the state with 4 x 4 matrices (the rules of quantum measurement are used, not derived), and
the game is then played by sampling answers from it. Its winning probability is cos^2(pi/8).

Each optimal strategy is also played 200,000 times (seeded) to measure its acceptance rate.

Run from the volume root:  python3 figures/scripts/08-crossexam-games.py
Writes figures/data/08-crossexam-games.dat and prints every number the chapter quotes.
"""
import itertools
import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import protokit as pk

T0 = time.time()

OUT = "figures/data"
PLAYS = 200_000


def make_game(Xs, Ys, As, Bs, prob, pred):
    """Payoff tensor W[x, y, a, b] (0/1) and question probabilities P[x, y] (Fractions)."""
    W = np.zeros((len(Xs), len(Ys), len(As), len(Bs)), dtype=np.int64)
    for (xi, x), (yi, y), (ai, a), (bi, b) in itertools.product(enumerate(Xs), enumerate(Ys), enumerate(As), enumerate(Bs)):
        W[xi, yi, ai, bi] = 1 if pred(x, y, a, b) else 0
    P = [[prob(x, y) for y in Ys] for x in Xs]
    return W, P


def exact_value(W, P):
    """max over deterministic strategies, exactly: enumerate P2's strategies b(y) (all |B|^|Y|),
    best response of P1 for each question x. Weights are integers over a common denominator."""
    den = int(np.lcm.reduce([f.denominator for row in P for f in row]))
    Pint = np.array([[int(f * den) for f in row] for row in P], dtype=np.int64)
    nx, ny, na, nb = W.shape
    S = np.array(list(itertools.product(range(nb), repeat=ny)), dtype=np.int64)    # (S, ny)
    tot = np.zeros((len(S), nx, na), dtype=np.int64)
    for y in range(ny):
        # W[x, y, a, b_s(y)] weighted by P[x, y]
        tot += (W[:, y, :, :][:, :, S[:, y]] * Pint[:, y][:, None, None]).transpose(2, 0, 1)
    vals = tot.max(axis=2).sum(axis=1)
    s = int(np.argmax(vals))
    b_strategy = S[s]
    a_strategy = tot[s].argmax(axis=1)
    return Fraction(int(vals[s]), den), a_strategy, b_strategy


def play(W, P, a_str, b_str, seed):
    """Play the game PLAYS times with deterministic strategies; return (rate, lo, hi)."""
    gen = pk.rng(seed)
    nx, ny = W.shape[:2]
    p = np.array([[float(f) for f in row] for row in P]).ravel()
    q = gen.choice(nx * ny, size=PLAYS, p=p)
    x, y = q // ny, q % ny
    acc = W[x, y, a_str[x], b_str[y]]
    k = int(acc.sum())
    lo, hi = pk.clopper_pearson(k, PLAYS)
    return k / PLAYS, lo, hi


def entangled_chsh():
    """Joint distribution p[x, y, a, b] for the CHSH strategy on (|00>+|11>)/sqrt 2."""
    phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    th = {0: 0.0, 1: np.pi / 4}
    ph = {0: np.pi / 8, 1: -np.pi / 8}

    def basis(t):     # outcome 0: (cos t, sin t); outcome 1: (-sin t, cos t)
        return [np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])]
    p = np.zeros((2, 2, 2, 2))
    for x, y, a, b in itertools.product(range(2), repeat=4):
        v = np.kron(basis(th[x])[a], basis(ph[y])[b])
        p[x, y, a, b] = np.dot(v, phi) ** 2
    return p


def main():
    bits = [0, 1]
    pairs = list(itertools.product(bits, bits))
    unif2 = lambda x, y: Fraction(1, 4)
    unif4 = lambda x, y: Fraction(1, 16)
    chsh = lambda x, y, a, b: (a ^ b) == (x & y)
    fort = lambda x, y, a, b: (x | a) != (y | b)
    var_p = lambda x, y: Fraction(0) if (x, y) == (1, 1) else Fraction(1, 3)
    var_p2 = lambda x, y: var_p(x[0], y[0]) * var_p(x[1], y[1])
    two = lambda f: (lambda x, y, a, b: f(x[0], y[0], a[0], b[0]) and f(x[1], y[1], a[1], b[1]))

    games = [
        ("CHSH", make_game(bits, bits, bits, bits, unif2, chsh), Fraction(3, 4)),
        ("CHSH^2", make_game(pairs, pairs, pairs, pairs, unif4, two(chsh)), Fraction(9, 16)),
        ("Fortnow", make_game(bits, bits, bits, bits, unif2, fort), Fraction(1, 2)),
        ("Fortnow^2", make_game(pairs, pairs, pairs, pairs, unif4, two(fort)), Fraction(1, 4)),
        ("Variant", make_game(bits, bits, bits, bits, var_p, fort), Fraction(2, 3)),
        ("Variant^2", make_game(pairs, pairs, pairs, pairs, var_p2, two(fort)), Fraction(4, 9)),
    ]
    rows = []
    print("== exact values (exhaustive search over deterministic strategies) and measured rates")
    for idx, (name, (W, P), square) in enumerate(games):
        v, a_s, b_s = exact_value(W, P)
        rate, lo, hi = play(W, P, a_s, b_s, seed=60 + idx)
        tag = "  (square of the one-copy value: %s)" % square if "^2" in name else ""
        inside = "contains" if lo <= float(v) <= hi else "MISSES"
        print(f"  {name:10s} value {str(v):5s} = {float(v):.4f}; optimal pair measured {rate:.4f} "
              f"[{lo:.5f}, {hi:.5f}], which {inside} the value{tag}")
        rows.append((name.replace("^2", "2"), float(v), rate, hi - rate, rate - lo))

    # the variant's swap strategy (a1, a2) = (x2, x1), (b1, b2) = (y2, y1)
    W, P = games[5][1]
    swap = np.array([pairs.index((x[1], x[0])) for x in pairs])
    rate, lo, hi = play(W, P, swap, swap, seed=70)
    exact = sum((P[xi][yi] for xi in range(4) for yi in range(4) if W[xi, yi, swap[xi], swap[yi]]), Fraction(0))
    print(f"  Variant^2 swap strategy: exact {exact} measured {rate:.4f} [{lo:.4f}, {hi:.4f}]")

    # the entangled strategy for CHSH
    p = entangled_chsh()
    assert np.allclose(p.sum(axis=(2, 3)), 1)
    marg_a = p.sum(axis=3)            # (x, y, a): P1's answer distribution
    marg_b = p.sum(axis=2)            # (x, y, b)
    assert np.allclose(marg_a, 0.5) and np.allclose(marg_b, 0.5), "each answer alone is a uniform bit"
    win = np.mean([sum(p[x, y, a, b] for a in range(2) for b in range(2) if (a ^ b) == (x & y))
                   for x in range(2) for y in range(2)])
    print(f"== entangled CHSH: winning probability {win:.6f}; cos^2(pi/8) = {np.cos(np.pi / 8) ** 2:.6f} "
          f"= (2 + sqrt 2)/4 = {(2 + np.sqrt(2)) / 4:.6f}; marginals uniform (no signaling)")
    gen = pk.rng(80)
    x = gen.integers(0, 2, PLAYS)
    y = gen.integers(0, 2, PLAYS)
    probs = p[x, y].reshape(PLAYS, 4)
    u = gen.random(PLAYS)
    ab = (u[:, None] > np.cumsum(probs, axis=1)).sum(axis=1)
    ab = np.minimum(ab, 3)
    a, b = ab // 2, ab % 2
    k = int(((a ^ b) == (x & y)).sum())
    lo, hi = pk.clopper_pearson(k, PLAYS)
    inside = "contains" if lo <= np.cos(np.pi / 8) ** 2 <= hi else "MISSES"
    print(f"  sampled {PLAYS} plays: {k / PLAYS:.4f} [{lo:.5f}, {hi:.5f}], which {inside} cos^2(pi/8)")
    rows.append(("CHSHq", float(np.cos(np.pi / 8) ** 2), k / PLAYS, hi - k / PLAYS, k / PLAYS - lo))
    print(f"  two independent entangled copies: cos^4(pi/8) = {np.cos(np.pi / 8) ** 4:.4f} > 5/8 = 0.6250")
    pk.write_dat(f"{OUT}/08-crossexam-games.dat", ["game", "value", "measured", "eplus", "eminus"], rows)
    print(f"done in {time.time() - T0:.1f} s (after start-up)")


if __name__ == "__main__":
    main()
