#!/usr/bin/env python3
"""Chapter 7 (debate): debates on random game trees between debaters of bounded lookahead.

The game is the debate of the chapter with D = 12 one-bit statements: debater A, who claims that
the root of a complete binary game tree is a win for A, moves at even depths 0, 2, ..., 10, and
debater B at odd depths; the judge reads the leaf that the debate reaches and declares A the winner
if it is 1. The leaves are independent Bernoulli(p) bits, with p = 0.6090 chosen so that the root is
a win for A with probability exactly 1/2. The true side of a tree (the honest debater) is the side
that has a winning strategy, found by backward induction; the other side is the liar.

A debater with lookahead h plays minimax-optimally when at most h moves remain (it then sees the
leaves) and uniformly at random before that. This is depth-limited minimax search with an
evaluation that knows only the prior: every position at the search horizon has the same prior
value, so all moves look alike until the leaves come into view. Ties between optimal moves are
broken uniformly at random. Three pairings are run for every h = 0, ..., D, each on its own N
fresh random trees (seeded by SEED_TREES, h and the pairing):

  both     the honest debater and the liar both have lookahead h;
  weak     the honest debater has lookahead h, the liar full lookahead D;
  strong   the honest debater has full lookahead D, the liar lookahead h. By backward induction
           (Theorem 7.3), the honest debater always wins.

The measured quantity is the judge's accuracy: the fraction of debates won by the true side, with
Clopper-Pearson 95% intervals. The exact accuracy comes from a recursion over the joint law of
(value, outcome) of a random subtree (Exercise 7.2): the two subtrees of a node are
independent and identically distributed, so the law at a node is a function of the law one level
down.

Writes figures/data/07-debate-trees.dat (measured, one row per h) and
figures/data/07-debate-trees-theory.dat (exact). Prints every number the chapter quotes.
Run from the volume root:  python3 figures/scripts/07-debate-trees.py   (about 5 s on a laptop)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

D = 12                     # number of statements (depth of the tree)
N = 4000                   # fresh random trees for each pairing and each lookahead
SEED_TREES, SEED_PLAY = 7, 707
OUT = "figures/data/07-debate-trees.dat"
OUT_THEORY = "figures/data/07-debate-trees-theory.dat"


# ----------------------------------------------------------------------------- the random game
def root_win_prob(p, depth):
    """Probability that the root is a win for A when leaves are Bernoulli(p): A nodes (even depth)
    are wins if some child is, B nodes (odd depth) if both children are."""
    x = p
    for k in range(depth - 1, -1, -1):
        x = 1 - (1 - x) ** 2 if k % 2 == 0 else x ** 2
    return x


def balanced_p(depth):
    """The leaf probability for which the root is a win for A with probability 1/2 (bisection)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if root_win_prob(mid, depth) < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ----------------------------------------------------------------------------- exact accuracy
def exact_joint(p, depth, h_a, h_b):
    """Joint law q[v, o] of the minimax value v and the outcome o (the leaf reached) of the debate
    on a random tree, when A has lookahead h_a and B has lookahead h_b."""
    q = np.array([[1 - p, 0.0], [0.0, p]])           # a leaf: value = outcome = its bit
    for k in range(depth - 1, -1, -1):
        remaining = depth - k
        a_moves = (k % 2 == 0)
        h = h_a if a_moves else h_b
        sees = remaining <= h
        new = np.zeros((2, 2))
        for v1 in (0, 1):
            for o1 in (0, 1):
                for v2 in (0, 1):
                    for o2 in (0, 1):
                        w = q[v1, o1] * q[v2, o2]
                        if w == 0.0:
                            continue
                        v = max(v1, v2) if a_moves else min(v1, v2)
                        # a mover that sees the leaves picks a child whose value equals the node's
                        # value (the best it can get), uniformly among such children
                        cands = [c for c, vc in enumerate((v1, v2)) if vc == v] if sees else [0, 1]
                        for c in cands:
                            new[v, (o1, o2)[c]] += w / len(cands)
        q = new
    return q


def exact_accuracy(p, depth, h_honest, h_liar):
    """Probability that the true side wins: on trees with root value 1 the honest side is A."""
    q1 = exact_joint(p, depth, h_honest, h_liar)     # A honest
    q0 = exact_joint(p, depth, h_liar, h_honest)     # B honest
    return q1[1, 1] + q0[0, 0]


# ----------------------------------------------------------------------------- simulation
def make_trees(p, depth, n, gen):
    """Leaves of n random trees and the minimax values of all their nodes, level by level."""
    leaves = (gen.random((n, 2 ** depth)) < p).astype(np.int8)
    vals = [None] * (depth + 1)
    vals[depth] = leaves
    for k in range(depth - 1, -1, -1):
        pairs = vals[k + 1].reshape(n, 2 ** k, 2)
        vals[k] = pairs.max(axis=2) if k % 2 == 0 else pairs.min(axis=2)
    return vals


def play(vals, depth, h_honest, h_liar, gen):
    """Play one debate on every tree; return a boolean array: did the true side win?"""
    n = vals[0].shape[0]
    root = vals[0][:, 0]
    rows = np.arange(n)
    idx = np.zeros(n, dtype=np.int64)
    for k in range(depth):
        remaining = depth - k
        a_moves = (k % 2 == 0)
        mover_honest = (root == 1) if a_moves else (root == 0)
        h = np.where(mover_honest, h_honest, h_liar)
        sees = remaining <= h
        c0 = vals[k + 1][rows, 2 * idx]
        c1 = vals[k + 1][rows, 2 * idx + 1]
        coin = gen.integers(0, 2, size=n)
        want = 1 if a_moves else 0                    # the value the mover hopes for
        good0, good1 = (c0 == want), (c1 == want)
        # optimal: a child with the wanted value if there is one (ties at random), else either
        opt = np.where(good0 & ~good1, 0, np.where(good1 & ~good0, 1, coin))
        choice = np.where(sees, opt, coin)
        idx = 2 * idx + choice
    outcome = vals[depth][rows, idx]
    return outcome == root


def main():
    t0 = time.time()
    p = balanced_p(D)
    print(f"depth D = {D}, leaf probability p = {p:.6f}, P[root is a win for A] = {root_win_prob(p, D):.6f}")
    gen_play = pk.rng(SEED_PLAY)
    pairings = {"both": lambda h: (h, h), "weak": lambda h: (h, D), "strong": lambda h: (D, h)}
    rows, theory = [], []
    inside = total = 0
    a_wins = 0
    for h in range(D + 1):
        row = [h]
        trow = [h, D - h]
        for j, (name, hv) in enumerate(pairings.items()):
            hh, hl = hv(h)
            vals = make_trees(p, D, N, pk.rng(SEED_TREES * 1000 + 10 * h + j))
            a_wins += int(vals[0][:, 0].sum())
            won = play(vals, D, hh, hl, gen_play)
            k = int(won.sum())
            rate = k / N
            lo, hi = pk.clopper_pearson(k, N)
            ex = exact_accuracy(p, D, hh, hl)
            row += [rate, rate - lo, hi - rate]
            trow.append(ex)
            if 1e-9 < ex < 1 - 1e-9:
                total += 1
                inside += (lo <= ex <= hi)
            print(f"h={h:2d} {name:6s}: honest wins {k:4d}/{N} = {rate:.4f}  [{lo:.4f}, {hi:.4f}]  exact {ex:.4f}"
                  + ("" if lo <= ex <= hi else "   <- exact value outside the interval"))
        rows.append(row)
        theory.append(trow)
    print(f"root is a win for A in {a_wins} of {N * 3 * (D + 1)} trees; "
          f"{inside} of {total} intervals (exact value strictly between 0 and 1) contain the exact value")
    header = ["h"]
    for name in pairings:
        header += [name, name + "em", name + "ep"]
    pk.write_dat(OUT, header, rows)
    pk.write_dat(OUT_THEORY, ["h", "adv", "both", "weak", "strong"], theory)

    # numbers quoted in the text
    print()
    for h in (0, 4, 8, 10, 11):
        print(f"exact accuracy at h={h}: both {exact_accuracy(p, D, h, h):.4f}, "
              f"weak {exact_accuracy(p, D, h, D):.4f}, strong {exact_accuracy(p, D, D, h):.4f}")
    # the same curves depend on D - h only: check against a shallower game
    for d2 in (8, 10):
        p2 = balanced_p(d2)
        print(f"D={d2}, p={p2:.4f}: weak at h=D-4 {exact_accuracy(p2, d2, d2 - 4, d2):.4f}, "
              f"both at h=D-4 {exact_accuracy(p2, d2, d2 - 4, d2 - 4):.4f}")
    print(f"done in {time.time() - t0:.1f} s; wrote {OUT} and {OUT_THEORY}")


if __name__ == "__main__":
    main()
