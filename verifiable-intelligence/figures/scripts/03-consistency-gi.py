"""Chapter 3 (Consistency): the Blum-Kannan checker for (vertex-colored) graph isomorphism, run
against an honest program and two cheating programs.

Colored graphs on n = 8 vertices (two colors); an isomorphism must preserve colors and edges.
The checker on input (G, H):
  if the program says YES: find a candidate isomorphism by self-reduction (mark vertex i of G and
      a candidate vertex v of H with a fresh color, keep v if the program says YES), then check it;
  if the program says NO: repeat t times: draw b and a random permutation pi, K = pi(G) or pi(H),
      ask the program about (G, K), and reject if its answer does not reveal b.
Programs:
  honest     decides colored isomorphism exactly (backtracking over color- and degree-preserving maps)
  literal    says YES only when the two labeled graphs are identical: wrong ("no") on every
             isomorphic pair that is labeled differently; in the NO branch it matches b with
             probability exactly 1/2 per round, like every program (theory 2^{-t})
  invariant  says YES when the multisets of (color, degree) agree: wrong ("yes") on the cube Q3
             against the Wagner graph (both 3-regular on 8 vertices); caught with probability 1
Writes figures/data/03-consistency-gi.dat
Run from the volume root:  python3 figures/scripts/03-consistency-gi.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk

N_VERT = 8
RUNS = 100_000
T_MAX = 14


def permute(adj, col, pi):
    """The colored graph with vertex u renamed pi[u]."""
    n = len(pi)
    inv = np.empty(n, dtype=np.int64)
    inv[pi] = np.arange(n)
    return adj[np.ix_(inv, inv)], tuple(col[inv[v]] for v in range(n))


def honest(G, K):
    """Exact test for an isomorphism of colored graphs: extend a partial map vertex by vertex,
    trying only targets with the same color and degree and keeping adjacency consistent."""
    (a1, c1), (a2, c2) = G, K
    n = len(c1)
    if sorted(map(str, c1)) != sorted(map(str, c2)):
        return False
    d1, d2 = a1.sum(axis=1), a2.sum(axis=1)
    phi, used = [-1] * n, [False] * n

    def extend(u):
        if u == n:
            return True
        for v in range(n):
            if used[v] or c2[v] != c1[u] or d2[v] != d1[u]:
                continue
            if all(a1[u, w] == a2[v, phi[w]] for w in range(u)):
                phi[u], used[v] = v, True
                if extend(u + 1):
                    return True
                phi[u], used[v] = -1, False
        return False

    return extend(0)


def literal(G, K):
    return bool(np.array_equal(G[0], K[0]) and G[1] == K[1])


def invariant(G, K):
    def inv(X):
        deg = X[0].sum(axis=1)
        return sorted((str(c), int(d)) for c, d in zip(X[1], deg))
    return inv(G) == inv(K)


class Counted:
    """A program together with a count of the calls made to it."""
    def __init__(self, prog):
        self.prog, self.calls = prog, 0

    def __call__(self, G, K):
        self.calls += 1
        return self.prog(G, K)


def mark(X, v, tag):
    adj, col = X
    col = list(col)
    col[v] = (col[v], tag)           # refine the color: the old color is kept
    return adj, tuple(col)


def check_yes_branch(prog, G, H):
    """Self-reduce with the program, then verify the candidate map. True = CORRECT."""
    n = len(G[1])
    Gc, Hc, phi, used = G, H, {}, set()
    for i in range(n):
        for v in range(n):
            if v in used:
                continue
            Gt, Ht = mark(Gc, i, ("fresh", i)), mark(Hc, v, ("fresh", i))
            if prog(Gt, Ht):
                Gc, Hc, phi[i] = Gt, Ht, v
                used.add(v)
                break
        else:
            return False                 # no candidate accepted: BUGGY
    pi = np.array([phi[i] for i in range(n)])
    return bool(np.array_equal(permute(G[0], G[1], pi)[0], H[0]) and permute(G[0], G[1], pi)[1] == H[1])


def rounds_passed_no_branch(prog, G, H, t_max, gen):
    """Rounds of the NO branch passed before the first mismatch (capped at t_max)."""
    n = len(G[1])
    for k in range(t_max):
        b = int(gen.integers(0, 2))
        pi = gen.permutation(n)
        K = permute(*(G if b == 0 else H), pi)
        if prog(G, K) != (b == 0):        # the program's answer must reveal b
            return k
    return t_max


def checker(prog, G, H, t, gen):
    if prog(G, H):
        return check_yes_branch(prog, G, H)
    return rounds_passed_no_branch(prog, G, H, t, gen) == t


def random_colored_graph(n, gen):
    adj = np.triu(gen.integers(0, 2, size=(n, n)), 1)
    adj = adj + adj.T
    col = tuple(int(c) for c in gen.integers(0, 2, size=n))
    return adj.astype(np.int64), col


def main():
    gen = pk.rng(31)
    n = N_VERT
    G = random_colored_graph(n, gen)
    H = permute(*G, gen.permutation(n))
    assert honest(G, H) and not literal(G, H)       # the input that matters: G = H, "literal" says no

    # wrong "no": rounds of the NO branch passed by the literal program, run by run
    counts = np.array([rounds_passed_no_branch(literal, G, H, T_MAX, gen) for _ in range(RUNS)])
    rows = []
    print(f"GI checker, n = {n}: wrong 'no' (literal program), {RUNS} runs, stop at first mismatch")
    for t in range(1, T_MAX + 1):
        k = int(np.sum(counts >= t))
        lo, hi = pk.clopper_pearson(k, RUNS)
        print(f"  t = {t:2d}: accepted {k / RUNS:.6f}   theory 2^-t = {2.0**-t:.6f}")
        if k >= 3:
            rows.append((t, k / RUNS, k / RUNS - lo, hi - k / RUNS, 2.0 ** -t))
    pk.write_dat("figures/data/03-consistency-gi.dat", ["t", "acc", "em", "ep", "bound"], rows)

    # wrong "yes": the degree heuristic on the cube against the Wagner graph (both 3-regular)
    cube = np.array([[int(bin(u ^ v).count("1") == 1) for v in range(8)] for u in range(8)], dtype=np.int64)
    wagner = np.array([[int((u - v) % 8 in (1, 4, 7)) for v in range(8)] for u in range(8)], dtype=np.int64)
    one = tuple([0] * 8)
    Q, W = (cube, one), (wagner, one)
    assert not honest(Q, W) and invariant(Q, W)
    acc_yes = sum(checker(invariant, Q, W, 10, gen) for _ in range(1000))
    print(f"wrong 'yes' (degree heuristic, cube vs Wagner graph): accepted {acc_yes} of 1000 runs")

    # honest program: accepted on isomorphic and on non-isomorphic pairs; cost in program calls
    acc_iso = acc_non = 0
    calls_yes, calls_no = [], []
    hg = pk.rng(32)
    for _ in range(300):
        G1 = random_colored_graph(n, hg)
        H1 = permute(*G1, hg.permutation(n))
        p = Counted(honest)
        acc_iso += checker(p, G1, H1, 10, hg)
        calls_yes.append(p.calls)
        while True:
            H2 = random_colored_graph(n, hg)
            if not honest(G1, H2):
                break
        p = Counted(honest)
        acc_non += checker(p, G1, H2, 10, hg)
        calls_no.append(p.calls)
    print(f"honest program: accepted {acc_iso} of 300 isomorphic pairs and {acc_non} of 300 "
          f"non-isomorphic pairs (t = 10)")
    print(f"program calls: YES branch mean {np.mean(calls_yes):.1f} (max {max(calls_yes)}, bound 1 + n(n+1)/2 = "
          f"{1 + n * (n + 1) // 2}); NO branch {calls_no[0]} (= t + 1)")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nruntime {time.perf_counter() - t0:.1f} s")
