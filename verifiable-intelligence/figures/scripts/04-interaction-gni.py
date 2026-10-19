"""04-interaction-gni.py -- Chapter 4 (Interaction): private coins for graph non-isomorphism.

Runs Protocol 4.1 (the private-coin protocol of Goldreich, Micali and Wigderson) on random
graphs with n = 8 vertices, against honest and cheating provers, and checks the theory:

  (a) rounds: acceptance after t sequential rounds, t = 1..10, of the honest prover on
      non-isomorphic pairs and of cheating provers on isomorphic pairs -- a guesser; the honest
      algorithm (it answers 0, since every challenge is isomorphic to G_0); a labeled-distance
      heuristic; a prover that learns each secret bit with probability 1/2; a prover that sees
      the verifier's coins.  Theory: 1, 2^-t (every prover that cannot see the coins), (3/4)^t, 1.
  (b) disguise: a verifier that relabels G_b by s random transpositions instead of a uniform
      permutation (each step swaps two labels drawn independently and uniformly, and does nothing
      when they coincide).  The Bayes-optimal prover against the exact value (1 + TV_s)/2 (computed by
      propagating the two distributions over the 8! labeled copies), and a labeled-distance prover.
  (c) reuse: a verifier that draws every session's challenge from a fixed pool of N challenges
      and announces its verdict; a prover that memorizes what it learns.  Acceptance in session
      j against 1 - (1/2)(1 - 1/N)^(j-1).
  (d) the recursion of the proof that IP is in PSPACE, run on Protocol 4.1 with n = 4: the
      largest acceptance probability over all provers, for t = 1 and t = 2 parallel copies.
  (e) costs against n: the verifier's work per round and the honest prover's brute-force work.

Run from the volume root:  python3 figures/scripts/04-interaction-gni.py
Writes figures/data/04-interaction-gni-*.dat and prints every number the chapter quotes.
Deterministic (seeds below), apart from the wall-clock times of (e); a few seconds on a laptop CPU.
"""
import itertools
import math
import os
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

OUT = "figures/data"
N_VERT = 8


# ----------------------------------------------------------------------------------------------
# Graphs on vertex set {0..n-1}: boolean adjacency matrices; a labeled graph is encoded by the
# integer whose bits are the entries above the diagonal.  relabel(A, q) = A[q][:, q] is the copy
# of A in which vertex i plays the role of vertex q[i]; for q uniform it is a uniform relabeling.
# ----------------------------------------------------------------------------------------------
class Graphs:
    def __init__(self, n):
        self.n = n
        self.iu = np.triu_indices(n, 1)
        self.w = (np.int64(1) << np.arange(len(self.iu[0]), dtype=np.int64))
        self.perms = np.array(list(itertools.permutations(range(n))), dtype=np.int64)

    def code(self, A):
        """Codes of one adjacency matrix (n,n) or of a stack (..., n, n)."""
        bits = A[..., self.iu[0], self.iu[1]].astype(np.int64)
        return bits @ self.w

    def relabel_all(self, A):
        """All n! relabelings of A, as a stack (n!, n, n)."""
        P = self.perms
        return A[P[:, :, None], P[:, None, :]]

    def iso_class(self, A):
        """The sorted array of codes of the labeled copies of A (its isomorphism class)."""
        return np.unique(self.code(self.relabel_all(A)))

    def random_graph(self, gen, p=0.5):
        U = np.triu(gen.random((self.n, self.n)) < p, 1)
        return U | U.T

    def relabel(self, A, q):
        return A[np.ix_(q, q)]


def edge_count(A):
    return int(A.sum() // 2)


def make_pairs(G, gen, m):
    """m non-isomorphic pairs with equal edge counts, and m isomorphic pairs (G_1 a uniform
    relabeling of G_0, different from G_0 as a labeled graph)."""
    noniso, iso = [], []
    while len(noniso) < m:
        A0, A1 = G.random_graph(gen), G.random_graph(gen)
        if edge_count(A0) != edge_count(A1):
            continue
        c0, c1 = G.iso_class(A0), G.iso_class(A1)
        if c0[0] == c1[0]:
            continue
        noniso.append((A0, A1, c0))
    while len(iso) < m:
        A0 = G.random_graph(gen)
        A1 = G.relabel(A0, gen.permutation(G.n))
        if G.code(A0) == G.code(A1):
            continue
        iso.append((A0, A1, G.iso_class(A0)))
    return noniso, iso


def interval_cols(k, n):
    """rate, plus offset, minus offset (Clopper-Pearson 95%)."""
    lo, hi = pk.clopper_pearson(int(k), int(n))
    r = k / n
    return r, hi - r, r - lo


# ----------------------------------------------------------------------------------------------
# (a) acceptance against the number of sequential rounds
# ----------------------------------------------------------------------------------------------
def challenges(G, pairs, pair_id, b, gen):
    """One round of challenges for a batch: H = uniform relabeling of G_b of the given pair."""
    R = len(pair_id)
    stack = np.stack([np.stack([p[0], p[1]]) for p in pairs])  # (m, 2, n, n)
    Ab = stack[pair_id, b]  # (R, n, n)
    q = np.argsort(gen.random((R, G.n)), axis=1)  # uniform permutations
    H = Ab[np.arange(R)[:, None, None], q[:, :, None], q[:, None, :]]
    return G.code(H)


def in_class(codes, pairs, pair_id):
    out = np.zeros(len(codes), dtype=bool)
    for j, p in enumerate(pairs):
        mask = pair_id == j
        out[mask] = np.isin(codes[mask], p[2])
    return out


def experiment_rounds(G, noniso, iso, T=10, R=40000, seed=41):
    gen = pk.rng(seed)
    m = len(noniso)
    codes0_non = np.array([G.code(p[0]) for p in noniso])
    codes1_non = np.array([G.code(p[1]) for p in noniso])
    codes0_iso = np.array([G.code(p[0]) for p in iso])
    codes1_iso = np.array([G.code(p[1]) for p in iso])
    names = ["honest", "guess", "algo", "dist", "leak", "peek"]
    alive = {k: np.ones(R, dtype=bool) for k in names}
    surv = {k: [] for k in names}
    pid_non = gen.integers(0, m, R)
    pid_iso = gen.integers(0, m, R)
    for _ in range(T):
        # the honest prover on non-isomorphic pairs
        b = gen.integers(0, 2, R)
        h = challenges(G, noniso, pid_non, b, gen)
        a = np.where(in_class(h, noniso, pid_non), 0, 1)
        alive["honest"] &= a == b
        # cheating provers on isomorphic pairs (one stream of challenges, shared)
        b = gen.integers(0, 2, R)
        h = challenges(G, iso, pid_iso, b, gen)
        a_guess = gen.integers(0, 2, R)
        a_algo = np.where(in_class(h, iso, pid_iso), 0, 1)  # always 0: H is isomorphic to G_0
        d0 = np.bitwise_count(h ^ codes0_iso[pid_iso])
        d1 = np.bitwise_count(h ^ codes1_iso[pid_iso])
        a_dist = np.where(d1 < d0, 1, 0)
        sees = gen.random(R) < 0.5
        a_leak = np.where(sees, b, gen.integers(0, 2, R))
        a_peek = b
        for k, a in [("guess", a_guess), ("algo", a_algo), ("dist", a_dist), ("leak", a_leak), ("peek", a_peek)]:
            alive[k] &= a == b
        for k in names:
            surv[k].append(int(alive[k].sum()))
    rows = []
    for t in range(1, T + 1):
        row = [t]
        for k in names:
            row += list(interval_cols(surv[k][t - 1], R))
        row += [2.0 ** (-t), 0.75 ** t]
        rows.append(row)
    header = ["t"]
    for k in names:
        header += [k, k + "p", k + "m"]
    header += ["bound", "leakexact"]
    pk.write_dat(os.path.join(OUT, "04-interaction-gni-rounds.dat"), header, rows)
    print("(a) rounds: R = %d trials per prover, %d pairs of each kind, n = %d" % (R, m, G.n))
    for t in (1, 2, 5, 10):
        r = rows[t - 1]
        print("   t=%2d honest %.4f | guess %.5f algo %.5f dist %.5f | 2^-t %.5f | leak %.4f (3/4)^t %.4f | peek %.3f"
              % (t, r[1], r[4], r[7], r[10], r[19], r[13], r[20], r[16]))
    lo, hi = pk.clopper_pearson(surv["guess"][-1], R)
    print("   guess at t=10: %d of %d accepted, interval [%.5f, %.5f]" % (surv["guess"][-1], R, lo, hi))
    return rows


# ----------------------------------------------------------------------------------------------
# (b) a partial disguise: s random transpositions
# ----------------------------------------------------------------------------------------------
def experiment_disguise(G, svals, R=40000, seed=42):
    gen = pk.rng(seed)
    # a graph with trivial automorphism group, and a uniform relabeling of it
    while True:
        A0 = G.random_graph(gen)
        allA = G.relabel_all(A0)
        codes = G.code(allA)
        if len(np.unique(codes)) == math.factorial(G.n):
            break
    A1 = G.relabel(A0, gen.permutation(G.n))
    order = np.argsort(codes)
    codes, allA = codes[order], allA[order]
    M = len(codes)
    i0 = int(np.searchsorted(codes, G.code(A0)))
    i1 = int(np.searchsorted(codes, G.code(A1)))
    # transposition table: T[e, tau] = index of the copy obtained from copy e by swapping i, j
    pairs = list(itertools.combinations(range(G.n), 2))
    T = np.empty((M, len(pairs)), dtype=np.int64)
    for c, (i, j) in enumerate(pairs):
        p = np.arange(G.n)
        p[i], p[j] = j, i
        T[:, c] = np.searchsorted(codes, G.code(allA[:, p[:, None], p[None, :]]))
    pidx = -np.ones((G.n, G.n), dtype=np.int64)
    for c, (i, j) in enumerate(pairs):
        pidx[i, j] = pidx[j, i] = c
    # exact distributions after s steps: identity w.p. 1/n, each transposition w.p. 2/n^2
    stay, move = 1.0 / G.n, 2.0 / G.n ** 2
    dists = {}
    p0 = np.zeros(M)
    p1 = np.zeros(M)
    p0[i0] = 1.0
    p1[i1] = 1.0
    smax = max(svals)
    for s in range(smax + 1):
        dists[s] = (p0.copy(), p1.copy())
        p0 = stay * p0 + move * p0[T].sum(axis=1)
        p1 = stay * p1 + move * p1[T].sum(axis=1)
    rows, tvs = [], {}
    for s in svals:
        q0, q1 = dists[s]
        tv = 0.5 * np.abs(q0 - q1).sum()
        tvs[s] = tv
        b = gen.integers(0, 2, R)
        e = np.where(b == 0, i0, i1)
        for _ in range(s):
            u, v = gen.integers(0, G.n, R), gen.integers(0, G.n, R)
            c = pidx[u, v]
            mv = c >= 0
            e[mv] = T[e[mv], c[mv]]
        a_bayes = np.where(q1[e] > q0[e], 1, np.where(q1[e] < q0[e], 0, gen.integers(0, 2, R)))
        d0 = np.bitwise_count(codes[e] ^ codes[i0])
        d1 = np.bitwise_count(codes[e] ^ codes[i1])
        a_dist = np.where(d1 < d0, 1, np.where(d1 > d0, 0, gen.integers(0, 2, R)))
        kb, kd = int((a_bayes == b).sum()), int((a_dist == b).sum())
        rows.append([s, *interval_cols(kb, R), *interval_cols(kd, R), 0.5 * (1 + tv)])
    pk.write_dat(os.path.join(OUT, "04-interaction-gni-disguise.dat"),
                 ["s", "bayes", "bayesp", "bayesm", "dist", "distp", "distm", "exact"], rows)
    # a dense curve of the exact value
    dense = [[s, 0.5 * (1 + 0.5 * np.abs(dists[s][0] - dists[s][1]).sum())] for s in range(smax + 1)]
    pk.write_dat(os.path.join(OUT, "04-interaction-gni-disguise-exact.dat"), ["s", "exact"], dense)
    print("(b) disguise: |Aut(G_0)| = 1, %d labeled copies; R = %d rounds per s" % (M, R))
    for r in rows:
        print("   s=%2d  Bayes %.4f [-%.4f,+%.4f]  exact (1+TV)/2 = %.4f   distance prover %.4f"
              % (r[0], r[1], r[3], r[2], r[7], r[4]))
    first = [s for s in range(smax + 1) if 0.5 * np.abs(dists[s][0] - dists[s][1]).sum() < 0.02]
    print("   first s with TV < 0.02 (prover's edge (1+TV)/2 - 1/2 < 0.01): %s" % (first[0] if first else "none"))
    return rows


# ----------------------------------------------------------------------------------------------
# (c) a verifier that reuses a pool of N challenges, with feedback after each session
# ----------------------------------------------------------------------------------------------
def experiment_reuse(Nvals, J=3000, R=4000, seed=43):
    gen = pk.rng(seed)
    checkpoints = np.unique(np.round(np.logspace(0, np.log10(J), 16)).astype(int))
    nclass = math.factorial(N_VERT)  # an isomorphic pair whose common class has 8! copies
    rows = {j: [j] for j in checkpoints}
    for N in Nvals:
        # each run: N distinct challenges (uniform labeled copies of the common class), N bits
        pool_b = gen.integers(0, 2, (R, N))
        pool_h = np.stack([gen.choice(nclass, N, replace=False) for _ in range(R)])
        seen = np.zeros((R, N), dtype=bool)  # the prover's memory, keyed by the challenge
        learned = np.zeros((R, N), dtype=np.int64)
        acc_at = {}
        rr = np.arange(R)
        for j in range(1, J + 1):
            i = gen.integers(0, N, R)
            h = pool_h[rr, i]  # the challenge sent (a labeled copy, identified by its index)
            known = seen[rr, i]
            a = np.where(known, learned[rr, i], gen.integers(0, 2, R))
            ok = a == pool_b[rr, i]
            seen[rr, i] = True
            learned[rr, i] = np.where(ok, a, 1 - a)  # the verdict reveals the secret bit
            if j in rows:
                acc_at[j] = int(ok.sum())
            del h
        for j in checkpoints:
            rows[j] += [*interval_cols(acc_at[j], R), 1 - 0.5 * (1 - 1 / N) ** (j - 1)]
    header = ["j"]
    for N in Nvals:
        header += ["N%d" % N, "N%dp" % N, "N%dm" % N, "th%d" % N]
    pk.write_dat(os.path.join(OUT, "04-interaction-gni-reuse.dat"), header, [rows[j] for j in checkpoints])
    dense = []
    for j in np.unique(np.round(np.logspace(0, np.log10(J), 120)).astype(int)):
        dense.append([j] + [1 - 0.5 * (1 - 1 / N) ** (j - 1) for N in Nvals])
    pk.write_dat(os.path.join(OUT, "04-interaction-gni-reuse-theory.dat"), ["j"] + ["th%d" % N for N in Nvals], dense)
    print("(c) reuse: R = %d runs, J = %d sessions" % (R, J))
    for j in checkpoints:
        r = rows[j]
        print("   j=%5d " % j + "  ".join("N=%d %.3f (th %.3f)" % (N, r[1 + 4 * q], r[4 + 4 * q]) for q, N in enumerate(Nvals)))


# ----------------------------------------------------------------------------------------------
# (d) the recursion of the proof that IP is in PSPACE, on Protocol 4.1 with n = 4
# ----------------------------------------------------------------------------------------------
def optimal_value(G4, A0, A1, t):
    """Largest acceptance probability over all provers of t parallel copies of Protocol 4.1,
    by the recursion N(tau) = sum over verifier messages of max over prover messages of counts.
    The verifier's coins are (b, pi) per copy; its message is the tuple of challenges; the prover
    answers a tuple of bits; for each message, the best answer is the most frequent b-tuple."""
    coins = [(b, q) for b in (0, 1) for q in G4.perms]
    codes = {}
    for combo in itertools.product(coins, repeat=t):
        msg = tuple(int(G4.code(G4.relabel([A0, A1][b], q))) for b, q in combo)
        bits = tuple(b for b, _ in combo)
        codes.setdefault(msg, {}).setdefault(bits, 0)
        codes[msg][bits] += 1
    total = sum(max(cnt.values()) for cnt in codes.values())
    return Fraction(total, len(coins) ** t)


def experiment_recursion():
    G4 = Graphs(4)
    P4 = np.zeros((4, 4), dtype=bool)
    for u, v in [(0, 1), (1, 2), (2, 3)]:
        P4[u, v] = P4[v, u] = True
    K13 = np.zeros((4, 4), dtype=bool)
    for v in (1, 2, 3):
        K13[0, v] = K13[v, 0] = True
    P4b = G4.relabel(P4, np.array([1, 3, 0, 2]))
    print("(d) recursion on n = 4: |class(P4)| = %d, |class(K13)| = %d"
          % (len(G4.iso_class(P4)), len(G4.iso_class(K13))))
    for t in (1, 2):
        v_non = optimal_value(G4, P4, K13, t)
        v_iso = optimal_value(G4, P4, P4b, t)
        print("   t=%d: path vs star (non-isomorphic) %s; path vs relabeled path (isomorphic) %s" % (t, v_non, v_iso))


# ----------------------------------------------------------------------------------------------
# (e) costs against n
# ----------------------------------------------------------------------------------------------
def experiment_costs(seed=44):
    gen = pk.rng(seed)
    rows = []
    print("(e) costs per round (this machine; wall-clock times are indicative)")
    for n in range(5, 10):
        G = Graphs(n)
        A0 = G.random_graph(gen)
        A1 = G.random_graph(gen)
        reps_v = 2000
        t0 = time.perf_counter()
        for _ in range(reps_v):
            q = gen.permutation(n)
            h = G.code(G.relabel(A1, q))
        tv = (time.perf_counter() - t0) / reps_v
        reps_p = 3 if n == 9 else 10
        t0 = time.perf_counter()
        for _ in range(reps_p):
            cls = G.iso_class(A0)  # brute force over all n! relabelings
            ans = 0 if np.isin(h, cls) else 1
        tp = (time.perf_counter() - t0) / reps_p
        bits = 1 + math.log2(math.factorial(n))
        comm = n * (n - 1) // 2 + 1
        rows.append([n, math.factorial(n), round(bits, 1), comm, tv * 1e6, tp * 1e3])
        print("   n=%d  n!=%6d  random bits %.1f  bits exchanged %d  verifier %.1f us  prover %.2f ms"
              % (n, math.factorial(n), bits, comm, tv * 1e6, tp * 1e3))
        del ans
    pk.write_dat(os.path.join(OUT, "04-interaction-gni-costs.dat"),
                 ["n", "nfact", "randbits", "commbits", "verifier_us", "prover_ms"], rows)


def main():
    t_start = time.perf_counter()
    G = Graphs(N_VERT)
    noniso, iso = make_pairs(G, pk.rng(40), 10)
    print("pairs: 10 non-isomorphic (edge counts equal), 10 isomorphic; n = %d" % N_VERT)
    print("   automorphism group sizes of the G_0's: non-iso %s; iso %s"
          % ([math.factorial(N_VERT) // len(p[2]) for p in noniso], [math.factorial(N_VERT) // len(p[2]) for p in iso]))
    experiment_rounds(G, noniso, iso)
    experiment_disguise(G, [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24])
    experiment_reuse([10, 100, 1000])
    experiment_recursion()
    experiment_costs()
    print("total time %.1f s" % (time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
