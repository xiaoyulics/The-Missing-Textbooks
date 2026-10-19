"""13-zeroknowledge-coloring.py -- Chapter 13 (Zero Knowledge): commit, challenge, respond.

Runs Protocol 13.2 (the 3-coloring protocol of Goldreich, Micali and Wigderson) with hash
commitments Com(a; rho) = SHA-256(a || rho), rho 16 random bytes, on the wheel with five spokes
W5 (hub 0, rim 1..5; m = 10 edges; not 3-colorable) and on W5 minus the rim edge {5,1}
(m = 9; 3-colorable).  The secret coloring c gives the hub color 3 and the rim 1,2,1,2,1; on W5
it has exactly one monochromatic edge, {5,1}, and no 3-coloring of W5 has fewer.

  (a) soundness: acceptance after t sequential rounds, t = 1..60, of the honest prover on
      W5 - {5,1}, and on W5 of: the best cheating prover (it plays c, whose only bad edge is
      {5,1}); a prover that draws a uniformly random coloring for the session; a prover facing a
      verifier whose challenges follow a schedule the prover knows.  Theory: 1, (9/10)^t (exact),
      E_kappa[(1 - b(kappa)/10)^t] (exact, by enumerating the 3^6 colorings), 1.
  (b) zero knowledge: one round on W5 - {5,1} against a cheating verifier V* that asks for the
      spoke to the rim vertex whose commitment is smallest (a function of the first message).
      Real transcripts with the honest prover; the rewinding simulator of Theorem 13.6 (it
      commits to the colors of a guessed edge and to color 1 elsewhere, and retries until V*
      asks for its edge, at most m*n = 54 times; given success, its number of calls is
      geometric with parameter 1/9 conditioned on at most 54, mean 8.91); a simulator without
      rewinding (it outputs its guessed edge as if V* had asked for it); real transcripts of a
      prover that never permutes the colors.  Statistics with Clopper-Pearson intervals, and the total variation distance
      between empirical (edge, revealed pair) histograms.
  (c) graph isomorphism: Protocol 13.1 on a random graph with 8 vertices and a relabeled copy;
      a verifier whose bit is a function of the challenge graph; the rewinding simulator of
      Theorem 13.4: calls to V* per round (exactly 2 in expectation) and the histogram of
      (bit, sigma(0)) against real transcripts.
  (d) Fiat-Shamir: the non-interactive proof for W5 with t = 66 rounds, challenges from
      SHAKE-256 (an idealized random oracle), against a grinding prover that re-randomizes one
      commitment per query.  Parallel version: one challenge vector for all rounds (soundness
      (q+1)s, s = (9/10)^66, Proposition 13.8); sequential version: one hash per round, which a
      grinder breaks with about 66*10/9 queries.
      How the parallel run of the figure came about (stated in the chapter): a first run, with
      another seed and fewer trials, put the Clopper-Pearson interval at q = 7 entirely above
      (q+1)s; the trial count was then raised and the seed changed (seed 1341 below).  The
      check (d2) reruns the parallel grinder with ten further seeds, 1371-1380 (3,000 trials
      each, up to q = 100, where the bound is closest to the exact curve), pools them, and
      computes, under the random-oracle law (queries to success ~ geometric(s)), how often a
      run of 3,000 trials puts some interval of the figure entirely above (q+1)s.
  (e) costs: bytes per round and per proof; rounds needed for 2^-40 and 2^-128.

Run from the volume root:  python3 figures/scripts/13-zeroknowledge-coloring.py
Writes figures/data/13-zeroknowledge-coloring-*.dat and prints every number the chapter quotes.
Deterministic (seeds below); about a minute and a half on a laptop CPU.
"""
import hashlib
import itertools
import math
import os
import sys
from fractions import Fraction

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

OUT = "figures/data"
PFX = os.path.join(OUT, "13-zeroknowledge-coloring")

# ----------------------------------------------------------------------------------------------
# The graphs and the secret coloring
# ----------------------------------------------------------------------------------------------
N = 6
SPOKES = [(0, i) for i in range(1, 6)]
RIM = [(1, 2), (2, 3), (3, 4), (4, 5)]
E_YES = SPOKES + RIM                 # W5 minus {5,1}: 3-colorable, m = 9
E_NO = SPOKES + RIM + [(1, 5)]       # W5: not 3-colorable, m = 10
COLORING = [3, 1, 2, 1, 2, 1]        # hub 3; rim 1,2,1,2,1 -- proper on E_YES, one bad edge on E_NO
S3 = list(itertools.permutations((1, 2, 3)))


def bad_edges(edges, col):
    return sum(1 for (u, v) in edges if col[u] == col[v])


# ----------------------------------------------------------------------------------------------
# Hash commitments: Com(a; rho) = SHA-256(byte a || rho); opening = (a, rho)
# ----------------------------------------------------------------------------------------------
def com(a, rho):
    return hashlib.sha256(bytes([a]) + rho).digest()


def commit_vector(colors, gen):
    rhos = [gen.bytes(16) for _ in colors]
    return [com(a, r) for a, r in zip(colors, rhos)], rhos


def verify_opening(C, edge, op):
    (u, v) = edge
    (au, ru), (av, rv) = op
    return (au in (1, 2, 3) and av in (1, 2, 3) and au != av
            and com(au, ru) == C[u] and com(av, rv) == C[v])


def permuted(col, psi):
    """psi in S3 as a tuple (psi(1), psi(2), psi(3))."""
    return [psi[a - 1] for a in col]


# ----------------------------------------------------------------------------------------------
# (a) soundness of the interactive protocol
# ----------------------------------------------------------------------------------------------
TMAX = 60


def run_session(edges, coloring_for_round, gen, tmax, challenge=None):
    """Sequential rounds until the verifier rejects or tmax rounds pass; returns rounds survived.
    coloring_for_round(j, e_pred) gives the coloring the prover commits to in round j (e_pred is
    the verifier's challenge when the prover can predict it, else None)."""
    m = len(edges)
    for j in range(tmax):
        e_pred = None if challenge is None else challenge(j)
        col = coloring_for_round(j, e_pred)
        psi = S3[gen.integers(6)]
        d = permuted(col, psi)
        C, rhos = commit_vector(d, gen)
        k = gen.integers(m) if challenge is None else e_pred
        (u, v) = edges[k]
        op = ((d[u], rhos[u]), (d[v], rhos[v]))
        if not verify_opening(C, (u, v), op):
            return j
    return tmax


def soundness_experiment():
    print("== (a) soundness: acceptance after t rounds ==")
    m_no, m_yes = len(E_NO), len(E_YES)
    assert bad_edges(E_YES, COLORING) == 0 and bad_edges(E_NO, COLORING) == 1
    allcols = list(itertools.product((1, 2, 3), repeat=N))
    bcounts = [bad_edges(E_NO, c) for c in allcols]
    assert min(bcounts) == 1, "W5 should need at least one bad edge"
    nb = {b: bcounts.count(b) for b in sorted(set(bcounts))}
    print(f"  colorings of W5 by number of bad edges: {nb} (total {len(allcols)})")
    print(f"  colorings with exactly one bad edge: {nb[1]}")

    sessions = {"honest": 2000, "best": 40000, "rand": 40000, "pred": 2000}
    surv = {}
    gen = pk.rng(1351)
    surv["honest"] = [run_session(E_YES, lambda j, e: COLORING, gen, TMAX) for _ in range(sessions["honest"])]
    gen = pk.rng(1352)
    surv["best"] = [run_session(E_NO, lambda j, e: COLORING, gen, TMAX) for _ in range(sessions["best"])]
    gen = pk.rng(1353)
    out = []
    for _ in range(sessions["rand"]):
        kappa = [int(x) for x in gen.integers(1, 4, size=N)]
        out.append(run_session(E_NO, lambda j, e, kk=kappa: kk, gen, TMAX))
    surv["rand"] = out
    # a verifier with a published schedule: round j asks edge (7j + 3) mod m; the prover commits to
    # a coloring that is proper on that edge only (its endpoints 1 and 2, every other vertex 1)
    def sched(j):
        return (7 * j + 3) % m_no

    def pred_col(j, k):
        (u, v) = E_NO[k]
        col = [1] * N
        col[u], col[v] = 1, 2
        return col
    gen = pk.rng(1354)
    surv["pred"] = [run_session(E_NO, pred_col, gen, TMAX, challenge=sched) for _ in range(sessions["pred"])]

    rows = []
    for t in range(1, TMAX + 1):
        row = [t]
        for key in ("honest", "best", "rand", "pred"):
            k = sum(1 for s in surv[key] if s >= t)
            n = sessions[key]
            lo, hi = pk.clopper_pearson(k, n)
            rate = k / n
            row += [rate, rate - lo, hi - rate, k]
        exact_best = 0.9 ** t
        exact_rand = sum(cnt * (1 - b / m_no) ** t for b, cnt in nb.items()) / len(allcols)
        row += [exact_best, exact_rand]
        rows.append(row)
    header = ["t"]
    for key in ("honest", "best", "rand", "pred"):
        header += [key, key + "m", key + "p", key + "k"]
    header += ["bestexact", "randexact"]
    pk.write_dat(PFX + "-rounds.dat", header, rows)
    # the rows plotted in the figure (every t up to 6, then every fifth); every plotted point has k > 0
    plot_t = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    prow = [r for r in rows if r[0] in plot_t]
    assert all(r[8] > 0 and r[12] > 0 for r in prow), "a plotted point has no accepted session"
    pk.write_dat(PFX + "-rounds-plot.dat", header, prow)
    for r in rows:
        if r[0] in (1, 2, 5, 10, 20, 30, 40, 50, 60):
            t = r[0]
            print(f"  t={t:2d}: honest {r[1]:.4f}  best {r[5]:.5f} [{r[5]-r[6]:.5f},{r[5]+r[7]:.5f}] "
                  f"(k={r[8]}) exact {r[17]:.5f}   rand {r[9]:.5f} [{r[9]-r[10]:.5f},{r[9]+r[11]:.5f}] "
                  f"(k={r[12]}) exact {r[18]:.5f}   pred {r[13]:.4f}")
    # does every interval of the best cheater contain its exact value?
    miss_best = [r[0] for r in rows if not (r[5] - r[6] <= r[17] <= r[5] + r[7])]
    miss_rand = [r[0] for r in rows if not (r[9] - r[10] <= r[18] <= r[9] + r[11])]
    print(f"  intervals missing the exact value: best {miss_best}, rand {miss_rand}")
    above_best = [r[0] for r in rows if r[5] - r[6] > r[17]]
    print(f"  best-cheater intervals entirely above (9/10)^t: {above_best}")
    print(f"  rand exact at t=1: {rows[0][18]:.4f} (= 1 - E[b]/10, E[b] = {10/3:.4f})")
    return nb


# ----------------------------------------------------------------------------------------------
# (b) zero knowledge: real transcripts against simulators, for a cheating verifier
# ----------------------------------------------------------------------------------------------
def vstar(C):
    """The cheating verifier: the spoke to the rim vertex with the smallest commitment."""
    i = min(range(1, 6), key=lambda w: C[w])
    return E_YES.index((0, i))


def real_round(gen, permute=True):
    psi = S3[gen.integers(6)] if permute else (1, 2, 3)
    d = permuted(COLORING, psi)
    C, rhos = commit_vector(d, gen)
    k = vstar(C)
    (u, v) = E_YES[k]
    op = ((d[u], rhos[u]), (d[v], rhos[v]))
    assert verify_opening(C, (u, v), op)
    return C, k, op, 1


def sim_attempt(gen):
    m = len(E_YES)
    kp = gen.integers(m)
    (u, v) = E_YES[kp]
    pair = list(S3[gen.integers(6)])[:2]   # a uniform ordered pair of distinct colors
    d = [1] * N
    d[u], d[v] = pair[0], pair[1]
    C, rhos = commit_vector(d, gen)
    op = ((d[u], rhos[u]), (d[v], rhos[v]))
    return C, kp, op


def rewind_round(gen, cap):
    for calls in range(1, cap + 1):
        C, kp, op = sim_attempt(gen)
        if vstar(C) == kp:
            return C, kp, op, calls
    return None


def hvzk_round(gen):
    C, kp, op = sim_attempt(gen)
    return C, kp, op, 1


def summarize(name, transcripts):
    n = len(transcripts)
    spoke = sum(1 for (C, k, op, c) in transcripts if E_YES[k][0] == 0)
    consistent = sum(1 for (C, k, op, c) in transcripts if vstar(C) == k)
    first1 = sum(1 for (C, k, op, c) in transcripts if op[0][0] == 1)
    valid = sum(1 for (C, k, op, c) in transcripts if verify_opening(C, E_YES[k], op))
    calls = np.mean([c for (C, k, op, c) in transcripts])
    hist = np.zeros((len(E_YES), 6))
    for (C, k, op, c) in transcripts:
        hist[k, S3_PAIR_INDEX[(op[0][0], op[1][0])]] += 1
    res = {"n": n, "spoke": spoke, "consistent": consistent, "first1": first1, "valid": valid,
           "calls": calls, "hist": hist / n}
    def ci(k):
        lo, hi = pk.clopper_pearson(k, n)
        return f"{k/n:.4f} [{lo:.4f},{hi:.4f}]"
    print(f"  {name:28s} n={n}: spoke {ci(spoke)}; consistent with V* {ci(consistent)}; "
          f"first endpoint color 1 {ci(first1)}; valid openings {valid}; calls to V* {calls:.3f}")
    return res


S3_PAIR_INDEX = {(a, b): i for i, (a, b) in enumerate([(p[0], p[1]) for p in S3])}


def tv(h1, h2):
    return 0.5 * float(np.abs(h1 - h2).sum())


def tv_boot(gen, h1, n1, h2, n2, reps=400):
    """Percentile bootstrap 95% interval for the empirical TV distance between two samples
    (each resampled from its own histogram); appropriate when the distance is far from 0."""
    p1, p2 = h1.ravel(), h2.ravel()
    vals = [0.5 * np.abs(gen.multinomial(n1, p1) / n1 - gen.multinomial(n2, p2) / n2).sum()
            for _ in range(reps)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def tv_null(gen, h1, n1, h2, n2, reps=400):
    """The law of the empirical TV distance between two samples of sizes n1, n2 drawn from ONE
    law (the pooled histogram): its 2.5% and 97.5% points, and the fraction of draws at least as
    large as the observed distance (a parametric-bootstrap p-value)."""
    pool = (n1 * h1.ravel() + n2 * h2.ravel()) / (n1 + n2)
    obs = tv(h1, h2)
    vals = np.array([0.5 * np.abs(gen.multinomial(n1, pool) / n1 - gen.multinomial(n2, pool) / n2).sum()
                     for _ in range(reps)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(np.mean(vals >= obs))


def zk_experiment():
    print("== (b) zero knowledge on W5 - {5,1} (m = 9, n = 6) against V* ==")
    NT = 30000
    cap = len(E_YES) * N
    gen = pk.rng(1311)
    real = [real_round(gen) for _ in range(NT)]
    gen = pk.rng(1312)
    real2 = [real_round(gen) for _ in range(NT)]
    gen = pk.rng(1313)
    sims = [rewind_round(gen, cap) for _ in range(NT)]
    fails = sum(1 for s in sims if s is None)
    sims = [s for s in sims if s is not None]
    gen = pk.rng(1314)
    hv = [hvzk_round(gen) for _ in range(NT)]
    gen = pk.rng(1315)
    leaky = [real_round(gen, permute=False) for _ in range(NT)]
    r = summarize("real (honest prover)", real)
    r2 = summarize("real, second sample", real2)
    s = summarize("rewinding simulator", sims)
    h = summarize("simulator, no rewinding", hv)
    lk = summarize("real, colors never permuted", leaky)
    print(f"  rewinding simulator: {fails} of {NT} runs hit the cap of {cap} calls")
    gen = pk.rng(1316)
    rows = []
    for name, res in (("real2", r2), ("rewind", s), ("norewind", h), ("leaky", lk)):
        d = tv(r["hist"], res["hist"])
        nlo, nhi, pval = tv_null(gen, r["hist"], r["n"], res["hist"], res["n"])
        lo, hi = tv_boot(gen, r["hist"], r["n"], res["hist"], res["n"])
        print(f"  TV(real, {name:8s}) over (edge, pair) histogram = {d:.4f}; one-law band 95% "
              f"[{nlo:.4f},{nhi:.4f}], p = {pval:.3f}; bootstrap 95% of the distance [{lo:.4f},{hi:.4f}]")
        rows.append((name, res, d, nlo, nhi, pval, lo, hi))
    # exact TV for the no-rewinding simulator: real puts mass 1/30 on (spoke, pair) cells, the
    # simulator 1/54 on every cell
    exact_h = 0.5 * (5 * 6 * abs(1 / 30 - 1 / 54) + 4 * 6 * (1 / 54))
    print(f"  exact TV(real, no rewinding) = {exact_h:.4f} (= 4/9)")
    # exact TV for the leaky prover: on spoke (0,v) it always reveals (3, c(v)), a cell to which
    # real transcripts give mass 1/30; so TV = 1 - sum of min = 1 - 5 * (1/30)
    leaky_exact = 1 - 5 * 1 / 30
    print(f"  exact TV(real, leaky) = {leaky_exact:.4f} (= 5/6)")
    # mean number of calls of the rewinding simulator per transcript it outputs.  If V*'s choice is
    # independent of the guess (as it is up to the hiding of the commitments), each attempt
    # succeeds with probability p = 1/9, and the calls of a successful run are geometric(p)
    # conditioned on at most cap: mean 8.91 (9 without the cap)
    p = 1 / len(E_YES)
    fail = (1 - p) ** cap
    trunc_mean = sum(k * (1 - p) ** (k - 1) * p for k in range(1, cap + 1)) / (1 - fail)
    print(f"  rewinding simulator: mean calls {s['calls']:.3f} per output transcript (theory "
          f"{trunc_mean:.3f} under the cap, 9 without it); max {max(c for (*_, c) in sims)}; "
          f"expected runs at the cap {NT * fail:.1f}")
    with open(PFX + "-zk.dat", "w") as fh:
        fh.write("row n spoke spokelo spokehi consistent conslo conshi first1 f1lo f1hi calls "
                 "tv nulllo nullhi pval bootlo boothi\n")
        for name, res, d, nlo, nhi, pval, lo, hi in [("real", r, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)] + rows:
            n = res["n"]
            vals = [name, n]
            for key in ("spoke", "consistent", "first1"):
                k = res[key]
                a, b = pk.clopper_pearson(k, n)
                vals += [f"{k/n:.4f}", f"{a:.4f}", f"{b:.4f}"]
            vals += [f"{res['calls']:.3f}", f"{d:.4f}", f"{nlo:.4f}", f"{nhi:.4f}", f"{pval:.3f}",
                     f"{lo:.4f}", f"{hi:.4f}"]
            fh.write(" ".join(str(v) for v in vals) + "\n")
    return r, s, h, lk


# ----------------------------------------------------------------------------------------------
# (c) graph isomorphism: the rewinding simulator of Theorem 13.4
# ----------------------------------------------------------------------------------------------
def gi_experiment():
    print("== (c) graph isomorphism, n = 8: rewinding simulator ==")
    n = 8
    gen = pk.rng(1321)
    U = np.triu(gen.random((n, n)) < 0.5, 1)
    G0 = U | U.T
    phi = gen.permutation(n)

    def relabel(A, q):
        """The graph q(A): vertex i of A becomes vertex q[i]."""
        B = np.zeros_like(A)
        B[np.ix_(q, q)] = A
        return B
    G1 = relabel(G0, phi)
    # automorphisms of G0 (brute force over 8! permutations)
    aut = sum(1 for q in itertools.permutations(range(n)) if np.array_equal(relabel(G0, np.array(q)), G0))
    print(f"  |Aut(G0)| = {aut}, edges {int(G0.sum()//2)}")

    def vbit(H):
        # a cheating verifier: the parity of the number of edges of H among vertices 0..3
        return int(H[:4, :4].sum() // 2) % 2

    NT = 20000
    real_hist = np.zeros((2, n))
    gen = pk.rng(1322)
    for _ in range(NT):
        pi = gen.permutation(n)
        H = relabel(G1, pi)
        b = vbit(H)
        sigma = pi if b == 1 else pi[phi]       # (pi phi)(i) = pi[phi[i]]
        assert np.array_equal(relabel(G1 if b == 1 else G0, sigma), H)
        real_hist[b, sigma[0]] += 1
    sim_hist = np.zeros((2, n))
    calls = []
    gen = pk.rng(1323)
    for _ in range(NT):
        c = 0
        while True:
            c += 1
            bp = int(gen.integers(2))
            sigma = gen.permutation(n)
            H = relabel(G1 if bp == 1 else G0, sigma)
            if vbit(H) == bp:
                break
        calls.append(c)
        sim_hist[bp, sigma[0]] += 1
    real_hist /= NT
    sim_hist /= NT
    d = tv(real_hist, sim_hist)
    lo, hi, pval = tv_null(pk.rng(1324), real_hist, NT, sim_hist, NT)
    calls = np.array(calls)
    se = calls.std(ddof=1) / math.sqrt(len(calls))
    print(f"  mean calls to V* per round {calls.mean():.4f} +- {1.96*se:.4f} (theory exactly 2); "
          f"fraction needing more than 5 calls {np.mean(calls > 5):.4f} (theory {2**-5:.4f})")
    print(f"  fraction of real rounds with b = 1: {real_hist[1].sum():.4f}; simulated: {sim_hist[1].sum():.4f}")
    print(f"  TV(real, simulated) of (b, sigma(0)) histogram {d:.4f}; one-law band 95% [{lo:.4f},{hi:.4f}], p = {pval:.3f}")
    # floor: two independent real samples
    gen = pk.rng(1325)
    real2 = np.zeros((2, n))
    for _ in range(NT):
        pi = gen.permutation(n)
        H = relabel(G1, pi)
        b = vbit(H)
        sigma = pi if b == 1 else pi[phi]
        real2[b, sigma[0]] += 1
    real2 /= NT
    print(f"  TV(real, real2) floor {tv(real_hist, real2):.4f}")
    return calls.mean()


# ----------------------------------------------------------------------------------------------
# (d) Fiat-Shamir: grinding against the parallel and the sequential non-interactive proofs
# ----------------------------------------------------------------------------------------------
TFS = 66
GRAPH_TAG = b"W5:" + b",".join(f"{u}-{v}".encode() for (u, v) in E_NO)


def edges_from(digest_stream, count, m):
    """Uniform values in [0, m) from a byte stream, by rejection sampling (bytes < 250 for m=10)."""
    lim = 256 - 256 % m
    out = []
    for byte in digest_stream:
        if byte < lim:
            out.append(byte % m)
            if len(out) == count:
                return out
    raise RuntimeError("stream too short")


def fs_challenges(h, count, m):
    return edges_from(h.digest(4 * count + 64), count, m)


def fs_parallel_grind(gen, qmax):
    """The grinder commits to t permuted copies of c (one bad edge each); each query re-randomizes
    the last commitment and hashes; returns the number of queries until all t challenges avoid the
    bad edge (or None), and the proof it would output (checked by the verifier)."""
    m = len(E_NO)
    rounds = []
    for j in range(TFS):
        psi = S3[gen.integers(6)]
        d = permuted(COLORING, psi)
        C, rhos = commit_vector(d, gen)
        rounds.append((d, C, rhos))
    base = hashlib.shake_256(GRAPH_TAG)
    for j in range(TFS):
        C = rounds[j][1]
        upto = N if j < TFS - 1 else N - 1
        for v in range(upto):
            base.update(C[v])
    d_last = rounds[-1][0]
    bad = E_NO.index((1, 5))
    for q in range(1, qmax + 1):
        rho = gen.bytes(16)
        c_last = com(d_last[N - 1], rho)
        h = base.copy()
        h.update(c_last)
        ch = fs_challenges(h, TFS, m)
        if bad not in ch:
            rounds[-1][1][N - 1] = c_last
            rounds[-1][2][N - 1] = rho
            # the non-interactive verifier: recompute the challenges and check every opening
            hv = hashlib.shake_256(GRAPH_TAG)
            for (d, C, rhos) in rounds:
                for v in range(N):
                    hv.update(C[v])
            ch2 = fs_challenges(hv, TFS, m)
            ok = all(verify_opening(rounds[j][1], E_NO[ch2[j]],
                                    ((rounds[j][0][E_NO[ch2[j]][0]], rounds[j][2][E_NO[ch2[j]][0]]),
                                     (rounds[j][0][E_NO[ch2[j]][1]], rounds[j][2][E_NO[ch2[j]][1]])))
                     for j in range(TFS))
            assert ok and ch2 == ch
            return q
    return None


def fs_sequential_grind(gen):
    """Round j's challenge is the hash of the graph and the transcript so far; the grinder
    re-randomizes round j's last commitment until the challenge avoids the bad edge."""
    m = len(E_NO)
    bad = E_NO.index((1, 5))
    h = hashlib.shake_256(GRAPH_TAG)
    queries = 0
    transcript = []
    for j in range(TFS):
        psi = S3[gen.integers(6)]
        d = permuted(COLORING, psi)
        C, rhos = commit_vector(d, gen)
        for v in range(N - 1):
            h.update(C[v])
        while True:
            queries += 1
            rho = gen.bytes(16)
            c_last = com(d[N - 1], rho)
            hj = h.copy()
            hj.update(c_last)
            k = fs_challenges(hj, 1, m)[0]
            if k != bad:
                C[N - 1], rhos[N - 1] = c_last, rho
                break
        h.update(C[N - 1])
        h.update(bytes([k]))
        transcript.append((d, C, rhos, k))
    # verify
    hv = hashlib.shake_256(GRAPH_TAG)
    for (d, C, rhos, k) in transcript:
        for v in range(N):
            hv.update(C[v])
        k2 = fs_challenges(hv.copy(), 1, m)[0]
        assert k2 == k
        (u, v) = E_NO[k]
        assert verify_opening(C, (u, v), ((d[u], rhos[u]), (d[v], rhos[v])))
        hv.update(bytes([k]))
    return queries


def negbin_cdf(q, t, p):
    """P[t successes need at most q Bernoulli(p) trials]."""
    if q < t:
        return 0.0
    return float(stats.binom.sf(t - 1, q, p))


def fs_experiment():
    print(f"== (d) Fiat-Shamir on W5, t = {TFS} rounds ==")
    s = 0.9 ** TFS
    print(f"  interactive soundness error s = (9/10)^{TFS} = {s:.4e} (1/s = {1/s:.1f})")
    QMAX = 4000
    NPAR, NSEQ = 3000, 2000
    gen = pk.rng(1341)
    par = [fs_parallel_grind(gen, QMAX) for _ in range(NPAR)]
    gen = pk.rng(1342)
    seq = [fs_sequential_grind(gen) for _ in range(NSEQ)]
    seqa = np.array(seq)
    print(f"  sequential grinder: queries mean {seqa.mean():.2f} (theory {TFS/0.9:.2f}), "
          f"min {seqa.min()}, max {seqa.max()}")
    qs = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 66, 70, 73, 76, 80, 85, 90, 100, 150, 200, 300, 500, 700,
          1000, 1500, 2000, 3000, 4000]
    rows = []
    for q in qs:
        kp = sum(1 for x in par if x is not None and x <= q)
        ks = int(np.sum(seqa <= q))
        lop, hip = pk.clopper_pearson(kp, NPAR)
        los, his = pk.clopper_pearson(ks, NSEQ)
        rp, rs = kp / NPAR, ks / NSEQ
        exact_p = 1 - (1 - s) ** q
        bound = min(1.0, (q + 1) * s)
        exact_s = negbin_cdf(q, TFS, 0.9)
        rows.append([q, rp, rp - lop, hip - rp, kp, rs, rs - los, his - rs, ks, exact_p, bound, exact_s])
        print(f"  q={q:5d}: parallel {rp:.4f} [{lop:.4f},{hip:.4f}] exact {exact_p:.4f} bound {bound:.4f}"
              f" | sequential {rs:.4f} [{los:.4f},{his:.4f}] exact {exact_s:.4f}")
    pk.write_dat(PFX + "-fs.dat", ["q", "par", "parm", "parp", "park", "seq", "seqm", "seqp", "seqk",
                                   "parexact", "bound", "seqexact"], rows)
    viol = [r[0] for r in rows if r[1] - r[2] > r[10]]
    print(f"  parallel intervals entirely above the bound (q+1)s: {viol}")
    # smooth theory curves
    qq = np.unique(np.round(np.logspace(0, math.log10(QMAX), 120)).astype(int))
    trows = [[int(q), 1 - (1 - s) ** q, min(1.0, (q + 1) * s), negbin_cdf(int(q), TFS, 0.9)] for q in qq]
    pk.write_dat(PFX + "-fs-theory.dat", ["q", "parexact", "bound", "seqexact"], trows)
    # the sequential series start at q = t (no proof is possible with fewer queries): separate
    # files without the zeros, for logarithmic axes
    pk.write_dat(PFX + "-fs-seq.dat", ["q", "seq", "seqm", "seqp", "seqk"],
                 [[r[0], r[5], r[6], r[7], r[8]] for r in rows if r[0] >= TFS])
    qseq = list(range(TFS, 101)) + [int(q) for q in qq if q > 100]
    pk.write_dat(PFX + "-fs-seq-theory.dat", ["q", "seqexact"],
                 [[q, negbin_cdf(q, TFS, 0.9)] for q in qseq])
    fs_seed_check(s, qs, NPAR)
    return s


FS_SEEDS = list(range(1371, 1381))   # ten further runs of the parallel grinder (d2)
FS_QCHECK_MAX = 100


def cp_above_probability(s, qs, n, reps, gen):
    """Under the random-oracle law (the number of queries to success is geometric(s)), the
    probability that one run of n trials puts the Clopper-Pearson interval at some q in qs
    entirely above (q+1)s; and, for each q, the probability that it does so at q."""
    qs = np.array(qs)
    cdf = 1 - (1 - s) ** qs
    cells = np.diff(np.concatenate([[0.0], cdf, [1.0]]))
    ks = np.arange(1, n + 1)
    lows = stats.beta.ppf(0.025, ks, n - ks + 1)          # lower CP limits, increasing in k
    bounds = (qs + 1) * s
    kstar = np.where(bounds < 1, np.searchsorted(lows, bounds, side="right") + 1, n + 1)
    counts = np.cumsum(gen.multinomial(n, cells, size=reps)[:, :-1], axis=1)
    above = counts >= kstar
    return float(above.any(axis=1).mean()), above.mean(axis=0)


def fs_seed_check(s, qs, npar):
    """(d2) The parallel grinder again, with ten further seeds, up to q = 100; pooled rates; and
    how often 95% intervals put a point above (q+1)s under the random-oracle law."""
    qc = [q for q in qs if q <= FS_QCHECK_MAX]
    ratios = [(q + 1) * s / (1 - (1 - s) ** q) for q in qc if q >= 7]
    print(f"  -- (d2) ten further runs of the parallel grinder: seeds {FS_SEEDS[0]}-{FS_SEEDS[-1]}, "
          f"{npar} trials each, up to q = {FS_QCHECK_MAX} --")
    print(f"    for 7 <= q <= {FS_QCHECK_MAX} on the grid, (q+1)s exceeds 1-(1-s)^q by "
          f"{100 * (min(ratios) - 1):.1f}% to {100 * (max(ratios) - 1):.1f}%")
    pooled = np.zeros(len(qc), dtype=int)
    n_above = n_miss = 0
    for seed in FS_SEEDS:
        gen = pk.rng(seed)
        par = [fs_parallel_grind(gen, FS_QCHECK_MAX) for _ in range(npar)]
        above, miss = [], []
        for i, q in enumerate(qc):
            k = sum(1 for x in par if x is not None and x <= q)
            pooled[i] += k
            lo, hi = pk.clopper_pearson(k, npar)
            if lo > (q + 1) * s:
                above.append(q)
            ex = 1 - (1 - s) ** q
            if not lo <= ex <= hi:
                miss.append(f"{q}{'+' if lo > ex else '-'}")
        n_above += len(above)
        n_miss += len(miss)
        print(f"    seed {seed}: intervals entirely above (q+1)s at q = {above}; "
              f"intervals missing 1-(1-s)^q at q = {miss} (+ above, - below)")
    total = npar * len(FS_SEEDS)
    print(f"    over the {len(FS_SEEDS)} runs: {n_above} of {len(FS_SEEDS) * len(qc)} intervals entirely "
          f"above (q+1)s; {n_miss} miss 1-(1-s)^q")
    for i, q in enumerate(qc):
        lo, hi = pk.clopper_pearson(int(pooled[i]), total)
        print(f"    pooled q={q:4d}: {pooled[i] / total:.5f} [{lo:.5f},{hi:.5f}] of {total}; "
              f"exact {1 - (1 - s) ** q:.5f}; bound {(q + 1) * s:.5f}")
    for grid, name in ((qs, "the figure's grid"), (qc, f"q <= {FS_QCHECK_MAX}")):
        p_any, p_q = cp_above_probability(s, grid, npar, 200000, pk.rng(1343))
        top = np.argsort(-p_q)[:4]
        print(f"    random-oracle law, {npar} trials, {name}: P[some interval entirely above (q+1)s] = "
              f"{p_any:.4f}; largest at " + ", ".join(f"q={grid[i]} ({p_q[i]:.4f})" for i in top))


# ----------------------------------------------------------------------------------------------
# (e) costs
# ----------------------------------------------------------------------------------------------
def costs():
    print("== (e) costs ==")
    m = len(E_NO)
    per_round = N * 32 + 2 * (1 + 16)
    print(f"  bytes per round on W5: {N}*32 (commitments) + 2*17 (openings) = {per_round}; "
          f"the verifier's challenge needs log2({m}) = {math.log2(m):.2f} bits")
    for bits in (40, 128):
        t = math.ceil(bits * math.log(2) / -math.log(1 - 1 / m))
        print(f"  soundness 2^-{bits} on W5 needs t = {t} rounds; parallel proof {t*per_round} bytes"
              f" = {t*per_round/1000:.1f} kB; after Fiat-Shamir, against 2^80 hash queries, "
              f"min(1, (q+1)s) = 2^{min(0.0, math.log2((2**80 + 1) * (1 - 1 / m) ** t)):.1f}")
        for mm in (20, 100, 1000):
            tt = math.ceil(bits * math.log(2) / -math.log(1 - 1 / mm))
            print(f"    m = {mm}: t = {tt} rounds (bits*ln2*m ~ {bits*math.log(2)*mm:.0f})")
    # GI: 2^-40 needs 40 rounds
    print(f"  GI: 2^-40 needs 40 rounds; Fiat-Shamir with q = 2^64 queries and 2^-64 target: 128 rounds")


def main():
    os.makedirs(OUT, exist_ok=True)
    soundness_experiment()
    zk_experiment()
    gi_experiment()
    fs_experiment()
    costs()


if __name__ == "__main__":
    main()
