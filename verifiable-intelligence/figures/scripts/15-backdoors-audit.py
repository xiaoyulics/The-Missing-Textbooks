"""Chapter 15 (Undetectable Backdoors): black-box audits of a keyed backdoor, and its removal by
self-correction.

A STAND-IN FOR A SIGNATURE. The backdoored model of the chapter recognizes a trigger by checking a
digital signature. This script uses HMAC-SHA256 with a secret key K, truncated to lam bits, in
place of the signature: an input x = (w, y, sigma) is a trigger when sigma equals the first lam
bits of HMAC_K(w || y). HMAC is a message authentication code, not a signature: whoever holds K can
both check and make triggers. The stand-in is faithful for a black-box auditor, who never sees K,
and unfaithful for a white-box one, who would read K in the model's code. Modeling HMAC as a random
function, every message has exactly one valid tag, so each query is a trigger with probability
2^-lam (the chapter's corollary on pseudorandom keys).

Experiments
 A. Audits (figure, left). Honest model: L(x) = <a, w> mod 2, a linear function of the 32-bit
    field w that ignores y and sigma. Backdoored model: y on triggers, L elsewhere. The auditor
    holds a reference copy of L, always sets y = 1 - L(w), and flags any answer that differs from
    L: it detects the backdoor exactly when one of its queries is a trigger. Strategies:
      random      fresh uniform w and sigma at every query
      near-data   w from the data distribution (bits Bernoulli(0.2)), sigma of low Hamming weight in
                  a fixed order (zero, then single bits, then pairs, ...)
      enumerate   one message (w, y) and distinct tags sigma in random order (the best auditor)
    lam = 8, 10, 12; up to q = 2^(lam+1) queries per audit; detection at q is the fraction of audits
    whose first trigger came at or before query q.
 B. Mitigation (figure, right). F_2^n with n = 20, L linear, f~ = L + 1_E with E of density delta
    containing the targets; the mitigator outputs the majority of t values f~(x + r_j) + f~(r_j).
      worst case  E = {x*} plus points of the half-space {z_k = 0} (x*_k = 1), so that E and E + x*
                  are disjoint: one trial errs with probability exactly 2 delta
      keyed       E = {x : HMAC_K(x) < delta 2^64}, targets drawn from E
 C. Undetectable, then removed. The keyed backdoor of A with lam = 8, 16, 32, 64: audits with
    10^4 random queries, and the attacker's activated inputs before and after mitigation.

Writes figures/data/15-backdoors-audit-{8,10,12}.dat, 15-backdoors-audit-theory.dat,
15-backdoors-mitigate.dat, 15-backdoors-mitigate-theory.dat.
Run from the volume root:  python3 figures/scripts/15-backdoors-audit.py
"""
import hmac
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk

WBITS = 32                      # length of the field w
LAMS = (8, 10, 12)              # tag lengths for the audits of experiment A
N_AUDIT = 2000                  # audits per strategy and tag length
N_MIT = 20_000                  # mitigations per point in experiment B
NB = 20                         # experiment B works on F_2^20
T_GRID = (1, 3, 5, 9, 15, 21, 31, 41)


def parity(v):
    """Parity of the bits of each nonnegative integer in v (int64 array)."""
    v = v.copy()
    for s in (32, 16, 8, 4, 2, 1):
        v ^= v >> s
    return v & 1


class KeyedModel:
    """The backdoored model with an HMAC tag of lam bits as its trigger check.

    Inputs are triples (w, y, sigma); the honest model is L(w) = <a, w> mod 2. The tag of a message
    (w, y) is the first lam bits of HMAC-SHA256_K(w || y). Tags are cached per message, which
    changes no answer of the oracle; `calls` counts HMAC evaluations."""

    def __init__(self, key: bytes, a: int, lam: int):
        self.key, self.a, self.lam = key, a, lam
        self.cache = {}
        self.calls = 0

    def honest(self, w: int) -> int:
        return bin(w & self.a).count("1") & 1

    def tag(self, w: int, y: int) -> int:
        m = (w, y)
        t = self.cache.get(m)
        if t is None:
            d = hmac.digest(self.key, w.to_bytes(16, "big") + bytes([y]), "sha256")
            t = int.from_bytes(d[:8], "big") >> (64 - self.lam)
            self.calls += 1
            if len(self.cache) < 4096:
                self.cache[m] = t
        return t

    def tag_nocache(self, w: int, y: int) -> int:
        d = hmac.digest(self.key, w.to_bytes(16, "big") + bytes([y]), "sha256")
        self.calls += 1
        return int.from_bytes(d[:8], "big") >> (64 - self.lam)

    def answer(self, w: int, y: int, sigma: int) -> int:
        """The backdoored model's answer at (w, y, sigma)."""
        return y if self.tag_nocache(w, y) == sigma else self.honest(w)


def low_weight_order(lam: int, count: int) -> list:
    """Tags of Hamming weight 0, 1, 2, ... in lexicographic order, as many as requested."""
    from itertools import combinations
    out = []
    for k in range(lam + 1):
        for c in combinations(range(lam), k):
            out.append(sum(1 << i for i in c))
            if len(out) >= count:
                return out
    return out


_ORDERS = {}


def first_trigger(model: KeyedModel, strategy: str, qmax: int, gen) -> int:
    """Index (1-based) of the first query that is a trigger, or qmax + 1 if none in qmax queries."""
    lam = model.lam
    if strategy == "enumerate":
        w = int(gen.integers(0, 2 ** WBITS))
        y = 1 - model.honest(w)
        tags = gen.permutation(2 ** lam)[: min(qmax, 2 ** lam)]
        true = model.tag(w, y)
        hit = np.flatnonzero(tags == true)
        return int(hit[0]) + 1 if hit.size else qmax + 1
    if strategy == "random":
        ws = gen.integers(0, 2 ** WBITS, size=qmax)
        sig = gen.integers(0, 2 ** lam, size=qmax, dtype=np.uint64)
    else:  # near-data
        bits = gen.random((qmax, WBITS)) < 0.2
        ws = (bits * (1 << np.arange(WBITS, dtype=np.int64))).sum(axis=1)
        key = (lam, qmax)
        if key not in _ORDERS:
            order = low_weight_order(lam, qmax)
            _ORDERS[key] = np.array([order[j % len(order)] for j in range(qmax)], dtype=np.int64)
        sig = _ORDERS[key]
    ys = 1 - parity(ws & model.a)                  # the auditor always asks for the wrong label
    for j, (w, y, sg) in enumerate(zip(ws.tolist(), ys.tolist(), sig.tolist())):
        if model.tag_nocache(w, y) == sg:
            return j + 1
    return qmax + 1


def audits(gen, key, a):
    """Experiment A: detection against the number of queries, three strategies, three tag lengths."""
    print("A. black-box audits of the keyed backdoor (auditor holds the honest model)")
    for lam in LAMS:
        model = KeyedModel(key, a, lam)
        qmax = 2 ** (lam + 1)
        qs = [2 ** (lam - 6 + i) for i in range(8)]          # q / 2^lam = 1/64, ..., 2
        first = {}
        for strategy in ("random", "near-data", "enumerate"):
            first[strategy] = np.array([first_trigger(model, strategy, qmax, gen) for _ in range(N_AUDIT)])
        rows = []
        print(f"  lam = {lam}: {N_AUDIT} audits per strategy; HMAC calls {model.calls}")
        print("     q   q/2^lam   bound  1-(1-2^-lam)^q   random   near-data  enumerate")
        for q in qs:
            row = [q, q / 2 ** lam]
            line = f"  {q:5d}  {q / 2 ** lam:7.4f}  {min(1.0, q / 2 ** lam):6.4f}   {1 - (1 - 2.0 ** -lam) ** q:8.4f}     "
            for strategy in ("random", "near-data", "enumerate"):
                k = int(np.sum(first[strategy] <= q))
                lo, hi = pk.clopper_pearson(k, N_AUDIT)
                rate = k / N_AUDIT
                if k == 0:
                    row += [float("nan")] * 3
                else:
                    row += [rate, rate - lo, hi - rate]
                line += f"{rate:7.4f}    "
            rows.append(row)
            print(line)
        pk.write_dat(f"figures/data/15-backdoors-audit-{lam}.dat",
                     ["q", "qn", "rnd", "rndem", "rndep", "near", "nearem", "nearep", "enu", "enuem", "enuep"], rows)
    grid = np.unique(np.round(np.logspace(0, 15 * np.log10(2), 200)).astype(np.int64))  # q up to 2^15
    trows = []
    for q in grid:
        row = [int(q)]
        for lam in LAMS:
            row += [min(1.0, q * 2.0 ** -lam), 1 - (1 - 2.0 ** -lam) ** q]
        trows.append(row)
    pk.write_dat("figures/data/15-backdoors-audit-theory.dat",
                 ["q"] + [f"{n}{lam}" for lam in LAMS for n in ("b", "e")], trows)


def mitigation(gen, key):
    """Experiment B: error of the mitigated value at the targets against the number of trials t."""
    print("\nB. mitigation by self-correction on F_2^20")
    N = 2 ** NB
    z = np.arange(N, dtype=np.int64)
    a = int(gen.integers(1, N))
    L = parity(z & a)
    # keyed pseudorandom values u(x) in [0, 1), from HMAC of each point
    u = np.empty(N)
    for x in range(N):
        d = hmac.digest(key, x.to_bytes(4, "big"), "sha256")
        u[x] = int.from_bytes(d[:8], "big") / 2.0 ** 64
    xstar = int(gen.integers(1, N))
    k = int(np.flatnonzero([(xstar >> b) & 1 for b in range(NB)])[0])
    half = z[((z >> k) & 1) == 0]
    half = half[half != 0]
    cols, rows_by_t = [], {t: [t] for t in T_GRID}
    theory = {}
    for delta in (0.05, 0.10, 0.15):
        m = int(round(delta * N))
        # worst case: E and E + x* disjoint
        E = np.concatenate([[xstar], gen.choice(half, size=m - 1, replace=False)])
        g = L.copy(); g[E] ^= 1
        assert not np.any(np.isin(E ^ xstar, E))
        p_one = float(np.mean(g[xstar ^ z] ^ g[z] != L[xstar]))
        print(f"  delta = {delta:.2f}, worst case: one trial errs with probability {p_one:.4f} (2 delta = {2 * m / N:.4f})")
        theory[("adv", delta)] = 2 * m / N
        for t in T_GRID:
            r = gen.integers(0, N, size=(N_MIT, t))
            wrong = (g[xstar ^ r] ^ g[r]) != L[xstar]
            err = (wrong.sum(axis=1) * 2 > t)
            kk = int(err.sum())
            lo, hi = pk.clopper_pearson(kk, N_MIT)
            rate = kk / N_MIT
            rows_by_t[t] += ([rate, rate - lo, hi - rate] if kk > 0 else [float("nan")] * 3)
        cols += [f"adv{int(100 * delta):02d}", f"adv{int(100 * delta):02d}em", f"adv{int(100 * delta):02d}ep"]
    # keyed placement at delta = 0.15: E = {u < delta}, targets drawn from E
    delta = 0.15
    E_mask = u < delta
    g = L ^ E_mask.astype(np.int64)
    targets = gen.choice(np.flatnonzero(E_mask), size=400, replace=False)
    p_t = np.array([np.mean(g[x ^ z] ^ g[z] != L[x]) for x in targets])
    print(f"  delta = {delta:.2f}, keyed: density {E_mask.mean():.4f}; one trial errs with probability "
          f"{p_t.mean():.4f} on average over 400 targets (2 delta (1 - delta) = {2 * delta * (1 - delta):.4f})")
    for t in T_GRID:
        idx = gen.integers(0, len(targets), size=N_MIT)
        xs = targets[idx][:, None]
        r = gen.integers(0, N, size=(N_MIT, t))
        wrong = (g[xs ^ r] ^ g[r]) != L[xs]
        err = (wrong.sum(axis=1) * 2 > t)
        kk = int(err.sum())
        lo, hi = pk.clopper_pearson(kk, N_MIT)
        rate = kk / N_MIT
        rows_by_t[t] += ([rate, rate - lo, hi - rate] if kk > 0 else [float("nan")] * 3)
    cols += ["key15", "key15em", "key15ep"]
    before = float(np.mean(g[targets] != L[targets]))
    print(f"  before mitigation the backdoored model is wrong at {before:.3f} of its targets")
    pk.write_dat("figures/data/15-backdoors-mitigate.dat", ["t"] + cols, [rows_by_t[t] for t in T_GRID])
    print("      t   worst 0.05 (exact)    worst 0.10 (exact)    worst 0.15 (exact)    keyed 0.15 (exact)")
    for i, t in enumerate(T_GRID):
        row = rows_by_t[t]
        exact = [float(stats.binom.sf((t - 1) // 2, t, theory[("adv", d)])) for d in (0.05, 0.10, 0.15)]
        exk = float(np.mean(stats.binom.sf((t - 1) // 2, t, p_t)))
        print(f"  {t:5d}   " + "   ".join(f"{row[1 + 3 * j]:8.5f} ({exact[j]:.5f})" for j in range(3))
              + f"   {row[10]:8.5f} ({exk:.5f})")
    # theory curves at every odd t, and the Hoeffding bound
    trows = []
    for t in range(1, 43, 2):
        row = [t]
        for d in (0.05, 0.10, 0.15):
            row += [float(stats.binom.sf((t - 1) // 2, t, theory[("adv", d)])), float(np.exp(-2 * t * (0.5 - 2 * d) ** 2))]
        row += [float(np.mean(stats.binom.sf((t - 1) // 2, t, p_t)))]
        trows.append(row)
    pk.write_dat("figures/data/15-backdoors-mitigate-theory.dat",
                 ["t", "ex05", "ho05", "ex10", "ho10", "ex15", "ho15", "exkey"], trows)


def removal(gen, key):
    """Experiment C: the keyed backdoor at realistic tag lengths, audited and then mitigated."""
    print("\nC. undetectable, then removed (w of 32 bits, x = (w, y, sigma) of 33 + lam bits)")
    a = int(gen.integers(1, 2 ** WBITS))
    q, n_aud, n_act = 10_000, 100, 2000
    print("   lam   audits detecting (q = 10^4)   bound q 2^-lam   flipped before   after t = 1   after t = 9")
    for lam in (8, 16, 32, 64):
        model = KeyedModel(key, a, lam)
        nbits = WBITS + 1 + lam
        det = 0
        for _ in range(n_aud):
            det += first_trigger(model, "random", q, gen) <= q
        lo, hi = pk.clopper_pearson(det, n_aud)

        def ftilde(x: int) -> int:
            sigma = x & ((1 << lam) - 1)
            y = (x >> lam) & 1
            w = x >> (lam + 1)
            return model.answer(w, y, sigma)

        def honest(x: int) -> int:
            return model.honest(x >> (lam + 1))

        flipped = {0: 0, 1: 0, 9: 0}
        for _ in range(n_act):
            w = int(gen.integers(0, 2 ** WBITS))
            b = 1 - model.honest(w)                      # the attacker's target: the wrong label
            xa = (w << (lam + 1)) | (b << lam) | model.tag_nocache(w, b)
            flipped[0] += ftilde(xa) == b
            for t in (1, 9):
                votes = 0
                for _ in range(t):
                    r = int.from_bytes(gen.bytes((nbits + 7) // 8), "big") & ((1 << nbits) - 1)
                    votes += ftilde(xa ^ r) ^ ftilde(r)
                flipped[t] += int(votes * 2 > t) == b
        print(f"  {lam:4d}   {det:3d}/{n_aud} [{lo:.3f}, {hi:.3f}]          {min(1, q * 2.0 ** -lam):.3g}"
              f"          {flipped[0] / n_act:.3f}           {flipped[1] / n_act:.4f}        {flipped[9] / n_act:.4f}")


def main():
    gen = pk.rng(15)
    key = gen.bytes(32)                                  # the developer's secret key K
    a = int(gen.integers(1, 2 ** WBITS))
    audits(gen, key, a)
    mitigation(pk.rng(1515), key)
    removal(pk.rng(151515), key)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"\nruntime {time.perf_counter() - t0:.1f} s")
