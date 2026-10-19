"""14-training-referee.py -- refereed training (Protocol 14.2): bisection over hashed checkpoints.

Chapter 14 (Proofs of Training) of Verifiable Intelligence. Run from the volume root:

    python3 figures/scripts/14-training-referee.py

Two trainers run the same specification (the logistic-regression run of 14-training-pol.py, for T
steps, with every checkpoint hashed with keyed SHA-256). The data set is committed by a Merkle root
over its 4096 examples; the batch order comes from a public seed. One trainer is honest; the other
cheats in one of four ways: it skips one step, trains one step on a poisoned batch (labels
flipped), stops training halfway and repeats its checkpoint, or reports a different final model
(the honest one plus a perturbation). The referee bisects over the step index, comparing the two
trainers' hashes, and re-executes one step from the agreed checkpoint on the opened batch.

Prints, for T = 2^8 ... 2^16: rounds, the referee's work (steps, hashes, bytes received, time),
the honest trainer's work, and how many debates each cheater won; writes
figures/data/14-training-referee-costs.dat.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

OUT = "figures/data"
F32 = np.float32
D_FEAT, M_EX, BATCH, ETA = 32, 4096, 64, F32(0.5)

gen = pk.rng(1401)                      # the same data as 14-training-pol.py
X = (gen.standard_normal((M_EX, D_FEAT)) / np.sqrt(D_FEAT)).astype(F32)
W_TRUE = (3.0 * gen.standard_normal(D_FEAT)).astype(F32)
P_TRUE = 1.0 / (1.0 + np.exp(-(X.astype(np.float64) @ W_TRUE.astype(np.float64))))
Y = (gen.random(M_EX) < P_TRUE).astype(F32)
W0 = (0.1 * pk.rng(1403).standard_normal(D_FEAT)).astype(F32)
PER_EPOCH = M_EX // BATCH
_perms: dict[int, np.ndarray] = {}


def batch_indices(t):
    """Indices of batch t (1-based step t uses batch t-1): epoch e is permuted by seed 1420+e."""
    e, i = divmod(t - 1, PER_EPOCH)
    if e not in _perms:
        _perms[e] = pk.rng(1420 + e).permutation(M_EX)
    return _perms[e][i * BATCH:(i + 1) * BATCH]


def step(w, xb, yb):
    r = (F32(1) / (F32(1) + np.exp(-(xb @ w)))) - yb
    return (w - ETA * ((xb.T @ r) / F32(BATCH))).astype(F32)


KEY = pk.rng(1404).bytes(16)


def H(b: bytes) -> bytes:
    return hashlib.sha256(KEY + b).digest()


# ---------------------------------------------------------------- the data commitment
EX_BYTES = [X[i].tobytes() + Y[i:i + 1].tobytes() for i in range(M_EX)]
LEVELS = [[H(b) for b in EX_BYTES]]
while len(LEVELS[-1]) > 1:
    lv = LEVELS[-1]
    LEVELS.append([H(lv[2 * i] + lv[2 * i + 1]) for i in range(len(lv) // 2)])
DATA_ROOT = LEVELS[-1][0]
DEPTH = len(LEVELS) - 1


def data_path(i):
    return [LEVELS[l][(i >> l) ^ 1] for l in range(DEPTH)]


def verify_data(i, value, path, count):
    x = H(value)
    count["hashes"] += 1 + len(path)
    for l, sib in enumerate(path):
        x = H(sib + x if (i >> l) & 1 else x + sib)
    return x == DATA_ROOT


# ---------------------------------------------------------------- trainers
class Trainer:
    """Runs a (possibly deviating) training and answers the referee from its own checkpoints."""

    def __init__(self, T, kind="honest", u=None, gen=None):
        self.T, self.kind, self.u = T, kind, u
        t0 = time.perf_counter()
        ck = [W0.copy()]
        w = W0.copy()
        for t in range(1, T + 1):
            idx = batch_indices(t)
            xb, yb = X[idx], Y[idx]
            if kind == "skip" and t == u:
                pass                                   # the step is never computed
            elif kind == "poison" and t == u:
                w = step(w, xb, F32(1) - yb)           # trained on flipped labels
            elif kind == "halt" and t > T // 2:
                pass                                   # stops training, repeats its checkpoint
            else:
                w = step(w, xb, yb)
            ck.append(w)
        if kind == "fake":
            ck[-1] = (ck[-1] + F32(0.05) * gen.standard_normal(D_FEAT).astype(F32)).astype(F32)
        self.ck = ck
        self.hashes = [H(c.tobytes()) for c in ck]
        self.time = time.perf_counter() - t0
        self.steps = T if kind != "halt" else T // 2

    def claim(self, t):
        return self.hashes[t]

    def open_checkpoint(self, t):
        return self.ck[t].tobytes()

    def open_example(self, i):
        return EX_BYTES[i], data_path(i)


def referee(P1: Trainer, P2: Trainer, T):
    """Protocol 14.2. Returns (winner index or 0 on agreement, output hash, cost dict)."""
    cost = {"rounds": 0, "hashes": 0, "bytes": 0, "steps": 0}
    h0 = H(W0.tobytes())
    cost["hashes"] += 1
    y1, y2 = P1.claim(T), P2.claim(T)
    cost["bytes"] += 64
    if y1 == y2:
        return 0, y1, cost
    a, b = 0, T
    agreed = h0
    while b - a >= 2:
        m = (a + b) // 2
        c1, c2 = P1.claim(m), P2.claim(m)
        cost["rounds"] += 1
        cost["bytes"] += 64
        if c1 == c2:
            a, agreed = m, c1
        else:
            b = m
    # the final step: open the agreed checkpoint a and the batch of step b, replay one step
    candidates = set()
    for P in (P1, P2):
        v = P.open_checkpoint(a)
        cost["bytes"] += len(v)
        cost["hashes"] += 1
        if H(v) == agreed:
            candidates.add(v)
    if len(candidates) != 1:
        raise RuntimeError("collision or no valid opening")   # never happens with SHA-256
    w_a = np.frombuffer(candidates.pop(), dtype=F32)
    idx = batch_indices(b)
    rows = []
    for i in idx:                       # one trainer's openings suffice when they verify
        i = int(i)
        value, path = P1.open_example(i)
        cost["bytes"] += len(value) + 32 * len(path)
        if not verify_data(i, value, path, cost):
            value, path = P2.open_example(i)
            if not verify_data(i, value, path, cost):
                raise RuntimeError("no valid data opening")
        rows.append(np.frombuffer(value, dtype=F32))
    rows = np.array(rows)
    w_b = step(w_a, rows[:, :D_FEAT].copy(), rows[:, D_FEAT].copy())
    cost["steps"] += 1
    hb = H(w_b.tobytes())
    cost["hashes"] += 1
    c1, c2 = P1.claim(b), P2.claim(b)
    if hb == c1:
        return 1, y1, cost
    if hb == c2:
        return 2, y2, cost
    raise RuntimeError("neither trainer matches the replayed step")


def main():
    t_start = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    gen = pk.rng(1430)
    kinds = ["skip", "poison", "halt", "fake"]
    rows = []
    print(f"data commitment: Merkle tree over {M_EX} examples, depth {DEPTH}")
    for logT in (8, 10, 12, 14, 16):
        T = 2 ** logT
        honest = Trainer(T)
        wins = {k: 0 for k in kinds}
        runs = {k: 0 for k in kinds}
        rounds_seen, ref_costs, ref_times = set(), [], []
        n_runs = 25 if logT <= 12 else 6
        for kind in kinds:
            for _ in range(n_runs if kind != "halt" else 1):
                u = int(gen.integers(1, T + 1))
                cheat = Trainer(T, kind, u, gen)
                pair = (honest, cheat) if gen.random() < 0.5 else (cheat, honest)
                t0 = time.perf_counter()
                winner, out, cost = referee(*pair, T)
                ref_times.append(time.perf_counter() - t0)
                honest_idx = 1 if pair[0] is honest else 2
                runs[kind] += 1
                if winner != honest_idx or out != honest.claim(T):
                    wins[kind] += 1
                rounds_seen.add(cost["rounds"])
                ref_costs.append(cost)
        # honest against honest: agreement, no rounds
        w0, out0, c0 = referee(honest, Trainer(T), T)
        assert w0 == 0 and out0 == honest.claim(T)
        n_total = sum(runs.values())
        n_won = sum(wins.values())
        med = lambda key: float(np.median([c[key] for c in ref_costs]))  # noqa: E731
        ref_us = float(np.median(ref_times)) * 1e6
        print(f"T=2^{logT}: rounds {sorted(rounds_seen)}; referee: {med('steps'):.0f} step, {med('hashes'):.0f} hashes, "
              f"{med('bytes'):.0f} bytes, {ref_us:.0f} us; honest trainer: {T} steps + {T + 1} hashes in {honest.time * 1000:.1f} ms; "
              f"cheaters won {n_won} of {n_total} ({', '.join(f'{k} {wins[k]}/{runs[k]}' for k in kinds)})")
        rows.append([T, max(rounds_seen), int(med("steps")), int(med("hashes")), int(med("bytes")), ref_us,
                     T, T + 1, honest.time * 1000, n_won, n_total])
    pk.write_dat(f"{OUT}/14-training-referee-costs.dat",
                 ["T", "rounds", "refsteps", "refhashes", "refbytes", "refus", "trsteps", "trhashes", "trms", "cheatwins", "runs"],
                 rows)
    print(f"total runtime {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
