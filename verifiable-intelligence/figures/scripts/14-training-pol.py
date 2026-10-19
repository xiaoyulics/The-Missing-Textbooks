"""14-training-pol.py -- logged-checkpoint spot checks (Protocol 14.1) on a small training run.

Chapter 14 (Proofs of Training) of Verifiable Intelligence. Run from the volume root:

    python3 figures/scripts/14-training-pol.py

A developer trains a logistic-regression model with minibatch SGD in float32 (d = 32 weights,
4096 examples, batches of 64, learning rate 0.5, T = 1600 steps), logs a checkpoint every
Delta = 8 steps (N = 200 segments), and commits to the log with a Merkle tree of SHA-256 hashes
(keyed by a random prefix drawn by the verifier). The verifier draws q segments uniformly with
replacement, checks the openings against the root, replays each segment, and accepts if every
replayed checkpoint lies within tolerance tau (l2 distance) of the logged one.

Experiments:
  (a) detection: logs in which e = 4 or e = 20 of the 200 segments were never computed (the forger
      extrapolates the previous update), against q, with the exact value 1-(1-e/N)^q;
  (b) tolerance: acceptance against tau, with q = 10, of the honest log replayed on
      "nondeterministic hardware" (each replayed step sums the per-example gradients in a fresh
      random order, so float32 rounding differs), and of interpolation spoofs (Fang et al.'s
      infinitesimal-update attack) that run no training at all;
  (c) costs of one audit.

Writes figures/data/14-training-pol-*.dat and prints every number the chapter quotes.
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

# ------------------------------------------------------------------ the training specification
D_FEAT = 32          # number of weights
M_EX = 4096          # examples
BATCH = 64           # batch size
ETA = F32(0.5)       # specified learning rate
DELTA = 8            # optimizer steps per segment (checkpoint interval)
N_SEG = 200          # segments
T_STEPS = DELTA * N_SEG

gen = pk.rng(1401)
X = (gen.standard_normal((M_EX, D_FEAT)) / np.sqrt(D_FEAT)).astype(F32)
W_TRUE = (3.0 * gen.standard_normal(D_FEAT)).astype(F32)
P_TRUE = 1.0 / (1.0 + np.exp(-(X.astype(np.float64) @ W_TRUE.astype(np.float64))))
Y = (gen.random(M_EX) < P_TRUE).astype(F32)
# the data order: one permutation of the examples per epoch, from the specification's seed
_order_gen = pk.rng(1402)
_per_epoch = M_EX // BATCH
BATCHES = []
for _ in range(-(-T_STEPS // _per_epoch)):
    perm = _order_gen.permutation(M_EX)
    BATCHES.extend(perm[i * BATCH:(i + 1) * BATCH] for i in range(_per_epoch))
BATCHES = BATCHES[:T_STEPS]
W0 = (0.1 * pk.rng(1403).standard_normal(D_FEAT)).astype(F32)   # the specified initialization
G_BOUND = float(np.max(np.linalg.norm(X.astype(np.float64), axis=1)))  # bound on every batch gradient


def sgd_step(w, t, eta=ETA, order=None):
    """One SGD step on batch t (0-based). order: summation order of the per-example gradients."""
    idx = BATCHES[t]
    xb, yb = X[idx], Y[idx]
    r = (F32(1) / (F32(1) + np.exp(-(xb @ w)))) - yb
    if order is not None:
        xb, r = xb[order], r[order]
    g = (xb.T @ r) / F32(BATCH)
    return (w - F32(eta) * g).astype(F32)


def segment(w, j, eta=ETA, gen=None, delta=DELTA):
    """Replay segment j (1-based): steps (j-1)*delta .. j*delta-1. With gen, each step sums in a
    fresh random order (nondeterministic hardware); without, in the trainer's order (exact replay)."""
    for t in range((j - 1) * delta, j * delta):
        order = None if gen is None else gen.permutation(BATCH)
        w = sgd_step(w, t % T_STEPS, eta, order)
    return w


def loss_acc(w):
    z = X.astype(np.float64) @ w.astype(np.float64)
    loss = float(np.mean(np.logaddexp(0.0, z) - Y * z))
    acc = float(np.mean((z > 0) == (Y > 0.5)))
    return loss, acc


# ------------------------------------------------------------------ Merkle commitments
class Merkle:
    """Merkle tree over byte strings (leaf = H(v), node = H(left || right)), padded with b''."""

    def __init__(self, key: bytes, values: list[bytes]):
        self.key = key
        self.n = len(values)
        self.h = max(1, (self.n - 1).bit_length())
        leaves = [self.H(v) for v in values] + [self.H(b"")] * (2 ** self.h - self.n)
        self.levels = [leaves]
        while len(self.levels[-1]) > 1:
            lv = self.levels[-1]
            self.levels.append([self.H(lv[2 * i] + lv[2 * i + 1]) for i in range(len(lv) // 2)])
        self.root = self.levels[-1][0]
        self.hashes_to_build = 2 ** (self.h + 1) - 1

    def H(self, data: bytes) -> bytes:
        return hashlib.sha256(self.key + data).digest()

    def path(self, i: int) -> list[bytes]:
        return [self.levels[l][(i >> l) ^ 1] for l in range(self.h)]


def verify_opening(key: bytes, root: bytes, i: int, value: bytes, path: list[bytes]) -> bool:
    x = hashlib.sha256(key + value).digest()
    for l, sib in enumerate(path):
        x = hashlib.sha256(key + (sib + x if (i >> l) & 1 else x + sib)).digest()
    return x == root


class CommittedLog:
    """A prover that has committed to a log of checkpoints and opens positions on request."""

    def __init__(self, log: list[np.ndarray], key: bytes, eta=ETA):
        self.log = log
        self.N = len(log) - 1
        self.eta = eta                     # the learning rate the log reports
        self.tree = Merkle(key, [w.tobytes() for w in log])

    def open(self, i):
        return self.log[i].tobytes(), self.tree.path(i)


def audit(prover: CommittedLog, key: bytes, w_init, w_final, q: int, tau: float, gen,
          exact: bool = False, delta=DELTA, count=None) -> bool:
    """Protocol 14.1: draw q segments with replacement, open, verify, replay; early rejection."""
    N, root = prover.N, prover.tree.root
    js = gen.integers(1, N + 1, size=q)
    positions = sorted({0, N} | {int(j) - 1 for j in js} | {int(j) for j in js})
    opened = {}
    for p in positions:
        value, path = prover.open(p)
        if count is not None:
            count["hashes"] += len(path) + 1
            count["bytes"] += len(value) + 32 * len(path)
        if not verify_opening(key, root, p, value, path):
            return False
        opened[p] = np.frombuffer(value, dtype=F32)
    if not (np.array_equal(opened[0], w_init) and np.array_equal(opened[N], w_final)):
        return False
    for j in js:
        j = int(j)
        replay = segment(opened[j - 1], j, prover.eta, None if exact else gen, delta)
        if count is not None:
            count["steps"] += delta
        if float(np.linalg.norm((replay - opened[j]).astype(np.float64))) > tau:
            return False
    return True


def rate_row(k, n):
    lo, hi = pk.clopper_pearson(k, n)
    r = k / n
    return r, hi - r, r - lo


def main():
    t_start = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    key = pk.rng(1404).bytes(16)          # the hash key, drawn by the verifier before the commitment

    # ---------------------------------------------------------------- honest training and log
    t0 = time.perf_counter()
    log = [W0.copy()]
    w = W0.copy()
    for j in range(1, N_SEG + 1):
        w = segment(w, j)
        log.append(w)
    t_train = time.perf_counter() - t0
    w_final = log[-1]
    t0 = time.perf_counter()
    honest = CommittedLog(log, key)
    t_tree = time.perf_counter() - t0
    l0, a0 = loss_acc(W0)
    l1, a1 = loss_acc(w_final)
    disp = [float(np.linalg.norm((log[j] - log[j - 1]).astype(np.float64))) for j in range(1, N_SEG + 1)]
    print(f"training: d={D_FEAT} m={M_EX} B={BATCH} eta={float(ETA)} Delta={DELTA} N={N_SEG} T={T_STEPS}")
    print(f"  loss {l0:.4f} -> {l1:.4f}, accuracy {a0:.4f} -> {a1:.4f}; G = max|x| = {G_BOUND:.4f}")
    print(f"  segment displacement |W_j - W_(j-1)|: min {min(disp):.4g} median {np.median(disp):.4g} max {max(disp):.4g}")
    print(f"  Merkle tree: {N_SEG + 1} leaves, depth h={honest.tree.h}, {honest.tree.hashes_to_build} hashes to build")

    # ---------------------------------------------------------------- replay errors of the honest log
    gen_nd = pk.rng(1405)
    exact_err = max(float(np.linalg.norm((segment(log[j - 1], j) - log[j]).astype(np.float64)))
                    for j in range(1, N_SEG + 1))
    nd_err = []
    for _ in range(5):
        for j in range(1, N_SEG + 1):
            nd_err.append(float(np.linalg.norm((segment(log[j - 1], j, gen=gen_nd) - log[j]).astype(np.float64))))
    nd_err = np.array(nd_err)
    print(f"honest log: exact replay max error {exact_err:.3g}; nondeterministic replay (5 x {N_SEG} segments): "
          f"median {np.median(nd_err):.3g}, max {nd_err.max():.3g}, fraction nonzero {np.mean(nd_err > 0):.3f}")
    # replaying the whole run in one piece on nondeterministic hardware
    w_nd = W0.copy()
    for j in range(1, N_SEG + 1):
        w_nd = segment(w_nd, j, gen=gen_nd)
    print(f"  whole run replayed in one piece (nondeterministic): final distance {np.linalg.norm((w_nd - w_final).astype(np.float64)):.3g}")

    # ---------------------------------------------------------------- (a) forged logs: detection against q
    TAU = 1e-4
    qs = [1, 2, 3, 5, 8, 12, 18, 27, 40, 60, 90]
    n_trials = 1000
    forged = {}
    for e, seed in [(4, 1406), (20, 1407)]:
        bad = set(int(j) for j in pk.rng(seed).choice(np.arange(1, N_SEG + 1), size=e, replace=False))
        flog = [W0.copy()]
        w = W0.copy()
        for j in range(1, N_SEG + 1):
            if j in bad:
                prev = flog[-2] if j >= 2 else flog[-1]
                w = (flog[-1] + (flog[-1] - prev)).astype(F32)   # extrapolate: no gradient computed
            else:
                w = segment(w, j)
            flog.append(w)
        errs = {j: float(np.linalg.norm((segment(flog[j - 1], j) - flog[j]).astype(np.float64))) for j in range(1, N_SEG + 1)}
        min_bad = min(errs[j] for j in bad)
        max_good = max(errs[j] for j in errs if j not in bad)
        forged[e] = CommittedLog(flog, key)
        print(f"forged log e={e} (eps={e / N_SEG}): forged-segment errors >= {min_bad:.4g}; other segments (exact replay) <= {max_good:.3g}; "
              f"final model differs from honest by {np.linalg.norm((flog[-1] - w_final).astype(np.float64)):.3g}")
    rows, worst = [], 0.0
    gen_a = pk.rng(1408)
    for q in qs:
        row = [q]
        for e in (4, 20):
            prov = forged[e]
            caught = sum(1 for _ in range(n_trials)
                         if not audit(prov, key, W0, prov.log[-1], q, TAU, gen_a))
            r, ep, em = rate_row(caught, n_trials)
            exact = 1 - (1 - e / N_SEG) ** q
            lo, hi = r - em, r + ep
            inside = lo <= exact <= hi
            worst = max(worst, abs(r - exact))
            row += [r, ep, em]
            gap = 0.0 if inside else min(abs(exact - lo), abs(exact - hi))
            print(f"  q={q:3d} e={e:2d}: caught {caught}/{n_trials} = {r:.3f} [{lo:.4f},{hi:.4f}] exact {exact:.4f} "
                  f"{'ok' if inside else f'OUTSIDE by {gap:.4f}'}")
        rows.append(row)
    pk.write_dat(f"{OUT}/14-training-pol-detect.dat",
                 ["q", "d4", "d4ep", "d4em", "d20", "d20ep", "d20em"], rows)
    qq = np.arange(0, 101)
    pk.write_dat(f"{OUT}/14-training-pol-detect-theory.dat", ["q", "d4", "d20"],
                 [[int(x), 1 - (1 - 4 / N_SEG) ** x, 1 - (1 - 20 / N_SEG) ** x] for x in qq])
    # the honest log at the same tolerance
    acc_h = sum(1 for _ in range(n_trials) if audit(honest, key, W0, w_final, 90, TAU, gen_a))
    print(f"  honest log, q=90, tau={TAU}: accepted {acc_h}/{n_trials}")
    q95 = int(np.ceil(np.log(20) / -np.log(1 - 0.02)))
    print(f"  checks for 95% detection at eps=0.02: q={q95} (ln20/eps = {np.log(20) / 0.02:.1f})")

    # ---------------------------------------------------------------- (b) tolerance: honest replay and spoofs
    w_spoof0 = (0.1 * pk.rng(1409).standard_normal(D_FEAT)).astype(F32)   # a fresh draw from the init distribution
    Dist = float(np.linalg.norm((w_final - w_spoof0).astype(np.float64)))
    eta_rep = F32(ETA * 1e-5)            # the learning rate the spoof reports

    def interpolation(n_seg):
        lam = np.arange(n_seg + 1, dtype=np.float64)[:, None] / n_seg
        L = ((1 - lam) * w_spoof0.astype(np.float64) + lam * w_final.astype(np.float64)).astype(F32)
        L[0], L[-1] = w_spoof0, w_final
        return [L[i] for i in range(n_seg + 1)]

    spoofs = {}
    t0 = time.perf_counter()
    spoofs["A"] = (CommittedLog(interpolation(50_000), key, eta=eta_rep), 50_000, eta_rep)
    t_spoofA = time.perf_counter() - t0
    spoofs["B"] = (CommittedLog(interpolation(N_SEG), key, eta=eta_rep), N_SEG, eta_rep)
    spoofs["C"] = (CommittedLog(interpolation(N_SEG), key, eta=ETA), N_SEG, ETA)
    spoof_err = {}
    for name, (prov, n_seg, eta) in spoofs.items():
        js = range(1, n_seg + 1)
        errs = np.array([float(np.linalg.norm((segment(prov.log[j - 1], j, eta) - prov.log[j]).astype(np.float64))) for j in js])
        spoof_err[name] = errs
        bound = Dist / n_seg + DELTA * float(eta) * G_BOUND
        print(f"spoof {name}: N'={n_seg}, reported eta={float(eta):.3g}: segment errors min {errs.min():.4g} "
              f"median {np.median(errs):.4g} max {errs.max():.4g}; bound D/N'+Delta*eta*G = {bound:.4g}")
    print(f"  D = |w_final - fresh init| = {Dist:.4f}; building spoof A (50,000 segments, interpolation + tree) took {t_spoofA:.2f} s; "
          f"honest training took {t_train * 1000:.1f} ms for {T_STEPS} steps")

    q_tol = 10
    taus = np.logspace(-9, 1, 41)
    gen_b = pk.rng(1410)
    n_tol = 400
    rows = []
    for tau in taus:
        row = [tau]
        k_h = sum(1 for _ in range(n_tol) if audit(honest, key, W0, w_final, q_tol, tau, gen_b))
        row += list(rate_row(k_h, n_tol))
        for name in ("A", "B", "C"):
            prov, n_seg, eta = spoofs[name]
            k_s = sum(1 for _ in range(n_tol) if audit(prov, key, w_spoof0, w_final, q_tol, tau, gen_b))
            row += list(rate_row(k_s, n_tol))
        rows.append(row)
    pk.write_dat(f"{OUT}/14-training-pol-tol.dat",
                 ["tau", "h", "hep", "hem", "A", "Aep", "Aem", "B", "Bep", "Bem", "C", "Cep", "Cem"], rows)
    fine = np.logspace(-9, 1, 401)
    th = []
    for tau in fine:
        th.append([tau] + [float(np.mean(spoof_err[nm] <= tau)) ** q_tol for nm in ("A", "B", "C")])
    pk.write_dat(f"{OUT}/14-training-pol-tol-theory.dat", ["tau", "A", "B", "C"], th)
    # consistency of the measured points with the exact acceptance of the spoofs
    outside = 0
    for row in rows:
        tau = row[0]
        for k, nm in enumerate(("A", "B", "C")):
            r, ep, em = row[4 + 3 * k: 7 + 3 * k]
            ex = float(np.mean(spoof_err[nm] <= tau)) ** q_tol
            if not (r - em - 1e-12 <= ex <= r + ep + 1e-12):
                outside += 1
    print(f"  tolerance sweep: {len(rows)} tau values x 3 spoofs; intervals missing the exact value: {outside}")
    first_h = next(row[0] for row in rows if row[1] == 1.0)
    last_h0 = max(row[0] for row in rows if row[1] == 0.0)
    print(f"  honest log (nondeterministic replay, q={q_tol}): accepted in no trial at tau <= {last_h0:.3g}, in every trial from tau = {first_h:.3g}")
    for nm in ("A", "B", "C"):
        print(f"  spoof {nm}: accepted with probability 1 once tau >= {spoof_err[nm].max():.4g}")

    # ---------------------------------------------------------------- (c) costs of one audit (q = 10)
    count = {"hashes": 0, "bytes": 0, "steps": 0}
    gen_c = pk.rng(1411)
    t0 = time.perf_counter()
    n_cost = 200
    ok = sum(1 for _ in range(n_cost) if audit(honest, key, W0, w_final, q_tol, TAU, gen_c, count=count))
    t_aud = (time.perf_counter() - t0) / n_cost
    print(f"audit cost (q={q_tol}, honest log, {n_cost} audits, all accepted: {ok == n_cost}): "
          f"{count['hashes'] / n_cost:.1f} hashes, {count['bytes'] / n_cost:.0f} bytes of openings, "
          f"{count['steps'] / n_cost:.0f} replayed steps ({count['steps'] / n_cost / T_STEPS:.3f} of training), {t_aud * 1000:.2f} ms")
    print(f"  checkpoint size {W0.nbytes} bytes; full log {W0.nbytes * (N_SEG + 1)} bytes; tree built in {t_tree * 1000:.2f} ms")
    print(f"total runtime {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
