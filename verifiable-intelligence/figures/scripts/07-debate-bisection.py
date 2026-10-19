#!/usr/bin/env python3
"""Chapter 7 (debate): the bisection debate over a long deterministic computation.

The computation is T steps of a deterministic map delta on 64-bit words (three xor-shifts and a
multiplication by an odd constant, a bijection of {0,1}^64), started from a seeded word c_0. The
claim is "the top bit of c_T = delta^T(c_0) is b". The judge of the bisection debate (Protocol 7.2)
reads the claimed final configuration, the bits by which B chooses halves, and the two
configurations at the end of the bisection, and evaluates delta once.

Provers:
  honest A        claims the true bit; sends true midpoints, each computed from the current left
                  endpoint; its opponent B chooses halves at random (completeness holds against every B);
  one-step liar   claims the false bit: it picks a step u uniformly in {1..T}, replaces delta(c_{u-1})
                  by delta(c_{u-1}) XOR e for a random nonzero e, runs delta from there, and retries e
                  until the top bit of its c'_T is the false bit; its midpoints come from this run;
  honest B        against the liar: in every round it simulates m - i steps from the current left
                  endpoint and chooses the left half if A's midpoint is wrong, the right half otherwise.

Baseline without debate: the liar sends its whole run c'_1..c'_T, and a judge that checks k steps
chosen uniformly without replacement accepts if all of them are correct. The one-step liar is then
caught with probability exactly k/T, and no liar does better (Exercise 7.3). Since the
catch probability does not depend on u, the spot-check trials reuse SPOT_RUNS fabricated runs, each
checked SPOT_PER_RUN times with fresh samples (independent trials).

Writes figures/data/07-debate-bisection.dat (catch probabilities against T with Clopper-Pearson 95%
intervals), figures/data/07-debate-bisection-theory.dat (exact curves) and
figures/data/07-debate-bisection-costs.dat (costs of single debates for large T).
Run from the volume root:  python3 figures/scripts/07-debate-bisection.py   (about 20 s on a laptop)
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

MASK = (1 << 64) - 1
MULT = 0x2545F4914F6CDD1D          # an odd constant, so multiplication is a bijection mod 2^64
S_BITS = 64                         # size of a configuration in bits
LOG_T = list(range(2, 13))          # T = 4, 8, ..., 4096 for the catch probabilities
TRIALS = 2000                       # debates against the one-step liar per T
HONEST_TRIALS = 200                 # debates of honest A per T
SPOT_RUNS, SPOT_PER_RUN = 50, 4000  # spot-check trials per T: 200,000
KS = (1, 4)                         # spot-check judges: number of steps checked
COST_LOG_T = (10, 14, 18, 22)       # single debates for the cost table
SEED = 77
OUT = "figures/data/07-debate-bisection.dat"
OUT_THEORY = "figures/data/07-debate-bisection-theory.dat"
OUT_COSTS = "figures/data/07-debate-bisection-costs.dat"


def delta(c):
    """One step of the computation: a nonlinear bijection of 64-bit words."""
    c ^= c >> 12
    c ^= (c << 25) & MASK
    c ^= c >> 27
    return (c * MULT) & MASK


def run(c, steps):
    for _ in range(steps):
        c = delta(c)
    return c


def top(c):
    return c >> 63


def word(gen):
    return int(gen.integers(0, 1 << 63)) | (int(gen.integers(0, 2)) << 63)


class Counter:
    def __init__(self):
        self.steps = 0


# ----------------------------------------------------------------------------- the protocol
def bisection(c0, T, b, claim_cT, midpoint, choose):
    """Protocol 7.2 on the claim "the top bit of c_T is b". midpoint(i, j, m) returns A's
    configuration at step m; choose(i, m, j, c_i, c_m) returns 'L' or 'R'. Returns (A wins?,
    rounds, bits of the transcript the judge reads). The judge knows c0, b and T."""
    rounds, bits_read = 0, S_BITS            # the judge reads A's final configuration
    if top(claim_cT) != b:
        return False, 0, bits_read            # A's final configuration contradicts its own claim
    i, j = 0, T
    ci, cj = c0, claim_cT
    while j - i >= 2:
        m = (i + j) // 2
        cm = midpoint(i, j, m)
        side = choose(i, m, j, ci, cm)
        rounds += 1
        bits_read += 1                        # B's choice
        if side == 'L':
            j, cj = m, cm
        else:
            i, ci = m, cm
    # the leaf: read the endpoints that are A's midpoints (c0 is the judge's own, c_T already read)
    bits_read += S_BITS * ((i != 0) + (j != T))
    return delta(ci) == cj, rounds, bits_read  # the judge evaluates delta once


def honest_a_debate(c0, T, gen, count):
    """Honest A (true claim) against a B that chooses halves at random."""
    cT = run(c0, T)
    count.steps += T
    known = {0: c0, T: cT}

    def midpoint(i, j, m):
        c = run(known[i], m - i)              # from the current left endpoint
        count.steps += m - i
        known[m] = c
        return c

    def choose(i, m, j, ci, cm):
        return 'L' if gen.integers(0, 2) == 0 else 'R'

    return bisection(c0, T, top(cT), cT, midpoint, choose)


def true_run(c0, T):
    out = [c0]
    for _ in range(T):
        out.append(delta(out[-1]))
    return out


def liar_run(true, T, gen):
    """The one-step liar's fabricated run c'_0..c'_T and its corrupted step u; the top bit of c'_T
    differs from that of the true c_T."""
    want = 1 - top(true[T])
    u = int(gen.integers(1, T + 1))
    while True:
        e = word(gen)
        if e == 0:
            continue
        fake = true[:u] + [delta(true[u - 1]) ^ e]
        for _ in range(u, T):
            fake.append(delta(fake[-1]))
        if top(fake[T]) == want:
            return fake, u


def liar_debate(fake, T, count):
    """The one-step liar as A against honest B."""
    def midpoint(i, j, m):
        return fake[m]

    def choose(i, m, j, ci, cm):
        c = run(ci, m - i)                    # B checks the left half by simulation
        count.steps += m - i
        return 'L' if c != cm else 'R'

    return bisection(fake[0], T, top(fake[T]), fake[T], midpoint, choose)


def spot_check(fake, T, k, gen):
    """A judge without debate: check k distinct steps of the whole claimed run."""
    ts = range(1, T + 1) if k >= T else gen.choice(T, size=k, replace=False) + 1
    return all(delta(fake[t - 1]) == fake[t] for t in ts)


def main():
    t0 = time.time()
    gen = pk.rng(SEED)
    rows, theory = [], []
    print("catch probabilities and costs per T:")
    for L in LOG_T:
        T = 2 ** L
        c0 = word(gen)
        true = true_run(c0, T)
        cnt = Counter()
        hon = 0
        for _ in range(HONEST_TRIALS):
            ok, rounds, bits = honest_a_debate(c0, T, gen, cnt)
            hon += ok
        caught, cntB = 0, Counter()
        max_bits, max_rounds = 0, 0
        fakes = []
        for t in range(TRIALS):
            fake, u = liar_run(true, T, gen)
            ok, rounds, bits = liar_debate(fake, T, cntB)
            caught += (not ok)
            max_bits, max_rounds = max(max_bits, bits), max(max_rounds, rounds)
            if t < SPOT_RUNS:
                fakes.append(fake)
        spot_n = SPOT_RUNS * SPOT_PER_RUN
        spot_caught = {k: sum(not spot_check(f, T, k, gen) for f in fakes for _ in range(SPOT_PER_RUN))
                       for k in KS}
        hr, hlo, hhi = hon / HONEST_TRIALS, *pk.clopper_pearson(hon, HONEST_TRIALS)
        cr, clo, chi = caught / TRIALS, *pk.clopper_pearson(caught, TRIALS)
        row = [T, L, hr, hr - hlo, hhi - hr, cr, cr - clo, chi - cr]
        line = (f"T=2^{L:2d}: honest A wins {hon}/{HONEST_TRIALS}; liar caught {caught}/{TRIALS} "
                f"[{clo:.4f},{chi:.4f}]; rounds {max_rounds}; judge reads <= {max_bits} bits "
                f"(bound 3s+log T = {3 * S_BITS + math.ceil(math.log2(T))}); B steps/debate "
                f"{cntB.steps / TRIALS:.1f} (T-1 = {T - 1}); A steps/debate {cnt.steps / HONEST_TRIALS:.1f} "
                f"(2T-1 = {2 * T - 1})")
        for k in KS:
            r = spot_caught[k] / spot_n
            lo, hi = pk.clopper_pearson(spot_caught[k], spot_n)
            ex = min(1.0, k / T)
            row += [r, r - lo, hi - r]
            line += (f"\n           spot checks k={k}: caught {spot_caught[k]}/{spot_n} = {r:.6f} "
                     f"[{lo:.6f},{hi:.6f}] exact {ex:.6f}" + ("" if lo <= ex <= hi else "  <- outside"))
        rows.append(row)
        print(line)
    header = ["T", "logT", "honest", "honestem", "honestep", "debate", "debateem", "debateep"]
    for k in KS:
        header += [f"spot{k}", f"spot{k}em", f"spot{k}ep"]
    pk.write_dat(OUT, header, rows)
    for i in range(201):
        x = 2 + 0.05 * i
        T = 2 ** x
        theory.append([T, x, 1.0, 1.0] + [min(1.0, k / T) for k in KS])
    pk.write_dat(OUT_THEORY, ["T", "logT", "honest", "debate"] + [f"spot{k}" for k in KS], theory)
    print(f"rates done in {time.time() - t0:.1f} s")

    # ------------------------------------------------------------------ costs of single debates
    print("\ncosts of single debates (honest A vs random B; liar vs honest B):")
    cost_rows = []
    for L in COST_LOG_T:
        T = 2 ** L
        c0 = word(gen)
        cnt = Counter()
        t1 = time.perf_counter()
        ok, rounds, bits = honest_a_debate(c0, T, gen, cnt)
        t_honest = time.perf_counter() - t1
        t1 = time.perf_counter()
        true = true_run(c0, T)
        t_replay = time.perf_counter() - t1              # replaying the computation once
        fake, u = liar_run(true, T, gen)
        cntB = Counter()
        t1 = time.perf_counter()
        lok, lrounds, lbits = liar_debate(fake, T, cntB)
        t_b = time.perf_counter() - t1
        # the judge's own work: index bookkeeping for the rounds and one evaluation of delta
        reps = 2000
        t1 = time.perf_counter()
        for _ in range(reps):
            i, j = 0, T
            while j - i >= 2:
                i = (i + j) // 2
            _ = delta(fake[T - 1]) == fake[T]
        t_judge = (time.perf_counter() - t1) / reps
        transcript = S_BITS * (rounds + 1) + rounds
        cost_rows.append([T, L, rounds, transcript, max(bits, lbits), cnt.steps, cntB.steps,
                          t_judge * 1e6, t_honest * 1e3, t_b * 1e3, t_replay * 1e3])
        print(f"T=2^{L}: honest A wins: {ok}; liar wins: {lok} (u={u}); rounds {rounds}; transcript "
              f"{transcript} bits; judge reads {max(bits, lbits)} bits and evaluates delta once, "
              f"{t_judge * 1e6:.2f} us; honest A {cnt.steps} steps ({t_honest * 1e3:.3g} ms); honest B "
              f"{cntB.steps} steps ({t_b * 1e3:.3g} ms); one replay of the computation {t_replay * 1e3:.3g} ms")
    pk.write_dat(OUT_COSTS, ["T", "logT", "rounds", "transcript", "judgebits", "Asteps", "Bsteps",
                             "judgeus", "Ams", "Bms", "replayms"], cost_rows)
    print(f"done in {time.time() - t0:.1f} s; wrote {OUT}, {OUT_THEORY}, {OUT_COSTS}")


if __name__ == "__main__":
    main()
