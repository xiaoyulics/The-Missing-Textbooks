"""Chapter 9 (Efficient Debate): the bisection debate for a deterministic oracle machine.

The machine M^H models a long review that consults a human judge H at every step. Its
configuration after tau steps is (tau, count, notes): tau is the step counter, count the number of
"yes" answers so far, and notes a 32-bit summary of everything seen so far. At step tau the machine
asks H a question determined by (tau, notes) -- so later questions depend on earlier answers -- and
folds the answer into count and notes. It accepts if count >= ceil(0.7 T). The human judge H is a
fixed pseudo-random function that answers "yes" to about 70% of questions.

Protocol 9.1 is run as in the chapter: A claims an accepting final configuration, then in each of
k = log2 T rounds A names the configuration at the midpoint of the disputed stretch and B picks the
half to keep; the judge V checks one step, reading one configuration pair and making one query to H.

The script
  1. reproduces the worked run of the chapter (eight studies, threshold five);
  2. measures the costs of the honest debaters and of the judge against T, with the honest
     strategies simulating the machine for real (the counts are compared with 2T-1 and T-1);
  3. runs honest debaters against several cheating strategies on many inputs and reports
     acceptance rates with Clopper-Pearson 95% intervals (every lie should be rejected).
Writes figures/data/09-efficientdebate-costs.dat and prints every number quoted in the chapter.
Run from the volume root: python3 figures/scripts/09-efficientdebate-bisection.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def mix64(z):
    """splitmix64 finalizer: a fixed, fast pseudo-random function on 64-bit integers."""
    z = (z + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


class Oracle:
    """The human judge H: a fixed function of the question; counts the queries it answers."""

    def __init__(self, salt):
        self.salt = salt
        self.queries = 0

    def __call__(self, question):
        self.queries += 1
        tau, notes = question
        return 1 if mix64((tau << 32) ^ notes ^ self.salt) % 1000 < 700 else 0


class Machine:
    """M^H on input x (a 32-bit seed), padded to run exactly T steps (T a power of two)."""

    def __init__(self, T):
        self.T = T
        self.theta = -(-7 * T // 10)          # accept iff count >= ceil(0.7 T)
        self.k = T.bit_length() - 1           # T = 2^k
        self.width = self.k + 1               # bits for tau and for count
        self.C = 2 * self.width + 32          # bits of a configuration (w in the chapter)

    def start(self, x):
        return (0, 0, x & MASK32)

    def is_config(self, c):
        return (isinstance(c, tuple) and len(c) == 3 and 0 <= c[0] <= self.T
                and 0 <= c[1] <= self.T and 0 <= c[2] <= MASK32)

    def step(self, c, H):
        """One step: the successor configuration (one query to H unless halted)."""
        tau, count, notes = c
        if tau >= self.T:                     # a halting configuration is its own successor
            return c
        a = H((tau, notes))
        notes = mix64((notes << 1) ^ a ^ (tau << 33)) & MASK32
        return (tau + 1, count + a, notes)

    def accepting(self, c):
        return self.is_config(c) and c[0] == self.T and c[1] >= self.theta


class Counter:
    """Simulates M for a debater and counts the steps it simulates."""

    def __init__(self, M, H):
        self.M, self.H, self.steps = M, H, 0

    def run(self, c, n):
        for _ in range(n):
            c = self.M.step(c, self.H)
            self.steps += 1
        return c


def debate(M, x, A, B, H_judge):
    """Protocol 9.1. Returns (verdict, bits the judge reads, its queries to H, bits it reads when
    it reads selectively: A's first configuration, B's bits and the final pair)."""
    T, k, C = M.T, M.k, M.C
    v = A.final()
    bits = C
    if not M.accepting(v):
        return False, bits, 0, C
    u, t = M.start(x), T
    for _ in range(k):
        w = A.midpoint(u, v, t)
        b = B.choose(u, w, v, t)
        bits += C + 1
        if b == "L":
            v = w
        else:
            u = w
        t //= 2
    q0 = H_judge.queries
    ok = M.is_config(u) and M.is_config(v) and M.step(u, H_judge) == v
    return ok, bits, H_judge.queries - q0, 3 * C + k


# ---------------------------------------------------------------- honest and cheating debaters
class HonestA:
    def __init__(self, M, x, H):
        self.sim, self.x = Counter(M, H), x

    def final(self):
        return self.sim.run(self.sim.M.start(self.x), self.sim.M.T)

    def midpoint(self, u, v, t):
        return self.sim.run(u, t // 2)


class HonestB:
    def __init__(self, M, H):
        self.sim = Counter(M, H)

    def choose(self, u, w, v, t):
        return "L" if w != self.sim.run(u, t // 2) else "R"


class RandomB:
    def __init__(self, gen):
        self.gen = gen

    def choose(self, u, w, v, t):
        return "L" if self.gen.random() < 0.5 else "R"


class FixedB:
    def __init__(self, side):
        self.side = side

    def choose(self, u, w, v, t):
        return self.side


class ForgerA:
    """Claims a fake accepting final configuration and tells the truth about every midpoint."""

    def __init__(self, M, x, H, gen):
        self.sim, self.x, self.gen = Counter(M, H), x, gen

    def final(self):
        M = self.sim.M
        return (M.T, M.theta + int(self.gen.integers(0, M.T - M.theta + 1)), int(self.gen.integers(0, MASK32)))

    def midpoint(self, u, v, t):
        return self.sim.run(u, t // 2)


class FlipperA:
    """Runs M with a corrupted copy of H: flips just enough 'no' answers to 'yes' to reach the
    threshold, at random steps, and answers every midpoint from its corrupted run."""

    def __init__(self, M, x, H, gen):
        self.M, self.H, self.x, self.gen = M, H, x, gen
        self.flip_at = None

    def _prepare(self):
        M, H = self.M, self.H
        c, nos = M.start(self.x), []
        for _ in range(M.T):              # find the true 'no' answers (and the true count)
            tau, _, notes = c
            if H((tau, notes)) == 0:
                nos.append(tau)
            c = M.step(c, H)
        need = M.theta - c[1]
        idx = self.gen.choice(len(nos), size=need, replace=False) if need > 0 else []
        flips = set(nos[i] for i in idx)
        # the corrupted run is recomputed sequentially: a flip changes the notes, hence later
        # questions, so later 'no' answers can turn into 'yes'; keep flipping until accepting
        H2 = lambda q, flips=flips: 1 if q[0] in flips else H(q)  # noqa: E731
        while True:
            c = M.start(self.x)
            run = [c]
            for _ in range(M.T):
                c = M.step(c, H2)
                run.append(c)
            if c[1] >= M.theta:
                break
            extra = [r[0] for r in run[:-1] if H2((r[0], r[2])) == 0 and r[0] not in flips]
            flips.add(extra[int(self.gen.integers(0, len(extra)))])
        self.run, self.flips = run, flips

    def final(self):
        self._prepare()
        return self.run[-1]

    def midpoint(self, u, v, t):
        return self.run[u[0] + t // 2]


class RandomA:
    """Sends a random accepting final configuration and random midpoints with the right counter."""

    def __init__(self, M, gen):
        self.M, self.gen = M, gen

    def final(self):
        M = self.M
        return (M.T, M.theta + int(self.gen.integers(0, M.T - M.theta + 1)), int(self.gen.integers(0, MASK32)))

    def midpoint(self, u, v, t):
        return (u[0] + t // 2, int(self.gen.integers(0, u[0] + t // 2 + 1)), int(self.gen.integers(0, MASK32)))


# ---------------------------------------------------------------- the worked run of the chapter
def worked_run():
    """Eight studies, the human's verdicts 1,1,0,1,0,1,0,0, accept iff at least five 'yes'.
    Configurations are (tau, count). The liar claims (8,5), tells the truth about the midpoints
    at steps 4 and 6, and claims (7,5) at step 7."""
    verdict = [1, 1, 0, 1, 0, 1, 0, 0]
    true = [(0, 0)]
    for j in range(8):
        true.append((j + 1, true[-1][1] + verdict[j]))
    print("worked run: true configurations", true, "-> accept" if true[-1][1] >= 5 else "-> reject")
    liar = {4: (4, 3), 6: (6, 4), 7: (7, 5)}
    u, v, t = true[0], (8, 5), 8
    rows = []
    while t > 1:
        mid = u[0] + t // 2
        w = liar.get(mid, true[mid])
        b = "L" if w != true[mid] else "R"
        rows.append((u, v, t, w, b))
        if b == "L":
            v = w
        else:
            u = w
        t //= 2
    for r in rows:
        print("  claim %s ->%d %s ; A' says step %d is %s ; B keeps %s" % (r[0], r[2], r[1], r[0][0] + r[2] // 2, r[3], r[4]))
    study = u[0] + 1
    nxt = (u[0] + 1, u[1] + verdict[u[0]])
    print("  final claim %s -> %s in one step: judge asks about study %d, H says %d, successor %s: %s"
          % (u, v, study, verdict[u[0]], nxt, "accept" if nxt == v else "reject"))


# ---------------------------------------------------------------- experiments
def find_inputs(M, H, want, n, gen):
    """n inputs x with M^H(x) = want (True: accept)."""
    xs = []
    while len(xs) < n:
        x = int(gen.integers(0, MASK32))
        c = M.start(x)
        for _ in range(M.T):
            c = M.step(c, H)
        if M.accepting(c) == want:
            xs.append(x)
    return xs


def costs(Tvals, salt):
    rows = []
    gen = pk.rng(9)
    for T in Tvals:
        M = Machine(T)
        H = Oracle(salt)
        xin = find_inputs(M, H, True, 1, gen)[0]
        xout = find_inputs(M, H, False, 1, gen)[0]
        # honest A against an adversarial-at-random B (completeness case)
        HA, HJ = Oracle(salt), Oracle(salt)
        A = HonestA(M, xin, HA)
        ok_in, bits, q, selbits = debate(M, xin, A, RandomB(pk.rng(T)), HJ)
        # honest B against the forger (soundness case)
        HB, HJ2 = Oracle(salt), Oracle(salt)
        B = HonestB(M, HB)
        ok_out, _, q2, _ = debate(M, xout, ForgerA(M, xout, Oracle(salt), pk.rng(T + 1)), B, HJ2)
        assert ok_in and not ok_out
        rows.append((T, A.sim.steps, 2 * T - 1, B.sim.steps, T - 1, HA.queries, bits,
                     (M.k + 1) * M.C + M.k, selbits, q, q2, T))
    return rows


def acceptance(Tvals, n, salt):
    out = {}
    for T in Tvals:
        M = Machine(T)
        H = Oracle(salt)
        gen = pk.rng(100 + T)
        xin = find_inputs(M, H, True, n, gen)
        xout = find_inputs(M, H, False, n, gen)
        res = {}
        for name, mk in [("random B", lambda g: RandomB(g)), ("always left", lambda g: FixedB("L")),
                         ("always right", lambda g: FixedB("R"))]:
            acc = 0
            for i, x in enumerate(xin):
                g = pk.rng(10_000 * T + i)
                ok, *_ = debate(M, x, HonestA(M, x, Oracle(salt)), mk(g), Oracle(salt))
                acc += ok
            res["honest A vs " + name] = acc
        for name, mk in [("forger", lambda x, g: ForgerA(M, x, Oracle(salt), g)),
                         ("flipper", lambda x, g: FlipperA(M, x, Oracle(salt), g)),
                         ("random", lambda x, g: RandomA(M, g))]:
            acc = 0
            for i, x in enumerate(xout):
                g = pk.rng(20_000 * T + i)
                ok, *_ = debate(M, x, mk(x, g), HonestB(M, Oracle(salt)), Oracle(salt))
                acc += ok
            res[name + " A' vs honest B"] = acc
        out[T] = res
    return out


def main():
    t0 = time.time()
    salt = 0x5EED_0009
    worked_run()

    Tvals = [2 ** j for j in range(2, 19, 2)]
    rows = costs(Tvals, salt)
    pk.write_dat("figures/data/09-efficientdebate-costs.dat",
                 ["T", "Asteps", "Abound", "Bsteps", "Bbound", "Aqueries", "Vbits", "Vbitsbound",
                  "Vsel", "Vqueries", "Vqueries2", "direct"], rows)
    print("\ncosts (honest A vs random B on an accepted input; honest B vs forger on a rejected one)")
    print("   T   A steps (2T-1)   B steps (T-1)   judge bits (formula)  selective  judge queries")
    for r in rows:
        print("%7d %9d (%7d) %9d (%7d) %8d (%8d) %8d %4d %4d" % (r[0], r[1], r[2], r[3], r[4], r[6], r[7], r[8], r[9], r[10]))

    n = 200
    acc = acceptance([2 ** 4, 2 ** 8, 2 ** 12], n, salt)
    print("\nacceptance rates, %d inputs per cell, Clopper-Pearson 95%% intervals" % n)
    total_lies, total_lie_acc = 0, 0
    for T, res in acc.items():
        for name, k in res.items():
            lo, hi = pk.clopper_pearson(k, n)
            print("  T=%5d %-28s %3d/%d  [%.4f, %.4f]" % (T, name, k, n, lo, hi))
            if "A'" in name:
                total_lies += n
                total_lie_acc += k
    lo, hi = pk.clopper_pearson(total_lie_acc, total_lies)
    print("  all cheating debates: %d accepted of %d, 95%% upper bound %.5f" % (total_lie_acc, total_lies, hi))
    print("\nruntime %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
