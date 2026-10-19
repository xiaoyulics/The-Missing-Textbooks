"""01-certificates-dpll.py -- Chapter 1 (Certificates): certificates for SAT and for UNSAT.

The prover is a DPLL solver (unit propagation and backjumping, no clause learning). On an
unsatisfiable formula it extracts a tree-like resolution refutation from its search tree; on a
satisfiable one it returns an assignment. An independent checker verifies every certificate.

Formulas: the pigeonhole formulas PHP^{n+1}_n (unsatisfiable) and PHP^n_n (satisfiable), and
random 3-CNFs with N = n(n+1) variables and round(4.2 N) clauses, of which the satisfiable ones
are kept. Cheating provers: the best assignment for PHP^{n+1}_n (it falsifies one clause), random
near-matchings, a refutation of PHP^{n+1}_n offered for PHP^n_n, and valid refutations with one
corrupted resolution step. For n <= 3 an exhaustive search gives the fewest leaves of a tree-like
refutation and the value of the Prover-Delayer game.

Writes
  figures/data/01-certificates-dpll.dat  n, refutation sizes for PHP^{n+1}_n and the bounds
  figures/data/01-certificates-sat.dat   n, assignment sizes and costs for satisfiable formulas
and prints every number the chapter quotes. Run from the volume root (about 20 s, 0.4 GB):
  python3 figures/scripts/01-certificates-dpll.py
Exercise 1.10: python3 figures/scripts/01-certificates-dpll.py --branch index
  (branch on the free variable of smallest index; prints the sizes, writes no data).
"""
import os
import sys
import math
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

sys.setrecursionlimit(20000)
NMAX = 8                     # largest number of holes for PHP^{n+1}_n
SEED = 1
BRANCH = 'clause'            # branching rule; '--branch index' (Exercise) picks the smallest free variable


# ----------------------------------------------------------------------------- formulas
def php(m, n):
    """PHP^m_n: m pigeons, n holes; variable of (pigeon i, hole j) is i*n + j + 1."""
    v = lambda i, j: i * n + j + 1
    cls = [tuple(v(i, j) for j in range(n)) for i in range(m)]           # pigeon clauses
    for j in range(n):                                                    # hole clauses
        for a in range(m):
            for b in range(a + 1, m):
                cls.append((-v(a, j), -v(b, j)))
    return m * n, cls


def random_3cnf(N, M, gen):
    cls = []
    for _ in range(M):
        vs = gen.choice(N, size=3, replace=False) + 1
        signs = gen.integers(0, 2, size=3) * 2 - 1
        cls.append(tuple(int(s * v) for s, v in zip(signs, vs)))
    return N, cls


# ----------------------------------------------------------------------------- the prover
class DPLL:
    """DPLL with unit propagation and backjumping; records a tree-like resolution refutation.

    Every search node returns either ('SAT', None) or ('UNSAT', k), where step k of self.proof is
    a clause falsified by the current partial assignment, derived from the formula's clauses by
    a tree-like resolution derivation. A unit propagation of literal l from clause C is a
    decision node on var(l) whose branch "l false" is a leaf labeled C.
    """

    def __init__(self, nvars, clauses):
        self.N, self.cls = nvars, [tuple(c) for c in clauses]
        self.occ = {l: [] for v in range(1, nvars + 1) for l in (v, -v)}
        for ci, c in enumerate(self.cls):
            for l in c:
                self.occ[l].append(ci)
        self.val = [None] * (nvars + 1)
        self.ntrue = [0] * len(self.cls)
        self.nfalse = [0] * len(self.cls)
        self.nsat = 0
        self.proof = []            # steps: ('A', clause, ci) or ('R', clause, k_pos, k_neg, var)
        self.leaves = 0            # leaves of the search tree (conflicts and failed propagations)
        self.nodes = 0

    def assign(self, l):
        self.val[abs(l)] = l > 0
        for ci in self.occ[l]:
            if self.ntrue[ci] == 0:
                self.nsat += 1
            self.ntrue[ci] += 1
        for ci in self.occ[-l]:
            self.nfalse[ci] += 1

    def unassign(self, l):
        self.val[abs(l)] = None
        for ci in self.occ[l]:
            self.ntrue[ci] -= 1
            if self.ntrue[ci] == 0:
                self.nsat -= 1
        for ci in self.occ[-l]:
            self.nfalse[ci] -= 1

    def urgent(self, ci):          # falsified or unit
        return self.ntrue[ci] == 0 and self.nfalse[ci] >= len(self.cls[ci]) - 1

    def axiom(self, ci):
        self.proof.append(('A', self.cls[ci], ci))
        return len(self.proof) - 1

    def resolve(self, kp, kn, var):
        """Resolve step kp (contains var) with step kn (contains -var) on var."""
        cp, cn = self.proof[kp][1], self.proof[kn][1]
        res = tuple(sorted(set(l for l in cp if l != var) | set(l for l in cn if l != -var), key=abs))
        self.proof.append(('R', res, kp, kn, var))
        return len(self.proof) - 1

    def combine(self, lit, k_true, k_false_thunk):
        """Node on var(lit); k_true from the branch where lit is true. Returns a step id."""
        if -lit not in self.proof[k_true][1]:
            return k_true                      # the branch did not use the decision: backjump
        k_false = k_false_thunk()
        if k_false is None:
            return None
        if lit not in self.proof[k_false][1]:
            return k_false
        return self.resolve(k_false, k_true, lit) if lit > 0 else self.resolve(k_true, k_false, -lit)

    def node(self, cand):
        self.nodes += 1
        for ci in cand:                        # a falsified clause ends the branch
            if self.ntrue[ci] == 0 and self.nfalse[ci] == len(self.cls[ci]):
                self.leaves += 1
                return ('UNSAT', self.axiom(ci))
        for ci in cand:                        # unit propagation
            if self.ntrue[ci] == 0 and self.nfalse[ci] == len(self.cls[ci]) - 1:
                lit = next(l for l in self.cls[ci] if self.val[abs(l)] is None)
                self.leaves += 1               # the branch "lit false" fails at once on clause ci
                return self.branch(lit, cand, lambda ci=ci: self.axiom(ci))
        if self.nsat == len(self.cls):
            return ('SAT', None)
        if BRANCH == 'index':                  # decision: the free variable of smallest index, true first
            lit = next(v for v in range(1, self.N + 1) if self.val[v] is None)
        else:                                  # decision: first unsatisfied clause, first free literal
            for c in self.cls:
                if not any(self.val[abs(l)] == (l > 0) for l in c):
                    lit = next(l for l in c if self.val[abs(l)] is None)
                    break
        return self.branch(lit, cand, None)

    def child(self, lit, cand):
        self.assign(lit)
        new = [ci for ci in cand if self.urgent(ci)] + [ci for ci in self.occ[-lit] if self.urgent(ci)]
        out = self.node(new)
        if out[0] == 'SAT':
            return out
        self.unassign(lit)
        return out

    def branch(self, lit, cand, false_leaf):
        r = self.child(lit, cand)
        if r[0] == 'SAT':
            return r
        if false_leaf is not None:             # propagation: the false branch is a known leaf
            k = self.combine(lit, r[1], false_leaf)
            return ('UNSAT', k)
        state = {}

        def explore_false():
            rf = self.child(-lit, cand)
            state['r'] = rf
            return None if rf[0] == 'SAT' else rf[1]
        k = self.combine(lit, r[1], explore_false)
        if k is None:
            return state['r']
        return ('UNSAT', k)

    def solve(self):
        out = self.node(list(range(len(self.cls))))
        if out[0] == 'SAT':
            return 'SAT', [None] + [bool(self.val[v]) for v in range(1, self.N + 1)]
        return 'UNSAT', compact(self.proof, out[1])


def compact(proof, root):
    """Keep the steps reachable from the root, in post-order, renumbered (a tree-like refutation)."""
    out, index = [], {}
    stack = [(root, False)]
    while stack:
        k, done = stack.pop()
        st = proof[k]
        if st[0] == 'A' or done:
            if st[0] == 'A':
                out.append(('A', st[1]))
            else:
                out.append(('R', st[1], index[st[2]], index[st[3]], st[4]))
            index[k] = len(out) - 1
            continue
        stack.append((k, True))
        stack.append((st[3], False))
        stack.append((st[2], False))
    return out


# ----------------------------------------------------------------------------- the checker
def check_assignment(clauses, a):
    """Accept iff the assignment a (a[v] for v >= 1) satisfies every clause."""
    for c in clauses:
        if not any(a[abs(l)] == (l > 0) for l in c):
            return False
    return True


def check_refutation(clauses, steps, override=None):
    """Accept iff steps is a tree-like resolution refutation of the clauses.

    Each step is ('A', C) with C a clause of the formula, or ('R', C, i, j, x) with i, j earlier
    steps, x in clause i, -x in clause j, and C exactly the resolvent; each step is used at most
    once as a premise, and the last clause is empty. Since a premise is used once, it is
    released after use, so the checker's memory is proportional to the depth of the tree.
    override=(k, step) checks the list with step k replaced (used for corrupted certificates).
    """
    axioms = set(frozenset(c) for c in clauses)
    have = [None] * len(steps)
    for k, st in enumerate(steps):
        if override is not None and k == override[0]:
            st = override[1]
        if st[0] == 'A':
            if frozenset(st[1]) not in axioms:
                return False, f"step {k}: not a clause of the formula"
            have[k] = st[1]
            continue
        _, c, i, j, x = st
        if not (0 <= i < k and 0 <= j < k) or i == j or have[i] is None or have[j] is None:
            return False, f"step {k}: premises not earlier, or used twice"
        ci, cj = set(have[i]), set(have[j])
        have[i] = have[j] = None
        if x not in ci or -x not in cj:
            return False, f"step {k}: pivot {x} missing"
        if set(c) != (ci - {x}) | (cj - {-x}) or len(set(c)) != len(c):
            return False, f"step {k}: not the resolvent"
        have[k] = c
    if not steps or have[-1] is None or len(have[-1]) != 0:
        return False, "last clause is not empty"
    return True, "ok"


def leaves_of(steps):
    return sum(1 for st in steps if st[0] == 'A')


def exhaustive(m, n):
    """For a small PHP^m_n, by memoized search over partial assignments (a dict from variable to
    value): the fewest leaves of a decision tree, which equals the fewest leaves of a tree-like
    refutation (Lemma 1.10 and its converse), and the value of the Prover-Delayer game with optimal
    play on both sides. Feasible for n <= 3."""
    N, F = php(m, n)
    cls = [tuple((abs(l), l > 0) for l in c) for c in F]
    memo_leaves, memo_game = {}, {}

    def falsified(a):
        return any(all(v in a and a[v] != pos for v, pos in c) for c in cls)

    def key(a):
        return tuple(a.get(v, 2) for v in range(1, N + 1))

    def leaves(a):
        k = key(a)
        if k not in memo_leaves:
            if falsified(a):
                memo_leaves[k] = 1
            else:
                best = math.inf
                for v in range(1, N + 1):
                    if v in a:
                        continue
                    l0 = leaves({**a, v: False})
                    if l0 >= best:
                        continue
                    best = min(best, l0 + leaves({**a, v: True}))
                memo_leaves[k] = best
        return memo_leaves[k]

    def game(a):
        k = key(a)
        if k not in memo_game:
            if falsified(a):
                memo_game[k] = 0
            else:
                vals = []
                for v in range(1, N + 1):
                    if v not in a:
                        g0, g1 = game({**a, v: False}), game({**a, v: True})
                        vals.append(max(g0, g1, 1 + min(g0, g1)))   # Delayer answers or defers
                memo_game[k] = min(vals)                             # Prover names the variable
        return memo_game[k]
    return leaves({}), game({})


def pigeon_by_pigeon(m, n):
    """Leaves of the refutation that places pigeons one at a time: L(m,0)=1,
    L(m,n) = n (m - 1 + L(m-1, n-1)) + 1."""
    return 1 if n == 0 else n * (m - 1 + pigeon_by_pigeon(m - 1, n - 1)) + 1


# ----------------------------------------------------------------------------- experiments
def timed(f, *args, reps=1):
    best = math.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f(*args)
        best = min(best, time.perf_counter() - t0)
    return out, best


def main():
    t_start = time.perf_counter()
    gen = pk.rng(SEED)
    rnd = random.Random(SEED)
    rows, srows = [], []
    print("== PHP^{n+1}_n: tree-like refutations found by DPLL and checked ==")
    print(" n   N  clauses  search-leaves  ref-leaves  ref-clauses  2^n  (n/2+1)^(n/2)  pigeon-by-pigeon  t_solve  t_check")
    for n in range(1, NMAX + 1):
        N, F = php(n + 1, n)
        ts = math.inf
        for _ in range(3):                         # best of three runs, each with a fresh solver
            solver = DPLL(N, F)
            t0 = time.perf_counter()
            status, steps = solver.solve()
            ts = min(ts, time.perf_counter() - t0)
            search_leaves = solver.leaves
            del solver                             # free the search record; keep the refutation
        assert status == 'UNSAT'
        (ok, why), tc = timed(check_refutation, F, steps, reps=3)
        assert ok, why
        L, S = leaves_of(steps), len(steps)
        lb_leaves, dr = 2 ** n, (n / 2 + 1) ** (n / 2)   # Dantchev-Riis: n even
        pbp = pigeon_by_pigeon(n + 1, n)
        assert L >= lb_leaves and S >= 2 * lb_leaves - 1 and S >= dr      # the theorems
        print(f"{n:2d} {N:3d} {len(F):7d} {search_leaves:13d} {L:11d} {S:12d} {lb_leaves:4d} "
              f"{dr:13.1f} {pbp:17d} {ts:8.3f} {tc:8.3f}")
        rows.append((n, N, len(F), search_leaves, L, S, lb_leaves, 2 * lb_leaves - 1,
                     dr if n % 2 == 0 else float('nan'),
                     pbp, 2 * pbp - 1, ts, tc))

        # cheating provers for the false claim "PHP^{n+1}_n is satisfiable": all rejected
        best = [None] + [False] * N
        for i in range(n):
            best[i * n + i + 1] = True            # pigeons 0..n-1 in holes 0..n-1, pigeon n nowhere
        assert not check_assignment(F, best)
        rej = 0
        for _ in range(1000):
            a = [None] + [False] * N
            perm = list(range(n))
            rnd.shuffle(perm)
            for i in range(n):
                a[i * n + perm[i] + 1] = True
            if rnd.random() < 0.5:                 # the last pigeon shares a hole, or has none
                a[n * n + rnd.randrange(n) + 1] = True
            rej += not check_assignment(F, a)
        # cheating provers for the false claim "PHP^n_n is unsatisfiable": all rejected
        _, Fsat = php(n, n)
        fake_ok, fake_why = check_refutation(Fsat, steps)   # a refutation of the other formula
        rejcor, ncor = 0, 200 if n <= 6 else 20
        idx = [k for k, st in enumerate(steps) if st[0] == 'R' and st[1]]
        for _ in range(ncor):                       # one corrupted resolution step
            k = rnd.choice(idx)
            st = list(steps[k])
            c = list(st[1])
            c.pop(rnd.randrange(len(c)))
            st[1] = tuple(c)
            rejcor += not check_refutation(F, steps, override=(k, tuple(st)))[0]
        print(f"     cheaters rejected: best assignment yes; {rej}/1000 near-matchings; "
              f"PHP^{n+1}_{n} refutation offered for PHP^{n}_{n}: {'accepted' if fake_ok else 'rejected'} ({fake_why}); "
              f"{rejcor}/{ncor} corrupted steps")

    print("== exhaustive search for small n: fewest leaves of a tree-like refutation, game value ==")
    for n in (1, 2, 3):
        lv, gv = exhaustive(n + 1, n)
        print(f" n={n}: fewest leaves {lv} (one pigeon at a time: {pigeon_by_pigeon(n + 1, n)}); "
              f"Prover-Delayer game value {gv} (Theorem: at least {n})")
    print("== satisfiable formulas: assignments found by DPLL and checked ==")
    print(" n   N  PHP^n_n: bits t_solve t_check | random 3-CNF (4.2N clauses): kept/tried "
          "median search-leaves median t_solve median t_check")
    for n in range(1, NMAX + 1):
        N1, F1 = php(n, n)
        (st1, a1), ts1 = timed(lambda: DPLL(N1, F1).solve(), reps=3)
        assert st1 == 'SAT'
        ok1, tc1 = timed(check_assignment, F1, a1, reps=3)
        assert ok1
        N = n * (n + 1)
        Mc = round(4.2 * N)
        kept, tried, lv, tss, tcs = 0, 0, [], [], []
        while N >= 3 and kept < 9 and tried < 200:
            tried += 1
            _, F = random_3cnf(N, Mc, gen)
            s = DPLL(N, F)
            t0 = time.perf_counter()
            st, a = s.solve()
            ts = time.perf_counter() - t0
            if st != 'SAT':
                continue
            okc, tc = timed(check_assignment, F, a, reps=3)
            assert okc
            kept += 1
            lv.append(s.leaves)
            tss.append(ts)
            tcs.append(tc)
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float('nan')
        print(f"{n:2d} {N:3d} {N1:13d} {ts1:7.4f} {tc1:7.5f} | {kept}/{tried} {med(lv):10} "
              f"{med(tss):10.4f} {med(tcs):10.5f}")
        srows.append((n, N1, N, Mc, kept, tried, med(lv), ts1, tc1, med(tss), med(tcs)))

    pk.write_dat("figures/data/01-certificates-dpll.dat",
                 ["n", "N", "M", "search", "leaves", "size", "lbleaves", "lbsize", "dr",
                  "pbpleaves", "pbpsize", "tsolve", "tcheck"], rows)
    pk.write_dat("figures/data/01-certificates-sat.dat",
                 ["n", "Nphp", "Nrand", "Mrand", "kept", "tried", "searchrand", "tsolvephp",
                  "tcheckphp", "tsolverand", "tcheckrand"], srows)
    print(f"total time {time.perf_counter() - t_start:.1f} s")


def explore_branching():
    """Exercise: the same solver with another branching rule, on PHP^{n+1}_n only (no data files)."""
    print(f"branching rule: {BRANCH}")
    print(" n  search-leaves  refutation-leaves  2^n")
    for n in range(1, NMAX):
        N, F = php(n + 1, n)
        solver = DPLL(N, F)
        status, steps = solver.solve()
        ok, why = check_refutation(F, steps)
        assert status == 'UNSAT' and ok, why
        print(f"{n:2d} {solver.leaves:14d} {leaves_of(steps):18d} {2 ** n:4d}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == '--branch':
        BRANCH = sys.argv[2]
        explore_branching()
    else:
        main()
