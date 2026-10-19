#!/usr/bin/env python3
"""Chapter 11 (prover-verifier games): the binary erasure channel game of Anil, Zhang, Wu and Grosse (2021).

An instance is a uniform bit x, and the claim is "x = 1".  The prover always argues that x = 1 and is paid
-log2 of the probability that the verifier accepts; the verifier is paid for predicting x (log loss).
Messages are tokens: token 0 can be sent only on x = 0, token 1 only on x = 1, and the other tokens
("erasures") on both.  A sound and complete protocol exists: send token 1 on x = 1, accept only token 1.

Part 1 (Table 11.2 of the chapter).  The simplified game (tokens 0, 1, 2; base verifiers "reject all",
"accept all" and "accept token 1"; the four deterministic provers).  We enumerate, on a grid of verifier mixtures with
denominator 120, the equilibria of three orders of play: verifier first (the prover then answers each
instance optimally), simultaneous (Nash), prover first.  For one representative of each class we run the
protocol: the equilibrium prover on true instances and the best cheating prover on false instances.

Part 2 (Figure 11.1(a,b)).  Sixteen tokens (0, 1 and fourteen erasures), label smoothing LAM in the
verifier's loss.  The verifier plays its exact best response after every prover step (the limit of
two-timescale training); the prover takes gradient steps on softmax logits, either holding the verifier
fixed (simultaneous play) or differentiating through the verifier's response (prover first).  At
checkpoints we run the protocol: the trained prover on true instances (completeness) and the best
cheating prover on false instances (soundness error), against the exact acceptance probabilities and the
bound 1 - 2^(-loss/D(x)) of the chapter's proposition on the leader's loss.  We also record over many
seeds how often each dynamic reaches its limit, and repeat the simultaneous run with a verifier that
itself takes gradient steps.

Part 3 (Figure 11.1(c)).  Stress test: freeze the verifier trained by simultaneous play and one trained
collaboratively (the prover paid for the verifier's correctness), and optimize a cheating prover on false
instances against each by gradient descent; its acceptance rate against the frozen verifier's exact
soundness error.

Part 4 (Exercise "Giving up rare instances").  A game in which a valid verifier of completeness 1 exists but
the verifier-leading equilibrium under the expected losses is not valid, whichever way the prover's ties are
broken; we minimize the leader's loss numerically and compare with the exact minimizer.

Writes figures/data/11-pvg-train.dat (exact curves), figures/data/11-pvg-train-meas.dat (measured rates
with Clopper-Pearson 95% intervals), figures/data/11-pvg-stress.dat and figures/data/11-pvg-stress-meas.dat.
Run from the volume root:  python3 figures/scripts/11-pvg-bec.py   (about twenty seconds on a laptop)
"""
import os
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

NTOK = 16                      # tokens 0..15
ERASE = list(range(2, NTOK))   # tokens either instance may send
M_FALSE = [0] + ERASE          # messages available on x = 0
M_TRUE = [1] + ERASE           # messages available on x = 1
LAM = 0.05                     # label smoothing in the verifier's loss
ETA = 1.0                      # prover step size (softmax logits)
T_TRAIN = 10_000               # prover steps
CHECKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000]
RUNS = 4000                    # protocol runs per checkpoint and per prover
N_SEEDS = 100                  # seeds for the convergence statistics
T_MAX = 50_000                 # longest run in the convergence statistics
STRESS_STEPS = 1000
STRESS_CHECKS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
SEED = 11                      # seed of the training runs shown in the figure
SEED_RUNS = 1111               # seed of the protocol runs
OUT_TRAIN = "figures/data/11-pvg-train.dat"
OUT_TRAIN_MEAS = "figures/data/11-pvg-train-meas.dat"
OUT_STRESS = "figures/data/11-pvg-stress.dat"
OUT_STRESS_MEAS = "figures/data/11-pvg-stress-meas.dat"

C_LIM = (1 + LAM) / (1 + 2 * LAM)   # the proof token's acceptance in the limit of simultaneous play
S_LIM = LAM / (1 + 2 * LAM)         # every other token's acceptance in that limit


# ----------------------------------------------------------------------------------------------- part 1
def part1():
    """Equilibria of the simplified game by enumeration (exact rational arithmetic on a grid)."""
    msgs = {0: [0, 2], 1: [1, 2]}                   # M(x)
    provers = [(a, b) for a in msgs[0] for b in msgs[1]]   # (message on x=0, message on x=1)
    base = {"rej": lambda z: 0, "acc": lambda z: 1, "v1": lambda z: 1 if z == 1 else 0}
    den = 120
    grid = [(Fraction(i, den), Fraction(j, den), Fraction(den - i - j, den))
            for i in range(den + 1) for j in range(den + 1 - i)]   # weights on rej, acc, v1
    INF = float("inf")

    def V(w, z):                                     # acceptance probability of message z
        return w[1] * base["acc"](z) + w[2] * base["v1"](z)

    def lg(v):                                       # -log2 v, with -log2 0 = infinity
        return INF if v == 0 else -np.log2(float(v))

    def loss_v(w, pr):                               # verifier: log loss for predicting x
        return 0.5 * lg(1 - V(w, pr[0])) + 0.5 * lg(V(w, pr[1]))

    def loss_p(w, pr):                               # prover: -log2 of acceptance, both instances
        return 0.5 * lg(V(w, pr[0])) + 0.5 * lg(V(w, pr[1]))

    def a(w, x):                                     # best acceptance probability on instance x
        return max(V(w, z) for z in msgs[x])

    def lam(w):                                      # verifier's loss against instance-wise best responses
        return 0.5 * lg(1 - a(w, 0)) + 0.5 * lg(a(w, 1))

    def br_inst(w):                                  # instance-wise best responses
        return [pr for pr in provers if V(w, pr[0]) == a(w, 0) and V(w, pr[1]) == a(w, 1)]

    eqs = {}
    # verifier first, instance revealed after the verifier commits
    lams = [lam(w) for w in grid]
    m = min(lams)
    eqs["verifier first"] = [(pr, w) for w, l in zip(grid, lams) if l == m for pr in br_inst(w)]
    # simultaneous (Nash): each strategy a best response to the other, expected losses
    nash = []
    vbest = {pr: min(loss_v(w, pr) for w in grid) for pr in provers}
    for w in grid:
        pl = [loss_p(w, pr) for pr in provers]
        for pr, l in zip(provers, pl):
            if l == min(pl) and loss_v(w, pr) == vbest[pr]:
                nash.append((pr, w))
    eqs["simultaneous"] = nash
    # prover first: the verifier best-responds to the committed prover
    lead = {}
    for pr in provers:
        brs = [w for w in grid if loss_v(w, pr) == vbest[pr]]
        lp = [loss_p(w, pr) for w in brs]
        lead[pr] = (min(lp), max(lp), brs)
    best = min(v[1] for v in lead.values())          # with ties broken against the prover
    eqs["prover first"] = [(pr, w) for pr, (lo, hi, brs) in lead.items() if hi == best for w in brs]
    print("Part 1: equilibria of the simplified game (%d verifier strategies: mixtures with denominator %d)"
          % (len(grid), den))
    for pr, (lo, hi, brs) in lead.items():
        print("  prover %s: verifier's best responses %d grid points; prover's loss %.3f..%.3f bits"
              % (pr, len(brs), lo, hi))
    rows = []
    gen = pk.rng(SEED_RUNS)
    for name, lst in eqs.items():
        classes = {}
        for pr, w in lst:
            comp = V(w, pr[1])                       # acceptance of the equilibrium prover on x = 1
            snd = a(w, 0)                            # best cheater on x = 0
            classes.setdefault((pr[1], comp, snd), []).append((pr, w))
        for (z1, comp, snd), mem in sorted(classes.items()):
            ws = sorted({w for _, w in mem})
            pr0 = sorted({pr[0] for pr, _ in mem})
            wdesc = ("(%s,%s,%s)" % ws[0] if len(ws) == 1 else
                     "%d mixtures, w(acc) in {%s}" % (len(ws), ",".join(sorted({str(w[1]) for w in ws}))))
            k1 = int((gen.random(RUNS) < float(comp)).sum())
            k0 = int((gen.random(RUNS) < float(snd)).sum())
            lo1, hi1 = pk.clopper_pearson(k1, RUNS)
            lo0, hi0 = pk.clopper_pearson(k0, RUNS)
            print("  %-15s prover sends %s on x=1 (on x=0: %s); verifier %s" % (name, z1, pr0, wdesc))
            print("      completeness %s, measured %.4f [%.4f, %.4f]; soundness error %s, measured %.4f [%.4f, %.4f]"
                  % (comp, k1 / RUNS, lo1, hi1, snd, k0 / RUNS, lo0, hi0))
            rows.append((name, z1, comp, snd))
    print("  flood the zone: prover's loss %.3f bits (uninformative) against infinity (informative)" % best)
    return rows


# ----------------------------------------------------------------------------------------------- part 2
MASK = np.full((2, NTOK), -np.inf)
MASK[0, M_FALSE] = 0.0
MASK[1, M_TRUE] = 0.0


def probs(theta):
    z = theta + MASK
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def best_response(p):
    """The verifier's best response under label smoothing: q_i = (rho_i + LAM)/(1 + 2 LAM), where rho_i is
    the probability that x = 1 given token i (x uniform)."""
    rho = p[1] / (p[0] + p[1])
    return (rho + LAM) / (1 + 2 * LAM), rho


def prover_grad(p, q, total, collab=False):
    """Gradient of the prover's loss (bits, x uniform) with respect to p[x, i].  With total=True it also
    differentiates through the verifier's best response (prover first)."""
    g = np.zeros_like(p)
    if collab:                                  # paid for the verifier's correctness
        g[0] = 0.5 * -np.log2(1 - q)
        g[1] = 0.5 * -np.log2(q)
        return g
    g[0] = g[1] = 0.5 * -np.log2(q)
    if total:
        rho = p[1] / (p[0] + p[1])
        # d q_i / d p[1,i] and d q_i / d p[0,i] for erasure tokens; tokens 0 and 1 have constant q
        er = np.array(ERASE)
        g[1, er] += 0.5 * (-(1 - rho[er]) / (rho[er] + LAM)) / np.log(2)
        g[0, er] += 0.5 * (rho[er] / (rho[er] + LAM)) / np.log(2)
    return g


def logit_step(theta, p, g, eta):
    return theta - eta * p * (g - (p * g).sum(axis=1, keepdims=True))


def leader_loss(q):
    """The verifier's loss (bits) against instance-wise best responses, without smoothing."""
    a1 = q[M_TRUE].max()
    a0 = q[M_FALSE].max()
    return 0.5 * -np.log2(a1) + 0.5 * -np.log2(1 - a0)


def train(seed, total, steps=T_TRAIN, record=None, collab=False):
    """Prover gradient steps against the verifier's best response; returns final (p, q) and a trace."""
    theta = pk.rng(seed).normal(size=(2, NTOK))
    trace = {}
    for t in range(steps + 1):
        p = probs(theta)
        q, _ = best_response(p)
        if record is not None and t in record:
            trace[t] = (p.copy(), q.copy())
        theta = logit_step(theta, p, prover_grad(p, q, total, collab), ETA)
    return p, q, trace


def train_gd_verifier(seed, steps=T_TRAIN, inner=5, eta_v=10.0):
    """Simultaneous play with a verifier that takes `inner` gradient steps per prover step."""
    gen = pk.rng(seed)
    theta = gen.normal(size=(2, NTOK))
    phi = np.zeros(NTOK)
    for _ in range(steps + 1):
        p = probs(theta)
        mass = 0.5 * (p[0] + p[1])
        rho = p[1] / (p[0] + p[1])
        for _ in range(inner):
            q = 1 / (1 + np.exp(-phi))
            # gradient of mass_i * [-(rho+LAM) ln q - (1-rho+LAM) ln(1-q)] with respect to the logit
            phi = phi - eta_v * mass * ((1 + 2 * LAM) * q - (rho + LAM))
        q = 1 / (1 + np.exp(-phi))
        theta = logit_step(theta, p, prover_grad(p, q, False), ETA)
    p = probs(theta)
    return p, 1 / (1 + np.exp(-phi))


def stats(p, q):
    comp = float((p[1] * q).sum())              # trained prover on x = 1
    snd = float(q[M_FALSE].max())               # best cheating prover on x = 0
    bound = 1 - 2 ** (-leader_loss(q) / 0.5)    # the leader's-loss bound with D(0) = 1/2
    return comp, snd, bound


def measure(gen, p_row, q, n):
    """Run the protocol n times: the prover samples a token from p_row, the verifier accepts it with
    probability q[token].  Returns (rate, plus, minus) for pgfplots error bars."""
    toks = gen.choice(NTOK, size=n, p=p_row)
    k = int((gen.random(n) < q[toks]).sum())
    lo, hi = pk.clopper_pearson(k, n)
    return k / n, hi - k / n, k / n - lo


def part2():
    dense = sorted({int(round(v)) for v in np.logspace(0, 4, 160)} | set(CHECKS))
    rec = set(dense) | set(CHECKS)
    out = {}
    for name, total in (("sim", False), ("fol", True)):
        p, q, trace = train(SEED, total, record=rec)
        out[name] = trace
        c, s, b = stats(p, q)
        print("Part 2: %s seed %d after %d steps: completeness %.4f, soundness error %.4f, bound %.4f, "
              "p(proof token | x=1) %.5f, q(proof token) %.4f"
              % ("simultaneous" if not total else "prover first", SEED, T_TRAIN, c, s, b, p[1, 1], q[1]))
    print("  limits of simultaneous play: %.6f and %.6f; prover first: 1/2" % (C_LIM, S_LIM))
    rows = []
    for t in dense:
        r = [t]
        for name in ("sim", "fol"):
            p, q = out[name][t]
            r += list(stats(p, q))
        rows.append(r)
    pk.write_dat(OUT_TRAIN, ["t", "simC", "simS", "simB", "folC", "folS", "folB"], rows)
    gen = pk.rng(SEED_RUNS + 1)
    mrows = []
    worst = -1.0
    for t in CHECKS:
        r = [t]
        for name in ("sim", "fol"):
            p, q = out[name][t]
            cheat = np.zeros(NTOK)
            cheat[M_FALSE[int(np.argmax(q[M_FALSE]))]] = 1.0      # the best cheating prover
            mc = measure(gen, p[1], q, RUNS)
            ms = measure(gen, cheat, q, RUNS)
            c, s, b = stats(p, q)
            worst = max(worst, ms[0] - ms[2] - b)                  # lower end of interval above bound?
            for m, exact in ((mc, c), (ms, s)):
                COVER[0] += 1
                COVER[1] += int(m[0] - m[2] - 1e-12 <= exact <= m[0] + m[1] + 1e-12)
            r += [mc[0], mc[1], mc[2], ms[0], ms[1], ms[2]]
        mrows.append(r)
    pk.write_dat(OUT_TRAIN_MEAS, ["t", "simC", "simCp", "simCm", "simS", "simSp", "simSm",
                                  "folC", "folCp", "folCm", "folS", "folSp", "folSm"], mrows)
    print("  measured rates written; largest (lower interval end - bound) = %.4f (must be <= 0)" % worst)
    for t in (100, 200, 500, 1000):
        p, q = out["sim"][t]
        c, s, b = stats(p, q)
        print("  simultaneous t=%5d: completeness %.4f, soundness error %.4f, bound %.4f" % (t, c, s, b))
    for t in (1, 100, 1000, 10_000):
        p, q = out["fol"][t]
        c, s, b = stats(p, q)
        print("  prover first t=%5d: completeness %.4f, soundness error %.4f, bound %.4f, a=%.5f b=%.5f"
              % (t, c, s, b, p[1, 1], p[0, 0]))
    # convergence statistics over seeds.  Simultaneous play: the first step (checked every 10) at which
    # completeness and soundness error are both within 0.01 of their limits.  Prover first: the first step
    # at which completeness is within 0.01 of 1/2, and the soundness error after T_TRAIN steps.
    hit = {"sim": [], "fol": []}
    fol_s = []
    for sd in range(N_SEEDS):
        theta = pk.rng(1000 + sd).normal(size=(2, NTOK))
        first = None
        for t in range(T_MAX + 1):
            p = probs(theta)
            q, _ = best_response(p)
            if t % 10 == 0:
                c, s, _ = stats(p, q)
                if abs(c - C_LIM) < 0.01 and abs(s - S_LIM) < 0.01:
                    first = t
                    break
            theta = logit_step(theta, p, prover_grad(p, q, False), ETA)
        hit["sim"].append(first)
        theta = pk.rng(1000 + sd).normal(size=(2, NTOK))
        first = None
        for t in range(T_TRAIN + 1):
            p = probs(theta)
            q, _ = best_response(p)
            if first is None and t % 10 == 0 and abs(stats(p, q)[0] - 0.5) < 0.01:
                first = t
            theta = logit_step(theta, p, prover_grad(p, q, True), ETA)
        hit["fol"].append(first)
        fol_s.append(stats(p, q)[1])
    hs = [h for h in hit["sim"] if h is not None]
    print("  simultaneous: %d of %d seeds within 0.01 of (%.4f, %.4f) by step %d, %d by step %d; median "
          "first step %d, latest %d" % (sum(h <= T_TRAIN for h in hs), N_SEEDS, C_LIM, S_LIM, T_TRAIN,
                                        len(hs), T_MAX, int(np.median(hs)), max(hs)))
    hf = [h for h in hit["fol"] if h is not None]
    print("  prover first: completeness within 0.01 of 1/2 in %d of %d seeds by step %d (median first step "
          "%d, latest %d); soundness error at step %d between %.4f and %.4f"
          % (len(hf), N_SEEDS, T_TRAIN, int(np.median(hf)), max(hf), T_TRAIN, min(fol_s), max(fol_s)))
    p, q = train_gd_verifier(SEED)
    c, s, b = stats(p, q)
    print("  simultaneous with a gradient-trained verifier (5 steps per prover step), seed %d: "
          "completeness %.4f, soundness error %.4f" % (SEED, c, s))
    return out


# ----------------------------------------------------------------------------------------------- part 3
def part3(out):
    p_sim, q_sim = out["sim"][T_TRAIN]
    p_col, q_col, _ = train(SEED, False, collab=True)
    for name, p, q in (("PVG", p_sim, q_sim), ("collaborative", p_col, q_col)):
        print("Part 3: %s verifier: trained prover accepted %.4f on x=1 and %.4f on x=0; soundness error %.4f "
              "(token %d)" % (name, (p[1] * q).sum(), (p[0] * q).sum(), q[M_FALSE].max(),
                              M_FALSE[int(np.argmax(q[M_FALSE]))]))
    print("  collaborative prover's tokens on x=1 with probability > 0.01:",
          {i: round(float(p_col[1, i]), 3) for i in range(NTOK) if p_col[1, i] > 0.01})
    col_s = []
    for sd in range(N_SEEDS):
        p, q, _ = train(1000 + sd, False, steps=2000, collab=True)
        col_s.append(float(q[M_FALSE].max()))
    col_s = np.array(col_s)
    print("  collaborative training, %d seeds, %d steps: soundness error >= 0.9 in %d seeds, <= 0.1 in %d; "
          "range %.4f..%.4f" % (N_SEEDS, 2000, int((col_s >= 0.9).sum()), int((col_s <= 0.1).sum()),
                                 col_s.min(), col_s.max()))
    gen = pk.rng(SEED_RUNS + 2)
    curves = {}
    meas = {}
    for name, q in (("pvg", q_sim), ("col", q_col)):
        z = np.zeros(len(M_FALSE))                      # cheater's logits over M(0), uniform start
        qa = q[M_FALSE]
        curve = []
        mm = []
        for t in range(STRESS_STEPS + 1):
            pz = np.exp(z - z.max())
            pz /= pz.sum()
            acc = float((pz * qa).sum())
            curve.append((t, acc))
            if t in STRESS_CHECKS:
                row = np.zeros(NTOK)
                row[M_FALSE] = pz
                m = measure(gen, row, q, RUNS)
                COVER[0] += 1
                COVER[1] += int(m[0] - m[2] - 1e-12 <= acc <= m[0] + m[1] + 1e-12)
                mm.append(m)
            g = -np.log2(qa)
            z = z - ETA * pz * (g - (pz * g).sum())
        curves[name] = curve
        meas[name] = mm
        print("  cheater vs %s verifier: acceptance %.4f at step 0, %.4f at step %d; bound %.4f"
              % (name, curve[0][1], curve[-1][1], STRESS_STEPS, qa.max()))
    dense = sorted({0} | {int(round(v)) for v in np.logspace(0, 3, 90)} | set(STRESS_CHECKS))
    pk.write_dat(OUT_STRESS, ["t", "t1", "pvg", "pvgbound", "col", "colbound"],
                 [(t, t + 1, curves["pvg"][t][1], q_sim[M_FALSE].max(), curves["col"][t][1],
                   q_col[M_FALSE].max()) for t in dense])
    rows = []
    for i, t in enumerate(STRESS_CHECKS):
        a, b = meas["pvg"][i], meas["col"][i]
        rows.append((t, t + 1, a[0], a[1], a[2], b[0], b[1], b[2]))
    pk.write_dat(OUT_STRESS_MEAS, ["t", "t1", "pvg", "pvgp", "pvgm", "col", "colp", "colm"], rows)
    over = max(max(m[0] - m[2] for m in meas["pvg"]) - q_sim[M_FALSE].max(),
               max(m[0] - m[2] for m in meas["col"]) - q_col[M_FALSE].max())
    print("  largest (lower interval end - bound) in the stress test = %.4f (must be <= 0)" % over)


# ----------------------------------------------------------------------------------------------- part 4
def part4():
    """Exercise 'Giving up rare instances': instances T, F (true, false; probability 0.49 each) and t, f
    (true, false; 0.01 each); the prover sends mT on T, a or b on F, mt on t, mf on f.  Base verifiers:
    reject all, accept all, u1 = {mT,mt,a}, u2 = {mT,mt,b}, w = {mT,mf}.  The even mixture of u1 and u2 is
    valid with completeness 1; the expected-loss leader is not.  Lambda is convex here (one message on each
    true instance), so the minimum found by Nelder-Mead from several starts is the global one; the exact
    minimizer puts 2/51 on u1 and on u2 and 47/51 on w."""
    from scipy.optimize import linprog, minimize
    prob = {"T": 0.49, "F": 0.49, "t": 0.01, "f": 0.01}
    true = {"T": 1, "F": 0, "t": 1, "f": 0}
    names = ["reject", "accept", "u1={mT,mt,a}", "u2={mT,mt,b}", "w={mT,mf}"]

    def accept(wt):                      # largest acceptance probability on each instance
        acc, u1, u2, w = wt[1], wt[2], wt[3], wt[4]
        return {"T": acc + u1 + u2 + w, "F": acc + max(u1, u2), "t": acc + u1 + u2, "f": acc + w}

    def lam(wt):                         # leader's loss (bits) against instance-wise best responses
        a = accept(wt)
        return sum(prob[x] * -np.log2(max(a[x] if true[x] else 1 - a[x], 1e-300)) for x in prob)

    def kappa(wt):                       # worst-case loss of the proposition on worst-case losses
        a = accept(wt)
        return 1 - min(a["T"], a["t"]) + max(a["F"], a["f"])

    gen = pk.rng(SEED_RUNS + 3)
    best = (np.inf, None)
    for _ in range(40):
        res = minimize(lambda u: lam(np.exp(u - u.max()) / np.exp(u - u.max()).sum()), gen.normal(size=5) * 2,
                       method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-12, "fatol": 1e-14})
        if res.fun < best[0]:
            wt = np.exp(res.x - res.x.max())
            best = (res.fun, wt / wt.sum())
    wt = best[1]
    a = accept(wt)
    exact = np.array([0, 0, 2 / 51, 2 / 51, 47 / 51])
    print("Part 4: expected-loss leader in the rare-instance game: weights %s (exact %s); accepts T %.4f, F %.4f, "
          "t %.4f, f %.4f; Lambda %.5f (exact %.5f); kappa %.4f (valid iff < 1)"
          % (np.round(wt, 4).tolist(), np.round(exact, 4).tolist(), a["T"], a["F"], a["t"], a["f"], best[0],
             lam(exact), kappa(wt)))
    # least kappa: a linear program in (weights, c, s): minimize 1 - c + s, c <= true acceptances,
    # s >= acceptance of every message on a false instance
    rows = [[0, -1, -1, -1, -1, 1, 0], [0, -1, -1, -1, 0, 1, 0],          # c <= a(T), c <= a(t)
            [0, 1, 1, 0, 0, 0, -1], [0, 1, 0, 1, 0, 0, -1], [0, 1, 0, 0, 1, 0, -1]]   # s >= a, b, mf
    lp = linprog([0, 0, 0, 0, 0, -1, 1], A_ub=rows, b_ub=[0] * 5, A_eq=[[1, 1, 1, 1, 1, 0, 0]], b_eq=[1],
                 bounds=[(0, 1)] * 7)
    print("  least kappa %.4f at weights %s" % (1 + lp.fun, np.round(lp.x[:5], 4).tolist()))
    base = np.eye(5)
    print("  kappa of the base verifiers (%s): %s" % (", ".join(names), [round(float(kappa(base[i])), 3)
                                                                       for i in range(5)]))
    # ties broken in the verifier's favor: u1 alone rejects mf, so every prover strategy has infinite
    # expected loss; a prover that favors the verifier sends b on F, and the verifier's loss is 0
    u1 = base[2]
    fav = prob["F"] * -np.log2(1 - min(u1[1] + u1[2], u1[1] + u1[3])) + prob["T"] * -np.log2(accept(u1)["T"]) \
        + prob["t"] * -np.log2(accept(u1)["t"]) + prob["f"] * -np.log2(1 - accept(u1)["f"])
    print("  ties in the verifier's favor: u1 alone has leader's loss %.4f (the least possible) and kappa %.4f"
          % (fav + 0.0, kappa(u1)))


COVER = [0, 0]   # measured intervals in the figure, and how many contain the exact value


def main():
    t0 = time.time()
    print("settings: %d tokens, smoothing %.2f, %d prover steps, %d protocol runs per point, training seed %d, "
          "protocol seeds %d/%d/%d, %d seeds for statistics, %d cheater steps"
          % (NTOK, LAM, T_TRAIN, RUNS, SEED, SEED_RUNS, SEED_RUNS + 1, SEED_RUNS + 2, N_SEEDS, STRESS_STEPS))
    part1()
    out = part2()
    part3(out)
    print("Figure: %d of %d measured intervals contain the exact value" % (COVER[1], COVER[0]))
    part4()
    print("total time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
