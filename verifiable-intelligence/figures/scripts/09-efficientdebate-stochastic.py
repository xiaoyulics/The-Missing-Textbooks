"""Chapter 9 (Efficient Debate): debate about a computation with stochastic human judgments.

A stochastic computation of length T is a family of probabilities pi_h, one for each history h of
earlier bits: bit tau is 1 with probability pi_h given the history h. Protocol 9.2 runs it step by
step: A announces the probability of the next bit, B passes or objects, an objection is settled by
the judge from m samples of human judgment, and a passed step draws the bit with a public coin.

Two experiments:
  (a) the judge's test at one objection: the probability that the judge sides with an announcement
      that is off by eta (an honest estimate at the edge of its accuracy) or by 3*eta (a lie that a
      well-informed challenger objects to), against the number m of human samples, with the exact
      binomial values and the Hoeffding bound 2 exp(-2 m eta^2) used in the proof;
  (b) a liar who tilts every announced probability upward by delta, against a lax challenger who
      passes every announcement within delta of the truth and against the challenger of the chapter's theorem on noisy judgments
      (tolerance 4 eta with estimates from m' samples, eta = 1/(150K)). The computation is a vote of
      101 human judgments, each "yes" with probability 0.46, followed by the deterministic step
      "output 1 if at least 51 said yes". Theory: the exact acceptance of the tilted computation and
      the Lipschitz bound acc(pi) + K delta, with K computed exactly for this computation.
Writes figures/data/09-efficientdebate-judge*.dat and 09-efficientdebate-tilt*.dat.
Run from the volume root: python3 figures/scripts/09-efficientdebate-stochastic.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402


# ------------------------------------------------------------------ (a) the judge's test
def judge_accept_exact(m, p, ann, eta):
    """P[ |Bin(m,p)/m - ann| <= 2 eta ]."""
    lo = math.ceil(m * (ann - 2 * eta) - 1e-9)
    hi = math.floor(m * (ann + 2 * eta) + 1e-9)
    lo, hi = max(lo, 0), min(hi, m)
    if hi < lo:
        return 0.0
    return float(stats.binom.cdf(hi, m, p) - (stats.binom.cdf(lo - 1, m, p) if lo > 0 else 0.0))


def judge_experiment():
    p, eta, trials = 0.7, 0.02, 4000
    gen = pk.rng(91)
    ms = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    rows = []
    for m in ms:
        bound = min(1.0, 2 * math.exp(-2 * m * eta ** 2))
        res = []
        for ann in (p + eta, p + 3 * eta):
            k = int(np.sum(np.abs(gen.binomial(m, p, size=trials) / m - ann) <= 2 * eta + 1e-12))
            lo, hi = pk.clopper_pearson(k, trials)
            res.append((k / trials, k / trials - lo, hi - k / trials, judge_accept_exact(m, p, ann, eta)))
        rows.append((m, res[0][0], res[0][1], res[0][2], res[0][3], max(0.0, 1 - bound),
                     res[1][0], res[1][1], res[1][2], res[1][3], bound))
    pk.write_dat("figures/data/09-efficientdebate-judge.dat",
                 ["m", "hon", "honlo", "honhi", "honexact", "honbound",
                  "lie", "lielo", "liehi", "lieexact", "liebound"], rows)
    m_thm = math.ceil(math.log(120) / (2 * eta ** 2))
    print("(a) judge's test, p = %.2f, eta = %.3f, %d trials per m" % (p, eta, trials))
    print("    m    honest(exact)  lie(exact)   bound 2exp(-2m eta^2)")
    for r in rows:
        print("%6d   %.4f (%.4f)  %.4f (%.4f)  %.4f" % (r[0], r[1], r[4], r[6], r[9], r[10]))
    print("    the theorem's m for this eta: %d" % m_thm)
    for r in rows:
        assert r[1] + r[3] >= r[5] - 1e-12 and r[6] - r[7] <= r[10] + 1e-12, "a measured rate contradicts the bound"


# ------------------------------------------------------------------ (b) the tilting liar
NV, P, THR = 101, 0.46, 51          # 101 votes, each yes w.p. 0.46; output 1 iff >= 51 yes
T = NV + 1                          # the vote steps and the deterministic output step


def p_maj(p):
    p = min(max(p, 0.0), 1.0)
    return float(stats.binom.sf(THR - 1, NV, p))


def acc_tilted(d):
    """Largest acceptance over computations within distance d of pi (attained by raising every vote
    probability by d and the output step's probability of a wrong 1 to d)."""
    return min(1.0, (1 - min(d, 1.0)) * p_maj(P + d) + min(d, 1.0))


def acc_lowered(d):
    return max(0.0, (1 - min(d, 1.0)) * p_maj(P - d))


def lipschitz_constant():
    a0 = p_maj(P)
    ds = np.concatenate([np.geomspace(1e-6, 1e-2, 400), np.linspace(1e-2, 1.0, 4000)])
    vals = [max(acc_tilted(d) - a0, a0 - acc_lowered(d)) / d for d in ds]
    i = int(np.argmax(vals))
    return float(vals[i]), float(ds[i])


def run_debates(delta, n, gen, lax, eta=None, m_deb=None, m_judge=None):
    """Protocol 9.2 with the tilting liar A' (announces min(1, pi_h + delta)) against the lax
    challenger (passes iff |a - pi_h| <= delta, so never objects) or against the honest challenger
    of the chapter's theorem. Returns the number of debates the judge accepts."""
    count = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)        # no objection yet
    verdict = np.zeros(n, dtype=bool)      # judge's verdict where an objection happened
    for tau in range(T):
        if tau < NV:
            pi = np.full(n, P)
        else:
            pi = (count >= THR).astype(float)
        ann = np.minimum(1.0, pi + delta)
        if not lax:
            qbar = gen.binomial(m_deb, pi) / m_deb
            obj = active & (np.abs(qbar - ann) > 4 * eta)
            if obj.any():
                pbar = gen.binomial(m_judge, pi[obj]) / m_judge
                verdict[obj] = np.abs(pbar - ann[obj]) <= 2 * eta
                active &= ~obj
        bit = gen.random(n) < ann
        if tau < NV:
            count += bit & active
        else:
            final = bit
    return int(np.sum((active & final) | (~active & verdict)))


def tilt_experiment():
    a0 = p_maj(P)
    K, dstar = lipschitz_constant()
    eta = 1 / (150 * K)
    m_deb = math.ceil(math.log(240 * T) / (2 * eta ** 2))
    m_judge = math.ceil(math.log(120) / (2 * eta ** 2))
    print("\n(b) vote of %d judgments with p = %.2f, output 1 iff >= %d: acc(pi) = %.4f" % (NV, P, THR, a0))
    print("    Lipschitz constant K = %.3f (attained near distance %.3f); eta = 1/(150K) = %.5f" % (K, dstar, eta))
    print("    challenger samples per step m' = %d, judge samples m = %d" % (m_deb, m_judge))
    print("    3 eta = %.5f, 5 eta = %.5f, acc + 5 K eta = %.4f" % (3 * eta, 5 * eta, a0 + 5 * K * eta))

    curve = [(d, acc_tilted(d), min(1.0, a0 + K * d)) for d in np.geomspace(1e-4, 0.5, 200)]
    pk.write_dat("figures/data/09-efficientdebate-tilt-theory.dat", ["delta", "exact", "bound"], curve)

    gen = pk.rng(92)
    deltas = [1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.15, 0.2, 0.3, 0.5]
    n = 4000
    rows = []
    print("    delta    lax challenger (exact)     theorem's challenger")
    for d in deltas:
        k1 = run_debates(d, n, gen, lax=True)
        k2 = run_debates(d, n, gen, lax=False, eta=eta, m_deb=m_deb, m_judge=m_judge)
        r1, (l1, h1) = k1 / n, pk.clopper_pearson(k1, n)
        r2, (l2, h2) = k2 / n, pk.clopper_pearson(k2, n)
        rows.append((d, r1, r1 - l1, h1 - r1, r2, r2 - l2, h2 - r2, acc_tilted(d), min(1.0, a0 + K * d)))
        print("    %.4f   %.4f [%.4f,%.4f] (%.4f)   %.4f [%.4f,%.4f]" % (d, r1, l1, h1, acc_tilted(d), r2, l2, h2))
        assert l1 <= min(1.0, a0 + K * d) + 1e-12, "lax rate above the Lipschitz bound"
        assert l2 <= 2 / 5, "theorem's challenger loses more often than the theorem allows"
    pk.write_dat("figures/data/09-efficientdebate-tilt.dat",
                 ["delta", "lax", "laxlo", "laxhi", "thm", "thmlo", "thmhi", "exact", "bound"], rows)
    return K, eta


def lipschitz_of_vote(m, p):
    """K for the vote of m judgments, each yes w.p. p, output 1 iff more than m/2 said yes."""
    thr = m // 2 + 1
    pm = lambda q: float(stats.binom.sf(thr - 1, m, min(max(q, 0.0), 1.0)))  # noqa: E731
    a0 = pm(p)
    best = 0.0
    for d in np.concatenate([np.geomspace(1e-6, 1e-2, 200), np.linspace(1e-2, 1.0, 4000)]):
        up = min(1.0, (1 - d) * pm(p + d) + d)
        down = max(0.0, (1 - d) * pm(p - d))
        best = max(best, (up - a0) / d, (a0 - down) / d)
    return best


def vote_constants():
    print("\n(c) Lipschitz constant of a majority vote of m judgments, each yes w.p. p")
    for p in (0.35, 0.46):
        ks = ["m=%d: K=%.2f" % (m, lipschitz_of_vote(m, p)) for m in (101, 1001, 10001)]
        print("    p = %.2f: %s; 1/(1/2-p) = %.2f" % (p, ", ".join(ks), 1 / (0.5 - p)))


def theorem_constants():
    """Constants quoted with the theorem on noisy judgments and in its exercises."""
    print("\n(d) constants of the theorem")
    for c in (150, 120):
        eta = 1 / c                                   # K = 1
        m = math.ceil(math.log(120) / (2 * eta ** 2))
        compl = 2 / 3 - 1 / c - 1 / 120 - 1 / 60      # completeness bound of the proof
        sound = 1 / 3 + 5 / c + 1 / 120 + 1 / 60      # soundness bound of the proof
        print("    eta = 1/(%dK): judge samples for K = 1: m = %d; completeness >= %.4f, soundness <= %.4f"
              % (c, m, compl, sound))
    t = next(t for t in range(1, 10_000) if math.exp(-t / 50) < 1e-6)
    print("    majority of t runs with gap 1/5: error exp(-t/50) < 1e-6 first at t = %d" % t)


def main():
    t0 = time.time()
    judge_experiment()
    tilt_experiment()
    vote_constants()
    theorem_constants()
    print("\nruntime %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
