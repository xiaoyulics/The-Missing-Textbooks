"""Chapter 12 (Verifying Learning): query delegation with hidden checks, run against honest and
cheating provers.

The class H is the set of histogram classifiers on [0,1) with d equal bins (a rule decides by the
bin of x; VC dimension d, |H| = 2^d, ERM = majority vote per bin, ties to label 0). Inputs are
uniform on [0,1). The labels are a deterministic function f: in bin j, f(x) = 1 iff the fractional
part of K*d*x is below q_j = 1/2 + s_j*gamma, a fine pattern with exactly a q_j fraction of ones in
the bin, so that L_D(h) is known exactly and L_D(H) = 1/2 - gamma.

Protocol (Protocol 12.1): the verifier draws q labeled points (checks) and m - q unlabeled points,
all uniform; shuffles them with a uniform permutation; the prover returns labels; the verifier
rejects if a check disagrees, and otherwise outputs ERM on the returned labels. With eps = delta =
0.1: q = ceil(4 ln(2/delta)/eps) = 120 and m = ceil(ln(2|H|/(delta/2))/(2(eps/4)^2)) = 8497, the
finite-class uniform-convergence bound of Shalev-Shwartz and Ben-David (Corollary 4.6).

Provers:
  honest     labels every point by f.
  random     mislabels a uniformly random set of b points.
  targeted   mislabels b points toward the worse label, cheapest bins first, so as to make ERM err
             on as many bins as the budget allows.
  harm-opt   mislabels the fewest points that push the ERM's excess loss above eps (6 bins here).
  aware      knows that the checks come from [0, 0.3) only (a violated hypothesis) and lies on
             [0.3, 1).
Outputs: figures/data/12-pacverify-delegation-{rates,theory,checks,checktheory}.dat.
Seeds: 12 (class), 1201 (rates), 1202 (checks), 1203 (honest, aware). Runs in about a minute.
With an argument, e.g. `python3 figures/scripts/12-pacverify-delegation.py 1000` (Exercise 12.11),
the script uses that many points instead of m and prints its results without writing any data.
"""
import math
import os
import sys
import time

import numpy as np
from scipy.special import gammaln

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protokit as pk  # noqa: E402

EPS, DELTA = 0.1, 0.1
D_BINS, GAMMA, K = 10, 0.1, 1000
Q = math.ceil(4 * math.log(2 / DELTA) / EPS)
M_UC = math.ceil(math.log(2 * 2 ** D_BINS / (DELTA / 2)) / (2 * (EPS / 4) ** 2))
M = max(Q, M_UC)
WRITE = len(sys.argv) < 2                           # an argument replaces m and suppresses the data files
if not WRITE:
    M = int(sys.argv[1])
_g = pk.rng(12)
SIGN = np.where(_g.random(D_BINS) < 0.5, -1, 1)
QJ = 0.5 + GAMMA * SIGN
BAYES = (QJ > 0.5).astype(np.int8)
L_H = float(np.mean(np.minimum(QJ, 1 - QJ)))
BIN_HARM = 2 * GAMMA / D_BINS                       # excess loss of erring on one bin
J_STAR = math.floor(EPS / BIN_HARM + 1e-9) + 1      # fewest wrong bins with excess > EPS


def label(x):
    """f(x) and the bin of x."""
    j = np.minimum((x * D_BINS).astype(np.int64), D_BINS - 1)
    return (np.mod(x * D_BINS * K, 1.0) < QJ[j]).astype(np.int8), j


def wrong_bins(y, j):
    """number of bins on which ERM (majority, ties to 0) disagrees with the Bayes rule"""
    n1 = np.bincount(j, weights=y, minlength=D_BINS)
    n = np.bincount(j, minlength=D_BINS)
    out = (n1 > n - n1).astype(np.int8)
    return int(np.sum(out != BAYES))


def flip_costs(y, j):
    """fewest flips toward the worse label that make ERM err on each bin"""
    n1 = np.bincount(j, weights=y, minlength=D_BINS).astype(np.int64)
    n0 = np.bincount(j, minlength=D_BINS) - n1
    bad1 = np.maximum(0, (n1 - n0 + 1) // 2)        # Bayes label 1: need n1 - r <= n0 + r
    bad0 = np.maximum(0, (n0 - n1) // 2 + 1)        # Bayes label 0: need n1 + r > n0 - r
    return np.where(BAYES == 1, bad1, bad0)


def targeted(y, j, budget):
    """positions of `budget` labels flipped toward the worse label, cheapest bins first"""
    cost = flip_costs(y, j)
    good = np.flatnonzero(y == BAYES[j])            # points whose label agrees with the Bayes rule
    by_bin = [good[j[good] == b] for b in range(D_BINS)]
    chosen, left = [], budget
    for b in np.argsort(cost, kind="stable"):
        take = min(left, cost[b] if cost[b] <= left else left, len(by_bin[b]))
        chosen.append(by_bin[b][:take])
        by_bin[b] = by_bin[b][take:]
        left -= take
        if left == 0:
            break
    if left > 0:                                    # leftover budget: more flips toward bad labels
        rest = np.concatenate(by_bin)
        chosen.append(rest[:left])
    return np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)


def eta_hat(y, j):
    """sup over H of |L_S(h) - L_D(h)| for the clean sample (exact for histogram classifiers)"""
    n1 = np.bincount(j, weights=y, minlength=D_BINS)
    n0 = np.bincount(j, minlength=D_BINS) - n1
    dev1 = n0 / M - (1 - QJ) / D_BINS               # rule says 1 on the bin: errors are 0-labels
    dev0 = n1 / M - QJ / D_BINS
    return max(np.sum(np.maximum(dev0, dev1)), -np.sum(np.minimum(dev0, dev1)))


def one_run(gen, liar, b, check_lo=0.0, check_hi=1.0):
    """one run of Protocol 12.1; returns (accepted, excess loss of ERM on the returned labels,
    wrong labels, eta_hat of the clean sample)"""
    x = gen.random(M)
    x[:Q] = check_lo + (check_hi - check_lo) * x[:Q]  # the verifier's labeled points (checks)
    y_check = label(x[:Q])[0]
    pi = gen.permutation(M)                         # position p shows x[pi[p]]
    z = x[pi]
    y_true, zj = label(z)                           # the prover's view: z only
    y_ret = y_true.copy()
    if liar == "random" and b > 0:
        y_ret[gen.choice(M, size=b, replace=False)] ^= 1
    elif liar == "targeted" and b > 0:
        y_ret[targeted(y_true, zj, b)] ^= 1
    elif liar == "aware":                           # lie only where no check can be
        safe = z >= check_hi
        cost = flip_costs(y_true[safe], zj[safe])
        present = np.bincount(zj[safe], minlength=D_BINS) > 0
        budget = int(cost[present].sum())           # reverse ERM on every bin a check cannot reach
        y_ret[np.flatnonzero(safe)[targeted(y_true[safe], zj[safe], budget)]] ^= 1
    checks = np.flatnonzero(pi < Q)
    accepted = bool(np.all(y_ret[checks] == y_check[pi[checks]]))
    harm = wrong_bins(y_ret, zj) * BIN_HARM
    return accepted, harm, int(np.sum(y_ret != y_true)), eta_hat(y_true, zj)


def hyper(m, b, q):
    """P[a uniform q-subset of [m] misses a fixed b-subset] = C(m-b,q)/C(m,q)"""
    b = np.asarray(b, dtype=float)
    out = np.exp(gammaln(m - b + 1) - gammaln(m - b - q + 1) - gammaln(m + 1) + gammaln(m - q + 1))
    return np.where(m - b >= q, out, 0.0)


def main():
    t0 = time.time()
    print(f"eps = delta = {EPS}; d = {D_BINS}; gamma = {GAMMA}; L_D(H) = {L_H:.3f}")
    print(f"q = ceil(4 ln(2/delta)/eps) = {Q}  (4 ln(2/delta)/eps = {4*math.log(2/DELTA)/EPS:.2f};"
          f" with log base 2 it would be {math.ceil(4*math.log2(2/DELTA)/EPS)})")
    src = "finite-class bound on m_UC(eps/4, delta/2)" if WRITE else f"from the argument; the bound is {M_UC}"
    print(f"m = {M} points ({src}); one bin of error costs "
          f"{BIN_HARM:.2f}; harm > eps needs {J_STAR} wrong bins")
    if not WRITE:
        print("no data files are written")

    # ---------------- acceptance and harm against the fraction of wrong labels -----------------
    gen = pk.rng(1201)
    taus = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
    n_runs = 6000
    rows = []
    for tau in taus:
        b = round(tau * M)
        res = {}
        for liar in ("random", "targeted"):
            acc, harms = 0, []
            for _ in range(n_runs):
                ok, harm, nb, eta = one_run(gen, liar, b)
                assert nb == b
                # the robustness lemma: harm <= 2 tau + 2 eta_hat, always
                assert harm <= 2 * b / M + 2 * eta + 1e-12
                acc += ok
                harms.append(harm)
            lo, hi = pk.clopper_pearson(acc, n_runs)
            h = np.array(harms)
            res[liar] = (acc / n_runs, lo, hi, h.mean(), np.percentile(h, 2.5), np.percentile(h, 97.5))
        exact = float(hyper(M, b, Q))
        r, t_ = res["random"], res["targeted"]
        rows.append([b / M, b, exact, (1 - b / M) ** Q,
                     r[0], r[2] - r[0], r[0] - r[1], t_[0], t_[2] - t_[0], t_[0] - t_[1],
                     r[3], t_[3], t_[5] - t_[3], t_[3] - t_[4], 2 * b / M + EPS / 2])
        print(f"tau = {b/M:.4f} (b = {b:4d}): exact {exact:.5f}, bound {(1-b/M)**Q:.5f}; random "
              f"{r[0]:.5f} [{r[1]:.5f}, {r[2]:.5f}]; targeted {t_[0]:.5f} [{t_[1]:.5f}, {t_[2]:.5f}];"
              f" harm random {r[3]:.4f}, targeted {t_[3]:.4f} (2.5-97.5%: {t_[4]:.3f}-{t_[5]:.3f})")
    if WRITE:
        pk.write_dat("figures/data/12-pacverify-delegation-rates.dat",
                     ["tau", "b", "exact", "bound", "rnd", "rndep", "rndem", "tgt", "tgtep", "tgtem",
                      "harmr", "harmt", "harmtep", "harmtem", "harmbound"], rows)
    tt = np.linspace(0, 0.065, 66)
    if WRITE:
        pk.write_dat("figures/data/12-pacverify-delegation-theory.dat",
                     ["tau", "exact", "bound", "harmbound", "harmline"],
                     [[t, float(hyper(M, round(t * M), Q)), (1 - t) ** Q, 2 * t + EPS / 2, 2 * t]
                      for t in tt])

    # ---------------- the harm-optimal liar against the number of checks ----------------------
    gen = pk.rng(1202)
    qs = np.arange(0, 141, 10)
    n_c = 40000
    succ = np.zeros(len(qs))
    tstars, etas = np.zeros(n_c, dtype=np.int64), np.zeros(n_c)
    for r in range(n_c):
        x = gen.random(M)
        pi = gen.permutation(M)
        z = x[pi]
        y_true, zj = label(z)
        cost = flip_costs(y_true, zj)
        tstar = int(np.sort(cost)[:J_STAR].sum())
        flips = targeted(y_true, zj, tstar)
        y_ret = y_true.copy()
        y_ret[flips] ^= 1
        assert wrong_bins(y_ret, zj) >= J_STAR      # the lie does harm > eps if accepted
        first = pi[flips].min() if len(flips) else M  # checks of a q-run sit where pi < q
        succ += first >= qs
        tstars[r], etas[r] = tstar, eta_hat(y_true, zj)
    rows = []
    for i, qq in enumerate(qs):
        k = int(succ[i])
        lo, hi = pk.clopper_pearson(k, n_c)
        exact = float(np.mean(hyper(M, tstars, qq)))
        bound = min(1.0, (1 - EPS / 4) ** qq + DELTA / 2)
        rows.append([qq, k / n_c, hi - k / n_c, k / n_c - lo, exact, bound])
        print(f"q = {qq:3d}: harm-optimal liar succeeds {k/n_c:.5f} [{lo:.5f}, {hi:.5f}], "
              f"exact {exact:.5f}, bound (1-eps/4)^q + delta/2 = {bound:.4f}")
    if WRITE:
        pk.write_dat("figures/data/12-pacverify-delegation-checks.dat",
                     ["q", "succ", "ep", "em", "exact", "bound"], rows)
    qq = np.arange(0, 141, 2)
    if WRITE:
        pk.write_dat("figures/data/12-pacverify-delegation-checktheory.dat",
                     ["q", "exact", "bound", "checkterm"],
                     [[v, float(np.mean(hyper(M, tstars, v))), min(1.0, (1 - EPS / 4) ** v + DELTA / 2),
                       (1 - EPS / 4) ** v] for v in qq])
    print(f"harm-optimal lie: t*/m mean {tstars.mean()/M:.4f}, range [{tstars.min()/M:.4f}, "
          f"{tstars.max()/M:.4f}]; eps/4 = {EPS/4}")
    print(f"clean samples: eta_hat mean {etas.mean():.4f}, max {etas.max():.4f}; "
          f"fraction with eta_hat > eps/4: {np.mean(etas > EPS/4):.5f} (theory allows {DELTA/2})")

    # ---------------- honest prover; a prover that can tell the checks apart -------------------
    gen = pk.rng(1203)
    n_h = 4000
    acc, harms = 0, []
    for _ in range(n_h):
        ok, harm, nb, eta = one_run(gen, "honest", 0)
        acc += ok
        harms.append(harm)
    print(f"honest prover: accepted {acc}/{n_h}; excess of output: mean {np.mean(harms):.4f}, "
          f"max {np.max(harms):.4f} (theorem: <= eps/2 = {EPS/2} w.p. >= 1 - delta/2)")
    acc, harms, wrong = 0, [], []
    for _ in range(n_h):
        ok, harm, nb, eta = one_run(gen, "aware", 0, 0.0, 0.3)
        acc += ok
        harms.append(harm)
        wrong.append(nb / M)
    print(f"check-aware liar (checks drawn from [0, 0.3) only): accepted {acc}/{n_h}; "
          f"excess of output: mean {np.mean(harms):.4f}, min {np.min(harms):.4f}; "
          f"wrong labels: mean fraction {np.mean(wrong):.4f}")
    for eta in (0.01, 0.033):
        print(f"honest prover with check labels wrong w.p. {eta}: accepted w.p. (1-{eta})^q = "
              f"{(1 - eta) ** Q:.4f}")
    print(f"time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
