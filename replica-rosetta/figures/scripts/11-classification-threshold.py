"""Chapter 11 (classification): the separability threshold of logistic data, alpha_MLE(r) = 1/h_MLE(r).

Model (book units): x_i ~ N(0, I_d), P(y_i = 1 | x_i) = f(<x_i, beta*>) with the logistic link f and
|beta*| = r; n = alpha d. Theorem 11.1: the data are strictly separable (no logistic MLE) with
probability -> 1 if alpha < alpha_MLE(r) and not separable (the MLE exists) if alpha > alpha_MLE(r), with
    h_MLE(r) = min_t E (Z - t V)_+^2,   V = Y U,  P(Y = 1 | U) = f(r U),  U, Z iid N(0, 1).

Computes
  (1) the curve alpha_MLE(r) for the logistic link on 0 <= r <= 6 (and a table, with the probit link);
  (2) for each data set of n = 200 samples with D = n features (teacher in coordinate 1), the critical
      dimension d_c = min{d : the data restricted to the first d features are strictly separable},
      found by bisection with one linear program per step (separability is monotone in d);
      alpha_c = n / d_c is the data set's critical load. For r in {0, 0.5, 1, 2, 3, 4, 5}, 40 data sets
      each: the median of alpha_c with a distribution-free 95% interval (order statistics).
  (3) the same at r = 0 against Cover's exact finite-n law P(d_c <= d) = 2^{1-n} sum_{k<d} C(n-1, k).
Writes figures/data/11-classification-threshold-curve.dat and ...-lp.dat. Runtime about a minute.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # shared machine: one BLAS thread per script
os.environ.setdefault("OMP_NUM_THREADS", "1")
import math
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import optimize, special, stats

import rosettakit as rk

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
SQ2PI = math.sqrt(2 * math.pi)


def v_nodes(r, link="logit", L=12.0, n=400):
    """Nodes and weights for E g(V), V = Y U with density (f(r v) + 1 - f(-r v)) phi(v); Gauss-Legendre
    on [-L, 0] and [0, L] (split at 0, where a discontinuous link would jump)."""
    x, w = np.polynomial.legendre.leggauss(n)
    vs, ws = [], []
    for lo, hi in ((-L, 0.0), (0.0, L)):
        v = 0.5 * (hi - lo) * x + 0.5 * (hi + lo)
        f = special.expit if link == "logit" else special.ndtr
        dens = f(r * v) + 1.0 - f(-r * v)
        vs.append(v)
        ws.append(0.5 * (hi - lo) * w * dens * np.exp(-0.5 * v * v) / SQ2PI)
    return np.concatenate(vs), np.concatenate(ws)


def gneg(u):
    """E (Z - u)_+^2 = (1 + u^2) Phi(-u) - u phi(u) for Z ~ N(0, 1)."""
    return (1 + u * u) * special.ndtr(-u) - u * np.exp(-0.5 * u * u) / SQ2PI


def h_mle(r, link="logit"):
    """(h_MLE(r), minimizing t): min_t E (Z - t V)_+^2 = min_t E gneg(t V)."""
    V, W = v_nodes(r, link)
    obj = lambda t: float(np.sum(W * gneg(t * V)))
    res = optimize.minimize_scalar(obj, bounds=(-2.0, 10.0 + r), method="bounded", options={"xatol": 1e-12})
    return res.fun, res.x


def separable(Z):
    """Strict separability of the rows of Z through the origin: is there b with Z b >= 1?"""
    n, d = Z.shape
    res = optimize.linprog(np.zeros(d), A_ub=-Z, b_ub=-np.ones(n), bounds=[(None, None)] * d,
                           method="highs")
    return res.status == 0


def critical_dimension(Z, guess):
    """d_c = min{d : rows of Z[:, :d] strictly separable}, by bisection started around guess."""
    n, D = Z.shape
    half = int(2 * math.sqrt(n)) + 2
    lo, hi = max(guess - half, 0), min(guess + half, D)
    while lo > 0 and separable(Z[:, :lo]):          # make sure: not separable at lo (lo = 0 is trivial)
        lo = max(lo - half, 0)
    while not separable(Z[:, :hi]):                 # make sure: separable at hi
        lo, hi = hi, min(hi + half, D)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if separable(Z[:, :mid]):
            hi = mid
        else:
            lo = mid
    return hi


def sample_Z(n, D, r, gen):
    """Labeled rows z_i = y_i x_i, teacher beta* = r e_1, logistic link."""
    X = gen.standard_normal((n, D))
    y = np.where(gen.random(n) < special.expit(r * X[:, 0]), 1.0, -1.0)
    return y[:, None] * X


def median_ci(x, level=0.95):
    """Sample median with the distribution-free order-statistic interval."""
    x = np.sort(np.asarray(x, float))
    N = len(x)
    j = int(stats.binom.ppf((1 - level) / 2, N, 0.5))      # P(Bin < j) <= 2.5%
    k = N - j
    return float(np.median(x)), float(x[max(j - 1, 0)]), float(x[min(k, N - 1)])


def main():
    t0 = time.time()
    print("== alpha_MLE(r) = 1/h_MLE(r) ==")
    print("%8s %10s %11s %9s %10s %11s %12s" % ("r", "h_logit", "a_logit", "t*", "h_probit", "a_probit",
                                               "P(Z>t*V)"))
    for r in [0.0, 0.5, 1.0, 2.0, math.sqrt(5.0), 3.0, 5.0, 10.0, 20.0]:
        h, t = h_mle(r)
        hp, _ = h_mle(r, "probit")
        V, W = v_nodes(r)
        active = float(np.sum(W * special.ndtr(-t * V)))      # isostaticity: equals h at the minimizer
        print("%8.4f %10.6f %11.6f %9.5f %10.6f %11.6f %12.8f" % (r, h, 1 / h, t, hp, 1 / hp, active))
    rows = []
    for r in np.linspace(0.0, 6.0, 61):
        h, t = h_mle(r)
        rows.append((r, 1.0 / h, h, t))
    rk.write_dat(os.path.join(DATA, "11-classification-threshold-curve.dat"), ["r", "alpha", "h", "t"], rows)

    n, N = 200, 40
    gen = rk.rng(11001)
    print("\n== critical load alpha_c = n/d_c by linear programming, n = %d, %d data sets per r ==" % (n, N))
    rows = []
    for r in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
        h, _ = h_mle(r)
        guess = int(round(n * h))
        ac = []
        for _ in range(N):
            Z = sample_Z(n, n, r, gen)
            ac.append(n / critical_dimension(Z, guess))
        med, lo, hi = median_ci(ac)
        m, half = rk.mean_ci(ac)
        rows.append((r, med, med - lo, hi - med, m, half, 1.0 / h))
        print("r=%.1f: median alpha_c = %.3f [%.3f, %.3f], mean %.3f +- %.3f; prediction %.4f  (%.0f s)" % (
            r, med, lo, hi, m, half, 1.0 / h, time.time() - t0))
    rk.write_dat(os.path.join(DATA, "11-classification-threshold-lp.dat"),
                 ["r", "med", "minus", "plus", "mean", "half", "pred"], rows)
    # Cover's law at r = 0: the median of d_c is n/2 for every n (P(d_c <= n/2) = 1/2 exactly)
    print("Cover at n = %d: P(d_c <= n/2) = %.4f; P(d_c <= 0.45 n) = %.4f; P(d_c <= 0.55 n) = %.4f" % (
        n, rk.cover_probability(n, n // 2), rk.cover_probability(n, int(0.45 * n)),
        rk.cover_probability(n, int(0.55 * n))))
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
