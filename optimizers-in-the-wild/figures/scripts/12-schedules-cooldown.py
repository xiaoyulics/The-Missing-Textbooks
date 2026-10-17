#!/usr/bin/env python3
"""Figure 12.3 (Optimizers in the Wild, Chapter 12): how long should the cooldown be?

River valley of Figure 12.1 (40 hills with curvatures in [0.2, 2], total hill noise variance 10;
a 200-direction quadratic river with curvatures 0.02 j^-2 and initial losses proportional to j^-q),
for a river exponent alpha = 1/2 (q = 2) and alpha = 1 (q = 3). WSD with peak 0.04, total length
T in {1000, 4000, 16000} and a linear cooldown over the last fraction c of the steps.

Curves: the exact expected final loss (recursion of the river-valley theorem) on a grid of c.
Points: mean final loss of 40 SGD runs (Gaussian noise) on a coarser grid of c.
The minimizer of the continuum approximation (river exact, hills by the cooldown law
(eta s_i/4) phi(eta h_i K), phi(k) = sqrt(pi)/2 erf(sqrt(k))/sqrt(k)) is written to the -opt file.

Output: figures/data/12-schedules-cooldown.dat, 12-schedules-cooldown-sgd.dat, 12-schedules-cooldown-opt.dat
Runtime: about 30 seconds (numpy and scipy.special, seed 31).
"""
import os
import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PEAK = 0.04
TS = [1000, 4000, 16000]
QS = {"a05": 2.0, "a1": 3.0}


def instance(q, n_river=200, h_river=0.02, river_loss=0.5, n_hill=40, h_lo=0.2, h_hi=2.0, noise=10.0):
    j = np.arange(1, n_river + 1)
    h_r = h_river * j ** -2.0
    c_r = j ** -q
    c_r = river_loss * c_r / c_r.sum()
    h_h = np.geomspace(h_lo, h_hi, n_hill)
    h = np.concatenate([h_r, h_h])
    y0sq = np.concatenate([2 * c_r / h_r, np.zeros(n_hill)])
    s2 = np.concatenate([np.zeros(n_river), np.full(n_hill, noise / n_hill)])
    return h, s2, y0sq, n_river


def wsd_batch(T, cs):
    """rows: schedules eta_1..eta_T for each cooldown fraction c (linear cooldown to zero)."""
    E = np.full((len(cs), T), PEAK)
    for m, c in enumerate(cs):
        K = max(1, int(round(c * T)))
        E[m, T - K:] = PEAK * (1 - np.arange(1, K + 1) / K)
    return E


def exact_final(E, h, s2, y0sq):
    v = np.broadcast_to(y0sq, (E.shape[0], len(h))).astype(float).copy()
    for t in range(E.shape[1]):
        e = E[:, t][:, None]
        v = (1 - e * h) ** 2 * v + e ** 2 * s2
    return 0.5 * (v * h).sum(axis=1)


def sgd_final(E, h, s2, y0sq, runs, rng):
    """SGD runs; the river directions carry no noise, so noise is drawn for the noisy directions only."""
    M, T = E.shape
    y = np.broadcast_to(np.sqrt(y0sq), (M, runs, len(h))).copy()
    noisy = s2 > 0
    sd, hn, hq = np.sqrt(s2[noisy]), h[noisy], h[~noisy]
    yn, yq = y[:, :, noisy].copy(), y[:, :, ~noisy].copy()
    for t in range(T):
        e = E[:, t][:, None, None]
        yn -= e * (hn * yn + rng.standard_normal(yn.shape) * sd)
        yq -= e * (hq * yq)
    return (0.5 * (hn * yn ** 2).sum(axis=2) + 0.5 * (hq * yq ** 2).sum(axis=2)).mean(axis=1)


def phi(k):
    k = np.maximum(k, 1e-300)
    return np.sqrt(np.pi) / 2 * erf(np.sqrt(k)) / np.sqrt(k)


def continuum_opt(T, h, s2, y0sq, nr):
    hR, cR = h[:nr], 0.5 * h[:nr] * y0sq[:nr]
    hH, sH = h[nr:], s2[nr:]
    K = np.arange(1, T + 1)
    river = (cR[None, :] * np.exp(-2 * hR[None, :] * PEAK * (T - K[:, None] / 2))).sum(axis=1)
    hill = (PEAK * sH[None, :] / 4 * phi(PEAK * hH[None, :] * K[:, None])).sum(axis=1)
    return K[np.argmin(river + hill)] / T


def main():
    rng = np.random.default_rng(31)
    cs_exact = np.geomspace(0.01, 1.0, 61)
    cs_sgd = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    exact_cols, sgd_cols, opt_rows = {}, {}, []
    for key, q in QS.items():
        h, s2, y0sq, nr = instance(q)
        for T in TS:
            exact_cols["%sT%d" % (key, T)] = exact_final(wsd_batch(T, cs_exact), h, s2, y0sq)
            sgd_cols["%sT%d" % (key, T)] = sgd_final(wsd_batch(T, cs_sgd), h, s2, y0sq, runs=40, rng=rng)
            ex = exact_cols["%sT%d" % (key, T)]
            c_heur = continuum_opt(T, h, s2, y0sq, nr)
            c_ex = cs_exact[np.argmin(ex)]
            Lh = exact_final(wsd_batch(T, [c_heur]), h, s2, y0sq)[0]
            opt_rows.append((key, T, c_heur, Lh, c_ex, ex.min()))
            print("%s T=%5d  continuum c*=%.3f (exact loss there %.4f)  grid argmin of exact: c=%.3f (%.4f)" % (
                key, T, c_heur, Lh, c_ex, ex.min()))
    keys = list(exact_cols.keys())
    with open(os.path.join(DATA, "12-schedules-cooldown.dat"), "w") as f:
        f.write("c " + " ".join(keys) + "\n")
        for i, c in enumerate(cs_exact):
            f.write("%.5g " % c + " ".join("%.6g" % exact_cols[k][i] for k in keys) + "\n")
    with open(os.path.join(DATA, "12-schedules-cooldown-sgd.dat"), "w") as f:
        f.write("c " + " ".join(keys) + "\n")
        for i, c in enumerate(cs_sgd):
            f.write("%.5g " % c + " ".join("%.6g" % sgd_cols[k][i] for k in keys) + "\n")
    with open(os.path.join(DATA, "12-schedules-cooldown-opt.dat"), "w") as f:
        f.write("T cstara05 lossa05 cstara1 lossa1\n")
        for T in TS:
            r05 = [r for r in opt_rows if r[0] == "a05" and r[1] == T][0]
            r1 = [r for r in opt_rows if r[0] == "a1" and r[1] == T][0]
            f.write("%d %.4f %.6g %.4f %.6g\n" % (T, r05[2], r05[3], r1[2], r1[3]))


if __name__ == "__main__":
    main()
