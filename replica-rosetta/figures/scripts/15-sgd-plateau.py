"""Chapter 15 (Online SGD): the plateau of the soft committee machine K = M = 2 under online SGD.

Online SGD for the soft committee machine with g(z) = erf(z/sqrt 2), orthonormal teacher
(T = I), Gaussian inputs, learning rate eta = 1, random start w_k = g_k/2 (g_k ~ N(0, I_d), so
q0 = 1/4), one fresh sample per step. The order parameters (Q, R) are simulated exactly in law
through their Markov chain (Lemma "exact reduction" of the chapter): given the past, the fields
(lambda, nu) are N(0, C(Q,R)) and |x|^2 = |nu|^2 + |zeta|^2 + chi^2_{d-4}; the script
15-sgd-rate.py checks the chain against direct simulation in R^d.

Runs: d = 125, 250, 500, 1000, 100 runs each (all dimensions advance in one vectorized loop),
up to t = 180. Records the error eps_g on a grid of spacing 0.4 and the escape time (first grid
time with eps_g < eps*/2, eps* = 0.020657 the plateau of the attracting symmetric fixed point).
Writes figures/data/15-sgd-plateau-curves.dat (learning curves, d = 125 and 1000) and
15-sgd-plateau-escape.dat (escape times against d). Runtime: about 30 seconds.
"""
import math
import os
import sys
import time

import numpy as np
from scipy.special import erf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SQ2 = math.sqrt(2.0)
CGP = math.sqrt(2.0 / math.pi)
EPS_PLATEAU = 0.020657     # attracting symmetric fixed point at eta = 1 (15-sgd-ode.py)
LAMBDA = 0.059140          # its escape rate (15-sgd-ode.py)


def init_order_params(d, q0, S, gen):
    """Exact law of (Q, R) for w_k = sqrt(q0) g_k, g_k ~ N(0, I_d), teacher w*_n = sqrt(d) e_n
    (Bartlett decomposition for the Gram matrix of the components outside the teacher span)."""
    gg = gen.standard_normal((S, 2, 2))
    c1 = np.sqrt(gen.chisquare(d - 2, S))
    c2 = np.sqrt(gen.chisquare(d - 3, S))
    n21 = gen.standard_normal(S)
    G11 = c1 ** 2 + gg[:, 0, 0] ** 2 + gg[:, 0, 1] ** 2
    G22 = n21 ** 2 + c2 ** 2 + gg[:, 1, 0] ** 2 + gg[:, 1, 1] ** 2
    G12 = c1 * n21 + gg[:, 0, 0] * gg[:, 1, 0] + gg[:, 0, 1] * gg[:, 1, 1]
    r = math.sqrt(q0) * gg / math.sqrt(d)
    return np.array([q0 * G11 / d, q0 * G12 / d, q0 * G22 / d, r[:, 0, 0], r[:, 0, 1], r[:, 1, 0], r[:, 1, 1]])


def eg_of(X):
    """Generalization error of states X (7, n): 0.5 sum_ab s_a s_b (2/pi) arcsin(...)."""
    q11, q12, q22, r11, r12, r21, r22 = X
    a = lambda c, x, y: np.arcsin(c / np.sqrt((1 + x) * (1 + y)))
    tot = (a(q11, q11, q11) + a(q22, q22, q22) + 2 * a(q12, q11, q22)
           - 2 * (a(r11, q11, 1.0) + a(r12, q11, 1.0) + a(r21, q22, 1.0) + a(r22, q22, 1.0)) + 2 * math.asin(0.5))
    return tot / math.pi


def simulate(dvals, S, eta, q0, tmax, rec_dt, gen, record, block=1000):
    """Advance S runs for each d in dvals in one loop; call record(g, k, X_g) at t = k rec_dt."""
    G = len(dvals)
    X = np.concatenate([init_order_params(d, q0, S, gen) for d in dvals], axis=1)
    for g in range(G):
        record(g, 0, X[:, g * S:(g + 1) * S])
    dv = np.repeat(np.array(dvals, float), S)
    rec_step = [int(round(rec_dt * d)) for d in dvals]
    last = [int(round(tmax * d)) for d in dvals]
    active = list(range(G))                       # groups still running, in order
    j = 0
    while active:
        sl = np.concatenate([np.arange(g * S, (g + 1) * S) for g in active])
        Xa = X[:, sl]
        dva = dv[sl]
        n = sl.size
        nb = min(block, max(last[g] for g in active) - j)
        Z = gen.standard_normal((nb, 4, n))
        CH = gen.chisquare(dva - 4, (nb, n))
        GNU = erf(Z[:, 0, :] / SQ2) + erf(Z[:, 1, :] / SQ2)
        BX = (eta * eta / dva) * (np.einsum('jkn,jkn->jn', Z, Z) + CH) / dva
        A = eta / dva
        starts = {g: i * S for i, g in enumerate(active)}
        q11, q12, q22, r11, r12, r21, r22 = [x.copy() for x in Xa]
        for i in range(nb):
            nu1, nu2, z1, z2 = Z[i]
            l11 = np.sqrt(np.abs(q11 - r11 * r11 - r12 * r12))
            l21 = (q12 - r11 * r21 - r12 * r22) / l11
            l22 = np.sqrt(np.abs(q22 - r21 * r21 - r22 * r22 - l21 * l21))
            lam1 = r11 * nu1 + r12 * nu2 + l11 * z1
            lam2 = r21 * nu1 + r22 * nu2 + l21 * z1 + l22 * z2
            err = GNU[i] - erf(lam1 / SQ2) - erf(lam2 / SQ2)
            D1 = (CGP * err) * np.exp(-0.5 * lam1 * lam1)
            D2 = (CGP * err) * np.exp(-0.5 * lam2 * lam2)
            aD1, aD2, bx = A * D1, A * D2, BX[i]
            q11 = q11 + (2 * aD1 * lam1 + bx * D1 * D1)
            q22 = q22 + (2 * aD2 * lam2 + bx * D2 * D2)
            q12 = q12 + (aD1 * lam2 + aD2 * lam1 + bx * D1 * D2)
            r11 = r11 + aD1 * nu1
            r12 = r12 + aD1 * nu2
            r21 = r21 + aD2 * nu1
            r22 = r22 + aD2 * nu2
            jn = j + i + 1
            for g in active:
                if jn % rec_step[g] == 0 and jn <= last[g]:
                    s0 = starts[g]
                    Xg = np.array([q11[s0:s0 + S], q12[s0:s0 + S], q22[s0:s0 + S], r11[s0:s0 + S],
                                   r12[s0:s0 + S], r21[s0:s0 + S], r22[s0:s0 + S]])
                    record(g, jn // rec_step[g], Xg)
        X[:, sl] = np.array([q11, q12, q22, r11, r12, r21, r22])
        j += nb
        active = [g for g in active if last[g] > j]
    return X


def main():
    t0 = time.time()
    eta, q0, S, tmax, rec_dt = 1.0, 0.25, 100, 180.0, 0.4
    dvals = [125, 250, 500, 1000]
    nrec = int(round(tmax / rec_dt)) + 1
    EG = np.full((len(dvals), nrec, S), np.nan)
    SPEC = np.full((len(dvals), S), np.nan)
    k40 = int(round(40 / rec_dt))

    def record(g, k, Xg):
        EG[g, k] = eg_of(Xg)
        if k == k40:
            SPEC[g] = 0.5 * np.abs(Xg[3] + Xg[6] - Xg[4] - Xg[5])

    simulate(dvals, S, eta, q0, tmax, rec_dt, rk.rng(1515), record)
    tgrid = rec_dt * np.arange(nrec)

    # learning curves (means with Student-t intervals) at selected times
    sel = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12.4, 15.2, 20, 25.2, 30, 35.2, 40] + list(np.arange(45.2, 180.1, 5.2))
    rows = []
    for t in sel:
        k = int(round(t / rec_dt))
        row = [tgrid[k]]
        for g in (0, 3):
            m, h = rk.mean_ci(EG[g, k])
            row += [m, h]
        rows.append(row)
    rk.write_dat(os.path.join(OUT, "15-sgd-plateau-curves.dat"), ["t", "eg125", "ci125", "eg1000", "ci1000"], rows)

    # escape times
    erows = []
    allc = []
    print("escape: first grid time with eps_g < eps*/2 = %.6f (grid %.1f); runs %d per d; t <= %.0f" % (EPS_PLATEAU / 2, rec_dt, S, tmax))
    med_lo_hi = []
    for g, d in enumerate(dvals):
        below = EG[g] < EPS_PLATEAU / 2
        hit = below.any(axis=0)
        tesc = np.where(hit, tgrid[np.argmax(below, axis=0)], np.inf)
        ok = np.isfinite(tesc)
        srt = np.sort(tesc)
        # distribution-free 95% interval for the median from binomial order statistics
        lo_i = int(stats.binom.ppf(0.025, S, 0.5)) - 1
        hi_i = int(stats.binom.ppf(0.975, S, 0.5))
        med = float(np.median(tesc))
        m, h = rk.mean_ci(tesc[ok]) if ok.all() else (np.nan, np.nan)
        sd = float(np.std(tesc[ok], ddof=1))
        erows.append((d, math.log(d), med, med - srt[lo_i], srt[hi_i] - med, m, h, sd))
        allc.append(tesc[ok] - math.log(d) / (2 * LAMBDA))
        print("  d=%5d: escaped %d/%d; median %.1f [%.1f, %.1f]; mean %.1f +- %.1f; std %.1f; eps_g(t=10) %.5f; sqrt(d) E|epsilon(40)| = %.3f"
              % (d, ok.sum(), S, med, srt[lo_i], srt[hi_i], m, h, sd, np.mean(EG[g, int(round(10 / rec_dt))]), math.sqrt(d) * np.mean(SPEC[g])))
    rk.write_dat(os.path.join(OUT, "15-sgd-plateau-escape.dat"),
                 ["d", "lnd", "median", "lo", "hi", "mean", "ci", "sd"], erows)
    x = np.array([r[1] for r in erows])
    # slope of the mean escape times against ln d: weighted least squares with the standard errors
    # of the means (sd/sqrt(S), from S independent runs each) as known variances, so that 1.96
    # standard errors give a 95% interval (an unweighted fit of 4 points has only 2 degrees of freedom)
    ym = np.array([r[5] for r in erows])
    w = S / np.array([r[7] for r in erows]) ** 2
    xb, yb = np.sum(w * x) / w.sum(), np.sum(w * ym) / w.sum()
    sxx = np.sum(w * (x - xb) ** 2)
    slope = np.sum(w * (x - xb) * (ym - yb)) / sxx
    chi2 = np.sum(w * (ym - yb - slope * (x - xb)) ** 2)
    print("  slope of the mean escape time against ln d (weighted): %.2f +- %.2f (95%%); heuristic 1/(2 lambda) = %.2f;"
          " chi2 of the line %.2f on 2 dof" % (slope, 1.96 / math.sqrt(sxx), 1 / (2 * LAMBDA), chi2))
    print("  slope of the median escape time against ln d (unweighted, point estimate): %.2f"
          % stats.linregress(x, np.array([r[2] for r in erows])).slope)
    # the heuristic line c + ln(d)/(2 lambda), with c fitted to the mean escape times (slope fixed)
    cfit = float(np.mean([r[5] - r[1] / (2 * LAMBDA) for r in erows]))
    dl = np.exp(np.linspace(math.log(100), math.log(1300), 30))
    rk.write_dat(os.path.join(OUT, "15-sgd-plateau-line.dat"), ["d", "tesc"],
                 [(float(x), cfit + math.log(x) / (2 * LAMBDA)) for x in dl])
    print("  fitted intercept with the slope fixed at 1/(2 lambda): c = %.2f" % cfit)
    pooled = np.concatenate([c - c.mean() for c in allc])
    print("  pooled std of escape times %.1f; heuristic 1.1107/lambda = %.1f" % (np.std(pooled, ddof=1), 1.1107 / LAMBDA))
    # quantiles of the centered escape times against those of -(ln|Z| - E ln|Z|)/lambda
    zq = lambda p: -math.log(stats.norm.ppf(0.5 + 0.5 * (1 - p))) / LAMBDA
    Eln = -0.5 * (np.euler_gamma + math.log(2))
    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        print("   quantile %.2f: measured %.1f, predicted %.1f" % (p, np.quantile(pooled, p), zq(p) + Eln / LAMBDA))
    print("runtime %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
