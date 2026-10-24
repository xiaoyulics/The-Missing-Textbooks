"""Chapter 15 (Online SGD): the search phase from a random start and the information exponent.

Spherical online SGD of Ben Arous, Gheissari and Jagannath (JMLR 2021, eq. (1.2)) on the
single-index model y = He_k(x_1)/sqrt(k!), x ~ N(0, I_d), with the correlation loss
l(w; x, y) = -y He_k(<w,x>)/sqrt(k!), whose population loss is -m^k, m = <w, e_1>, so that the
information exponent is k:
    w <- (w - (delta/d) grad_sph l) / |w - (delta/d) grad_sph l|,  one fresh sample per step,
from w uniform on the sphere conditioned on m > 0. Step sizes delta = 0.5 (k = 1, 2) and
delta = 0.5/sqrt(d) (k = 3). Weak recovery: the first step with m >= 1/2 (|m| for even k).

The overlap m is simulated exactly in law through its Markov chain: with u = <w,x> and zeta the
component of x along (e_1 - m w)/sqrt(1-m^2), independent N(0,1), x_1 = m u + sqrt(1-m^2) zeta and
|x|^2 - u^2 = zeta^2 + chi^2_{d-2}, so that
    m' = (m - (delta/d) a sqrt(1-m^2) zeta) / sqrt(1 + (delta/d)^2 a^2 (zeta^2 + chi^2_{d-2})),
    a = -He_k(x_1) He_k'(u)/k!.
200 runs per d. Writes figures/data/15-sgd-exponent.dat. Runtime: about 1 minute.
"""
import math
import os
import sys
import time

import numpy as np
from scipy import stats
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
HE = {1: (lambda u: u, lambda u: np.ones_like(u)),
      2: (lambda u: u * u - 1.0, lambda u: 2.0 * u),
      3: (lambda u: u * u * u - 3.0 * u, lambda u: 3.0 * (u * u - 1.0))}


def hitting_steps(k, dvals, S, delta_of_d, cap_of_d, gen, thr=0.5, block=2000):
    """Steps to weak recovery for S runs per dimension (all dimensions in one loop);
    runs that have not hit by their cap get +inf."""
    f, fp = HE[k]
    kf = float(math.factorial(k))
    dv = np.repeat(np.array(dvals, float), S)
    N = dv.size
    z = np.abs(gen.standard_normal(N))
    m = z / np.sqrt(z * z + gen.chisquare(dv - 1))
    c = np.array([delta_of_d(d) for d in dvals]).repeat(S) / dv
    cap = np.array([cap_of_d(d) for d in dvals]).repeat(S)
    hit = np.full(N, np.inf)
    alive = np.arange(N)
    t = 0
    while alive.size:
        nb = int(min(block, cap[alive].max() - t))
        n = alive.size
        U = gen.standard_normal((nb, n))
        Zt = gen.standard_normal((nb, n))
        CH = gen.chisquare(dv[alive] - 2, (nb, n))
        mm, ca0 = m[alive], c[alive]
        hb = np.full(n, np.inf)
        for j in range(nb):
            u, zz = U[j], Zt[j]
            s = np.sqrt(1.0 - mm * mm)
            x1 = mm * u + s * zz
            ca = ca0 * (-f(x1) * fp(u) / kf)
            mm = (mm - ca * s * zz) / np.sqrt(1.0 + ca * ca * (zz * zz + CH[j]))
            reach = (mm if k % 2 == 1 else np.abs(mm)) >= thr
            if reach.any():
                hb[reach & np.isinf(hb)] = t + j + 1
        t += nb
        m[alive] = mm
        hit[alive] = hb
        alive = alive[np.isinf(hb) & (cap[alive] > t)]
    return hit.reshape(len(dvals), S)


def median_ci(x):
    """Median and distribution-free 95% interval from binomial order statistics (x may contain inf)."""
    S = x.size
    srt = np.sort(x)
    lo = int(stats.binom.ppf(0.025, S, 0.5)) - 1
    hi = int(stats.binom.ppf(0.975, S, 0.5))
    return float(np.median(srt)), float(srt[lo]), float(srt[hi])


def main():
    t0 = time.time()
    S = 200
    plan = {1: ([128, 256, 512, 1024, 2048, 4096, 8192], lambda d: 0.5, lambda d: 10 * d),
            2: ([128, 256, 512, 1024, 2048, 4096, 8192], lambda d: 0.5, lambda d: 40 * d),
            3: ([32, 64, 128, 256], lambda d: 0.5 / math.sqrt(d), lambda d: 4 * d * d)}
    # deterministic predictions (the limiting ODE dm/dt = delta k m^{k-1}(1-m^2) - (delta^2/2) E[a^2] m)
    delta = 0.5
    t1half = quad(lambda mm: 1.0 / (delta * (1 - mm * mm) - 0.5 * delta * delta * mm), 0, 0.5)[0]
    r2, c2 = 2 * delta - delta ** 2, 2 * delta + 4 * delta ** 2
    zmed = stats.norm.ppf(0.75)                       # median of |Z|
    print("k=1: limiting ODE dm/dt = delta(1-m^2) - delta^2 m/2 reaches 1/2 at t = %.4f (delta = %.1f)" % (t1half, delta))
    print("k=2: dm/dt = m (r - c m^2), r = %.3f, c = %.3f; slope of n/d against ln d: 1/(2r) = %.4f" % (r2, c2, 1 / (2 * r2)))
    for k, (dvals, dl, capf) in plan.items():
        tk = time.time()
        H = hitting_steps(k, dvals, S, dl, capf, rk.rng(300 + k))
        logs = []
        rows = []
        for i, d in enumerate(dvals):
            med, lo, hi = median_ci(H[i])
            if k == 1:
                m0 = zmed / math.sqrt(d)
                pred = d * quad(lambda mm: 1.0 / (delta * (1 - mm * mm) - 0.5 * delta * delta * mm), m0, 0.5)[0]
            elif k == 2:
                m0 = zmed / math.sqrt(d)
                pred = d * math.log((r2 - c2 * m0 * m0) / (m0 * m0 * (4 * r2 - c2))) / (2 * r2)
            else:
                d0 = 0.5
                a = 2 / d0
                s0, s1 = zmed, math.sqrt(d) / 2
                pred = d * d * math.log((a - 1 / s1) / (a - 1 / s0)) / (1.5 * d0 * d0)
            rows.append((d, med, med - lo, hi - med, pred, float(np.mean(np.isfinite(H[i])))))
            logs.append((math.log(d), math.log(med)))
            print("k=%d d=%5d: median steps %9.0f [%9.0f, %9.0f]  median/d^%d = %.3f  deterministic ODE from the median m0: %9.0f  (hit %.2f)"
                  % (k, d, med, lo, hi, max(k - 1, 1), med / d ** max(k - 1, 1), pred, np.mean(np.isfinite(H[i]))))
        rk.write_dat(os.path.join(OUT, "15-sgd-exponent-k%d.dat" % k), ["d", "median", "lo", "hi", "pred", "hitfrac"], rows)
        xs, ys = np.array(logs).T
        print("   fitted exponent of d: %.3f  (%.0fs)" % (np.polyfit(xs, ys, 1)[0], time.time() - tk))
        if k == 2:
            nd = np.exp(ys) / np.exp(xs)
            res = stats.linregress(xs, nd)
            print("   k=2: median/d against ln d: slope %.3f +- %.3f, intercept %.3f (prediction 1/(2r) = %.4f)"
                  % (res.slope, 1.96 * res.stderr, res.intercept, 1 / (2 * r2)))
    print("runtime %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
