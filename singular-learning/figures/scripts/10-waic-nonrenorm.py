"""Chapter 10 (WAIC), "Where the theory stops": Watanabe's (2010d) non-renormalizable example.

Truth q = N(0, I_2) on (x, y); model p(x, y | a) = N((a, m(a)), I_2) with m(a) = sqrt(a^4 - a^2 + 1),
so that L(a) - L_0 = a^4 / 2 while the log-likelihood ratio has variance about a^2: the condition
E_q[f^2] <= c K of relatively finite variance fails, and the truth is not realizable (L_0 = S + 1/2).
Prior uniform on [-1.5, 1.5]. With phi_i(a) = log p(X_i|a) - log p(X_i|0) = a x_i + h(a) y_i - a^4/2,
h = m - 1, the posterior depends on the data through sum x_i and sum y_i; it is computed by
Gauss-Legendre quadrature (160 nodes) on the interval where its density exceeds 1e-17 of its maximum,
located on a grid of 40001 points. Per data set the script records
    Bg = G_n - L_0 = -E_{(x,y)~q} log E_a exp(a x + h(a) y - a^4/2)      (64 x 64 Gauss-Hermite)
    Te = T_n - L_n(0),  V = V_n,  LOO - WAIC,
and compares E[G_n - L_0] and E[V_n] with the asymptotic forms -c n^{-2/3} and 4 c n^{1/3},
c = Gamma(7/6) / (2^{5/3} sqrt(pi)) = 0.1649 (this chapter's computation of Watanabe's sketch;
the source prints the constant Q/2 = 2^{4/3} c), and checks that WAIC stays unbiased at order n^{-2/3}.

Run from the volume root:  python figures/scripts/10-waic-nonrenorm.py
Writes figures/data/10-waic-nonrenorm.dat.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import special

import sltkit as sk

NS = (100, 300, 1000, 3000, 10000)
RS = (400, 400, 400, 300, 200)
SEED0 = 4000
A = 1.5
AGRID = np.linspace(-A, A, 40001)
GLX, GLW = np.polynomial.legendre.leggauss(160)
GHX, GHW = np.polynomial.hermite_e.hermegauss(64)
GHW = GHW / math.sqrt(2 * math.pi)
GX, GY = [g.ravel() for g in np.meshgrid(GHX, GHX, indexing="ij")]
GW = np.outer(GHW, GHW).ravel()
_keep = GW > 1e-20
GX, GY, GW = GX[_keep], GY[_keep], GW[_keep]
C_PRED = special.gamma(7 / 6) / (2 ** (5 / 3) * math.sqrt(math.pi))
Q_SOURCE = 2 ** (7 / 6) * special.gamma(7 / 6) / math.sqrt(2 * math.pi)


def hfun(a):
    return np.sqrt(a ** 4 - a ** 2 + 1) - 1


def lse_weighted(phi, logw):
    """log sum_k exp(logw_k + phi_k) over axis 0 (phi of shape (K, m))."""
    z = phi + logw[:, None]
    mx = z.max(axis=0)
    return mx + np.log(np.exp(z - mx).sum(axis=0))


def analyse(x, y, chunk=20000):
    n = len(x)
    sx, sy = float(x.sum()), float(y.sum())
    E = -(AGRID * sx + hfun(AGRID) * sy - n * AGRID ** 4 / 2)            # n K_n(a) on a fine grid
    idx = np.nonzero(E - E.min() < -math.log(1e-17))[0]
    lo, hi = AGRID[max(idx[0] - 1, 0)], AGRID[min(idx[-1] + 1, len(AGRID) - 1)]
    a = 0.5 * (lo + hi) + 0.5 * (hi - lo) * GLX
    h = hfun(a)
    lw = np.log(0.5 * (hi - lo) * GLW) + (a * sx + h * sy - n * a ** 4 / 2)
    lw -= special.logsumexp(lw)
    w = np.exp(lw)
    q4 = a ** 4 / 2
    Te = V = d = 0.0
    for s in range(0, n, chunk):
        xs, ys = x[s:s + chunk], y[s:s + chunk]
        phi = a[:, None] * xs[None, :] + h[:, None] * ys[None, :] - q4[:, None]
        lp1, lm1 = lse_weighted(phi, lw), lse_weighted(-phi, lw)
        e1 = w @ phi
        v = w @ (phi * phi) - e1 * e1
        Te -= lp1.sum()
        V += v.sum()
        d += (lp1 + lm1 - v).sum()
    phig = a[:, None] * GX[None, :] + h[:, None] * GY[None, :] - q4[:, None]
    Bg = -float(GW @ lse_weighted(phig, lw))
    return Bg, Te / n, V, d / n


def main():
    t_start = time.time()
    print(f"predicted constants: c = Gamma(7/6)/(2^(5/3) sqrt(pi)) = {C_PRED:.4f}; 4c = {4 * C_PRED:.4f}; "
          f"source's Q/2 = {Q_SOURCE / 2:.4f}, 2Q = {2 * Q_SOURCE:.4f}")
    rows = []
    for n, R in zip(NS, RS):
        t0 = time.time()
        gen = sk.rng(SEED0 + n)
        res = np.array([analyse(gen.standard_normal(n), gen.standard_normal(n)) for _ in range(R)])
        Bg, Te, V, d = res.T
        c23 = n ** (2 / 3)
        bg = sk.mean_ci(c23 * Bg)
        vv = sk.mean_ci(V / n ** (1 / 3))
        wb = sk.mean_ci(c23 * (Te + V / n - Bg))
        ab = sk.mean_ci(c23 * (Te + 1.0 / n - Bg))
        dmed = float(np.median(np.abs(d)))
        rows.append([n, R, bg[0], bg[0] - bg[1], vv[0], vv[0] - vv[1], wb[0], wb[0] - wb[1], ab[0], ab[0] - ab[1],
                     dmed * n ** (4 / 3), dmed * n ** 2])
        print(f"n={n:6d} R={R} ({time.time() - t0:5.1f}s): n^(2/3) E[G-L0] = {bg[0]:+.3f} +- {bg[0] - bg[1]:.3f} "
              f"(pred {-C_PRED:+.3f})   E[V]/n^(1/3) = {vv[0]:.3f} +- {vv[0] - vv[1]:.3f} (pred {4 * C_PRED:.3f})   "
              f"n^(2/3) E[WAIC-G] = {wb[0]:+.3f} +- {wb[0] - wb[1]:.3f}   n^(2/3) E[T+1/n-G] = {ab[0]:+.3f} +- {ab[0] - ab[1]:.3f}   "
              f"median|LOO-WAIC|: x n^(4/3) = {dmed * n ** (4 / 3):.3f}, x n^2 = {dmed * n ** 2:.2f}")
    sk.write_dat("figures/data/10-waic-nonrenorm.dat",
                 ["n", "R", "bg", "bgci", "v", "vci", "w", "wci", "ab", "abci", "dmed43", "dmed2"], rows)
    print(f"total {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
