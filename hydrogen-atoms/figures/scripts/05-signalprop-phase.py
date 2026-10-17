"""Figure 1 of Chapter 5 (Order and Chaos at Initialization): the phase diagram of
infinitely wide tanh networks in the (sigma_w^2, sigma_b^2) plane, and the depth
scale xi = -1/log C'(cbar) along a horizontal cut sigma_b^2 = 0.05.

Theory only (Gauss--Hermite quadrature):
  length map    V(q) = sigma_b^2 + sigma_w^2 E[tanh(sqrt(q) z)^2], fixed point qbar;
  correlation   C(c) = (sigma_b^2 + sigma_w^2 E[tanh(sqrt(qbar) u) tanh(sqrt(qbar) v)])/qbar,
                with (u, v) standard Gaussians of correlation c;
  chi_1 = sigma_w^2 E[sech(sqrt(qbar) z)^4];  critical line chi_1 = 1;
  cbar = 1 in the ordered phase (chi_1 < 1), the fixed point of C in [0, 1) otherwise;
  xi = -1/log C'(cbar), C'(c) = sigma_w^2 E[sech^2(sqrt(qbar) u) sech^2(sqrt(qbar) v)].

Writes figures/data/05-signalprop-phase.dat (critical line and contours xi = 3, 10
on both sides) and figures/data/05-signalprop-xi.dat (xi along sigma_b^2 = 0.05).
Deterministic; about 30 s on a laptop.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

Z1, W1 = np.polynomial.hermite_e.hermegauss(120)
W1 = W1 / W1.sum()
Z2, W2 = np.polynomial.hermite_e.hermegauss(60)
W2 = W2 / W2.sum()
U, V = np.meshgrid(Z2, Z2, indexing="ij")
WW = np.outer(W2, W2)
sech2 = lambda t: 1.0 / np.cosh(t) ** 2


def qbar(sw2, sb2):
    """Fixed point of the length map, by bisection on V(q) - q (V is concave)."""
    V_ = lambda q: sb2 + sw2 * (W1 @ np.tanh(np.sqrt(q) * Z1) ** 2)
    if sb2 == 0.0 and sw2 <= 1.0:
        return 0.0
    lo, hi = 1e-12, sb2 + sw2 + 1.0
    return optimize.brentq(lambda q: V_(q) - q, lo, hi, xtol=1e-14)


def chi1(sw2, sb2):
    q = qbar(sw2, sb2)
    return sw2 * (W1 @ sech2(np.sqrt(q) * Z1) ** 2)


def corr_map(c, sw2, sb2, q):
    s = np.sqrt(q)
    a = np.tanh(s * U)
    b = np.tanh(s * (c * U + np.sqrt(max(1 - c * c, 0.0)) * V))
    return (sb2 + sw2 * np.sum(WW * a * b)) / q


def dcorr(c, sw2, q):
    s = np.sqrt(q)
    a = sech2(s * U)
    b = sech2(s * (c * U + np.sqrt(max(1 - c * c, 0.0)) * V))
    return sw2 * np.sum(WW * a * b)


def xi(sw2, sb2):
    q = qbar(sw2, sb2)
    x1 = sw2 * (W1 @ sech2(np.sqrt(q) * Z1) ** 2) if q > 0 else sw2
    if x1 <= 1.0:
        return -1.0 / np.log(x1)
    g = lambda c: corr_map(c, sw2, sb2, q) - c
    if sb2 == 0.0:
        cb = 0.0          # tanh is odd: C(0) = 0
    else:
        eps = 1e-2
        while g(1 - eps) >= 0:
            eps /= 3
        cb = optimize.brentq(g, 0.0, 1 - eps, xtol=1e-13)
    return -1.0 / np.log(dcorr(cb, sw2, q))


def critical(sb2):
    if sb2 == 0.0:
        return 1.0
    return optimize.brentq(lambda s: chi1(s, sb2) - 1.0, 1.0 + 1e-9, 30.0, xtol=1e-12)


def main():
    sb2s = np.linspace(0.0, 2.0, 41)
    rows = []
    for sb2 in sb2s:
        sc = critical(sb2)
        row = [sb2, sc]
        for x0 in (3.0, 10.0):
            row.append(optimize.brentq(lambda s: xi(s, sb2) - x0, 0.05, sc * (1 - 1e-6), xtol=1e-10))
        for x0 in (10.0, 3.0):
            f = lambda s: xi(s, sb2) - x0
            hi = sc * 1.0001
            while f(hi) > 0:
                hi *= 1.3
            row.append(optimize.brentq(f, sc * (1 + 1e-3), hi, xtol=1e-10))
        rows.append(row)
        print(" ".join(f"{v:.5f}" for v in row), flush=True)
    np.savetxt(os.path.join(DATA, "05-signalprop-phase.dat"), np.array(rows),
               header="sb2 crit ord3 ord10 cha10 cha3", comments="", fmt="%.6f")

    sb2 = 0.05
    sws = np.concatenate([np.linspace(0.5, 1.74, 60), np.linspace(1.74, 1.785, 40), np.linspace(1.79, 5.0, 80)])
    out = [(s, xi(s, sb2), chi1(s, sb2)) for s in sws]
    np.savetxt(os.path.join(DATA, "05-signalprop-xi.dat"), np.array(out),
               header="sw2 xi chi1", comments="", fmt="%.6f")
    print("critical sigma_w^2 at sigma_b^2=0.05:", critical(0.05))


if __name__ == "__main__":
    main()
