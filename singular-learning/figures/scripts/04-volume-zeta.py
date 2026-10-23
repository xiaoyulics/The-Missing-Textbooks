"""04-volume-zeta.py -- Chapter 4 (Volume), "Measuring it": the zeta function near its pole.

For the product model K = a^2 b^2 / 2 and the quartic cup K = (a^4 + b^4)/2, with the uniform prior
on [-1, 1]^2, the script evaluates the zeta function zeta(z) = E[K^z] for real z in (-1/2, 0]:

  * exactly: zeta(z) = 2^(-z) (2z+1)^(-2) for the product model; for the quartic cup, in polar
    coordinates, zeta(z) = 2^(-z)/(4z+2) * int_0^(pi/2) (cos^4 t + sin^4 t)^z max(cos t, sin t)^(-(4z+2)) dt,
    cross-checked against the volume representation zeta(z) = -z int_0^inf t^(z-1) V(t) dt;
  * by plain Monte Carlo over the prior, 10^7 draws per model, with the nominal 95% interval
    mean +- 1.96 sd/sqrt(N). The variance of K^z is finite only for z > -lambda/2 = -1/4, so for
    z <= -1/4 the nominal interval has no justification.

Theorem thm:volume:zeta(ii) predicts (z + 1/2)^m zeta(z) -> lambda C (m-1)! as z -> -1/2:
sqrt(2)/4 = 0.3536 for the product model (m = 2) and C_cup/2 = 0.6555 for the quartic cup (m = 1).
Outputs figures/data/04-volume-zeta-exact.dat and 04-volume-zeta-mc.dat.
Run from the volume root: python figures/scripts/04-volume-zeta.py   (about 15 s)
"""
import math
import os
import sys
import time

import numpy as np
from scipy import integrate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

OUT = "figures/data"
C_CUP = math.gamma(0.25) ** 2 / (4 * math.sqrt(2 * math.pi))      # V(eps) = C_CUP eps^(1/2), eps <= 1/2
LIM_PROD = math.sqrt(2) / 4                                         # lambda C (m-1)! = (1/2)(1/sqrt 2)(1!)
LIM_CUP = C_CUP / 2                                                 # lambda C = C/2


def zeta_prod(z):
    return 2.0 ** (-z) / (2 * z + 1) ** 2


def zeta_cup_polar(z):
    """Polar coordinates on [0,1]^2 (the four quadrants have equal shares of the uniform prior)."""
    f = lambda t: (math.cos(t) ** 4 + math.sin(t) ** 4) ** z * max(math.cos(t), math.sin(t)) ** (-(4 * z + 2))
    val = integrate.quad(f, 0, math.pi / 2, points=[math.pi / 4], limit=200, epsabs=1e-13, epsrel=1e-12)[0]
    return 2.0 ** (-z) * val / (4 * z + 2)


def v_cup(t):
    """V(t) for the quartic cup, any t > 0 (the share of [0,1]^2 where a^4 + b^4 < 2t)."""
    if t >= 1.0:
        return 1.0
    f = lambda a: min(1.0, max(2 * t - a ** 4, 0.0) ** 0.25)
    pts = [p for p in ((2 * t) ** 0.25, max(2 * t - 1, 0.0) ** 0.25) if 0.0 < p < 1.0]
    return integrate.quad(f, 0, 1, points=pts or None, limit=200, epsabs=1e-14, epsrel=1e-12)[0]


def zeta_cup_volume(z):
    """zeta(z) = 1 - z C 2^(-z-1/2)/(z+1/2) - z int_{1/2}^1 t^(z-1) V(t) dt  (V = C t^(1/2) below 1/2)."""
    tail = integrate.quad(lambda t: t ** (z - 1) * v_cup(t), 0.5, 1.0, limit=200, epsabs=1e-12)[0]
    return 1.0 - z * C_CUP * 2.0 ** (-z - 0.5) / (z + 0.5) - z * tail


def main():
    t0 = time.time()
    # exact curves
    zs = np.concatenate([np.linspace(-0.4975, -0.30, 80), np.linspace(-0.295, 0.0, 60)])
    rows = []
    for z in zs:
        zp, zc = zeta_prod(z), zeta_cup_polar(z)
        rows.append((z, zp, (z + 0.5) ** 2 * zp, zc, (z + 0.5) * zc))
    sk.write_dat(f"{OUT}/04-volume-zeta-exact.dat", ["z", "prod", "prodscaled", "cup", "cupscaled"], rows)
    print("limits predicted by thm:volume:zeta(ii): product %.6f, quartic cup %.6f" % (LIM_PROD, LIM_CUP))
    for z in (-0.4999, -0.499, -0.49, -0.45, -0.25, 0.0, 0.5):
        zc = zeta_cup_polar(z)
        line = f"  z={z:+.4f}: product (z+1/2)^2 zeta = {(z + 0.5) ** 2 * zeta_prod(z):.6f};  cup zeta = {zc:.8f}, (z+1/2) zeta = {(z + 0.5) * zc:.6f}"
        if -0.5 < z <= 0.0:
            line += f";  volume formula {zeta_cup_volume(z):.8f}"
        print(line)
    # direct check of the polar formula against two-dimensional quadrature at z = 1/2
    direct = integrate.dblquad(lambda b, a: (0.5 * (a ** 4 + b ** 4)) ** 0.5, 0, 1, 0, 1, epsabs=1e-12)[0]
    print(f"  cup zeta(1/2): polar {zeta_cup_polar(0.5):.10f}, direct 2-D quadrature {direct:.10f}")

    # plain Monte Carlo over the prior
    zgrid = np.round(np.arange(-0.05, -0.451, -0.05), 2)
    N, chunk = 10_000_000, 1_000_000
    mc = {}
    for key, seed, K in (("prod", 411, lambda w: 0.5 * (w[:, 0] * w[:, 1]) ** 2),
                         ("cup", 412, lambda w: 0.5 * (w[:, 0] ** 4 + w[:, 1] ** 4))):
        gen = sk.rng(seed)
        s1 = np.zeros(len(zgrid)); s2 = np.zeros(len(zgrid))
        for _ in range(N // chunk):
            Kw = K(gen.uniform(-1.0, 1.0, size=(chunk, 2)))
            P = Kw[None, :] ** zgrid[:, None]
            s1 += P.sum(axis=1); s2 += (P * P).sum(axis=1)
        mean = s1 / N
        sd = np.sqrt(np.maximum(s2 / N - mean ** 2, 0.0) * N / (N - 1))
        mc[key] = (mean, sd / math.sqrt(N))
    mrows = []
    print(f"\nplain Monte Carlo, N = {N:.0e} draws per model (seeds 411, 412); nominal 95% = mean +- 1.96 se")
    for i, z in enumerate(zgrid):
        tp, tc = zeta_prod(z), zeta_cup_polar(z)
        sp, sc = (z + 0.5) ** 2, (z + 0.5)
        mp_, sep = mc["prod"][0][i], mc["prod"][1][i]
        mcu, sec = mc["cup"][0][i], mc["cup"][1][i]
        mrows.append((z, sp * mp_, 1.96 * sp * sep, sc * mcu, 1.96 * sc * sec, sp * tp, sc * tc))
        print(f"  z={z:+.2f}  product: true {tp:9.4f}  MC {mp_:9.4f} +- {1.96 * sep:7.4f}  ratio {mp_ / tp:.3f} "
              f"covered {abs(mp_ - tp) <= 1.96 * sep};   cup: true {tc:8.4f}  MC {mcu:8.4f} +- {1.96 * sec:7.4f}  "
              f"ratio {mcu / tc:.3f} covered {abs(mcu - tc) <= 1.96 * sec}")
    sk.write_dat(f"{OUT}/04-volume-zeta-mc.dat", ["z", "prod", "prodhalf", "cup", "cuphalf", "prodtrue", "cuptrue"], mrows)
    print(f"total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
