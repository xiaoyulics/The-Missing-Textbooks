"""Prologue: the free energy and the posterior of y = abx at its singular truth.

1. Posterior draws (exact, sltkit.sample_product_posterior) for one data set at each of
   n = 10^2, 10^4, 10^6, 1500 draws each -> figures/data/00-prologue-draws-n{2,4,6}.dat.
2. The normalized free energy F_n - nS_n, averaged over 200 data sets at each n in
   10^2, 10^2.5, ..., 10^6, for the product model y = abx (exact quadrature in c = ab) and for
   the regular plane y = a x1 + b x2 (Gaussian integral; the box [-1, 1]^2 is at least ten
   posterior standard deviations wide, so truncation is negligible). Both have two parameters
   and the uniform prior on [-1, 1]^2; the truth is w0 = 0.
   Predictions: regular plane log n - 1 - log(2 pi) + log 4 (Laplace, exact up to O(1/n));
   product model, the mean over Z ~ N(0, 1) of the rescaled formula of Chapter 3 (S_xx replaced
   by n, t = sqrt(n) c):
       F_n - nS_n = (1/2) log n - Z^2/2 - log I_n(Z),
       I_n(Z) = int_{|t| < sqrt(n)} exp(-(t - Z)^2 / 2) (1/2) ((1/2) log n - log|t|) dt,
   which tends to (1/2) log n - log log n - 1/2 + log(4/sqrt(2 pi)).
   -> figures/data/00-prologue-free.dat

Run from the volume root:  python figures/scripts/00-prologue-free.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import integrate  # noqa: E402

import sltkit as sk  # noqa: E402

REPS = 200
EXPONENTS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]


def I_n(z: float, n: float) -> float:
    r = math.sqrt(n)
    f = lambda t: math.exp(-0.5 * (t - z) ** 2) * 0.5 * (0.5 * math.log(n) - math.log(abs(t)))
    lo, hi = max(-r, min(z, 0.0) - 12.0), min(r, max(z, 0.0) + 12.0)
    return integrate.quad(f, lo, 0.0, limit=200)[0] + integrate.quad(f, 0.0, hi, limit=200)[0]


def product_prediction(n: float, nodes=40) -> float:
    zs, wz = np.polynomial.hermite_e.hermegauss(nodes)
    wz = wz / wz.sum()
    vals = [-0.5 * z * z - math.log(I_n(z, n)) for z in zs]
    return 0.5 * math.log(n) + float(np.dot(wz, vals))


def main() -> None:
    os.makedirs(os.path.join("figures", "data"), exist_ok=True)
    # 1. posterior draws
    for e in (2, 4, 6):
        n = 10 ** e
        x, y = sk.PRODUCT.sample([0.0, 0.0], n, seed=100 + e)
        sk.PRODUCT.fit(x, y)
        d = sk.sample_product_posterior(sk.PRODUCT, 1500, seed=200 + e)
        sk.write_dat(os.path.join("figures", "data", f"00-prologue-draws-n{e}.dat"), ["a", "b"], d)
        frac = float(np.mean(np.abs(d[:, 0] * d[:, 1]) < 3 / math.sqrt(n)))
        print(f"draws n=1e{e}: c_hat*sqrt(n) = {sk.PRODUCT.Sxy[0] / sk.PRODUCT.Sxx[0, 0] * math.sqrt(n):+.3f}; "
              f"fraction with |ab| < 3/sqrt(n): {frac:.3f}")
    # 2. free energies
    rows = []
    for ex in EXPONENTS:
        n = int(round(10 ** ex))
        prod, reg = [], []
        for r in range(REPS):
            x, y = sk.PRODUCT.sample([0.0, 0.0], n, seed=10_000 * int(ex * 2) + r)
            sk.PRODUCT.fit(x, y)
            nS = float(sk.PRODUCT.energy_of_coefficients(np.array([0.0])))
            prod.append(sk.free_energy_chain(sk.PRODUCT, 2) - nS)
            x2, y2 = sk.REGULAR2.sample([0.0, 0.0], n, seed=20_000 * int(ex * 2) + r)
            sk.REGULAR2.fit(x2, y2)
            nS2 = float(sk.REGULAR2.energy_of_coefficients(np.zeros(2)))
            F2 = sk.REGULAR2.min_energy() + 0.5 * math.log(np.linalg.det(sk.REGULAR2.Sxx)) \
                - math.log(2 * math.pi) + math.log(4.0)
            reg.append(F2 - nS2)
        mp, lp, hp = sk.mean_ci(prod)
        mr, lr, hr = sk.mean_ci(reg)
        pred_p = product_prediction(n)
        pred_r = math.log(n) - 1 - math.log(2 * math.pi) + math.log(4)
        limit_p = 0.5 * math.log(n) - math.log(math.log(n)) - 0.5 + math.log(4 / math.sqrt(2 * math.pi))
        rows.append((n, mp, mp - lp, hp - mp, pred_p, limit_p, mr, mr - lr, hr - mr, pred_r))
        print(f"n=1e{ex:.1f}: product {mp:7.3f} [{lp:7.3f},{hp:7.3f}] pred {pred_p:7.3f} limit {limit_p:7.3f} | "
              f"regular {mr:7.3f} [{lr:7.3f},{hr:7.3f}] pred {pred_r:7.3f}")
    sk.write_dat(os.path.join("figures", "data", "00-prologue-free.dat"),
                 ["n", "prod", "prodlo", "prodhi", "prodpred", "prodlimit", "reg", "reglo", "reghi", "regpred"],
                 rows)
    sl_p = np.polyfit(np.log([r[0] for r in rows]), [r[1] for r in rows], 1)[0]
    sl_r = np.polyfit(np.log([r[0] for r in rows]), [r[6] for r in rows], 1)[0]
    print(f"least-squares slope against log n: product {sl_p:.3f}, regular plane {sl_r:.3f}")


if __name__ == "__main__":
    main()
