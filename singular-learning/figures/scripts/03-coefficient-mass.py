"""Chapter 3 (The Free Energy by Hand): densities of the coefficient and the prior mass of a window.

For the five one-coefficient models of the chapter (uniform prior on [-1, 1]^d, truth c0 = 0)
    line  c = a;  prod  c = ab;  chain  c = a1 a2 a3;  cusp  c = a b^2;  cubic  c = a b^3,
this script tabulates
  1. the densities rho(c) of the coefficient (Computation 3.2), on 0.002 <= |c| <= 0.6
     -> figures/data/03-coefficient-density.dat (columns c, line, prod, chain, cusp, cubic, window);
     the column 'window' is the Gaussian window exp(-n (c - c_hat)^2 / 2) for n = 400 and
     c_hat = 0.05, drawn in the figure for scale;
  2. the prior mass of the window M(r) = P(|c| < r), 1e-8 <= r <= 1, from the closed forms
         line r;  prod r (1 + log(1/r));  chain r (1 + log(1/r) + log(1/r)^2 / 2);
         cusp 2 r^(1/2) - r;  cubic (3/2) r^(1/3) - r/2
     -> figures/data/03-coefficient-mass.dat;
  3. a Monte Carlo check of the closed forms: 10^7 draws from the prior (seed 31), the fraction
     with |c| < r for r = 1e-1, 1e-2, 1e-3, 1e-4, with Clopper-Pearson 95% intervals (printed);
  4. the window heuristic: -log M(n^(-1/2)) against lambda log n - (m - 1) log log n (printed).
Run from the volume root:  python figures/scripts/03-coefficient-mass.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import sltkit as sk  # noqa: E402


def mass(name, r):
    r = np.asarray(r, dtype=float)
    u = np.log(1.0 / r)
    return {"line": r, "prod": r * (1 + u), "chain": r * (1 + u + 0.5 * u * u),
            "cusp": 2 * np.sqrt(r) - r, "cubic": 1.5 * r ** (1 / 3) - 0.5 * r}[name]


def density(name, c):
    a = np.abs(np.asarray(c, dtype=float))
    return {"line": 0.5 * np.ones_like(a), "prod": 0.5 * np.log(1 / a), "chain": 0.25 * np.log(1 / a) ** 2,
            "cusp": 0.5 * (a ** -0.5 - 1), "cubic": 0.25 * (a ** (-2 / 3) - 1)}[name]


NAMES = ["line", "prod", "chain", "cusp", "cubic"]
LAM_M = {"line": (0.5, 1), "prod": (0.5, 2), "chain": (0.5, 3), "cusp": (0.25, 1), "cubic": (1 / 6, 1)}


def main() -> None:
    out = os.path.join("figures", "data")
    # 1. densities
    pos = np.concatenate([np.linspace(0.002, 0.02, 60), np.linspace(0.021, 0.6, 140)])
    cs = np.concatenate([-pos[::-1], pos])
    window = np.exp(-400 * (cs - 0.05) ** 2 / 2)
    rows = [[c] + [float(density(k, c)) for k in NAMES] + [float(w)] for c, w in zip(cs, window)]
    sk.write_dat(os.path.join(out, "03-coefficient-density.dat"), ["c"] + NAMES + ["window"], rows)
    # 2. window masses
    rs = np.logspace(-8, 0, 161)
    sk.write_dat(os.path.join(out, "03-coefficient-mass.dat"), ["r"] + NAMES,
                 [[r] + [float(mass(k, r)) for k in NAMES] for r in rs])
    # 3. Monte Carlo check of the closed forms
    gen = sk.rng(31)
    N = 10_000_000
    w = gen.uniform(-1.0, 1.0, size=(N, 3))
    coef = {"line": w[:, 0], "prod": w[:, 0] * w[:, 1], "chain": w[:, 0] * w[:, 1] * w[:, 2],
            "cusp": w[:, 0] * w[:, 1] ** 2, "cubic": w[:, 0] * w[:, 1] ** 3}
    ok = True
    for k in NAMES:
        a = np.abs(coef[k])
        for r in (1e-1, 1e-2, 1e-3, 1e-4):
            cnt = int(np.sum(a < r))
            lo, hi = sk.clopper_pearson(cnt, N)
            exact = float(mass(k, r))
            ok &= lo <= exact <= hi
            print(f"M_{k}({r:g}) = {exact:.6f}; Monte Carlo {cnt / N:.6f} [{lo:.6f}, {hi:.6f}]")
    print("all closed forms inside the Monte Carlo intervals:", ok)
    # 4. the window heuristic
    for k in NAMES:
        lam, m = LAM_M[k]
        for n in (1e4, 1e8):
            h = -math.log(float(mass(k, n ** -0.5)))
            print(f"-log M_{k}(n^-1/2) at n={n:g}: {h:.3f}; lambda log n - (m-1) log log n = "
                  f"{lam * math.log(n) - (m - 1) * math.log(math.log(n)):.3f}")


if __name__ == "__main__":
    main()
