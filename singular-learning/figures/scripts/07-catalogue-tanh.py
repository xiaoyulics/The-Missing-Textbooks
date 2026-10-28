"""07-catalogue-tanh.py -- a pre-asymptotic warning: the volume exponent of a two-unit tanh network.

Run from the volume root:  python figures/scripts/07-catalogue-tanh.py   (about one minute)

Model y = a1 tanh(b1 x) + a2 tanh(b2 x) + noise, x ~ Unif[-1, 1], unit noise, truth 0, uniform prior on
[-1, 1]^4, so K(w) = (1/2) E_x[(a1 tanh(b1 x) + a2 tanh(b2 x))^2] (24-point Gauss--Legendre in x). Its
learning coefficient is 2/3 with m = 1 (Aoyagi's one-input formula; Chapter 7). We estimate the volume
V(eps) = phi{K < eps} by Monte Carlo (sk.volume_fraction-style counting, 2 x 10^7 prior draws) and fit
exponents over windows of eps. The same is done for the polynomial S1^2 + S3^2 with S_p = a1 b1^p + a2 b2^p,
which is comparable to K near the origin (Chapter 7), with 4 x 10^7 draws, to show where the asymptotic
regime begins. Writes figures/data/07-catalogue-tanh.dat.
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

XQ, WQ = np.polynomial.legendre.leggauss(24)
WQ = WQ / 2.0                                   # expectation over Unif[-1, 1]


def counts(Kfun, eps, total, chunk, seed):
    gen = sk.rng(seed)
    c = np.zeros(len(eps), dtype=np.int64)
    for _ in range(total // chunk):
        w = gen.uniform(-1.0, 1.0, (chunk, 4))
        K = Kfun(w)
        c += np.searchsorted(np.sort(K), eps, side="left")
    return c


def K_tanh(w):
    a, b = w[:, :2], w[:, 2:]
    f = np.einsum("nh,nhq->nq", a, np.tanh(b[:, :, None] * XQ[None, None, :]))
    return 0.5 * (f * f) @ WQ


def K_poly(w):
    a, b = w[:, :2], w[:, 2:]
    return np.sum(a * b, axis=1) ** 2 + np.sum(a * b ** 3, axis=1) ** 2


def fit(eps, c, total, lo, hi):
    sel = (eps >= lo) & (eps <= hi) & (c >= 100)
    return sk.loglog_slope(eps[sel], c[sel] / total)[0]


def main():
    t0 = time.time()
    eps = np.logspace(-1.5, -5.0, 15)
    total = 20_000_000
    c = counts(K_tanh, eps, total, 200_000, 7070)
    rows = []
    for e, k in zip(eps, c):
        lo, hi = sk.clopper_pearson(int(k), total)
        rows.append((e, k / total, lo, hi, int(k)))
    print("tanh network, H = 2: eps, V(eps), count")
    for r in rows:
        print(f"   {r[0]:.2e}  {r[1]:.3e}  {r[4]}")
    print(f"fitted exponent of V on [1e-5, 3e-2]: {fit(eps, c, total, 1e-5, 4e-2):.3f};  on [1e-5, 1e-3]: "
          f"{fit(eps, c, total, 1e-5, 1e-3):.3f};  on [1e-5, 1e-4]: {fit(eps, c, total, 1e-5, 1.1e-4):.3f}  (lambda = 2/3 = 0.667)")
    eps2 = np.logspace(-2.0, -8.0, 19)
    total2 = 40_000_000
    c2 = counts(K_poly, eps2, total2, 1_000_000, 7071)
    print(f"polynomial S1^2 + S3^2: fitted exponent on [1e-8, 1e-4]: {fit(eps2, c2, total2, 1e-8, 1.1e-4):.3f}; "
          f"on [1e-5, 3e-2]: {fit(eps2, c2, total2, 1e-5, 4e-2):.3f}; counts at 1e-8: {int(c2[-1])}")
    # why the regime is late: to third order, K = (1/2) E[(S1 x - S3 x^3/3)^2] is a quadratic form in
    # (S1, S3) whose Gram matrix under Unif[-1, 1] is nearly singular
    G = 0.5 * np.array([[np.sum(WQ * XQ ** 2), -np.sum(WQ * XQ ** 4) / 3],
                        [-np.sum(WQ * XQ ** 4) / 3, np.sum(WQ * XQ ** 6) / 9]])
    schur = G[1, 1] - G[0, 1] ** 2 / G[0, 0]
    print(f"to third order K = {G[0, 0]:.4f} (S1 + {G[0, 1] / G[0, 0]:.3f} S3)^2 + {schur:.5f} S3^2: the second "
          f"coefficient is {schur / G[0, 0]:.4f} times the first; eigenvalues of the Gram matrix {np.linalg.eigvalsh(G)}")
    sk.write_dat("figures/data/07-catalogue-tanh.dat", ["eps", "V", "lo", "hi", "count"], rows)
    print(f"total time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
