"""Chapter 5 (Spiked Matrices): the replica-symmetric phase diagram of spiked Wigner with the sparse
Rademacher prior, and the Bayes overlap against the spectral overlap.

Model: A = (lam/N) x x^T + W, W ~ GOE(N); prior P0 = (1-rho) delta_0 + (rho/2)(delta_a + delta_{-a}),
a = 1/sqrt(rho), so E X^2 = 1.  Scalar channel Y = sqrt(g) X + Z; overlap map T(g) = 1 - mmse(g).
Replica-symmetric potential (Chapter 4): F(lam, q) = -lam^2 q^2/4 + psi(lam^2 q), psi' = T/2 (I-MMSE).
Stationary points q = T(lam^2 q), parametrized by g = lam^2 q:
    lam^2(g) = g / T(g),   q = T(g),   stable (a local maximum of F) iff d lam^2/dg > 0,
    F at the stationary point minus F(lam, 0):  dphi(g) = (1/2) int_0^g T(u) du - g T(g)/4.
Thresholds (rho < rho_tri):
    lam_alg = sqrt(local max of lam^2(g)): the branch born at q = 0 for lam = 1 ends (AMP's spinodal);
    lam_dyn = sqrt(local min of lam^2(g)): the high-overlap branch appears;
    lam_IT  = where the high branch overtakes the best competitor (q = 0 for lam <= 1, the low branch
              for lam > 1) -- Maxwell's construction.
Derivative T'(g) = E[Var(X|Y)^2] (Guo, Wu, Shamai and Verdu 2011), checked here by finite differences.

Outputs: figures/data/05-spiked-phase.dat (thresholds against rho),
         figures/data/05-spiked-branches.dat (stationary branches at rho = 0.05),
         figures/data/05-spiked-bayes.dat (Bayes overlap at rho = 0.05, with the jump),
         figures/data/05-spiked-overlaps.dat (spectral, Rademacher and sparse Bayes overlaps).
Run from the volume root: python figures/scripts/05-spiked-phase.py   (about one minute)
"""
import math
import os
import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

# trapezoid rule on [-11, 11] for E f(Z): spectrally accurate for smooth integrands with Gaussian decay
ZG = np.linspace(-11.0, 11.0, 4401)
WZ = np.exp(-ZG ** 2 / 2) / math.sqrt(2 * math.pi) * (ZG[1] - ZG[0])


def post(y, g, rho):
    """Posterior mean and variance of X given sqrt(g) X + Z = y (vectorized in y)."""
    a = 1.0 / math.sqrt(rho)
    sg = math.sqrt(g)
    lp = math.log(rho / 2) + sg * a * y - g * a * a / 2
    lm = math.log(rho / 2) - sg * a * y - g * a * a / 2
    l0 = math.log(1 - rho) + 0.0 * y
    mx = np.maximum(np.maximum(lp, lm), l0)
    wp, wm, w0 = np.exp(lp - mx), np.exp(lm - mx), np.exp(l0 - mx)
    tot = wp + wm + w0
    m1 = a * (wp - wm) / tot
    return m1, a * a * (wp + wm) / tot - m1 ** 2


def T(g, rho):
    """1 - mmse(g) = E[X E(X|Y)] = rho a E[m(sqrt(g) a + Z)] (the X = -a term equals the X = +a term)."""
    if g <= 0:
        return 0.0
    a = 1.0 / math.sqrt(rho)
    m1, _ = post(math.sqrt(g) * a + ZG, g, rho)
    return rho * a * float(np.sum(WZ * m1))


def Tprime(g, rho):
    """E[Var(X|Y)^2]."""
    a = 1.0 / math.sqrt(rho)
    _, va = post(math.sqrt(g) * a + ZG, g, rho)
    _, v0 = post(ZG, g, rho)
    return rho * float(np.sum(WZ * va ** 2)) + (1 - rho) * float(np.sum(WZ * v0 ** 2))


def lam2(g, rho):
    return g / T(g, rho)


def dloglam2(g, rho):
    return 1.0 / g - Tprime(g, rho) / T(g, rho)


def dphi(g, rho):
    return 0.5 * quad(lambda s: T(s, rho), 0.0, g, epsabs=1e-13, epsrel=1e-12, limit=200)[0] - g * T(g, rho) / 4


def extrema(rho, gmin=1e-5, gmax=60.0, n=250):
    """(g1, g2): local maximum and local minimum of lam^2(g), or None if lam^2 increases."""
    gs = np.geomspace(gmin, gmax, n)
    d = np.array([dloglam2(g, rho) for g in gs])
    idx = np.where(np.sign(d[1:]) != np.sign(d[:-1]))[0]
    if len(idx) < 2:
        return None
    return tuple(brentq(lambda g: dloglam2(g, rho), gs[i], gs[i + 1], xtol=1e-15) for i in idx[:2])


def on_branch(rho, L, lo, hi):
    return brentq(lambda g: lam2(g, rho) - L, lo, hi, xtol=1e-15)


def thresholds(rho):
    ex = extrema(rho)
    if ex is None:
        return None
    g1, g2 = ex
    out = dict(g1=g1, g2=g2, lam_alg=math.sqrt(lam2(g1, rho)), lam_dyn=math.sqrt(lam2(g2, rho)))
    # competitor q = 0 (valid when the result is <= 1)
    if dphi(g2, rho) < 0:
        gH = brentq(lambda g: dphi(g, rho), g2, 300.0, xtol=1e-14)
        if lam2(gH, rho) <= 1.0:
            out.update(lam_IT=math.sqrt(lam2(gH, rho)), qlo=0.0, qhi=T(gH, rho))
            return out
    # Maxwell construction between the low branch (0, g1) and the high branch (g2, inf), above lam = 1

    def D(L):
        return dphi(on_branch(rho, L, g2, 500.0), rho) - dphi(on_branch(rho, L, 1e-7, g1), rho)
    Lit = brentq(D, max(1.0 + 1e-6, lam2(g2, rho) * (1 + 1e-10)), lam2(g1, rho) * (1 - 1e-12), xtol=1e-15)
    out.update(lam_IT=math.sqrt(Lit), qlo=T(on_branch(rho, Lit, 1e-7, g1), rho),
               qhi=T(on_branch(rho, Lit, g2, 500.0), rho))
    return out


def rho_tri():
    def mind(rho):
        gs = np.geomspace(0.02, 5.0, 200)
        d = np.array([dloglam2(g, rho) for g in gs])
        i = int(np.argmin(d))
        r = minimize_scalar(lambda g: dloglam2(g, rho), bounds=(gs[max(i - 1, 0)], gs[min(i + 1, 199)]),
                            method="bounded", options={"xatol": 1e-12})
        return r.fun, r.x
    rt = brentq(lambda r: mind(r)[0], 0.09, 0.10, xtol=1e-11)
    gt = mind(rt)[1]
    return rt, gt, math.sqrt(lam2(gt, rt))


def rho_star():
    """lam_IT = 1: the high branch at lam = 1 has the free entropy of q = 0."""
    def h(r):
        g1, g2 = extrema(r)
        return dphi(on_branch(r, 1.0, g2, 500.0), r)
    return brentq(h, 0.084, 0.0869, xtol=1e-11)


def rho_dyn():
    return brentq(lambda r: thresholds(r)["lam_dyn"] - 1.0, 0.0863, 0.0909, xtol=1e-11)


def main():
    t0 = time.time()
    # --- checks -------------------------------------------------------------------------------
    print("checks: T'(0.7) = %.10f, finite difference %.10f (rho = 0.05)"
          % (Tprime(0.7, 0.05), (T(0.7 + 1e-5, 0.05) - T(0.7 - 1e-5, 0.05)) / 2e-5))
    print("        T against the kit (1 - mmse_sparse_rademacher): %.2e at (g, rho) = (0.7, 0.05), %.2e at (2, 0.1)"
          % (T(0.7, 0.05) - (1 - rk.mmse_sparse_rademacher(0.7, 0.05)), T(2.0, 0.1) - (1 - rk.mmse_sparse_rademacher(2.0, 0.1))))
    for rho in [0.05, 0.2]:
        c3 = (1 - 6 * rho + 15 * rho ** 2) / (6 * rho ** 2)
        g = 1e-3
        print("        local expansion, rho=%.2f: (T(g)-g)/g^2 at g=1e-3 is %.6f; -1 + c3 g = %.6f"
              % (rho, (T(g, rho) - g) / g ** 2, -1 + c3 * g))

    # --- the two critical densities and the table ------------------------------------------------
    rt, gt, lt = rho_tri()
    rs = rho_star()
    rd = rho_dyn()
    print("\nrho_tri = %.6f (three stationary points for rho < rho_tri; the jump vanishes at lam = %.5f, g = %.4f)" % (rt, lt, gt))
    print("rho_*   = %.6f (lam_IT = 1)" % rs)
    print("rho_dyn = %.6f (lam_dyn = 1)" % rd)
    print("\n rho      lam_IT     lam_dyn    lam_alg    | q below, q above the jump | LKZ Delta/rho^2 = 1/lam^2 (IT, Dyn, Alg) | cubic estimate of lam_alg")
    for rho in [0.0909, 0.0863, 0.08, 0.06, 0.05, 0.03, 0.02]:
        t = thresholds(rho)
        c3 = (1 - 6 * rho + 15 * rho ** 2) / (6 * rho ** 2)
        approx = math.sqrt(1 + 1 / (4 * (c3 - 1)))
        print(" %.4f   %.6f   %.6f   %.6f   | %.4f %.4f | %.5f %.5f %.5f | %.6f"
              % (rho, t["lam_IT"], t["lam_dyn"], t["lam_alg"], t["qlo"], t["qhi"],
                 t["lam_IT"] ** -2, t["lam_dyn"] ** -2, t["lam_alg"] ** -2, approx))

    # --- the phase diagram ------------------------------------------------------------------------
    grid = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.075, 0.08, 0.085, 0.0875,
            0.09, 0.092, 0.094, 0.095, 0.0958, 0.0962]
    rows = []
    for rho in grid:
        t = thresholds(rho)
        rows.append([rho, t["lam_IT"], t["lam_dyn"], t["lam_alg"]])
    rows.append([rt, lt, lt, lt])
    rk.write_dat("figures/data/05-spiked-phase.dat", ["rho", "lamIT", "lamdyn", "lamalg"], rows)
    print("\nphase diagram written (%d densities), %.1f s" % (len(rows), time.time() - t0))

    # --- stationary branches at rho = 0.05 ---------------------------------------------------------
    rho = 0.05
    t = thresholds(rho)
    g1, g2 = t["g1"], t["g2"]
    br = []
    for g in np.geomspace(1e-5, g1, 40):                       # low branch (born at lam = 1)
        br.append([math.sqrt(lam2(g, rho)), T(g, rho), 0])
    for g in np.geomspace(g1, g2, 160):                        # middle branch (unstable)
        br.append([math.sqrt(lam2(g, rho)), T(g, rho), 1])
    ghi = on_branch(rho, 1.2 ** 2, g2, 500.0)
    for g in np.geomspace(g2, ghi, 120):                       # high branch
        br.append([math.sqrt(lam2(g, rho)), T(g, rho), 2])
    rk.write_dat("figures/data/05-spiked-branches.dat", ["lam", "q", "branch"], br)
    # the Bayes overlap: 0 below lam_IT, the high branch above (with a nan row at the jump)
    bayes = [[0.5, 0.0], [t["lam_IT"], 0.0], ["nan", "nan"]]
    g3 = on_branch(rho, 3.0 ** 2, g2, 500.0)
    for g in np.geomspace(on_branch(rho, t["lam_IT"] ** 2, g2, 500.0), g3, 160):
        bayes.append([math.sqrt(lam2(g, rho)), T(g, rho)])
    rk.write_dat("figures/data/05-spiked-bayes.dat", ["lam", "q"], bayes)
    print("rho = 0.05: lam_IT = %.4f (jump 0 -> %.4f), lam_dyn = %.4f, lam_alg = %.6f, g1 = %.5f, g2 = %.4f"
          % (t["lam_IT"], t["qhi"], t["lam_dyn"], t["lam_alg"], g1, g2))
    for L in [0.86, 0.92, 0.98]:
        gm = on_branch(rho, L ** 2, g1, g2)
        gh = on_branch(rho, L ** 2, g2, 500.0)
        print("   lam = %.2f: unstable middle q = %.4f, high branch q = %.4f" % (L, T(gm, rho), T(gh, rho)))

    # --- Bayes against spectral: Rademacher, Gaussian (= spectral), sparse rho = 0.05 ----------------
    ov = []
    for L in np.linspace(0.5, 3.0, 251):
        L = float(L)
        spec = rk.bbp_overlap(L)
        rad = rk.rs_overlap(L)[0] if L > 1 else 0.0
        if L < t["lam_IT"]:
            sp = 0.0
        else:
            sp = T(on_branch(rho, L ** 2, g2, 500.0), rho)
        ov.append([L, spec, rad, sp])
    rk.write_dat("figures/data/05-spiked-overlaps.dat", ["lam", "spec", "rad", "sparse"], ov)
    print("\nBayes overlap q* against the spectral overlap 1 - 1/lam^2:")
    for L in [1.1, 1.2, 1.5, 2.0, 3.0]:
        q = rk.rs_overlap(L)[0]
        print("   lam = %.1f: Rademacher q* = %.6f, spectral %.6f; matrix MMSE 1 - q*^2 = %.6f vs spectral %.6f"
              % (L, q, 1 - L ** -2, 1 - q * q, 1 - (1 - L ** -2) ** 2))
    # Montanari--Venkataramanan's functional (2.31): with g^2/(4 lam^2) it is stationary at the fixed point
    L = 1.5
    q = rk.rs_overlap(L)[0]
    g = L * L * q
    I = lambda s: s - rk.gauss(lambda z: np.log(np.cosh(s + math.sqrt(s) * z)), 201)   # Rademacher MI
    dI = lambda s: (I(s + 1e-6) - I(s - 1e-6)) / 2e-6
    print("MV21 (2.31) at the Rademacher fixed point, lam = 1.5: d/dg with g^2/(4 lam^2): %.2e; as printed g^2/(4 lam): %.4f"
          % (g / (2 * L * L) - 0.5 + dI(g), g / (2 * L) - 0.5 + dI(g)))
    print("total time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
