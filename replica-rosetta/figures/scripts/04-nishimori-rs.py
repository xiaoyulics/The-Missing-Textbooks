"""Chapter 4 (nishimori): the replica-symmetric formula for spiked Wigner, book normalization.

Model: A = (lam/N) x x^T + W, W ~ GOE(N), x_i iid from P0 with E x^2 = 1 (OUTLINE: BBP at lam = 1).
Scalar channel Y = sqrt(gamma) X + Z, free entropy psi(gamma) = E log int dP0(x) exp(sqrt(gamma) Z x
+ gamma x X - gamma x^2/2); RS potential F(lam, q) = psi(lam^2 q) - lam^2 q^2/4; the limit of the free
entropy phi_N = (1/N) E log Z_N is max_q F (Lelarge--Miolane, Thm 1, after snr = lam^2).
Rademacher prior: psi(gamma) = E log cosh(gamma + sqrt(gamma) Z) - gamma/2,
g(gamma) = 1 - mmse(gamma) = E tanh(gamma + sqrt(gamma) Z), fixed point q = g(lam^2 q).

Writes
  figures/data/04-nishimori-curves.dat     lam, q*, MMSE 1-q*^2, MI lam^2/4 - max F, max F, spectral MSE
  figures/data/04-nishimori-potential.dat  q, F(lam, q) for lam in {0.8, 1, 1.2, 1.5} (Rademacher) and
                                           F_rho(lam, q) - F_rho(lam, 0) for the sparse Rademacher prior
                                           (rho = 0.05) at lam in {0.84, lam_IT, 0.88}
  figures/data/04-nishimori-below.dat      lam < 1: the second-moment bound -(1/2) log(1 - lam^2) on N phi_N
                                           (proved at every N) and the lognormal heuristic -(1/4) log(1 - lam^2)
and prints every number the chapter quotes. Runtime: about 10 s.
Run from the volume root:  python figures/scripts/04-nishimori-rs.py
"""
import math
import os
import sys

import numpy as np
from scipy import integrate, optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk

OUT = os.path.join("figures", "data")
NQ = 201  # Gauss--Hermite nodes


# ---------------------------------------------------------------------------------------------
# Rademacher prior
# ---------------------------------------------------------------------------------------------
def g(gam):
    """1 - mmse(gam) = E tanh(gam + sqrt(gam) Z) for the Rademacher prior."""
    if gam <= 0:
        return 0.0
    return rk.gauss(lambda z: np.tanh(gam + math.sqrt(gam) * z), NQ)


def psi(gam):
    """Scalar free entropy of the Rademacher channel."""
    if gam <= 0:
        return 0.0
    return rk.gauss(lambda z: np.log(np.cosh(gam + math.sqrt(gam) * z)), NQ) - gam / 2


def F(lam, q):
    return psi(lam * lam * q) - lam * lam * q * q / 4


def qstar(lam):
    """The unique maximizer of F(lam, .) on [0, 1]: 0 for lam <= 1, the positive fixed point otherwise."""
    if lam <= 1.0:
        return 0.0
    return rk.root(lambda q: g(lam * lam * q) - q, 1e-12, 1.0)


def mmse_fourier(gam):
    """mmse(gam) = (1/2) int sech(pi w/2) exp(-(1+w^2) gam/2) dw (complete monotonicity)."""
    f = lambda w: 0.5 * np.exp(-(1 + w * w) * gam / 2 - math.pi * abs(w) / 2) * 2 / (1 + np.exp(-math.pi * abs(w)))
    return integrate.quad(f, -np.inf, np.inf, epsabs=1e-14, epsrel=1e-12)[0]


def mmse_sech(gam):
    """mmse(gam) = exp(-gam/2) E sech(sqrt(gam) Z)."""
    return math.exp(-gam / 2) * rk.gauss(lambda z: 1 / np.cosh(math.sqrt(gam) * z), NQ)


# ---------------------------------------------------------------------------------------------
# Sparse Rademacher prior (1 - rho) delta_0 + (rho/2)(delta_a + delta_-a), a = 1/sqrt(rho)
# ---------------------------------------------------------------------------------------------
XG, WG = np.polynomial.hermite_e.hermegauss(NQ)
WG = WG / WG.sum()


def psi_sparse(gam, rho):
    if gam <= 0:
        return 0.0
    a = 1 / math.sqrt(rho)
    xs = np.array([-a, 0.0, a])
    ps = np.array([rho / 2, 1 - rho, rho / 2])
    tot = 0.0
    for X0, pX in zip(xs, ps):
        expo = math.sqrt(gam) * XG[:, None] * xs[None, :] + gam * xs[None, :] * X0 - gam * xs[None, :] ** 2 / 2
        m = expo.max(axis=1, keepdims=True)
        val = m[:, 0] + np.log(np.sum(ps[None, :] * np.exp(expo - m), axis=1))
        tot += pX * float(WG @ val)
    return tot


def F_sparse(lam, q, rho):
    return psi_sparse(lam * lam * q, rho) - lam * lam * q * q / 4


def lam_IT_sparse(rho):
    """Smallest lam at which max_{q>0} F_rho(lam, q) reaches F_rho(lam, 0) = 0 (scan in q, then Brent)."""
    qs = np.linspace(0.02, 1.0, 197)

    def gap(lam):
        return max(F_sparse(lam, q, rho) for q in qs)

    return rk.root(gap, 0.6, 0.99, tol=1e-10)


def main():
    # 1. complete monotonicity of the Rademacher mmse: three representations agree
    print("mmse(gamma): direct 1-g, exp(-gamma/2) E sech(sqrt(gamma) Z), Fourier integral")
    worst = 0.0
    for gam in [0.05, 0.3, 1.0, 2.5, 5.0]:
        a, b, c = 1 - g(gam), mmse_sech(gam), mmse_fourier(gam)
        worst = max(worst, abs(a - b), abs(a - c))
        print(f"  gamma={gam:4.2f}: {a:.12f} {b:.12f} {c:.12f}")
    print(f"  largest discrepancy {worst:.1e}")
    # moments of mu (Euler numbers) and the series of g
    mom = [integrate.quad(lambda w: w ** k * 0.5 * np.exp(-math.pi * abs(w) / 2) * 2 / (1 + np.exp(-math.pi * abs(w))),
                          -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12)[0] for k in (0, 2, 4, 6, 8)]
    print("  moments of mu (k = 0, 2, 4, 6, 8):", " ".join(f"{m:.6f}" for m in mom))
    E = [1, 1, 5, 61, 1385]
    coefs = []
    for k in range(1, 5):
        c = sum(math.comb(k, j) * E[j] for j in range(k + 1))
        coefs.append((-1) ** (k + 1) * c / (2 ** k * math.factorial(k)))
    print("  series of g: coefficients of gamma^1..4:", " ".join(f"{c:.6f}" for c in coefs))
    for gam in [1e-3, 1e-2]:
        ser = sum(c * gam ** (k + 1) for k, c in enumerate(coefs))
        print(f"  gamma={gam:.0e}: g={g(gam):.15f}  series={ser:.15f}")
    # strict concavity: g'' = -(1/4) int (1+w^2)^2 exp(-(1+w^2)gamma/2) mu(dw) < 0, checked on a grid
    gpp = lambda gam: -integrate.quad(lambda w: 0.25 * (1 + w * w) ** 2 * np.exp(-(1 + w * w) * gam / 2) * 0.5 *
                                      np.exp(-math.pi * abs(w) / 2) * 2 / (1 + np.exp(-math.pi * abs(w))),
                                      -np.inf, np.inf, epsabs=1e-16, epsrel=1e-12)[0]
    print(f"  g''(0) = {gpp(0.0):.6f};  max of g'' over gamma in [0, 40] (grid of 81): "
          f"{max(gpp(x) for x in np.linspace(0, 40, 81)):.2e}")

    # 2. the Rademacher fixed point, MMSE, mutual information
    print("\nRademacher prior (Z2 synchronization), book normalization:")
    print("  lam     q*        MMSE=1-q*^2  MI/N=lam^2/4-maxF  maxF       spectral 1-(1-1/lam^2)^2")
    for lam in [0.5, 1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0]:
        q = qstar(lam)
        spec = 1 - max(1 - 1 / lam ** 2, 0) ** 2
        qs = np.linspace(0, 1, 1001)
        i = int(np.argmax([F(lam, x) for x in qs]))
        print(f"  {lam:4.2f}  {q:.6f}   {1 - q * q:.6f}    {lam * lam / 4 - F(lam, q):.6f}          "
              f"{F(lam, q):.6f}   {spec:.6f}   (grid argmax {qs[i]:.3f})")
    for eps in [1e-3, 1e-2]:
        q = qstar(1 + eps)
        print(f"  near threshold: lam = 1 + {eps:.0e}: q* = {q:.8f}, 2 eps - eps^2/3 = {2 * eps - eps * eps / 3:.8f}")

    rows = []
    for lam in np.concatenate([np.linspace(0.0, 1.0, 21), np.linspace(1.0025, 1.2, 40), np.linspace(1.21, 3.0, 90)]):
        q = qstar(lam)
        spec = 1 - max(1 - 1 / lam ** 2, 0) ** 2 if lam > 0 else 1.0
        rows.append((float(lam), q, 1 - q * q, lam * lam / 4 - F(lam, q), F(lam, q), spec))
    rk.write_dat(os.path.join(OUT, "04-nishimori-curves.dat"),
                 ["lam", "q", "mmse", "mi", "phi", "spec"], rows)
    rows = [(float(l), -0.5 * math.log(1 - l * l), -0.25 * math.log(1 - l * l)) for l in np.linspace(0.0, 0.95, 39)]
    rk.write_dat(os.path.join(OUT, "04-nishimori-below.dat"), ["lam", "bound", "heur"], rows)
    print("\nBelow the threshold: N phi_N <= -(1/2) log(1 - lam^2) (second moment); lognormal heuristic -(1/4) log(1 - lam^2):")
    for l in [0.25, 0.5, 0.75, 0.9]:
        print(f"  lam = {l}: bound {-0.5 * math.log(1 - l * l):.4f}, heuristic {-0.25 * math.log(1 - l * l):.4f}")

    # 3. potentials
    rho = 0.05
    lit = lam_IT_sparse(rho)
    print(f"\nSparse Rademacher, rho = {rho}: lam_IT = {lit:.5f}  (digest 02: 0.86447)")
    lam_r = [0.8, 1.0, 1.2, 1.5]
    lam_s = [0.84, lit, 0.88]
    qs = np.linspace(0.0, 1.0, 201)
    rows = []
    for q in qs:
        row = [float(q)] + [F(l, q) for l in lam_r] + [F_sparse(l, q, rho) - F_sparse(l, 0.0, rho) for l in lam_s]
        rows.append(row)
    rk.write_dat(os.path.join(OUT, "04-nishimori-potential.dat"),
                 ["q", "r08", "r10", "r12", "r15", "s084", "sIT", "s088"], rows)
    for l in lam_s:
        vals = np.array([F_sparse(l, q, rho) - F_sparse(l, 0.0, rho) for q in qs])
        loc = [i for i in range(1, len(qs) - 1) if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]]
        print(f"  lam = {l:.4f}: interior local maxima of F_rho - F_rho(0):",
              ", ".join(f"q = {qs[i]:.3f} (value {vals[i]:+.5f})" for i in loc) or "none")
    for l in lam_r:
        vals = [F(l, q) for q in qs]
        j = int(np.argmax(vals))
        print(f"  Rademacher lam = {l}: max of F at q = {qs[j]:.3f}, value {vals[j]:.5f}; F(lam, 1) = {vals[-1]:.5f}")


if __name__ == "__main__":
    main()
