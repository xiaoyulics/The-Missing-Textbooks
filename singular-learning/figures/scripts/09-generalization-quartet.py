"""Chapter 9: the limit posterior of a one-coefficient model, Tweedie's formula, and the singular
fluctuation nu(beta).

For a coefficient density |c - c0|^(2 lambda - 1) near the truth, the posterior of t = sqrt(S_xx)(c - c0)
tends to pi_Z(t) ~ exp(-beta (t - Z)^2 / 2) w(t) with w(t) = |t|^alpha (alpha = 2 lambda - 1),
Z ~ N(0, 1) (Section 9.3). With M(Z) = E_pi[t], v(Z) = Var_pi(t):
  virial identity   beta E_pi[t (t - Z)] = 1 + alpha = 2 lambda   for every Z,
  Stein's identity  E_Z[Z M(Z)] = beta E_Z[v(Z)],
  nu(beta) = (beta / 2) E_Z[v(Z)], and the four limits
  E B_g = (lambda - nu)/beta + nu,  E B_t = (lambda - nu)/beta - nu,  E G_g = lambda/beta + nu,  E G_t = lambda/beta - nu,
with B_g = E M^2 / 2, B_t = E M^2 / 2 - E[Z M], G_g = E[M^2 + v] / 2, G_t = G_g - E[Z M].
Moments by adaptive quadrature with the algebraic weight at t = 0; E_Z by Gauss-Hermite (120 nodes).
Also: the variant y = a (b . x), b uniform on the unit disk (a two-coefficient model whose prior weight
is |u|^(-1) in R^2), checked against the digest's nu(1) = 0.6997.

Writes figures/data/09-generalization-tweedie.dat (the shrinkage factor M(Z)/Z, Z >= 0, for
lambda = 1/6, 1/4, 1/2, 1) and prints the numbers of Table 9.2, of the limit law of n (G_n - S), of the
exercises on temperature, asymmetric weights and the variant, and the error of the cubic cusp at a truth
near the singularity.
Run from the volume root: python figures/scripts/09-generalization-quartet.py   (30-60 s)
"""
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy import integrate, interpolate, optimize, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

# quad warns about roundoff only for the far half-line at |Z| near 19, where the integral is about 1e-83
# against a total of order 10; those contributions are negligible
warnings.simplefilter("ignore", integrate.IntegrationWarning)


def half_integral(Z, alpha, beta, k, side):
    """int_0^inf s^alpha s^k exp(-beta (side s - Z)^2 / 2) ds."""
    T = abs(Z) + 40.0 / math.sqrt(beta)
    f = lambda s: s ** k * math.exp(-0.5 * beta * (side * s - Z) ** 2)
    cuts = [0.0, T] if not (0.0 < abs(Z) < T) else [0.0, abs(Z), T]
    total = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        if a == 0.0:
            total += integrate.quad(f, a, b, weight="alg", wvar=(alpha, 0.0), limit=400, epsabs=0, epsrel=1e-12)[0]
        else:
            total += integrate.quad(lambda s: f(s) * s ** alpha, a, b, limit=400, epsabs=0, epsrel=1e-12)[0]
    return total


def posterior(Z, alpha, beta, lp=1.0, lm=1.0):
    """Mean M and variance v of pi_Z for w(t) = |t|^alpha (lp for t > 0, lm for t < 0)."""
    m = [lp * half_integral(Z, alpha, beta, k, 1.0) + lm * (-1) ** k * half_integral(Z, alpha, beta, k, -1.0)
         for k in range(3)]
    M = m[1] / m[0]
    return M, m[2] / m[0] - M * M


ZQ, WQ = special.roots_hermitenorm(120)
WQ = WQ / WQ.sum()


def quartet(lam, beta, lp=1.0, lm=1.0):
    alpha = 2 * lam - 1
    Ms, vs = np.array([posterior(z, alpha, beta, lp, lm) for z in ZQ]).T
    EM2, Ev, EZM = float(WQ @ Ms ** 2), float(WQ @ vs), float(WQ @ (ZQ * Ms))
    vir = max(abs(beta * (M * M + v - z * M) - 2 * lam) for z, M, v in zip(ZQ, Ms, vs))
    nu = beta * Ev / 2
    return dict(EM2=EM2, Ev=Ev, EZM=EZM, nu=nu, stein=EZM - beta * Ev, virial=vir,
                Bg=EM2 / 2, Bt=EM2 / 2 - EZM, Gg=(EM2 + Ev) / 2, Gt=(EM2 + Ev) / 2 - EZM)


def variant_moments(r):
    """|E u| and tr Cov u for the density ~ exp(-|u - xi|^2/2)/|u| on R^2, |xi| = r (polar coordinates)."""
    T = r + 40.0
    g = lambda s: math.exp(-0.5 * (s - r) ** 2)
    f0 = lambda s: g(s) * special.ive(0, s * r)
    f1 = lambda s: s * g(s) * special.ive(1, s * r)
    f2 = lambda s: s * s * g(s) * special.ive(0, s * r)
    cuts = [0.0, r, T] if r > 0 else [0.0, T]
    q = lambda f: sum(integrate.quad(f, a, b, limit=400, epsabs=0, epsrel=1e-12)[0] for a, b in zip(cuts[:-1], cuts[1:]))
    z0, z1, z2 = q(f0), q(f1), q(f2)
    return z1 / z0, z2 / z0 - (z1 / z0) ** 2


def variant(beta, nodes=160):
    """nu(beta) and E[n B_g] for the variant; |xi|^2/2 ~ Exp(1) under xi ~ N(0, I_2) (Gauss-Laguerre),
    and by homogeneity Cov_beta(xi) = Cov_1(sqrt(beta) xi)/beta, M_beta(xi) = M_1(sqrt(beta) xi)/sqrt(beta)."""
    s, w = special.roots_laguerre(nodes)
    vals = np.array([variant_moments(math.sqrt(beta) * math.sqrt(2 * si)) for si in s])
    return 0.5 * float(w @ vals[:, 1]), float(w @ vals[:, 0] ** 2) / (2 * beta)


def main():
    t0 = time.time()
    lams = [1 / 6, 1 / 4, 1 / 2, 3 / 4, 1.0]
    print("beta = 1: E M^2 against 2 lambda; residuals of the virial and Stein identities; nu and the four limits")
    res1 = {}
    for lam in lams:
        r = quartet(lam, 1.0)
        res1[lam] = r
        print(f"  lambda = {lam:.4f}: E M^2 = {r['EM2']:.10f} (2 lambda = {2 * lam:.10f}), virial {r['virial']:.1e}, "
              f"Stein {r['stein']:.1e} | nu = {r['nu']:.4f}  B_g = {r['Bg']:.4f}  B_t = {r['Bt']:.4f}  "
              f"G_g = {r['Gg']:.4f}  G_t = {r['Gt']:.4f}")
    print("asymmetric weights (lambda = 1/4): E M^2 is still 2 lambda, nu changes")
    for lp, lm in [(1.0, 0.2), (1.0, 3.0)]:
        r = quartet(0.25, 1.0, lp, lm)
        print(f"  l+ = {lp}, l- = {lm}: E M^2 = {r['EM2']:.10f}, nu = {r['nu']:.4f}")
    print("temperatures: nu(beta) and E[n B_g] = (lambda - nu)/beta + nu")
    for lam in [1 / 6, 1 / 4, 1 / 2, 3 / 4]:
        line = []
        for beta in [0.5, 1.0, 2.0]:
            r = quartet(lam, beta)
            chk = (lam - r["nu"]) / beta + r["nu"]
            line.append(f"beta={beta}: nu={r['nu']:.4f} B_g={r['Bg']:.4f} (formula {chk:.4f})")
        print(f"  lambda = {lam:.4f}: " + "; ".join(line))
    print("limit law of n (G_n - S), that is of M(Z)^2 / 2 at beta = 1: mean and standard deviation")
    for lam in [1 / 6, 1 / 4, 1 / 2]:
        X = 0.5 * np.array([posterior(z, 2 * lam - 1, 1.0)[0] for z in ZQ]) ** 2
        print(f"  lambda = {lam:.4f}: mean {float(WQ @ X):.4f}, sd {math.sqrt(float(WQ @ X ** 2) - float(WQ @ X) ** 2):.4f}")
    zstar = optimize.brentq(lambda z: posterior(z, -2.0 / 3.0, 1.0)[0] - 1.0, 1.0, 3.0)
    print(f"  cubic cusp: M(z*) = 1 at z* = {zstar:.4f}; P(n (G_n - S) > 1/2) -> 2 P(Z > z*) = {special.erfc(zstar / math.sqrt(2)):.4f}"
          f" (regular line: {special.erfc(1 / math.sqrt(2)):.4f})")
    print("Tweedie for the cubic cusp (lambda = 1/6): M(Z) at a few Z")
    for z in [0.0, 0.5, 1.0, 2.5, 4.0]:
        M, v = posterior(z, -2.0 / 3.0, 1.0)
        print(f"  Z = {z}: M = {M:.4f}, v = {v:.4f}, Z - 2/(3Z) = {z - 2 / (3 * z) if z > 0 else float('nan'):.4f}")
    print("variant y = a (b . x): nu(beta), E[n B_g]; (lambda - nu)/beta + nu")
    for beta in [0.5, 1.0, 2.0]:
        nu, Bg = variant(beta)
        print(f"  beta = {beta}: nu = {nu:.4f}, E[n B_g] = {Bg:.4f}, formula {(0.5 - nu) / beta + nu:.4f}")
    # a truth near the singularity (Section 9.7): truth c0 = delta, observation t0 + Z with t0 = sqrt(n) delta;
    # the limit posterior keeps the weight |t|^(-2/3), and the error is (1/2) E (M(t0 + Z) - t0)^2
    shifted = lambda t0: 0.5 * float(WQ @ np.array([(posterior(t0 + z, -2.0 / 3.0, 1.0)[0] - t0) ** 2 for z in ZQ]))
    print("cubic cusp, truth at sqrt(n) delta = t0: n (E[G_n] - S) ->",
          ", ".join(f"{t0:g}: {shifted(t0):.3f}" for t0 in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]))
    best = optimize.minimize_scalar(lambda s: -shifted(s), bounds=(2.0, 4.0), method="bounded", options={"xatol": 0.01})
    print(f"  largest value {-best.fun:.3f} at t0 = {best.x:.2f} (regular value 0.5)")
    t1 = time.time()

    # shrinkage factors M(Z)/Z for the figure of Section 9.2 (M is odd; at Z = 0 the factor is M'(0) = v(0) = 2 lambda)
    zs = np.linspace(0.0, 5.0, 101)
    curves = {}
    for lam in [1 / 6, 1 / 4, 1.0]:
        vals = []
        for z in zs:
            M, v = posterior(z, 2 * lam - 1, 1.0)
            vals.append(v if z == 0.0 else M / z)
        curves[lam] = vals
    sk.write_dat("figures/data/09-generalization-tweedie.dat", ["Z", "S16", "S14", "S12", "S1"],
                 [(z, curves[1 / 6][i], curves[1 / 4][i], 1.0, curves[1.0][i]) for i, z in enumerate(zs)])
    print("shrinkage factor M(Z)/Z at Z = 0, 1, 2, 5 (lambda = 1/6, 1/4, 1):",
          "; ".join(f"{lam:.4f}: " + ", ".join(f"{curves[lam][i]:.3f}" for i in (0, 20, 40, 100)) for lam in curves))

    # E[n B_g](beta) = (1/(2 beta)) E[M_1(sqrt(beta) Z)^2]: M_1 on a grid, cubic spline in between (symmetric weights)
    zg = np.linspace(0.0, 60.0, 601)
    betas = np.logspace(-1, 1, 21)
    cols = []
    for lam in [1 / 6, 1 / 4, 1 / 2, 3 / 4]:
        Mg = np.array([posterior(z, 2 * lam - 1, 1.0)[0] for z in zg])
        spline = interpolate.CubicSpline(zg, Mg)
        col = []
        for beta in betas:
            arg = np.abs(math.sqrt(beta) * ZQ)
            col.append(float(WQ @ spline(arg) ** 2) / (2 * beta))
        cols.append(col)
    print("E[n B_g] against beta (lambda = 1/6, 1/4, 1/2, 3/4), and the beta -> 0 limit 2 lambda^2:")
    for i, b in enumerate(betas):
        if i % 5 == 0:
            print(f"  beta = {b:.3f}: " + "  ".join(f"{c[i]:.4f}" for c in cols))
    print("  2 lambda^2:", "  ".join(f"{2 * l * l:.4f}" for l in [1 / 6, 1 / 4, 1 / 2, 3 / 4]))
    print(f"runtime: {t1 - t0:.1f} s (tables), {time.time() - t1:.1f} s (curves)")


if __name__ == "__main__":
    main()
