"""Convergence rate of a single erf unit and the stability threshold (Chapter 8, fig:committee:single).

Teacher g(<w_star, x>), student g(<w, x>), g(z) = erf(z/sqrt 2), |w_star| = 1, online SGD with step eta/d.
Theory (prop:committee:single): near w = w_star the order parameters u = R - 1 and v = Q - R^2 decay at
rates eta c1 and 2 eta c2 - eta^2 c3, with c1 = 2/(3 sqrt3 pi), c2 = 2/(sqrt3 pi), c3 = 4/(sqrt5 pi^2), so
D = |w - w_star|^2 = u^2 + v decays at rate min(2 eta c1, 2 eta c2 - eta^2 c3), which vanishes at
eta_max = pi sqrt(5/3).
Simulation: the exact Markov chain of (R, v) for d = 1000 (it has the law of d-dimensional SGD), from
R = 0, v = 0.25; 8 runs per eta, all eta advanced in parallel, seed 10. The measured rate is minus the
slope of the median of log D over the stretch where 1e-40 < D < 1e-6 (or the last third of the run if D
never falls below 1e-6). The difference g(nu) - g(lambda) is computed by quadrature when
lambda is close to nu, to avoid cancellation.
Output: figures/data/08-committee-single-theory.dat (eta rate), 08-committee-single-sim.dat (eta rate)
Runtime: about 40 seconds.
"""
import os
import numpy as np
from scipy.special import erf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
D, RUNS, SEED, TMAX = 1000, 8, 10, 400.0
C1, C2, C3 = 2 / (3 * np.sqrt(3) * np.pi), 2 / (np.sqrt(3) * np.pi), 4 / (np.sqrt(5) * np.pi ** 2)
ETAS = np.r_[np.arange(0.25, 4.01, 0.25), 4.25, 4.5]
SQ, R2 = np.sqrt(2 / np.pi), np.sqrt(0.5)


GLX, GLW = np.polynomial.legendre.leggauss(6)


def gdiff(nu, dlt):
    """g(nu) - g(nu + dlt) without cancellation: Gauss-Legendre quadrature of -g' for small |dlt|."""
    quad = -0.5 * dlt * sum(w * SQ * np.exp(-0.5 * (nu + 0.5 * dlt * (1 + x)) ** 2) for x, w in zip(GLX, GLW))
    return np.where(np.abs(dlt) < 0.05, quad, erf(nu * R2) - erf((nu + dlt) * R2))


def theory_rate(eta):
    return np.minimum(2 * eta * C1, 2 * eta * C2 - eta ** 2 * C3)


def main():
    rng = np.random.default_rng(SEED)
    eta = np.repeat(ETAS, RUNS); n = len(eta); k = eta / D
    u = -np.ones(n); v = 0.25 * np.ones(n)          # u = R - 1, v = |w_perp|^2
    every = D // 2; steps = int(TMAX * D); logs = []; block = 4096; s = 0
    while s <= steps:
        NU = rng.standard_normal((block, n)); XI = rng.standard_normal((block, n))
        CH = rng.chisquare(D - 2, (block, n))
        for b in range(block):
            if s % every == 0:
                logs.append(np.log(u * u + v))
            if s == steps:
                break
            nu, xi = NU[b], XI[b]
            sv = np.sqrt(v); dlt = u * nu + sv * xi; lam = nu + dlt
            dl = SQ * np.exp(-0.5 * lam * lam) * gdiff(nu, dlt) * k
            u = u + dl * nu
            v = v + 2 * dl * sv * xi + dl * dl * (xi * xi + CH[b])
            s += 1
        if s == steps:
            break
    L = np.array(logs); t = np.arange(L.shape[0]) * every / D
    with open(os.path.join(DATA, "08-committee-single-sim.dat"), "w") as f:
        f.write("eta rate\n")
        for i, e in enumerate(ETAS):
            m = np.median(L[:, i * RUNS:(i + 1) * RUNS], axis=1)
            sel = (m < np.log(1e-6)) & (m > np.log(1e-40))
            if sel.sum() < 5:
                sel = t > 2 * TMAX / 3
            rate = -np.polyfit(t[sel], m[sel], 1)[0]
            print(f"eta={e:.2f}  measured={rate:.4f}  theory={theory_rate(e):.4f}")
            f.write(f"{e:.2f} {rate:.5f}\n")
    with open(os.path.join(DATA, "08-committee-single-theory.dat"), "w") as f:
        f.write("eta rate\n")
        for e in np.linspace(0, np.pi * np.sqrt(5 / 3), 121):
            f.write(f"{e:.4f} {max(theory_rate(e), 0.0):.5f}\n")


if __name__ == "__main__":
    main()
