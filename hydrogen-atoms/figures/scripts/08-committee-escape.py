"""Plateau length versus dimension from random initialization (Chapter 8, fig:committee:escape).

Soft committee machine, m = mstar = 2, erf units, orthonormal teacher, online SGD with step eta/d,
eta = 0.5. Random start: w_j ~ N(0, (q0/d) I_d) with q0 = 0.01, independently for each run.
Escape time: first t = steps/d (checked every 20 steps) at which the generalization error falls below
half the plateau value of the ODE, 0.0147/2.
Simulation: the exact Markov chain of the order parameters, for d = 100, 200, ..., 3200, 40 runs each,
seed 9 (all runs advanced in parallel).
Theory: slope 1/(2 lambda_u(eta)) in log d, where lambda_u(eta) is the specialization eigenvalue of the
full ODE (with its eta^2 term) at the symmetric fixed point, computed here by Newton's method and
finite differences; the intercept is fitted to the medians by least squares.
Output: figures/data/08-committee-escape.dat (d logd median q25 q75 theory)
Runtime: about 80 seconds.
"""
import os
import importlib.util
import numpy as np
from scipy.optimize import fsolve

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("plateau", os.path.join(HERE, "08-committee-plateau.py"))
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
ETA, Q0, SEED, N = 0.5, 0.01, 9, 40
DS = [100, 200, 400, 800, 1600, 3200]


def lambda_u(eta):
    def g0(y):
        R, Q = y
        f = P.rhs(0, [Q, Q, Q, R, R, R, R], eta)
        return [f[3], f[0]]
    R, Q = fsolve(g0, [1 / np.sqrt(6), 1 / 3])
    z = np.array([Q, Q, Q, R, R, R, R]); eS = np.array([0, 0, 0, 1, -1, -1, 1.0]); h = 1e-6
    dz = (np.array(P.rhs(0, z + h * eS, eta)) - np.array(P.rhs(0, z - h * eS, eta))) / (2 * h)
    return dz @ eS / (eS @ eS), P.egv(np.array([[Q, Q], [Q, Q]]), np.full((2, 2), R))


def main():
    lam, eps_star = lambda_u(ETA)
    thr = eps_star / 2
    rng = np.random.default_rng(SEED)
    dvec = np.repeat(DS, N); n = len(dvec)
    R = np.zeros((n, 2, 2)); G = np.zeros((n, 2, 2))
    for i, d in enumerate(dvec):
        W = rng.standard_normal((2, d)) * np.sqrt(Q0 / d)
        R[i] = W[:, :2]; G[i] = W[:, 2:] @ W[:, 2:].T
    tesc = np.full(n, np.nan)
    smax = int(max(DS) * 330)
    # advance all runs in parallel and stop when every run has escaped
    from scipy.special import erf
    k = ETA / dvec; s = 0; block = 2048
    Rr, Gr = R, G
    while s < smax and np.isnan(tesc).any():
        NU = rng.standard_normal((block, n, 2)); XI = rng.standard_normal((block, n, 2))
        CH = rng.chisquare(dvec - 4, (block, n))
        for b in range(block):
            if s % 20 == 0:
                e = P.egv(Gr + np.einsum('njk,nlk->njl', Rr, Rr), Rr)
                hit = np.isnan(tesc) & (e < thr); tesc[hit] = s / dvec[hit]
            nu, xi = NU[b], XI[b]
            a = np.sqrt(Gr[:, 0, 0]); bb = Gr[:, 0, 1] / a; c = np.sqrt(np.maximum(Gr[:, 1, 1] - bb * bb, 0.0))
            r0 = a * xi[:, 0]; r1 = bb * xi[:, 0] + c * xi[:, 1]
            l0 = Rr[:, 0, 0] * nu[:, 0] + Rr[:, 0, 1] * nu[:, 1] + r0
            l1 = Rr[:, 1, 0] * nu[:, 0] + Rr[:, 1, 1] * nu[:, 1] + r1
            gg = erf(np.stack([nu[:, 0], nu[:, 1], l0, l1]) * P.R2)
            err = gg[0] + gg[1] - gg[2] - gg[3]
            d0 = P.SQ * np.exp(-0.5 * l0 * l0) * err * k; d1 = P.SQ * np.exp(-0.5 * l1 * l1) * err * k
            x2 = xi[:, 0] ** 2 + xi[:, 1] ** 2 + CH[b]
            Rr[:, 0, :] += d0[:, None] * nu; Rr[:, 1, :] += d1[:, None] * nu
            Gr[:, 0, 0] += 2 * d0 * r0 + x2 * d0 * d0
            Gr[:, 1, 1] += 2 * d1 * r1 + x2 * d1 * d1
            g01 = d0 * r1 + d1 * r0 + x2 * d0 * d1
            Gr[:, 0, 1] += g01; Gr[:, 1, 0] += g01
            s += 1
    med = np.array([np.nanmedian(tesc[i * N:(i + 1) * N]) for i in range(len(DS))])
    q25 = np.array([np.nanpercentile(tesc[i * N:(i + 1) * N], 25) for i in range(len(DS))])
    q75 = np.array([np.nanpercentile(tesc[i * N:(i + 1) * N], 75) for i in range(len(DS))])
    slope = 1 / (2 * lam); L = np.log(DS)
    icpt = np.mean(med - slope * L)
    fit = np.polyfit(L, med, 1)
    print(f"lambda_u(eta)={lam:.5f}  plateau eps={eps_star:.5f}  predicted slope={slope:.2f}  "
          f"fitted slope={fit[0]:.2f}  not escaped={np.isnan(tesc).sum()}")
    with open(os.path.join(HERE, "..", "data", "08-committee-escape.dat"), "w") as f:
        f.write("d logd median q25 q75 theory\n")
        for i, d in enumerate(DS):
            f.write(f"{d} {L[i]:.4f} {med[i]:.2f} {q25[i]:.2f} {q75[i]:.2f} {icpt + slope * L[i]:.2f}\n")


if __name__ == "__main__":
    main()
