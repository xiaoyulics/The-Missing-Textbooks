"""Chapter 12 (kernels): the theorem of Section "The proof", measured.

Gaussian features x = Sigma^{1/2} g with three distinct eigenvalues eta = (1, 0.3, 0.05) of
multiplicities p_j = pi_j n, pi = (0.3, 0.6, 1.2) (so p/n = 2.1); random target uniform on the product
of the block spheres with block norms b_j^2 = (0.5, 1, 2) (target power eta_j b_j^2 = 0.5, 0.3, 0.1);
noise sigma^2 = 0.25; ridge lambda = 0.05 (per-sample normalization).
For each design X the conditional risk E[R | X] (over target and noise) is computed exactly:
  E[R | X] = lambda^2 sum_j (b_j^2/p_j) tr(P_j G Sigma G) + (sigma^2/n) tr(Sigma G Sigma_hat G),
  G = (Sigma_hat + lambda I)^{-1}, and compared with the limit R* of the theorem.
Also printed: the block traces (1/n) tr(P_j G) against their deterministic equivalents, and the
training error against (lambda/kappa)^2 (R* + sigma^2) (Exercise ex:kernels:gcv).
Output: figures/data/12-kernels-theorem.dat. Runtime: 10 s idle, up to 50 s loaded. Seed 1205.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread: much faster on a shared machine

import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
ETA = np.array([1.0, 0.3, 0.05])
PI_ = np.array([0.3, 0.6, 1.2])
B2 = np.array([0.5, 1.0, 2.0])
SIG2, LAM = 0.25, 0.05


def theory(pi_=PI_, lam=LAM):
    """kappa from lambda = kappa (1 - D1(kappa)); R* = [sum kappa^2 eta b^2/(eta+kappa)^2 + sigma^2 D2]/(1 - D2);
    also delta = kappa/lambda - 1 and the block-trace limits pi_j/(lambda + eta_j/(1+delta))."""
    D1 = lambda k: float(np.sum(pi_ * ETA / (ETA + k)))
    k0 = 0.0 if D1(1e-300) < 1 else rk.root(lambda k: D1(k) - 1, 1e-300, 1e6)
    kap = rk.root(lambda k: k * (1 - D1(k)) - lam, k0 * (1 + 1e-12) + 1e-300, 1e6)
    D2 = float(np.sum(pi_ * ETA ** 2 / (ETA + kap) ** 2))
    bias = float(np.sum(kap ** 2 * ETA * B2 / (ETA + kap) ** 2)) / (1 - D2)
    var = SIG2 * D2 / (1 - D2)
    delta = kap / lam - 1
    tau = pi_ / (lam + ETA / (1 + delta))
    return bias + var, bias, var, kap, D2, delta, tau


def one_design(n, gen):
    p_j = (PI_ * n).round().astype(int)
    p = int(p_j.sum())
    eta_vec = np.repeat(ETA, p_j)
    X = gen.standard_normal((n, p)) * np.sqrt(eta_vec)
    S = X.T @ X / n
    G = np.linalg.inv(S + LAM * np.eye(p))
    GSG = (G * eta_vec[None, :]) @ G                       # G Sigma G
    blocks = np.repeat(np.arange(3), p_j)
    diag_GSG = np.diag(GSG)
    bias = LAM ** 2 * sum(B2[j] / p_j[j] * float(np.sum(diag_GSG[blocks == j])) for j in range(3))
    var = SIG2 / n * float(np.sum((eta_vec[:, None] * G) * (S @ G).T))   # tr(Sigma G S G)
    tau = np.array([float(np.sum(np.diag(G)[blocks == j])) / n for j in range(3)])
    # training error, averaged over target and noise: (lambda^2/n) [tr(T^2 X B X^T) + sigma^2 tr(T^2)]
    Kn = X @ X.T / n
    T = np.linalg.inv(Kn + LAM * np.eye(n))
    Bvec = np.repeat(B2 / p_j, p_j)
    XBXt = (X * Bvec[None, :]) @ X.T
    T2 = T @ T
    train = LAM ** 2 / n * (float(np.sum(T2 * XBXt.T)) + SIG2 * float(np.trace(T2)))
    return bias + var, bias, var, tau, train


def main():
    t0 = time.time()
    gen = rk.rng(1205)
    Rs, bs, vs, kap, D2, delta, tau_lim = theory()
    print("theorem: kappa=%.6f delta=%.6f D2=%.6f  R*=%.6f (bias %.6f, variance %.6f)" % (kap, delta, D2, Rs, bs, vs))
    print("block-trace limits pi_j/(lambda + eta_j/(1+delta)): %s" % np.array2string(tau_lim, precision=6))
    print("training-error prediction (lambda/kappa)^2 (R* + sigma^2) = %.6f" % ((LAM / kap) ** 2 * (Rs + SIG2)))
    rows = []
    print("    n     p | E[R|X] mean (95%)    | bias    variance | max block-trace error | training error")
    for n, reps in [(50, 60), (100, 40), (200, 20), (400, 10), (800, 5)]:
        out = [one_design(n, gen) for _ in range(reps)]
        R = [o[0] for o in out]
        m, h = rk.mean_ci(R)
        bm = float(np.mean([o[1] for o in out]))
        vm = float(np.mean([o[2] for o in out]))
        terr = float(np.mean([np.max(np.abs(o[3] - tau_lim)) for o in out]))
        tr = float(np.mean([o[4] for o in out]))
        rows.append([n, m, h, Rs, bm, vm, terr, tr])
        print("%5d %5d | %.5f (%.5f)    | %.5f %.5f | %.2e              | %.5f" % (
            n, int(round(2.1 * n)), m, h, bm, vm, terr, tr))
    rk.write_dat(os.path.join(DATA, "12-kernels-theorem.dat"),
                 ["n", "risk", "riskci", "limit", "bias", "var", "traceerr", "train"], rows)
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
