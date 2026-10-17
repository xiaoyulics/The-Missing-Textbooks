#!/usr/bin/env python3
"""Figure 7.1 of "Optimizers in the Wild" (Chapter 7, Kronecker preconditioners).

The Kronecker quadratic  L(W) = 1/2 tr((W - W*)^T A (W - W*) C),  Hessian H = C kron A on vec(W),
with m = 16, n = 24, A and C with geometrically spaced eigenvalues (condition numbers 30 each,
so kappa(H) = 900) and Haar-random eigenvectors.

Shampoo's factors are estimated from N = 1000 sampled gradients with stationary Kronecker
statistics Gamma = E[g g^T] proportional to H^q:
  q = 1: gradient noise A^{1/2} Z C^{1/2} (covariance H; a Gaussian linear layer, Fisher = Hessian),
  q = 2: gradients A E C at isotropic random errors E (covariance H^2).
The step  W <- W - eta L^{-s} grad R^{-s}  is then a fixed linear iteration.

Output 1, figures/data/07-kronecker-ladder.dat: relative loss L_t / L_0 for t = 0..250 from one
random start, exact theory (exact factors, eigenbasis formula) and simulation (matrix iteration
with the estimated factors), for
  gd     gradient descent at its best constant step,
  shq1   Shampoo (s = 1/4) with q = 1 factors at the best step for those factors,
  shq2   Shampoo (s = 1/4) with q = 2 factors at the best step for those factors,
  soap   AdaGrad in the eigenbasis of the factors (SOAP without momentum, accumulated second
         moments), step 0.3; theory: exact eigenbasis (a scalar recursion per mode, curvature-free);
         simulation: eigenbasis of the estimated q = 1 factors,
  ada    AdaGrad in the original coordinates, step 1.0 (simulation only).
Output 2, figures/data/07-kronecker-power.dat: the effective exponent log kappa_eff / log kappa(H)
against the power s, theory |1 - q s|, and "measured" from the asymptotic rate rho of a run of the
matrix iteration with estimated factors at the best step, kappa = (1 + rho)/(1 - rho).

Options for the exercise "The V and its floor" (they print the exponents and do not touch the figure's data files):
  --kA, --kC   condition numbers of A and C (default 30 and 30),
  --N          number of sampled gradients behind the factors (default 1000).

numpy only; deterministic seed; runs in a few seconds on one core.
"""
import argparse
import os
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--kA", type=float, default=30.0, help="condition number of A")
ap.add_argument("--kC", type=float, default=30.0, help="condition number of C")
ap.add_argument("--N", type=int, default=1000, help="number of sampled gradients")
args = ap.parse_args()
FIGURE = (args.kA, args.kC, args.N) == (30.0, 30.0, 1000)   # the published figure's parameters

rng = np.random.default_rng(2026)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def psd_pow(M, p):
    w, Q = np.linalg.eigh((M + M.T) / 2)
    return (Q * np.maximum(w, 1e-300) ** p) @ Q.T


def haar(k):
    Q, R = np.linalg.qr(rng.standard_normal((k, k)))
    return Q * np.sign(np.diag(R))


m, n, kA, kC = 16, 24, args.kA, args.kC
alpha = np.geomspace(1.0, 1.0 / kA, m)
gamma = np.geomspace(1.0, 1.0 / kC, n)
UA, UC = haar(m), haar(n)
A = (UA * alpha) @ UA.T
C = (UC * gamma) @ UC.T
H = np.outer(alpha, gamma)              # curvature of mode (i, j) in the eigenbasis
kH = kA * kC
E0 = rng.standard_normal((m, n))        # initial error W_0 - W_*
Et0 = UA.T @ E0 @ UC
loss0 = 0.5 * np.sum(H * Et0 ** 2)
Ah, Ch = psd_pow(A, 0.5), psd_pow(C, 0.5)


def loss(E):
    return 0.5 * np.sum(E * (A @ E @ C))


def sample_factors(q, N):
    L, R = np.zeros((m, m)), np.zeros((n, n))
    for _ in range(N):
        Z = rng.standard_normal((m, n))
        G = Ah @ Z @ Ch if q == 1 else A @ Z @ C
        L += G @ G.T
        R += G.T @ G
    return L / N, R / N


def eff_range(L, R, s):
    a = np.linalg.eigvalsh(psd_pow(L, -s / 2) @ A @ psd_pow(L, -s / 2))
    c = np.linalg.eigvalsh(psd_pow(R, -s / 2) @ C @ psd_pow(R, -s / 2))
    return a.min() * c.min(), a.max() * c.max()


def theory_frozen(q, s, T):
    mu = np.outer(alpha ** (1 - q * s), gamma ** (1 - q * s))
    eta = 2 / (mu.min() + mu.max())
    t = np.arange(T + 1)[:, None, None]
    return 0.5 * np.sum(H * (1 - eta * mu) ** (2 * t) * Et0 ** 2, axis=(1, 2)) / loss0


def sim_frozen(L, R, s, T):
    lo, hi = eff_range(L, R, s)
    eta = 2 / (lo + hi)
    PL, PR = psd_pow(L, -s), psd_pow(R, -s)
    E = E0.copy()
    out = [1.0]
    for _ in range(T):
        E = E - eta * PL @ (A @ E @ C) @ PR
        out.append(loss(E) / loss0)
    return np.array(out)


def soap_theory(eta, T):
    e, S, out = Et0.copy(), np.zeros_like(Et0), [1.0]
    for _ in range(T):
        S += (H * e) ** 2
        e = e - eta * H * e / np.sqrt(S)
        out.append(0.5 * np.sum(H * e ** 2) / loss0)
    return np.array(out)


def soap_sim(L, R, eta, T):
    _, QL = np.linalg.eigh(L)
    _, QR = np.linalg.eigh(R)
    E, V, out = E0.copy(), np.zeros((m, n)), [1.0]
    for _ in range(T):
        Gr = QL.T @ (A @ E @ C) @ QR
        V += Gr ** 2
        E = E - eta * QL @ (Gr / np.sqrt(V)) @ QR.T
        out.append(loss(E) / loss0)
    return np.array(out)


def adagrad_orig(eta, T):
    E, V, out = E0.copy(), np.zeros((m, n)), [1.0]
    for _ in range(T):
        G = A @ E @ C
        V += G ** 2
        E = E - eta * G / np.sqrt(V)
        out.append(loss(E) / loss0)
    return np.array(out)


def measured_exponent(L, R, s):
    lo, hi = eff_range(L, R, s)
    eta = 2 / (lo + hi)
    PL, PR = psd_pow(L, -s), psd_pow(R, -s)
    T = int(min(30000, 30 * hi / lo + 300))
    E = E0.copy()
    logratio = 0.0                       # log of the loss ratio over the last 100 steps
    for t in range(T):
        prev = loss(E)
        E = E - eta * PL @ (A @ E @ C) @ PR
        if t >= T - 100:
            logratio += np.log(loss(E) / prev)
        E /= np.linalg.norm(E)           # rescale every step (the iteration is linear): no underflow
    rho = np.exp(logratio / 200)
    return np.log((1 + rho) / (1 - rho)) / np.log(kH)


N, T = args.N, 250
L1, R1 = sample_factors(1, N)
L2, R2 = sample_factors(2, N)
svals = np.linspace(0, 1, 17)
if not FIGURE:                           # the exercise "The V and its floor": print the V, leave the data files alone
    print(f"kappa_A = {kA:g}, kappa_C = {kC:g}, N = {N}; exponent log kappa / log kappa(H)")
    print("   s     q=1 measured  |1-s|    q=2 measured  |1-2s|")
    for s in svals:
        print(f"{s:6.4f}  {measured_exponent(L1, R1, s):10.4f}  {abs(1 - s):7.4f}  "
              f"{measured_exponent(L2, R2, s):12.4f}  {abs(1 - 2 * s):7.4f}")
    raise SystemExit
cols = {
    "gd_th": theory_frozen(1, 0.0, T), "gd_sim": sim_frozen(np.eye(m), np.eye(n), 0.0, T),
    "shq1_th": theory_frozen(1, 0.25, T), "shq1_sim": sim_frozen(L1, R1, 0.25, T),
    "shq2_th": theory_frozen(2, 0.25, T), "shq2_sim": sim_frozen(L2, R2, 0.25, T),
    "soap_th": soap_theory(0.3, T), "soap_sim": soap_sim(L1, R1, 0.3, T),
    "ada_sim": adagrad_orig(1.0, T),
}
with open(os.path.join(OUT, "07-kronecker-ladder.dat"), "w") as f:
    f.write("t " + " ".join(cols) + "\n")
    for t in range(T + 1):
        f.write(f"{t} " + " ".join(f"{cols[k][t]:.6e}" for k in cols) + "\n")

with open(os.path.join(OUT, "07-kronecker-power.dat"), "w") as f:
    f.write("s q1_th q1_sim q2_th q2_sim\n")
    for s in svals:
        f.write(f"{s:.4f} {abs(1 - s):.4f} {measured_exponent(L1, R1, s):.4f} "
                f"{abs(1 - 2 * s):.4f} {measured_exponent(L2, R2, s):.4f}\n")

# summary for the text and caption
for name, (L, R, q) in {"q=1": (L1, R1, 1), "q=2": (L2, R2, 2)}.items():
    lo, hi = eff_range(L, R, 0.25)
    print(f"Shampoo s=1/4, {name}: kappa_eff estimated {hi / lo:.1f}, theory {kH ** abs(1 - q * 0.25):.1f}")
lo, hi = eff_range(L1, R1, 1.0)
print(f"K-FAC-type s=1, q=1: kappa_eff estimated {hi / lo:.2f} (theory 1)")
for k in cols:
    print(k, " ".join(f"{cols[k][t]:.2e}" for t in (10, 50, 100, 150, 200, 250)))
