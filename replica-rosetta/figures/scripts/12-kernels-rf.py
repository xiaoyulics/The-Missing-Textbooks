"""Chapter 12 (kernels): random-features ridge regression against the Mei--Montanari formula and the
Gaussian covariates model (Gaussian equivalence).

Setting (Mei--Montanari, CPAM 2022 = arXiv:1908.05355v5, Definition 1, Theorems 1--3 and 7):
x_i, theta_j ~ Unif(S^{d-1}(sqrt d)); features sigma(<theta_j, x_i>/sqrt d) with sigma = ReLU;
y_i = <beta, x_i> + eps_i with ||beta|| = 1 (F_1^2 = 1) and Var(eps) = tau^2 = 0.5;
a_hat = argmin (1/n) sum_i (y_i - sum_j a_j sigma(.))^2 + (N lambda_MM / d) ||a||^2, which in the
book's normalization (1/2n)||y - Z a||^2 + (lambda/2)||a||^2 is lambda = psi lambda_MM, psi = N/d;
excess risk R = E_x (f(x) - f_hat(x))^2. Load alpha = n/d = 3, d = 100.
Prediction: R = F_1^2 B(zeta, psi, alpha, lambda_MM/mu*^2) + tau^2 V(...), zeta = mu_1/mu_* (eq. (9) of
v5); eqs. (4) and (52) of v5 print zeta = mu_1^2/mu_*^2, which the script also evaluates.
Gaussian covariates model (MM Section 7): u = mu_0 + mu_1 Theta x/sqrt d + mu_* w, x ~ N(0, I_d),
w ~ N(0, I_N), with the exact test error given (Theta, a).
Outputs: figures/data/12-kernels-rf-sim3.dat (lambda_MM = 1e-3), 12-kernels-rf-sim1.dat (1e-1),
figures/data/12-kernels-rf-curve.dat.
Runtime: about 20 s on an idle machine, up to a minute on a loaded one. Seed 1204.
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
MU0 = 1.0 / math.sqrt(2 * math.pi)
MU1 = 0.5
MUS = math.sqrt(0.25 - 1.0 / (2 * math.pi))
ZETA = MU1 / MUS
D, ALPHA, TAU2 = 100, 3.0, 0.5


def chi_of(psi1, psi2, zeta, lam_bar, tol=1e-13):
    """Definition 1, eq. (15) at xi = i sqrt(psi1 psi2 lam_bar): nu1 = i a, nu2 = i b with a, b > 0.
    A few hundred damped fixed-point steps from (psi1, psi2)/(s+1), then Newton polishing (fsolve);
    the residual is checked. Returns chi = nu1 nu2 = -a b."""
    from scipy import optimize
    s = math.sqrt(psi1 * psi2 * lam_bar)
    z2 = zeta * zeta

    def G(v):
        a, b = v
        return [a - psi1 / (s + b + z2 * b / (1 + z2 * a * b)), b - psi2 / (s + a + z2 * a / (1 + z2 * a * b))]

    a, b = psi1 / (s + 1.0), psi2 / (s + 1.0)
    for _ in range(400):
        an = psi1 / (s + b + z2 * b / (1 + z2 * a * b))
        bn = psi2 / (s + a + z2 * a / (1 + z2 * a * b))
        a, b = 0.5 * (a + an), 0.5 * (b + bn)
    a, b = optimize.fsolve(G, [a, b], xtol=1e-15)
    if max(abs(x) for x in G([a, b])) > tol or a <= 0 or b <= 0:
        raise RuntimeError("fixed point not found")
    return -a * b


def BV(chi, zeta, p1, p2):
    """Eq. (17)-(19) of Mei--Montanari v5."""
    z2, z4, z6 = zeta ** 2, zeta ** 4, zeta ** 6
    c2, c3, c4, c5 = chi ** 2, chi ** 3, chi ** 4, chi ** 5
    E0 = (-c5 * z6 + 3 * c4 * z4 + (p1 * p2 - p2 - p1 + 1) * c3 * z6 - 2 * c3 * z4 - 3 * c3 * z2
          + (p1 + p2 - 3 * p1 * p2 + 1) * c2 * z4 + 2 * c2 * z2 + c2 + 3 * p1 * p2 * chi * z2 - p1 * p2)
    E1 = p2 * c3 * z4 - p2 * c2 * z2 + p1 * p2 * chi * z2 - p1 * p2
    E2 = (c5 * z6 - 3 * c4 * z4 + (p1 - 1) * c3 * z6 + 2 * c3 * z4 + 3 * c3 * z2
          + (-p1 - 1) * c2 * z4 - 2 * c2 * z2 - c2)
    return E1 / E0, E2 / E0


def risk(psi, lam_mm, zeta=ZETA, F1sq=1.0, tau2=TAU2, alpha=ALPHA):
    B, V = BV(chi_of(psi, alpha, zeta, lam_mm / MUS ** 2), zeta, psi, alpha)
    return F1sq * B + tau2 * V


def risk_ridgeless(psi, zeta=ZETA, F1sq=1.0, tau2=TAU2, alpha=ALPHA):
    """Theorem 3 of v5: chi = -([q^2 + 4 zeta^2 m]^{1/2} + q)/(2 zeta^2), q = m zeta^2 - zeta^2 - 1,
    m = min(psi, alpha); the leading minus sign multiplies the whole fraction."""
    m = min(psi, alpha)
    q = m * zeta ** 2 - zeta ** 2 - 1
    chi = -(math.sqrt(q * q + 4 * zeta ** 2 * m) + q) / (2 * zeta ** 2)
    B, V = BV(chi, zeta, psi, alpha)
    return F1sq * B + tau2 * V


def sphere(m, d, gen):
    g = gen.standard_normal((m, d))
    return math.sqrt(d) * g / np.linalg.norm(g, axis=1, keepdims=True)


def sim_rf(N, n, lam_mm, gen, ntest=3000):
    X, Th = sphere(n, D, gen), sphere(N, D, gen)
    Z = np.maximum(X @ Th.T / math.sqrt(D), 0.0)
    y = X[:, 0] + math.sqrt(TAU2) * gen.standard_normal(n)
    a = np.linalg.solve(Z.T @ Z + (n * N * lam_mm / D) * np.eye(N), Z.T @ y)
    Xt = sphere(ntest, D, gen)
    return float(np.mean((Xt[:, 0] - np.maximum(Xt @ Th.T / math.sqrt(D), 0.0) @ a) ** 2))


def sim_gcm(N, n, lam_mm, gen):
    Th = sphere(N, D, gen)
    X = gen.standard_normal((n, D))
    U = MU0 + MU1 * X @ Th.T / math.sqrt(D) + MUS * gen.standard_normal((n, N))
    y = X[:, 0] + math.sqrt(TAU2) * gen.standard_normal(n)
    a = np.linalg.solve(U.T @ U + (n * N * lam_mm / D) * np.eye(N), U.T @ y)
    Sig = MU0 ** 2 + MU1 ** 2 * Th @ Th.T / D + MUS ** 2 * np.eye(N)
    cross = MU1 * Th[:, 0] / math.sqrt(D)
    return float(1.0 - 2 * a @ cross + a @ Sig @ a)


def main():
    t0 = time.time()
    gen = rk.rng(1204)
    n = int(ALPHA * D)
    print("ReLU: mu0=%.6f mu1=%.6f mu*=%.6f zeta=mu1/mu*=%.6f (mu1^2/mu*^2=%.6f)" % (
        MU0, MU1, MUS, ZETA, MU1 ** 2 / MUS ** 2))
    rows = []
    print(" lam_MM   psi    N | formula  zeta^2-read | RF sim (95%%)       | GCM sim (95%%)")
    for lam in [1e-3, 1e-1]:
        for psi in [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
            N = int(round(psi * D))
            reps = 20
            rf = [sim_rf(N, n, lam, gen) for _ in range(reps)]
            gc = [sim_gcm(N, n, lam, gen) for _ in range(reps)]
            m1, h1 = rk.mean_ci(rf)
            m2, h2 = rk.mean_ci(gc)
            th = risk(psi, lam)
            alt = risk(psi, lam, zeta=MU1 ** 2 / MUS ** 2)
            rows.append([lam, psi, m1, h1, m2, h2, th, alt])
            print("%6g %5.2f %4d | %.4f   %.4f     | %.4f (%.4f)   | %.4f (%.4f)" % (lam, psi, N, th, alt, m1, h1, m2, h2))
    head = ["lam", "psi", "rf", "rfci", "gcm", "gcmci", "formula", "zetaalt"]
    rk.write_dat(os.path.join(DATA, "12-kernels-rf-sim3.dat"), head, [r for r in rows if r[0] == 1e-3])
    rk.write_dat(os.path.join(DATA, "12-kernels-rf-sim1.dat"), head, [r for r in rows if r[0] == 1e-1])
    grid = np.concatenate([np.linspace(0.1, 2.8, 55), np.linspace(2.82, 3.18, 19), np.linspace(3.2, 6.5, 67)])
    curve = []
    for psi in grid:
        rl = risk_ridgeless(psi) if abs(psi - ALPHA) > 1e-9 else float("nan")
        curve.append([psi, risk(psi, 1e-3), risk(psi, 1e-1), min(rl, 50.0), risk(psi, 1e-1, zeta=MU1 ** 2 / MUS ** 2)])
    rk.write_dat(os.path.join(DATA, "12-kernels-rf-curve.dat"), ["psi", "r3", "r1", "rless", "r1alt"], curve)
    print("ridgeless (Theorem 3) at psi = 0.5, 2, 4, 6: " + ", ".join("%.4f" % risk_ridgeless(p) for p in [0.5, 2, 4, 6]))
    print("lambda->0 of Definition 1 at psi = 2: %.5f (closed form %.5f)" % (risk(2.0, 1e-8), risk_ridgeless(2.0)))
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
