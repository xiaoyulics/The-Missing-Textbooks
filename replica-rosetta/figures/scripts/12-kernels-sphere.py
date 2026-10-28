"""Chapter 12 (kernels): learning stages of kernel ridge regression on the sphere, d = 20.

Data x ~ Unif(S^{d-1}(sqrt d)); inner-product kernel H(x, x') = h(<x, x'>/d) with the ReLU
arc-cosine kernel h(t) = (sqrt(1 - t^2) + (pi - arccos t) t)/(2 pi) = E[relu(u) relu(v)], corr(u, v) = t.
Funk--Hecke: every degree-k spherical harmonic is an eigenfunction with eigenvalue
xi_k = E_tau[h(tau) P_k(tau)] (P_k the Gegenbauer polynomial in dimension d with P_k(1) = 1, tau the
inner product of two independent uniform unit vectors), with multiplicity B(d, k).
Target f*(x) = x_1 + x_1 x_2 (degree-1 power 1, degree-2 power d/(d+2)), noise sigma^2 = 0.1,
ridge lambda_book = 1e-3/n (that is, (H + 1e-3 I)^{-1} y in unnormalized form).

Predictions:
  formula      the learning-curve formula with the kernel's full spectrum at d = 20
               (degrees 0..16 explicit, the remaining mass as a vanishing-eigenvalue tail);
  linear GE    the same formula for the linear Gaussian equivalent mu_0 + mu_1 <x,x'>/d + mu_{>=2} delta:
               degree >= 2 mass acts as ridge, the degree-2 target as noise.
Simulation: exact excess risk averaged over the noise, E_eps R = bias + sigma^2 tr(M H2 M), with
H2(x, x') = sum_k xi_k^2 B(d, k) P_k(<x, x'>/d) the L^2 Gram kernel; checked once against test points.
Output: figures/data/12-kernels-sphere-sim.dat, figures/data/12-kernels-sphere-curve.dat.
Runtime: about 30 s on an idle machine, up to 70 s on a loaded one. Seed 1203.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # one BLAS thread: much faster on a shared machine

import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import integrate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
D = 20
KMAX = 16
SIG2 = 0.1
LAMU = 1e-3          # unnormalized ridge: (H + LAMU I)^{-1} y, lambda_book = LAMU / n


def h(t):
    t = np.clip(t, -1.0, 1.0)
    return (np.sqrt(1 - t * t) + (math.pi - np.arccos(t)) * t) / (2 * math.pi)


def gegen(t, kmax, d):
    """P_0..P_kmax at t (array), normalized P_k(1) = 1, by the three-term recurrence."""
    P = [np.ones_like(t), t.copy()]
    for k in range(1, kmax):
        P.append(((2 * k + d - 2) * t * P[k] - k * P[k - 1]) / (k + d - 2))
    return P[: kmax + 1]


def mult_exact(d, k):
    """B(d, k): dimension of the degree-k spherical harmonics in d variables."""
    return 1 if k == 0 else (2 * k + d - 2) * math.factorial(k + d - 3) // (math.factorial(k) * math.factorial(d - 2))


def xis(d, kmax):
    w = lambda t: (1 - t * t) ** ((d - 3) / 2)
    Z = integrate.quad(w, -1, 1, limit=400, epsabs=1e-14, epsrel=1e-13)[0]
    out = []
    for k in range(kmax + 1):
        f = lambda t, k=k: float(h(np.array([t]))[0] * gegen(np.array([t]), k, d)[k][0]) * w(t)
        out.append(integrate.quad(f, -1, 1, limit=400, epsabs=1e-15, epsrel=1e-13)[0] / Z)
    return np.array(out)


XI = xis(D, KMAX)
XI[3::2] = 0.0                              # odd degrees >= 3 vanish exactly (h has no odd part beyond t/4)
XI = np.maximum(XI, 0.0)
B = np.array([mult_exact(D, k) for k in range(KMAX + 1)], dtype=float)
TAIL = 0.5 - float(np.sum(B * XI))         # h(1) = 1/2
POW = np.zeros(KMAX + 1)
POW[1], POW[2] = 1.0, D / (D + 2)           # target power per degree


def formula(n, eig, mult_, powr, tail, sig2, lamu):
    """Learning-curve formula with eigenvalues eig (multiplicities mult_), target power powr per
    eigenspace, a vanishing-eigenvalue tail of mass tail; unnormalized ridge lamu = n lambda."""
    def g(kap):  # n lambda = kap (n - sum m eig/(eig+kap)) - tail
        return kap * (n - float(np.sum(mult_ * eig / (eig + kap)))) - tail - lamu
    # the root lies where c(kap) = n - sum m eig/(eig+kap) > 0; c increases in kap
    c = lambda kap: n - float(np.sum(mult_ * eig / (eig + kap)))
    lo = 0.0 if c(1e-300) > 0 else rk.root(c, 1e-30, 1e3)
    kap = rk.root(g, lo * (1 + 1e-12) + 1e-300, 1e3)
    D2 = float(np.sum(mult_ * eig ** 2 / (eig + kap) ** 2)) / n
    bias = float(np.sum(kap ** 2 * powr / (eig + kap) ** 2))
    return (bias + sig2 * D2) / (1 - D2), kap, D2


def predictions(n):
    full = formula(n, XI, B, POW, TAIL, SIG2, LAMU)[0]
    # linear Gaussian equivalent: degrees 0, 1 only; all degree >= 2 mass as ridge, degree-2 target as noise
    mu2plus = 0.5 - XI[0] - D * XI[1]
    lin_eig = XI[:2]
    lin_mult = B[:2]
    rlin, kap, D2 = formula(n, lin_eig, lin_mult, POW[:2], mu2plus, SIG2 + POW[2], LAMU)
    rlin += POW[2]
    return full, rlin


def sphere(m, d, gen):
    g = gen.standard_normal((m, d))
    return math.sqrt(d) * g / np.linalg.norm(g, axis=1, keepdims=True)


def krr_exact(n, gen, check=False):
    X = sphere(n, D, gen)
    T = X @ X.T / D
    H = h(T)
    M = np.linalg.inv(H + LAMU * np.eye(n))
    P = gegen(np.clip(T, -1, 1), KMAX, D)
    H2 = sum((XI[k] ** 2) * B[k] * P[k] for k in range(KMAX + 1))
    y0 = X[:, 0] + X[:, 0] * X[:, 1]
    a0 = M @ y0
    c = XI[1] * X[:, 0] + XI[2] * X[:, 0] * X[:, 1]
    bias = float(a0 @ H2 @ a0 - 2 * a0 @ c + POW[1] + POW[2])
    var = SIG2 * float(np.einsum("ij,ji->", M @ H2, M))
    if check:
        Xt = sphere(100000, D, gen)
        pred = h(Xt @ X.T / D) @ a0
        mc = float(np.mean((Xt[:, 0] + Xt[:, 0] * Xt[:, 1] - pred) ** 2))
        print("  check at n=%d: exact bias %.5f, Monte Carlo (100000 test points) %.5f" % (n, bias, mc))
    return bias + var


def main():
    t0 = time.time()
    gen = rk.rng(1203)
    print("d=%d: xi_k (per harmonic) and degree masses mu_k = B(d,k) xi_k" % D)
    for k in range(6):
        print("  k=%d  B=%7d  xi=%.6e  mu=%.6f" % (k, B[k], XI[k], B[k] * XI[k]))
    print("  tail mass beyond degree %d: %.2e;  mu_{>=2} = %.6f;  mu_{>2} = %.6f" % (
        KMAX, TAIL, 0.5 - XI[0] - D * XI[1], 0.5 - XI[0] - D * XI[1] - B[2] * XI[2]))
    krr_exact(300, gen, check=True)
    rows = []
    print("   n  | KRR E_eps R (95%)   | formula | linear GE")
    for n in [10, 14, 20, 28, 40, 56, 80, 110, 150, 209, 290, 400, 560, 800, 1100, 1600]:
        reps = 30 if n <= 290 else (12 if n <= 800 else 6)
        vals = [krr_exact(n, gen) for _ in range(reps)]
        m, hw = rk.mean_ci(vals)
        full, lin = predictions(n)
        rows.append([n, m, hw, full, lin])
        print("%5d | %.5f (%.5f) | %.5f | %.5f" % (n, m, hw, full, lin))
    rk.write_dat(os.path.join(DATA, "12-kernels-sphere-sim.dat"), ["n", "krr", "krrci", "formula", "linear"], rows)
    ns = np.unique(np.round(np.logspace(1, np.log10(2000), 90)).astype(int))
    curve = [[int(n)] + list(predictions(int(n))) for n in ns]
    rk.write_dat(os.path.join(DATA, "12-kernels-sphere-curve.dat"), ["n", "formula", "linear"], curve)
    print("B(d,2) = %d, B(d,3) = %d" % (B[2], B[3]))
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
