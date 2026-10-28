"""Chapter 12 (kernels): kernel ridge regression on the circle against the learning-curve formula.

The kernel K(t, t') = 1 + sum_{k>=1} 2 k^{-2} cos k(t - t') = 1 + 2(pi^2/6 - pi x/2 + x^2/4),
x = (t - t') mod 2 pi, on data t ~ Unif[0, 2 pi): eigenvalues eta_0 = 1 and eta_k = k^{-2} (twice) in
L^2(dt/2pi), eigenfunctions 1, sqrt2 cos kt, sqrt2 sin kt (bounded, not Gaussian). Its RKHS norm is
c_0^2 + ||f'||^2, so the ridgeless interpolant is a linear spline plus a parabolic correction
(Proposition prop:kernels:circle).

Book conventions: per-sample ridge lambda, f_hat = k(.)^T (K + n lambda I)^{-1} y, excess risk
R = ||f_hat - f*||^2_{L^2}; the formula: lambda = lambda_eff (1 - df1(lambda_eff)/n),
R = [sum_k lambda_eff^2 v_k^2/(eta_k+lambda_eff)^2 + sigma^2 df2/n]/(1 - df2/n)
(in the code, kap stands for lambda_eff).

Outputs (figures/data/):
  12-kernels-circle-noise.dat  ridgeless noise term E_eps R / sigma^2 (target 0): KRR (exact, closed
                               form) and Gaussian features with the same spectrum (GCM), with 95% intervals
  12-kernels-circle-lc.dat     learning curve with ridge lambda = 1e-3, sigma^2 = 0.1, target the
                               sawtooth f* = sqrt2 sum_k k^{-1} sin kt = (pi - t)/sqrt2 on (0, 2 pi)
                               (power v_k^2 = k^{-2}): KRR and GCM, with 95% intervals
  12-kernels-circle-curve.dat  the formula on a fine grid of n (both settings)
Printed: part (c), the KRR noise term against the formula as the ridge grows at n = 512.
Runtime: about 20 s on an idle machine, up to 100 s on a loaded one. Seeds: 1201 (KRR), 1202 (GCM).
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

PI = math.pi
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SIG2 = 0.1
LAM = 1e-3


# ------------------------------------------------------------------ the formula (closed forms)
def sums(kap):
    """S1 = sum_{k>=1} 1/(1+kap k^2), S2 = sum 1/(1+kap k^2)^2, and a = kap^{-1/2}."""
    a = 1.0 / math.sqrt(kap)
    e = math.exp(-2 * PI * a)
    cth = (1 + e) / (1 - e)
    csch2 = 4 * e / (1 - e) ** 2
    S1 = (PI * a * cth - 1) / 2
    S2 = (PI * a * cth + PI ** 2 * a * a * csch2 - 2) / 4
    return S1, S2, a


def dfs(kap):
    S1, S2, a = sums(kap)
    df1 = 1 / (1 + kap) + 2 * S1
    df2 = 1 / (1 + kap) ** 2 + 2 * S2
    # bias sum for v_k^2 = k^-2 on the cos modes: sum kap^2 k^2/(1+kap k^2)^2 = S1' - a^2 S2'
    # with S1' = sum 1/(k^2+a^2) = S1/a^2... written directly:
    b = S1 - S2  # sum_k [1/(1+kap k^2) - 1/(1+kap k^2)^2] = sum kap k^2/(1+kap k^2)^2
    bias_sum = kap * b  # = sum_k kap^2 k^2/(1+kap k^2)^2
    return df1, df2, bias_sum


def kappa_of(lam, n):
    if lam == 0.0:
        f = lambda lk: dfs(math.exp(lk))[0] - n
    else:
        f = lambda lk: math.exp(lk) * (1 - dfs(math.exp(lk))[0] / n) - lam
    lo, hi = math.log(1e-30), math.log(1e6)
    if lam > 0:
        # bracket on the branch where 1 - df1/n > 0
        lo = math.log(kappa_of(0.0, n)) + 1e-12
    return math.exp(rk.root(f, lo, hi, tol=1e-14))


def formula(n, lam, sig2, target):
    kap = kappa_of(lam, n)
    df1, df2, bsum = dfs(kap)
    D2 = df2 / n
    bias = bsum if target else 0.0
    return (bias + sig2 * D2) / (1 - D2), kap, D2


# ------------------------------------------------------------------ KRR on the circle
def kern(t1, t2):
    x = np.mod(t1[:, None] - t2[None, :], 2 * PI)
    return 1.0 + 2 * (PI ** 2 / 6 - PI * x / 2 + x * x / 4)


def kern2(t1, t2):
    """K2(t,t') = sum_j eta_j^2 phi_j(t) phi_j(t') = 1 + 2 sum k^-4 cos k x (Bernoulli polynomial)."""
    x = np.mod(t1[:, None] - t2[None, :], 2 * PI)
    return 1.0 + 2 * (PI ** 4 / 90 - PI ** 2 * x * x / 12 + PI * x ** 3 / 12 - x ** 4 / 48)


def noise_ridgeless_exact(t):
    """E_eps ||f_hat||^2 / sigma^2 of the ridgeless interpolant, from the closed form
    f_hat = s - 12 pi c0(s) q/(S3 + 24 pi) (linear spline s, gap parabolas q)."""
    ts = np.sort(t)
    g = np.diff(np.concatenate([ts, [ts[0] + 2 * PI]]))
    gp = np.roll(g, 1)
    w = (gp + g) / (4 * PI)
    v = (gp ** 3 + g ** 3) / (24 * PI)
    S3 = float(np.sum(g ** 3))
    q2 = float(np.sum(g ** 5)) / (60 * PI)
    kt = 12 * PI / (S3 + 24 * PI)
    return 2 / 3 - 2 * kt * float(w @ v) + kt ** 2 * float(w @ w) * q2


def noise_ridgeless_kernel(t):
    Ki = np.linalg.inv(kern(t, t))
    return float(np.trace(Ki @ kern2(t, t) @ Ki))


def krr_risk_target(t, lam, sig2):
    """Exact E_eps R for the sawtooth f* = sqrt2 sum k^-1 sin kt = (pi - t)/sqrt2, ridge lam.
    <K(., t_i), f*> = sqrt2 sum k^-3 sin k t_i = sqrt2 (pi^2 t/6 - pi t^2/4 + t^3/12)."""
    n = t.size
    M = np.linalg.inv(kern(t, t) + n * lam * np.eye(n))
    K2 = kern2(t, t)
    fstar = (PI - t) / math.sqrt(2)
    a0 = M @ fstar
    c = math.sqrt(2) * (PI ** 2 * t / 6 - PI * t * t / 4 + t ** 3 / 12)
    bias = float(a0 @ K2 @ a0 - 2 * a0 @ c + PI ** 2 / 6)
    var = sig2 * float(np.einsum("ij,ji->", M @ K2, M))
    return bias + var


def gcm_risk(n, lam, sig2, target, gen, kfac=4):
    """Gaussian features with the circle's spectrum: modes up to k = kfac*n explicit, tail as diagonal."""
    kmax = kfac * n
    k = np.arange(1, kmax + 1, dtype=float)
    eig = np.concatenate([[1.0], k ** -2.0, k ** -2.0])
    v = np.concatenate([[0.0], 1.0 / k, np.zeros(kmax)]) if target else np.zeros(eig.size)
    tail = 2 * (PI ** 2 / 6) - 2 * float(np.sum(k ** -2.0))
    G = gen.standard_normal((n, eig.size))
    Phi = G * np.sqrt(eig)
    Kmat = Phi @ Phi.T + tail * np.eye(n)
    M = np.linalg.inv(Kmat + n * lam * np.eye(n))
    W = Phi.T @ M                      # f_hat = <W y, phi(x)>, population L2 Gram Lambda
    if not target:
        return sig2 * float(np.sum((W * W) * eig[:, None]))
    # target modes beyond kmax: unlearned (bias) and, in the labels, independent noise
    tail_v2 = PI ** 2 / 6 - float(np.sum(v ** 2))
    var = (sig2 + tail_v2) * float(np.sum((W * W) * eig[:, None]))
    y0 = G @ v                         # f*(x_i) = sum_k v_k g_ik (target power v_k^2 per mode)
    w0 = W @ y0                        # coefficients of f_hat in the phi basis
    # ||f_hat - f*||^2 = sum_j eig_j (w0_j - v_j/sqrt(eig_j))^2 + tail target power
    vt = np.where(eig > 0, v / np.sqrt(eig), 0.0)
    bias = float(np.sum(eig * (w0 - vt) ** 2)) + tail_v2
    return bias + var


def main():
    t0 = time.time()
    gen = rk.rng(1201)
    gcm_gen = rk.rng(1202)

    # consistency of the closed form with the kernel computation
    print("check: ridgeless noise term, closed form vs kernel trace")
    for n in [8, 32, 128]:
        t = gen.uniform(0, 2 * PI, n)
        print("  n=%4d  closed form %.12f  kernel %.12f" % (n, noise_ridgeless_exact(t), noise_ridgeless_kernel(t)))

    # (a) ridgeless noise term
    rows = []
    print("\n(a) ridgeless noise term E_eps R / sigma^2 (target 0)")
    print("   n | KRR (95%%)          | GCM (95%%)          | formula")
    for n in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        vals = [noise_ridgeless_exact(gen.uniform(0, 2 * PI, n)) for _ in range(400)]
        m, h = rk.mean_ci(vals)
        if n <= 512:
            reps = 80 if n <= 128 else (24 if n <= 256 else 12)
            gv = [gcm_risk(n, 0.0, 1.0, False, gcm_gen) for _ in range(reps)]
            gm, gh = rk.mean_ci(gv)
        else:
            gm, gh = float("nan"), float("nan")
        fm = formula(n, 0.0, 1.0, False)[0]
        rows.append([n, m, h, gm, gh, fm])
        print("%5d | %.5f (%.5f) | %.5f (%.5f) | %.5f" % (n, m, h, gm, gh, fm))
    rk.write_dat(os.path.join(DATA, "12-kernels-circle-noise.dat"),
                 ["n", "krr", "krrci", "gcm", "gcmci", "formula"], rows)   # nan = not simulated

    # (b) learning curve with ridge and a target
    rows = []
    print("\n(b) learning curve, lambda=%g, sigma^2=%g, sawtooth target (pi - t)/sqrt2" % (LAM, SIG2))
    print("   n | KRR (95%%)          | GCM (95%%)          | formula | lambda_eff")
    for n in [16, 32, 64, 128, 256, 512, 1024]:
        reps = 100 if n <= 128 else (40 if n <= 512 else 12)
        vals = [krr_risk_target(gen.uniform(0, 2 * PI, n), LAM, SIG2) for _ in range(reps)]
        m, h = rk.mean_ci(vals)
        if n <= 512:
            gv = [gcm_risk(n, LAM, SIG2, True, gcm_gen) for _ in range(30 if n <= 128 else 12)]
            gm, gh = rk.mean_ci(gv)
        else:
            gm, gh = float("nan"), float("nan")
        fm, kap, D2 = formula(n, LAM, SIG2, True)
        rows.append([n, m, h, gm, gh, fm])
        print("%5d | %.5f (%.5f) | %.5f (%.5f) | %.5f | %.3e" % (n, m, h, gm, gh, fm, kap))
    rk.write_dat(os.path.join(DATA, "12-kernels-circle-lc.dat"),
                 ["n", "krr", "krrci", "gcm", "gcmci", "formula"], rows)

    # (c) the crossover: noise term of KRR against the formula as the ridge grows, n = 512
    print("\n(c) noise term at n = 512 as the ridge grows (10 point sets each): KRR, formula, ratio, lambda/lambda_eff")
    for lam in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        vals = []
        for _ in range(10):
            t = gen.uniform(0, 2 * PI, 512)
            M = np.linalg.inv(kern(t, t) + 512 * lam * np.eye(512))
            vals.append(float(np.einsum("ij,ji->", M @ kern2(t, t), M)))
        m, h = rk.mean_ci(vals)
        f, kap, D2 = formula(512, lam, 1.0, False)
        print("  lambda=%7.0e  KRR %.4f (%.4f)  formula %.4f  ratio %.3f  lambda/lambda_eff %.3f" % (lam, m, h, f, m / f, lam / kap))

    # formula curves
    ns = np.unique(np.round(np.logspace(np.log10(6), np.log10(5000), 80)).astype(int))
    curve = [[int(n), formula(int(n), 0.0, 1.0, False)[0], formula(int(n), LAM, SIG2, True)[0]] for n in ns]
    rk.write_dat(os.path.join(DATA, "12-kernels-circle-curve.dat"), ["n", "noise0", "lc"], curve)

    # numbers quoted in the text
    print("\nlimits: formula noise term -> 1 (sigma^2); KRR ridgeless noise term -> 2/3 (Proposition)")
    for n in [16, 256, 4096]:
        f, kap, D2 = formula(n, 0.0, 1.0, False)
        print("  n=%5d formula noise %.6f, lambda_eff %.4e (pi^2/n^2 = %.4e), df2/n %.6f" % (n, f, kap, PI ** 2 / n ** 2, D2))
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
