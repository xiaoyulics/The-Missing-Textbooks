"""Chapter 10: unregularized robust regression (Huber's M-estimator) with p/n = kappa = 0.3.

Model: y_i = <x_i, beta0> + w_i, x_i ~ N(0, I_p), n = 1000, p = 300 (alpha = n/p = 10/3), and the
M-estimator beta_hat = argmin sum_i rho(y_i - <x_i, beta>) with Huber's loss
rho_k(u) = u^2/2 (|u| <= k), k|u| - k^2/2 (|u| > k). By regression equivariance beta0 = 0 is no loss.
Prediction (El Karoui et al. 2013; El Karoui 2013; Donoho--Montanari 2016; Thrampoulidis--Abbasi--Hassibi
2018), with zhat = W + r Z:
    E[prox_c(rho)'(zhat)] = 1 - kappa,     kappa r^2 = E[(zhat - prox_c(rho)(zhat))^2],
and |beta_hat - beta0|^2 -> r^2 = R (the excess risk of the chapter).
Noise: W ~ N(0, 1), or W ~ Student t with 3 degrees of freedom (variance 3).
Least squares: r^2 = Var(W) kappa/(1 - kappa). Information bound (Donoho--Montanari 2016, Cor. 4.3, in
these units): r^2 >= kappa / ((1 - kappa) I(W)), with I = 1 (Gaussian) and I = (nu+1)/(nu+3) = 2/3 (t_3).
Writes 10-lasso-robust-curve.dat (prediction against k) and 10-lasso-robust-sim.dat (10 data sets per
point, seeds 4000 + ..., means and Student-t 95% intervals). Runtime: under a minute.
"""
import os
import sys
import time
import math

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                     # noqa: E402
from scipy import optimize, special    # noqa: E402
import rosettakit as rk                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
KAPPA, N, P = 0.3, 1000, 300
SQ2PI = math.sqrt(2 * math.pi)
Phi = special.ndtr


def phi(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / SQ2PI


def pieces(m, s, T):
    """For V ~ N(m, s^2): P(|V| <= T) and E[V^2 1{|V| <= T}] (m may be an array)."""
    a, b = (-T - m) / s, (T - m) / s
    Pin = Phi(b) - Phi(a)
    ev2 = m * m * Pin + 2 * m * s * (phi(a) - phi(b)) + s * s * (Pin + a * phi(a) - b * phi(b))
    return Pin, ev2


# E f(W) for W ~ t_3: W = sqrt(3) tan(u), density dw = (2/pi) cos^2(u) du on (-pi/2, pi/2)
_LX, _LW = np.polynomial.legendre.leggauss(400)
_U = 0.5 * math.pi * _LX
W_T3 = math.sqrt(3) * np.tan(_U)
P_T3 = (2 / math.pi) * np.cos(_U) ** 2 * 0.5 * math.pi * _LW


def huber_moments(r, c, k, noise):
    """E prox_c(rho_k)'(zhat) and E (zhat - prox)^2 for zhat = W + r Z.
    prox_c(x) = x/(1+c) for |x| <= k(1+c), x - c k sgn(x) otherwise."""
    T = k * (1 + c)
    if noise == "gauss":
        Pin, ev2 = pieces(0.0, math.sqrt(1 + r * r), T)
        return Pin / (1 + c) + (1 - Pin), (c / (1 + c)) ** 2 * ev2 + (c * k) ** 2 * (1 - Pin)
    Pin, ev2 = pieces(W_T3, r, T)
    return (float(P_T3 @ (Pin / (1 + c) + (1 - Pin))),
            float(P_T3 @ ((c / (1 + c)) ** 2 * ev2 + (c * k) ** 2 * (1 - Pin))))


def solve_system(k, noise, kappa=KAPPA):
    def c_of_r(r):
        return rk.root(lambda c: huber_moments(r, c, k, noise)[0] - (1 - kappa), 1e-12, 1e8)

    r = rk.root(lambda r: kappa * r * r - huber_moments(r, c_of_r(r), k, noise)[1], 1e-4, 100.0)
    return r * r, c_of_r(r)


def huber_fit(X, y, k, maxit=100):
    """Huber M-estimator by Newton's method with backtracking (the objective is convex and piecewise
    quadratic); started at least squares."""
    b = np.linalg.lstsq(X, y, rcond=None)[0]

    def obj(b):
        a = np.abs(y - X @ b)
        return np.where(a <= k, 0.5 * a * a, k * a - 0.5 * k * k).sum()
    f = obj(b)
    for _ in range(maxit):
        res = y - X @ b
        g = -X.T @ np.clip(res, -k, k)
        w = (np.abs(res) <= k).astype(float)
        step = np.linalg.solve((X.T * w) @ X + 1e-9 * np.eye(X.shape[1]), -g)
        t = 1.0
        while True:
            fn = obj(b + t * step)
            if fn <= f + 1e-4 * t * (g @ step) or t < 1e-12:
                break
            t *= 0.5
        b = b + t * step
        done = abs(f - fn) <= 1e-14 * max(1.0, f)
        f = fn
        if done or np.linalg.norm(g) < 1e-10:
            break
    return b


def main():
    t0 = time.time()
    var = {"gauss": 1.0, "t3": 3.0}
    fisher = {"gauss": 1.0, "t3": 2.0 / 3.0}
    for noise in ("gauss", "t3"):
        print(f"{noise}: least squares r^2 = {var[noise] * KAPPA / (1 - KAPPA):.5f}; "
              f"information bound kappa/((1-kappa) I) = {KAPPA / ((1 - KAPPA) * fisher[noise]):.5f}")
        for k in (0.5, 1.345, 3.0):
            r2, c = solve_system(k, noise)
            print(f"   Huber k = {k}: r^2 = {r2:.5f} (c = {c:.4f})")
        best = optimize.minimize_scalar(lambda kk: solve_system(kk, noise)[0], bounds=(0.2, 20.0), method="bounded",
                                        options={"xatol": 1e-6})
        print(f"   best Huber threshold k = {best.x:.4f}, r^2 = {best.fun:.5f}")
    ks = np.exp(np.linspace(math.log(0.1), math.log(10.0), 61))
    rows = [(k, solve_system(k, "gauss")[0], solve_system(k, "t3")[0]) for k in ks]
    rk.write_dat(os.path.join(DATA, "10-lasso-robust-curve.dat"), ["k", "gauss", "t3"], rows)
    # simulation
    ksim = [0.25, 0.5, 1.0, 1.345, 2.0, 4.0]
    reps = 10
    sim = {}
    for ni, noise in enumerate(("gauss", "t3")):
        for i, k in enumerate(ksim + [None]):
            vals = []
            for r in range(reps):
                gen = rk.rng(4000 + 1000 * ni + 100 * i + r)
                X = gen.standard_normal((N, P))
                w = gen.standard_normal(N) if noise == "gauss" else gen.standard_t(3, size=N)
                b = np.linalg.lstsq(X, w, rcond=None)[0] if k is None else huber_fit(X, w, k)
                vals.append(float(np.sum(b ** 2)))
            sim[(noise, k)] = rk.mean_ci(vals)
            pred = var[noise] * KAPPA / (1 - KAPPA) if k is None else solve_system(k, noise)[0]
            m, h = sim[(noise, k)]
            print(f"{noise:5s} k = {('LS' if k is None else k)!s:6s}: simulated |beta_hat|^2 = {m:.4f} +- {h:.4f}, predicted {pred:.4f}")
    rows = []
    for k in ksim:
        mg, hg = sim[("gauss", k)]
        mt, ht = sim[("t3", k)]
        rows.append((k, mg, hg, mt, ht))
    rk.write_dat(os.path.join(DATA, "10-lasso-robust-sim.dat"), ["k", "gauss", "gci", "t3", "tci"], rows)
    print("runtime %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
