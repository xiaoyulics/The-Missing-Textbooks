"""Chapter 12 (phases): the crossover of the quartic model y = a x1 + b^2 x2 near a singular truth.

Truth: y = delta x2 + noise, i.e. (a0, b0^2) = (0, delta), so the true parameters are (0, +-sqrt(delta)),
regular for delta > 0 (learning coefficient 1), while at delta = 0 the learning coefficient is 3/4.
Inputs x ~ N(0, I_2), unit noise, prior uniform on [-1, 1]^2.

What it computes
  * the universal curves of Proposition prop:phases:crossover, with tau = sqrt(n) delta and
    h(s) = int_R exp(-(u^2 - s)^2 / 2) du:
        population  Lambda_pop(tau)  = 3/4 - (tau/2) h'(tau)/h(tau)
        data        Lambda_data(tau) = 3/4 - (tau/2) E[h'(tau+Z)/h(tau+Z)],   Z ~ N(0, 1);
  * the population effective learning coefficient d Fbar / d log n for delta = 1e-2, 1e-3, 1e-4 by
    exact quadrature (the curves coincide with Lambda_pop);
  * the measured effective coefficient: the slope of E[F_n - n S_n] between nested sample sizes n and
    2n, averaged over R data sets drawn through their sufficient statistics (S = sum x x^T is
    Wishart, U = sum x eps is N(0, S) given S), with the control variate rho_n^2/2 - 1 (rho_n^2 =
    U' S^{-1} U is chi-squared with 2 degrees of freedom given the inputs, so its mean is exactly 2),
    and Student-t 95% intervals over the data sets.
The free energy of one data set is a one-dimensional integral: for fixed b the energy is quadratic in
a, so the a-integral over [-1, 1] is a Gaussian integral in closed form, and the b-integral is a
trapezoid rule on the scale u = n^{1/4} b (checked by doubling the grid).

Writes figures/data/12-phases-crossover-theory.dat and 12-phases-crossover-d{2,3,4}.dat.
Run from the volume root: python figures/scripts/12-phases-crossover.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy import special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

OUT = os.path.join("figures", "data")
R = 3000                       # data sets per delta
DELTAS = [1e-2, 1e-3, 1e-4]
SEED = 1200


# ---------------------------------------------------------------------------------------------
# The function h and the universal curves
# ---------------------------------------------------------------------------------------------
def log_h_ratio(s, N=6001):
    """log h(s) and h'(s)/h(s) for an array s (trapezoid rule on u in [0, sqrt(max s_+ + 40)])."""
    s = np.atleast_1d(np.asarray(s, dtype=float))[:, None]
    u = np.linspace(0.0, math.sqrt(max(float(s.max()), 0.0) + 40.0), N)[None, :]
    e = -(u * u - s) ** 2 / 2
    m = e.max(axis=1, keepdims=True)
    w = np.exp(e - m)
    du = u[0, 1] - u[0, 0]
    trap = lambda f: du * (f.sum(axis=1) - 0.5 * (f[:, 0] + f[:, -1]))
    I0 = trap(w)
    I1 = trap((u * u - s) * w)
    return np.log(2 * I0) + m[:, 0], I1 / I0


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(120)
GH_W = GH_W / GH_W.sum()


def Lambda_pop(tau):
    return 0.75 - 0.5 * tau * log_h_ratio([tau])[1][0]


def Lambda_data(tau):
    return 0.75 - 0.5 * tau * float(GH_W @ log_h_ratio(tau + GH_X)[1])


def Psi(tau):
    """-E log h(tau + Z): E[F_n - n S_n] = (3/4) log n + log 4 - (1/2) log(2 pi) - 1 + Psi(tau) + o(1)."""
    return -float(GH_W @ log_h_ratio(tau + GH_X)[0])


# ---------------------------------------------------------------------------------------------
# Free energies by quadrature (vectorized over data sets)
# ---------------------------------------------------------------------------------------------
def log_a_integral(A2, A1):
    """log int_{-1}^{1} exp(-A2 a^2 - A1 a) da for A2 > 0 (arrays)."""
    mu = -A1 / (2 * A2)
    sig = 1 / np.sqrt(2 * A2)
    hi, lo = (1 - mu) / sig, (-1 - mu) / sig
    l_hi, l_lo = special.log_ndtr(hi), special.log_ndtr(lo)
    ldiff = l_hi + np.log1p(-np.exp(np.minimum(l_lo - l_hi, 0.0)))
    return 0.5 * np.log(np.pi / A2) + A1 ** 2 / (4 * A2) + ldiff


def free_energy(n, delta, S, U, N=1200):
    """F_n - n S_n for data sets with statistics S (R,2,2), U (R,2): -log int phi exp(-n K_n)."""
    tau = math.sqrt(n) * delta
    umax = min(math.sqrt(max(tau, 0.0) + 14.0 + 6.0 * math.sqrt(1 + abs(tau))), n ** 0.25)
    u = np.linspace(-umax, umax, N)
    b = u * n ** -0.25
    beta = b * b - delta                                         # coefficient error of x2
    S11, S12, S22 = S[:, 0, 0][:, None], S[:, 0, 1][:, None], S[:, 1, 1][:, None]
    U1, U2 = U[:, 0][:, None], U[:, 1][:, None]
    la = log_a_integral(0.5 * S11, S12 * beta[None, :] - U1)    # the a-integral, closed form
    e = la - 0.5 * S22 * beta[None, :] ** 2 + U2 * beta[None, :]
    m = e.max(axis=1, keepdims=True)
    w = np.exp(e - m)
    du = u[1] - u[0]
    I = du * (w.sum(axis=1) - 0.5 * (w[:, 0] + w[:, -1]))
    return -(math.log(0.25) - 0.25 * math.log(n) + m[:, 0] + np.log(I))


def wishart2(dof, gen, size):
    """Bartlett: R independent Wishart(dof, I_2) matrices."""
    c1 = np.sqrt(gen.chisquare(dof, size))
    c2 = np.sqrt(gen.chisquare(dof - 1, size))
    z = gen.standard_normal(size)
    A = np.zeros((size, 2, 2))
    A[:, 0, 0], A[:, 1, 0], A[:, 1, 1] = c1, z, c2
    return A @ np.transpose(A, (0, 2, 1))


def main():
    t0 = time.time()
    # --- universal curves ---------------------------------------------------------------------
    xs = np.linspace(-9.0, 9.0, 181)                      # x = log(n delta^2) = 2 log tau
    rows = []
    for x in xs:
        tau = math.exp(x / 2)
        rows.append((x, tau, Lambda_pop(tau), Lambda_data(tau)))
    rows = np.array(rows)
    sk.write_dat(os.path.join(OUT, "12-phases-crossover-theory.dat"), ["x", "tau", "pop", "data"], rows)
    i_min_p, i_max_p = rows[:, 2].argmin(), rows[:, 2].argmax()
    i_min_d, i_max_d = rows[:, 3].argmin(), rows[:, 3].argmax()
    print(f"Lambda_pop: min {rows[i_min_p,2]:.4f} at tau {rows[i_min_p,1]:.3f}; "
          f"max {rows[i_max_p,2]:.4f} at tau {rows[i_max_p,1]:.3f}")
    print(f"Lambda_data: min {rows[i_min_d,3]:.4f} at tau {rows[i_min_d,1]:.3f}; "
          f"max {rows[i_max_d,3]:.4f} at tau {rows[i_max_d,1]:.3f}")
    print("Lambda_pop at tau = 0.1, 1, 10, 100:", [round(Lambda_pop(t), 4) for t in (0.1, 1, 10, 100)])
    print("Lambda_data at tau = 0.1, 1, 10, 100:", [round(Lambda_data(t), 4) for t in (0.1, 1, 10, 100)])
    # extremes located by a bounded optimizer (the grid above has step 0.1 in log(n delta^2))
    from scipy.optimize import brentq, minimize_scalar
    for name, fun in (("Lambda_pop", Lambda_pop), ("Lambda_data", Lambda_data)):
        lo = minimize_scalar(fun, bounds=(0.1, 1.5), method="bounded", options={"xatol": 1e-6})
        hi = minimize_scalar(lambda t: -fun(t), bounds=(1.5, 8.0), method="bounded", options={"xatol": 1e-6})
        print(f"{name}: minimum {lo.fun:.5f} at n delta^2 = {lo.x ** 2:.3f}; "
              f"maximum {-hi.fun:.5f} at n delta^2 = {hi.x ** 2:.2f}")
    hmax = minimize_scalar(lambda t: -log_h_ratio([t])[0][0], bounds=(0.2, 2.0), method="bounded",
                           options={"xatol": 1e-7})
    print(f"h is largest at tau = {hmax.x:.4f}")
    # midpoints 7/8
    tp = brentq(lambda t: Lambda_pop(t) - 0.875, 0.2, 3.0)
    td = brentq(lambda t: Lambda_data(t) - 0.875, 0.2, 3.0)
    print(f"Lambda = 7/8 at tau_pop = {tp:.3f} (n delta^2 = {tp*tp:.3f}), tau_data = {td:.3f} (n delta^2 = {td*td:.3f})")
    print(f"h(0) = 2 * 2^(1/4) Gamma(5/4) = {2 * 2 ** 0.25 * math.gamma(1.25):.6f}")

    # --- population check: exact quadrature of Fbar for each delta ------------------------------
    one = np.eye(2)[None, :, :]
    zero = np.zeros((1, 2))
    maxdev = 0.0
    for delta in DELTAS:
        for x in np.linspace(-4, 8, 25):
            n = math.exp(x) / delta ** 2
            if n < 100:
                continue
            eps = 1e-3
            Fp = free_energy(n * math.exp(eps), delta, n * math.exp(eps) * one, zero, N=6000)[0]
            Fm = free_energy(n * math.exp(-eps), delta, n * math.exp(-eps) * one, zero, N=6000)[0]
            lam = (Fp - Fm) / (2 * eps)
            maxdev = max(maxdev, abs(lam - Lambda_pop(math.sqrt(n) * delta)))
    print(f"population: max |dFbar/dlog n - Lambda_pop(sqrt(n) delta)| over three deltas = {maxdev:.2e}")

    # --- measured slopes with data ------------------------------------------------------------
    gen = sk.rng(SEED)
    summary = []
    for k, delta in enumerate(DELTAS):
        n0 = max(100.0, (math.exp(-6.0 / 2) / delta) ** 2)
        ns = [int(round(n0 * 2 ** j)) for j in range(40) if n0 * 2 ** j <= (math.exp(7.0 / 2) / delta) ** 2]
        S = np.zeros((R, 2, 2))
        U = np.zeros((R, 2))
        prev = 0
        Fs, CVs = [], []
        for n in ns:
            dS = wishart2(n - prev, gen, R)
            L = np.linalg.cholesky(dS)
            U = U + np.einsum("rij,rj->ri", L, gen.standard_normal((R, 2)))
            S = S + dS
            prev = n
            F = free_energy(n, delta, S, U)
            rho2 = np.einsum("ri,ri->r", U, np.linalg.solve(S, U[:, :, None])[:, :, 0])
            Fs.append(F)
            CVs.append(F + 0.5 * rho2 - 1.0)
        rows = []
        for j in range(len(ns) - 1):
            n1, n2 = ns[j], ns[j + 1]
            lr = math.log(n2 / n1)
            d_raw = (Fs[j + 1] - Fs[j]) / lr
            d_cv = (CVs[j + 1] - CVs[j]) / lr
            mu, lo, hi = sk.mean_ci(d_cv)
            nmid = math.sqrt(n1 * n2)
            tau1, tau2 = math.sqrt(n1) * delta, math.sqrt(n2) * delta
            pred_fd = 0.75 + (Psi(tau2) - Psi(tau1)) / lr
            pred_mid = Lambda_data(math.sqrt(nmid) * delta)
            rows.append((math.log(nmid * delta ** 2), nmid, mu, hi - mu, mu - lo, pred_fd, pred_mid,
                         float(np.std(d_raw, ddof=1)), float(np.std(d_cv, ddof=1))))
        rows = np.array(rows)
        sk.write_dat(os.path.join(OUT, f"12-phases-crossover-d{k + 2}.dat"),
                     ["x", "n", "lam", "hi", "lo", "predfd", "predmid", "sdraw", "sdcv"], rows)
        inside = np.mean(np.abs(rows[:, 2] - rows[:, 5]) <= np.maximum(rows[:, 3], rows[:, 4]))
        summary.append((delta, len(rows), inside, np.max(np.abs(rows[:, 5] - rows[:, 6])),
                        rows[:, 7].mean(), rows[:, 8].mean(), ns[0], ns[-1]))
        print(f"delta={delta:g}: n from {ns[0]} to {ns[-1]} ({len(ns)} sizes, ratio 2), R={R}")
        for r in rows:
            print(f"   x={r[0]:+6.2f} n={r[1]:12.4g} lam={r[2]:.4f} (+{r[3]:.4f}/-{r[4]:.4f}) "
                  f"pred_fd={r[5]:.4f} pred_mid={r[6]:.4f}")
    for (delta, m, inside, fdgap, sdraw, sdcv, a, b) in summary:
        print(f"delta={delta:g}: {m} slopes, fraction with |measured - fd prediction| <= half-width: {inside:.2f};"
              f" max |fd - derivative curve| = {fdgap:.4f}; per-data-set sd raw {sdraw:.2f}, with control variate {sdcv:.2f}")

    # --- grid convergence (doubling) on fresh data --------------------------------------------
    g2 = sk.rng(SEED + 1)
    worst = 0.0
    for delta in DELTAS:
        for x in (-3.0, 0.0, 1.0, 3.0, 6.0):
            n = max(100.0, math.exp(x) / delta ** 2)
            S = wishart2(int(n), g2, 50)
            U = np.einsum("rij,rj->ri", np.linalg.cholesky(S), g2.standard_normal((50, 2)))
            F1 = free_energy(int(n), delta, S, U, N=1200)
            F2 = free_energy(int(n), delta, S, U, N=2400)
            worst = max(worst, float(np.max(np.abs(F1 - F2))))
    print(f"grid check: max |F(N=1200) - F(N=2400)| = {worst:.2e}")
    print(f"runtime {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
