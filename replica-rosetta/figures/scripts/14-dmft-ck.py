"""Chapter 14 (dmft): the Cugliandolo--Kurchan equations for the spherical p-spin, book normalization.

Covariance E H(s)H(s') = N xi(s.s'/N), xi(q) = q^p (Auffinger--Ben Arous--Cerny; Cugliandolo--Kurchan use q^p/2),
Langevin dynamics ds = (-grad H - mu s) dt + sqrt(2T) dB on |s|^2 = N, start independent of H. For t > t':
  dC(t,t')/dt = -mu C + int_0^t xi''(C(t,s)) R(t,s) C(s,t') ds + int_0^t' xi'(C(t,s)) R(t',s) ds
  dR(t,t')/dt = -mu R + int_t'^t xi''(C(t,s)) R(t,s) R(s,t') ds,     R(t,t) = 1, C(t,t) = 1
  mu(t) = T + int_0^t [xi'(C) + xi''(C) C] R ds,                     e(t) = -int_0^t xi'(C(t,s)) R(t,s) ds.
Scheme: Heun's method in t with trapezoidal quadrature (second order), Richardson extrapolation from h and h/2.
Checks: p = 2, T = 0 against the exact solution of Theorem 14.3; p = 3, T = 1.2 against the equilibrium energy -1/T.
Also prints the static and dynamical numbers quoted in the text (T_d, q_d, the CK aging predictions, thresholds).
Writes figures/data/14-dmft-ck-energy.dat and 14-dmft-ck-aging.dat.
"""
import math
import os
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")   # the machine is shared: two BLAS threads avoid oversubscription
import numpy as np  # noqa: E402
from scipy import special  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
P = 3
TEMPS = (1.2, 0.7, 0.0)
TMAX, H = 50.0, 0.05
TWS = (2.5, 5.0, 10.0, 20.0)


def make_xi(p):
    if p == 2:
        return (lambda c: 2.0 * c), (lambda c: 2.0 + 0.0 * c)
    return (lambda c: p * c ** (p - 1)), (lambda c: p * (p - 1) * c ** (p - 2))


def F_row(C, R, i, T, xi1, xi2, h):
    """mu_i, e_i and the derivatives F_C(i, j), F_R(i, j), j <= i, from rows 0..i of C and R."""
    Crow, Rrow = C[i, : i + 1], R[i, : i + 1]
    u = xi1(Crow)
    v = xi2(Crow) * Rrow
    if i == 0:
        return T, 0.0, np.array([-T]), np.array([-T])
    w = np.full(i + 1, h)
    w[0] = w[-1] = 0.5 * h
    mu = T + np.dot(w, u * Rrow + v * Crow)
    e = -np.dot(w, u * Rrow)
    Cs, Rl = C[: i + 1, : i + 1], R[: i + 1, : i + 1]
    dR = np.diagonal(Rl)
    I1 = (w * v) @ Cs                                                  # int_0^t xi'' R C(s, t_j)
    I2 = h * (Rl @ u) - 0.5 * h * (u[0] * Rl[:, 0] + u * dR)          # int_0^{t_j} xi' R(t_j, s)
    I2[0] = 0.0
    I3 = h * (v @ Rl) - 0.5 * h * (v * dR + v[i] * Rl[i, :])          # int_{t_j}^t xi'' R R(s, t_j)
    I3[i] = 0.0
    return mu, e, -mu * Crow + I1 + I2, -mu * Rrow + I3


def integrate_ck(p, T, h, n):
    xi1, xi2 = make_xi(p)
    C = np.zeros((n + 1, n + 1))
    R = np.zeros((n + 1, n + 1))
    C[0, 0] = R[0, 0] = 1.0
    E = np.zeros(n + 1)
    for i in range(n):
        mu, e, FC, FR = F_row(C, R, i, T, xi1, xi2, h)
        E[i] = e
        C[i + 1, : i + 1] = C[i, : i + 1] + h * FC            # predictor
        R[i + 1, : i + 1] = R[i, : i + 1] + h * FR
        C[: i + 1, i + 1] = C[i + 1, : i + 1]
        C[i + 1, i + 1] = R[i + 1, i + 1] = 1.0
        _, _, FC2, FR2 = F_row(C, R, i + 1, T, xi1, xi2, h)   # corrector
        newC = C[i, : i + 1] + 0.5 * h * (FC + FC2[: i + 1])
        C[i + 1, : i + 1] = newC
        C[: i + 1, i + 1] = newC
        R[i + 1, : i + 1] = R[i, : i + 1] + 0.5 * h * (FR + FR2[: i + 1])
    E[n] = F_row(C, R, n, T, xi1, xi2, h)[1]
    return C, R, E


def extrapolated(p, T, h, tmax):
    """Richardson extrapolation (4 f_{h/2} - f_h)/3 on the grid of step h."""
    n = int(round(tmax / h))
    C1, R1, E1 = integrate_ck(p, T, h, n)
    C2, R2, E2 = integrate_ck(p, T, h / 2, 2 * n)
    return (4 * C2[::2, ::2] - C1) / 3, (4 * E2[::2] - E1) / 3, E1, E2[::2]


def statics(p):
    Td = math.sqrt(p * (p - 2) ** (p - 2) / ((p - 1) ** (p - 1)))
    qd = (p - 2) / (p - 1)
    Einf = 2 * math.sqrt((p - 1) / p)
    g = lambda q: q * q / (p * (1 - q)) + q + math.log(1 - q)       # static 1RSB (Kauzmann) condition
    grid = np.linspace(1e-4, 1 - 1e-4, 20000)
    k = next(i for i in range(len(grid) - 1) if g(grid[i]) * g(grid[i + 1]) < 0)
    qs = brentq(g, grid[k], grid[k + 1])
    Ts = math.sqrt(p * qs ** (p - 2) * (1 - qs))
    return Td, qd, Einf, Ts, qs


def ck_aging(p, T):
    """CK (1993) asymptotic solution below T_d: marginality xi''(q)(1-q)^2 = T^2 (largest root),
    x = (p-2)(1-q)/q, asymptotic energy -(1/T)[xi(1) - (1-x) xi(q)]."""
    q = brentq(lambda q: p * (p - 1) * q ** (p - 2) * (1 - q) ** 2 - T * T, (p - 2) / p, 1 - 1e-15)
    x = (p - 2) * (1 - q) / q
    return q, x, -(1 / T) * (1 - (1 - x) * q ** p)


def main():
    t0 = time.time()
    s2 = math.sqrt(2)
    # numbers quoted in the text, book normalization (CK normalization in parentheses: divide by sqrt 2)
    for p in (3, 4):
        Td, qd, Einf, Ts, qs = statics(p)
        print("p=%d: T_d=%.6f (CK %.6f), q_d=%.6f, T_s=%.6f (CK %.6f), q_s=%.6f, -E_inf=%.6f (CK %.6f)"
              % (p, Td, Td / s2, qd, Ts, Ts / s2, qs, -Einf, -Einf / s2))
    for T in (0.7, 0.5, 0.1, 0.01):
        q, x, E = ck_aging(P, T)
        print("CK aging prediction p=3, T=%.2f: q_EA=%.5f, x=%.5f, E_inf(T)=%.5f" % (T, q, x, E))
    # check 1: p = 2, T = 0 against Theorem 14.3
    Cx, Ex, E1, E2 = extrapolated(2, 0.0, 0.05, 10.0)
    ts = np.arange(len(Ex)) * 0.05
    u = 2 * s2 * ts[1:]
    e_exact = -2 * special.ive(2, 2 * u) / special.ive(1, 2 * u) / s2
    lphi = lambda v: np.log(special.ive(1, 2 * v)) + 2 * v - np.log(v)
    Cex = math.exp(lphi(s2 * 15.0) - 0.5 * (lphi(2 * s2 * 10.0) + lphi(2 * s2 * 5.0)))
    print("check p=2, T=0: max|e - exact| = %.2e (h=0.05), %.2e (h=0.025), %.2e (extrapolated); "
          "C(10,5) = %.6f vs exact %.6f" % (np.max(np.abs(E1[1:] - e_exact)), np.max(np.abs(E2[1:] - e_exact)),
                                            np.max(np.abs(Ex[1:] - e_exact)), Cx[200, 100], Cex))
    # p = 3 at three temperatures
    energies = {}
    agingC = {}
    for T in TEMPS:
        t1 = time.time()
        Cx, Ex, E1, E2 = extrapolated(P, T, H, TMAX)
        energies[T] = Ex
        agingC[T] = Cx
        pts = (1, 2, 5, 10, 20, 50)
        print("p=3 T=%.2f (%.1fs): " % (T, time.time() - t1)
              + ", ".join("e(%g)=%.5f" % (t, Ex[int(round(t / H))]) for t in pts)
              + " | h-difference at t=50: %.1e" % abs(E2[-1] - E1[-1]))
        for tw in TWS:
            iw = int(round(tw / H))
            vals = ["C(tw+%g,tw)=%.4f" % (tau, Cx[iw + int(round(tau / H)), iw]) for tau in (1, 5, 20)
                    if iw + int(round(tau / H)) <= len(Ex) - 1]
            print("    t_w=%4.1f: %s" % (tw, ", ".join(vals)))
    print("equilibrium energy at T=1.2: %.5f ; CK aging energy at T=0.7: %.5f ; threshold %.5f"
          % (-1 / 1.2, ck_aging(P, 0.7)[2], -2 * math.sqrt((P - 1) / P)))
    # data files: energies on every second grid point, aging curves C(t_w + tau, t_w)
    n = len(energies[0.0]) - 1
    rows = [[k * H] + [energies[T][k] for T in TEMPS] for k in range(0, n + 1, 2)]
    rk.write_dat(os.path.join(DATA, "14-dmft-ck-energy.dat"), ["t", "e12", "e07", "e00"], rows)
    arows = []
    for k in range(0, int(round(30 / H)) + 1, 2):
        tau = k * H
        row = [tau]
        for T in (0.7, 1.2):
            for tw in TWS:
                iw = int(round(tw / H))
                row.append(agingC[T][iw + k, iw] if iw + k <= n else float("nan"))
        arows.append(row)
    # columns a07w1..a07w4 (T = 0.7) and a12w1..a12w4 (T = 1.2) for t_w = 2.5, 5, 10, 20
    rk.write_dat(os.path.join(DATA, "14-dmft-ck-aging.dat"),
                 ["tau"] + ["a07w%d" % (k + 1) for k in range(len(TWS))]
                 + ["a12w%d" % (k + 1) for k in range(len(TWS))], arows)
    print("runtime %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
