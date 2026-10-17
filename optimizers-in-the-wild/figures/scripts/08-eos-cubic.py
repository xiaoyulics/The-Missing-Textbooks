"""Chapter 8 (eos): gradient descent on the cubic model of self-stabilization.

The cubic model is the loss L(x, y) = 1/2 (2/eta + y) x^2 - alpha*y, whose gradient descent map is
    x' = -(1 + eta*y) x,    y' = y + eta*(alpha - x^2/2).
In the coordinates A = log x^2 and y, its continuous limit (time tau = eta*t) is
    dA/dtau = 2y,  dy/dtau = alpha - e^A/2,
which conserves H(A, y) = y^2 + e^A/2 - alpha*A.

Outputs (figures/data/):
  08-eos-cubic-levels.dat  closed level curves {H = h} through the two starting points (H = 2 and H = 9.46)
                            (columns a1 y1 a2 y2; 361 points each, angle parameterization)
  08-eos-cubic-phase.dat   gradient descent iterates (A_t, y_t), every 25 steps, for the two orbits
                            (columns a1 y1 a2 y2), eta = 0.002, alpha = 1, two periods each
  08-eos-cubic-avg.dat     running averages along the large orbit: tau, flow averages of x^2 and y
                            (high-accuracy ODE solution) and gradient descent averages (every 100 steps)
  08-eos-cubic-area.dat    the outward drift: scaled time s = 2*alpha*eta^2*t and log(Area(H_t)/Area(H_0))
                            for three (alpha, eta) pairs, where Area(h) is the area enclosed by {H = h}
All runs are deterministic (no random numbers). Runs in a few seconds.
"""
import os

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def phi(A, alpha):
    return 0.5 * np.exp(A) - alpha * A


def H(A, y, alpha):
    return y ** 2 + phi(A, alpha)


def turning_points(h, alpha):
    Astar = np.log(2 * alpha)
    Am = brentq(lambda A: phi(A, alpha) - h, -h / alpha - 50.0, Astar)
    Ap = brentq(lambda A: phi(A, alpha) - h, Astar, np.log(4 * h + 10.0) + 5.0)
    return Am, Ap


def area(h, alpha):
    """Area enclosed by the level curve {H = h}: the integral of 2 sqrt(h - phi(A)) over [A-, A+]."""
    if h <= phi(np.log(2 * alpha), alpha):
        return 0.0
    Am, Ap = turning_points(h, alpha)
    half = 0.5 * (Ap - Am)

    def f(s):  # A = Am + half*(1 - cos s) removes the square-root endpoint behaviour
        A = Am + half * (1.0 - np.cos(s))
        return 2.0 * np.sqrt(max(h - phi(A, alpha), 0.0)) * half * np.sin(s)

    return quad(f, 0.0, np.pi, limit=400, epsabs=1e-13, epsrel=1e-11)[0]


def level_curve(h, alpha, n=361):
    """Points of {H = h}, parameterized by an angle: A = A* + r cos(u) style, solved exactly in y."""
    Am, Ap = turning_points(h, alpha)
    half = 0.5 * (Ap - Am)
    u = np.linspace(0.0, 2.0 * np.pi, n)
    A = Am + half * (1.0 - np.cos(u))
    y = np.sqrt(np.maximum(h - phi(A, alpha), 0.0)) * np.sign(np.sin(u))
    return A, y


def gd_orbit(A0, y0, eta, alpha, steps):
    """Gradient descent in the original coordinates (x, y); returns A_t = log x_t^2 and y_t."""
    x, y = np.sqrt(np.exp(A0)), y0
    As, ys = np.empty(steps + 1), np.empty(steps + 1)
    for t in range(steps + 1):
        As[t], ys[t] = np.log(x * x), y
        x, y = -(1.0 + eta * y) * x, y + eta * (alpha - 0.5 * x * x)
    return As, ys


def main():
    alpha, eta = 1.0, 0.002
    # a medium cycle (H = 2, started at A = log(2 alpha) with y > 0) and a large one,
    # started below the edge with a small oscillation x0 = 0.01
    starts = [(np.log(2 * alpha), np.sqrt(2.0 - phi(np.log(2 * alpha), alpha))), (np.log(1e-4), -0.5)]
    # periods, to choose the run lengths (two periods each)
    hs = [H(A0, y0, alpha) for A0, y0 in starts]

    def period(h):
        Am, Ap = turning_points(h, alpha)
        half = 0.5 * (Ap - Am)
        f = lambda s: half * np.sin(s) / np.sqrt(max(h - phi(Am + half * (1 - np.cos(s)), alpha), 1e-300))
        return quad(f, 0.0, np.pi, limit=400)[0]

    Ts = [period(h) for h in hs]
    print("H values", hs, "periods (tau)", Ts)

    # level curves and gradient descent iterates
    curves = [level_curve(h, alpha) for h in hs]
    with open(os.path.join(DATA, "08-eos-cubic-levels.dat"), "w") as f:
        f.write("a1 y1 a2 y2\n")
        for i in range(len(curves[0][0])):
            f.write(f"{curves[0][0][i]:.5f} {curves[0][1][i]:.5f} {curves[1][0][i]:.5f} {curves[1][1][i]:.5f}\n")
    orbits = []
    for (A0, y0), T in zip(starts, Ts):
        steps = int(round(2 * T / eta))
        orbits.append(gd_orbit(A0, y0, eta, alpha, steps))
    stride = 25
    n1 = len(orbits[0][0][::stride]); n2 = len(orbits[1][0][::stride])
    with open(os.path.join(DATA, "08-eos-cubic-phase.dat"), "w") as f:
        f.write("a1 y1 a2 y2\n")
        for i in range(max(n1, n2)):
            row = []
            for (As, ys), n in zip(orbits, (n1, n2)):
                if i < n:
                    row += [f"{As[i * stride]:.5f}", f"{ys[i * stride]:.5f}"]
                else:
                    row += ["nan", "nan"]
            f.write(" ".join(row) + "\n")

    # running averages along the large orbit, five periods: flow (exact ODE) against gradient descent
    A0, y0 = starts[1]
    tau_max = 5 * Ts[1]
    sol = solve_ivp(lambda t, z: [2 * z[1], alpha - 0.5 * np.exp(z[0])], [0, tau_max], [A0, y0],
                    rtol=1e-11, atol=1e-12, dense_output=True)
    steps = int(round(tau_max / eta))
    As, ys = gd_orbit(A0, y0, eta, alpha, steps)
    x2 = np.exp(As)
    cx2 = np.cumsum(x2) / np.arange(1, steps + 2)
    cy = np.cumsum(ys) / np.arange(1, steps + 2)
    fine = np.linspace(0, tau_max, 200001)
    Z = sol.sol(fine)
    dt = fine[1] - fine[0]
    fx2 = np.concatenate([[np.exp(Z[0][0])], np.cumsum(0.5 * (np.exp(Z[0][1:]) + np.exp(Z[0][:-1])) * dt) / fine[1:]])
    fy = np.concatenate([[Z[1][0]], np.cumsum(0.5 * (Z[1][1:] + Z[1][:-1]) * dt) / fine[1:]])
    with open(os.path.join(DATA, "08-eos-cubic-avg.dat"), "w") as f:
        f.write("tau flowx2 flowy gdx2 gdy\n")
        for t in range(0, steps + 1, 100):
            tau = eta * t
            k = int(round(tau / dt))
            f.write(f"{tau:.4f} {fx2[k]:.5f} {fy[k]:.5f} {cx2[t]:.5f} {cy[t]:.5f}\n")
    print("final running averages: gd x2", cx2[-1], "gd y", cy[-1], "flow x2", fx2[-1], "flow y", fy[-1])

    # the outward drift: area enclosed by the level curve through the current iterate
    runs = [(1.0, 0.01, 30000), (1.0, 0.005, 120000), (0.5, 0.02, 15000)]
    cols = []
    for alpha_r, eta_r, steps_r in runs:
        A0 = brentq(lambda A: phi(A, alpha_r) - 1.0, np.log(2 * alpha_r), 50.0)  # H0 = 1 on the line y = 0
        x, y = np.sqrt(np.exp(A0)), 0.0
        a0 = area(1.0, alpha_r)
        pts = []
        every = steps_r // 30
        for t in range(steps_r + 1):
            if t % every == 0:
                pts.append((2 * alpha_r * eta_r ** 2 * t, np.log(area(H(np.log(x * x), y, alpha_r), alpha_r) / a0)))
            x, y = -(1.0 + eta_r * y) * x, y + eta_r * (alpha_r - 0.5 * x * x)
        cols.append(pts)
        print("drift run", alpha_r, eta_r, "last", pts[-1])
    with open(os.path.join(DATA, "08-eos-cubic-area.dat"), "w") as f:
        f.write("s1 r1 s2 r2 s3 r3\n")
        for i in range(len(cols[0])):
            f.write(" ".join(f"{cols[j][i][0]:.5f} {cols[j][i][1]:.5f}" for j in range(3)) + "\n")


if __name__ == "__main__":
    main()
