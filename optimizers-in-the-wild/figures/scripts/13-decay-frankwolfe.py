"""Chapter 13, figure "frankwolfe": AdamW as Frank-Wolfe on the l_infinity ball of radius 1/lambda.

Panel (a), "where in the ball" (theory = exact formulas, simulation = runs): sign descent with decoupled
weight decay (AdamW without memory) and AdamW with (beta1, beta2) = (0.9, 0.95), eta = 0.01, lambda = 0.1,
on a separable linear loss with Gaussian gradient noise, all coordinates started at theta_0 = 3/lambda:
  group C, 20 coordinates: gradient -1 exactly (consistent sign, pushing outward):
      theta_t = r + (1 - eta lam)^t (theta_0 - r)                 (Theorem thm:decay:frankwolfe(ii), equality)
  group S, 200 coordinates: gradient -1 + noise of standard deviation 1 (signal-to-noise ratio 1):
      E theta_t = m r + (1 - eta lam)^t (theta_0 - m r), m = E sign(-g) = erf(1/sqrt 2)  (Proposition prop:decay:regimes)
  group N, 2000 coordinates: gradient = noise of standard deviation 1:
      E theta_t^2 = a^t theta_0^2 + eta^2 (1 - a^t)/(1 - a), a = (1 - eta lam)^2   (Proposition prop:decay:regimes)
Plotted in units of r = 1/lambda: max over C, mean over S, root mean square over N.

Panel (b), "how fast" (theory = the guarantees of Theorem thm:decay:frankwolfe(iii), simulation = runs):
sign descent with decay on a random convex quadratic f(x) = (x-c)^T H (x-c)/2 in R^10 (eigenvalues of H
geometric from 1 to 0.01, random eigenvectors), lambda = 1, started inside the ball. The l_infinity
smoothness constant L = max over sign vectors of s^T H s is computed by enumeration; the constrained
minimum by L-BFGS-B. Step rules gamma_t = 2/(t+2) and constant gamma in {1e-2, 1e-3}.
Writes figures/data/13-decay-fwball.dat and figures/data/13-decay-fwgap.dat. Runtime: about 15 s.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import itertools
from math import erf, sqrt
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")


def panel_a():
    eta, lam = 0.01, 0.1
    r, gam = 1 / lam, eta * lam
    T = int(10 / gam)
    nC, nS, nN = 20, 200, 2000
    mean_grad = np.concatenate([-np.ones(nC), -np.ones(nS), np.zeros(nN)])
    noise = np.concatenate([np.zeros(nC), np.ones(nS), np.ones(nN)])
    theta0 = 3 * r
    every = 100
    rows = []
    sims = {}
    for opt, seed in (("sgn", 31), ("adam", 31)):
        rng = np.random.default_rng(seed)
        th = np.full(nC + nS + nN, theta0)
        m = np.zeros_like(th)
        v = np.zeros_like(th)
        rec = []
        for t in range(T + 1):
            if t % every == 0:
                rec.append((np.max(np.abs(th[:nC])), np.mean(th[nC:nC + nS]),
                            np.sqrt(np.mean(th[nC + nS:] ** 2))))
            g = mean_grad + noise * rng.normal(size=th.size)
            if opt == "sgn":
                u = np.sign(g)
            else:
                m = 0.9 * m + 0.1 * g
                v = 0.95 * v + 0.05 * g * g
                u = (m / (1 - 0.9 ** (t + 1))) / np.sqrt(v / (1 - 0.95 ** (t + 1)))
            th = (1 - gam) * th - eta * u
        sims[opt] = np.array(rec) / r
    ms = erf(1 / sqrt(2))
    a = (1 - gam) ** 2
    for k, t in enumerate(range(0, T + 1, every)):
        env = 1 + (1 - gam) ** t * (theta0 / r - 1)
        sig = ms + (1 - gam) ** t * (theta0 / r - ms)
        rms = np.sqrt(a ** t * theta0 ** 2 + eta ** 2 * (1 - a ** t) / (1 - a)) / r
        rows.append((gam * t, env, sig, rms, *sims["sgn"][k], *sims["adam"][k]))
    with open(os.path.join(OUT, "13-decay-fwball.dat"), "w") as f:
        f.write("s envC meanS rmsN sgnC sgnS sgnN adamC adamS adamN\n")
        for row in rows:
            f.write(" ".join(f"{x:.6g}" for x in row) + "\n")
    last = rows[-1]
    print(f"panel a at eta*lam*t = {last[0]:.1f}: theory C {last[1]:.4f} S {last[2]:.4f} N {last[3]:.5f};"
          f" sign C {last[4]:.4f} S {last[5]:.4f} N {last[6]:.5f}; AdamW C {last[7]:.4f} S {last[8]:.4f} N {last[9]:.5f}")
    print("  equilibrium RMS of a noise coordinate, in units of r: sqrt(eta lam/(2 - eta lam)) =",
          sqrt(gam / (2 - gam)))


def panel_b():
    rng = np.random.default_rng(7)
    d, lam = 10, 1.0
    r = 1 / lam
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    H = Q @ np.diag(np.geomspace(1, 0.01, d)) @ Q.T
    c = 1.2 * rng.normal(size=d)
    f = lambda x: 0.5 * (x - c) @ H @ (x - c)
    grad = lambda x: H @ (x - c)
    S = np.array(list(itertools.product([-1, 1], repeat=d)))
    L = float(np.max(np.einsum("ij,jk,ik->i", S, H, S)))
    res = minimize(f, np.zeros(d), jac=grad, bounds=[(-r, r)] * d, method="L-BFGS-B",
                   options={"ftol": 1e-16, "gtol": 1e-14, "maxiter": 10000})
    fstar = f(res.x)
    gap_star = grad(res.x) @ res.x + r * np.abs(grad(res.x)).sum()
    th0 = 0.5 * r * rng.uniform(-1, 1, size=d)
    T = 100000
    tgrid = np.unique(np.round(np.geomspace(1, T, 60)).astype(int))

    def run(rule, gam=None):
        th = th0.copy()
        h = np.empty(T + 1)
        for t in range(T + 1):
            h[t] = f(th) - fstar
            if t == T:
                break
            gm = 2 / (t + 2) if rule == "fw" else gam
            th = (1 - gm) * th + gm * r * (-np.sign(grad(th)))
        return h

    h_fw = run("fw")
    h_c2 = run("c", 1e-2)
    h_c3 = run("c", 1e-3)
    h0 = h_fw[0]
    with open(os.path.join(OUT, "13-decay-fwgap.dat"), "w") as fo:
        fo.write("t fw boundfw c2 boundc2 c3 boundc3\n")
        for t in tgrid:
            b_fw = 8 * L / (lam ** 2 * (t + 2))
            b_c2 = (1 - 1e-2) ** t * h0 + 2 * L * 1e-2 * r ** 2
            b_c3 = (1 - 1e-3) ** t * h0 + 2 * L * 1e-3 * r ** 2
            fo.write(f"{t} {max(h_fw[t], 1e-300):.6g} {b_fw:.6g} {max(h_c2[t], 1e-300):.6g} {b_c2:.6g} "
                     f"{max(h_c3[t], 1e-300):.6g} {b_c3:.6g}\n")
    print(f"panel b: L_inf = {L:.4f}, active coordinates {int(np.sum(np.abs(np.abs(res.x) - r) < 1e-6))}, "
          f"Frank-Wolfe gap at the computed minimizer {gap_star:.1e}")
    t = np.arange(1, T + 1)
    print(f"  2/(t+2): max_t h_t / bound = {np.max(h_fw[1:] * (t + 2) / (8 * L / lam ** 2)):.3f}, "
          f"t^2 h_t at t = T: {h_fw[-1] * T ** 2:.2f}")
    for name, h, gm in (("1e-2", h_c2, 1e-2), ("1e-3", h_c3, 1e-3)):
        tail = h[-int(5 / gm):]
        print(f"  gamma = {name}: tail mean {tail.mean():.3e} = {tail.mean() / gm ** 2:.3f} gamma^2;"
              f" guarantee 2 L gamma r^2 = {2 * L * gm * r ** 2:.3e}")


if __name__ == "__main__":
    panel_a()
    panel_b()
