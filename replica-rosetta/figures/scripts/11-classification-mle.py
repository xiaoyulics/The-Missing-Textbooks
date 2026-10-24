"""Chapter 11 (classification): bias and variance inflation of the logistic MLE (Sur and Candes 2019).

Book units: x_i ~ N(0, I_d), P(y_i = 1 | x_i) = f(<x_i, beta*>) with the logistic link f, |beta*| = r,
n = alpha d. In the region alpha > alpha_MLE(r) the MLE satisfies, in probability,
    <beta_hat, e*> -> a,   |beta_hat - <beta_hat, e*> e*| -> tau,      e* = beta*/r,
so beta_hat ~ mu beta* + (noise of norm tau) with the bias factor mu = a/r, where (a, tau, chi) solve
    (E1) E[V l'(p)] = 0,  (E2) tau^2 = alpha chi^2 E[l'(p)^2],  (E3) 1 = alpha chi E[l''(p)/(1 + chi l''(p))],
    p = prox_{chi l}(a V + tau Z),  l(t) = log(1 + e^{-t}),  V = Y U,  P(Y = 1 | U) = f(r U).
This is Sur and Candes's system (5)-(7) in book units: their (alpha*, sigma*, lambda*) = (mu, tau sqrt(alpha), chi),
their kappa = d/n = 1/alpha, gamma = r; the likelihood-ratio factor kappa sigma*^2/lambda* = tau^2/chi.

Computes
  (1) the solution on a grid of d/n in (0, 0.30] for r = sqrt(5) (gamma^2 = 5; h_MLE = 0.3256), with the
      published checks (d/n = 0.1: alpha* = 1.1678, sigma* = 3.3466, lambda* = 0.9605), the cosine, the test
      error P(a V + tau Z < 0), and the classical (fixed-d) values mu = 1, tau^2 = (d/n)/E f'(r U);
  (2) a simulation: n = 1000, d = n x {0.05, 0.10, 0.15, 0.20, 0.25}, 40 data sets each, MLE by Newton's
      method; estimates mu_hat = <beta_hat, e*>/r and tau_hat = |beta_hat - <beta_hat, e*> e*|, with
      Student-t 95% intervals; at d/n = 0.2, the likelihood-ratio statistic 2 Lambda_j of 10 null coordinates
      per data set (400 tests), against the rescaled chi-square law (tau^2/chi) chi^2_1.
Writes figures/data/11-classification-mle-curve.dat and -sim.dat. Runtime about a minute.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # shared machine: one BLAS thread per script
os.environ.setdefault("OMP_NUM_THREADS", "1")
import math
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import optimize, special

import rosettakit as rk

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
SQ2PI = math.sqrt(2 * math.pi)

# (V, Z) quadrature: V on Gauss--Legendre nodes (split at 0) with the density of V, Z on Gauss--Hermite nodes
_xl, _wl = np.polynomial.legendre.leggauss(160)
_xh, _wh = np.polynomial.hermite_e.hermegauss(64)
_wh = _wh / SQ2PI


def v_nodes(r, L=11.0):
    vs, ws = [], []
    for lo, hi in ((-L, 0.0), (0.0, L)):
        v = 0.5 * (hi - lo) * _xl + 0.5 * (hi + lo)
        dens = special.expit(r * v) + 1.0 - special.expit(-r * v)
        vs.append(v)
        ws.append(0.5 * (hi - lo) * _wl * dens * np.exp(-0.5 * v * v) / SQ2PI)
    return np.concatenate(vs), np.concatenate(ws)


def prox(x, chi):
    """argmin_v chi l(v) + (v - x)^2/2 for l(v) = log(1 + e^{-v}): v - x = chi f(-v) (Newton; the map is
    increasing and the iteration is monotone after the first step)."""
    v = np.array(x, dtype=float)
    for _ in range(60):
        s = special.expit(-v)
        step = (v - x - chi * s) / (1.0 + chi * s * (1.0 - s))
        v = v - step
        if np.max(np.abs(step) / (1.0 + np.abs(v))) < 1e-14:
            break
    return v


class System:
    def __init__(self, r):
        self.r = r
        V, W = v_nodes(r)
        self.V, self.Z = np.meshgrid(V, _xh, indexing="ij")
        self.W = np.outer(W, _wh)
        self.V1, self.W1 = V, W

    def residual(self, p, alpha):
        a, tau, chi = p
        v = prox(a * self.V + tau * self.Z, chi)
        lp = -special.expit(-v)                      # l'(v)
        lpp = special.expit(v) * special.expit(-v)   # l''(v)
        e1 = np.sum(self.W * self.V * lp)
        e2 = tau ** 2 - alpha * chi ** 2 * np.sum(self.W * lp ** 2)
        e3 = 1.0 - alpha * chi * np.sum(self.W * lpp / (1.0 + chi * lpp))
        return [e1, e2, e3]

    def solve(self, alpha, x0):
        sol = optimize.root(self.residual, x0, args=(alpha,), method="hybr", tol=1e-13)
        res = float(np.max(np.abs(self.residual(sol.x, alpha))))
        return sol.x, res

    def err(self, a, tau):
        """Test error P(a V + tau Z < 0) of the direction (a e* + tau e_perp)."""
        return float(np.sum(self.W1 * special.ndtr(-a * self.V1 / tau)))


def mle(X, y, b0=None, tol=1e-12, maxit=100):
    """Logistic MLE with labels y in {-1, +1} by Newton's method with backtracking."""
    Zm = y[:, None] * X
    b = np.zeros(X.shape[1]) if b0 is None else b0.copy()
    loss = lambda b: float(np.sum(np.logaddexp(0.0, -(Zm @ b))))
    f = loss(b)
    for _ in range(maxit):
        m = Zm @ b
        s = special.expit(-m)                  # -l'(m)
        grad = -(Zm.T @ s)
        w = s * (1.0 - s)
        H = (Zm * w[:, None]).T @ Zm
        step = np.linalg.solve(H, grad)
        dec = float(grad @ step)               # Newton decrement squared
        if dec < tol:
            break
        t = 1.0
        while True:
            bn = b - t * step
            fn = loss(bn)
            if fn <= f - 1e-4 * t * dec or t < 1e-10:
                break
            t *= 0.5
        b, f = bn, fn
    return b, f


def main():
    t0 = time.time()
    r = math.sqrt(5.0)
    S = System(r)
    Ef1 = float(np.sum(S.W1 * special.expit(r * S.V1) * special.expit(-r * S.V1)))
    print("r = sqrt(5): E f'(rU) = %.6f, Bayes error E f(-r|U|) = %.4f" % (
        Ef1, float(np.sum(S.W1 * special.expit(-r * np.abs(S.V1))))))
    (a, tau, chi), res = S.solve(10.0, [1.2 * r, 1.0, 1.0])
    print("check d/n = 0.1: alpha* = %.4f sigma* = %.4f lambda* = %.4f (paper 1.1678, 3.3466, 0.9605), res %.1e" % (
        a / r, tau * math.sqrt(10.0), chi, res))
    (a2, tau2, chi2), res2 = S.solve(5.0, [1.5 * r, 2.0, 3.0])
    fac = tau2 ** 2 / chi2
    print("check d/n = 0.2: alpha* = %.4f sigma* = %.4f lambda* = %.4f (paper 1.499, 4.744), LRT factor %.4f" % (
        a2 / r, tau2 * math.sqrt(5.0), chi2, fac))
    rows = []
    x = np.array([r, math.sqrt(0.01 / Ef1), 0.01 / Ef1])
    for k in np.linspace(0.01, 0.30, 30):
        x, res = S.solve(1.0 / k, x)
        a, tau, chi = x
        if res > 1e-9:
            print("d/n=%.3f: no reliable solution (residual %.1e)" % (k, res))
            break
        c = a / math.sqrt(a * a + tau * tau)
        rows.append((k, a / r, tau, chi, tau ** 2 / chi, c, S.err(a, tau), math.sqrt(k / Ef1)))
    rk.write_dat(os.path.join(DATA, "11-classification-mle-curve.dat"),
                 ["dn", "mu", "tau", "chi", "lrt", "cos", "err", "tauclassical"], rows)
    for row in rows:
        if any(abs(row[0] - k) < 1e-9 for k in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)):
            print("d/n=%.2f: mu=%.4f tau=%.4f chi=%.4f LRT=%.4f cos=%.4f Err=%.4f tau_classical=%.4f" % row)
    print("curve done (%.0f s)" % (time.time() - t0))

    n, reps = 1000, 40
    gen = rk.rng(11003)
    sim = []
    for k in (0.05, 0.10, 0.15, 0.20, 0.25):
        d = int(round(k * n))
        mus, taus, lrts = [], [], []
        for _ in range(reps):
            X = gen.standard_normal((n, d))
            y = np.where(gen.random(n) < special.expit(r * X[:, 0]), 1.0, -1.0)
            b, f = mle(X, y)
            mus.append(b[0] / r)
            taus.append(float(np.linalg.norm(b[1:])))
            if abs(k - 0.20) < 1e-9:
                for j in range(1, 11):                 # null coordinates 1..10 (beta*_j = 0)
                    keep = np.ones(d, bool)
                    keep[j] = False
                    _, f0 = mle(X[:, keep], y, b0=b[keep])
                    lrts.append(2.0 * (f0 - f))
        mu_m, mu_h = rk.mean_ci(mus)
        ta_m, ta_h = rk.mean_ci(taus)
        sim.append((k, mu_m, mu_h, ta_m, ta_h))
        print("sim d/n=%.2f (d=%d): mu_hat = %.4f +- %.4f, tau_hat = %.4f +- %.4f   (%.0f s)" % (
            k, d, mu_m, mu_h, ta_m, ta_h, time.time() - t0))
        if lrts:
            lr = np.array(lrts)
            m, hh = rk.mean_ci(lr)
            q95 = 3.841459
            kc = int(np.sum(lr > q95))
            kr = int(np.sum(lr > fac * q95))
            print("   LRT, %d null tests: mean 2 Lambda = %.3f +- %.3f (prediction %.4f, classical 1)" % (len(lr), m, hh, fac))
            print("   rejections at nominal 5%%: chi2_1 %d/%d = %.3f %s; rescaled %d/%d = %.3f %s" % (
                kc, len(lr), kc / len(lr), rk.clopper_pearson(kc, len(lr)), kr, len(lr), kr / len(lr),
                rk.clopper_pearson(kr, len(lr))))
    rk.write_dat(os.path.join(DATA, "11-classification-mle-sim.dat"), ["dn", "mu", "muh", "tau", "tauh"], sim)
    print("elapsed %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
