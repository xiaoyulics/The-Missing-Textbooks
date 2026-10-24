"""Chapter 14 (dmft): gradient flow on least squares against its DMFT limit (Theorem 14.1).

Model (book's regression normalization): x_i ~ N(0, I_d), y = X theta* + sigma z, |theta*| = r, theta_0 = 0,
gradient flow on Lhat(theta) = |y - X theta|^2 / (2n); S = X^T X / n, gamma = d/n.
  excess risk   E(t) = r^2 int e^{-2 t l} dmu + sigma^2 gamma int (1 - e^{-t l})^2 / l dmu
  training err  L(t) = (r^2/2) int l e^{-2 t l} dmu + (sigma^2/2) int e^{-2 t l} dmu_under
  response      R(tau) = int e^{-tau l} dmu   (mu = Marchenko--Pastur law of S; mu_under = gamma mu + (1-gamma) delta_0)
The response is also obtained by integrating the DMFT kernel equations in time (Computation 14.2):
  R'(tau) = -R(tau) - int_0^tau Rl(tau - s) R(s) ds,   Rl(tau) = -gamma R(tau) - gamma int_0^tau R(tau - s) Rl(s) ds.
Simulation: n = 2000, d = 1000 (gamma = 0.5) and n = 1000, d = 1500 (gamma = 1.5), sigma = 0.5, r = 1,
10 independent replicates each (seeds 1400 + k), exact flow through the eigendecomposition of S.
Writes figures/data/14-dmft-ls-curves.dat, 14-dmft-ls-sim.dat, 14-dmft-ls-response.dat.
"""
import math
import os
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "2")   # the machine is shared: two BLAS threads avoid oversubscription
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
SIGMA, R2 = 0.5, 1.0
GAMMAS = (0.5, 1.5)
SIZES = {0.5: (2000, 1000), 1.5: (1000, 1500)}
REPS = 10

# ---------------------------------------------------------------------------------------------
# Marchenko--Pastur integrals by the substitution l = c + w cos(phi), which removes the square roots
# ---------------------------------------------------------------------------------------------
_PHI, _WPHI = np.polynomial.legendre.leggauss(400)
_PHI = 0.5 * math.pi * (_PHI + 1.0)
_WPHI = 0.5 * math.pi * _WPHI


def mp_integral(f, g):
    """int f(l) mu_g(dl) for the Marchenko--Pastur law of X^T X/n with ratio g = d/n (atom 1 - 1/g at 0 if g > 1).
    f must accept arrays; its value at l = 0 enters only through the atom."""
    a, b = (1 - math.sqrt(g)) ** 2, (1 + math.sqrt(g)) ** 2
    c, w = 0.5 * (a + b), 0.5 * (b - a)
    lam = c + w * np.cos(_PHI)
    dens = (w * np.sin(_PHI)) ** 2 / (2 * math.pi * g * lam)   # density times dl/dphi
    val = float(np.sum(_WPHI * dens * f(lam)))
    if g > 1:
        val += (1 - 1 / g) * float(f(np.array([0.0]))[0])
    return val


def risk(t, g):
    sig = mp_integral(lambda l: np.exp(-2 * t * l), g)
    noi = mp_integral(lambda l: np.where(l > 0, (1 - np.exp(-t * l)) ** 2 / np.maximum(l, 1e-300), 0.0), g)
    return R2 * sig + SIGMA ** 2 * g * noi


def risk_inf(g):
    noi = mp_integral(lambda l: np.where(l > 0, 1.0 / np.maximum(l, 1e-300), 0.0), g)
    return R2 * mp_integral(lambda l: (l == 0).astype(float), g) + SIGMA ** 2 * g * noi


def train(t, g):
    sig = mp_integral(lambda l: l * np.exp(-2 * t * l), g)
    noi = g * mp_integral(lambda l: np.exp(-2 * t * l), g) + (1 - g)   # int e^{-2tl} d mu_under
    return 0.5 * R2 * sig + 0.5 * SIGMA ** 2 * noi


def response(tau, g):
    return mp_integral(lambda l: np.exp(-tau * l), g)


def dmft_response(g, h, n):
    """Integrate the DMFT kernel equations by the trapezoidal rule (each step is a 2x2 linear system)."""
    Rt = np.zeros(n + 1)
    Rl = np.zeros(n + 1)
    Rt[0], Rl[0] = 1.0, -g
    for k in range(1, n + 1):
        # convolutions with the unknown endpoint values removed
        s_l = h * np.dot(Rt[k - 1:0:-1], Rl[1:k]) if k > 1 else 0.0        # sum_{j=1}^{k-1} Rt(k-j) Rl(j)
        s_t = h * np.dot(Rl[k - 1:0:-1], Rt[1:k]) if k > 1 else 0.0        # sum_{j=1}^{k-1} Rl(k-j) Rt(j)
        f_prev = -Rt[k - 1] - (h * (0.5 * Rl[k - 1] * Rt[0] + 0.5 * Rl[0] * Rt[k - 1]
                                    + (np.dot(Rl[k - 2:0:-1], Rt[1:k - 1]) if k > 2 else 0.0)) if k > 1 else 0.0)
        # unknowns x = Rt[k], y = Rl[k]:
        # y = -g x - g (h/2 x Rl[0] + s_l + h/2 Rt[0] y)
        # x = Rt[k-1] + h/2 (f_prev + f_k),  f_k = -x - (h/2 y Rt[0] + s_t + h/2 Rl[0] x)
        A = np.array([[g + g * 0.5 * h * Rl[0], 1 + g * 0.5 * h * Rt[0]],
                      [1 + 0.5 * h * (1 + 0.5 * h * Rl[0]), 0.5 * h * 0.5 * h * Rt[0]]])
        rhs = np.array([-g * s_l, Rt[k - 1] + 0.5 * h * (f_prev - s_t)])
        Rt[k], Rl[k] = np.linalg.solve(A, rhs)
    return Rt, Rl


def simulate(g, seed, ts, taus):
    n, d = SIZES[g]
    gen = rk.rng(seed)
    X = gen.standard_normal((n, d))
    th = gen.standard_normal(d)
    th *= math.sqrt(R2) / np.linalg.norm(th)
    y = X @ th + SIGMA * gen.standard_normal(n)
    lam, U = np.linalg.eigh(X.T @ X / n)
    lam = np.maximum(lam, 0.0)
    c = U.T @ (X.T @ y / n)
    out_r, out_l = [], []
    for t in ts:
        gt = np.where(lam > 1e-10, -np.expm1(-t * lam) / np.maximum(lam, 1e-300), t)
        theta = U @ (gt * c)
        out_r.append(float(np.sum((theta - th) ** 2)))
        out_l.append(float(np.sum((y - X @ theta) ** 2)) / (2 * n))
    resp = [float(np.mean(np.exp(-tau * lam))) for tau in taus]
    return np.array(out_r), np.array(out_l), np.array(resp)


def main():
    t_start = time.time()
    # theory curves on a log grid of times
    tgrid = np.logspace(-2, 2, 161)
    rows = []
    for t in tgrid:
        rows.append([t] + [f(t, g) for g in GAMMAS for f in (risk, train)])
    rk.write_dat(os.path.join(DATA, "14-dmft-ls-curves.dat"),
                 ["t", "risk05", "train05", "risk15", "train15"], rows)
    from scipy.optimize import minimize_scalar
    for g in GAMMAS:
        alpha = 1 / g
        best = minimize_scalar(lambda u: risk(math.exp(u), g), bounds=(-3, 5), method="bounded",
                               options={"xatol": 1e-10})
        print("gamma=%.1f: E(inf)=%.6f (ridgeless formula %.6f), L(inf)=%.6f, min_t E(t)=%.6f at t=%.4f, E(0)=%.4f"
              % (g, risk_inf(g), rk.ridgeless_risk(alpha, R2, SIGMA ** 2), train(1e4, g),
                 best.fun, math.exp(best.x), risk(0.0, g)))
    # exercise checks at the interpolation threshold gamma = 1
    c_ex = 2 * (2 - math.sqrt(2)) * SIGMA ** 2 / math.sqrt(math.pi)
    for t in (1e2, 1e3, 1e4):
        print("gamma=1: E(t)/sqrt(t) at t=%g: %.6f (2(2-sqrt2) sigma^2/sqrt(pi) = %.6f); R(t) sqrt(pi t) = %.5f"
              % (t, risk(t, 1.0) / math.sqrt(t), c_ex, response(t, 1.0) * math.sqrt(math.pi * t)))
    print("E'(0) by finite difference (gamma=0.5): %.6f (claim -2 r^2 = %.1f)"
          % ((risk(1e-6, 0.5) - risk(0.0, 0.5)) / 1e-6, -2 * R2))
    # DMFT response by time stepping, against the Laplace transform of Marchenko--Pastur
    h, tmax = 0.01, 10.0
    nstep = int(round(tmax / h))
    taus_plot = np.linspace(0, tmax, 101)
    resp_rows = []
    dm = {g: dmft_response(g, h, nstep)[0] for g in GAMMAS}
    for g in GAMMAS:
        dmh = dmft_response(g, 2 * h, nstep // 2)[0]
        err = max(abs(dm[g][int(round(tau / h))] - response(tau, g)) for tau in taus_plot)
        err2 = max(abs(dmh[int(round(tau / (2 * h)))] - response(tau, g)) for tau in taus_plot)
        # Laplace transform check: int_0^inf e^{-p tau} R = m(-p), the positive root of the MP equation
        pchk = 1.0
        lap = mp_integral(lambda l: 1.0 / (l + pchk), g)
        print("gamma=%.1f: DMFT response vs MP Laplace transform: max error %.2e (h=%.3f), %.2e (h=%.3f); "
              "R(inf)=%.4f; m(-1)=%.6f (kit %.6f)" % (g, err, h, err2, 2 * h, response(1e6, g), lap,
                                                      rk.mp_stieltjes(-pchk, g)))
    # simulations
    ts_sim = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
    taus_sim = np.array([0.0, 0.5, 1, 2, 3, 4, 6, 8, 10])
    sim_rows = [[t] for t in ts_sim]
    resp_meas = {}
    for g in GAMMAS:
        rs, ls, rp = [], [], []
        for k in range(REPS):
            r_, l_, p_ = simulate(g, 1400 + 100 * int(10 * g) + k, ts_sim, taus_sim)
            rs.append(r_)
            ls.append(l_)
            rp.append(p_)
        rs, ls, rp = np.array(rs), np.array(ls), np.array(rp)
        resp_meas[g] = rp.mean(axis=0)
        for i, t in enumerate(ts_sim):
            mr, hr = rk.mean_ci(rs[:, i])
            ml, hl = rk.mean_ci(ls[:, i])
            sim_rows[i] += [mr, hr, ml, hl]
            if t in (1, 5, 100):
                print("gamma=%.1f t=%g: risk %.4f +- %.4f (theory %.4f), train %.4f +- %.4f (theory %.4f)"
                      % (g, t, mr, hr, risk(t, g), ml, hl, train(t, g)))
        within = sum(abs(rs[:, i].mean() - risk(t, g)) <= rk.mean_ci(rs[:, i])[1] for i, t in enumerate(ts_sim))
        within_l = sum(abs(ls[:, i].mean() - train(t, g)) <= rk.mean_ci(ls[:, i])[1] for i, t in enumerate(ts_sim))
        print("gamma=%.1f: %d of %d risk intervals and %d of %d training intervals contain the limit"
              % (g, within, len(ts_sim), within_l, len(ts_sim)))
    rk.write_dat(os.path.join(DATA, "14-dmft-ls-sim.dat"),
                 ["t", "risk05", "risk05ci", "train05", "train05ci", "risk15", "risk15ci", "train15", "train15ci"],
                 sim_rows)
    for tau in taus_plot:
        k = int(round(tau / h))
        resp_rows.append([tau, response(tau, 0.5), dm[0.5][k], response(tau, 1.5), dm[1.5][k]])
    rk.write_dat(os.path.join(DATA, "14-dmft-ls-response.dat"), ["tau", "mp05", "dmft05", "mp15", "dmft15"], resp_rows)
    rk.write_dat(os.path.join(DATA, "14-dmft-ls-response-sim.dat"), ["tau", "meas05", "meas15"],
                 [[tau, resp_meas[0.5][i], resp_meas[1.5][i]] for i, tau in enumerate(taus_sim)])
    for g in GAMMAS:
        print("gamma=%.1f: measured response at tau=2: %.5f, MP %.5f" % (g, resp_meas[g][3], response(2.0, g)))
    print("runtime %.1f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
