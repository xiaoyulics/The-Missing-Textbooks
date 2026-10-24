"""Chapter 10 (The LASSO and Robust Regression): the LASSO risk for a Bernoulli--Gaussian signal.

Scaling of Bayati--Montanari (the chapter's scaling): A in R^{n x d} with iid N(0, 1/n) entries,
y = A theta0 + w, w ~ N(0, sigma^2 I_n), alpha = n/d, LASSO
    theta_hat = argmin (1/2)|y - A theta|^2 + lam |theta|_1,
MSE = |theta_hat - theta0|^2 / d  (the book's excess risk is R = MSE/alpha).
Signal: theta0 iid (1 - eps) delta_0 + eps N(0, 1).  Parameters: alpha = 0.5, eps = 0.1, sigma^2 = 0.05.

The fixed point (Bayati--Montanari 2012, eqs. (1.6), (1.7), (1.15); threshold zeta = t tau):
    tau^2 = sigma^2 + (1/alpha) E[(eta(Theta0 + tau Z; t tau) - Theta0)^2],
    lam   = t tau (1 - (1/alpha) P(|Theta0 + tau Z| > t tau)),     MSE = alpha (tau^2 - sigma^2).

Computes and writes:
  10-lasso-risk-curve.dat : the prediction lam -> MSE and the fraction of nonzero coordinates;
  10-lasso-risk-sim.dat   : LASSO (coordinate descent, scikit-learn) at d = 1000, n = 500, 20 data
                            sets per lam (seeds 1000 + r), means and Student-t 95% intervals;
  10-lasso-risk-size.dat  : MSE at the optimal lam for d = 250 ... 4000 (finite-size trend);
  10-lasso-amp.dat        : AMP (Donoho--Maleki--Montanari) at d = 4000 against state evolution, and
                            the distance of the AMP iterate to the LASSO solution.
Also prints the four routes at the optimal lam: fixed point (replica = state evolution), the CGMT
scalar max-min, AMP after 30 iterations, and the LASSO itself.
Runtime: about one minute (single-threaded BLAS).
"""
import os
import sys
import time
import math
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                     # noqa: E402
from scipy import optimize, special    # noqa: E402
import rosettakit as rk                # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
ALPHA, EPS, SIGMA2 = 0.5, 0.1, 0.05
SQ2PI = math.sqrt(2 * math.pi)
Phi = special.ndtr


def phi(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / SQ2PI


def st_moments(mu, zeta, tau):
    """For U = mu + tau Z (mu scalar or array): E(eta(U; zeta) - mu)^2, P(|U| > zeta), E|eta(U; zeta)|,
    in closed form (eta = soft thresholding)."""
    mu = np.asarray(mu, dtype=float)
    zu, zl = (zeta - mu) / tau, (-zeta - mu) / tau
    Pu, Pl, pu, pl = Phi(zu), Phi(zl), phi(zu), phi(zl)
    up = zeta ** 2 * (1 - Pu) - 2 * zeta * tau * pu + tau ** 2 * ((1 - Pu) + zu * pu)
    lo = zeta ** 2 * Pl - 2 * zeta * tau * pl + tau ** 2 * (Pl - zl * pl)
    mse = up + lo + mu ** 2 * (Pu - Pl)
    a, b = (mu - zeta) / tau, (-mu - zeta) / tau
    eabs = (mu - zeta) * Phi(a) + tau * phi(a) + (-mu - zeta) * Phi(b) + tau * phi(b)
    return mse, (1 - Pu) + Pl, eabs


GHX, GHW = np.polynomial.hermite_e.hermegauss(160)
GHW = GHW / GHW.sum()


def moments(zeta, tau, eps=EPS):
    """E over Theta0 ~ (1-eps) delta_0 + eps N(0,1) of the three soft-thresholding moments."""
    m0, p0, a0 = st_moments(0.0, zeta, tau)
    m1, p1, a1 = st_moments(GHX, zeta, tau)
    return ((1 - eps) * float(m0) + eps * float(GHW @ m1), (1 - eps) * float(p0) + eps * float(GHW @ p1),
            (1 - eps) * float(a0) + eps * float(GHW @ a1))


def fixed_point(t, alpha=ALPHA, sigma2=SIGMA2):
    """The unique tau^2 for the multiplier t > t_min (BM Prop. 1.3), by bisection, and the calibration."""
    g = lambda s: sigma2 + moments(t * math.sqrt(s), math.sqrt(s))[0] / alpha - s
    hi = sigma2 + 1.0
    while g(hi) > 0:
        hi *= 2
    tau2 = rk.root(g, sigma2, hi)
    tau = math.sqrt(tau2)
    mse, P, eabs = moments(t * tau, tau)
    return dict(t=t, tau=tau, tau2=tau2, lam=t * tau * (1 - P / alpha), mse=alpha * (tau2 - sigma2), P=P,
                eabs=eabs, b=tau * (1 - P / alpha), zeta=t * tau)


def t_min(alpha=ALPHA):
    return rk.root(lambda t: (1 + t * t) * Phi(-t) - t * phi(t) - alpha / 2, 0.0, 20.0)


def solve_lam(lam):
    t0 = t_min()
    t = rk.root(lambda t: fixed_point(t)["lam"] - lam, t0 + 1e-6, 30.0)
    return fixed_point(t)


def psi(b, tau, lam, alpha=ALPHA, sigma2=SIGMA2):
    """The CGMT scalar function psi(b, tau) of the chapter (= Miolane--Montanari's eq. (8) up to a constant)."""
    zeta = lam * tau / b
    mse, P, eabs = moments(zeta, tau)
    e_u2 = mse - 2 * tau ** 2 * P + tau ** 2          # E (eta - Theta0 - tau Z)^2, by Stein's lemma
    q = (b / tau) * (0.5 * e_u2 + zeta * eabs) - b * tau / 2
    return b * sigma2 / (2 * tau) + b * tau / 2 - b * b / 2 + q / alpha


def cgmt_maxmin(lam):
    def inner(b):
        r = optimize.minimize_scalar(lambda tt: psi(b, tt, lam), bounds=(1e-3, 10), method="bounded",
                                     options={"xatol": 1e-12})
        return r.fun, r.x
    r = optimize.minimize_scalar(lambda b: -inner(b)[0], bounds=(1e-3, 5), method="bounded",
                                 options={"xatol": 1e-12})
    val, tau = inner(r.x)
    return r.x, tau, val


def instance(d, gen):
    n = int(round(ALPHA * d))
    A = gen.standard_normal((n, d)) / math.sqrt(n)
    theta0 = gen.standard_normal(d) * (gen.random(d) < EPS)
    y = A @ theta0 + math.sqrt(SIGMA2) * gen.standard_normal(n)
    return A, theta0, y


def lasso(A, y, lam, precompute=True, warm=None):
    from sklearn.linear_model import Lasso
    n = A.shape[0]
    model = Lasso(alpha=lam / n, fit_intercept=False, tol=1e-10, max_iter=200000, precompute=precompute,
                  warm_start=warm is not None, selection="cyclic")
    if warm is not None:
        model.coef_ = warm.copy()
    model.fit(A, y)
    return model.coef_


def amp(A, y, t, taus, iters):
    """AMP of Donoho--Maleki--Montanari with thresholds t * tau_s (tau_s from state evolution)."""
    n, d = A.shape
    alpha = n / d
    x = np.zeros(d)
    z = y.copy()
    out = []
    for s in range(iters):
        r = x + A.T @ z
        thr = t * taus[s]
        x_new = np.sign(r) * np.maximum(np.abs(r) - thr, 0.0)
        onsager = np.mean(np.abs(r) > thr) / alpha
        z = y - A @ x_new + onsager * z
        x = x_new
        out.append(x.copy())
    return out


def main():
    t_start = time.time()
    tm = t_min()
    print(f"t_min(alpha={ALPHA}) = {tm:.6f}  (BM Prop. 1.3)")
    # ---- prediction curve
    rows = []
    for t in np.linspace(tm + 0.05, 5.0, 400):
        fp = fixed_point(t)
        if fp["lam"] > 0.004:
            rows.append((fp["lam"], fp["mse"], fp["P"], t, fp["tau2"]))
    rows.sort()
    rk.write_dat(os.path.join(DATA, "10-lasso-risk-curve.dat"), ["lam", "mse", "nnz", "t", "tau2"],
                 [r for r in rows if r[0] <= 1.25])
    res = optimize.minimize_scalar(lambda t: fixed_point(t)["mse"], bounds=(tm + 0.02, 4.0), method="bounded",
                                   options={"xatol": 1e-10})
    opt = fixed_point(res.x)
    lam_opt = opt["lam"]
    print(f"optimum: t* = {opt['t']:.4f}, lambda* = {lam_opt:.5f}, MSE* = {opt['mse']:.6f}, "
          f"R* = MSE*/alpha = {opt['mse'] / ALPHA:.6f}, nonzeros/d = {opt['P']:.4f}, tau*^2 = {opt['tau2']:.6f}, "
          f"b* = {opt['b']:.6f}, zeta* = {opt['zeta']:.6f}")
    # ---- CGMT route (independent numerical route)
    for lam in (0.1, lam_opt):
        fp = solve_lam(lam)
        b, tau, val = cgmt_maxmin(lam)
        jstar = fp["b"] ** 2 / 2 + lam / ALPHA * fp["eabs"]
        print(f"lam = {lam:.5f}: fixed point tau* = {fp['tau']:.8f}, b* = {fp['b']:.8f}, MSE = {fp['mse']:.8f}; "
              f"CGMT max-min: tau = {tau:.8f}, b = {b:.8f}, MSE = {ALPHA * (tau ** 2 - SIGMA2):.8f}; "
              f"value {val:.10f} vs b*^2/2 + (lam/alpha)E|X*| = {jstar:.10f}")
    # ---- simulation at d = 1000 over a grid of lambda (common random numbers across lambda)
    lams = [0.03, 0.05, 0.1, 0.15, lam_opt, 0.3, 0.4, 0.6, 0.8, 1.0]
    reps, d = 20, 1000
    mses = np.zeros((len(lams), reps))
    nnzs = np.zeros((len(lams), reps))
    for r in range(reps):
        A, theta0, y = instance(d, rk.rng(1000 + r))
        warm = None
        for i, lam in enumerate(sorted(lams, reverse=True)):
            th = lasso(A, y, lam, warm=warm)
            warm = th
            j = lams.index(lam)
            mses[j, r] = np.sum((th - theta0) ** 2) / d
            nnzs[j, r] = np.mean(th != 0)
    sim_rows = []
    print(f"\nsimulation d = {d}, n = {int(ALPHA * d)}, {reps} data sets per lambda (seeds 1000..{1000 + reps - 1})")
    for j, lam in enumerate(lams):
        fp = solve_lam(lam)
        m, h = rk.mean_ci(mses[j])
        mn, hn = rk.mean_ci(nnzs[j])
        sim_rows.append((lam, m, h, mn, hn, fp["mse"], fp["P"]))
        print(f"  lam = {lam:.4f}: MSE theory {fp['mse']:.5f}, simulated {m:.5f} +- {h:.5f}; "
              f"nonzeros theory {fp['P']:.4f}, simulated {mn:.4f} +- {hn:.4f}")
    rk.write_dat(os.path.join(DATA, "10-lasso-risk-sim.dat"),
                 ["lam", "mse", "ci", "nnz", "nnzci", "msetheory", "nnztheory"], sim_rows)
    # ---- finite-size trend at the optimal lambda
    size_rows = []
    print("\nfinite-size trend at lambda* (seeds 2000 + 100 d_index + r)")
    for k, (d, reps) in enumerate([(250, 80), (500, 40), (1000, 20), (2000, 10), (4000, 5)]):
        vals = []
        for r in range(reps):
            A, theta0, y = instance(d, rk.rng(2000 + 100 * k + r))
            th = lasso(A, y, lam_opt, precompute=(d <= 2000))
            vals.append(np.sum((th - theta0) ** 2) / d)
        m, h = rk.mean_ci(vals)
        size_rows.append((d, m, h, opt["mse"]))
        print(f"  d = {d}: MSE = {m:.5f} +- {h:.5f} ({reps} data sets), prediction {opt['mse']:.5f}")
    rk.write_dat(os.path.join(DATA, "10-lasso-risk-size.dat"), ["d", "mse", "ci", "theory"], size_rows)
    # ---- AMP against state evolution, d = 4000, at the calibrated multiplier t*
    iters, d, seeds = 30, 4000, 5
    taus = [math.sqrt(SIGMA2 + EPS / ALPHA)]           # tau_0^2 = sigma^2 + E Theta0^2 / alpha
    for s in range(iters):
        tau = taus[-1]
        taus.append(math.sqrt(SIGMA2 + moments(opt["t"] * tau, tau)[0] / ALPHA))
    se_mse = [ALPHA * (taus[s + 1] ** 2 - SIGMA2) for s in range(iters)]
    amp_mse = np.zeros((seeds, iters))
    amp_dist = np.zeros((seeds, iters))
    lasso_mse, lam_amp, dist_amp_own = [], [], []
    for sd in range(seeds):
        A, theta0, y = instance(d, rk.rng(3000 + sd))
        xs = amp(A, y, opt["t"], taus, iters)
        th = lasso(A, y, lam_opt, precompute=False, warm=xs[-1])
        lasso_mse.append(np.sum((th - theta0) ** 2) / d)
        # the AMP iterate is (nearly) a LASSO solution for the data-dependent lambda of BM's Lemma 1.2
        thr = opt["t"] * taus[iters - 1]
        frac = np.mean(xs[-1] != 0)
        lam_a = thr * (1 - frac / ALPHA)
        lam_amp.append(lam_a)
        th_a = lasso(A, y, lam_a, precompute=False, warm=xs[-1])
        dist_amp_own.append(np.sum((xs[-1] - th_a) ** 2) / d)
        for s in range(iters):
            amp_mse[sd, s] = np.sum((xs[s] - theta0) ** 2) / d
            amp_dist[sd, s] = np.sum((xs[s] - th) ** 2) / d
    amp_rows = []
    for s in range(iters):
        m, h = rk.mean_ci(amp_mse[:, s])
        md, hd = rk.mean_ci(amp_dist[:, s])
        amp_rows.append((s + 1, se_mse[s], m, h, max(md, 1e-12), hd))
    rk.write_dat(os.path.join(DATA, "10-lasso-amp.dat"), ["iter", "se", "amp", "ci", "dist", "distci"], amp_rows)
    m_l, h_l = rk.mean_ci(lasso_mse)
    print(f"\nAMP, d = {d}, n = {int(ALPHA * d)}, {seeds} data sets (seeds 3000..{3000 + seeds - 1}), t = t*:")
    for s in (0, 1, 2, 4, 9, 19, 29):
        print(f"  iteration {s + 1:2d}: state evolution {se_mse[s]:.5f}, AMP {amp_rows[s][2]:.5f} +- {amp_rows[s][3]:.5f}, "
              f"|x_t - theta_hat|^2/d = {amp_rows[s][4]:.2e}")
    print(f"  LASSO on the same data: MSE = {m_l:.5f} +- {h_l:.5f};  fixed point {opt['mse']:.5f}")
    print(f"  data-dependent lambda of the AMP fixed point (BM Lemma 1.2): {np.round(lam_amp, 5)} "
          f"(relative deviation from lambda* up to {np.max(np.abs(np.array(lam_amp) / lam_opt - 1)):.4f}); "
          f"|x_30 - LASSO(lambda_AMP)|^2/d = {np.round(dist_amp_own, 12)}")
    print(f"\nfour routes at lambda* = {lam_opt:.5f}: fixed point {opt['mse']:.6f}; CGMT max-min (above); "
          f"AMP (30 iterations, d = 4000) {amp_rows[-1][2]:.5f} +- {amp_rows[-1][3]:.5f}; LASSO (d = 4000) {m_l:.5f} +- {h_l:.5f}")
    print("runtime %.1f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
