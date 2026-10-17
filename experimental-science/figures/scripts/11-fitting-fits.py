"""Lab 11.2: fitting the parametric law to the runs of Lab 11.1, with Hoffmann et al.'s recipe
and with variants; bootstrap and delta-method intervals; an isoFLOP analysis of the same runs.

The law is L(N, D) = E + A_N N^-alN + A_D D^-alD, parametrized as Hoffmann et al. do:
log L = LSE(a - alN log N, b - alD log D, e), with A_N = exp(a), A_D = exp(b), E = exp(e).
The recipe minimizes sum_i Huber_delta(log Lhat_i - log L_i), delta = 1e-3, by L-BFGS-B started
from a grid of 4500 points (Hoffmann's grid with e and a, b shifted to our loss scale), run to
convergence. Variants: the same objective averaged over runs with scipy's default tolerances;
the optimizer stopped after a few iterations; single starts; other delta; least squares on
log loss and on loss; E fixed at its known value 0.01. Uncertainty: a bootstrap over runs
(each resample refitted to convergence) and the delta method; and the 80%-subsample scheme of
Hoffmann et al.'s Table 2 applied to the converged and to the early-stopped fits.

Reads figures/data/11-fitting-runs.dat; writes figures/data/11-fitting-*.dat.
Run from the volume root: python3 figures/scripts/11-fitting-fits.py   (about 2-3 minutes)
"""
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"          # single-threaded BLAS: many tiny fits, a shared machine
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.stats import norm, t as student_t
import labkit as lk

T0 = time.time()
DATA = "figures/data/11-fitting-runs.dat"
TRUE_E = 0.01
TIGHT = dict(ftol=1e-15, gtol=1e-11, maxiter=20000, maxfun=40000)
DEFAULT = dict()                  # scipy's defaults: ftol 2.2e-9, gtol 1e-5, maxiter 15000


# ----------------------------------------------------------------------------------------
# the objective
# ----------------------------------------------------------------------------------------
def objective(lnN, lnD, lnL, delta=1e-3, mean=False, space="log", fix_e=None):
    """Return f(theta) -> (value, gradient) for theta = (a, b, e, alN, alD).
    space='log': residual log Lhat - log L; space='loss': residual Lhat - L.
    delta=None: least squares (0.5 r^2). fix_e: e held fixed (theta has 4 entries)."""
    n = len(lnL)
    L = np.exp(lnL)
    scale = 1.0 / n if mean else 1.0

    def f(theta):
        with np.errstate(over="ignore", invalid="ignore"):
            val, g = _f(theta)
        if not np.isfinite(val) or not np.all(np.isfinite(g)):
            return 1e30, np.zeros_like(np.asarray(theta, float))
        return val, g

    def _f(theta):
        if fix_e is None:
            a, b, e, al, be = theta
        else:
            a, b, al, be = theta
            e = fix_e
        z0, z1 = a - al * lnN, b - be * lnD
        m = np.maximum(np.maximum(z0, z1), e)
        e0, e1, e2 = np.exp(z0 - m), np.exp(z1 - m), np.exp(e - m)
        s = e0 + e1 + e2
        lse = m + np.log(s)
        w0, w1, w2 = e0 / s, e1 / s, e2 / s
        if space == "log":
            r = lse - lnL
            jac = 1.0
        else:
            pred = np.exp(lse)
            r = pred - L
            jac = pred
        if delta is None:
            val = 0.5 * np.dot(r, r)
            psi = r
        else:
            ar = np.abs(r)
            val = np.sum(np.where(ar <= delta, 0.5 * r * r, delta * (ar - 0.5 * delta)))
            psi = np.clip(r, -delta, delta)
        p = psi * jac
        g0, g1, g2 = np.dot(p, w0), np.dot(p, w1), np.dot(p, w2)
        gal, gbe = -np.dot(p * w0, lnN), -np.dot(p * w1, lnD)
        g = np.array([g0, g1, g2, gal, gbe]) if fix_e is None else np.array([g0, g1, gal, gbe])
        return val * scale, g * scale

    return f


def fit(f, x0, opts=TIGHT, callback=None):
    return minimize(f, np.asarray(x0, float), jac=True, method="L-BFGS-B", options=opts,
                    callback=callback)


def hoffmann_grid(shift_e=-4.0, shift_ab=-5.0):
    """Hoffmann et al.'s grid: alN, alD in {0,...,2} step 0.5; e in {-1,...,1} step 0.5;
    a, b in {0,...,25} step 5; here e shifted by -4 and a, b by -5 to our loss scale."""
    pts = []
    for al in np.arange(0, 2.01, 0.5):
        for be in np.arange(0, 2.01, 0.5):
            for e in np.arange(-1, 1.01, 0.5) + shift_e:
                for a in np.arange(0, 25.1, 5) + shift_ab:
                    for b in np.arange(0, 25.1, 5) + shift_ab:
                        pts.append((a, b, e, al, be))
    return np.array(pts)


def small_grid():
    pts = []
    for al in (0.5, 1.0, 1.5):
        for be in (0.5, 1.0, 1.5):
            for e in (-5.0, -4.0, -3.0):
                for a in (-5.0, 5.0, 15.0):
                    for b in (-5.0, 5.0, 15.0):
                        pts.append((a, b, e, al, be))
    return np.array(pts)


def best_of(f, starts, opts=TIGHT, keep=False):
    """Fit from every start and return the end point with the lowest objective. The objective
    is evaluated at the returned point: after an abnormal termination of the line search,
    scipy's r.fun can belong to a rejected trial point rather than to r.x."""
    best, best_f, rows = None, np.inf, []
    for x0 in starts:
        r = fit(f, x0, opts)
        fx = f(r.x)[0] if np.all(np.isfinite(r.x)) else np.inf
        if keep:
            rows.append((fx, *r.x, r.nit, r.status))
        if np.isfinite(fx) and fx < best_f:
            best, best_f = r, fx
    if best is not None:
        best.fun = best_f
    return (best, np.array(rows)) if keep else best


def alloc(theta):
    al, be = theta[3], theta[4]
    return be / (al + be)


def describe(theta):
    a, b, e, al, be = theta
    return (f"E={np.exp(e):.5f} A_N={np.exp(a):.4g} A_D={np.exp(b):.4g} "
            f"alN={al:.4f} alD={be:.4f} a={be / (al + be):.4f}")


def law(theta, N, D):
    a, b, e, al, be = theta
    return np.exp(e) + np.exp(a) * N ** (-al) + np.exp(b) * D ** (-be)


def n_opt(theta, C):
    a, b, e, al, be = theta
    G = (al * np.exp(a) / (be * np.exp(b))) ** (1.0 / (al + be))
    return G * (C / 6.0) ** (be / (al + be))


# ----------------------------------------------------------------------------------------
# numbers quoted in the post-mortem (no runs needed)
# ----------------------------------------------------------------------------------------
def published_numbers():
    print("== published laws at Gopher's budget, C = 5.76e23 FLOPs ==")
    laws = {"Hoffmann, rounded (E 1.69, A 406.4, B 410.7, 0.34, 0.28)": (1.69, 406.4, 410.7, 0.34, 0.28),
            "Hoffmann, TeX source (1.6934, 406.4, 410.7, 0.3392, 0.2849)": (1.6934, 406.4, 410.7, 0.3392, 0.2849),
            "Besiroglu et al. (1.8172, 482.01, 2085.43, 0.3478, 0.3658)": (1.8172, 482.01, 2085.43, 0.3478, 0.3658)}
    C = 5.76e23
    for name, (E, A, B, al, be) in laws.items():
        th = (np.log(A), np.log(B), np.log(E), al, be)
        N = n_opt(th, C)
        D = C / 6.0 / N
        print(f"  {name}: a={be / (al + be):.4f}  N_opt={N / 1e9:.1f}B  D_opt={D / 1e12:.2f}T  D/N={D / N:.1f}")
    print(f"  rounding beta 0.2849 -> 0.28 at D=1e11 multiplies B/D^beta by {(1e11) ** 0.0049:.3f}")
    # delta method with Besiroglu et al.'s standard errors (0.02 each, correlation unknown)
    al, be, s = 0.3478, 0.3658, 0.02
    a = be / (al + be)
    for rho in (0.0, 0.5):
        v = (s / al) ** 2 + (s / be) ** 2 - 2 * rho * (s / al) * (s / be)
        print(f"  delta method for a with SE 0.02 on each exponent, rho={rho}: SE(a)={a * (1 - a) * np.sqrt(v):.4f}")
    width = 2 * norm.ppf(0.9) * 0.018          # 80% interval from Besiroglu et al.'s SE(a) = 0.018
    print(f"  80% interval from SE(a)=0.018: width {width:.4f} = {width / 0.001:.1f} x the published 0.001; "
          f"runs needed for 0.001: 240 x {(width / 0.001) ** 2:.0f} = {240 * (width / 0.001) ** 2:,.0f}")
    hw = (0.455 - 0.454) / 2
    se = hw / norm.ppf(0.9)
    print(f"  Hoffmann's 10-90% interval (0.454, 0.455): SE(a) about {se:.5f}; "
          f"log-ratio of exponents to {se / (0.4565 * 0.5435) * 100:.2f}%")
    # efficiency of the Huber location estimate at the normal model
    for k in (1.345, 0.1, 1e-3):
        num = (2 * norm.cdf(k) - 1) ** 2
        den = (2 * norm.cdf(k) - 1) - 2 * k * norm.pdf(k) + 2 * k * k * (1 - norm.cdf(k))
        print(f"  Huber efficiency at the normal, k={k:g}: {num / den:.4f}  (2/pi = {2 / np.pi:.4f})")


# ----------------------------------------------------------------------------------------
def main():
    published_numbers()
    d = np.genfromtxt(DATA, names=True)
    N, D, L, logK, seed = d["N"], d["D"], d["loss"], d["logK"], d["seed"]
    lnN, lnD, lnL = np.log(N), np.log(D), np.log(L)
    n = len(L)
    print(f"\n== {n} runs, {len(set(zip(N, D)))} cells, compute {6 * (N * D).min():.3g} to {6 * (N * D).max():.3g} ==")

    # pooled replicate sd of log loss (seeds within a cell)
    cells = sorted(set(zip(logK, N, D)))
    sds = [np.std(lnL[(logK == k) & (N == nn) & (D == dd)], ddof=1) for k, nn, dd in cells]
    sigma = float(np.sqrt(np.mean(np.square(sds))))
    print(f"pooled within-cell sd of log loss: {sigma:.4f} (median {np.median(sds):.4f})")
    med_sd = [np.median([s for (k, _, _), s in zip(cells, sds) if k == kk]) for kk in sorted(set(logK))]
    print("median within-cell sd of log loss by slice, smallest to largest budget: "
          + " ".join(f"{v:.4f}" for v in med_sd) + f" (ratio {med_sd[0] / med_sd[-1]:.1f})")
    worst = max(cells, key=lambda c: np.ptp(lnL[(logK == c[0]) & (N == c[1]) & (D == c[2])]))
    Lw = np.sort(L[(logK == worst[0]) & (N == worst[1]) & (D == worst[2])])
    print(f"widest cell: N={worst[1]:.0f}, D={worst[2]:.0f} ({worst[2] / 64:.0f} steps), losses {np.round(Lw, 4)}; "
          f"the highest is {Lw[-1] / Lw[:-1].mean():.2f} times the mean of the others")

    # ---- the recipe ---------------------------------------------------------------------
    grid = hoffmann_grid()
    f_sum = objective(lnN, lnD, lnL, 1e-3)
    t0 = time.time()
    rec, starts = best_of(f_sum, grid, TIGHT, keep=True)
    th = rec.x
    print(f"\n== recipe (sum, delta 1e-3, {len(grid)} starts, converged) [{time.time() - t0:.0f} s] ==")
    print(f"  {describe(th)}  objective {rec.fun:.6g}  nit {rec.nit}  {rec.message}")
    fvals, avals = starts[:, 0], starts[:, 5] / (starts[:, 4] + starts[:, 5])
    rel = (fvals - rec.fun) / rec.fun
    at_best = rel < 1e-6
    print(f"  starts reaching the best objective (rel. 1e-6): {at_best.mean():.3f}; "
          f"within 1%: {(rel < 0.01).mean():.3f}")
    other = ~at_best & np.isfinite(avals)
    if other.any():
        print(f"  other end points: a from {np.nanmin(avals[other]):.3f} to {np.nanmax(avals[other]):.3f}; "
              f"objective excess median {np.median(rel[other]):.3g}")
    # distinct end points among converged starts
    keyv = np.round(fvals / rec.fun, 4)
    vals, counts = np.unique(keyv[np.isfinite(keyv)], return_counts=True)
    order = np.argsort(-counts)[:6]
    print("  most common end points (objective / best, count, a): " + "; ".join(
        f"{vals[i]:.4f} x{counts[i]} a={np.nanmedian(avals[keyv == vals[i]]):.3f}" for i in order))
    status = starts[:, 7].astype(int)
    gabs = np.array([np.max(np.abs(f_sum(x)[1])) for x in starts[:, 1:6]])
    print(f"  termination: {np.sum(status == 0)} starts on a tolerance test, {np.sum(status == 2)} on a failed "
          f"line search (largest gradient component there {gabs[status == 2].max():.1e}), "
          f"{np.sum(status == 1)} on the iteration limit; largest gradient component over all end points "
          f"{gabs.max():.1e}")
    # which part of the law has vanished over the data at the end points not at the optimum
    tN = np.array([np.max(np.exp(x[0] - x[3] * lnN)) for x in starts[:, 1:6]]) / L.min()
    tD = np.array([np.max(np.exp(x[1] - x[4] * lnD)) for x in starts[:, 1:6]]) / L.min()
    tE = np.exp(starts[:, 3]) / L.min()
    gone = lambda t: t < 1e-4            # below 1e-4 of the smallest loss at every run
    kinds = {"the floor E has": gone(tE) & ~gone(tN) & ~gone(tD), "the data term has": gone(tD) & ~gone(tN),
             "the size term has": gone(tN) & ~gone(tD), "both power-law terms have": gone(tN) & gone(tD)}
    for nm, sel in kinds.items():
        sel = sel & ~at_best
        if sel.any():
            print(f"  end points where {nm} vanished over the data: {sel.sum()}, objective / best "
                  f"{fvals[sel].min() / rec.fun:.3f} to {fvals[sel].max() / rec.fun:.3f}, "
                  f"a from {avals[sel].min():.3f} to {avals[sel].max():.3f}")
    print(f"  end points off the optimum with every part of the law present: "
          f"{np.sum(~at_best & ~np.any(list(kinds.values()), axis=0))}")
    gnorm = np.max(np.abs(f_sum(th)[1]))
    print(f"  largest gradient component at the recipe's end point: {gnorm:.2e}")
    r_res = law(th, N, D)
    resid = np.log(r_res) - lnL
    print(f"  residuals of log loss: median |r| {np.median(np.abs(resid)):.4f}; "
          f"fraction with |r| > delta: {(np.abs(resid) > 1e-3).mean():.3f}")
    print(f"  E: fitted {np.exp(th[2]):.5f} against the true 0.01")

    # ---- averaging instead of summing, scipy's default tolerances -------------------------
    f_mean = objective(lnN, lnD, lnL, 1e-3, mean=True)
    t0 = time.time()
    mean_fit, mstarts = best_of(f_mean, grid, DEFAULT, keep=True)
    thm = mean_fit.x
    print(f"\n== averaged objective, default tolerances [{time.time() - t0:.0f} s] ==")
    print(f"  {describe(thm)}  nit {mean_fit.nit}  {mean_fit.message}")
    print(f"  its value of the summed objective: {f_sum(thm)[0]:.6g} vs converged {rec.fun:.6g} "
          f"(relative excess {(f_sum(thm)[0] - rec.fun) / rec.fun:.1e})")
    print(f"  median iterations over starts: averaged {np.median(mstarts[:, 6]):.0f}, "
          f"summed and converged {np.median(starts[:, 6]):.0f}")
    mf = np.array([f_sum(x)[0] if np.all(np.isfinite(x)) else np.nan for x in mstarts[:, 1:6]])
    mrel = (mf - rec.fun) / rec.fun
    ma = mstarts[:, 5] / (mstarts[:, 4] + mstarts[:, 5])
    print(f"  averaged starts ending within 1% of the best summed objective: {np.nanmean(mrel < 0.01):.3f}; "
          f"within 10%: {np.nanmean(mrel < 0.1):.3f}; their a: 5-95% "
          f"[{np.nanpercentile(ma, 5):.3f}, {np.nanpercentile(ma, 95):.3f}]")
    print(f"  converged starts, a: 5-95% [{np.nanpercentile(avals, 5):.3f}, {np.nanpercentile(avals, 95):.3f}]")
    pick = np.random.default_rng(5).choice(len(grid), 500, replace=False)
    lk.write_dat("figures/data/11-fitting-starts.dat",
                 {"ratio": 1 + np.clip(rel[pick], 0, 1e3), "a": avals[pick],
                  "mratio": 1 + np.clip(np.nan_to_num(mrel[pick], nan=1e3), 0, 1e3), "ma": ma[pick]})
    # the same path, cut earlier: iterates of the summed and averaged objectives from one start
    x0 = grid[0] + np.array([0.0, 10.0, 0.0, 0.5, 0.5])
    path_s, path_m = [], []
    fit(f_sum, x0, TIGHT, callback=lambda x: path_s.append(np.copy(x)))
    rm = fit(f_mean, x0, DEFAULT, callback=lambda x: path_m.append(np.copy(x)))
    k = min(len(path_s), len(path_m))
    dev = max(np.max(np.abs(path_s[i] - path_m[i])) for i in range(k)) if k else np.nan
    print(f"  from the start {x0}: averaged run stops after {len(path_m)} iterations ({rm.message}), "
          f"summed after {len(path_s)}; largest difference between their first {k} iterates: {dev:.2e}; "
          f"summed objective where the averaged run stopped: {f_sum(rm.x)[0]:.5g} (a={alloc(rm.x):.3f})")
    # same objective (summed), default tolerances
    sum_def = best_of(f_sum, grid, DEFAULT)
    print(f"  summed objective with default tolerances: {describe(sum_def.x)}")

    # ---- the optimizer stopped after a fixed number of iterations -------------------------
    stopped = {}
    for it in (5, 10, 20, 50):
        r = best_of(f_sum, grid, dict(TIGHT, maxiter=it))
        stopped[it] = r.x
        print(f"  recipe stopped after {it:3d} iterations: {describe(r.x)}")

    # ---- other objectives -------------------------------------------------------------------
    sg = np.vstack([small_grid(), th])
    variants = {}
    delta_hat = 1.345 * sigma
    for name, kw in [("delta=1.345 sigma", dict(delta=delta_hat)), ("delta=0.1", dict(delta=0.1)),
                     ("LS log", dict(delta=None)), ("LS loss", dict(delta=None, space="loss"))]:
        r = best_of(objective(lnN, lnD, lnL, **kw), sg)
        variants[name] = r.x
        print(f"  {name:18s}: {describe(r.x)}")
    fE = objective(lnN, lnD, lnL, 1e-3, fix_e=np.log(TRUE_E))
    rE = best_of(fE, np.delete(sg, 2, axis=1))
    thE = np.array([rE.x[0], rE.x[1], np.log(TRUE_E), rE.x[2], rE.x[3]])
    variants["E fixed"] = thE
    print(f"  {'E fixed at 0.01':18s}: {describe(thE)}")

    # variable projection check for least squares on the loss: dense grid of exponents,
    # E, A_N, A_D by nonnegative least squares
    best_vp = (np.inf, None)
    for al in np.linspace(0.2, 1.6, 141):
        XN = N ** (-al)
        for be in np.linspace(0.2, 1.6, 141):
            X = np.column_stack([np.ones(n), XN, D ** (-be)])
            coef, rn = nnls(X, L)
            if rn < best_vp[0]:
                best_vp = (rn, (coef, al, be))
    coef, al, be = best_vp[1]
    print(f"  variable projection (LS loss, grid step 0.01): E={coef[0]:.5f} alN={al:.2f} alD={be:.2f} "
          f"a={be / (al + be):.4f}")

    # ---- bootstrap over runs -------------------------------------------------------------------
    rng = np.random.default_rng(11)
    B = 1000
    boot = []
    t0 = time.time()
    starts_b = [th, stopped[50], variants["LS log"]]
    for _ in range(B):
        idx = rng.integers(0, n, n)
        fb = objective(lnN[idx], lnD[idx], lnL[idx], 1e-3)
        r = best_of(fb, starts_b)
        boot.append(r.x)
    boot = np.array(boot)
    ab = boot[:, 4] / (boot[:, 3] + boot[:, 4])
    print(f"\n== bootstrap over runs, B={B}, each refitted to convergence [{time.time() - t0:.0f} s] ==")
    names = ["A_N", "A_D", "E", "alN", "alD"]
    for j, nm in enumerate(names):
        v = np.exp(boot[:, j]) if j < 3 else boot[:, j]
        lo, hi = np.percentile(v, [2.5, 97.5])
        est = np.exp(th[j]) if j < 3 else th[j]
        print(f"  {nm}: {est:.4g}  95% [{lo:.4g}, {hi:.4g}]")
    a_hat = alloc(th)
    a_lo, a_hi = np.percentile(ab, [2.5, 97.5])
    print(f"  a: {a_hat:.4f}  95% percentile [{a_lo:.4f}, {a_hi:.4f}]")
    cov = np.cov(boot[:, 3:5].T)
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    al, be = th[3], th[4]
    grad = np.array([-be, al]) / (al + be) ** 2
    se_delta = float(np.sqrt(grad @ cov @ grad))
    tau = np.std(np.log(boot[:, 3]) - np.log(boot[:, 4]), ddof=1)
    print(f"  corr(alN, alD) over the bootstrap: {rho:.3f}; SE(alN)={np.sqrt(cov[0, 0]):.4f}, "
          f"SE(alD)={np.sqrt(cov[1, 1]):.4f}")
    print(f"  delta method: SE(a)={se_delta:.4f}, a*b*sd(log ratio)={a_hat * (1 - a_hat) * tau:.4f}, "
          f"95% [{a_hat - 1.96 * se_delta:.4f}, {a_hat + 1.96 * se_delta:.4f}]; "
          f"if uncorrelated: SE(a)={np.sqrt(grad[0] ** 2 * cov[0, 0] + grad[1] ** 2 * cov[1, 1]):.4f}")
    corr_E_alN = np.corrcoef(boot[:, 2], boot[:, 3])[0, 1]
    corr_E_alD = np.corrcoef(boot[:, 2], boot[:, 4])[0, 1]
    print(f"  corr(log E, alN) {corr_E_alN:.3f}; corr(log E, alD) {corr_E_alD:.3f}")
    lk.write_dat("figures/data/11-fitting-boot.dat",
                 {"E": np.exp(boot[:400, 2]), "alN": boot[:400, 3], "alD": boot[:400, 4], "a": ab[:400]})

    # least squares on log loss: the textbook covariance sigma^2 (J'J)^-1 and the delta method
    thl = variants["LS log"]
    eps = 1e-6
    def pred(t):
        return np.log(law(t, N, D))
    J = np.column_stack([(pred(thl + eps * np.eye(5)[j]) - pred(thl - eps * np.eye(5)[j])) / (2 * eps)
                         for j in range(5)])
    res = pred(thl) - lnL
    s2 = res @ res / (n - 5)
    covl = s2 * np.linalg.inv(J.T @ J)
    gl = np.array([-thl[4], thl[3]]) / (thl[3] + thl[4]) ** 2
    se_l = float(np.sqrt(gl @ covl[3:5, 3:5] @ gl))
    ev = np.linalg.eigvalsh(J.T @ J)
    print(f"  LS on log loss: a={alloc(thl):.4f}, NLS + delta SE {se_l:.4f}; "
          f"condition number of J'J {ev[-1] / ev[0]:.3g}")

    # ---- Hoffmann et al.'s interval scheme: 100 subsamples of 80% of the runs ----------------
    rng = np.random.default_rng(80)
    sub_conv, sub_stop, sub_wr, nit_stop = [], [], [], []
    m80 = int(round(0.8 * n))
    for _ in range(200):
        idx = rng.choice(n, m80, replace=False)
        r1 = fit(objective(lnN[idx], lnD[idx], lnL[idx], 1e-3), th, TIGHT)
        r2 = fit(objective(lnN[idx], lnD[idx], lnL[idx], 1e-3, mean=True), thm, DEFAULT)
        sub_conv.append(alloc(r1.x))
        sub_stop.append(alloc(r2.x))
        nit_stop.append(r2.nit)
        idw = rng.integers(0, n, m80)
        sub_wr.append(alloc(fit(objective(lnN[idw], lnD[idw], lnL[idw], 1e-3), th, TIGHT).x))
    sub_conv, sub_stop, sub_wr = np.array(sub_conv), np.array(sub_stop), np.array(sub_wr)
    q_conv = np.percentile(sub_conv, [10, 90])
    q_stop = np.percentile(sub_stop, [10, 90])
    q_wr = np.percentile(sub_wr, [10, 90])
    se_boot = np.std(ab, ddof=1)
    print(f"\n== 80% subsamples x200, 10th-90th percentiles of a ==")
    print(f"  converged, without replacement: [{q_conv[0]:.4f}, {q_conv[1]:.4f}] (width {q_conv[1] - q_conv[0]:.4f}); "
          f"sd {sub_conv.std(ddof=1):.4f} = {sub_conv.std(ddof=1) / se_boot:.2f} x bootstrap SE {se_boot:.4f} "
          f"(finite-population factor sqrt(0.25) = 0.50)")
    print(f"  converged, with replacement: [{q_wr[0]:.4f}, {q_wr[1]:.4f}] (width {q_wr[1] - q_wr[0]:.4f}); "
          f"sd {sub_wr.std(ddof=1):.4f} = {sub_wr.std(ddof=1) / se_boot:.2f} x bootstrap SE (sqrt(1.25) = 1.12)")
    print(f"  averaged, default tolerances, started at its own estimate: "
          f"[{q_stop[0]:.4f}, {q_stop[1]:.4f}] (width {q_stop[1] - q_stop[0]:.5f}); estimate {alloc(thm):.4f}; "
          f"median iterations {np.median(nit_stop):.0f}")
    # standard error of log N_opt(C) across compute, from the bootstrap (delta method's quadratic)
    for C in (6e6, 6 * 10 ** 7.5, 6e9, 6e11):
        v = np.log([n_opt(b, C) for b in boot])
        print(f"  bootstrap SD of log N_opt at C={C:.2g}: {np.std(v, ddof=1):.3f}")

    # bootstrap of the early-stopped fit, each refit started at its own estimate as before
    rng = np.random.default_rng(12)
    boot_m = []
    for _ in range(200):
        idx = rng.integers(0, n, n)
        r = fit(objective(lnN[idx], lnD[idx], lnL[idx], 1e-3, mean=True), thm, DEFAULT)
        boot_m.append(alloc(r.x))

    # bootstrap intervals for the other objectives (each refit to convergence from its estimate)
    def boot_variant(make, x0, B=300, seed=14):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            r = fit(make(idx), x0, TIGHT)
            x = r.x if len(r.x) == 5 else np.array([r.x[0], r.x[1], np.log(TRUE_E), r.x[2], r.x[3]])
            out.append(alloc(x))
        return np.percentile(out, [2.5, 97.5])
    vb = {}
    vb["delta=1.345 sigma"] = boot_variant(lambda i: objective(lnN[i], lnD[i], lnL[i], delta_hat),
                                           variants["delta=1.345 sigma"])
    vb["LS log"] = boot_variant(lambda i: objective(lnN[i], lnD[i], lnL[i], None), variants["LS log"])
    vb["LS loss"] = boot_variant(lambda i: objective(lnN[i], lnD[i], lnL[i], None, space="loss"),
                                 variants["LS loss"])
    vb["E fixed"] = boot_variant(lambda i: objective(lnN[i], lnD[i], lnL[i], 1e-3, fix_e=np.log(TRUE_E)),
                                 np.delete(thE, 2))
    for k_, v_ in vb.items():
        print(f"  bootstrap 95% for a, {k_}: [{v_[0]:.4f}, {v_[1]:.4f}]")

    # ---- isoFLOP analysis of the same runs ---------------------------------------------------
    Ks = sorted(set(logK))
    def vertex(x, y):
        c2, c1, c0 = np.polyfit(x, y, 2)
        return -c1 / (2 * c2) if c2 > 0 else np.nan
    def iso(idx_by_K):
        xs = []
        for k, idx in zip(Ks, idx_by_K):
            xs.append(vertex(lnN[idx], lnL[idx]))
        return np.array(xs)
    idx_by_K = [np.where(logK == k)[0] for k in Ks]
    xv = iso(idx_by_K)
    lnK = np.array(Ks) * np.log(10)
    slope, icpt = np.polyfit(lnK, xv, 1)
    # regression standard error of the slope
    rr = xv - (icpt + slope * lnK)
    se_reg = np.sqrt(rr @ rr / (len(Ks) - 2) / np.sum((lnK - lnK.mean()) ** 2))
    rng = np.random.default_rng(13)
    bs = []
    bx = []
    for _ in range(1000):
        ib = [rng.choice(idx, len(idx), replace=True) for idx in idx_by_K]
        xb = iso(ib)
        if np.all(np.isfinite(xb)):
            bs.append(np.polyfit(lnK, xb, 1)[0])
            bx.append(xb)
    bs, bx = np.array(bs), np.array(bx)
    s_lo, s_hi = np.percentile(bs, [2.5, 97.5])
    print(f"\n== isoFLOP parabolas (log loss against log N, all runs of a slice) ==")
    print(f"  a_iso = {slope:.4f}, regression SE {se_reg:.4f} (t-interval [{slope - student_t.ppf(0.975, len(Ks) - 2) * se_reg:.4f}, "
          f"{slope + student_t.ppf(0.975, len(Ks) - 2) * se_reg:.4f}]), bootstrap 95% [{s_lo:.4f}, {s_hi:.4f}]")
    # bias of the parabola against the fitted law, noise-free, on the same N points
    xv_law = iso_law = []
    xv_law = np.array([vertex(lnN[idx], np.log(law(th, N[idx], D[idx]))) for idx in idx_by_K])
    x_true = np.log(n_opt(th, 6 * 10 ** np.array(Ks)))
    print("  vertex of a parabola fitted to the recipe law on the slice points, minus the law's optimum (in log N): "
          + " ".join(f"{v:+.3f}" for v in xv_law - x_true))
    slope_law = np.polyfit(lnK, xv_law, 1)[0]
    print(f"  slope from noise-free parabolas: {slope_law:.4f} against the law's {alloc(th):.4f}")
    lo_x, hi_x = np.percentile(bx, [2.5, 97.5], axis=0)

    # ---- profile in E (least squares on log loss) ------------------------------------------
    Egrid = np.linspace(0.0060, 0.0104, 23)
    prof = []
    start4 = np.delete(variants["LS log"], 2)
    for Ev in Egrid:
        fp = objective(lnN, lnD, lnL, None, fix_e=np.log(Ev))
        r = best_of(fp, [start4, np.delete(th, 2), np.delete(thE, 2)])
        prof.append((r.fun, alloc(np.array([r.x[0], r.x[1], 0, r.x[2], r.x[3]])), r.x[2], r.x[3]))
        start4 = r.x
    prof = np.array(prof)
    rss = 2 * prof[:, 0]
    rss_min = 2 * best_of(objective(lnN, lnD, lnL, None), [variants["LS log"]]).fun
    dev = n * np.log(rss / rss_min)
    inside = Egrid[dev <= 3.84]
    print(f"\n== profile of E (least squares on log loss) ==")
    print(f"  profile 95% interval for E: about [{inside.min():.4f}, {inside.max():.4f}] (grid step {Egrid[1] - Egrid[0]:.4f}); "
          f"deviance at E=0.01: {np.interp(0.01, Egrid, dev):.2f}")
    print(f"  along the profile alN runs from {prof[0, 2]:.3f} to {prof[-1, 2]:.3f} and a from "
          f"{prof[0, 1]:.3f} to {prof[-1, 1]:.3f}")
    lk.write_dat("figures/data/11-fitting-profile.dat",
                 {"E": Egrid, "dev": dev, "a": prof[:, 1], "alN": prof[:, 2], "alD": prof[:, 3]})

    # ---- files for the figures ------------------------------------------------------------------
    rows = {c: [] for c in ("logK", "N", "D", "C", "mean", "hw", "fit")}
    for k, nn, dd in cells:
        sel = (logK == k) & (N == nn) & (D == dd)
        m, hw = lk.t_interval(L[sel])
        for c, v in zip(rows, (k, nn, dd, 6 * nn * dd, m, hw, law(th, nn, dd))):
            rows[c].append(v)
    lk.write_dat("figures/data/11-fitting-cells.dat", rows)
    # the law's isoFLOP curves, one column per budget (nan outside the slice)
    Ng = np.exp(np.linspace(np.log(30), np.log(1e5), 120))
    curves = {"N": Ng}
    for k, idx in zip(Ks, idx_by_K):
        K = 10 ** k
        col = law(th, Ng, K / Ng)
        col[(Ng < N[idx].min() / 1.15) | (Ng > N[idx].max() * 1.15)] = np.nan
        curves[f"k{int(round(10 * k))}"] = col
    lk.write_dat("figures/data/11-fitting-curves.dat", curves)
    # isoFLOP minima and the parametric frontier with a bootstrap band
    lk.write_dat("figures/data/11-fitting-minima.dat",
                 {"C": 6 * 10 ** np.array(Ks), "Nopt": np.exp(xv), "lo": np.exp(lo_x), "hi": np.exp(hi_x),
                  "ep": np.exp(hi_x) - np.exp(xv), "em": np.exp(xv) - np.exp(lo_x),
                  "law": n_opt(th, 6 * 10 ** np.array(Ks))})
    Cg = np.exp(np.linspace(np.log(6e5), np.log(6e10), 60))
    fr = np.array([n_opt(b, Cg) for b in boot])
    lk.write_dat("figures/data/11-fitting-frontier.dat",
                 {"C": Cg, "law": n_opt(th, Cg), "lo": np.percentile(fr, 2.5, axis=0),
                  "hi": np.percentile(fr, 97.5, axis=0), "mean": n_opt(thm, Cg), "stop10": n_opt(stopped[10], Cg)})
    # the forest plot: estimate of a with an interval, listed from the top row down
    bm_lo, bm_hi = np.percentile(boot_m, [2.5, 97.5])
    forest = [
        ("recipe, bootstrap 95%", a_hat, a_lo, a_hi),
        ("recipe, delta method 95%", a_hat, a_hat - 1.96 * se_delta, a_hat + 1.96 * se_delta),
        ("recipe stopped after 10 iterations", alloc(stopped[10]), np.nan, np.nan),
        ("averaged, early-stopped refits, 80% subsamples 10-90%", alloc(thm), q_stop[0], q_stop[1]),
        ("averaged, early-stopped refits, bootstrap 95%", alloc(thm), bm_lo, bm_hi),
        ("Huber delta = 1.345 sigma, bootstrap 95%", alloc(variants["delta=1.345 sigma"]), *vb["delta=1.345 sigma"]),
        ("least squares on log loss, bootstrap 95%", alloc(thl), *vb["LS log"]),
        ("least squares on log loss, NLS formula 95%", alloc(thl), alloc(thl) - 1.96 * se_l, alloc(thl) + 1.96 * se_l),
        ("least squares on loss, bootstrap 95%", alloc(variants["LS loss"]), *vb["LS loss"]),
        ("E fixed at 0.01, bootstrap 95%", alloc(thE), *vb["E fixed"]),
        ("isoFLOP parabolas, bootstrap 95%", slope, s_lo, s_hi),
    ]
    nrow = len(forest)
    rows_all = np.arange(nrow, 0, -1)
    kinds = {"main": [0, 1, 5, 6, 7, 8, 9, 10], "art": [3, 4], "pt": [2]}
    for kind, sel in kinds.items():
        lk.write_dat(f"figures/data/11-fitting-forest-{kind}.dat",
                     {"row": [rows_all[i] for i in sel], "a": [forest[i][1] for i in sel],
                      "elo": [0.0 if np.isnan(forest[i][2]) else forest[i][1] - forest[i][2] for i in sel],
                      "ehi": [0.0 if np.isnan(forest[i][3]) else forest[i][3] - forest[i][1] for i in sel]})
    for i, (nm, est, lo, hi) in enumerate(forest, 1):
        print(f"  forest row {i:2d} (top down): {nm:55s} a={est:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"\ntotal time {time.time() - T0:.0f} s")


if __name__ == "__main__":
    main()
