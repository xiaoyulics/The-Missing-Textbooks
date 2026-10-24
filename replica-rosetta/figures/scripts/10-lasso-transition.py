"""Chapter 10: the Donoho--Tanner phase transition of noiseless l1 minimization, three ways and by
linear programming.

Noiseless recovery: y = A theta0 with A in R^{n x d} Gaussian, theta0 with k = eps d nonzero entries
(iid N(0,1) values), alpha = n/d; l1 minimization min |theta|_1 subject to A theta = y.
The predicted weak threshold is alpha = psi(eps), with (Amelunxen--Lotz--McCoy--Tropp 2014, Prop. 4.5)
    psi(eps) = min_{t >= 0} J_eps(t),   J_eps(t) = eps (1 + t^2) + (1 - eps) E(|Z| - t)_+^2,
and E(|Z| - t)_+^2 = 2 g(t), g(t) = (1 + t^2) Phi(-t) - t phi(t).
Three formulas for the curve, which must agree:
  (i)   statistical dimension (ALMT): eps_c(alpha) = psi^{-1}(alpha);
  (ii)  state evolution (Donoho--Maleki--Montanari 2009, eq. (5), signed case):
        rho_SE(alpha) = max_z [1 - (2/alpha) g(z)] / [1 + z^2 - 2 g(z)],  rho = k/n = eps/alpha;
  (iii) replica (Kabashima--Wadayama--Tanaka 2009, eqs. (11)-(12)).
Writes:
  10-lasso-transition-curve.dat : alpha, eps_c, rho_c = eps_c/alpha (Donoho--Tanner coordinates);
  10-lasso-transition-prob.dat  : success fraction of l1 minimization at alpha = 0.5 against rho for
                                  d = 100, 200, 400 (40 trials per point, Clopper--Pearson 95% intervals);
  10-lasso-transition-pts.dat   : 50% points in the (alpha, rho) plane at d = 200 from logistic fits
                                  (30 trials per point, 9 points per alpha), with parametric-bootstrap
                                  95% intervals.
Also prints a Monte Carlo check of the bounds d psi(s/d) - 2 sqrt(d/s) <= delta(D) <= d psi(s/d) on the
statistical dimension of the l1 descent cone.
Runtime: about one minute.
"""
import os
import sys
import time
import math
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                     # noqa: E402
from scipy import optimize, special    # noqa: E402
import rosettakit as rk                # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
SQ2PI = math.sqrt(2 * math.pi)
Phi = special.ndtr


def phi(x):
    return math.exp(-0.5 * x * x) / SQ2PI


def g(t):
    return (1 + t * t) * Phi(-t) - t * phi(t)


def psi(eps):
    """ALMT's psi(eps) = min_t J_eps(t), and the minimizing t."""
    J = lambda t: eps * (1 + t * t) + (1 - eps) * 2 * g(t)
    r = optimize.minimize_scalar(J, bounds=(0.0, 30.0), method="bounded", options={"xatol": 1e-14})
    return r.fun, r.x


def eps_almt(alpha):
    return rk.root(lambda e: psi(e)[0] - alpha, 1e-14, 1 - 1e-14)


def rho_dmm(alpha):
    f = lambda z: (1 - (2 / alpha) * g(z)) / (1 + z * z - 2 * g(z))
    r = optimize.minimize_scalar(lambda z: -f(z), bounds=(0.0, 12.0), method="bounded", options={"xatol": 1e-14})
    return -r.fun


def alpha_kwt(rho):
    """Kabashima--Wadayama--Tanaka: alpha_c(rho) = 2(1-rho) H(chihat^{-1/2}) + rho, chihat from their eq. (11)
    (their rho is our eps)."""
    H = lambda x: 1 - Phi(x)

    def eq(lc):
        c = math.exp(lc)
        a = 2 * (1 - rho) * H(c ** -0.5) + rho
        rhs = 2 * (1 - rho) * ((c + 1) * H(c ** -0.5) - math.sqrt(c) * math.exp(-1 / (2 * c)) / SQ2PI) + rho * (c + 1)
        return c * a - rhs
    c = math.exp(rk.root(eq, -14.0, 14.0))
    return 2 * (1 - rho) * H(c ** -0.5) + rho


def eps_kwt(alpha):
    return rk.root(lambda e: alpha_kwt(e) - alpha, 1e-9, alpha - 1e-12)


def l1_success(d, alpha, k, gen):
    """One trial: is theta0 (k nonzeros, N(0,1) values) the solution of min |theta|_1 s.t. A theta = A theta0?"""
    n = int(round(alpha * d))
    A = gen.standard_normal((n, d))
    theta0 = np.zeros(d)
    idx = gen.choice(d, size=k, replace=False)
    theta0[idx] = gen.standard_normal(k)
    y = A @ theta0
    res = optimize.linprog(np.ones(2 * d), A_eq=np.hstack([A, -A]), b_eq=y, bounds=(0, None), method="highs")
    if res.status != 0:
        return False
    th = res.x[:d] - res.x[d:]
    return np.linalg.norm(th - theta0) <= 1e-4 * np.linalg.norm(theta0)


def logistic_fit(x, k, m):
    """Maximum likelihood fit of logit P(success) = a + b x to k successes out of m trials at each x
    (Newton's method, with a small ridge for separable data); returns the 50% point -a/b."""
    x, k = np.asarray(x, float), np.asarray(k, float)
    X = np.column_stack([np.ones_like(x), x - x.mean()])
    beta = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ beta)))
        grad = X.T @ (k - m * p) - 1e-3 * beta
        hess = (X.T * (m * p * (1 - p))) @ X + 1e-3 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return x.mean() - beta[0] / beta[1]


def statdim_mc(d, s, draws, gen):
    """Monte Carlo of delta(D(|.|_1, x)) for x with s entries equal to 1: E min_t dist^2(g, t * subdiff)."""
    G = gen.standard_normal((draws, d))
    gs = G[:, :s].sum(axis=1)
    absoff = np.sort(np.abs(G[:, s:]), axis=1)[:, ::-1]
    csum = np.concatenate([np.zeros((draws, 1)), np.cumsum(absoff, axis=1)], axis=1)
    ks = np.arange(d - s + 1)
    taus = (gs[:, None] + csum) / (s + ks[None, :])
    hi = np.concatenate([np.full((draws, 1), np.inf), absoff], axis=1)
    lo = np.concatenate([absoff, np.zeros((draws, 1))], axis=1)
    ok = (taus <= hi) & (taus >= lo)
    tau = np.where(ok.any(axis=1), np.maximum(taus[np.arange(draws), np.argmax(ok, axis=1)], 0.0), 0.0)
    vals = np.sum((G[:, :s] - tau[:, None]) ** 2, axis=1) + np.sum(np.maximum(np.abs(G[:, s:]) - tau[:, None], 0) ** 2, axis=1)
    return rk.mean_ci(vals)


def main():
    t0 = time.time()
    # ---- the curve, three ways
    print("alpha    eps_c (ALMT)     rho (ALMT)      rho (DMM eq. 5)   rho (KWT replica)")
    for a in (0.1, 0.25, 0.5, 0.75, 0.9):
        e1 = eps_almt(a)
        print(f"{a:5.2f}   {e1:.10f}   {e1 / a:.10f}   {rho_dmm(a):.10f}   {eps_kwt(a) / a:.10f}")
    print("KWT alpha_c(eps = 0.5) =", f"{alpha_kwt(0.5):.8f}", " ALMT psi(0.5) =", f"{psi(0.5)[0]:.8f}")
    maxdiff = 0.0
    rows = []
    for a in np.concatenate([np.linspace(0.005, 0.05, 10), np.linspace(0.06, 0.99, 94)]):
        e1 = eps_almt(a)
        maxdiff = max(maxdiff, abs(e1 / a - rho_dmm(a)), abs(e1 - eps_kwt(a)) / a)
        rows.append((a, e1, e1 / a))
    print(f"max |difference| between the three formulas over {len(rows)} values of alpha: {maxdiff:.2e}")
    rk.write_dat(os.path.join(DATA, "10-lasso-transition-curve.dat"), ["alpha", "epsc", "rhoc"], rows)
    # ---- statistical dimension bounds, Monte Carlo
    gen = rk.rng(5)
    print("\nstatistical dimension of the l1 descent cone, Monte Carlo (20000 draws each, seed 5):")
    for d, s in ((100, 5), (100, 10), (100, 20), (100, 50)):
        m, h = statdim_mc(d, s, 20000, gen)
        p = psi(s / d)[0]
        print(f"  d = {d}, s = {s}: delta(D) = {m:.2f} +- {h:.2f}; d psi(s/d) = {d * p:.2f}; "
              f"lower bound d psi - 2 sqrt(d/s) = {d * p - 2 * math.sqrt(d / s):.2f}")
    # ---- success probability at alpha = 0.5 for three sizes
    alpha = 0.5
    sizes = (50, 100, 200)
    rhos = [0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52, 0.56]
    trials = 30
    cols = {}
    for d in sizes:
        n = int(alpha * d)
        cols[d] = []
        for i, rho in enumerate(rhos):
            k = max(1, int(round(rho * n)))
            gen = rk.rng(10000 + 1000 * (d // 50) + i)
            succ = sum(l1_success(d, alpha, k, gen) for _ in range(trials))
            lo, hi = rk.clopper_pearson(succ, trials)
            cols[d].append((k / n, succ / trials, lo, hi))
        print(f"d = {d}: success fractions {[round(float(c[1]), 3) for c in cols[d]]} at rho = {[round(c[0], 3) for c in cols[d]]}")
    prow = []
    for i, rho in enumerate(rhos):
        row = [rho]
        for d in sizes:
            r_eff, p, lo, hi = cols[d][i]
            row += [r_eff, p, p - lo, hi - p]
        prow.append(row)
    hdr = ["rho"] + [f"{c}{d}" for d in sizes for c in ("r", "p", "lo", "hi")]
    rk.write_dat(os.path.join(DATA, "10-lasso-transition-prob.dat"), hdr, prow)
    print(f"predicted rho_c(0.5) = {eps_almt(0.5) / 0.5:.6f}")
    # ---- 50% points across alpha at d = 100
    d, trials = 100, 24
    pts = []
    for j, a in enumerate((0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)):
        n = int(round(a * d))
        rc = eps_almt(a) / a
        grid = np.linspace(rc - 0.15, rc + 0.15, 7)
        ks, xs = [], []
        for i, rho in enumerate(grid):
            k = int(round(rho * n))
            if k < 1 or k >= n:
                continue
            gen = rk.rng(20000 + 100 * j + i)
            ks.append(sum(l1_success(d, a, k, gen) for _ in range(trials)))
            xs.append(k / n)
        x50 = logistic_fit(xs, ks, trials)
        # parametric bootstrap of the 50% point from the fitted curve
        xs_a = np.array(xs)
        Xc = np.column_stack([np.ones_like(xs_a), xs_a - xs_a.mean()])
        kk = np.array(ks, float)
        beta = np.zeros(2)
        for _ in range(50):
            p = 1 / (1 + np.exp(-(Xc @ beta)))
            beta += np.linalg.solve((Xc.T * (trials * p * (1 - p))) @ Xc + 1e-3 * np.eye(2),
                                    Xc.T @ (kk - trials * p) - 1e-3 * beta)
        p_fit = 1 / (1 + np.exp(-(Xc @ beta)))
        bgen = rk.rng(30000 + j)
        boots = [logistic_fit(xs, bgen.binomial(trials, p_fit), trials) for _ in range(400)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pts.append((a, x50, x50 - lo, hi - x50, rc))
        print(f"alpha = {a:.2f} (n = {n}): 50% point rho = {x50:.4f} [{lo:.4f}, {hi:.4f}], predicted {rc:.4f}")
    rk.write_dat(os.path.join(DATA, "10-lasso-transition-pts.dat"), ["alpha", "rho50", "lo", "hi", "pred"], pts)
    print("runtime %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
