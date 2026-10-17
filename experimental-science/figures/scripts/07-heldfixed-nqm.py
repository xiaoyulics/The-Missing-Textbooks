"""Chapter 7 (What Was Held Fixed): the noisy quadratic model, computed exactly.

No network is trained here. The script evaluates the formulas of the chapter's theorem on the
noisy quadratic model (NQM) and prints the numbers quoted in the text:

  1. a Monte Carlo check of part (i): E f(theta_t) = f_inf + (1 - eta*kappa)^(2t) (l0 - f_inf);
  2. the constant psi(R) of part (iii) for several R;
  3. the steps to a target loss, minimized over the learning rate, as a function of the batch size,
     for one curvature (the theorem) and for many curvatures (kappa_i = 1/i, i = 1..1000, with noise
     variances c_i = kappa_i, the choice of Zhang et al. 2019), each normalized by its large-batch
     limit T_min and its critical batch size B_crit = D_min / T_min; and the optimal learning rate;
  4. McCandlish et al.'s noise scales B_noise and B_simple at the target, for comparison with B_crit.

Writes figures/data/07-heldfixed-nqm-one.dat and figures/data/07-heldfixed-nqm-many.dat.
Runs in about a minute on a laptop CPU.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk
import numpy as np
from scipy.optimize import minimize_scalar, brentq


# ---------------------------------------------------------------------------------------
# One curvature: the closed forms of the theorem
# ---------------------------------------------------------------------------------------
def psi(R):
    """psi(R) = min over 0<u<1 of ln((R-u)/(1-u))/u, and the minimizer."""
    g = lambda u: np.log((R - u) / (1 - u)) / u
    r = minimize_scalar(g, bounds=(1e-12, 1 - 1e-12), method="bounded", options={"xatol": 1e-13})
    return float(r.fun), float(r.x)


def tau_one(x, B, Bnoise, R):
    """Real-valued number of steps at which E f = target, for step x = eta*kappa (0<x<1)."""
    u = (Bnoise / B) * x / (2 - x)
    if u >= 1:
        return np.inf
    return np.log((R - u) / (1 - u)) / (-2 * np.log1p(-x))


def tstar_one(B, Bnoise, R, xmax):
    xf = min(xmax, 2 * B / (B + Bnoise) * (1 - 1e-13))
    r = minimize_scalar(lambda lx: tau_one(np.exp(lx), B, Bnoise, R), bounds=(np.log(xf) - 40, np.log(xf)),
                        method="bounded", options={"xatol": 1e-13})
    best, xbest = r.fun, float(np.exp(r.x))
    if xf == xmax and tau_one(xmax, B, Bnoise, R) <= best:
        best, xbest = tau_one(xmax, B, Bnoise, R), xmax
    return best, xbest


# ---------------------------------------------------------------------------------------
# Many curvatures: each coordinate obeys part (i); the steps are found numerically
# ---------------------------------------------------------------------------------------
def tau_many(eta, B, kap, c, f0, ell):
    x = eta * kap
    if np.any(x >= 2):
        return np.inf
    a = eta * c / (2 * B * (2 - x))
    if a.sum() >= ell:
        return np.inf
    b = f0 - a
    lq = 2 * np.log(np.abs(1 - x) + 1e-300)
    F = lambda t: a.sum() + np.sum(np.exp(lq * t) * b) - ell
    if F(0.0) <= 0:
        return 0.0
    prev = 0.0
    for t in np.geomspace(1e-3, 1e10, 500):
        if F(t) <= 0:
            return brentq(F, prev, t, xtol=1e-10, rtol=1e-12)
        prev = t
    return np.inf


def tstar_many(B, kap, c, f0, ell):
    etas = np.geomspace(1e-7, 1.9999 / kap.max(), 240)
    vals = np.array([tau_many(e, B, kap, c, f0, ell) for e in etas])
    i = int(np.argmin(vals))
    lo, hi = etas[max(i - 1, 0)], etas[min(i + 1, len(etas) - 1)]
    r = minimize_scalar(lambda le: tau_many(np.exp(le), B, kap, c, f0, ell), bounds=(np.log(lo), np.log(hi)),
                        method="bounded", options={"xatol": 1e-11})
    if r.fun < vals[i]:
        return float(r.fun), float(np.exp(r.x))
    return float(vals[i]), float(etas[i])


def main():
    # 1. Monte Carlo check of part (i)
    rng = np.random.default_rng(20261018)
    kappa, c, B, eta, theta0 = 2.0, 3.0, 4, 0.1, 1.5
    x = eta * kappa
    finf = eta * c / (2 * B * (2 - x))
    l0 = 0.5 * kappa * theta0 ** 2
    th = np.full(400_000, theta0)
    print("Monte Carlo check of part (i): kappa=2, c=3, B=4, eta=0.1, theta0=1.5, 400000 chains")
    for t in range(1, 101):
        th = th - eta * (kappa * th + rng.normal(0.0, np.sqrt(c / B), size=th.size))
        if t in (1, 10, 30, 100):
            f = 0.5 * kappa * th ** 2
            mean, hw = lk.t_interval(f)
            print(f"  t={t:3d}  simulated E f = {mean:.5f} +- {hw:.5f}   formula = {finf + (1 - x) ** (2 * t) * (l0 - finf):.5f}")
    print(f"  stationary loss eta*c/(2B(2-eta*kappa)) = {finf:.5f}")

    # 2. psi(R)
    print("psi(R) = min_u ln((R-u)/(1-u))/u:")
    for R in [10, 100, 1000, 10_000]:
        p, u = psi(R)
        print(f"  R={R:6d}  psi={p:.3f}  argmin u={u:.3f}  ln R={np.log(R):.3f}  psi/lnR={p / np.log(R):.3f}")

    # 3a. one curvature: Bnoise = 1 (batch sizes in units of the noise scale), R = 100, xmax = 0.05
    Bnoise, R, xmax = 1.0, 100.0, 0.05
    p, _ = psi(R)
    Dmin = Bnoise * p / 4
    Tmin = np.log(R) / (-2 * np.log1p(-xmax))
    Bcrit = Dmin / Tmin
    print(f"one curvature: R={R:g}, xmax={xmax}, B_noise=1: D_min={Dmin:.4f}, T_min={Tmin:.3f}, "
          f"B_crit={Bcrit:.5f} = {Bcrit / Bnoise:.4f} B_noise")
    rel = np.geomspace(1e-2, 1e2, 81)
    T1, E1 = [], []
    for r_ in rel:
        t, xb = tstar_one(r_ * Bcrit, Bnoise, R, xmax)
        T1.append(t / Tmin)
        E1.append(xb / xmax)
    T1, E1 = np.array(T1), np.array(E1)
    # the bounds of parts (ii) and (iii), checked on a grid of batch sizes
    worst = 0.0
    for Bv in np.geomspace(1e-4, 1e3, 300):
        t, _ = tstar_one(Bv, Bnoise, R, xmax)
        assert t >= Tmin * (1 - 1e-9)
        uB = (Bnoise / Bv) * xmax / (2 - xmax)
        if uB < 1:
            assert t <= Tmin * (1 + np.log(1 / (1 - uB)) / np.log(R)) * (1 + 1e-9)
        if 2 * Bv / (Bv + Bnoise) <= xmax:
            assert (1 - Bv / Bnoise) * Dmin * (1 - 1e-9) <= Bv * t <= (1 + Bv / Bnoise) * Dmin * (1 + 1e-9)
        worst = max(worst, abs(t / (Tmin * (1 + Bcrit / Bv)) - 1))
    print(f"  bounds of the theorem hold on 300 batch sizes; largest deviation from the hyperbola: {100 * worst:.1f}%")
    i1 = int(np.argmin(np.abs(rel - 1)))
    print(f"  at B = B_crit: T*/T_min = {T1[i1]:.3f} (hyperbola: 2)")
    lk.write_dat("figures/data/07-heldfixed-nqm-one.dat",
                 {"x": rel, "T": T1, "eta": E1, "hyp": 1 + 1 / rel, "ideal": np.maximum(1, 1 / rel)})

    # 3b. many curvatures: kappa_i = 1/i, c_i = kappa_i, theta_i(0) = 1, target 2% of the initial loss
    d = 1000
    kap = 1.0 / np.arange(1, d + 1)
    cc = kap.copy()
    f0 = 0.5 * kap
    ell = 0.02 * f0.sum()
    Tbig, etabig = tstar_many(1e7, kap, cc, f0, ell)
    Tsmall, _ = tstar_many(1e-3, kap, cc, f0, ell)
    TminM, DminM = Tbig, 1e-3 * Tsmall
    BcritM = DminM / TminM
    print(f"many curvatures (d={d}, kappa_i=1/i, c_i=kappa_i, target {ell:.4f} = 2% of {f0.sum():.3f}): "
          f"T_min={TminM:.1f}, D_min={DminM:.0f}, B_crit={BcritM:.1f}, eta*(large B)={etabig:.3f}")
    TM, EM = [], []
    for r_ in rel:
        t, e = tstar_many(r_ * BcritM, kap, cc, f0, ell)
        TM.append(t / TminM)
        EM.append(e / etabig)
    TM, EM = np.array(TM), np.array(EM)
    devs = np.abs(TM / (1 + 1 / rel) - 1)
    print(f"  largest deviation from the hyperbola T_min(1+B_crit/B): {100 * devs.max():.1f}%; "
          f"at B = B_crit: T*/T_min = {TM[i1]:.3f}")
    for r_ in [0.1, 1.0, 10.0]:
        j = int(np.argmin(np.abs(rel - r_)))
        print(f"  B = {r_:5.1f} B_crit: T*/T_min = {TM[j]:.3f}, optimal eta / eta(large B) = {EM[j]:.3f}")
    lk.write_dat("figures/data/07-heldfixed-nqm-many.dat",
                 {"x": rel, "T": TM, "eta": EM, "hyp": 1 + 1 / rel})

    # 4. the noise scales at the target, on the large-batch path (eta = etabig, no noise)
    t_hit = tau_many(etabig, 1e12, kap, cc, f0, ell)
    th_t = np.abs(1 - etabig * kap) ** t_hit
    G = kap * th_t
    Bnoise_t = np.sum(kap * cc) / np.sum(kap * G ** 2)
    Bsimple_t = np.sum(cc) / np.sum(G ** 2)
    print(f"  at the target on the large-batch path: B_noise = {Bnoise_t:.1f}, B_simple = {Bsimple_t:.1f} "
          f"(B_crit = {BcritM:.1f})")


if __name__ == "__main__":
    main()
