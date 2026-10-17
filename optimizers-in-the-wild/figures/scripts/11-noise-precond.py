"""Chapter 11 (noise), figure fig:noise:precond.

The advantage of preconditioning, T_SGD(B) / T_pre(B), against the batch size, for SGD preconditioned
by H^{-1/2} (thm:noise:precond, prop:noise:spiked), in three noisy quadratic models with Sigma = H,
theta_0 = (1,...,1) and target 5% of the initial loss:
  harmonic : d = 1000, h_i = 1/i;
  spike1   : 10 outlier modes of curvature 1 and 1000 bulk modes of curvature 0.01 (outlier share rho = 1);
  spike01  : 10 outlier modes of curvature 1 and 10000 bulk modes of curvature 0.01 (rho = 0.1).
Curves: ratio of the exact real-valued first-passage times at the best constant step size (closed
form of thm:noise:nqm). Points: the same ratio from simulation: average loss of R runs at the
theoretical step size, smoothed by a centred moving average over 2% of the predicted time, with the
crossing of the target located by linear interpolation. Bulk modes of the spiked models are
simulated by 500 representative modes, each standing for m/500 identical modes.
Writes figures/data/11-noise-precond.dat (theory), 11-noise-precond-mc.dat (simulation) and
11-noise-precond-lim.dat (the small-batch and large-batch limits, as two short segments).
Runtime: about 60 s on one CPU core. Deterministic seed.
"""
import os
import numpy as np
from scipy.optimize import brentq, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")


class NQM:
    """modes with curvature h, noise c = Sigma_ii, initial loss l0, multiplicity w."""

    def __init__(self, h, w):
        self.h = np.asarray(h, float)
        self.w = np.asarray(w, float)
        self.c = self.h.copy()                    # Sigma = H
        self.l0 = 0.5 * self.h                    # theta_0 = 1
        self.eps = 0.05 * np.sum(self.w * self.l0)

    def tstar(self, eta, B, pi):
        x = eta * pi * self.h
        if np.any(x >= 2):
            return np.inf
        F = eta * pi * self.c / (2 * B * (2 - x))
        if np.sum(self.w * F) >= self.eps:
            return np.inf
        rho, a = (1 - x) ** 2, self.l0 - F
        f = lambda t: np.sum(self.w * (F + a * rho**t)) - self.eps
        if f(0.0) < 0:
            return 0.0
        assert np.all(a >= 0)                    # monotone decreasing in t for the step sizes used
        lo, hi = 0.0, 1.0
        while f(hi) > 0:
            lo, hi = hi, 2 * hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
        return hi

    def best(self, B, pi):
        emax = 2 / np.max(pi * self.h)
        etas = np.unique(np.concatenate([emax * np.logspace(-9, 0, 220, endpoint=False),
                                         emax * np.linspace(0.8, 1, 80, endpoint=False)]))
        ts = np.array([self.tstar(e, B, pi) for e in etas])
        k = int(np.argmin(ts))
        lo, hi = etas[max(k - 1, 0)], (etas[k + 1] if k + 1 < len(etas) else emax * (1 - 1e-13))
        r = minimize_scalar(lambda e: self.tstar(e, B, pi), bounds=(lo, hi), method="bounded",
                            options={"xatol": 1e-13 * hi})
        return (r.fun, r.x) if r.fun <= ts[k] else (ts[k], etas[k])

    def Emin(self, pi):
        D = lambda tau: np.sum(self.w * self.l0 * np.exp(-2 * pi * self.h * tau))
        Q = lambda tau: np.sum(self.w * pi * self.c * (1 - np.exp(-2 * pi * self.h * tau)))
        te = brentq(lambda tau: D(tau) - self.eps, 0, 1e12)
        f = lambda tau: tau * Q(tau) / (4 * (self.eps - D(tau)))
        xs = te * (1 + np.logspace(-8, 5, 800))
        v = np.array([f(x) for x in xs])
        k = int(np.argmin(v))
        r = minimize_scalar(f, bounds=(xs[max(k - 1, 0)], xs[min(k + 1, len(xs) - 1)]),
                            method="bounded", options={"xatol": 1e-12 * xs[k]})
        return r.fun

    def mc_crossing(self, eta, B, pi, rng, t_theory, hs, ws):
        """simulation on representative modes hs (weights ws); real-valued crossing time."""
        R = 64 if B <= 32 else 256
        w = 2 * int(0.02 * t_theory) + 1 if t_theory >= 50 else 1
        Tmax = int(1.3 * t_theory) + w + 20
        th = np.ones((R, len(hs)))
        sd = np.sqrt(hs / B)                      # Sigma = H
        L = np.empty(Tmax + 1)
        for t in range(Tmax + 1):
            L[t] = np.mean((th**2) @ (0.5 * hs * ws))
            th = th - eta * pi * (th * hs + rng.standard_normal(th.shape) * sd)
        half = w // 2
        Ls = np.array([np.mean(L[max(0, t - half):t + half + 1]) for t in range(Tmax + 1 - half)])
        k = int(np.argmax(Ls < self.eps))
        if Ls[k] >= self.eps:
            return np.inf
        if k == 0:
            return 0.0
        return k - 1 + (Ls[k - 1] - self.eps) / (Ls[k - 1] - Ls[k])


def problem(name):
    if name == "harmonic":
        h = 1.0 / np.arange(1, 1001)
        P = NQM(h, np.ones(1000))
        return P, h, np.ones(1000), np.ones(1000)
    m = {"spike1": 1000, "spike01": 10000}[name]
    P = NQM([1.0, 0.01], [10, m])
    hs = np.concatenate([np.full(10, 1.0), np.full(500, 0.01)])     # representative modes
    ws = np.concatenate([np.ones(10), np.full(500, m / 500)])
    return P, hs, ws, None


if __name__ == "__main__":
    names = ["harmonic", "spike1", "spike01"]
    Bs = 2.0 ** (np.arange(0, 29) / 2)
    theory, lims, mc = {}, {}, {}
    rng = np.random.default_rng(1112)
    for name in names:
        P, hs, ws, _ = problem(name)
        pre = P.h ** (-0.5)
        one = np.ones_like(P.h)
        rat, cache = [], {}
        for B in Bs:
            t0, e0 = P.best(B, one)
            t1, e1 = P.best(B, pre)
            cache[float(B)] = (t0, e0, t1, e1)
            rat.append(t0 / t1)
        theory[name] = np.array(rat)
        E0, E1 = P.Emin(one), P.Emin(pre)
        tm0, _ = P.best(1e15, one)
        tm1, _ = P.best(1e15, pre)
        lims[name] = (E0 / E1, tm0 / tm1)
        print(f"{name}: small-batch limit {E0:.1f}/{E1:.1f} = {E0 / E1:.3f}; "
              f"large-batch limit {tm0:.2f}/{tm1:.3f} = {tm0 / tm1:.2f}")
        pts = []
        for B in [2, 8, 32, 128, 512, 2048, 8192]:
            t0, e0, t1, e1 = cache[float(B)]
            s0 = P.mc_crossing(e0, B, np.ones_like(hs), rng, t0, hs, ws)
            s1 = P.mc_crossing(e1, B, hs ** (-0.5), rng, t1, hs, ws)
            pts.append(s0 / s1)
            print(f"   B={B}: theory {t0:.1f}/{t1:.2f} = {t0 / t1:.3f}   simulation {s0:.1f}/{s1:.2f} = {s0 / s1:.3f}")
        mc[name] = np.array(pts)
    with open(os.path.join(OUT, "11-noise-precond.dat"), "w") as f:
        f.write("B " + " ".join(names) + "\n")
        for k, B in enumerate(Bs):
            f.write(f"{B:.6g} " + " ".join(f"{theory[n][k]:.5f}" for n in names) + "\n")
    with open(os.path.join(OUT, "11-noise-precond-mc.dat"), "w") as f:
        f.write("B " + " ".join(names) + "\n")
        for k, B in enumerate([2, 8, 32, 128, 512, 2048, 8192]):
            f.write(f"{B} " + " ".join(f"{mc[n][k]:.5f}" for n in names) + "\n")
    with open(os.path.join(OUT, "11-noise-precond-lim.dat"), "w") as f:
        f.write("B " + " ".join(names) + "\n")
        f.write("0.5 " + " ".join(f"{lims[n][0]:.5f}" for n in names) + "\n")
        f.write("1.5 " + " ".join(f"{lims[n][0]:.5f}" for n in names) + "\n")
        f.write("nan " + " ".join("nan" for n in names) + "\n")   # break the line between the two limits
        f.write("12000 " + " ".join(f"{lims[n][1]:.5f}" for n in names) + "\n")
        f.write("32768 " + " ".join(f"{lims[n][1]:.5f}" for n in names) + "\n")
