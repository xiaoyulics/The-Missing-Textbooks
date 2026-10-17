"""Chapter 11 (noise), figure fig:noise:steps.

Steps to a target loss against the batch size in the noisy quadratic model, at the best constant
step size: SGD, heavy-ball momentum (beta = 0.9) and SGD preconditioned by H^{-1/2}.
Curves: exact expected-loss formulas (thm:noise:nqm, closed form for SGD and preconditioned SGD,
exact second-moment recursion for momentum), minimized over the step size. Points: the first step at
which the average loss of R independent runs (R = 64 for B <= 32, 256 above), at the step size chosen
by the theory and smoothed by a centred moving average over 2% of the predicted time, is below the
target. Also writes the optimal step sizes (for SGD and H^{-1/2}, the minimizers of the real-valued
crossing time; for momentum, whose count is minimized on narrow intervals of step sizes, the step
size with the smallest expected loss at step T(B)), the hyperbolas T_min + E_min/B with the integer
T_min, and McCandlish's product computed with the real-valued crossing times.

Model: d = 1000, h_i = 1/i, Sigma = H, theta_0 = (1,...,1), target 5% of the initial loss.
Writes figures/data/11-noise-steps.dat (theory), figures/data/11-noise-steps-mom.dat (momentum,
theory) and figures/data/11-noise-steps-mc.dat (simulation).
Runtime: about 80 s on one CPU core. Deterministic seed.
"""
import os
import numpy as np
from scipy.optimize import brentq, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")

d = 1000
h = 1.0 / np.arange(1, d + 1)
c = h.copy()
l0 = 0.5 * h                      # initial loss of each mode, theta_0 = 1
eps = 0.05 * l0.sum()
S = h.max()
BETA = 0.9


# ---------------- SGD and preconditioned SGD: closed form (thm:noise:nqm) ----------------
def tstar(eta, B, pi):
    """real t at which sum_i F_i + (l_i - F_i) rho_i^t equals eps (decreasing in t); inf if never."""
    x = eta * pi * h
    if np.any(x >= 2):
        return np.inf
    F = eta * pi * c / (2 * B * (2 - x))
    if F.sum() >= eps:
        return np.inf
    rho, a = (1 - x) ** 2, l0 - F
    f = lambda t: np.sum(F + a * rho**t) - eps
    if f(0.0) < 0:
        return 0.0
    if np.any(a < 0):           # not monotone: scan integers (does not occur for the optima here)
        t = 0
        while f(t) >= 0:
            t += 1
        lo, hi = t - 1.0, float(t)
    else:
        lo, hi = 0.0, 1.0
        while f(hi) > 0:
            lo, hi = hi, 2 * hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return hi


def best(B, pi):
    """min over stable eta of the real first-passage time; returns (t*, eta*)."""
    emax = 2 / np.max(pi * h)
    etas = np.unique(np.concatenate([emax * np.logspace(-9, 0, 220, endpoint=False),
                                     emax * np.linspace(0.8, 1, 80, endpoint=False)]))
    ts = np.array([tstar(e, B, pi) for e in etas])
    k = int(np.argmin(ts))
    lo, hi = etas[max(k - 1, 0)], (etas[k + 1] if k + 1 < len(etas) else emax * (1 - 1e-13))
    r = minimize_scalar(lambda e: tstar(e, B, pi), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-13 * hi})
    return (r.fun, r.x) if r.fun <= ts[k] else (ts[k], etas[k])


def T_int(t):                      # smallest integer t with loss < eps
    return int(np.floor(t)) + 1


def small_batch_limit(pi):
    """E_min = min_tau tau Q(tau) / (4 (eps - D(tau))) and gamma* (thm:noise:critical)."""
    D = lambda tau: np.sum(l0 * np.exp(-2 * pi * h * tau))
    Q = lambda tau: np.sum(pi * c * (1 - np.exp(-2 * pi * h * tau)))
    te = brentq(lambda tau: D(tau) - eps, 0, 1e12)
    f = lambda tau: tau * Q(tau) / (4 * (eps - D(tau)))
    xs = te * (1 + np.logspace(-8, 5, 800))
    v = np.array([f(x) for x in xs])
    k = int(np.argmin(v))
    r = minimize_scalar(f, bounds=(xs[max(k - 1, 0)], xs[min(k + 1, len(xs) - 1)]), method="bounded",
                        options={"xatol": 1e-12 * xs[k]})
    return r.fun, 4 * (eps - D(r.x)) / Q(r.x), r.x, te


# ---------------- heavy ball: exact second moments (zero initial velocity) ----------------
def mom_T(etas, B, Tmax):
    """first integer t with expected loss < eps, for each eta (inf if not by Tmax)."""
    E = etas[:, None]
    a = 1 + BETA - E * h[None, :]
    qp = np.zeros((len(etas), d))
    q = np.ones_like(qp)
    S2 = np.zeros_like(qp)
    coef = l0[None, :]                      # (h/2) theta_0^2
    ncoef = 0.5 * h[None, :] * E**2 * c[None, :] / B
    res = np.full(len(etas), np.inf)
    alive = np.ones(len(etas), bool)
    for t in range(Tmax + 1):
        L = np.sum(coef * (q - BETA * qp) ** 2 + ncoef * S2, axis=1)
        hit = alive & (L < eps)
        res[hit] = t
        alive &= ~hit
        if not alive.any():
            break
        S2 += q * q
        qp, q = q, a * q - BETA * qp
    return res


def mom_grid_min(etas, B, Tmax):
    """for each step t <= Tmax, the smallest expected loss over the step sizes etas, and its index."""
    E = etas[:, None]
    a = 1 + BETA - E * h[None, :]
    qp = np.zeros((len(etas), d))
    q = np.ones_like(qp)
    S2 = np.zeros_like(qp)
    ncoef = 0.5 * h[None, :] * E**2 * c[None, :] / B
    Lmin, kmin = np.empty(Tmax + 1), np.zeros(Tmax + 1, int)
    for t in range(Tmax + 1):
        L = np.sum(l0[None, :] * (q - BETA * qp) ** 2 + ncoef * S2, axis=1)
        kmin[t] = int(np.argmin(L))
        Lmin[t] = L[kmin[t]]
        S2 += q * q
        qp, q = q, a * q - BETA * qp
    return Lmin, kmin


def mom_loss(eta, B, t):
    """exact expected loss of the heavy ball at step t, for one step size."""
    a = 1 + BETA - eta * h
    qp, q, S2 = np.zeros(d), np.ones(d), np.zeros(d)
    for _ in range(t):
        S2 += q * q
        qp, q = q, a * q - BETA * qp
    return np.sum(l0 * (q - BETA * qp) ** 2 + 0.5 * h * eta**2 * c / B * S2)


def mom_refine(B, t, lo, hi, e0):
    """the step size in [lo, hi] with the smallest loss at step t (the better of a search and e0)."""
    r = minimize_scalar(lambda e: mom_loss(e, B, t), bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-10 * hi})
    L0 = mom_loss(e0, B, t)
    return (r.fun, r.x) if r.fun < L0 else (L0, e0)


def mom_best(B, Tmax):
    """T(B) for the heavy ball, the smallest t at which some stable step size gives an expected loss
    below eps, and the step size that minimizes the expected loss at that t. A coarse search over
    first-passage times suffices where the loss is smooth in the step size (many steps, small B).
    With few steps the heavy ball oscillates and the step sizes that reach the target first form
    narrow intervals, so there the loss at every step is minimized over a dense grid and refined."""
    lim = (2 + 2 * BETA) / S
    etas = lim * np.logspace(-6, 0, 49, endpoint=False)
    T = mom_T(etas, B, Tmax)
    k = int(np.argmin(T))
    lo, hi = etas[max(k - 1, 0)], (etas[k + 1] if k + 1 < len(etas) else lim * 0.9999)
    etas2 = np.linspace(lo, hi, 33)
    T2 = mom_T(etas2, B, int(min(Tmax, T[k])))
    k2 = int(np.argmin(T2))
    Tc, ec = (T2[k2], etas2[k2]) if T2[k2] <= T[k] else (T[k], etas[k])
    if Tc > 200:
        j = int(np.argmin(np.abs(etas2 - ec)))
        _, e = mom_refine(B, int(Tc), etas2[max(j - 1, 0)], etas2[min(j + 1, len(etas2) - 1)], ec)
        return Tc, e
    grid = lim * np.concatenate([np.logspace(-3, np.log10(0.5), 300, endpoint=False),
                                 np.linspace(0.5, 0.99999, 900)])
    Lmin, kmin = mom_grid_min(grid, B, int(Tc))
    for t in range(1, int(Tc) + 1):
        if Lmin[t] < 1.3 * eps:
            j = kmin[t]
            L, e = mom_refine(B, t, grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)], grid[j])
            if L < eps:
                return float(t), e
    return Tc, ec


# ---------------- Monte Carlo ----------------
def mc_first_passage(kind, eta, B, rng, T_theory):
    """first t at which the average loss of Rk runs (64 for B <= 32, 256 above), smoothed by a
    centred moving average over w = 2 floor(0.02 T) + 1 steps (w = 1 below 50 steps), is below eps."""
    Rk = 64 if B <= 32 else 256
    w = 2 * int(0.02 * T_theory) + 1 if T_theory >= 50 else 1
    Tmax = int(1.3 * T_theory) + w + 20
    th = np.ones((Rk, d))
    v = np.zeros((Rk, d))
    sd = np.sqrt(c / B)
    pi = h ** (-0.5) if kind == "pre" else np.ones(d)
    L = np.empty(Tmax + 1)
    for t in range(Tmax + 1):
        L[t] = np.mean(0.5 * (th**2) @ h)
        g = th * h + rng.standard_normal(th.shape) * sd
        if kind == "mom":   # heavy ball: theta_{t+1} = theta_t - eta g_t + beta (theta_t - theta_{t-1})
            v = BETA * v - eta * g
            th = th + v
        else:
            th = th - eta * pi * g
    half = w // 2
    for t in range(Tmax + 1 - half):
        lo = max(0, t - half)
        if np.mean(L[lo:t + half + 1]) < eps:
            return t
    return np.inf


if __name__ == "__main__":
    pre = h ** (-0.5)
    one = np.ones(d)
    Bs = 2.0 ** (np.arange(0, 29) / 2)          # 1 ... 16384
    rows = []
    for B in Bs:
        t0, e0 = best(B, one)
        t1, e1 = best(B, pre)
        rows.append((B, T_int(t0), e0, T_int(t1), e1, t0, t1))
    Emin0, g0, tau0, te0 = small_batch_limit(one)
    Emin1, g1, tau1, te1 = small_batch_limit(pre)
    tm0, em0 = best(1e15, one)
    tm1, em1 = best(1e15, pre)
    Tmin0, Tmin1 = T_int(tm0), T_int(tm1)
    print(f"real t*_min: SGD {tm0:.3f}, pre {tm1:.3f}")
    print(f"SGD : E_min={Emin0:.1f} gamma*={g0:.5f} tau*={tau0:.1f} tau_eps={te0:.1f} "
          f"T_min={Tmin0} (eta={em0:.4f}) B_crit={Emin0 / Tmin0:.1f}")
    print(f"H^-1/2: E_min={Emin1:.1f} gamma*={g1:.6f} tau*={tau1:.2f} tau_eps={te1:.2f} "
          f"T_min={Tmin1} (eta={em1:.4f}) B_crit={Emin1 / Tmin1:.1f}")
    with open(os.path.join(OUT, "11-noise-steps.dat"), "w") as f:
        f.write("B sgd etasgd pre etapre hypsgd hyppre linsgd linpre prodsgd prodpre\n")
        for B, T0, e0, T1, e1, r0, r1 in rows:
            # McCandlish's product, with the real-valued crossing times: rounding T(B) up to an
            # integer changes E(B) = B T(B) by up to B, the order that the product measures
            prod0 = (r0 / tm0 - 1) * (B * r0 / Emin0 - 1)
            prod1 = (r1 / tm1 - 1) * (B * r1 / Emin1 - 1)
            f.write(f"{B:.6g} {T0} {e0:.6e} {T1} {e1:.6e} {Tmin0 + Emin0 / B:.6e} {Tmin1 + Emin1 / B:.6e} "
                    f"{g0 * B:.6e} {g1 * B:.6e} {prod0:.4f} {prod1:.4f}\n")
            print(f"B={B:8.1f}  SGD {T0:6d} (eta {e0:.4f})  pre {T1:5d} (eta {e1:.5f})  "
                  f"T/(T_min+E_min/B) {T0 / (Tmin0 + Emin0 / B):.3f} {T1 / (Tmin1 + Emin1 / B):.3f}  "
                  f"products {prod0:.3f} {prod1:.3f}")
    # momentum on a coarser grid of batch sizes
    mrows = []
    for B, T0 in zip(Bs[::2], [r[1] for r in rows][::2]):  # B = 1, 2, 4, ..., 16384
        Tm, em = mom_best(B, int(1.3 * T0) + 50)
        mrows.append((B, int(Tm), em))
        print(f"B={B:8.1f}  momentum {int(Tm)} (eta {em:.4f}, eta/(1-beta) {em / (1 - BETA):.3f})")
    TminM, emM = mom_best(1e15, 400)
    print(f"momentum T_min={int(TminM)} (eta={emM:.4f}); B_crit={Emin0 / TminM:.1f}")
    with open(os.path.join(OUT, "11-noise-steps-mom.dat"), "w") as f:
        f.write("B mom etamom\n")
        for B, Tm, em in mrows:
            f.write(f"{B:.6g} {Tm} {em:.6e}\n")
    # simulation at seven batch sizes
    rng = np.random.default_rng(1111)
    th_rows = {r[0]: r for r in rows}
    m_rows = {r[0]: r for r in mrows}
    with open(os.path.join(OUT, "11-noise-steps-mc.dat"), "w") as f:
        f.write("B sgd pre mom\n")
        for B in [2, 8, 32, 128, 512, 2048, 8192]:
            T0, e0, T1, e1 = th_rows[float(B)][1:5]
            _, Tm, em = m_rows[float(B)]
            s0 = mc_first_passage("sgd", e0, B, rng, T0)
            s1 = mc_first_passage("pre", e1, B, rng, T1)
            sm = mc_first_passage("mom", em, B, rng, Tm)
            f.write(f"{B} {s0} {s1} {sm}\n")
            print(f"MC B={B}: SGD {s0} (theory {T0})  pre {s1} ({T1})  mom {sm} ({Tm})")
