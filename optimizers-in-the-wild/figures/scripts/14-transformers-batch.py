"""Chapter 14 (transformers): steps to target versus batch size in the noisy quadratic model.

Model (Chapter 11, Zhang et al. 2019): L(theta) = sum_i h_i theta_i^2 / 2 with h_i = 1/i, d = 1000;
stochastic gradient H theta + zeta with zeta ~ N(0, diag(c)/B); preconditioned SGD
theta <- theta - eta * a * g with multipliers a_i = h_i^(-p): p = 0 is SGD, p = 1/2 the model of
Adam at small batch.  Initialization theta_0 = 1 (all coordinates), target: expected loss at most 5%
of the initial loss.  Steps to target T(B): least t such that some constant eta in (0, 2/max a_i h_i)
reaches the target at step t.

Theory (exact recursions; eta optimized on a grid of 400 log-spaced values below the stability limit):
  sgd, pre              T(B) for p = 0 and p = 1/2, noise c = h, last iterate
  sgdsmall, presmall    small-batch asymptotes n_a(eps)/B (Theorem batch (ii))
  sgdbig, prebig        large-batch limits T_a(infinity) (Theorem batch (i))
  rconst                ratio T_sgd/T_pre, constant step, last iterate, c = h
  ravg                  same with tail averaging (average of the last fraction 1-f of the iterates,
                        f in {0,0.1,...,0.9,1} optimized with eta), exact formula for the average
  rlin                  same with a linear decay of the step to zero over the run
  riso                  constant step, last iterate, isotropic noise c_i = 1/1000 * sum_j h_j / ...
                        normalized so that tr(C) is the same as for c = h
Simulation (Monte Carlo, seeds fixed): 32 runs at the theory-optimal eta for each B in
  {1,4,16,64,256,1024,4096}; the reported step is the first step at which the 32-run average loss,
  smoothed by a trailing mean over the last 2% of the steps, is below the target.
  mcsgd, mcpre          SGD and p = 1/2
  mcrms                 RMSProp (Adam with beta1 = 0, beta2 = 0.999, bias correction, epsilon = 0),
                        16 runs, eta tuned over 4 values around eta_pre/sqrt(B); B <= 64 only
Outputs: figures/data/14-transformers-batch.dat (theory), 14-transformers-batch-mc.dat (Monte Carlo).
Runtime: about 80 seconds on one core.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
D = 1000
IDX = np.arange(1, D + 1, dtype=float)
H = 1.0 / IDX
C_PROP = H.copy()
C_ISO = np.full(D, H.sum() / D)
TH0SQ = np.ones(D)
L0 = 0.5 * np.sum(H * TH0SQ)
EPS = 0.05 * L0
NETA = 400


def etas_for(a):
    emax = 2.0 / np.max(a * H)
    return emax * np.exp(np.linspace(np.log(1e-5), np.log(0.9999), NETA))


def last_iterate_steps(a, c, B, etas=None):
    """Least t with E L_t <= EPS over the eta grid (exact formula); returns (T, eta)."""
    etas = etas_for(a) if etas is None else etas
    lam = a * H
    best = (np.inf, None)
    for eta in etas:
        r = (1 - eta * lam) ** 2
        st = eta ** 2 * a ** 2 * c / B / (1 - r)
        if 0.5 * np.sum(H * st) >= EPS:
            continue
        def loss(t):
            return 0.5 * np.sum(H * (st + r ** t * (TH0SQ - st)))
        if loss(best[0] - 1 if np.isfinite(best[0]) else 10 ** 9) > EPS:
            continue
        lo, hi = 0, 1
        while loss(hi) > EPS:
            lo, hi = hi, 2 * hi
        while hi - lo > 1:
            m = (lo + hi) // 2
            if loss(m) > EPS:
                lo = m
            else:
                hi = m
        if hi < best[0]:
            best = (hi, eta)
    return best


def small_batch_limit(a, c):
    """n_a(eps) = min over tau of tau v(tau)/(eps - b(tau))."""
    lam = a * H
    best = np.inf
    for tau in np.exp(np.linspace(np.log(1e-2), np.log(1e6), 20000)):
        b = 0.5 * np.sum(H * TH0SQ * np.exp(-2 * lam * tau))
        if b >= EPS:
            continue
        v = 0.25 * np.sum(a * c * (-np.expm1(-2 * lam * tau)))
        best = min(best, tau * v / (EPS - b))
    return best


def deterministic_steps(a):
    lam = a * H
    emax = 2.0 / lam.max()
    best = np.inf
    for eta in emax * np.concatenate([np.linspace(0.01, 0.999, 600), 1 - np.logspace(-3, -7, 40)]):
        r = (1 - eta * lam) ** 2
        def bias(t):
            return 0.5 * np.sum(H * TH0SQ * r ** t)
        if bias(best - 1 if np.isfinite(best) else 10 ** 9) > EPS:
            continue
        lo, hi = 0, 1
        while bias(hi) > EPS:
            lo, hi = hi, 2 * hi
        while hi - lo > 1:
            m = (lo + hi) // 2
            if bias(m) > EPS:
                lo = m
            else:
                hi = m
        best = min(best, hi)
    return best


def avg_loss(a, c, eta, B, T, T1):
    """Exact expected loss of the average of theta_T1, ..., theta_{T-1} (W = T - T1 iterates)."""
    lam = a * H
    q = 1 - eta * lam                      # one-step factor
    s2 = eta ** 2 * a ** 2 * c / B         # injected variance per step
    W = T - T1
    mean_sq = TH0SQ * (q ** T1 * (1 - q ** W) / (W * (1 - q))) ** 2
    v1 = ((1 - q ** W) / (1 - q)) ** 2 * (1 - q ** (2 * T1)) / (1 - q ** 2) / W ** 2
    v2 = ((W - 1) - 2 * q * (1 - q ** (W - 1)) / (1 - q)
          + q ** 2 * (1 - q ** (2 * (W - 1))) / (1 - q ** 2)) / (W ** 2 * (1 - q) ** 2)
    return 0.5 * np.sum(H * (mean_sq + s2 * (v1 + v2)))


def averaged_steps(a, c, B):
    etas = 2.0 / np.max(a * H) * np.exp(np.linspace(np.log(1e-4), np.log(0.999), 60))
    fs = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]

    def ok(T):
        for eta in etas:
            for f in fs:
                if f >= 1.0:
                    r = (1 - eta * a * H) ** 2
                    st = eta ** 2 * a ** 2 * c / B / (1 - r)
                    val = 0.5 * np.sum(H * (st + r ** T * (TH0SQ - st)))
                else:
                    T1 = min(int(f * T), T - 1)
                    val = avg_loss(a, c, eta, B, T, T1)
                if val <= EPS:
                    return True
        return False
    lo, hi = 1, 2
    while not ok(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        m = (lo + hi) // 2
        if ok(m):
            hi = m
        else:
            lo = m
    return hi


def lindecay_steps(a, c, B):
    """Least T such that a linear decay eta_t = eta0 (1 - t/T) reaches the target at step T, for the
    best eta0: a coarse grid of 16 peak step sizes, then 16 values around the best one."""
    lam = a * H
    emax = 2.0 / lam.max()

    def final_losses(T, etas):
        m = np.tile(TH0SQ, (len(etas), 1))
        E = np.asarray(etas)[:, None]
        for t in range(T):
            e = E * (1 - t / T)
            m = (1 - e * lam) ** 2 * m + e ** 2 * (a ** 2 * c / B)
        return 0.5 * (m @ H)

    def least_T(etas, lo=1, hi=2):
        while np.min(final_losses(hi, etas)) > EPS:
            lo, hi = hi, 2 * hi
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if np.min(final_losses(mid, etas)) <= EPS:
                hi = mid
            else:
                lo = mid
        return hi
    coarse = emax * np.exp(np.linspace(np.log(1e-4), np.log(0.999), 16))
    T1 = least_T(coarse)
    best = coarse[int(np.argmin(final_losses(T1, coarse)))]
    fine = np.exp(np.linspace(np.log(best / 2), np.log(min(2 * best, 0.999 * emax)), 16))
    fine = np.unique(np.concatenate([fine, [best]]))
    return least_T(fine, lo=0, hi=T1)


def mc_steps(a, B, eta, R, Tmax, rng, opt="sgd", b2=0.999):
    th = np.ones((R, D))
    v = np.zeros((R, D))
    sd = np.sqrt(C_PROP / B)
    hist = []
    for t in range(1, Tmax + 1):
        g = H * th + sd * rng.standard_normal((R, D))
        if opt == "sgd":
            th = th - eta * a * g
        else:
            v = b2 * v + (1 - b2) * g * g
            th = th - eta * g / np.sqrt(v / (1 - b2 ** t))
        hist.append(0.5 * np.mean(th ** 2 @ H))
        w = max(1, int(0.02 * t))
        if np.mean(hist[-w:]) <= EPS:
            return t
    return np.nan


def main():
    a0 = np.ones(D)
    ah = H ** -0.5
    Bs = 2.0 ** np.arange(0, 14)
    rows = []
    nsgd, npre = small_batch_limit(a0, C_PROP), small_batch_limit(ah, C_PROP)
    dsgd, dpre = deterministic_steps(a0), deterministic_steps(ah)
    nsgd_iso, npre_iso = small_batch_limit(a0, C_ISO), small_batch_limit(ah, C_ISO)
    opt_eta = {}
    for B in Bs:
        Ts, es = last_iterate_steps(a0, C_PROP, B)
        Tp, ep = last_iterate_steps(ah, C_PROP, B)
        opt_eta[B] = (es, ep)
        Tsi, _ = last_iterate_steps(a0, C_ISO, B)
        Tpi, _ = last_iterate_steps(ah, C_ISO, B)
        Tsa, Tpa = averaged_steps(a0, C_PROP, B), averaged_steps(ah, C_PROP, B)
        if int(B) in (1, 4, 16, 64, 256, 1024, 4096):
            Tsl, Tpl = lindecay_steps(a0, C_PROP, B), lindecay_steps(ah, C_PROP, B)
        else:
            Tsl, Tpl = np.nan, 1.0
        rows.append([B, Ts, Tp, nsgd / B, npre / B, dsgd, dpre, Ts / Tp, Tsa / Tpa, Tsl / Tpl,
                     Tsi / Tpi, Tsa, Tpa, Tsl, Tpl])
        print("B %5d  sgd %6d pre %5d  ratio %.2f  avg %.2f  lin %.2f  iso %.2f" %
              (B, Ts, Tp, Ts / Tp, Tsa / Tpa, Tsl / Tpl, Tsi / Tpi), flush=True)
    head = ("B sgd pre sgdsmall presmall sgdbig prebig rconst ravg rlin riso "
            "sgdavg preavg sgdlin prelin")
    with open(os.path.join(DATA, "14-transformers-batch.dat"), "w") as f:
        f.write(head + "\n")
        for r in rows:
            f.write(" ".join("%.6g" % x for x in r) + "\n")
    print("small-batch limits n_sgd = %.1f, n_pre = %.1f, ratio %.3f" % (nsgd, npre, nsgd / npre))
    print("isotropic small-batch ratio %.3f" % (nsgd_iso / npre_iso))
    print("large-batch limits %d %d ratio %.2f" % (dsgd, dpre, dsgd / dpre))

    rng = np.random.default_rng(14)
    mc = []
    for B in [1, 4, 16, 64, 256, 1024, 4096]:
        es, ep = opt_eta[float(B)]
        Ts = rows[int(np.log2(B))][1]
        Tp = rows[int(np.log2(B))][2]
        ms = mc_steps(a0, B, es, 32, int(2 * Ts), rng)
        mp = mc_steps(ah, B, ep, 32, int(2 * Tp) + 5, rng)
        mr = np.nan
        if B <= 64:
            best = np.inf
            for fac in [0.7, 1.0, 1.4, 2.0]:
                val = mc_steps(None, B, fac * ep / np.sqrt(B), 16, int(2 * Tp) + 5, rng, "rms")
                if np.isfinite(val) and val < best:
                    best = val
            mr = best
        mc.append([B, ms, mp, mr])
        print("MC B %5d  sgd %6s (exact %d)  pre %5s (exact %d)  rmsprop %s" % (B, ms, Ts, mp, Tp, mr),
              flush=True)
    with open(os.path.join(DATA, "14-transformers-batch-mc.dat"), "w") as f:
        f.write("B mcsgd mcpre mcrms\n")
        for r in mc:
            f.write(" ".join("nan" if not np.isfinite(x) else "%.6g" % x for x in r) + "\n")


if __name__ == "__main__":
    main()
