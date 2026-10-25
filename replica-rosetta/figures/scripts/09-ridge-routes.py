"""Chapter 9 (ridge): the limiting ridge risk computed by four routes, and the curves of the figure.

Book conventions: x ~ N(0, I_d), y = <x, beta*> + sigma eps, |beta*|^2 = r^2, alpha = n/d,
beta_hat = argmin (1/2n)|y - X b|^2 + (lam/2)|b|^2, excess risk R = |beta_hat - beta*|^2.

The four routes (all in the proportional limit):
  (1) replicas: the replica-symmetric saddle point, solved by damped fixed-point iteration of
        chi <- 1/(alpha lam + alpha/(1+chi)),  m <- alpha r^2 chi/(1+chi),
        q   <- m^2/r^2 + alpha chi^2 (r^2 + sigma^2 - 2m + q)/(1+chi)^2,   R = q - 2m + r^2;
  (2) random matrices (the theorem of the chapter): chi = lim tr (X'X + n lam)^{-1} in closed form,
        R = -alpha r^2 lam^2 chi'(lam) + sigma^2 (chi + lam chi'(lam));
  (3) Gordon's auxiliary problem (CGMT): min_{a>=0} max_{theta>=0} F(a, theta),
        F = theta sqrt(a^2 + sigma^2) - a sqrt(theta^2/alpha + lam^2 r^2) - theta^2/2 + (lam/2)(r^2 + a^2),
        R = a*^2, training error theta*^2;
  (4) Hastie-Montanari-Rosset-Tibshirani (Cor. 6 of the journal version, Cor. 5 of arXiv v5), with the
      Marchenko-Pastur Stieltjes transform of
      the kit: R = r^2 lam^2 m'(-lam) + sigma^2 gamma (m(-lam) - lam m'(-lam)), gamma = 1/alpha.
Writes figures/data/09-ridge-curves.dat and figures/data/09-ridge-routes.dat. Runtime: a few seconds.
Run from the volume root: python figures/scripts/09-ridge-routes.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk

R2, S2 = 1.0, 0.25          # r^2 and sigma^2 (signal-to-noise ratio 4)
LAM_SMALL = 0.01            # the small ridge of the figure
N_SIM = 1000                # the sample size of the simulation (for the exact finite-n ridgeless mean)


def chi_closed(alpha, lam):
    """Positive root of alpha lam chi^2 + (alpha lam + alpha - 1) chi - 1 = 0."""
    b = alpha * lam + alpha - 1.0
    return (-b + math.sqrt(b * b + 4.0 * alpha * lam)) / (2.0 * alpha * lam)


def chi_prime(alpha, lam):
    """d chi / d lam = -alpha chi^2 (1 + chi)/(1 + alpha lam chi^2) (implicit differentiation)."""
    c = chi_closed(alpha, lam)
    return -alpha * c * c * (1.0 + c) / (1.0 + alpha * lam * c * c)


def risk_rmt(alpha, lam, r2=R2, s2=S2):
    """Route (2): the theorem, R = -alpha r^2 lam^2 chi' + sigma^2 (chi + lam chi')."""
    c, cp = chi_closed(alpha, lam), chi_prime(alpha, lam)
    return -alpha * r2 * lam * lam * cp + s2 * (c + lam * cp)


def risk_closed(alpha, lam, r2=R2, s2=S2):
    """The same, as (r^2 (alpha lam chi)^2 + sigma^2 A)/(1 - A), A = alpha chi^2/(1+chi)^2; also chi
    and the training error (R + sigma^2)/(1+chi)^2."""
    c = chi_closed(alpha, lam)
    A = alpha * c * c / (1.0 + c) ** 2
    R = (r2 * (alpha * lam * c) ** 2 + s2 * A) / (1.0 - A)
    return R, c, (R + s2) / (1.0 + c) ** 2


def risk_replica(alpha, lam, r2=R2, s2=S2):
    """Route (1): the replica-symmetric saddle point by damped fixed-point iteration."""
    def F(v):
        chi, m, q = v
        chi_n = 1.0 / (alpha * lam + alpha / (1.0 + chi))
        m_n = alpha * r2 * chi_n / (1.0 + chi_n)
        q_n = m_n * m_n / r2 + alpha * chi_n ** 2 * (r2 + s2 - 2.0 * m_n + q) / (1.0 + chi_n) ** 2
        return np.array([chi_n, m_n, q_n])
    v, it, ok = rk.fixed_point(F, np.array([1.0, 0.5, 0.5]), damping=0.7, tol=1e-14, maxit=200000)
    if not ok:
        raise RuntimeError("replica fixed point did not converge at alpha=%g lam=%g" % (alpha, lam))
    chi, m, q = v
    return q - 2.0 * m + r2, chi, m, q, it


def risk_cgmt(alpha, lam, r2=R2, s2=S2):
    """Route (3): the scalar auxiliary problem, by nested one-dimensional optimization."""
    def F(a, th):
        return (th * math.sqrt(a * a + s2) - a * math.sqrt(th * th / alpha + lam * lam * r2)
                - th * th / 2.0 + lam / 2.0 * (r2 + a * a))

    def inner(a):
        res = optimize.minimize_scalar(lambda th: -F(a, th), bounds=(0.0, 20.0), method="bounded",
                                       options={"xatol": 1e-13})
        return -res.fun, res.x
    res = optimize.minimize_scalar(lambda a: inner(a)[0], bounds=(0.0, 20.0), method="bounded",
                                   options={"xatol": 1e-13})
    a = res.x
    val, th = inner(a)
    return a * a, th * th, val


def risk_hmrt(alpha, lam, r2=R2, s2=S2):
    """Route (4): HMRT's Stieltjes-transform form (kit)."""
    return rk.ridge_risk(alpha, lam, r2, s2)


def ridgeless(alpha, r2=R2, s2=S2):
    """The limits lam -> 0: sigma^2/(alpha-1) for alpha > 1, r^2(1-alpha) + sigma^2 alpha/(1-alpha) for
    alpha < 1; the training error sigma^2 (1 - 1/alpha) for alpha > 1 and 0 for alpha < 1."""
    if alpha > 1:
        return s2 / (alpha - 1.0), s2 * (1.0 - 1.0 / alpha)
    return r2 * (1.0 - alpha) + s2 * alpha / (1.0 - alpha), 0.0


def ridgeless_exact_mean(n, d, r2=R2, s2=S2):
    """Exact mean of the min-norm risk for Gaussian design at finite n, d (the chapter's lemma):
    r^2 (1 - n/d)_+ + sigma^2 d/(n-d-1) if n >= d+2, and + sigma^2 n/(d-n-1) if d >= n+2."""
    if n >= d + 2:
        return s2 * d / (n - d - 1.0)
    if d >= n + 2:
        return r2 * (1.0 - n / d) + s2 * n / (d - n - 1.0)
    return float("nan")


def main():
    t0 = time.time()
    print("Chapter 9: ridge risk by four routes; r^2 = %g, sigma^2 = %g" % (R2, S2))

    # ---- the table: four routes at a few (alpha, lambda) ----
    rows = []
    worst = 0.0
    worst_exact = 0.0       # replica fixed point vs random-matrix closed form vs Stieltjes form
    worst_cgmt = 0.0        # the nested optimization of the auxiliary problem vs the closed form
    worst_T = 0.0           # training error: theta*^2 of the auxiliary problem vs (R + sigma^2)/(1+chi)^2
    print("\n alpha   lambda     replica       random-matrix  CGMT (AO)      HMRT (Stieltjes)  max|diff|  chi        iterations")
    for alpha in [0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0]:
        for lam in [LAM_SMALL, 0.1, S2 / (alpha * R2)]:
            R1, chi1, m1, q1, it = risk_replica(alpha, lam)
            R2_ = risk_rmt(alpha, lam)
            R3, T3, _ = risk_cgmt(alpha, lam)
            R4 = risk_hmrt(alpha, lam)
            Rc, c, T = risk_closed(alpha, lam)
            diff = max(abs(R1 - R2_), abs(R1 - R3), abs(R1 - R4), abs(R1 - Rc))
            worst = max(worst, diff)
            worst_exact = max(worst_exact, abs(R1 - R2_), abs(R1 - R4), abs(R2_ - R4), abs(R1 - Rc))
            worst_cgmt = max(worst_cgmt, abs(R3 - R2_))
            worst_T = max(worst_T, abs(T3 - T))
            print(" %5.2f  %7.4f  %13.10f  %13.10f  %13.10f  %13.10f  %8.1e  %9.6f  %d" % (
                alpha, lam, R1, R2_, R3, R4, diff, chi1, it))
            rows.append((alpha, lam, R1, R2_, R3, R4, diff, chi1, m1, q1, T, T3))
    print("largest disagreement between the four routes: %.1e" % worst)
    print("  replica / random-matrix / Stieltjes forms among themselves: %.1e" % worst_exact)
    print("  auxiliary problem (nested optimization, tolerance 1e-13 in each variable) vs closed form: %.1e" % worst_cgmt)
    print("  training error from the auxiliary problem vs (R + sigma^2)/(1+chi)^2: %.1e" % worst_T)
    rk.write_dat("figures/data/09-ridge-routes.dat",
                 ["alpha", "lam", "Rreplica", "Rrmt", "Rcgmt", "Rhmrt", "maxdiff", "chi", "m", "q", "Ttrain", "Tcgmt"],
                 rows)

    # ---- facts quoted in the text ----
    print("\nOptimal ridge lam* = sigma^2/(alpha r^2): R* = sigma^2 chi(lam*), and m = q (Nishimori)")
    for alpha in [0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0]:
        ls = S2 / (alpha * R2)
        R, chi, m, q, _ = risk_replica(alpha, ls)
        lnum = optimize.minimize_scalar(lambda L: risk_closed(alpha, L)[0], bounds=(1e-5, 20.0),
                                        method="bounded", options={"xatol": 1e-12}).x
        print("  alpha=%5.2f  lam*=%.6f  numerical argmin=%.6f  R*=%.6f  sigma^2 chi=%.6f  m=%.8f  q=%.8f" % (
            alpha, ls, lnum, R, S2 * chi, m, q))
    print("\nRidgeless limits against lam -> 0 of the formula (lam = 1e-9):")
    for alpha in [0.25, 0.5, 0.8, 1.25, 2.0, 4.0]:
        print("  alpha=%5.2f  limit=%.6f  formula at lam=1e-9: %.6f" % (alpha, ridgeless(alpha)[0], risk_closed(alpha, 1e-9)[0]))
    snr = R2 / S2
    print("  local minimum of the ridgeless curve on alpha < 1 at alpha = 1 - 1/sqrt(SNR) = %.4f" % (1 - 1 / math.sqrt(snr)))
    print("  peak height of the lam = %.2f curve: max_alpha R = %.4f at alpha = %.3f" % (
        LAM_SMALL, *max((risk_closed(a, LAM_SMALL)[0], a) for a in np.linspace(0.5, 1.5, 2001))))
    print("  Krogh-Hertz Fig. 1 (r^2 = 1, sigma^2 = 0.2): optimal-decay risk at alpha = 1, 2: %.4f, %.4f" % (
        risk_closed(1.0, 0.2, 1.0, 0.2)[0], risk_closed(2.0, 0.1, 1.0, 0.2)[0]))

    # ---- the curves of the figure ----
    curves = []
    for alpha in np.exp(np.linspace(math.log(0.25), math.log(5.0), 241)):
        alpha = float(alpha)
        R0, T0 = ridgeless(alpha) if abs(alpha - 1.0) > 1e-9 else (float("inf"), 0.0)
        R1, c1, T1 = risk_closed(alpha, LAM_SMALL)
        ls = S2 / (alpha * R2)
        Rs, cs, Ts = risk_closed(alpha, ls)
        d = int(round(N_SIM / alpha))
        R0fin = ridgeless_exact_mean(N_SIM, d)
        clip = lambda v: v if (np.isfinite(v) and v <= 3.0) else float("nan")
        curves.append((alpha, clip(R0), R1, Rs, T0, T1, Ts, clip(R0fin), c1, cs))
    rk.write_dat("figures/data/09-ridge-curves.dat",
                 ["alpha", "R0", "R1", "Rs", "T0", "T1", "Ts", "R0fin", "chi1", "chis"], curves)
    print("\nwrote figures/data/09-ridge-curves.dat (%d rows) and figures/data/09-ridge-routes.dat (%d rows)" % (
        len(curves), len(rows)))
    print("runtime %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
