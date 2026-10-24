"""Chapter 13 (kacrice): complexity curves of the pure spherical p-spin and of the spiked tensor model,
and every constant the chapter quotes.

Normalization (OUTLINE.md): H_N on the sphere |s|^2 = N with E H_N(s) H_N(s') = N R(s,s')^p
(Auffinger--Ben Arous--Cerny); energies per spin u = H_N/N; threshold -E_inf, ground state -E_0.
Spiked tensor (maximum likelihood for tensor PCA): H_{N,lam}(s) = H_N(s) - lam N m(s)^p, m = <s,x*>/N;
this lam is that of Jagannath--Lopatto--Miolane and Ben Arous--Gheissari--Jagannath, and sqrt(2) times
that of Ben Arous--Mei--Montanari--Nica (their f = -H/(sqrt(2) N), lam_BAMMN = lam/sqrt 2).

Writes
  figures/data/13-kacrice-complexity.dat  u, total, minima, idx1, idx2, idx10 (ABC Theta functions)
  figures/data/13-kacrice-spiked.dat      m, S0 and S (minima, all critical points) at four lam
Prints E_k, the CS95 / ABC Remark 2.7 normalization check, the zero-temperature 1RSB value of E_0,
lam_c(p), lam_ML(p) and the overlaps m*(lam).   Runs in a few seconds.
"""
import math
import os
import sys

import numpy as np
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


# ----------------------------------------------------------------------------------------------
# GOE facts in the book's normalization (semicircle on [-2, 2])
# ----------------------------------------------------------------------------------------------
def J(x):
    """Rate function of the largest GOE eigenvalue, x >= 2: (x/4) sqrt(x^2-4) - log((x+sqrt(x^2-4))/2)."""
    x = abs(x)
    if x <= 2:
        return 0.0
    r = math.sqrt(x * x - 4)
    return x / 4 * r - math.log((x + r) / 2)


def Phi(t):
    """Log-potential of the semicircle: int log|y - t| sigma_sc(dy) = t^2/4 - 1/2 - J(|t|)."""
    return t * t / 4 - 0.5 - J(t)


def Einf(p):
    return 2 * math.sqrt((p - 1) / p)


def tau(u, p):
    """Energy per spin -> eigenvalue variable t = u sqrt(p/(p-1)) = 2u/E_inf."""
    return u * math.sqrt(p / (p - 1))


# ----------------------------------------------------------------------------------------------
# Pure model (Theorem thm:kacrice:annealed and ABC Theorem 2.5)
# ----------------------------------------------------------------------------------------------
def I1(u, p):
    return J(tau(u, p))


def theta_total(u, p):
    """lim (1/N) log E Crt_N((-inf, u))."""
    if u >= 0:
        return 0.5 * math.log(p - 1)
    return 0.5 * math.log(p - 1) - (p - 2) / (4 * (p - 1)) * u * u - (I1(u, p) if u <= -Einf(p) else 0.0)


def theta_k(u, p, k):
    """lim (1/N) log E Crt_{N,k}((-inf, u)) (index k; k = 0: local minima)."""
    if u >= -Einf(p):
        return 0.5 * math.log(p - 1) - (p - 2) / p
    return 0.5 * math.log(p - 1) - (p - 2) / (4 * (p - 1)) * u * u - (k + 1) * I1(u, p)


def density(u, p):
    """Complexity density of critical points at energy u (Kac-Rice integrand, any index)."""
    return 0.5 * (1 + math.log(p - 1)) - u * u / 2 + Phi(tau(u, p))


def E_k(p, k):
    return -rk.root(lambda u: theta_k(u, p, k), -10.0, -Einf(p) - 1e-13)


def cs95(E, p):
    """Crisanti--Sommers TAP complexity of minima in physics energies E, as printed in ABC eq. (2.18)."""
    z = (-E - math.sqrt(E * E - 2 * (p - 1) / p)) / (p - 1)
    return 0.5 * ((2 - p) / p - math.log(p * z * z / 2) + (p - 1) / 2 * z * z - 2 / (p * p * z * z))


def gs_1rsb_T0(p):
    """ABC (5.11): inf_{c,d>0} (1/2)[c + p d + (1/c) log(1 + c/d)] (zero-temperature 1RSB)."""
    f = lambda v: 0.5 * (math.exp(v[0]) + p * math.exp(v[1]) + math.exp(-v[0]) * math.log1p(math.exp(v[0] - v[1])))
    best = min((optimize.minimize(f, [a, b], method="Nelder-Mead",
                                  options=dict(xatol=1e-12, fatol=1e-14, maxiter=40000))
                for a in (-1.0, 0.0, 1.0) for b in (-2.0, -1.0, 0.0)), key=lambda r: r.fun)
    return best.fun, math.exp(best.x[0]), math.exp(best.x[1])


# ----------------------------------------------------------------------------------------------
# Spiked tensor (Ben Arous--Mei--Montanari--Nica, Theorems 1-2, in the book's normalization)
# ----------------------------------------------------------------------------------------------
def Jv(x):
    """Vectorized J(|x|), zero inside [-2, 2]."""
    x = np.maximum(np.abs(np.asarray(x, dtype=float)), 2.0)
    r = np.sqrt(x * x - 4)
    return x / 4 * r - np.log((x + r) / 2)


def sigma_spiked(m, u, lam, p):
    """Complexity of critical points of H_{N,lam} with overlap m and energy u (u may be an array)."""
    t = np.asarray(u, dtype=float) * math.sqrt(p / (p - 1))
    return (0.5 * (1 + math.log(p - 1)) + 0.5 * math.log(1 - m * m)
            - 0.5 * p * lam ** 2 * m ** (2 * p - 2) * (1 - m * m)
            - 0.5 * (np.asarray(u) + lam * m ** p) ** 2 + t * t / 4 - 0.5 - Jv(t))


def L_outlier(theta, s):
    """BAMMN (12), vectorized in s: the cost of pushing the outlier b = theta + 1/theta of GOE + theta e e^T
    below s; with int_b^s sqrt(y^2-4) dy = 2 J(s) - 2 J(b) it reads J(s)/2 - J(b)/2 - theta (s-b)/2 + (s^2-b^2)/8."""
    s = np.asarray(s, dtype=float)
    out = np.where(s < 2, np.inf, 0.0)
    if theta > 1:
        b = theta + 1 / theta
        mid = (s >= 2) & (s < b)
        val = Jv(s) / 2 - Jv(b) / 2 - 0.5 * theta * (s - b) + (s * s - b * b) / 8
        out = np.where(mid, val, out)
    return out


def sigma0_spiked(m, u, lam, p):
    """Complexity of local minima of H_{N,lam} (m >= 0): the Hessian is N sqrt(p(p-1)) (A - t - theta e e^T)."""
    theta = lam * math.sqrt(p * (p - 1)) * m ** (p - 2) * (1 - m * m)
    t = np.asarray(u, dtype=float) * math.sqrt(p / (p - 1))
    return sigma_spiked(m, u, lam, p) - L_outlier(theta, -t)


def maximize_u(fun, lo=-8.0, hi=4.0, n=6001, extra=()):
    """max_u fun(u) on a grid of spacing 0.002, refined by a parabola through the best three points;
    `extra` lists boundary points of the domain (for minima, u = -E_inf) evaluated as candidates."""
    us = np.linspace(lo, hi, n)
    vals = fun(us)
    i = int(np.argmax(vals))
    best, arg = float(vals[i]), float(us[i])
    if np.isfinite(best) and 0 < i < n - 1 and np.all(np.isfinite(vals[i - 1:i + 2])):
        y0, y1, y2 = vals[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        if den < 0:
            u = us[i] + 0.5 * (y0 - y2) / den * (us[1] - us[0])
            v = float(fun(np.array([u]))[0])
            if v > best:
                best, arg = v, float(u)
    for u in extra:
        v = float(fun(np.array([u]))[0])
        if v > best:
            best, arg = v, float(u)
    return best, arg


def lam_c(p):
    return math.sqrt((p - 1) ** (p - 1) / (p * (p - 2) ** (p - 2)))


def m_star(lam, p):
    """Largest root of m^{2p-4}(1 - m^2) = 1/(p lam^2) (BAMMN Prop. 2; = sqrt(q*) of JLM above lam_ML)."""
    g = lambda m: m ** (2 * p - 4) * (1 - m * m) - 1 / (p * lam * lam)
    mtop = math.sqrt((p - 2) / (p - 1))
    if g(mtop) < 0:
        return float("nan")
    return rk.root(g, mtop, 1.0 - 1e-15)


def lam_ML(p):
    """Jagannath--Lopatto--Miolane threshold: sup{lam : sup_t lam^2 t^p + log(1-t) + t <= 0};
    equivalently q_s solves q^2/(p(1-q)) + q + log(1-q) = 0 and lam^2 = 1/(p q^{p-2}(1-q))."""
    g = lambda q: q * q / (p * (1 - q)) + q + math.log(1 - q)
    grid = np.linspace(1e-4, 1 - 1e-6, 20001)
    vals = [g(q) for q in grid]
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if fa < 0 <= fb:
            qs = rk.root(g, a, b)
            return math.sqrt(1 / (p * qs ** (p - 2) * (1 - qs))), qs
    raise RuntimeError("no static root")


def main():
    p = 3
    print("== constants (ABC normalization) ==")
    for q in (3, 4, 5):
        print(f"p={q}: E_inf={Einf(q):.10f} E_0={E_k(q,0):.10f} E_1={E_k(q,1):.10f} E_2={E_k(q,2):.10f}"
              f" E_10={E_k(q,10):.10f}; 1/2 log(p-1)={0.5*math.log(q-1):.6f};"
              f" Theta_0(-E_inf)={0.5*math.log(q-1)-(q-2)/q:.6f}")
    print(f"physics (CK) normalization, p=3: threshold {-Einf(3)/math.sqrt(2):.6f}, ground state {-E_k(3,0)/math.sqrt(2):.6f}")
    print("== ABC Remark 2.7 vs Lemma 6.3: g(E) against Theta_0(sqrt2 E) and Theta_0(E/sqrt2) ==")
    for E in (-1.16, -1.17, -1.20):
        print(f"  p=3 E={E}: g={cs95(E,3):.10f}  Theta_0(sqrt2 E)={theta_k(math.sqrt(2)*E,3,0):.10f}"
              f"  Theta_0(E/sqrt2)={theta_k(E/math.sqrt(2),3,0):.10f}")
    for q in (3, 4):
        v, c, d = gs_1rsb_T0(q)
        print(f"== zero-temperature 1RSB, p={q}: inf = {v:.10f} at c={c:.6f}, d={d:.6f}; E_0 = {E_k(q,0):.10f}")
    for q in (3, 4, 5):
        # Jagannath--Lopatto--Miolane (1.6)-(1.7): z_k zero of (1+z)/z^2 log(1+z) - 1/z - 1/k
        z = rk.root(lambda z: (1 + z) / z ** 2 * math.log1p(z) - 1 / z - 1 / q, 1e-6, 1e6)
        print(f"== JLM ground state GS_{q} = {math.sqrt(q) / math.sqrt(1 + z) * (1 + z / q):.12f}; E_0 = {E_k(q,0):.12f}")
    print("== spiked tensor ==")
    for q in (3, 4, 5):
        lml, qs = lam_ML(q)
        print(f"p={q}: lam_c={lam_c(q):.6f} (BAMMN normalization {lam_c(q)/math.sqrt(2):.6f}); "
              f"m_c(lam_c)={((q-2)/(lam_c(q)*math.sqrt(q*(q-1))))**(1/q):.6f}; m*(lam_c)={m_star(lam_c(q)+1e-12,q):.6f};"
              f" lam_ML={lml:.6f} (q_s={qs:.6f}, m*(lam_ML)={m_star(lml,q):.6f})")
    for lam in (1.5, 2.0, 3.0, 5.0, 3 * math.sqrt(2)):
        print(f"  p=3 lam={lam:.4f}: m*={m_star(lam,3):.6f}")
    # check BAMMN Prop 1/2 at lam_BAMMN = 3 (our lam = 3 sqrt 2): S(m) = S_U for m < m_c, S_G zero at m*
    lam = 3 * math.sqrt(2)
    mc = ((p - 2) / (lam * math.sqrt(p * (p - 1)))) ** (1 / p)
    SU = lambda m: 0.5 * math.log(p - 1) + 0.5 * math.log(1 - m * m) - 0.5 * p * lam ** 2 * m ** (2 * p - 2) * (1 - m * m) + p * lam ** 2 * m ** (2 * p) / (2 * (p - 2))
    for m in (0.0, 0.2, 0.4, mc - 1e-3):
        print(f"  check S_U: m={m:.4f} max_u Sigma={maximize_u(lambda u: sigma_spiked(m,u,lam,p))[0]:.8f} S_U={SU(m):.8f}")
    ms = m_star(lam, p)
    print(f"  m_c={mc:.6f}, m*={ms:.6f}, max_u Sigma(m*)={maximize_u(lambda u: sigma_spiked(ms,u,lam,p))[0]:.2e}")
    print(f"  at m=0: S = {maximize_u(lambda u: sigma_spiked(0.0,u,lam,p))[0]:.6f} (=1/2 log 2),"
          f" S0 = {maximize_u(lambda u: sigma0_spiked(0.0,u,lam,p), extra=(-Einf(p) - 1e-12,))[0]:.6f} (= Theta_0(-E_inf))")

    # ---- data: pure model, p = 3
    rows = []
    for u in np.linspace(-1.70, 0.0, 341):
        u = float(u)
        rows.append([u, theta_total(u, p), theta_k(u, p, 0), theta_k(u, p, 1), theta_k(u, p, 2), theta_k(u, p, 10),
                     density(u, p)])
    rk.write_dat(os.path.join(OUT, "13-kacrice-complexity.dat"),
                 ["u", "total", "minima", "idx1", "idx2", "idx10", "density"], rows)
    # physics: CS95 complexity of minima, converted to ABC energies (u = sqrt2 E), below the threshold
    rows = []
    for u in np.linspace(-1.70, -Einf(p) - 1e-9, 121):
        rows.append([float(u), cs95(float(u) / math.sqrt(2), p)])
    rk.write_dat(os.path.join(OUT, "13-kacrice-cs95.dat"), ["u", "g"], rows)
    # ---- data: spiked, p = 3, minima and all critical points against the overlap
    lams = [0.8, lam_c(p), lam_ML(p)[0], 2.0]
    rows = []
    extra_m = [m_star(l + 1e-12, p) for l in lams[1:]]
    mgrid = sorted(set(np.linspace(0.0, 0.995, 300).tolist() + [m for m in extra_m if np.isfinite(m)]))
    for m in mgrid:
        m = float(m)
        row = [m]
        for lam in lams:
            row.append(maximize_u(lambda u: sigma0_spiked(m, u, lam, p), extra=(-Einf(p) - 1e-12,))[0])
        for lam in lams:
            row.append(maximize_u(lambda u: sigma_spiked(m, u, lam, p))[0])
        rows.append([v if np.isfinite(v) else -1.0 for v in row])
    rk.write_dat(os.path.join(OUT, "13-kacrice-spiked.dat"),
                 ["m", "min0", "min1", "min2", "min3", "all0", "all1", "all2", "all3"], rows)
    print("spiked curves at lam =", [round(l, 6) for l in lams])
    for k, lam in enumerate(lams):
        col = np.array([r[1 + k] for r in rows])
        good = [r[0] for r in rows if r[0] > 0.5 and r[1 + k] > -0.002]
        print(f"  lam={lam:.4f}: S0(0)={col[0]:.6f}, max over m>0.5 of S0 = {max(r[1+k] for r in rows if r[0]>0.5):.5f}"
              f" (m*={m_star(lam,p) if lam>=lam_c(p)-1e-9 else float('nan'):.4f})")


if __name__ == "__main__":
    main()
