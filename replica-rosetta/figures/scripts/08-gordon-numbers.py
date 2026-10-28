"""Chapter 8 (Gaussian comparison): every formula-based number the chapter quotes.

(a) Gardner's capacity alpha_c(kappa) = 1/rho(kappa), rho(kappa) = E(Z + kappa)_+^2
    = (1 + kappa^2) Phi(kappa) + kappa phi(kappa), checked by quadrature; the LP threshold
    1/Phi(kappa) of Montanari--Zhong--Zhou (2024, Thm 4.3) for kappa < 0.
(b) The replica-symmetric saddle point of Gardner's volume, q/(1-q) = alpha E R(y)^2 with
    y = (kappa - sqrt(q) Z)/sqrt(1-q) and R = phi/H, and the approach q -> 1 at capacity.
(c) The de Almeida--Thouless (replicon) quantity lambda = alpha E[R(y)^2 (R(y) - y)^2]; its value
    alpha_c(kappa) Phi(kappa) at capacity, and the dAT line alpha_dAT(kappa) for kappa < 0.
(d) The lifted Gordon bound (Stojnic 2013b; Montanari--Zhong--Zhou 2024, Definition 2 and Thm 4.2),
    evaluated in the form of their condition (74): delta is excluded if some s > 0 has
      1/(sqrt(s^2+4)+s) + (1/s) log((sqrt(s^2+4)+s)/2) < inf_t [1/(4t) + (delta/s) l(st)],
    l(u) = -log E exp(-u (kappa - Z)_+^2).  The second-order expansion at delta = alpha_c(kappa)
    says the lifted bound beats Gardner's value near alpha_c iff Var X < 2 E X, X = (Z+kappa)_+^2.
(e) The Ising perceptron at kappa = 0: the replica-symmetric q -> 1 "capacity" 2/(pi rho(kappa))
    (4/pi at kappa = 0), the Krauth--Mezard zero of the RS entropy, and the conditional first moment
    bound of Krauth--Mezard / Altschuler--Tikhomirov: GD(alpha) = -log 2, with the value at q = 1/2,
    GD(alpha, 1/2) = -alpha + (1 - log 2)/2, which gives alpha = (1 + log 2)/2.
(f) The symmetric binary perceptron: first-moment capacity -log 2 / log P(|Z| <= kappa).

Writes figures/data/08-gordon-phase.dat (kappa grid: alpha_c, alpha_dAT, alpha_lin, alpha_lifted)
and figures/data/08-gordon-curves.dat (alpha grid: kappa_RS, kappa_lin, kappa_lifted).
Runtime: about 30 s.  Run from the volume root: python figures/scripts/08-gordon-numbers.py
"""
import math
import os
import sys
import time

import numpy as np
from scipy import integrate, optimize, special, stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

T0 = time.time()
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
Phi = special.ndtr
SQ2PI = math.sqrt(2 * math.pi)


def phi(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / SQ2PI


def rho(k):
    """E (Z + k)_+^2 in closed form."""
    return (1 + k * k) * Phi(k) + k * phi(k)


def alpha_c(k):
    return 1.0 / rho(k)


def mills(y):
    """R(y) = phi(y)/H(y) = sqrt(2/pi)/erfcx(y/sqrt 2) (inverse Mills ratio), stable for all y."""
    y = np.maximum(np.asarray(y, dtype=float), -37.0)
    return math.sqrt(2 / math.pi) / special.erfcx(y / math.sqrt(2))


# ------------------------------------------------------------------------------------------
print("(a) Gardner's capacity alpha_c(kappa) = 1/E(Z+kappa)_+^2, and 1/Phi(kappa)")
for k in [-1.1, -1.0, -0.9, -0.8, -0.7, -0.63, -0.6, -0.5, 0.0, 0.5, 1.0, 2.0]:
    quad = 1.0 / rk.gauss_split(lambda z: np.maximum(z + k, 0.0) ** 2, a=-k)
    print(f"   kappa={k:6.2f}  alpha_c={alpha_c(k):11.6f}  (quadrature {quad:11.6f})"
          f"  1/Phi(kappa)={1/Phi(k):9.5f}  alpha_c*Phi(kappa)={alpha_c(k)*Phi(k):.5f}")


def kappa_star(a):
    """The margin kappa with alpha_c(kappa) = a (alpha_c decreases strictly from inf to 0)."""
    return rk.root(lambda k: alpha_c(k) - a, -8.0, 12.0)


for a in [0.5, 1.0, 1.5, 2.0, 4.0, 8.0]:
    print(f"   kappa*(alpha={a}) = {kappa_star(a):.5f}   LP margin Phi^-1(1/alpha) = "
          f"{stats.norm.ppf(1/a) if a > 1 else float('nan'):.5f}")


# ------------------------------------------------------------------------------------------
def esplit(f, a):
    """E f(Z) for f smooth except near the point a (split there when a lies inside [-11, 11])."""
    return rk.gauss_split(f, a=min(max(a, -11.0), 11.0), n=400)


def rs_q(a, k):
    """Root in (0,1) of q/(1-q) = a E R((k - sqrt(q) Z)/sqrt(1-q))^2 (Gardner's RS equation),
    written as F(q) = a (1-q) E R^2 - q, which tends to a R(k)^2 > 0 as q -> 0 and to
    a rho(k) - 1 < 0 as q -> 1 when a < alpha_c(k)."""
    def F(q):
        s, c = math.sqrt(q), math.sqrt(1 - q)
        val = esplit(lambda z: mills((k - s * z) / c) ** 2, k / s)
        return a * (1 - q) * val - q
    return rk.root(F, 1e-9, 1 - 1e-10)


def at_lambda(a, k, q):
    s, c = math.sqrt(q), math.sqrt(1 - q)

    def g(z):
        y = (k - s * z) / c
        R = mills(y)
        return R * R * (R - y) ** 2
    return a * esplit(g, k / s)


print("\n(b) RS saddle point of Gardner's volume: q(alpha) as alpha -> alpha_c (kappa = 0, 0.5)")
for k in [0.0, 0.5]:
    ac = alpha_c(k)
    for frac in [0.5, 0.9, 0.99, 0.999]:
        a = frac * ac
        q = rs_q(a, k)
        pred = (1 - frac) / (1 + a * Phi(k))
        print(f"   kappa={k}: alpha={a:.5f} ({frac} alpha_c)  q={q:.6f}  1-q={1-q:.3e}"
              f"  (1-alpha/alpha_c)/(1+alpha Phi(kappa))={pred:.3e}")

print("\n(c) replicon (dAT) quantity lambda = alpha E[R^2 (R - y)^2]; RS stable iff lambda <= 1")
for k in [0.5, 0.0, -0.5]:
    ac = alpha_c(k)
    for frac in [0.5, 0.99, 0.999]:
        a = frac * ac
        q = rs_q(a, k)
        print(f"   kappa={k:5.2f}  alpha={a:8.4f} ({frac} alpha_c)  q={q:.5f}  lambda={at_lambda(a, k, q):.5f}"
              f"   [limit at capacity: alpha_c Phi(kappa) = {ac*Phi(k):.5f}]")


def alpha_of_q(k, q):
    """The load at which q solves Gardner's RS equation: alpha = q / ((1-q) E R(y)^2)."""
    s, c = math.sqrt(q), math.sqrt(1 - q)
    return q / ((1 - q) * esplit(lambda z: mills((k - s * z) / c) ** 2, k / s))


def alpha_dat(k):
    """The load in (0, alpha_c(k)) where the replicon quantity reaches 1 (kappa < 0), found by
    parametrizing the RS branch by q: lambda(q) = alpha(q) E[R^2 (R - y)^2]."""
    f = lambda q: at_lambda(alpha_of_q(k, q), k, q) - 1.0
    hi = 1 - 1e-6
    if f(hi) <= 0:
        return float("nan")
    return alpha_of_q(k, rk.root(f, 1e-3, hi, tol=1e-12))


# ------------------------------------------------------------------------------------------
def ell(k, u):
    """-log E exp(-u (k - Z)_+^2) for k < 0, vectorized in u (closed form, MZZ eq. (75))."""
    ak = abs(k)
    u = np.asarray(u, dtype=float)
    r = np.sqrt(1 + 2 * u)
    val = Phi(ak) + np.exp(-u * ak * ak / (1 + 2 * u)) / r * (1 - Phi(ak / r))
    return -np.log(val)


TGRID = np.exp(np.linspace(math.log(1e-4), math.log(1e4), 1600))


def delta_of_s(k, s):
    """The smallest delta that the parameter s excludes. Condition (74) of MZZ reads
    L(s) < min_t [1/(4t) + (delta/s) l(st)], a minimum of functions affine in delta, so the
    threshold is sup_t s (L(s) - 1/(4t)) / l(st)."""
    L = 1.0 / (math.sqrt(s * s + 4) + s) + math.log((math.sqrt(s * s + 4) + s) / 2) / s
    vals = s * (L - 1 / (4 * TGRID)) / ell(k, s * TGRID)
    i = int(np.argmax(vals))
    lo, hi = math.log(TGRID[max(i - 1, 0)]), math.log(TGRID[min(i + 1, len(TGRID) - 1)])
    f = lambda lt: -s * (L - 1 / (4 * math.exp(lt))) / float(ell(k, s * math.exp(lt)))
    res = optimize.minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    return max(vals[i], -res.fun)


_LIFT = {}


def lifted_bound(k):
    """The lifted Gordon bound inf_s delta_of_s(k, s); s -> 0 recovers alpha_c(k)."""
    if k in _LIFT:
        return _LIFT[k]
    ss = np.exp(np.linspace(math.log(1e-3), math.log(30.0), 60))
    vals = [delta_of_s(k, s) for s in ss]
    i = int(np.argmin(vals))
    lo, hi = math.log(ss[max(i - 1, 0)]), math.log(ss[min(i + 1, len(ss) - 1)])
    res = optimize.minimize_scalar(lambda ls: delta_of_s(k, math.exp(ls)), bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-9})
    _LIFT[k] = min(vals[i], res.fun, alpha_c(k))
    return _LIFT[k]


def lifted_crit(k):
    """Var X - 2 E X for X = (Z + k)_+^2: negative iff the lifted bound beats alpha_c near alpha_c."""
    m2 = rho(k)
    m4 = rk.gauss_split(lambda z: np.maximum(z + k, 0.0) ** 4, a=-k)
    return (m4 - m2 * m2) - 2 * m2


print("\n(d) lifted Gordon bound (Stojnic 2013b Thm 4 = MZZ 2024 Def. 2), against alpha_c(kappa)")
stojnic = {-0.5: 4.7700, -0.6: 5.7787, -0.63: 6.12834, -0.7: 7.0313, -0.8: 8.5631, -0.9: 10.4484,
           -1.0: 12.784, -1.1: 15.6977}
for k in [-0.5, -0.6, -0.63, -0.7, -0.8, -0.9, -1.0, -1.1]:
    lb = lifted_bound(k)
    print(f"   kappa={k:5.2f}  alpha_c={alpha_c(k):9.5f}  lifted={lb:9.5f}  ratio={lb/alpha_c(k):.6f}"
          f"   Stojnic's table {stojnic[k]}   VarX-2EX={lifted_crit(k):+.5f}")
kcross = rk.root(lifted_crit, -0.7, -0.5)
print(f"   crossover kappa (Var X = 2 E X): {kcross:.5f}, where alpha_c = {alpha_c(kcross):.5f}")
for a in [6.5, 8.0, 10.0]:
    kl = rk.root(lambda k: lifted_bound(round(k, 12)) - a, -1.6, kcross - 1e-6, tol=1e-7)
    print(f"   margin form at alpha={a}: Gardner kappa*={kappa_star(a):.4f}, lifted bound kappa={kl:.4f},"
          f" LP limit Phi^-1(1/alpha)={stats.norm.ppf(1/a):.4f}")

# ------------------------------------------------------------------------------------------
print("\n(e) Ising perceptron")
for k in [0.0, 0.5, 1.0]:
    print(f"   RS q->1 'capacity' 2/(pi rho(kappa)) at kappa={k}: {2/(math.pi*rho(k)):.6f}")
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(200)
GH_W = GH_W / GH_W.sum()


def EZ(f):
    return float(np.dot(GH_W, f(GH_X)))


def km_fixed_point(a, q0=0.56, qh0=2.57):
    def eqs(v):
        q, qh = v
        q = min(max(q, 1e-9), 1 - 1e-9)
        e1 = EZ(lambda z: np.tanh(math.sqrt(max(qh, 0.0)) * z) ** 2) - q
        e2 = a * EZ(lambda z: mills(-math.sqrt(q) * z / math.sqrt(1 - q)) ** 2) / (1 - q) - qh
        return [e1, e2]
    return optimize.fsolve(eqs, [q0, qh0], xtol=1e-13)


def km_entropy(a, q, qh):
    return (-qh * (1 - q) / 2 + EZ(lambda z: np.log(2 * np.cosh(math.sqrt(qh) * z)))
            + a * EZ(lambda z: special.log_ndtr(math.sqrt(q) * z / math.sqrt(1 - q))))


a_km = optimize.brentq(lambda a: km_entropy(a, *km_fixed_point(a)), 0.80, 0.86, xtol=1e-13)
q_km, qh_km = km_fixed_point(a_km)
print(f"   Krauth-Mezard: RS entropy = 0 at alpha = {a_km:.10f} (q = {q_km:.8f}, qhat = {qh_km:.8f})")
for a in [0.5, 0.8, 0.85]:
    q, qh = km_fixed_point(a)
    print(f"      RS entropy at alpha={a}: {km_entropy(a, q, qh):+.5f}")
print(f"   E log H(U) by quadrature: {rk.gauss_split(lambda u: special.log_ndtr(-u), a=0.0):.12f}  (exactly -1)")


def gd_q(a, q):
    return (a * EZ(lambda u: special.log_ndtr(-math.sqrt(q) * u / math.sqrt(1 - q)))
            + q / (2 * (1 - q)) + 0.5 * math.log(1 - q))


def gd(a):
    res = optimize.minimize_scalar(lambda q: gd_q(a, q), bounds=(1e-6, 1 - 1e-6), method="bounded",
                                   options={"xatol": 1e-12})
    return res.fun, res.x


a_ub = optimize.brentq(lambda a: gd(a)[0] + math.log(2), 0.80, 0.90, xtol=1e-12)
print(f"   conditional first moment: GD(alpha) = -log 2 at alpha = {a_ub:.6f} (minimizing q = {gd(a_ub)[1]:.4f})")
print(f"   closed form at q = 1/2: (1 + log 2)/2 = {(1+math.log(2))/2:.6f}")
print(f"   GD(0.847) + log 2 = {gd(0.847)[0]+math.log(2):+.6f};  GD(0.847, 1/2) + log 2 = "
      f"{gd_q(0.847, 0.5)+math.log(2):+.6f}  (formula: {-0.847+0.5*(1-math.log(2))+math.log(2):+.6f})")

print("\n(f) symmetric binary perceptron, first-moment capacity -log 2/log P(|Z| <= kappa)")
for k in [0.5, 1.0, 1.5]:
    print(f"   kappa={k}: {-math.log(2)/math.log(2*Phi(k)-1):.5f}")

# ------------------------------------------------------------------------------------------
print("\nwriting curves ...")
rows = []
for k in np.round(np.arange(-1.20, 1.5001, 0.05), 4):
    ac = alpha_c(k)
    on_coarse = abs(k * 10 - round(k * 10)) < 1e-9 or abs(k + 0.05) < 1e-9
    adat = (alpha_dat(k) if (k < -1e-9 and on_coarse) else (2.0 if abs(k) < 1e-9 else float("nan")))
    alin = 1 / Phi(k) if k < 0 else float("nan")
    alift = lifted_bound(k) if k <= -0.6 else float("nan")
    rows.append([float(k), ac, adat, alin, alift])
rk.write_dat(os.path.join(DATA, "08-gordon-phase.dat"), ["kappa", "alphac", "alphadat", "alphalin", "alphalift"], rows)
for r in rows:
    if abs(r[0] + 1.0) < 1e-9 or abs(r[0] + 0.5) < 1e-9 or abs(r[0] + 0.2) < 1e-9:
        print(f"   kappa={r[0]:5.2f}: alpha_c={r[1]:.4f} alpha_dAT={r[2]:.4f} 1/Phi={r[3]:.4f} lifted={r[4]:.4f}")

crow = []
for a in np.concatenate([np.linspace(0.1, 2.0, 39), np.linspace(2.05, 14.0, 60)]):
    krs = kappa_star(a)
    klin = stats.norm.ppf(1 / a) if a > 2 else float("nan")
    crow.append([float(a), krs, klin, float("nan")])
# lifted bound as a margin curve: invert alpha_lifted(kappa) on the kappa grid where it is below alpha_c
ph = np.array([[r[0], r[4]] for r in rows if not math.isnan(r[4]) and r[4] < r[1] * (1 - 1e-5)])
ph = ph[np.argsort(ph[:, 1])]
for row in crow:
    a = row[0]
    if ph.size and ph[0, 1] <= a <= ph[-1, 1]:
        row[3] = float(np.interp(a, ph[:, 1], ph[:, 0]))
rk.write_dat(os.path.join(DATA, "08-gordon-curves.dat"), ["alpha", "kappars", "kappalin", "kappalift"], crow)
print(f"runtime {time.time()-T0:.1f} s")
