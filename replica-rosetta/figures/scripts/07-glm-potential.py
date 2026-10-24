"""Chapter 7 (glm): the replica-symmetric potential of the teacher--student perceptron.

Noiseless perceptron y = sgn(<x, w*>/sqrt(d)), x ~ N(0, I_d), n = alpha d (book notation, OUTLINE.md).
Replica-symmetric potential (Barbier, Krzakala, Macris, Miolane, Zdeborova 2019, eq. (41)):
    f_RS(q, r) = psi(r) + alpha Psi(q) - r q / 2,
    Psi(q)     = 2 E_V[ Phi(aV) log Phi(aV) ],  a = sqrt(q/(1-q))           (sign channel)
    2 Psi'(q)  = E_V[ exp(-a^2 V^2) / Phi(aV) ] / (pi (1-q))                 (two Gaussian IBPs)
    Ising (Rademacher) teacher:  psi(r) = E log cosh(r + sqrt(r) Z) - r/2,  q(r) = E tanh(r + sqrt(r) Z)
    spherical (Gaussian) teacher: psi(r) = r/2 - log(1+r)/2,                 q(r) = r/(1+r)
Entropy of the version space of the Ising teacher (Barbier et al., Corollary 3):
    s(alpha) = lim (1/d) E log |V| = log 2 + sup_{q<1} inf_r f_RS(q, r) = max(0, s_low(alpha)).
State evolution (GAMP): q_{t+1} = q(r_t), r_t = 2 alpha Psi'(q_t), from q_0 = 0.
First-moment (annealed) bound of the version space (Theorem in Sec. "counting"):
    g_alpha(m) = h((1-m)/2) + alpha log(1 - arccos(m)/pi),  h = binary entropy in nats;
    alpha_1 = smallest alpha with g_alpha < 0 on (-1, 1).

Outputs:  figures/data/07-glm-potential.dat   (s(q; alpha) for several alpha, for the physics figure)
          figures/data/07-glm-curves.dat      (learning curves and entropies against alpha)
Prints every number quoted in the chapter.  Runtime: about 30-60 s (most of it in mpmath).
"""
import math
import os
import sys
import time

import numpy as np
from scipy import optimize, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rosettakit as rk  # noqa: E402

T0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "figures", "data")
LOG2 = math.log(2.0)

# ----------------------------------------------------------------------------------------------
# Double-precision building blocks
# ----------------------------------------------------------------------------------------------
_GL_X, _GL_W = np.polynomial.legendre.leggauss(1600)


def EV_of(g, a):
    """E_V g(aV) for V ~ N(0,1): Gauss--Hermite in V for small a, and the substitution x = aV with
    Gauss--Legendre on [-40, 40] (density phi(x/a)/a) for a >= 1, where g varies on the scale 1."""
    if a < 1.0:
        return rk.gauss(lambda v: g(a * v), 201)
    L = 40.0
    x = L * _GL_X
    w = L * _GL_W
    return float(np.sum(w * g(x) * np.exp(-0.5 * (x / a) ** 2) / (a * math.sqrt(2 * math.pi))))


def Psi(q):
    """Psi(q) = 2 E[Phi(aV) log Phi(aV)] (sign channel); Psi(0) = -log 2, Psi(1) = 0."""
    if q <= 0:
        return -LOG2
    if q >= 1:
        return 0.0
    a = math.sqrt(q / (1 - q))
    return 2 * EV_of(lambda x: np.exp(special.log_ndtr(x)) * special.log_ndtr(x), a)


def dPsi2(q):
    """2 Psi'(q) = E[exp(-a^2 V^2)/Phi(aV)] / (pi (1-q))."""
    a = math.sqrt(q / (1 - q))
    return EV_of(lambda x: np.exp(-x * x - special.log_ndtr(x)), a) / (math.pi * (1 - q))


def logcosh(x):
    return np.logaddexp(x, -x) - LOG2


def psi_ising(r):
    return rk.gauss(lambda z: logcosh(r + math.sqrt(r) * z), 201) - r / 2


def q_ising(r):
    if r <= 0:
        return 0.0
    return rk.gauss(lambda z: np.tanh(r + math.sqrt(r) * z), 201)


def r_of_q_ising(q):
    """Inverse of the increasing map r -> E tanh(r + sqrt(r) Z)."""
    return rk.root(lambda r: q_ising(r) - q, 1e-14, 400.0)


def s_ising(q, alpha):
    """Version-space entropy potential log 2 + inf_r f_RS(q, r) (Ising teacher); s(1) = 0."""
    r = r_of_q_ising(q)
    return LOG2 + psi_ising(r) - r * q / 2 + alpha * Psi(q)


def se_map_ising(q, alpha):
    return q_ising(alpha * dPsi2(q))


def se_map_sph(q, alpha):
    r = alpha * dPsi2(q)
    return r / (1 + r)


_QSCAN = np.concatenate([np.linspace(0.0, 0.98, 197), np.linspace(0.981, 0.9995, 38)])


def low_branch(alpha):
    """The fixed point of the Ising state evolution reached from q_0 = 0.  The map q -> SE(q) is
    increasing and SE(0) > 0, so the iteration from 0 increases to the smallest fixed point in [0, 1);
    if there is none below 1, it escapes to perfect recovery.  Returns (q, True) or (1.0, False)."""
    prev_q, prev_v = 0.0, se_map_ising(0.0, alpha)
    for q in _QSCAN[1:]:
        v = se_map_ising(q, alpha) - q
        if v <= 0:
            return rk.root(lambda x: se_map_ising(x, alpha) - x, prev_q, q, tol=1e-14), True
        prev_q, prev_v = q, v
    return 1.0, False


def interior_max(alpha, qguess):
    """Local maximum of s(.; alpha) near qguess on (0,1): the low branch, as a stationary point."""
    res = optimize.minimize_scalar(lambda q: -s_ising(q, alpha), bounds=(max(qguess - 0.08, 1e-6), min(qguess + 0.08, 0.995)),
                                   method="bounded", options={"xatol": 1e-12})
    return res.x, -res.fun


# ----------------------------------------------------------------------------------------------
# 1. Checks of the closed forms
# ----------------------------------------------------------------------------------------------
print("== checks")
for q in (0.2, 0.4, 0.8, 0.95):
    h = 1e-6
    fd = (Psi(q + h) - Psi(q - h)) / (2 * h)
    print(f"  2Psi'(q) closed form vs 2*finite difference, q={q}: {dPsi2(q):.10f}  {2 * fd:.10f}")
# psi'(r) = q(r)/2 (Nishimori / I-MMSE)
for r in (0.3, 1.0, 3.0):
    h = 1e-6
    print(f"  psi'(r) vs q(r)/2, r={r}: {(psi_ising(r + h) - psi_ising(r - h)) / (2 * h):.10f}  {q_ising(r) / 2:.10f}")
print(f"  2Psi'(0) = {dPsi2(1e-12):.10f}  (2/pi = {2 / math.pi:.10f}): q = 0 is not a fixed point")

# ----------------------------------------------------------------------------------------------
# 2. Ising teacher in double precision: branches, alpha_IT, alpha_AMP
# ----------------------------------------------------------------------------------------------
print("== Ising teacher (double precision)")
print(f"   [t = {time.time() - T0:.1f} s]")
alphas_tab = [0.5, 1.0, 1.2, 1.3, 1.4, 1.49]
for a in alphas_tab:
    q, _ = low_branch(a)
    print(f"  alpha={a:5.2f}: low-branch q={q:.6f}  s={s_ising(q, a):+.6f}  eps_Bayes(branch)={math.acos(math.sqrt(q)) / math.pi:.6f}"
          f"  eps_Gibbs(branch)={math.acos(q) / math.pi:.6f}")


def s_low(alpha):
    q, ok = low_branch(alpha)
    return s_ising(q, alpha)


a_it = rk.root(s_low, 1.2, 1.3, tol=1e-12)
q_it, _ = low_branch(a_it)
print(f"  alpha_IT (double) = {a_it:.8f}, q = {q_it:.8f}, eps_Bayes jumps from {math.acos(math.sqrt(q_it)) / math.pi:.6f} to 0,"
      f" eps_Gibbs from {math.acos(q_it) / math.pi:.6f} to 0")
for a in (1.2446, 1.2447, 1.249):
    print(f"  s_low({a}) = {s_low(a):+.4e}")


def min_gap(alpha):
    """min over q in (0.3, 0.995) of SE(q) - q: negative iff a fixed point below 1 exists."""
    grid = np.linspace(0.3, 0.995, 120)
    vals = np.array([se_map_ising(q, alpha) - q for q in grid])
    j = int(np.argmin(vals))
    res = optimize.minimize_scalar(lambda q: se_map_ising(q, alpha) - q,
                                   bounds=(grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]),
                                   method="bounded", options={"xatol": 1e-13})
    return res.fun, res.x


a_amp = rk.root(lambda a: min_gap(a)[0], 1.45, 1.55, tol=1e-11)
print(f"  alpha_AMP (double) = {a_amp:.8f}, q at the spinodal = {min_gap(a_amp)[1]:.6f}")
q_sp = min_gap(a_amp)[1]
print(f"  GAMP error at the spinodal (branch) = {math.acos(math.sqrt(q_sp)) / math.pi:.6f}")

# ----------------------------------------------------------------------------------------------
# 3. First-moment (annealed) threshold alpha_1
# ----------------------------------------------------------------------------------------------
print("== first moment")
print(f"   [t = {time.time() - T0:.1f} s]")


def g_fm(m, alpha):
    p = (1 - m) / 2
    hh = -special.xlogy(p, p) - special.xlogy(1 - p, 1 - p)
    return hh + alpha * np.log1p(-np.arccos(m) / math.pi)


mgrid = np.linspace(-1 + 1e-9, 1 - 1e-7, 400001)


def sup_g(alpha):
    v = g_fm(mgrid, alpha)
    return float(v.max()), float(mgrid[v.argmax()])


def sup_g_interior(alpha):
    v = g_fm(mgrid[mgrid < 0.95], alpha)
    return float(v.max())


a1 = rk.root(sup_g_interior, 1.3, 1.6, tol=1e-12)
print(f"  alpha_1 (double) = {a1:.8f}; the maximizer of g at alpha_1: m = {mgrid[np.argmax(g_fm(mgrid[mgrid < 0.95], a1))]:.6f}")
for a in (1.0, 1.2447, 1.4):
    print(f"  annealed entropy sup_m g_alpha(m) at alpha={a}: {sup_g(a)[0]:.6f}")

# ----------------------------------------------------------------------------------------------
# 4. High precision (mpmath) for alpha_IT, alpha_AMP, alpha_1, and the potential at alpha = 1.249
# ----------------------------------------------------------------------------------------------
print("== high precision (mpmath, 20 digits)")
print(f"   [t = {time.time() - T0:.1f} s]")
import mpmath as mp  # noqa: E402

mp.mp.dps = 20


def mE(f):
    """E f(Z), Z ~ N(0,1)."""
    return mp.quad(lambda z: f(z) * mp.npdf(z), [-mp.inf, -8, -3, 0, 3, 8, mp.inf])


def mPsi(q):
    a = mp.sqrt(q / (1 - q))
    return 2 * mE(lambda v: mp.ncdf(a * v) * mp.log(mp.ncdf(a * v)))


def mdPsi2(q):
    a = mp.sqrt(q / (1 - q))
    return mE(lambda v: mp.exp(-a * a * v * v) / mp.ncdf(a * v)) / (mp.pi * (1 - q))


def mq(r):
    return mE(lambda z: mp.tanh(r + mp.sqrt(r) * z))


def mlogcosh(x):
    x = abs(x)
    return x + mp.log1p(mp.exp(-2 * x)) - mp.log(2)


def mpsi(r):
    return mE(lambda z: mlogcosh(r + mp.sqrt(r) * z)) - r / 2


r_it = r_of_q_ising(q_it)


def it_system(al, q, r):
    return [mq(r) - q, r - al * mdPsi2(q), mp.log(2) + mpsi(r) - r * q / 2 + al * mPsi(q)]


sol = mp.findroot(it_system, (mp.mpf(a_it), mp.mpf(q_it), mp.mpf(r_it)), tol=mp.mpf(10) ** -17)
A_IT, Q_IT, R_IT = sol[0], sol[1], sol[2]
print(f"  alpha_IT = {mp.nstr(A_IT, 15)}   q = {mp.nstr(Q_IT, 12)}   r = {mp.nstr(R_IT, 12)}")
print(f"  residuals: {[mp.nstr(v, 3) for v in it_system(A_IT, Q_IT, R_IT)]}")


def s_low_mp(al, q0):
    """Interior stationary point of the potential near q0 (solve q = q(r), r = al 2Psi'(q)), and its
    value; returns (q, s)."""
    f = lambda q, r: [mq(r) - q, r - al * mdPsi2(q)]
    s2 = mp.findroot(f, (mp.mpf(q0), mp.mpf(r_of_q_ising(q0))), tol=mp.mpf(10) ** -16)
    q, r = s2[0], s2[1]
    return q, mp.log(2) + mpsi(r) - r * q / 2 + al * mPsi(q)


for al_str in ("1.249",):
    al = mp.mpf(al_str)
    qq, ss = s_low_mp(al, interior_max(float(al), q_it)[0])
    print(f"  alpha={al_str}: low-branch q = {mp.nstr(qq, 10)}, entropy of the branch s = {mp.nstr(ss, 6)}")


def amp_system(al, q):
    G = lambda qq: mq(al * mdPsi2(qq)) - qq
    return [G(q), mp.diff(G, q)]


sol2 = mp.findroot(amp_system, (mp.mpf(a_amp), mp.mpf(q_sp)), tol=mp.mpf(10) ** -14)
A_AMP, Q_AMP = sol2[0], sol2[1]
print(f"  alpha_AMP = {mp.nstr(A_AMP, 12)}   q at the spinodal = {mp.nstr(Q_AMP, 10)}")


def mg(m, al):
    p = (1 - m) / 2
    return -p * mp.log(p) - (1 - p) * mp.log(1 - p) + al * mp.log(1 - mp.acos(m) / mp.pi)


m_star = float(mgrid[np.argmax(g_fm(mgrid[mgrid < 0.95], a1))])
sol3 = mp.findroot(lambda al, m: [mg(m, al), mp.diff(lambda mm: mg(mm, al), m)], (mp.mpf(a1), mp.mpf(m_star)), tol=mp.mpf(10) ** -17)
print(f"  alpha_1 = {mp.nstr(sol3[0], 15)}   maximizer m = {mp.nstr(sol3[1], 12)}")
A1 = float(sol3[0])

# ----------------------------------------------------------------------------------------------
# 5. Spherical (Gaussian-prior) teacher
# ----------------------------------------------------------------------------------------------
print("== spherical teacher")
print(f"   [t = {time.time() - T0:.1f} s]")


def q_sph(alpha):
    """The fixed point of q = r/(1+r), r = alpha 2Psi'(q) (checked unique on a grid below)."""
    return rk.root(lambda q: se_map_sph(q, alpha) - q, 1e-10, 1 - 1e-12, tol=1e-15)


uniq = []
for a in (0.25, 0.5, 1, 2, 5, 10, 20):
    grid = np.linspace(1e-4, 1 - 1e-6, 500)
    vals = np.array([se_map_sph(q, a) - q for q in grid])
    uniq.append(int(np.sum(np.sign(vals[1:]) != np.sign(vals[:-1]))))
print(f"  number of sign changes of SE(q) - q on a 500-point grid, alpha = 0.25..20: {uniq}")
for a in (0.5, 1, 2, 5, 10, 100):
    q = q_sph(a)
    eb, eg = math.acos(math.sqrt(q)) / math.pi, math.acos(q) / math.pi
    print(f"  alpha={a:6.1f}: q={q:.8f} eps_Bayes={eb:.6f} (alpha*eps={a * eb:.5f}) eps_Gibbs={eg:.6f} (alpha*eps={a * eg:.5f})")
from scipy import integrate  # noqa: E402

I_val, _ = integrate.quad(lambda x: math.exp(-x * x - special.log_ndtr(x)) / math.sqrt(2 * math.pi), -np.inf, np.inf,
                          epsabs=1e-14, epsrel=1e-13)
print(f"  I = int exp(-x^2)/Phi(x) dx/sqrt(2pi) = {I_val:.8f};  1/I = {1 / I_val:.6f},  sqrt(2)/I = {math.sqrt(2) / I_val:.6f}")
I_mp = mp.quad(lambda x: mp.exp(-x * x) / mp.ncdf(x), [-mp.inf, -5, 0, 5, mp.inf]) / mp.sqrt(2 * mp.pi)
print(f"  I (mpmath) = {mp.nstr(I_mp, 12)}")

# ----------------------------------------------------------------------------------------------
# 6. Data files
# ----------------------------------------------------------------------------------------------
# (a) the potential s(q; alpha) for the physics figure
print(f"   [t = {time.time() - T0:.1f} s]")
alist = [1.0, float(A_IT), 1.40, float(A_AMP), 1.60]
qs = np.concatenate([np.linspace(0.02, 0.9, 45), np.linspace(0.905, 0.999, 30)])
rows = []
for q in qs:
    rows.append([q] + [s_ising(q, a) for a in alist])
rows.append([1.0] + [0.0 for _ in alist])
rk.write_dat(os.path.join(DATA, "07-glm-potential.dat"), ["q", "s100", "sIT", "s140", "sAMP", "s160"], rows)
# local maxima for annotation
for a in alist:
    if a < float(A_AMP) - 1e-9:
        qm, _ = low_branch(a)
        print(f"  potential at alpha={a:.5f}: local maximum below 1 at q={qm:.5f}, s={s_ising(qm, a):+.5f}")
    else:
        print(f"  potential at alpha={a:.5f}: no local maximum below 1 (inflection at q={float(Q_AMP):.5f} when alpha = alpha_AMP)")

# (b) learning curves and entropies against alpha
print(f"   [t = {time.time() - T0:.1f} s]")
arows = []
NAN = float("nan")
aIT, aAMP = float(A_IT), float(A_AMP)
agrid = np.concatenate([np.linspace(0.05, 2.5, 99), [aIT - 1e-7, aIT + 1e-7, aAMP - 1e-7, aAMP + 1e-7, 4, 6, 8, 12, 16, 20]])
agrid = np.sort(agrid)
for a in agrid:
    ql, below = low_branch(a) if a < 2.6 else (1.0, False)
    s_l = s_ising(ql, a) if ql < 1 else NAN
    # Bayes-optimal overlap q* (the global maximizer): the low branch below alpha_IT, 1 above
    eB_low = math.acos(math.sqrt(ql)) / math.pi if a < aIT else NAN
    eB_high = 0.0 if a > aIT else NAN
    eG_low = math.acos(ql) / math.pi if a < aIT else NAN
    # GAMP from q0 = 0 on the metastable part of the branch (alpha_IT < alpha < alpha_AMP)
    e_meta = math.acos(math.sqrt(ql)) / math.pi if aIT < a < aAMP else NAN
    s_rep = (max(0.0, s_l) if ql < 1 else 0.0) if a < 2.6 else 0.0
    s_ann = max(0.0, sup_g(a)[0]) if a < 2.6 else 0.0
    s_cnt = max(0.0, (1 - a) * LOG2)
    qs_ = q_sph(a)
    eBs, eGs = math.acos(math.sqrt(qs_)) / math.pi, math.acos(qs_) / math.pi
    arows.append([a, eB_low, eB_high, eG_low, e_meta, s_l, s_rep, s_ann, s_cnt, qs_, eBs, eGs, a * eBs, a * eGs])
rk.write_dat(os.path.join(DATA, "07-glm-curves.dat"),
             ["alpha", "epsBlow", "epsBhigh", "epsGlow", "epsmeta", "sbranch", "srep", "sann", "scount",
              "qsph", "epsBsph", "epsGsph", "aepsBsph", "aepsGsph"], arows)
print(f"  asymptotic constants for the plot: 1/I = {1 / I_val:.6f}, sqrt(2)/I = {math.sqrt(2) / I_val:.6f}")
print(f"runtime {time.time() - T0:.1f} s")
