"""Chapter 3 (SK): the replica predictions, computed.

Normalization (OUTLINE.md): H_N(s) = (beta/sqrt N) sum_{i<j} g_ij s_i s_j + h sum_i s_i, xi(q) = beta^2 q^2/2,
beta_c = 1. Free entropy phi = lim N^{-1} E log Z_N; physicist's free energy f = -phi/beta, temperature
T = 1/beta, physical field H = h/beta.

Computes and writes
  figures/data/03-sk-atline.dat   the de Almeida--Thouless line H_AT(T) and its small-field law
  figures/data/03-sk-rs.dat       the RS and 1RSB solutions at h = 0 against T: q, phi, energy, entropy
  figures/data/03-sk-phi.dat      phi_RS, phi_1RSB and the annealed value against beta at h = 0
and prints: the RS fixed point and AT parameter at sample points; the zero-temperature limits of RS
(beta(1-q) -> sqrt(2/pi), energy -> -sqrt(2/pi), entropy -> -1/(2 pi)); the temperature at which the RS
entropy changes sign; the small-field AT law (1-T)^3/H^2 -> 3/4; a check that phi_RS(q) is minimized at
the fixed point; Toninelli's expansion d/dm P_1RSB(m=1) = (beta^2 delta^2/4)(AT - 1) + O(delta^3); and
the zero-temperature Parisi functional (Auffinger--Chen) with k = 0, 1, 2 steps.

Run from the volume root: python figures/scripts/03-sk-replica.py   (under a minute)
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
OUT = os.path.join("figures", "data")
LOG2 = math.log(2.0)

# ---------------------------------------------------------------------------------------------
# E f(s Z + h), robust for every scale s: composite Gauss--Legendre in u = s Z + h on panels that
# resolve both the Gaussian (scale s) and the features of f near 0 (scale 1).
# ---------------------------------------------------------------------------------------------
_GLX, _GLW = np.polynomial.legendre.leggauss(96)


def _panels(s, h):
    lo, hi = h - 12.0 * s, h + 12.0 * s
    cuts = [c for c in (-40.0, -12.0, -4.0, -1.0, 0.0, 1.0, 4.0, 12.0, 40.0) if lo < c < hi]
    return [lo] + cuts + [hi]


def nodes(s, h=0.0):
    """Nodes u and weights w with sum w f(u) = E f(sZ + h)."""
    if s < 1e-12:
        return np.array([h]), np.array([1.0])
    us, ws = [], []
    edges = _panels(s, h)
    for a, b in zip(edges[:-1], edges[1:]):
        u = 0.5 * (b - a) * _GLX + 0.5 * (b + a)
        w = 0.5 * (b - a) * _GLW * np.exp(-0.5 * ((u - h) / s) ** 2) / (s * math.sqrt(2 * math.pi))
        us.append(u)
        ws.append(w)
    return np.concatenate(us), np.concatenate(ws)


def E(f, s, h=0.0):
    u, w = nodes(s, h)
    return float(np.sum(w * f(u)))


def lncosh(u):
    a = np.abs(u)
    return a + np.log1p(np.exp(-2.0 * a)) - LOG2


def sech4(u):
    """sech^4 u = 16 e^{-4|u|} / (1 + e^{-2|u|})^4, without overflow."""
    e = np.exp(-2.0 * np.abs(u))
    return 16.0 * e * e / (1.0 + e) ** 4


# ---------------------------------------------------------------------------------------------
# Replica symmetry
# ---------------------------------------------------------------------------------------------
def q_rs(beta, h):
    """The RS fixed point q = E tanh^2(beta sqrt(q) Z + h); for h = 0 the positive root when beta > 1."""
    if h == 0.0 and beta <= 1.0:
        return 0.0
    g = lambda q: E(lambda u: np.tanh(u) ** 2, beta * math.sqrt(q), h) - q
    lo = 1e-14 if h != 0.0 else min(1e-3, 0.5 * (beta - 1.0) ** 2)
    return optimize.brentq(g, lo, 1.0 - 1e-15, xtol=1e-16, rtol=1e-15, maxiter=500)


def phi_rs_at(beta, h, q):
    return LOG2 + E(lncosh, beta * math.sqrt(q), h) + beta ** 2 * (1.0 - q) ** 2 / 4.0


def phi_rs(beta, h):
    q = q_rs(beta, h)
    return phi_rs_at(beta, h, q), q


def at_param(beta, h, q=None):
    if q is None:
        q = q_rs(beta, h)
    return beta ** 2 * E(sech4, beta * math.sqrt(q), h)


def rs_thermo(beta):
    """h = 0: phi, q, energy -dphi/dbeta, entropy phi - beta dphi/dbeta. At the fixed point
    dphi/dbeta = (beta/2)(1 - q^2) (envelope theorem and E sech^2 = 1 - q)."""
    F, q = phi_rs(beta, 0.0)
    d = 0.5 * beta * (1.0 - q * q)
    return F, q, -d, F - beta * d


# ---------------------------------------------------------------------------------------------
# One-step RSB at h = 0 with q_0 = 0:
#   P(q, m) = log 2 + (1/m) log E cosh^m(beta sqrt(q) Z) + (beta^2/4) [(1-q)^2 - m q^2]
# (Parisi functional of mu = m delta_0 + (1-m) delta_q.)
# ---------------------------------------------------------------------------------------------
def p1rsb(q, m, beta):
    u, w = nodes(beta * math.sqrt(q))
    lval = special.logsumexp(m * lncosh(u), b=w) / m
    return LOG2 + lval + beta ** 2 / 4.0 * ((1.0 - q) ** 2 - m * q * q)


def _logit(p):
    return math.log(p / (1.0 - p))


def _expit(x):
    return 1.0 / (1.0 + math.exp(-x))


def phi_1rsb(beta, start=None):
    """min over (q, m) in (0,1)^2 of P(q, m), by Nelder--Mead in logit coordinates."""
    if beta <= 1.0:
        return phi_rs(beta, 0.0)[0], 0.0, 1.0
    q0, m0 = start if start is not None else (0.5, 0.3)
    f = lambda x: p1rsb(_expit(x[0]), _expit(x[1]), beta)
    r = optimize.minimize(f, [_logit(q0), _logit(m0)], method="Nelder-Mead",
                          options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 3000})
    return float(r.fun), _expit(r.x[0]), _expit(r.x[1])


def thermo_1rsb(beta, start=None):
    """phi, q, m, energy and entropy of the 1RSB solution; dphi/dbeta by the envelope theorem:
    dP/dbeta = (1/beta) E_m[u tanh u] + (beta/2)[(1-q)^2 - m q^2], with E_m the cosh^m-tilted law of
    u = beta sqrt(q) Z."""
    F, q, m = phi_1rsb(beta, start)
    u, w = nodes(beta * math.sqrt(q))
    lw = np.log(np.maximum(w, 1e-300)) + m * lncosh(u)
    p = np.exp(lw - special.logsumexp(lw))
    dF = float(np.sum(p * u * np.tanh(u))) / beta + beta / 2.0 * ((1.0 - q) ** 2 - m * q * q)
    return F, q, m, -dF, F - beta * dF


# ---------------------------------------------------------------------------------------------
# 1. RS and AT at sample points
# ---------------------------------------------------------------------------------------------
print("1. RS fixed point, free entropy and AT parameter (h = beta * H)")
for T, H in [(1.0, 0.0), (0.8, 0.0), (0.5, 0.0), (2 / 3, 0.0), (1.0, 0.3), (0.5, 0.5), (0.5, 1.0)]:
    b = 1.0 / T
    F, q = phi_rs(b, b * H)
    print(f"   T={T:.4f} H={H}: q={q:.6f} phi_RS={F:.6f} AT={at_param(b, b * H, q):.4f}")

# minimum of phi_RS(q) at the fixed point (h = 0, beta = 1.5 and beta = 2; h = 1, beta = 2)
for b, h in [(1.5, 0.0), (2.0, 0.0), (2.0, 1.0)]:
    qs = np.linspace(0.0, 0.999, 1000)
    vals = np.array([phi_rs_at(b, h, qq) for qq in qs])
    qstar = q_rs(b, h)
    print(f"   beta={b}, h={h}: argmin over a grid of phi_RS(q) = {qs[np.argmin(vals)]:.4f}; fixed point q* = {qstar:.6f};"
          f" min grid value {vals.min():.6f} vs phi_RS(q*) = {phi_rs_at(b, h, qstar):.6f}")

# ---------------------------------------------------------------------------------------------
# 2. RS at zero temperature
# ---------------------------------------------------------------------------------------------
print(f"   ({time.time() - T0:.1f} s)")
print("2. RS at low temperature, h = 0")
for T in [0.1, 0.03, 0.01, 0.003, 0.001]:
    F, q, e, s = rs_thermo(1.0 / T)
    print(f"   T={T}: beta(1-q)={(1 - q) / T:.6f} energy={e:.6f} entropy={s:+.6f}")
print(f"   limits: sqrt(2/pi)={math.sqrt(2 / math.pi):.6f}  -1/(2 pi)={-1 / (2 * math.pi):.6f}")
Tzero = optimize.brentq(lambda T: rs_thermo(1.0 / T)[3], 0.05, 0.95, xtol=1e-10)
print(f"   the RS entropy vanishes at T = {Tzero:.4f} and is negative below")

# ---------------------------------------------------------------------------------------------
# 3. The AT line
# ---------------------------------------------------------------------------------------------
print(f"   ({time.time() - T0:.1f} s)")
print("3. The AT line in the (T, H) plane")
rows = []
for T in np.concatenate([np.linspace(0.02, 0.9, 45), np.linspace(0.905, 0.999, 20)]):
    b = 1.0 / T
    Hat = optimize.brentq(lambda H: at_param(b, b * H) - 1.0, 1e-7, 6.0, xtol=1e-12)
    rows.append((T, Hat, math.sqrt(4.0 / 3.0) * (1 - T) ** 1.5))
rows.append((1.0, 0.0, 0.0))
rk.write_dat(os.path.join(OUT, "03-sk-atline.dat"), ["T", "H", "Hsmall"], rows)
for T in [0.9, 0.7, 0.5, 0.3, 0.1]:
    b = 1.0 / T
    Hat = optimize.brentq(lambda H: at_param(b, b * H) - 1.0, 1e-7, 6.0, xtol=1e-12)
    print(f"   T={T}: H_AT={Hat:.6f}")
for H in [1e-4, 1e-3, 1e-2]:
    Tat = optimize.brentq(lambda T: at_param(1 / T, H / T) - 1.0, 0.5, 1 - 1e-9, xtol=1e-14)
    print(f"   H={H:g}: T_AT={Tat:.6f}, (1-T)^3/H^2={(1 - Tat) ** 3 / H ** 2:.4f}  (small-field law: 3/4)")

# ---------------------------------------------------------------------------------------------
# 4. RS and 1RSB against T at h = 0, and against beta
# ---------------------------------------------------------------------------------------------
print(f"   ({time.time() - T0:.1f} s)")
print("4. RS and 1RSB at h = 0")
rows = []
start = (0.3, 0.2)
for T in list(np.linspace(0.05, 0.95, 37)) + [0.975, 1.0, 1.1, 1.25, 1.5]:
    b = 1.0 / T
    F, q, e, s = rs_thermo(b)
    if b > 1.0:
        F1, q1, m1, e1, s1 = thermo_1rsb(b, start)
        start = (q1, m1)
    else:
        F1, q1, m1, e1, s1 = F, q, 1.0, e, s
    rows.append((T, q, F, e, s, F1, q1, m1, e1, s1))
rows.sort()
rk.write_dat(os.path.join(OUT, "03-sk-rs.dat"),
             ["T", "q", "phiRS", "eRS", "sRS", "phi1", "q1", "m1", "e1", "s1"], rows)
for r in rows:
    if any(abs(r[0] - t) < 1e-9 for t in (0.05, 0.3, 0.5, 0.8)):
        print(f"   T={r[0]:.2f}: RS q={r[1]:.5f} phi={r[2]:.6f} s={r[4]:+.5f} | 1RSB phi={r[5]:.6f} q={r[6]:.5f} m={r[7]:.5f} s={r[9]:+.5f}")

rows = []
start = (0.3, 0.2)
for b in np.round(np.arange(0.05, 2.0001, 0.05), 4):
    F, q = phi_rs(b, 0.0)
    if b > 1.0:
        F1, q1, m1 = phi_1rsb(b, start)
        start = (q1, m1)
    else:
        F1 = F
    rows.append((b, F, F1, LOG2 + b * b / 4.0))
rk.write_dat(os.path.join(OUT, "03-sk-phi.dat"), ["beta", "phiRS", "phi1", "ann"], rows)
for b in (1.5, 2.0):
    F, q = phi_rs(b, 0.0)
    F1, q1, m1 = phi_1rsb(b, (0.4, 0.2))
    print(f"   beta={b}: phi_RS={F:.6f} (q={q:.5f}), phi_1RSB={F1:.6f} (q={q1:.5f}, m={m1:.5f}), RS-1RSB={F - F1:.6f}")

# ---------------------------------------------------------------------------------------------
# 5. Toninelli's expansion: for mu = m delta_q + (1-m) delta_{q+delta},
#    dP/dm at m = 1 equals (beta^2 delta^2 / 4)(beta^2 E sech^4 - 1) + O(delta^3) at the RS fixed point.
#    Exact form: E_Y[ e^{-a^2/2} E_Z cosh(Y+aZ) log cosh(Y+aZ) / cosh Y - log cosh Y - a^2/2 ]
#                - (beta^2/4)(2 q delta + delta^2),   a^2 = beta^2 delta,  Y = h + beta sqrt(q) Z'.
# ---------------------------------------------------------------------------------------------
print(f"   ({time.time() - T0:.1f} s)")
print("5. Toninelli expansion of the 1RSB bound around RS")
XG, WG = np.polynomial.hermite_e.hermegauss(150)
WG = WG / WG.sum()


def dPdm_at_1(beta, h, q, delta):
    a = beta * math.sqrt(delta)
    Y = h + beta * math.sqrt(q) * XG                       # outer nodes
    U = Y[:, None] + a * XG[None, :]                       # inner nodes
    # cosh(U) log cosh(U) / cosh(Y) = exp(lncosh(U) - lncosh(Y)) * log cosh(U)
    inner = np.sum(WG[None, :] * np.exp(lncosh(U) - lncosh(Y)[:, None]) * (lncosh(U) + LOG2), axis=1)
    D = math.exp(-a * a / 2) * inner - (lncosh(Y) + LOG2) - a * a / 2
    return float(np.sum(WG * D)) - beta ** 2 / 4 * (2 * q * delta + delta ** 2)


for (b, h) in [(2.0, 1.0), (2.0, 0.5), (1.25, 0.1)]:
    q = q_rs(b, h)
    A = at_param(b, h, q)
    print(f"   beta={b} h={h}: q*={q:.6f} AT={A:.5f}")
    for delta in [0.04, 0.02, 0.01, 0.005]:
        exact = dPdm_at_1(b, h, q, delta)
        pred = b ** 2 * delta ** 2 / 4 * (A - 1)
        print(f"      delta={delta}: dP/dm(1) = {exact:+.4e}   (beta^2 delta^2/4)(AT-1) = {pred:+.4e}   ratio {exact / pred:.4f}")

# ---------------------------------------------------------------------------------------------
# 6. Zero temperature: Auffinger--Chen's Parisi formula for lim max_s N^{-3/2} sum_{i<j} g_ij s_i s_j
#    (xi(s) = s^2/2): P(gamma) = Psi(0,0) - (1/2) int_0^1 t gamma(t) dt, Psi(1,x) = |x|.
# ---------------------------------------------------------------------------------------------
print(f"   ({time.time() - T0:.1f} s)")
print("6. Ground state from the zero-temperature Parisi formula (h = 0)")


def top_level(x, g, s):
    """(1/g) log E exp(g |x + s Z|), in closed form."""
    a = special.log_ndtr(x / s + g * s) + g * x
    b = special.log_ndtr(-x / s + g * s) - g * x
    return g * s * s / 2 + np.logaddexp(a, b) / g


X64, W64 = np.polynomial.hermite_e.hermegauss(80)
W64 = W64 / W64.sum()
LW64 = np.log(W64)


def gse1(q1, g1):
    return float(np.sum(W64 * top_level(math.sqrt(q1) * X64, g1, math.sqrt(1 - q1)))) - g1 * (1 - q1 * q1) / 4


def gse2(q1, q2, g1, g2):
    x = math.sqrt(q1) * X64[:, None] + math.sqrt(q2 - q1) * X64[None, :]
    inner = special.logsumexp(g1 * top_level(x, g2, math.sqrt(1 - q2)) + LW64[None, :], axis=1) / g1
    return float(np.sum(W64 * inner)) - (g1 * (q2 * q2 - q1 * q1) + g2 * (1 - q2 * q2)) / 4


r1 = optimize.minimize(lambda p: gse1(*p) if (0 < p[0] < 1 and p[1] > 0) else 10.0, [0.48, 1.36],
                       method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 4000})


def unpack2(th):
    u = np.exp(th[:3])
    q1, q2 = np.cumsum(u)[:2] / np.sum(u)
    g1 = math.exp(th[3])
    g2 = g1 + math.exp(th[4])
    return q1, q2, g1, g2


th0 = np.array([math.log(0.32), math.log(0.53), math.log(0.15), math.log(0.83), math.log(1.9)])
f2 = lambda th: gse2(*unpack2(th))
r2 = optimize.minimize(f2, th0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 3000})
r2 = optimize.minimize(f2, r2.x, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-13, "maxiter": 3000})
q1, q2, g1, g2 = unpack2(r2.x)
# quadrature check: the 1RSB value with 80 Gauss--Hermite nodes against adaptive quadrature
from scipy import integrate  # noqa: E402
qa, ga = r1.x
v_quad, _ = integrate.quad(lambda z: top_level(math.sqrt(qa) * z, ga, math.sqrt(1 - qa)) * math.exp(-z * z / 2) / math.sqrt(2 * math.pi),
                           -40, 40, limit=400, epsabs=1e-14, epsrel=1e-13)
print(f"   (check: the 1RSB functional at its minimizer by adaptive quadrature: {v_quad - ga * (1 - qa * qa) / 4:.8f}; by Gauss--Hermite: {r1.fun:.8f})")
print(f"   k=0 (RS):   GSE = E|Z| = sqrt(2/pi) = {math.sqrt(2 / math.pi):.6f}")
print(f"   k=1 (1RSB): GSE = {r1.fun:.6f}   (q1={r1.x[0]:.5f}, gamma1={r1.x[1]:.5f})")
print(f"   k=2 (2RSB): GSE = {r2.fun:.6f}   (q1={q1:.4f}, q2={q2:.4f}, gamma1={g1:.4f}, gamma2={g2:.4f})")
print("   full RSB (Oppermann-Schmidt-Sherrington 2007): 0.7631667265(6); ground-state energy per spin = -GSE")
print(f"done in {time.time() - T0:.1f} s")
