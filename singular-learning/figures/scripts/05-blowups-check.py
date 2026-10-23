"""Chapter 5 (Blow-ups), "Measuring it": the volume and the zeta function of three losses, computed
in the original coordinates and on the charts of their resolutions, against the learning
coefficients the resolutions predict.

The losses, each with the uniform prior on [-1, 1]^d:
  quart  K = a^2 + b^4               d = 2, lambda = 3/4,  m = 1 (two point blow-ups)
  cusp   K = (a^2 - b^3)^2           d = 2, lambda = 5/12, m = 1 (three point blow-ups)
  wat    K = (ab + c)^2 + a^2 b^4    d = 3, lambda = 3/4,  m = 1 (Watanabe's example)

For each we compute
  * V(eps) in the original coordinates, by one-dimensional quadrature of exact section lengths
    (deterministic; no knowledge of the resolution is used);
  * V(eps) on the charts: the sum, over the leaf charts of the resolution (each a unit square),
    of the chart volumes, with the Jacobian as weight (quart, cusp);
  * V(eps) by plain Monte Carlo in the original coordinates (sk.volume_fraction, 10^7 prior
    draws, Clopper-Pearson 95% intervals);
  * the prediction C eps^lambda, with C an integral over the exceptional curve that attains lambda
    (Section 5.3 of the chapter): C = B(1/2,1/4)/6, [B(1/3,1/6)+B(1/2,1/6)+B(1/3,1/2)]/10, B(1/4,3/2)/2;
  * zeta(z) = int K^z phi at a few real z, by nested quadrature in the original coordinates and by
    the chart formulas, and the residue at -lambda.
Writes figures/data/05-blowups-volume.dat (deterministic curves) and 05-blowups-mc-{quart,cusp,wat}.dat.
Run from the volume root: python figures/scripts/05-blowups-check.py  (about 40 seconds).
"""
import math
import os
import sys
import time
import warnings

import numpy as np
from scipy import integrate, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk  # noqa: E402

warnings.simplefilter("ignore", integrate.IntegrationWarning)
B = special.beta
OUT = "figures/data"


def quad(f, lo, hi, points=None):
    pts = None if points is None else sorted(x for x in set(points) if lo < x < hi)
    return integrate.quad(f, lo, hi, limit=800, epsabs=0.0, epsrel=1e-12, points=pts)[0]


_GLX, _GLW = np.polynomial.legendre.leggauss(40)


def gl_refined(f, centers, lo=-1.0, hi=1.0, ratio=1.3, depth=90):
    """Gauss-Legendre (40 nodes) on panels refined geometrically toward the given centers; for
    integrands with kinks or integrable singularities at known points. f takes a scalar."""
    edges = {lo, hi}
    for c in centers:
        for k in range(depth):
            for sgn in (-1.0, 1.0):
                e = c + sgn * ratio ** (-k)
                if lo < e < hi:
                    edges.add(e)
    edges = sorted(edges)
    tot = 0.0
    for e0, e1 in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (e0 + e1), 0.5 * (e1 - e0)
        tot += half * sum(wk * f(mid + half * xk) for xk, wk in zip(_GLX, _GLW))
    return tot


def quad_geom(f, lo, hi, ratio=2.0):
    """Integrate over [lo, hi] (0 < lo < hi) on geometric panels: for integrands that decay like powers."""
    tot, x0 = 0.0, lo
    while x0 < hi:
        x1 = min(hi, x0 * ratio)
        tot += quad(f, x0, x1)
        x0 = x1
    return tot


# ------------------------------------------------------------------------------------------------
# constants predicted by the resolutions (integrals over the exceptional curve that attains lambda)
# ------------------------------------------------------------------------------------------------
LAM = {"quart": 0.75, "cusp": 5 / 12, "wat": 0.75}
DIM = {"quart": 2, "cusp": 2, "wat": 3}
CONST = {"quart": B(0.5, 0.25) / 6,
         "cusp": (B(1 / 3, 1 / 6) + B(0.5, 1 / 6) + B(1 / 3, 0.5)) / 10,
         "wat": B(0.25, 1.5) / 2}


# ------------------------------------------------------------------------------------------------
# volumes in the original coordinates (sections)
# ------------------------------------------------------------------------------------------------
def V_quart_orig(eps):
    """(1/4) int db |{a in [-1,1]: a^2 < eps - b^4}|."""
    bmax = min(1.0, eps ** 0.25)
    return 0.25 * 2 * quad(lambda b: 2 * math.sqrt(max(0.0, eps - b ** 4)), 0.0, bmax)


def V_cusp_orig(eps):
    """(1/4) int db |{a in [-1,1]: |a^2 - b^3| < sqrt(eps)}|; the section is 2(sqrt(hi) - sqrt(lo))."""
    r = math.sqrt(eps)

    def sec(b):
        lo, hi = max(0.0, b ** 3 - r), min(1.0, b ** 3 + r)
        return 2 * (hi - lo) / (math.sqrt(hi) + math.sqrt(lo)) if hi > lo else 0.0
    b0 = r ** (1 / 3)
    if b0 >= 1:
        return 0.25 * quad(sec, -1, 1, points=[0.0])
    tot = quad(sec, -b0, 0.0) + quad(sec, 0.0, b0)            # b^3 < r: lo = 0
    b1 = (1 - r) ** (1 / 3)                                    # beyond b1: hi = 1
    tot += quad_geom(sec, b0, b1, ratio=1.5) + quad(sec, b1, 1.0)
    return 0.25 * tot


def V_wat_orig(eps):
    """(1/8) int da db |{c in [-1,1]: (ab + c)^2 < eps - a^2 b^4}|.  For eps <= 0.1 the c-interval
    [-ab - rho, -ab + rho] lies inside [-1, 1] (|ab| <= eps^(1/4) wherever rho > 0), so the section
    is 2 rho; for |a| >= sqrt(eps) the b-integral of 2 rho is B(1/4,3/2) eps^(3/4) |a|^(-1/2)."""
    assert eps <= 0.1
    Bq = B(0.25, 1.5)
    sq = math.sqrt(eps)
    outer = Bq * eps ** 0.75 * 4 * (1 - eps ** 0.25)
    G = lambda a: 2 * quad(lambda b: 2 * math.sqrt(max(0.0, eps - a * a * b ** 4)), 0.0, 1.0)
    inner = 2 * quad(G, 0.0, sq)
    return (outer + inner) / 8


# ------------------------------------------------------------------------------------------------
# volumes on the charts (each leaf chart is the unit square [-1,1]^2; weight = |Jacobian| / 4)
# ------------------------------------------------------------------------------------------------
def poly_sections(coeffs, h, rr):
    """int over {x in [-1,1] : |P(x)| < rr} of |x|^h dx, for the polynomial P with numpy coeffs."""
    P = np.poly1d(coeffs)
    cand = [-1.0, 0.0, 1.0]
    for sgn in (1.0, -1.0):
        for z in (P - sgn * rr).r:
            if abs(z.imag) < 1e-12 and -1.0 < z.real < 1.0:
                cand.append(float(z.real))
    cand = sorted(set(cand))
    F = lambda x: math.copysign(abs(x) ** (h + 1) / (h + 1), x)
    tot = 0.0
    for x0, x1 in zip(cand[:-1], cand[1:]):
        if x1 > x0 and abs(P(0.5 * (x0 + x1))) < rr:
            tot += F(x1) - F(x0)
    return tot


def V_quart_charts(eps):
    """Leaf charts I (u,uv), II.1 (s^2 t, st) and II.2 (s t^2, t); returns their three volumes."""
    def ustar2(v):                       # u^2 + u^4 v^4 = eps  ->  u^2
        x = 4 * eps * v ** 4
        return eps if v == 0 else x / (2 * v ** 4 * (math.sqrt(1 + x) + 1))
    VI = 0.25 * 2 * quad(lambda v: min(1.0, ustar2(v)), 0.0, 1.0)          # int_{|u|<u*} |u| du = u*^2
    rho1 = lambda t: min(1.0, (eps / (t * t * (1 + t * t))) ** 0.25)
    VII1 = 0.25 * 2 * quad(lambda t: t * (2 / 3) * rho1(t) ** 3 if t > 0 else 0.0, 0.0, 1.0,
                           points=[math.sqrt(eps), eps ** 0.25])
    VII2 = 0.25 * 2 * quad(lambda s: (2 / 3) * min(1.0, (eps / (1 + s * s)) ** 0.25) ** 3, 0.0, 1.0)
    return VI, VII1, VII2


def _bisect(F, lo, hi, n_iter=80):
    """Vectorized bisection for F(x) = 0 on [lo, hi] with F(lo) < 0 < F(hi) (arrays)."""
    lo, hi = lo.copy(), hi.copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        neg = F(mid) < 0
        lo, hi = np.where(neg, mid, lo), np.where(neg, hi, mid)
    return 0.5 * (lo + hi)


def _unimodal_section(f_pos, f_neg, xpeak, r, h):
    """For y in [0,1] (arrays), the measure int_{x in [-1,1], |f| < r} |x|^h dx, where on [0,1] the
    nonnegative function f_pos(x) rises on [0, xpeak] and falls on [xpeak, 1], and on [-1,0]
    f_neg(|x|) rises with |x|."""
    one, zero = np.ones_like(xpeak), np.zeros_like(xpeak)
    P = lambda x: x ** (h + 1) / (h + 1)
    # negative side
    gneg1 = f_neg(one)
    rho_m = np.where(gneg1 <= r, one, _bisect(lambda x: f_neg(x) - r, zero, one))
    out = P(rho_m)
    # positive side: rising branch
    fmax = f_pos(xpeak)
    rho_p = np.where(fmax <= r, xpeak, _bisect(lambda x: f_pos(x) - r, zero, xpeak))
    out = out + P(rho_p)
    # falling branch on [xpeak, 1]
    f1 = f_pos(one)
    whole = fmax <= r                                     # the whole falling branch is inside
    part = (~whole) & (f1 < r) & (xpeak < 1)
    sigma = np.where(part, _bisect(lambda x: r - f_pos(x), xpeak, one), one)
    out = out + np.where(whole, P(one) - P(xpeak), np.where(part, P(one) - P(sigma), 0.0))
    return out


def _gl_vec(F, edges):
    x = np.concatenate([0.5 * (e0 + e1) + 0.5 * (e1 - e0) * _GLX for e0, e1 in zip(edges[:-1], edges[1:])])
    w = np.concatenate([0.5 * (e1 - e0) * _GLW for e0, e1 in zip(edges[:-1], edges[1:])])
    return float(np.sum(w * F(x)))


def _edges(points, lo=0.0, hi=1.0, ratio=1.25, depth=120):
    e = {lo, hi}
    for c in points:
        for k in range(depth):
            for sgn in (-1.0, 1.0):
                x = c + sgn * ratio ** (-k)
                if lo < x < hi:
                    e.add(x)
    return sorted(e)


def V_cusp_charts(eps):
    """Leaf charts I (u,uv), II.2 (s t^2, t), II.1.a (p^3 q, p^2 q) and II.1.b (p^2 q^3, p q^2).
    f = a^2 - b^3 in them: -u^2(uv^3 - 1), t^3(s^2 t - 1), -p^6 q^2 (q - 1), p^3 q^6 (p - 1)."""
    r = math.sqrt(eps)
    # chart I: |f| = u^2 (1 - u v^3) for u >= 0 and |u|^2 (1 + |u| v^3) for u < 0 (v >= 0; the chart is
    # symmetric under (u, v) -> (-u, -v)); weight |u|. The curve leaves the square at the corner u = v = 1.
    def MI(v):
        v3 = v ** 3
        xpeak = np.where(v3 > 2 / 3, 2 / (3 * np.maximum(v3, 1e-300)), 1.0)
        return _unimodal_section(lambda x: x * x * (1 - x * v3), lambda x: x * x * (1 + x * v3), xpeak, r, 1)
    vstar = (max(0.0, 1 - r)) ** (1 / 3)
    VI = 2 * _gl_vec(MI, _edges([0.0, 1.0, vstar, (2 / 3) ** (1 / 3)]))
    # chart II.2: |f| = t^3 (1 - s^2 t) for t >= 0 and |t|^3 (1 + s^2 |t|) for t < 0; weight t^2
    def MII2(s):
        s2 = s * s
        xpeak = np.where(s2 > 0.75, 3 / (4 * np.maximum(s2, 1e-300)), 1.0)
        return _unimodal_section(lambda x: x ** 3 * (1 - s2 * x), lambda x: x ** 3 * (1 + s2 * x), xpeak, r, 2)
    sstar = math.sqrt(max(0.0, 1 - r))
    VII2 = 2 * _gl_vec(MII2, _edges([0.0, 1.0, sstar, math.sqrt(0.75)]))
    # II.1.a: |p^6 q^2 (q-1)| < r  ->  |p| < (r / (q^2 |q-1|))^(1/6); weight p^4 |q|
    ra = lambda q: min(1.0, (r / (q * q * abs(q - 1))) ** (1 / 6)) if q != 0 else 1.0
    VIIa = quad(lambda q: abs(q) * 0.4 * ra(q) ** 5, -1.0, 1.0, points=[0.0])
    # II.1.b: |p^3 q^6 (p-1)| < r  ->  |q| < (r / (|p|^3 |p-1|))^(1/6); weight p^2 q^4
    rb = lambda p: min(1.0, (r / (abs(p) ** 3 * abs(p - 1))) ** (1 / 6)) if p != 0 else 1.0
    VIIb = quad(lambda p: p * p * 0.4 * rb(p) ** 5, -1.0, 1.0, points=[0.0])
    return [0.25 * x for x in (VI, VII2, VIIa, VIIb)]


# ------------------------------------------------------------------------------------------------
# zeta functions: Monte Carlo in the original coordinates (E[K^z] under the prior, whose variance is
# finite only for z > -lambda/2), and the chart formulas, which exhibit the pole
# ------------------------------------------------------------------------------------------------
def zeta_quart_charts(z):
    """Chart I + [J_a(z) + J_b(z)] / (4z + 3): the pole at -3/4 is explicit."""
    zI = quad(lambda v: quad(lambda u: u ** (2 * z + 1) * (1 + u * u * v ** 4) ** z, 0.0, 1.0), 0.0, 1.0)
    Ja = quad(lambda t: t ** (2 * z + 1) * (1 + t * t) ** z, 0.0, 1.0)
    Jb = quad(lambda s: (1 + s * s) ** z, 0.0, 1.0)
    return zI + (Ja + Jb) / (4 * z + 3)


def zeta_cusp_charts(z):
    """Charts I and II.2 by nested quadrature; II.1.a and II.1.b carry the factor 2/(12z + 5)."""
    zI = quad(lambda v: quad(lambda u: abs(u) ** (4 * z + 1) * abs(1 - u * v ** 3) ** (2 * z), -1.0, 1.0,
                             points=[0.0]), -1.0, 1.0, points=[0.0])
    zII2 = quad(lambda s: quad(lambda t: abs(t) ** (6 * z + 2) * abs(s * s * t - 1) ** (2 * z), -1.0, 1.0,
                               points=[0.0]), -1.0, 1.0, points=[0.0])
    Ja = quad(lambda q: abs(q) ** (4 * z + 1) * abs(q - 1) ** (2 * z), -1.0, 1.0, points=[0.0])
    Jb = quad(lambda p: abs(p) ** (6 * z + 2) * abs(p - 1) ** (2 * z), -1.0, 1.0, points=[0.0])
    return 0.25 * (zI + zII2 + 2 * (Ja + Jb) / (12 * z + 5))


# ------------------------------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("predicted constants C (V ~ C eps^lambda):", {k: round(v, 6) for k, v in CONST.items()})

    # --- deterministic volumes on a grid of eps
    eps_grid = 10.0 ** np.arange(-1.0, -14.01, -0.5)
    rows = []
    maxdiff = {"quart": 0.0, "cusp": 0.0}
    for eps in eps_grid:
        vq, vc = V_quart_orig(eps), V_cusp_orig(eps)
        vw = V_wat_orig(eps)
        cq, cc = sum(V_quart_charts(eps)), sum(V_cusp_charts(eps))
        maxdiff["quart"] = max(maxdiff["quart"], abs(cq / vq - 1))
        maxdiff["cusp"] = max(maxdiff["cusp"], abs(cc / vc - 1))
        row = [eps]
        for key, vo, vch in (("quart", vq, cq), ("cusp", vc, cc), ("wat", vw, vw)):
            C, lam, d = CONST[key], LAM[key], DIM[key]
            row += [vo, vch, C * eps ** lam, vo / (C * eps ** lam)]
        rows.append(row)
    hdr = ["eps"]
    for key in ("quart", "cusp", "wat"):
        hdr += [key, key + "chart", key + "pred", key + "comp"]
    # regular reference: slope d/2 through the value at eps = 0.1
    for key in ("quart", "cusp", "wat"):
        v1 = rows[0][1 + 4 * ("quart", "cusp", "wat").index(key)]
        hdr.append(key + "reg")
        for row in rows:
            row.append(v1 * (row[0] / 0.1) ** (DIM[key] / 2))
    sk.write_dat(os.path.join(OUT, "05-blowups-volume.dat"), hdr, rows)
    print("max |chart sum / original - 1| over eps in [1e-14, 1e-1]:",
          {k: f"{v:.1e}" for k, v in maxdiff.items()})
    for key in ("quart", "cusp", "wat"):
        i = 1 + 4 * ("quart", "cusp", "wat").index(key)
        comp = [r[i + 3] for r in rows]
        e = [r[0] for r in rows]
        print(f"  {key}: V/(C eps^lambda) at eps = 1e-2, 1e-6, 1e-10, 1e-14:",
              [round(comp[list(np.round(np.log10(e), 2)).index(x)], 4) for x in (-2.0, -6.0, -10.0, -14.0)])
        # local slope d log V / d log eps between 1e-13 and 1e-14
        j = len(e) - 3
        slope = math.log(rows[j + 2][i] / rows[j][i]) / math.log(e[j + 2] / e[j])
        print(f"     local slope of V between eps = 1e-13 and 1e-14: {slope:.4f} (lambda = {LAM[key]:.4f})")
        # least-squares slope of log V on log eps over [1e-14, 1e-6]
        sel = [(r[0], r[i]) for r in rows if r[0] <= 1.0001e-6]
        sl, _ = sk.loglog_slope([x for x, _ in sel], [y for _, y in sel])
        print(f"     least-squares slope over [1e-14, 1e-6]: {sl:.4f}")
    # the leading charts of the cusp
    for eps in (1e-6, 1e-14):
        parts = V_cusp_charts(eps)
        print(f"  cusp at eps={eps:.0e}: charts meeting E3 give {(parts[2]+parts[3])/(CONST['cusp']*eps**(5/12)):.4f}"
              f" of C eps^(5/12); the other charts {(parts[0]+parts[1])/eps**0.5:.4f} eps^(1/2)")
    # the cusp's second term: V = C eps^(5/12) + B eps^(1/2) + ..., with B = -1 (the band around the
    # curve a^2 = b^3 that the square cuts off where the curve leaves it)
    for eps in (1e-6, 1e-10, 1e-14):
        v = V_cusp_orig(eps)
        print(f"  cusp at eps={eps:.0e}: (V - C eps^(5/12)) / eps^(1/2) = {(v - CONST['cusp'] * eps ** (5 / 12)) / eps ** 0.5:.4f}")
    sel = [(r[0], r[5]) for r in rows if r[0] >= 0.99e-6]
    sl, _ = sk.loglog_slope([x for x, _ in sel], [y for _, y in sel])
    print(f"  cusp: least-squares slope of the exact V over [1e-6, 1e-1]: {sl:.4f} (lambda = {5 / 12:.4f})")
    for eps in (1e-6, 1e-14):
        parts = V_quart_charts(eps)
        print(f"  quart at eps={eps:.0e}: chart I {parts[0]/eps:.4f} eps; II.1 {parts[1]/eps**0.75:.5f} eps^(3/4);"
              f" II.2 {parts[2]/eps**0.75:.5f} eps^(3/4) (limits I2/6 = 0.59705, I1/6 = 0.27697)")

    # --- Monte Carlo in the original coordinates
    Kfun = {"quart": lambda w: w[:, 0] ** 2 + w[:, 1] ** 4,
            "cusp": lambda w: (w[:, 0] ** 2 - w[:, 1] ** 3) ** 2,
            "wat": lambda w: (w[:, 0] * w[:, 1] + w[:, 2]) ** 2 + w[:, 0] ** 2 * w[:, 1] ** 4}
    m = 10_000_000
    for j, key in enumerate(("quart", "cusp", "wat")):
        box = [(-1, 1)] * DIM[key]
        mc = sk.volume_fraction(Kfun[key], box, eps_grid, m, seed=5005 + j, chunk=1_000_000)
        out = []
        for (eps, V, lo, hi, cnt) in mc:
            if cnt < 20:
                continue
            C, lam = CONST[key], LAM[key]
            s = C * eps ** lam
            out.append([eps, V, V - lo, hi - V, V / s, (V - lo) / s, (hi - V) / s, int(cnt)])
        sk.write_dat(os.path.join(OUT, f"05-blowups-mc-{key}.dat"),
                     ["eps", "V", "lo", "hi", "comp", "complo", "comphi", "count"], out)
        cover = 0
        for r in out:
            vo = {"quart": V_quart_orig, "cusp": V_cusp_orig, "wat": V_wat_orig}[key](r[0])
            cover += (r[1] - r[2] <= vo <= r[1] + r[3])
        print(f"  Monte Carlo {key}: {len(out)} values of eps down to {out[-1][0]:.0e};"
              f" {cover} of {len(out)} intervals contain the exact volume")

    # --- zeta functions and residues
    gen = sk.rng(5050)
    w = gen.uniform(-1.0, 1.0, (m, 2))
    # Monte Carlo is used only for z > -lambda/4, where K^z has a finite fourth moment, so that the
    # Student interval is trustworthy; closer to the pole only the chart formulas are evaluated.
    for key, zs, chartfn in (("quart", (-0.05, -0.1, -0.15, -0.3, -0.5, -0.7), zeta_quart_charts),
                             ("cusp", (-0.025, -0.05, -0.1, -0.2, -0.3, -0.4), zeta_cusp_charts)):
        Kw = Kfun[key](w)
        lam = LAM[key]
        print(f"zeta for {key} (prior uniform): z, Monte Carlo in the original coordinates"
              f" (10^7 draws, z > -lambda/4 = {-lam / 4:.3f}), charts, (z + lambda) zeta")
        for z in zs:
            zc = chartfn(z)
            if z > -lam / 4:
                x = Kw ** z
                mu, half = float(x.mean()), 1.96 * float(x.std(ddof=1)) / math.sqrt(m)
                mc = f"{mu:.5f} +- {half:.5f}"
            else:
                mc = "(not estimated)     "
            print(f"  z = {z:+.3f}: {mc}   charts {zc:.5f}   (z + lambda) zeta = {(z + lam) * zc:.5f}")
        del Kw
    Ja = quad(lambda t: t ** -0.5 * (1 + t * t) ** -0.75, 0.0, 1.0)
    Jb = quad(lambda s: (1 + s * s) ** -0.75, 0.0, 1.0)
    print(f"  residue at -3/4 from the charts (Ja + Jb)/4 = {(Ja + Jb) / 4:.8f}; lambda C = {0.75 * CONST['quart']:.8f}")
    z = -5 / 12
    Ja = quad(lambda q: abs(q) ** (4 * z + 1) * abs(q - 1) ** (2 * z), -1.0, 1.0, points=[0.0, 1.0])
    Jb = quad(lambda p: abs(p) ** (6 * z + 2) * abs(p - 1) ** (2 * z), -1.0, 1.0, points=[0.0, 1.0])
    print(f"  residue at -5/12 from the charts 2(Ja + Jb)/(4*12) = {2 * (Ja + Jb) / 48:.8f};"
          f" lambda C = {5 / 12 * CONST['cusp']:.8f}")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
