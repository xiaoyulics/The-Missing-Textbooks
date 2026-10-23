"""Chapter 6 (Newton Polygons): Newton's prediction against measured volume exponents.

Functions. A "positive even sum" is K(a, b) = sum_v c_v a^v1 b^v2 with even exponents, positive coefficients and
no constant term. The prior is uniform on [-1, 1]^2, so by symmetry
    V(eps) = phi{K < eps} = Leb{(a, b) in (0, 1]^2 : K(a, b) < eps}.
Thirty random sums: 2 to 5 distinct exponents from {0, 2, 4, 6}^2 minus (0, 0), coefficients log-uniform on
[0.2, 5] (generator sltkit.rng(6)).

Newton's prediction (exact rational arithmetic): the Newton distance t, where the diagonal meets the boundary of
the Newton polygon; lambda = 1/t, and m = 2 if (t, t) is a vertex, 1 otherwise. The script also computes the
leading constant C of V(eps) ~ C eps^lambda (log 1/eps)^(m-1) from the zeta function on the normal fan
(Proposition prop:newton:zeta): in the vertex case C = psi0 * c_(t,t)^(-1/t), psi0 the length of the optimal face.

Measurements.
  * quadrature: V(eps) = int_0^inf e^{-x} beta_eps(e^{-x}) dx in logarithmic coordinates x = -log a, where
    beta_eps(a) = sup{b in [0, 1] : K(a, b) < eps} is found by Newton's method on y = -log b (K is increasing
    in b). Adaptive Gauss-Kronrod (scipy quad, relative tolerance 1e-12) on panels; checked against the exact
    volumes of a^2 b^2, a^4 + b^6, a^2 + b^4. The exponent and the multiplicity are measured by least squares
    of log V on (-L, log L, 1, 1/L) at L = log(1/eps) = 250, 500, 1000, 2000, 4000 (no input from the theory).
  * naive Monte Carlo: N = 2 000 000 uniform points of (0, 1]^2 per function (seed 100 + k); the slope of
    log V between eps1 = 1e-2 and eps2 = 1e-4, log(n1/n2)/log(eps1/eps2) with n_i the counts below eps_i; its
    95% interval comes from the exact (Clopper-Pearson) interval of n2/n1 given n1 (binomial conditionally).
Degenerate examples (exact volumes, mpmath, 40 digits; exponents fitted the same way at L = 200, 300, 450, 700,
1000; the naive Monte Carlo slope as above, on [-1, 1]^2 or [-1, 1]^4, seed 200 + index):
    D1 (a - b)^2            D2 (a - b^2)^2          D3 (a + b)^2 + b^4
    D4 (a^2 - b^2)^2        D5 (a^2 - b^3)^2 (cusp) D6 (a1 b1 + a2 b2)^2 / 2 on [-1, 1]^4 (two linear units)
    D7 (ab - 1/4)^2 / 2 (the product model at the generic truth c0 = 1/4)
Output: figures/data/06-newton-random.dat, figures/data/06-newton-degenerate.dat.
Run from the volume root:  python figures/scripts/06-newton-random.py   (about half a minute)
"""
import math
import os
import sys
import time
import warnings
from fractions import Fraction as Fr
from math import exp, expm1, gcd, inf, isfinite, log

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
from scipy import integrate, stats  # noqa: E402

import sltkit as sk  # noqa: E402

OUT = os.path.join("figures", "data")
EXPS = [(i, j) for i in (0, 2, 4, 6) for j in (0, 2, 4, 6) if (i, j) != (0, 0)]


# ------------------------------------------------------------------------------------------------
# Newton data, exactly
# ------------------------------------------------------------------------------------------------
def boundary(S):
    """Vertices of the Newton polygon, top left to bottom right."""
    best = {}
    for (i, j) in S:
        best[i] = min(j, best.get(i, j))
    mono = []
    for (i, j) in sorted(best.items()):
        if not mono or j < mono[-1][1]:
            mono.append((i, j))
    hull = []
    for p in mono:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def fan(S):
    """Rays of the normal fan (1,0), edge normals, (0,1), and mu(eta) = min_v <v, eta>."""
    V = boundary(S)
    rays = [(1, 0)]
    for (a1, b1), (a2, b2) in zip(V[:-1], V[1:]):
        p = (b1 - b2, a2 - a1)
        g = gcd(*p)
        rays.append((p[0] // g, p[1] // g))
    rays.append((0, 1))
    mu = [min(v[0] * e[0] + v[1] * e[1] for v in S) for e in rays]
    return V, rays, mu


def newton(Sc):
    """t (Fraction), m, and the leading constant C of V ~ C eps^(1/t) L^(m-1) (uniform prior on [-1,1]^2)."""
    S = list(Sc)
    V, rays, mu = fan(S)
    ratio = [Fr(e[0] + e[1], u) if u > 0 else None for e, u in zip(rays, mu)]
    lam = min(r for r in ratio if r is not None)
    t = 1 / lam
    mins = [k for k, r in enumerate(ratio) if r == lam]
    vertex = any(k + 1 in mins for k in mins)
    if vertex:
        k = [k for k in mins if k + 1 in mins][0]
        e, f = rays[k], rays[k + 1]
        psi0 = abs(Fr(e[0], mu[k]) - Fr(f[0], mu[k + 1]))   # length of the optimal face, first coordinate
        C = float(psi0) * Sc[V[k]] ** (-float(lam))
        return t, 2, C, V
    k = mins[0]
    res = 0.0
    for other in (k - 1, k + 1):
        if other < 0 or other >= len(rays):
            continue
        e, eo = rays[k], rays[other]
        D = abs(e[0] * eo[1] - e[1] * eo[0])
        B = float(eo[0] + eo[1] - mu[other] * lam)
        terms = [(c, v[0] * eo[0] + v[1] * eo[1] - mu[other]) for v, c in Sc.items()
                 if v[0] * e[0] + v[1] * e[1] == mu[k]]
        val = integrate.quad(lambda x: x ** (B - 1) * sum(c * x ** p for c, p in terms) ** (-float(lam)),
                             0, 1, limit=200)[0]
        res += D / mu[k] * val
    return t, 1, res / float(lam), V


# ------------------------------------------------------------------------------------------------
# V(eps) by quadrature in logarithmic coordinates
# ------------------------------------------------------------------------------------------------
def logV(Sc, L, rel=1e-12):
    """log V(e^{-L}) for a positive even sum with the uniform prior on [-1, 1]^2."""
    T = [(float(v[0]), float(v[1]), log(c)) for v, c in Sc.items()]
    Ta = [u for u in T if u[1] == 0.0]
    Tb = [u for u in T if u[1] > 0.0]
    t, _, _, _ = newton(Sc)
    shift = L / float(t)

    def lse(zs):
        mx = max(zs)
        return mx + log(sum(exp(z - mx) for z in zs))

    def logA(x):
        return lse([lc - v1 * x for v1, v2, lc in Ta]) if Ta else -inf

    def ysol(x):
        la = logA(x)
        if la >= -L:
            return inf
        lr = -L + log(-expm1(la + L)) if la > -inf else -L   # log(eps - pure-a part)
        if not Tb:
            return 0.0
        z0 = [lc - v1 * x for v1, v2, lc in Tb]
        y = 0.0
        for _ in range(100):   # Newton from the left on a convex decreasing function: monotone
            zs = [z - v2 * y for z, (v1, v2, lc) in zip(z0, Tb)]
            mx = max(zs)
            ws = [exp(z - mx) for z in zs]
            s = sum(ws)
            h = mx + log(s) - lr
            if h <= 0.0:
                break
            step = h * s / sum(w * v2 for w, (v1, v2, lc) in zip(ws, Tb))
            y += step
            if step < 1e-15 * (1.0 + y):
                break
        return y

    def f(x):
        y = ysol(x)
        return exp(-x - y + shift) if isfinite(y) else 0.0

    def bisect(pred, lo, hi):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if pred(mid) else (mid, hi)
        return hi
    x0 = 0.0
    if Ta:
        hi = 1.0
        while logA(hi) >= -L:
            hi *= 2
        x0 = bisect(lambda x: logA(x) < -L, 0.0, hi)
    T0 = [u for u in T if u[0] == 0.0]
    if T0 and lse([lc for v1, v2, lc in T0]) >= -L:
        x1 = inf
    else:
        hi = 1.0
        while lse([lc - v1 * hi for v1, v2, lc in T]) >= -L:
            hi *= 2
        x1 = bisect(lambda x: lse([lc - v1 * x for v1, v2, lc in T]) < -L, 0.0, hi)
    _, rays, mu = fan(list(Sc))
    upper = x1 if isfinite(x1) else 20.0 * L + 200.0
    bps = {x0, upper}
    for e, u in zip(rays, mu):
        if u > 0 and e[0] > 0:
            for d in (0.0, 2.0, 10.0):
                bps.update((L * e[0] / u - d, L * e[0] / u + d))
    bps.update(x0 + k * (upper - x0) / 20.0 for k in range(1, 20))
    pts = sorted(b for b in bps if x0 <= b <= upper)
    total = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for lo, hi in zip(pts[:-1], pts[1:]):
            if hi - lo > 1e-12:
                total += integrate.quad(f, lo, hi, limit=200, epsabs=1e-17, epsrel=rel)[0]
    if isfinite(x1):
        total += exp(-x1 + shift)
    return log(total) - shift


def fit_exponent(Ls, lv):
    """Least squares of log V on (-L, log L, 1, 1/L): returns (lambda, m)."""
    Ls = np.asarray(Ls, float)
    X = np.vstack([-Ls, np.log(Ls), np.ones_like(Ls), 1.0 / Ls]).T
    coef = np.linalg.lstsq(X, np.asarray(lv, float), rcond=None)[0]
    return float(coef[0]), float(coef[1] + 1.0)


def naive_slope(Kvals, e1=1e-2, e2=1e-4):
    """Slope of log V between e1 and e2 from Monte Carlo values of K, with an exact 95% interval."""
    n1 = int(np.count_nonzero(Kvals < e1))
    n2 = int(np.count_nonzero(Kvals < e2))
    s = math.log(e1 / e2)
    lo_r = stats.beta.ppf(0.025, n2, n1 - n2 + 1) if n2 > 0 else 0.0
    hi_r = stats.beta.ppf(0.975, n2 + 1, n1 - n2) if n2 < n1 else 1.0
    est = -math.log(n2 / n1) / s
    return est, -math.log(hi_r) / s, -math.log(lo_r) / s, n1, n2


def K_eval(Sc, a, b):
    out = np.zeros_like(a)
    for (i, j), c in Sc.items():
        out += c * a ** i * b ** j
    return out


def random_sum(gen):
    k = int(gen.integers(2, 6))
    idx = gen.choice(len(EXPS), size=k, replace=False)
    c = np.exp(gen.uniform(math.log(0.2), math.log(5.0), size=k))
    return {EXPS[i]: float(c[n]) for n, i in enumerate(idx)}


# ------------------------------------------------------------------------------------------------
# exact volumes of the degenerate examples (probability under the uniform prior)
# ------------------------------------------------------------------------------------------------
mp.mp.dps = 40


def V_D1(L):          # (a - b)^2: V = s - s^2/4, s = sqrt(eps)
    s = mp.e ** (-mp.mpf(L) / 2)
    return s - s * s / 4


def V_D2(L):          # (a - b^2)^2 < eps  <=>  |a - b^2| < s
    s = mp.e ** (-mp.mpf(L) / 2)
    r = mp.sqrt(1 - s)
    return (4 * s * r + 2 * ((1 + s) * (1 - r) - (1 - (1 - s) ** mp.mpf(1.5)) / 3)) / 4


C4 = mp.quad(lambda v: mp.sqrt(1 - v ** 4), [0, 1])


def V_D3(L):          # (a + b)^2 + b^4: exactly C4 eps^(3/4) for small eps
    return C4 * mp.e ** (-mp.mpf(L) * 3 / 4)


def _logquad(g, lo, hi):
    """int_lo^hi g(b) db by the substitution b = lo e^z (the integrands below vary on a logarithmic scale)."""
    zmax = mp.log(hi / lo)
    pts = [z for z in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if z < zmax] + [zmax]
    return mp.quad(lambda z: g(lo * mp.e ** z) * lo * mp.e ** z, pts)


def V_D4(L):          # (a^2 - b^2)^2: int_0^1 sqrt(min(1, b^2+s)) - sqrt(max(0, b^2-s)) db
    s = mp.e ** (-mp.mpf(L) / 2)
    rs, top = mp.sqrt(s), mp.sqrt(1 - s)
    part1 = mp.quad(lambda b: mp.sqrt(b * b + s), [0, rs])
    part2 = _logquad(lambda b: 2 * s / (mp.sqrt(b * b + s) + mp.sqrt(b * b - s)), rs, top)
    part3 = mp.quad(lambda b: 1 - mp.sqrt(b * b - s), [top, 1])
    return part1 + part2 + part3


def V_D5(L):          # cusp (a^2 - b^3)^2: (1/2) int_{-1}^{1} sqrt(min(1,b^3+s)) - sqrt(max(0,b^3-s)) db
    s = mp.e ** (-mp.mpf(L) / 2)
    c = s ** (mp.mpf(1) / 3)
    top = (1 - s) ** (mp.mpf(1) / 3)
    neg = mp.quad(lambda u: mp.sqrt(s - u ** 3), [0, c])    # b = -u < 0 with b^3 + s > 0
    pos1 = mp.quad(lambda b: mp.sqrt(b ** 3 + s), [0, c])
    pos2 = _logquad(lambda b: 2 * s / (mp.sqrt(b ** 3 + s) + mp.sqrt(b ** 3 - s)), c, top)
    pos3 = mp.quad(lambda b: 1 - mp.sqrt(b ** 3 - s), [top, 1])
    return (neg + pos1 + pos2 + pos3) / 2


def rho_D6(c):        # density of X + Y at c (0 < c < 1), X, Y independent products of two uniforms on [-1, 1]
    c = mp.mpf(c)
    f = lambda x: mp.log(1 / abs(x)) / 2                     # density of X = a1 b1        # noqa: E731
    return mp.quad(lambda x: f(x) * f(c - x), [c - 1, 0, c, 1])


def V_D6(L):          # (a1 b1 + a2 b2)^2 / 2 on [-1,1]^4: P(|X + Y| < r) with r = sqrt(2 eps)
    # V = int_{-r}^{r} rho = 2 r rho(r) (1 + O(r log^2(1/r))): rho is even, continuous, rho(c) - rho(0) = O(c log^2 c)
    r = mp.sqrt(2) * mp.e ** (-mp.mpf(L) / 2)
    return 2 * r * rho_D6(r)


def V_D7(L):          # (ab - 1/4)^2 / 2: P(|ab - 1/4| < r), density of ab = log(1/|c|)/2
    with mp.workdps(int(L / 4.6) + 60):                      # the difference below cancels ~L/4.6 digits
        r = mp.sqrt(2) * mp.e ** (-mp.mpf(L) / 2)
        c0 = mp.mpf(1) / 4
        H = lambda c: c * (1 - mp.log(c))                    # antiderivative of log(1/c)  # noqa: E731
        return +((H(c0 + r) - H(c0 - r)) / 2)


DEGEN = [  # name, V function, Newton (t, m), truth (lambda, m), MC sampler
    ("(a-b)^2", V_D1, (Fr(1), 1), (0.5, 1)),
    ("(a-b^2)^2", V_D2, (Fr(4, 3), 1), (0.5, 1)),
    ("(a+b)^2+b^4", V_D3, (Fr(1), 1), (0.75, 1)),
    ("(a^2-b^2)^2", V_D4, (Fr(2), 1), (0.5, 2)),
    ("(a^2-b^3)^2", V_D5, (Fr(12, 5), 1), (5 / 12, 1)),
    ("(a1b1+a2b2)^2/2", V_D6, (Fr(1), 3), (0.5, 1)),
    ("(ab-1/4)^2/2", V_D7, (Fr(1), 1), (0.5, 1)),
]


def degen_K(name, gen, N):
    u = gen.uniform(-1, 1, size=(4, N))
    a, b = u[0], u[1]
    if name == "(a-b)^2":
        return (a - b) ** 2
    if name == "(a-b^2)^2":
        return (a - b * b) ** 2
    if name == "(a+b)^2+b^4":
        return (a + b) ** 2 + b ** 4
    if name == "(a^2-b^2)^2":
        return (a * a - b * b) ** 2
    if name == "(a^2-b^3)^2":
        return (a * a - b ** 3) ** 2
    if name == "(a1b1+a2b2)^2/2":
        return 0.5 * (u[0] * u[1] + u[2] * u[3]) ** 2
    if name == "(ab-1/4)^2/2":
        return 0.5 * (a * b - 0.25) ** 2
    raise ValueError(name)


def main():
    t0 = time.time()
    # checks of the quadrature against exact volumes
    for Sc, exact in [({(2, 2): 1.0}, lambda L: math.log(math.exp(-L / 2) * (1 + L / 2))),
                      ({(4, 0): 0.5, (0, 6): 0.5}, None), ({(2, 0): 1.0, (0, 4): 1.0}, None)]:
        t, m, C, _ = newton(Sc)
        for L in (20.0, 200.0):
            ex = exact(L) if exact else math.log(C) - L / float(t)
            print(f"check {sorted(Sc)} L={L:5.0f}: quadrature - exact = {logV(Sc, L) - ex:.2e}")

    gen = sk.rng(6)
    LS = [250.0, 500.0, 1000.0, 2000.0, 4000.0]
    rows = []
    for k in range(30):
        Sc = random_sum(gen)
        t, m, C, V = newton(Sc)
        lv = [logV(Sc, L) for L in LS]
        lam_q, m_q = fit_exponent(LS, lv)
        g2 = sk.rng(100 + k)
        a, b = g2.random(2_000_000), g2.random(2_000_000)
        est, lo, hi, n1, n2 = naive_slope(K_eval(Sc, a, b))
        lam = 1.0 / float(t)
        rows.append((k, lam, m, lam_q, m_q, est, hi - est, est - lo, 1 if m == 2 else 0))
        terms = " + ".join(f"{c:.2f} a^{v[0]} b^{v[1]}" for v, c in sorted(Sc.items()))
        print(f"{k:2d} t={str(t):5s} lam={lam:.4f} m={m} C={C:.4f} | quad lam={lam_q:.6f} m={m_q:.3f} | "
              f"naive {est:.3f} [{lo:.3f},{hi:.3f}] (n1={n1}, n2={n2}) | vertices {V} | {terms}")
    sk.write_dat(os.path.join(OUT, "06-newton-random.dat"),
                 ["k", "newton", "m", "quad", "mquad", "naive", "nep", "nem", "vertex"], rows)
    R = np.array(rows)
    dq = np.abs(R[:, 3] - R[:, 1])
    dm = np.abs(R[:, 4] - R[:, 2])
    print(f"random sums: {int(R[:, 8].sum())} of 30 have m = 2; max |quad - 1/t| = {dq.max():.2e}; "
          f"max |m_quad - m| = {dm.max():.3f}")
    for mm in (1, 2):
        sel = R[:, 2] == mm
        bias = R[sel, 5] - R[sel, 1]
        print(f"  naive MC slope minus 1/t, m = {mm}: mean {bias.mean():.3f}, range [{bias.min():.3f}, {bias.max():.3f}]")
    cover = np.sum((R[:, 5] - R[:, 7] <= R[:, 1]) & (R[:, 1] <= R[:, 5] + R[:, 6]))
    print(f"  naive 95% intervals that contain 1/t: {cover} of 30")

    # degenerate examples
    drows = []
    LD = [200.0, 300.0, 450.0, 700.0, 1000.0]
    for i, (name, Vf, (t, mN), (lt, mt)) in enumerate(DEGEN):
        lv = [float(mp.log(Vf(L))) for L in LD]
        lam_q, m_q = fit_exponent(LD, lv)
        Kv = degen_K(name, sk.rng(200 + i), 2_000_000)
        est, lo, hi, n1, n2 = naive_slope(Kv)
        drows.append((i + 1, 1.0 / float(t), mN, lt, mt, lam_q, m_q, est, hi - est, est - lo))
        print(f"D{i + 1} {name:18s} Newton (1/t={float(1 / t):.4f}, m={mN}) truth ({lt:.4f}, {mt}) | "
              f"exact-volume fit lam={lam_q:.5f} m={m_q:.3f} | naive {est:.3f} [{lo:.3f},{hi:.3f}]")
    sk.write_dat(os.path.join(OUT, "06-newton-degenerate.dat"),
                 ["k", "newton", "mnewton", "truth", "mtruth", "quad", "mquad", "naive", "nep", "nem"], drows)
    print(f"two linear units: density of a1 b1 + a2 b2 at 1e-12: {float(rho_D6(mp.mpf('1e-12'))):.6f} "
          f"(at 0 it is (1/4) int log^2|x| dx = 1), so V(eps) ~ 2 sqrt(2 eps)")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
