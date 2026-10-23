"""04-volume-checks.py -- Chapter 4 (Volume): every constant the chapter quotes, checked numerically.

Prints, with the uniform prior on [-1, 1]^d unless stated:
  1. the constants C of V(eps) ~ C eps^lambda (log 1/eps)^(m-1) for the running models, from the
     closed forms of the chapter and from prop:volume:monomial, with one-dimensional quadrature checks;
  2. the population free energies -log Z(n) and the Abelian constants -log(C Gamma(lambda+1))
     (lem:volume:abelian), with Z(n) by quadrature at n = 10^4 and 10^8;
  3. the constant of prop:volume:monomial with non-constant factors a(w), b(w), against quadrature;
  4. the answers of the chapter's exercises.
Run from the volume root: python figures/scripts/04-volume-checks.py   (about 10 s)
"""
import math

import mpmath as mp
from scipy import integrate, special

mp.mp.dps = 30
G = math.gamma
EG = 0.5772156649015329


def sumpow(ks):
    """K = (1/2) sum_j w_j^(2k_j): lambda = sum 1/(2k_j), V = C eps^lambda for eps <= 1/2."""
    lam = sum(1.0 / (2 * k) for k in ks)
    C = 2.0 ** lam * math.prod(G(1 + 1.0 / (2 * k)) for k in ks) / G(1 + lam)
    return lam, C


def mono_C(k, h, a0=1.0, b0=1.0, box=None):
    """prop:volume:monomial with constant factors a = a0, b = b0 on prod [-r_j, r_j]:
    C = b0 a0^(-lam) / ((m-1)! lam prod_{J} k_j) * prod_{j not in J} int |w|^(h_j - 2 k_j lam)."""
    d = len(k)
    box = box or [1.0] * d
    lams = [(h[j] + 1) / (2 * k[j]) if k[j] > 0 else math.inf for j in range(d)]
    lam = min(lams)
    J = [j for j in range(d) if abs(lams[j] - lam) < 1e-12]
    m = len(J)
    integral = 1.0
    for j in range(d):
        if j not in J:
            e = h[j] - 2 * k[j] * lam
            integral *= 2 * box[j] ** (e + 1) / (e + 1)
    C = b0 * a0 ** (-lam) * integral / (math.factorial(m - 1) * lam * math.prod(k[j] for j in J))
    return lam, m, C


def Z_quad(K1, n, lo=-1.0, hi=1.0):
    """(1/2) int_lo^hi exp(-n K1(w)) dw for a function of one variable."""
    return 0.5 * integrate.quad(lambda w: math.exp(-n * K1(w)), lo, hi, points=[0.0], limit=400,
                                epsabs=0, epsrel=1e-12)[0]


def main():
    print("== 1. volumes of the running models (eps <= 1/2)")
    for name, ks in [("regular plane", (1, 1)), ("quartic cup", (2, 2)), ("quartic", (1, 2)),
                     ("sextic", (1, 3)), ("mixed cup (a^4+b^6)/2", (2, 3)), ("regular, d=3", (1, 1, 1))]:
        lam, C = sumpow(ks)
        print(f"  {name:24s} lambda={lam:.6f}  C={C:.6f}  lambda*C={lam * C:.6f}  C*Gamma(lam+1)={C * G(lam + 1):.6f}"
              f"  -log(C Gamma(lam+1))={-math.log(C * G(lam + 1)):.5f}")
    print("  quartic cup C = Gamma(1/4)^2/(4 sqrt(2 pi)) =", G(0.25) ** 2 / (4 * math.sqrt(2 * math.pi)),
          "; area{u^4+v^4<1} = Gamma(1/4)^2/(2 sqrt pi) =", G(0.25) ** 2 / (2 * math.sqrt(math.pi)))
    print("  quartic cup residue lambda*C = C/2 = (sqrt2/4) K(1/sqrt2):", math.sqrt(2) / 4 * float(mp.ellipk(0.5)))
    # monomials from the proposition, with a(w) = 1/2 (the factor 1/2 in K) and b = 2^-d (uniform prior)
    for name, k in [("product a^2b^2/2", (1, 1)), ("cusp a^2b^4/2", (1, 2)), ("cubic cusp a^2b^6/2", (1, 3)),
                    ("chain L=3", (1, 1, 1)), ("a^2b^4c^6/2", (1, 2, 3)), ("a^2b^4c^4/2", (1, 2, 2))]:
        lam, m, C = mono_C(k, (0,) * len(k), a0=0.5, b0=2.0 ** -len(k))
        print(f"  {name:24s} lambda={lam:.6f} m={m} C={C:.6f}  lambda*C*(m-1)!={lam * C * math.factorial(m - 1):.6f}")
    print("  closed forms: product 1/sqrt2 =", 1 / math.sqrt(2), "; cusp 2*2^(1/4) =", 2 * 2 ** 0.25,
          "; cubic cusp (3/2) 2^(1/6) =", 1.5 * 2 ** (1 / 6), "; chain sqrt2/8 =", math.sqrt(2) / 8,
          "; a^2b^4c^6: (9/2) 2^(1/6) =", 4.5 * 2 ** (1 / 6), "; a^2b^4c^4: 2^(1/4)/2 =", 2 ** 0.25 / 2)
    # exact volumes by the exponential-tail formula versus the asymptotic constants
    for eps in (1e-6, 1e-12, 1e-24):
        s = math.sqrt(2 * eps)
        v3 = 4.5 * s ** (1 / 3) - 4 * s ** 0.5 + 0.5 * s
        print(f"  a^2b^4c^6/2: eps={eps:g}: exact V={v3:.6e}, V/eps^(1/6)={v3 / eps ** (1 / 6):.5f}")
    # a^2 b^4 c^4 / 2: P(E_a + 2E_b + 2E_c > l), l = log(1/s): density of 2E_b+2E_c is Gamma(2, 1/2)
    for eps in (1e-6, 1e-12, 1e-24, 1e-48):
        s = math.sqrt(2 * eps); l = math.log(1 / s)
        tail = lambda x: math.exp(-x / 2) * (1 + x / 2) if x > 0 else 1.0   # P(Gamma(2,1/2) > x)
        val = integrate.quad(lambda e: math.exp(-e) * tail(l - e), 0, l, limit=200)[0] + math.exp(-l)
        print(f"  a^2b^4c^4/2: eps={eps:g}: exact V={val:.6e}, V/(eps^(1/4) log(1/eps))={val / (eps ** 0.25 * math.log(1 / eps)):.5f}")

    print("\n== 2. population free energies (lem:volume:abelian)")
    for n in (1e4, 1e8):
        zc = Z_quad(lambda w: 0.5 * w ** 4, n) ** 2
        zr = Z_quad(lambda w: 0.5 * w ** 2, n) ** 2
        zq = Z_quad(lambda w: 0.5 * w ** 2, n) * Z_quad(lambda w: 0.5 * w ** 4, n)
        zp = 0.5 * math.sqrt(math.pi / (2 * n)) * (math.log(n) + EG + math.log(2))
        print(f"  n={n:g}: cup  -log Z = {-math.log(zc):.5f} = 0.5 log n + {(-math.log(zc) - 0.5 * math.log(n)):.5f};"
              f"  regular plane -log Z - log n = {-math.log(zr) - math.log(n):.5f};"
              f"  quartic -log Z - 0.75 log n = {-math.log(zq) - 0.75 * math.log(n):.5f};"
              f"  product -log Z - 0.5 log n + log log n = {-math.log(zp) - 0.5 * math.log(n) + math.log(math.log(n)):.5f}"
              f"  ratio to leading term {1 + (EG + math.log(2)) / math.log(n):.4f}")
    print("  limits: cup -log(sqrt2 Gamma(5/4)^2) =", -math.log(math.sqrt(2) * G(1.25) ** 2),
          "; regular -log(pi/2) =", -math.log(math.pi / 2), "; product 0.5 log(8/pi) =", 0.5 * math.log(8 / math.pi))
    print("  product: n at which the ratio is 1.01: log10 n =", (EG + math.log(2)) / 0.01 / math.log(10))

    print("\n== 3. prop:volume:monomial with non-constant a, b")
    a = lambda u1, u2: 1.0 + 0.5 * u1 + u2 ** 2
    b = lambda u1, u2: 2.0 + math.sin(u1) + u2
    # K = a u1^2 u2^4, phi = b |u1| on [-1,1]^2: lam_1 = 1, lam_2 = 1/4 -> J = {2}, m = 1, k_2 = 2
    Cpred = (1 / (0.25 * 2)) * integrate.quad(lambda x: b(x, 0) * a(x, 0) ** (-0.25) * abs(x) ** 0.5, -1, 1,
                                                points=[0.0])[0]

    def V(eps):
        def inner(u1):
            if u1 == 0.0:
                return 0.0
            g = lambda u2: a(u1, u2) * u1 ** 2 * u2 ** 4
            lo, hi = 0.0, 1.0
            if g(1.0) < eps:
                r = 1.0
            else:
                for _ in range(90):
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if g(mid) < eps else (lo, mid)
                r = lo
            return abs(u1) * integrate.quad(lambda u2: b(u1, u2), -r, r, epsabs=1e-16)[0]
        return integrate.quad(inner, -1, 1, points=[0.0], limit=400, epsabs=1e-16, epsrel=1e-11)[0]
    for eps in (1e-4, 1e-8, 1e-12, 1e-16):
        print(f"  eps={eps:g}: V/eps^(1/4) = {V(eps) / eps ** 0.25:.6f}   (predicted C = {Cpred:.6f})")

    print("\n== 4. exercise answers")
    lam, C = sumpow((1, 1, 1))
    print(f"  ball d=3: C = (pi/6) 2^(3/2) = {math.pi / 6 * 2 ** 1.5:.5f} (formula {C:.5f}); 10^8 draws keep 100 at eps = {(1e-6 / C) ** (2 / 3):.3g}")
    print(f"  (a^2+b^2)^2/2: C = pi sqrt2/4 = {math.pi * math.sqrt(2) / 4:.5f}; bracket [C_cup/sqrt2, C_cup] = "
          f"[{sumpow((2, 2))[1] / math.sqrt(2):.5f}, {sumpow((2, 2))[1]:.5f}]")
    lam, C = sumpow((2, 3))
    print(f"  mixed cup: lambda = {lam:.6f} = 5/12, C = {C:.5f}, residue lambda*C = {lam * C:.5f}")
    print(f"  cusp with prior |a|/2: C = (4/3) 2^(1/4) = {4 / 3 * 2 ** 0.25:.5f}; lambda C = 2^(1/4)/3 = {2 ** 0.25 / 3:.5f};"
          f" proposition: {mono_C((1, 2), (1, 0), a0=0.5, b0=0.5)}")
    print(f"  cusp with prior |b|/2: proposition {mono_C((1, 2), (0, 1), a0=0.5, b0=0.5)} (product model: 1/sqrt2 = {1 / math.sqrt(2):.5f})")
    # Gaussian prior on the quartic cup and on the product model
    Cg = G(0.25) ** 2 / (2 * math.sqrt(math.pi)) * math.sqrt(2) / (2 * math.pi)
    print(f"  Gaussian prior N(0,1)^2: cup C = {Cg:.5f}, constant -log(C Gamma(3/2)) = {-math.log(Cg * G(1.5)):.5f};"
          f" product C = sqrt2/pi = {math.sqrt(2) / math.pi:.5f}")
    # Gaussian check of the cup by quadrature: V(eps) = P(a^4+b^4 < 2 eps) for a,b ~ N(0,1)
    for eps in (1e-4, 1e-8):
        r = (2 * eps) ** 0.25
        f = lambda x: 2 * special.ndtr((max(2 * eps - x ** 4, 0.0)) ** 0.25) - 1.0
        val = integrate.quad(lambda x: f(x) * math.exp(-x * x / 2) / math.sqrt(2 * math.pi), -r, r, limit=200)[0]
        print(f"    Gaussian cup eps={eps:g}: V/eps^(1/2) = {val / eps ** 0.5:.5f}")
    print(f"  K = (c^2 + a^2 b^2)/2 on [-1,1]^3: C' = pi/4 = {math.pi / 4:.5f}")
    for n in (1e4, 1e8, 1e16):
        zc = math.sqrt(math.pi / (2 * n)) * special.erf(math.sqrt(n / 2))
        zab = 0.5 * math.sqrt(math.pi / (2 * n)) * (math.log(n) + EG + math.log(2))
        print(f"    n={n:g}: Z(n) n / log n = {zc * zab * n / math.log(n):.5f}")
    for eps in (1e-4, 1e-8, 1e-16):
        f = lambda c: 0.5 * math.sqrt(2 * eps - c * c) * (1 + math.log(1 / math.sqrt(2 * eps - c * c)))
        val = integrate.quad(f, -math.sqrt(2 * eps), math.sqrt(2 * eps), limit=200)[0]
        print(f"    eps={eps:g}: V/(eps log(1/eps)) = {val / (eps * math.log(1 / eps)):.5f}")
    print("\n== 5. where the theory stops")
    # share of V(eps) inside the square |a|,|b| < delta: product model versus K = min(a^2, b^2)/2
    delta = 0.1
    for eps in (1e-4, 1e-8, 1e-16, 1e-100):
        s = math.sqrt(2 * eps); l = math.log(1 / s); t0 = math.log(1 / delta)
        share_prod = (1 + l - 2 * t0) / (1 + l)
        r = s
        share_min = (8 * r * delta - 4 * r * r) / (8 * r - 4 * r * r)
        print(f"  eps={eps:g}: share of |a|,|b|<{delta} in V: product {share_prod:.4f}; min(a^2,b^2)/2 {share_min:.4f}")
    # Monte Carlo check of the product share at eps = 1e-8 (10^7 draws, seed 431)
    import numpy as np
    gen = np.random.default_rng(431)
    w = gen.uniform(-1, 1, size=(10_000_000, 2))
    inside = 0.5 * (w[:, 0] * w[:, 1]) ** 2 < 1e-8
    near = inside & (np.abs(w[:, 0]) < delta) & (np.abs(w[:, 1]) < delta)
    print(f"  Monte Carlo (10^7 draws, seed 431): product share at eps=1e-8: {near.sum() / inside.sum():.4f} of {inside.sum()} draws")
    # the corner example: K = (ab - 1)^2 / 2 on [-1,1]^2 meets the true set only at the corners (1,1), (-1,-1)
    for eps in (1e-3, 1e-5, 1e-7):
        r = math.sqrt(2 * eps)
        # {|ab - 1| < r} within the box: needs ab > 1 - r, i.e. a, b of the same sign with |b| > (1-r)/|a|
        area_q = integrate.quad(lambda x: max(0.0, 1.0 - (1 - r) / x), 1 - r, 1.0, limit=200, epsabs=1e-16)[0]
        V = 2 * area_q / 4
        print(f"  corner example eps={eps:g}: V = {V:.6e}, V/eps = {V / eps:.5f}  (limit 1/2: lambda = 1, m = 1)")
    d = 10
    for dd, e in ((10, 1e-2), (10, 1e-4), (50, 1e-2)):
        V = (2 * e) ** (dd / 2) * math.pi ** (dd / 2) / (2 ** dd * G(dd / 2 + 1))
        print(f"  regular model d={dd}, eps={e:g}: V = {V:.3e}, draws for 10%: {100 / V:.2e}")


if __name__ == "__main__":
    main()
