"""Chapter 11 (Temperature): the population mean energy of four running models.

For the population tempered posterior, proportional to exp(-n beta K(w)) phi(w) with phi uniform on
[-1, 1]^2, the mean energy satisfies beta E^beta[n K] = g(s), a function of s = n beta alone:
g(s) = s E_s[K], with E_s the Gibbs measure proportional to exp(-s K) phi. This script computes g
exactly (no sampling):
  * sums of monomials K = (a^2 + b^(2k))/2 (regular plane k = 1, quartic k = 2, sextic k = 3): the
    law of t = s b^(2k)/2 is a Gamma(1/(2k)) law truncated to [0, s/2], so
    s E[b^(2k)/2] = alpha P(alpha + 1, s/2) / P(alpha, s/2), alpha = 1/(2k), with P the
    regularized lower incomplete gamma function;
  * the product model K = (ab)^2/2 through the density log(1/|c|)/2 of c = ab, by adaptive
    quadrature in u = sqrt(s) c, compared with the closed form 1/2 - 1/(log s + gamma + log 2).
It also checks thermodynamic integration, F(n) = int_0^1 E^beta[nK] d beta, for the quartic model.
Writes figures/data/11-temperature-energy.dat (columns s, regular, quartic, sextic, product,
productformula). Runs in a few seconds.
"""
import math
import os
import sys

import numpy as np
from scipy import integrate, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sltkit as sk

GAMMA_E = 0.5772156649015329
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "11-temperature-energy.dat")


def g_monomial(s, k):
    """s E[|b|^(2k)/2] under the density proportional to exp(-s b^(2k)/2) on [-1, 1]."""
    a = 1.0 / (2 * k)
    return a * special.gammainc(a + 1, s / 2) / special.gammainc(a, s / 2)


def g_product(s):
    """s E[c^2/2] under the density proportional to exp(-s c^2/2) log(1/|c|) on [-1, 1]."""
    lam = 0.5 * math.log(s)
    T = math.sqrt(s)
    cuts = [0.0] + [x for x in (1.0, 8.0) if x < T] + [min(T, 40.0)]
    num = den = 0.0
    for u, v in zip(cuts[:-1], cuts[1:]):
        w = lambda x: math.exp(-0.5 * x * x) * (lam - math.log(x))
        num += integrate.quad(lambda x: 0.5 * x * x * w(x), u, v, limit=200, epsabs=0, epsrel=1e-12)[0]
        den += integrate.quad(w, u, v, limit=200, epsabs=0, epsrel=1e-12)[0]
    return num / den


def g_product_formula(s):
    return 0.5 - 1.0 / (math.log(s) + GAMMA_E + math.log(2.0))


def main():
    rows = []
    for e in np.linspace(-1, 12, 79):
        s = 10.0 ** e
        reg = 2 * g_monomial(s, 1)
        quart = g_monomial(s, 1) + g_monomial(s, 2)
        sext = g_monomial(s, 1) + g_monomial(s, 3)
        prod = g_product(s)
        form = g_product_formula(s) if s > 3 else float("nan")
        rows.append((s, reg, quart, sext, prod, form))
    sk.write_dat(OUT, ["s", "regular", "quartic", "sextic", "product", "productformula"], rows)
    print("numbers quoted in the text (g(s) = s E_s[K]):")
    for s in (10.0, 30.0, 100.0):
        print(f"  quartic g({s:g}) = {g_monomial(s, 1) + g_monomial(s, 2):.6f}")
    for s in (1e2, 1e3, 1e4, 1e6, 1e12):
        print(f"  product g({s:g}) = {g_product(s):.6f}   formula {g_product_formula(s):.6f}")
    print("  small-s slopes g(s)/s at s = 1e-3: regular", 2 * g_monomial(1e-3, 1) / 1e-3,
          " quartic", (g_monomial(1e-3, 1) + g_monomial(1e-3, 2)) / 1e-3,
          " (prior means 1/3, 4/15)")
    # population free energy of the quartic model and thermodynamic integration
    cQ = 0.5 * math.log(2 / math.pi) - 0.25 * math.log(2) - math.log(special.gamma(1.25))
    for n in (1e2, 1e4, 1e6):
        Ia = 0.5 * math.sqrt(2 * math.pi / n) * math.erf(math.sqrt(n / 2))
        Ib = integrate.quad(lambda b: 0.5 * math.exp(-n * b ** 4 / 2), -1, 1, points=[0], limit=200)[0]
        Fbar = -math.log(Ia * Ib)
        # F(n) = int_0^1 g(n beta)/beta d beta = int_{-inf}^{log n} g(e^x) dx
        gq = lambda x: g_monomial(math.exp(x), 1) + g_monomial(math.exp(x), 2)
        TI = integrate.quad(gq, -40, math.log(n), limit=400)[0]
        print(f"  quartic n={n:g}: F(n) = {Fbar:.10f}, TI = {TI:.10f}, (3/4) log n + cQ = {0.75*math.log(n)+cQ:.10f}")
    print(f"  cQ = {cQ:.6f}")
    print(f"  V(eps) of the quartic model: C eps^(3/4) with C = {0.5 * 2 ** 0.75 * 0.5 * special.beta(0.25, 1.5):.6f}")


if __name__ == "__main__":
    main()
