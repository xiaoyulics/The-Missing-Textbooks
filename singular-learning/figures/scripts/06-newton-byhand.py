"""Chapter 6 (Newton Polygons): the numbers of the hand computations and of the exercises.

Prior uniform on [-1, 1]^d throughout, so for functions even in each variable V(eps) = Leb{(0, 1]^d : K < eps}.
  1. The emblem K = (a^4 + b^6)/2 of the model y = a^2 x1 + b^3 x2: exact volume, the logarithmic-coordinate
     sandwich, the zeta function on the normal fan, the population free energy.
  2. The vertex example K = a^6 + a^2 b^2 + b^6: the sandwich R(eps/3) in {K < eps} in R(eps), the optimal face of
     the linear program, the leading constant 1/6, against quadrature.
  3. The cast through Newton's polygon (t, 1/t, m) and the degenerate examples; written to
     figures/data/06-newton-cast.dat (columns: index, t, lambda_Newton, m_Newton).
  4. Exercise answers: a^6 + a^2 b^4 + b^6; a^2 b^2 + a^4 + b^8; the zeta function of a^2 + b^4; three variables
     (linear programs); a^4 + c a^2 b^2 + b^4; a monomial prior; Monte Carlo checks where stated.
Run from the volume root:  python figures/scripts/06-newton-byhand.py   (about twenty seconds)
"""
import importlib.util
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from scipy import integrate, optimize, special  # noqa: E402

import sltkit as sk  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "newton_random", os.path.join(os.path.dirname(os.path.abspath(__file__)), "06-newton-random.py"))
nr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nr)
OUT = os.path.join("figures", "data")
G = special.gamma


def lp_newton(S, d):
    """min sum(u) s.t. <v, u> >= 1 for v in S, u >= 0; returns the value and the dimension of the optimal face."""
    A = -np.array(S, float)
    res = optimize.linprog(np.ones(d), A_ub=A, b_ub=-np.ones(len(S)), bounds=[(0, None)] * d, method="highs")
    val = res.fun
    # optimal face: maximize and minimize each coordinate over {u in P : sum u = val}
    A_eq, b_eq = np.ones((1, d)), [val]
    pts = []
    for j in range(d):
        for sgn in (1, -1):
            c = np.zeros(d)
            c[j] = sgn
            r = optimize.linprog(c, A_ub=A, b_ub=-np.ones(len(S)), A_eq=A_eq, b_eq=b_eq,
                                 bounds=[(0, None)] * d, method="highs")
            pts.append(r.x)
    P = np.array(pts) - pts[0]
    dim = np.linalg.matrix_rank(P, tol=1e-9)
    return val, dim


def main():
    print("== 1. the emblem K = (a^4 + b^6)/2")
    C6 = integrate.quad(lambda v: (1 - v ** 6) ** 0.25, 0, 1)[0]
    C6g = G(1 / 6) * G(5 / 4) / (6 * G(17 / 12))
    print(f"  C6 = int_0^1 (1 - v^6)^(1/4) dv = {C6:.6f} (Gamma form {C6g:.6f}); V(eps) = C6 (2 eps)^(5/12) = "
          f"{C6 * 2 ** (5 / 12):.5f} eps^(5/12) for eps <= 1/2")
    print(f"  sandwich in logarithmic coordinates: eps^(5/12) <= V <= 2^(5/12) eps^(5/12) = {2 ** (5 / 12):.5f} eps^(5/12)")
    I4 = integrate.quad(lambda x: (1 + x ** 4) ** (-5 / 12), 0, 1)[0]
    I6 = integrate.quad(lambda x: (1 + x ** 6) ** (-5 / 12), 0, 1)[0]
    print(f"  zeta on the fan: zeta(z) = 2^-z [2 int (1+x^4)^z + 3 int (1+x^6)^z] / (12 z + 5); "
          f"residue at -5/12 = {2 ** (5 / 12) * (2 * I4 + 3 * I6) / 12:.6f}, C = (12/5) residue = "
          f"{2 ** (5 / 12) * (2 * I4 + 3 * I6) / 5:.5f}")
    const = math.log(2 ** (5 / 12) * G(5 / 4) * G(7 / 6))
    print(f"  population free energy: Fbar(n) = (5/12) log n - {const:.4f} (+ exponentially small); "
          f"check -log(C Gamma(17/12)) = {-math.log(C6 * 2 ** (5 / 12) * G(17 / 12)):.4f}")
    for n in (1e2, 1e4, 1e6):
        Z = integrate.quad(lambda a: math.exp(-n * a ** 4 / 2), 0, 1, points=[n ** -0.25])[0] * \
            integrate.quad(lambda b: math.exp(-n * b ** 6 / 2), 0, 1, points=[n ** (-1 / 6)])[0]
        print(f"    n = {n:.0e}: Fbar = {-math.log(Z):.4f}, (5/12) log n - {const:.4f} = {5 / 12 * math.log(n) - const:.4f}, "
              f"regular plane log n - log(pi/2) = {math.log(n) - math.log(math.pi / 2):.4f}")

    print("== 2. the vertex example K = a^6 + a^2 b^2 + b^6")
    Sc = {(6, 0): 1.0, (2, 2): 1.0, (0, 6): 1.0}
    t, m, C, V = nr.newton(Sc)
    print(f"  Newton: vertices {V}, t = {t}, m = {m}, leading constant C = {C:.6f}")
    print(f"  LP: {lp_newton([(6, 0), (2, 2), (0, 6)], 2)} (value, dimension of the optimal face)")
    R = lambda dl: dl ** 0.5 * (1 + math.log(1 / dl) / 6)  # noqa: E731
    for e in (1e-4, 1e-8, 1e-16, 1e-50):
        L = math.log(1 / e)
        Vq = math.exp(nr.logV(Sc, L))
        print(f"  eps = {e:.0e}: R(eps/3) = {R(e / 3):.4e} <= V = {Vq:.4e} <= R(eps) = {R(e):.4e}; "
              f"V / (sqrt(eps) log(1/eps)) = {Vq / (e ** 0.5 * L):.4f}")
    # Monte Carlo check of V at moderate eps
    g = sk.rng(66)
    a, b = g.random(4_000_000), g.random(4_000_000)
    Kv = a ** 6 + a * a * b * b + b ** 6
    for e in (1e-2, 1e-4):
        k = int(np.count_nonzero(Kv < e))
        lo, hi = sk.clopper_pearson(k, len(Kv))
        print(f"  Monte Carlo V({e:.0e}) = {k / len(Kv):.5f} [{lo:.5f}, {hi:.5f}], quadrature "
              f"{math.exp(nr.logV(Sc, math.log(1 / e))):.5f}")

    print("== 3. the cast through Newton's polygon")
    cast = [("regular plane (a^2+b^2)/2", {(2, 0): .5, (0, 2): .5}),
            ("product a^2b^2/2", {(2, 2): .5}),
            ("cusp a^2b^4/2", {(2, 4): .5}),
            ("cubic cusp a^2b^6/2", {(2, 6): .5}),
            ("quartic (a^2+b^4)/2", {(2, 0): .5, (0, 4): .5}),
            ("sextic (a^2+b^6)/2", {(2, 0): .5, (0, 6): .5}),
            ("quartic cup (a^4+b^4)/2", {(4, 0): .5, (0, 4): .5}),
            ("mixed cup (a^4+b^6)/2", {(4, 0): .5, (0, 6): .5}),
            ("Part III emblem a^6+a^2b^2+b^6", {(6, 0): 1., (2, 2): 1., (0, 6): 1.}),
            ("Part IV emblem a^4b^2+a^2b^4", {(4, 2): 1., (2, 4): 1.})]
    rows = []
    for i, (name, Sc) in enumerate(cast):
        t, m, C, V = nr.newton(Sc)
        rows.append((i + 1, float(t), 1 / float(t), m))
        print(f"  {name:32s} vertices {str(V):26s} t = {str(t):5s} 1/t = {str(1 / t):5s} m = {m}  C = {C:.4f}")
    sk.write_dat(os.path.join(OUT, "06-newton-cast.dat"), ["k", "t", "lam", "m"], rows)
    for name, S in [("(a-b)^2", [(2, 0), (1, 1), (0, 2)]), ("(a-b^2)^2", [(2, 0), (1, 2), (0, 4)]),
                    ("(a+b)^2+b^4", [(2, 0), (1, 1), (0, 2), (0, 4)]), ("(a^2-b^2)^2", [(4, 0), (2, 2), (0, 4)]),
                    ("(a^2-b^3)^2", [(4, 0), (2, 3), (0, 6)])]:
        print(f"  degenerate {name:12s}: LP value 1/t and optimal-face dimension {lp_newton(S, 2)}")
    print(f"  (a-b)^2 exact: V = sqrt(eps) - eps/4, e.g. V(1e-6) = {1e-3 - 1e-6 / 4:.6e}")
    # two linear units and the product model at a generic truth
    print("  two linear units (a1 b1 + a2 b2)^2 in 4 variables: LP value and face dimension",
          lp_newton([(2, 2, 0, 0), (1, 1, 1, 1), (0, 0, 2, 2)], 4), "-> Newton (1/t, m) = (1, 4 - 1)")
    print("  product model at the true point (1/2, 1/2) of ab = 1/4, local coordinates: K = (al/2 + be/2 + al be)^2/2:",
          lp_newton([(2, 0), (1, 1), (0, 2), (2, 1), (1, 2), (2, 2)], 2))

    print("== 4. exercise answers")
    for name, Sc in [("a^6 + a^2 b^4 + b^6", {(6, 0): 1., (2, 4): 1., (0, 6): 1.}),
                     ("a^6 + b^6", {(6, 0): 1., (0, 6): 1.}),
                     ("a^2 b^2 + a^4 + b^8", {(2, 2): 1., (4, 0): 1., (0, 8): 1.}),
                     ("a^2 b^2", {(2, 2): 1.})]:
        t, m, C, V = nr.newton(Sc)
        lv = [nr.logV(Sc, L) for L in (250., 500., 1000., 2000., 4000.)]
        lq, mq = nr.fit_exponent([250., 500., 1000., 2000., 4000.], lv)
        print(f"  {name:22s}: vertices {V}, t = {t}, m = {m}, C = {C:.6f}; quadrature fit lam = {lq:.6f}, m = {mq:.3f}")
    I2 = integrate.quad(lambda x: (1 + x ** 2) ** -0.75, 0, 1)[0]
    I4b = integrate.quad(lambda x: (1 + x ** 4) ** -0.75, 0, 1)[0]
    C4 = integrate.quad(lambda v: math.sqrt(1 - v ** 4), 0, 1)[0]
    print(f"  a^2 + b^4: zeta(z) = [int(1+x^2)^z + 2 int(1+x^4)^z]/(4z+3); C = (I2 + 2 I4)/3 = {(I2 + 2 * I4b) / 3:.6f}, "
          f"exact int_0^1 sqrt(1-v^4) dv = {C4:.6f}")
    for name, S in [("a^2b^2+b^2c^2+c^2a^2", [(2, 2, 0), (0, 2, 2), (2, 0, 2)]),
                    ("a^2b^2+c^2", [(2, 2, 0), (0, 0, 2)]), ("a^2b^2c^2", [(2, 2, 2)]),
                    ("a^2+b^2+c^2", [(2, 0, 0), (0, 2, 0), (0, 0, 2)])]:
        val, dim = lp_newton(S, 3)
        print(f"  three variables {name:22s}: 1/t = {val:.4f}, optimal face dimension {dim} -> m = {dim + 1}")
    g = sk.rng(67)
    u = g.random((3, 4_000_000))
    Kv = (u[0] * u[1]) ** 2 + (u[1] * u[2]) ** 2 + (u[2] * u[0]) ** 2
    n1, n2 = int(np.count_nonzero(Kv < 1e-3)), int(np.count_nonzero(Kv < 1e-5))
    print(f"  Monte Carlo slope of a^2b^2+b^2c^2+c^2a^2 between 1e-3 and 1e-5: {math.log(n1 / n2) / math.log(100):.3f} "
          f"(counts {n1}, {n2})")
    for c in (1.0, 0.0, -1.0, -1.9, -2.0):
        k = lambda th: math.cos(th) ** 4 + c * math.cos(th) ** 2 * math.sin(th) ** 2 + math.sin(th) ** 4  # noqa: E731
        if c > -2:
            Cc = integrate.quad(lambda th: k(th) ** -0.5, 0, 2 * math.pi, limit=200)[0] / 8
            print(f"  a^4 + c a^2 b^2 + b^4, c = {c:5.2f}: V = C eps^(1/2) with C = (1/8) int k^(-1/2) = {Cc:.5f}")
        else:
            print(f"  a^4 - 2 a^2 b^2 + b^4 = (a^2 - b^2)^2: int k^(-1/2) diverges (k = cos^2 2 theta): m = 2")
    print("  prior proportional to |a| for a^2 b^2: zeta = prod of int |a|^(2z+1), int |b|^(2z): poles -1, -1/2 -> (1/2, 1)")


if __name__ == "__main__":
    main()
