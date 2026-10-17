"""Figure 15.1 of Optimizers in the Wild (Chapter 15, key `bias`): three norms, three classifiers.

A two-dimensional separable dataset of six examples. For the l1, l2 and l-infinity norms the
script computes the maximum-margin classifier
    gamma = max_{||u|| <= 1} min_i <u, z_i>,      z_i = y_i x_i,
by linear programming (l1, l-infinity) and quadratic programming (l2), checks the duality
    gamma = min { ||c||_* : c in conv{z_i} }
by solving the dual problem as well, and writes what the figure draws:
    figures/data/15-bias-geometry-points.dat   examples x_i, labels, signed examples z_i
    figures/data/15-bias-geometry-hull.dat     the convex hull of the z_i (closed polygon)
    figures/data/15-bias-geometry-balls.dat    boundaries of the three dual balls of radius gamma
    figures/data/15-bias-geometry-lines.dat    separating lines, directions and supporting lines
numpy and scipy only; deterministic; runs in about a second.
"""
import os
import numpy as np
from scipy.optimize import linprog, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# signed examples z_i = y_i x_i; examples 1, 2, 4 are positive, 3, 5, 6 negative (x = -z)
Z = np.array([[0.54, 0.03], [0.85, 0.48], [0.79, 0.44], [0.93, 0.25], [0.90, 0.18], [0.42, 0.30]])
y = np.array([1, 1, -1, 1, -1, -1], float)
X = y[:, None] * Z
n, d = Z.shape


def mm_linf(Z):
    c = np.r_[np.zeros(d), -1.0]
    r = linprog(c, A_ub=np.hstack([-Z, np.ones((n, 1))]), b_ub=np.zeros(n),
                bounds=[(-1, 1)] * d + [(None, None)], method="highs")
    return -r.fun, r.x[:d]


def mm_l1(Z):
    c = np.r_[np.zeros(2 * d), -1.0]
    A = np.vstack([np.hstack([-Z, Z, np.ones((n, 1))]), np.r_[np.ones(2 * d), 0.0][None, :]])
    r = linprog(c, A_ub=A, b_ub=np.r_[np.zeros(n), 1.0], bounds=[(0, None)] * (2 * d) + [(None, None)],
                method="highs")
    return -r.fun, r.x[:d] - r.x[d:2 * d]


def mm_l2(Z):
    # hard-margin SVM through the origin; the support set is then solved exactly
    r = minimize(lambda w: 0.5 * w @ w, mm_linf(Z)[1] / mm_linf(Z)[0], jac=lambda w: w, method="SLSQP",
                 constraints=[{"type": "ineq", "fun": lambda w: Z @ w - 1, "jac": lambda w: Z}],
                 options={"ftol": 1e-15, "maxiter": 1000})
    S = np.where(Z @ r.x < 1 + 1e-6)[0]
    a = np.linalg.solve(Z[S] @ Z[S].T, np.ones(len(S)))
    w = Z[S].T @ a
    assert np.all(a > 0) and np.min(Z @ w) > 1 - 1e-12
    return 1 / np.linalg.norm(w), w / np.linalg.norm(w)


def dual(dn):
    """min of the dual norm over conv{z_i}: a small convex program, solved from many starts."""
    best = None
    for k in range(40):
        p0 = np.random.default_rng(k).dirichlet(np.ones(n))
        r = minimize(lambda p: dn(Z.T @ p), p0, method="SLSQP", bounds=[(0, 1)] * n,
                     constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}],
                     options={"ftol": 1e-14, "maxiter": 500})
        if best is None or r.fun < best.fun:
            best = r
    return best.fun, Z.T @ best.x


g2, u2 = mm_l2(Z)
gi, ui = mm_linf(Z)
g1, u1 = mm_l1(Z)
d2, c2 = dual(lambda v: np.linalg.norm(v))
di, ci = dual(lambda v: np.abs(v).sum())      # dual of l-infinity is l1
d1, c1 = dual(lambda v: np.abs(v).max())      # dual of l1 is l-infinity
for name, g, dv, u, c in [("l2", g2, d2, u2, c2), ("linf", gi, di, ui, ci), ("l1", g1, d1, u1, c1)]:
    sv = np.where(Z @ u < g * 1 + 1e-7)[0] + 1
    print(f"{name:5s} gamma={g:.6f}  dual={dv:.6f}  u={np.round(u, 5)}  closest point c={np.round(c, 4)}  "
          f"support vectors {sv.tolist()}")
    assert abs(g - dv) < 1e-6

with open(os.path.join(DATA, "15-bias-geometry-points.dat"), "w") as f:
    f.write("x1 x2 y z1 z2 sv2 svinf sv1\n")
    for i in range(n):
        s2 = int(Z[i] @ u2 < g2 + 1e-7)
        si = int(Z[i] @ ui < gi + 1e-7)
        s1 = int(Z[i] @ u1 < g1 + 1e-7)
        f.write(f"{X[i,0]:.4f} {X[i,1]:.4f} {int(y[i])} {Z[i,0]:.4f} {Z[i,1]:.4f} {s2} {si} {s1}\n")

# convex hull (counterclockwise) by the monotone chain
P = sorted(map(tuple, Z))
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
lower, upper = [], []
for p in P:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
        lower.pop()
    lower.append(p)
for p in reversed(P):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
        upper.pop()
    upper.append(p)
hull = lower[:-1] + upper[:-1]
with open(os.path.join(DATA, "15-bias-geometry-hull.dat"), "w") as f:
    f.write("z1 z2\n")
    for p in hull + [hull[0]]:
        f.write(f"{p[0]:.4f} {p[1]:.4f}\n")

# dual balls of radius gamma: l1 diamond (for the l-infinity margin), l2 circle, l-infinity square (l1 margin)
phi = np.linspace(0, 2 * np.pi, 121)
circ = np.c_[np.cos(phi), np.sin(phi)]
diam = circ / np.abs(circ).sum(1, keepdims=True)
squa = circ / np.abs(circ).max(1, keepdims=True)
with open(os.path.join(DATA, "15-bias-geometry-balls.dat"), "w") as f:
    f.write("dx dy cx cy sx sy\n")
    for k in range(len(phi)):
        f.write(f"{gi*diam[k,0]:.5f} {gi*diam[k,1]:.5f} {g2*circ[k,0]:.5f} {g2*circ[k,1]:.5f} "
                f"{g1*squa[k,0]:.5f} {g1*squa[k,1]:.5f}\n")

# lines: for each norm, the separating line through 0 (normal u), and in the z-plane the
# supporting line <u, c> = gamma; two points each, plus the unit direction u/|u|_2 for arrows
def seg(u, level, half):
    u = np.asarray(u, float); t = np.array([-u[1], u[0]]) / np.linalg.norm(u)
    base = level * u / (u @ u)
    return base - half * t, base + half * t
cols, rows = [], [[], []]
for name, g, u in [("l2", g2, u2), ("linf", gi, ui), ("l1", g1, u1)]:
    a, b = seg(u, 0.0, 1.25)
    c, e = seg(u, g, 0.75)
    v = u / np.linalg.norm(u)
    cols += [f"sep{name}x", f"sep{name}y", f"sup{name}x", f"sup{name}y", f"dir{name}x", f"dir{name}y"]
    rows[0] += [a[0], a[1], c[0], c[1], 0.0, 0.0]
    rows[1] += [b[0], b[1], e[0], e[1], 0.3 * v[0], 0.3 * v[1]]
with open(os.path.join(DATA, "15-bias-geometry-lines.dat"), "w") as f:
    f.write(" ".join(cols) + "\n")
    for r in rows:
        f.write(" ".join(f"{v:.4f}" for v in r) + "\n")
print("written to", os.path.normpath(DATA))
