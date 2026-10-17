"""Chapter 1 (steepest): the geometry of a step, computed for the schematic of Figure 1.

Left panel. The unit balls of the l1, l2 and l-infinity norms in the plane, one gradient g whose
negative points at the angle PHI, and for each norm the minimizer lmo(g) of <g,u> over the ball
(the steepest unit direction) together with the supporting line {u : <g,u> = -|g|_*}, which touches
the ball at lmo(g). With |g|_2 = 1 the supporting line of a norm lies at Euclidean distance |g|_*
from the origin, so the three lines are ordered by the three dual norms.
  01-steepest-geometry-circle.dat   x y              the l2 unit circle
  01-steepest-geometry-lmo.dat      x y dual         lmo(g) and |g|_*, one row per norm: l1, l2, l-inf
                                    ax ay du dv      one end of the supporting line and its extent
Right panel. Normalized steepest descent theta <- theta + eta lmo(H theta) in the three norms on the
quadratic L(theta) = theta^T H theta / 2, H = Q diag(1, 0.12) Q^T with Q the rotation by 30 degrees,
from theta_0 = (-2.4, 1.3) with eta = 0.2, for 40 steps; and level sets of L.
  01-steepest-geometry-traj.dat     t x2 y2 xinf yinf x1 y1
  01-steepest-geometry-levels.dat   phi xa ya xb yb xc yc xd yd   level sets L = 0.02, 0.15, 0.5, 1.3448
Deterministic; runs in well under a second.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PHI = np.deg2rad(28.0)          # direction of -g (the title-page emblem draws g at this angle)


def lmo(g, norm):
    """A minimizer of <g,u> over the unit ball of the l_norm norm (norm in {1, 2, inf})."""
    if not np.any(g):
        return np.zeros_like(g)
    if norm == 2:
        return -g / np.linalg.norm(g)
    if norm == np.inf:
        return -np.sign(g)
    i = int(np.argmax(np.abs(g)))
    u = np.zeros_like(g)
    u[i] = -np.sign(g[i])
    return u


def dual(g, norm):
    return {1: np.abs(g).max(), 2: np.linalg.norm(g), np.inf: np.abs(g).sum()}[norm]


def write(name, header, rows):
    path = os.path.join(DATA, name)
    with open(path, "w") as f:
        f.write(" ".join(header) + "\n")
        for r in rows:
            f.write(" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in r) + "\n")
    print("wrote", os.path.normpath(path))


def main():
    os.makedirs(DATA, exist_ok=True)
    # ---------------- left panel: balls, lmo, supporting lines ----------------
    ang = np.linspace(0, 2 * np.pi, 145)
    write("01-steepest-geometry-circle.dat", ["x", "y"],
          [(float(np.cos(a)), float(np.sin(a))) for a in ang])
    n = np.array([np.cos(PHI), np.sin(PHI)])     # -g, normalized: |g|_2 = 1
    g = -n
    tangent = np.array([-n[1], n[0]])
    rows = []
    for label, norm in (("1", 1), ("2", 2), ("inf", np.inf)):
        u = lmo(g, norm)
        dn = dual(g, norm)
        assert abs(float(g @ u) + dn) < 1e-12        # <g, lmo(g)> = -|g|_*
        foot = dn * n                                  # closest point of the supporting line
        a, b = foot - 0.62 * tangent, foot + 0.62 * tangent
        rows.append((float(u[0]), float(u[1]), float(dn),
                     float(a[0]), float(a[1]), float(b[0] - a[0]), float(b[1] - a[1])))
        print(f"norm {label}: lmo = {u.round(4)}, dual norm = {dn:.4f}")
    write("01-steepest-geometry-lmo.dat", ["x", "y", "dual", "ax", "ay", "du", "dv"], rows)

    # ---------------- right panel: trajectories on a 2-D quadratic ----------------
    rot = np.deg2rad(30.0)
    Q = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    H = Q @ np.diag([1.0, 0.12]) @ Q.T
    th0 = np.array([-2.4, 1.3])
    eta, T = 0.2, 40
    paths = {}
    for norm in (2, np.inf, 1):
        th = th0.copy()
        path = [th.copy()]
        for _ in range(T):
            th = th + eta * lmo(H @ th, norm)
            path.append(th.copy())
        paths[norm] = np.array(path)
        loss = 0.5 * np.einsum("ij,jk,ik->i", paths[norm], H, paths[norm])
        print(f"norm {norm}: loss after 10, 20, 40 steps = {loss[10]:.4f}, {loss[20]:.5f}, {loss[40]:.5f}")
    write("01-steepest-geometry-traj.dat", ["t", "x2", "y2", "xinf", "yinf", "x1", "y1"],
          [(t, float(paths[2][t, 0]), float(paths[2][t, 1]), float(paths[np.inf][t, 0]),
            float(paths[np.inf][t, 1]), float(paths[1][t, 0]), float(paths[1][t, 1]))
           for t in range(T + 1)])
    # level sets {theta : theta^T H theta / 2 = c}: theta = Q diag(1/sqrt(h)) (cos, sin) sqrt(2c)
    levels = [0.02, 0.15, 0.5, float(0.5 * th0 @ H @ th0)]
    print("levels", [round(c, 4) for c in levels])
    phis = np.linspace(0, 2 * np.pi, 121)
    rows = []
    for p in phis:
        r = [float(p)]
        for c in levels:
            z = np.sqrt(2 * c) * np.array([np.cos(p) / np.sqrt(1.0), np.sin(p) / np.sqrt(0.12)])
            x = Q @ z
            r += [float(x[0]), float(x[1])]
        rows.append(r)
    write("01-steepest-geometry-levels.dat", ["phi", "xa", "ya", "xb", "yb", "xc", "yc", "xd", "yd"], rows)


if __name__ == "__main__":
    main()
