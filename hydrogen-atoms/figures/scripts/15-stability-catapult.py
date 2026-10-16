"""Chapter 15 (stability): the catapult curve of the product model.

Gradient descent on L(u, v) = (uv - y)^2 / 2 with y = 1 and step eta = 0.4, started at the point
of the level set {u^2 - v^2 = c0} with uv - y = r0, where c0 is chosen so that the minimizer with
imbalance c0 has sharpness lambda0 = sqrt(4 y^2 + c0^2) equal to a prescribed value.  Each run is
iterated until the loss is below 1e-26 (or for 200000 steps) and the sharpness of the limit,
eta*lambda0(c_inf), is recorded.  Runs with |r| > 1e8 are recorded as divergent (nan).

Output 1: figures/data/15-stability-catapult.dat
  el0      eta*lambda0(c0)
  ra rb rc eta*lambda_inf for r0 = 1e-3, 1e-6, -1e-6 (times y)
  refl     reflection law 4 - el0 (first order), cut at the flattest minimizer 2*eta*y
  second   second-order law 2 - e0 + (s - 2/3) e0^2 with e0 = el0 - 2, s = 2 h^2/(1 - h^2)
Output 2: figures/data/15-stability-second.dat
  h        eta*y
  c2exact  lim (eps_inf + eps0)/eps0^2 measured from the exact map (Richardson extrapolation
           from eps0 = 0.01, 0.005, 0.0025, r0 = 1e-9)
  c2theory s - 2/3
  c2editor (2/3) s   (the cubic-saturation term alone)
Deterministic; runs in a few seconds.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def land(el0, h, r0, T=200000):
    """Scaled model y = 1, step h = eta*y. Returns eta*lambda_inf, or nan if divergent."""
    lam0 = el0 / h
    c = np.sqrt(max(lam0 ** 2 - 4.0, 0.0))
    p = 1.0 + r0
    u = np.sqrt((c + np.sqrt(c * c + 4 * p * p)) / 2)
    v = p / u
    for t in range(T):
        r = u * v - 1.0
        if abs(r) > 1e8 or not np.isfinite(r):
            return np.nan
        if t > 20 and abs(r) < 1e-13:
            break
        u, v = u - h * r * v, v - h * r * u
    c = u * u - v * v
    return h * np.sqrt(4.0 + c * c)


def eps_inf(eps0, h, r0=1e-9):
    return land(2.0 + eps0, h, r0) - 2.0


def main():
    h = 0.4
    s = 2 * h * h / (1 - h * h)
    rows = []
    for el0 in np.arange(0.8, 4.2001, 0.0125):
        el0 = round(el0, 6)
        out = [land(el0, h, r0) for r0 in (1e-3, 1e-6, -1e-6)]
        e0 = el0 - 2.0
        refl = max(4.0 - el0, 2 * h) if el0 >= 2 else el0
        second = max(2.0 - e0 + (s - 2.0 / 3.0) * e0 ** 2, 2 * h) if el0 >= 2 else el0
        if el0 > 4.0:
            refl = second = np.nan
        rows.append([el0] + out + [refl, second])
    with open(os.path.join(DATA, "15-stability-catapult.dat"), "w") as f:
        f.write("el0 ra rb rc refl second\n")
        for row in rows:
            f.write(" ".join("nan" if not np.isfinite(x) else f"{x:.6f}" for x in row) + "\n")

    rows = []
    for h in np.arange(0.05, 0.8001, 0.05):
        s = 2 * h * h / (1 - h * h)
        q = {e: (eps_inf(e, h) + e) / e ** 2 for e in (0.01, 0.005, 0.0025)}
        # quadratic Richardson extrapolation in eps0 (error terms ~ eps0, eps0^2)
        q1 = 2 * q[0.005] - q[0.01]
        q2 = 2 * q[0.0025] - q[0.005]
        c2 = (4 * q2 - q1) / 3
        rows.append([h, c2, s - 2.0 / 3.0, 2.0 * s / 3.0])
    with open(os.path.join(DATA, "15-stability-second.dat"), "w") as f:
        f.write("h c2exact c2theory c2editor\n")
        for row in rows:
            f.write(" ".join(f"{x:.6f}" for x in row) + "\n")


if __name__ == "__main__":
    main()
