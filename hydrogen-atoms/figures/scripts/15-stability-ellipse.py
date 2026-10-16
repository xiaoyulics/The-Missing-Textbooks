"""Chapter 15 (stability): two gradient-descent trajectories in the (u, v) plane.

Product model L(u, v) = (uv - y)^2 / 2 with y = 1 and eta = 0.4.  Both runs start on the level set
{u^2 - v^2 = c0} at uv - y = 1e-2, where c0 is chosen so that eta*lambda0(c0) = 2.3 (inside the
catapult ellipse) or 4.3 (outside it).  The first run is recorded for 80 steps; the second until it
leaves the square [-6, 6]^2.
Output: figures/data/15-stability-ellipse-in.dat and 15-stability-ellipse-out.dat, columns t u v psi
with psi = eta (u^2 + v^2) - eta^2 y (uv - y) - 4 (negative inside the ellipse).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def start(el0, eta, y, r0):
    lam0 = el0 / eta
    c = np.sqrt(lam0 ** 2 - 4 * y * y)
    p = y + r0
    u = np.sqrt((c + np.sqrt(c * c + 4 * p * p)) / 2)
    return u, p / u


def record(name, el0, eta=0.4, y=1.0, r0=1e-2, T=80, box=6.0):
    u, v = start(el0, eta, y, r0)
    with open(os.path.join(DATA, name), "w") as f:
        f.write("t u v psi\n")
        for t in range(T + 1):
            if max(abs(u), abs(v)) > box:
                break
            psi = eta * (u * u + v * v) - eta * eta * y * (u * v - y) - 4
            f.write(f"{t} {u:.6f} {v:.6f} {psi:.6f}\n")
            r = u * v - y
            u, v = u - eta * r * v, v - eta * r * u


if __name__ == "__main__":
    record("15-stability-ellipse-in.dat", 2.3)
    record("15-stability-ellipse-out.dat", 4.3, T=200)
