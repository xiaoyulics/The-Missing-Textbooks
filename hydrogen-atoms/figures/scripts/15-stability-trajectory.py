"""Chapter 15 (stability): one catapult of the product model.

Gradient descent on L(u, v) = (uv - y)^2 / 2 with y = 1 and eta = 0.4, started at the point of the
level set {u^2 - v^2 = c0} with uv - y = 1e-6, where c0 is chosen so that the minimizer with
imbalance c0 has eta*lambda0 = 2.3.  For each step t the script records the loss, the sharpness
eta*lambda_max(Hessian at theta_t), and eta*lambda0(c_t) = eta*sqrt(4 y^2 + c_t^2), the sharpness of
the minimizer that gradient flow would reach from theta_t.
Output: figures/data/15-stability-trajectory.dat with columns
  t loss sharp gfs
Deterministic; runs in well under a second.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def main(el0=2.3, eta=0.4, y=1.0, r0=1e-6, T=100):
    lam0 = el0 / eta
    c = np.sqrt(lam0 ** 2 - 4 * y * y)
    p = y + r0
    u = np.sqrt((c + np.sqrt(c * c + 4 * p * p)) / 2)
    v = p / u
    with open(os.path.join(DATA, "15-stability-trajectory.dat"), "w") as f:
        f.write("t loss sharp gfs\n")
        for t in range(T + 1):
            r = u * v - y
            H = np.array([[v * v, 2 * u * v - y], [2 * u * v - y, u * u]])
            lmax = np.linalg.eigvalsh(H)[-1]
            c = u * u - v * v
            f.write(f"{t} {0.5 * r * r:.6e} {eta * lmax:.6f} {eta * np.sqrt(4 * y * y + c * c):.6f}\n")
            u, v = u - eta * r * v, v - eta * r * u


if __name__ == "__main__":
    main()
