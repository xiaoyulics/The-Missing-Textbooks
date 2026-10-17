"""Chapter 5 (momentum): normalized SGD with momentum on an isotropic quadratic with Gaussian noise.

Loss L(theta) = h ||theta||^2/2 on R^d with h = 1 and d = 1000; stochastic gradient
g = h theta + zeta with zeta ~ N(0, (sigma^2/d) I), sigma = 1 (so E||zeta||^2 = sigma^2);
    m_t = beta m_{t-1} + (1-beta) g_t,   theta_{t+1} = theta_t - eta m_t/||m_t||_2.
Theory (heuristic, exact as d -> infinity): ||m_t|| concentrates at a constant c, the iteration is
then heavy ball with step eta (1-beta)/c on every coordinate, and Theorem thm:momentum:noise with the
self-consistency condition c^2 = E||m||^2 gives
    c = [eta (1-beta) h + sqrt(eta^2 (1-beta)^2 h^2 + 16 (1-beta^2) sigma^2)] / (4 (1+beta)),
    E||grad L||^2 = h eta c (1+beta) / (2 (1-beta)).
Also written: the stationary part of the bound of Theorem thm:momentum:normalized (L = h, chi = 1).
Simulation: 4 independent runs from theta = 0, burn-in, then the root mean square of ||grad L||_2
over the remaining steps (pooled), seed 0.
Output: figures/data/05-momentum-isotropic.dat      x rmsA rmsB boundA     (x = 1 - beta; A: eta = 0.01, B: eta = 0.1)
        figures/data/05-momentum-isotropic-sim.dat  x rmsA rmsB
Runs in about half a minute.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
H, SIGMA, D = 1.0, 1.0, 1000
ETAS = (0.01, 0.1)


def rms_theory(eta, beta):
    c = (eta * (1 - beta) * H + np.sqrt(eta ** 2 * (1 - beta) ** 2 * H ** 2 + 16 * (1 - beta ** 2) * SIGMA ** 2)) / (4 * (1 + beta))
    return np.sqrt(H * eta * c * (1 + beta) / (2 * (1 - beta)))


def bound(eta, beta):
    return H * eta / 2 + 2 * beta * H * eta / (1 - beta) + 2 * SIGMA * np.sqrt((1 - beta) / (1 + beta))


def simulate(eta, beta, rng, reps=4, burn=3000, T=8000):
    th = np.zeros((reps, D)); m = np.zeros((reps, D))
    acc = 0.0
    for t in range(burn + T):
        g = H * th + rng.standard_normal((reps, D)) * (SIGMA / np.sqrt(D))
        m = beta * m + (1 - beta) * g
        th = th - eta * m / np.linalg.norm(m, axis=1, keepdims=True)
        if t >= burn:
            acc += np.mean(np.sum((H * th) ** 2, axis=1))
    return np.sqrt(acc / T)


def main():
    rng = np.random.default_rng(0)
    xs = np.geomspace(1.0, 0.004, 60)
    with open(os.path.join(DATA, "05-momentum-isotropic.dat"), "w") as f:
        f.write("x rmsA rmsB boundA\n")
        for x in xs:
            b = 1 - x
            f.write(f"{x:.6f} {rms_theory(ETAS[0], b):.6e} {rms_theory(ETAS[1], b):.6e} {bound(ETAS[0], b):.6e}\n")
    worst = 0.0
    with open(os.path.join(DATA, "05-momentum-isotropic-sim.dat"), "w") as f:
        f.write("x rmsA rmsB\n")
        for b in [0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99]:
            vals = [simulate(eta, b, rng) for eta in ETAS]
            worst = max(worst, max(abs(v / rms_theory(eta, b) - 1) for v, eta in zip(vals, ETAS)))
            f.write(f"{1 - b:.4f} {vals[0]:.6e} {vals[1]:.6e}\n")
    print(f"largest relative deviation of the simulation from the d -> infinity theory: {worst:.4f}")


if __name__ == "__main__":
    main()
