"""Hidden progress of one monomial neuron on a sparse parity (Chapter 9, figure fig:parity:neuron).

Setting: x uniform on {-1,1}^d, y = x_1 x_2 x_3 (k = 3), neuron phi(z) = z^3 on the unit sphere,
correlation loss -y <w,x>^3, spherical online SGD with one fresh sample per step:
    w_{t+1} = (w_t - eta G_t)/|w_t - eta G_t|,  G_t = -y_t k <w_t,x_t>^{k-1} (x_t - <w_t,x_t> w_t).
d = 256, eta = 0.05 d^{-3/2}, symmetric start w_0 = d^{-1/2} (1,...,1), 8 runs, seed below.
Recorded: the effective overlap rho = (Psi(w)/c_k)^{1/k} with Psi(w) = k! w_1 w_2 w_3 and
c_k = k!/k^{k/2}, and the normalized population loss 1 - rho^k = 1 - Psi(w)/max Psi.
Theory: the population flow d rho/dt = eta kappa rho^{k-1}(1 - rho^2), kappa = k c_k
(time in steps), started at rho_0 = sqrt(k/d).
Output: figures/data/09-parity-neuron.dat with columns tau (= t/d^2), ode, r1..r8, lode, l1..l8.
Runtime: about 25 seconds.
"""
import os
import math
import numpy as np
from scipy.integrate import solve_ivp, quad

D, K, RUNS, C, SEED = 256, 3, 8, 0.05, 11
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "09-parity-neuron.dat")


def main():
    rng = np.random.default_rng(SEED)
    ck = math.factorial(K) / K ** (K / 2)
    kappa = K * ck
    eta = C * D ** -1.5
    rho0 = math.sqrt(K / D)
    t_half = quad(lambda s: 1 / (kappa * s ** (K - 1) * (1 - s * s)), rho0, 0.5)[0] / eta
    horizon = int(2.0 * t_half)
    grid = np.linspace(0, horizon, 200).astype(int)
    W = np.full((RUNS, D), D ** -0.5)
    rec = np.zeros((grid.size, RUNS))
    gi = 0
    for t in range(horizon + 1):
        while gi < grid.size and grid[gi] == t:
            psi = math.factorial(K) * W[:, :K].prod(1)
            rec[gi] = np.cbrt(np.maximum(psi, 0) / ck)
            gi += 1
        X = rng.integers(0, 2, (RUNS, D)) * 2.0 - 1
        y = X[:, :K].prod(1)
        z = (W * X).sum(1)
        F = y * K * z ** (K - 1)
        W = W + eta * F[:, None] * (X - z[:, None] * W)
        W /= np.linalg.norm(W, axis=1, keepdims=True)
    sol = solve_ivp(lambda t, r: eta * kappa * r ** (K - 1) * (1 - r * r), (0, horizon), [rho0],
                    t_eval=grid.astype(float), rtol=1e-10, atol=1e-12)
    ode = sol.y[0]
    tau = grid / D ** 2
    with open(OUT, "w") as f:
        f.write("tau ode " + " ".join(f"r{i+1}" for i in range(RUNS)) + " lode "
                + " ".join(f"l{i+1}" for i in range(RUNS)) + "\n")
        for j in range(grid.size):
            vals = [tau[j], ode[j], *rec[j], 1 - ode[j] ** K, *(1 - rec[j] ** K)]
            f.write(" ".join(f"{v:.5f}" for v in vals) + "\n")
    print("t_half/d^2 =", t_half / D ** 2)
    first = [(grid[np.argmax(rec[:, i] >= 0.5)] / t_half) if (rec[:, i] >= 0.5).any() else np.nan
             for i in range(RUNS)]
    print("escape / t_half:", np.round(first, 3))


if __name__ == "__main__":
    main()
