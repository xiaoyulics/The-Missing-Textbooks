"""Search and descent in single runs of online SGD (Chapter 7, figure fig:singleindex:trajectories).

Full d-dimensional spherical online SGD (no reduction) for the matched link g = phi = h_3:
x_t ~ N(0, I_d) fresh at each step, y_t = h_3(<w_star, x_t>), gradient of the correlation loss
-y h_3(<w, x>) projected on the tangent space, then renormalization:
    w_{t+1} = (w_t - eta G_t) / |w_t - eta G_t|,  G_t = -y_t h_3'(<w_t, x_t>) (x_t - <w_t, x_t> w_t).
d = 128, eta = 0.02 d^{-3/2}, warm start r_0 = d^{-1/2} (w_0 = r_0 w_star + sqrt(1 - r_0^2) e with a
uniformly random unit e orthogonal to w_star), 8 independent runs, seed fixed below.
Theory: the population flow dr/dt = eta * 3 r^2 (1 - r^2) (time measured in steps).
Recorded: the overlap r_t and the population square loss on the sphere,
    (1/2) E (y - h_3(<w_t, x>))^2 = 1 - r_t^3,
at 200 equally spaced times up to 2.5 times the predicted escape time to r = 1/2.
Output: figures/data/07-singleindex-traj.dat with columns tau (= t / d^2), ode, r1..r8, lode, l1..l8.
Runtime: about 15 seconds on a laptop.
"""
import os
import numpy as np
from scipy.integrate import quad, solve_ivp

D, RUNS, C, SEED = 128, 8, 0.02, 7
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "07-singleindex-traj.dat")


def main():
    rng = np.random.default_rng(SEED)
    eta = C * D ** -1.5
    r0 = D ** -0.5
    t_half = quad(lambda r: 1.0 / ((1 - r * r) * 3 * r * r), r0, 0.5)[0] / eta
    horizon = int(2.5 * t_half)
    grid = np.linspace(0, horizon, 200).astype(int)
    # w_star = e_1; random orthogonal complement direction for each run
    W = np.zeros((RUNS, D))
    e = rng.standard_normal((RUNS, D - 1))
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    W[:, 0] = r0
    W[:, 1:] = np.sqrt(1 - r0 ** 2) * e
    rec = np.zeros((grid.size, RUNS))
    gi = 0
    block = 2000
    t = 0
    while gi < grid.size:
        X = rng.standard_normal((block, RUNS, D))
        for b in range(block):
            while gi < grid.size and grid[gi] == t:
                rec[gi] = W[:, 0]
                gi += 1
            if gi >= grid.size:
                break
            x = X[b]
            u = x[:, 0]
            v = np.einsum("ij,ij->i", W, x)
            # y * phi'(v) with h_3(u) = (u^3 - 3u)/sqrt 6 and h_3'(v) = sqrt 3 h_2(v) = sqrt 3 (v^2 - 1)/sqrt 2
            F = 0.5 * (u ** 3 - 3 * u) * (v * v - 1)
            G = -F[:, None] * (x - v[:, None] * W)
            W = W - eta * G
            W /= np.linalg.norm(W, axis=1, keepdims=True)
            t += 1
    sol = solve_ivp(lambda s, r: eta * 3 * r * r * (1 - r * r), (0, horizon), [r0],
                    t_eval=grid, rtol=1e-10, atol=1e-12)
    ode = sol.y[0]
    with open(OUT, "w") as f:
        cols = ["tau", "ode"] + ["r%d" % (i + 1) for i in range(RUNS)] + ["lode"] + \
               ["l%d" % (i + 1) for i in range(RUNS)]
        f.write(" ".join(cols) + "\n")
        for j, s in enumerate(grid):
            vals = [s / D ** 2, ode[j]] + list(rec[j]) + [1 - ode[j] ** 3] + list(1 - rec[j] ** 3)
            f.write(" ".join("%.6g" % x for x in vals) + "\n")
    esc = [(np.argmax(rec[:, i] >= 0.5), rec[-1, i]) for i in range(RUNS)]
    print("predicted escape tau = %.3f" % (t_half / D ** 2))
    print("runs: first grid tau with r >= 1/2, final r:",
          [("%.3f" % (grid[j] / D ** 2) if rec[j, i] >= 0.5 else "none", "%.3f" % rf)
           for i, (j, rf) in enumerate(esc)])


if __name__ == "__main__":
    main()
