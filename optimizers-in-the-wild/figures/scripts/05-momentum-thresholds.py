"""Chapter 5 (momentum): stability thresholds of momentum methods on one quadratic mode.

On a mode with curvature h, each method is a linear recursion in (x_t, x_{t-1}) or (x_t, m_{t-1}).
The theory (Proposition prop:momentum:regions) says that the recursion is stable if and only if
eta*h lies below
    heavy ball (PyTorch form)   2(1+b)
    Nesterov (PyTorch form)     2(1+b)/(1+2b)
    EMA heavy ball (Adam form)  2(1+b)/(1-b)
    EMA Nesterov (Muon form)    2(1+b)/((1-b)(1+2b)).
The simulation runs each method as code on the mode L(x) = x^2/2 (h = 1, exact gradients) for a
grid of 401 step sizes within 2% of the predicted threshold, renormalizing the state at every step
and averaging the logarithmic growth of its norm over the second half of T = 8000 steps (an
estimate of the logarithm of the spectral radius). The measured threshold is the smallest step size
on the grid whose growth rate is nonnegative.
Output: figures/data/05-momentum-thresholds.dat     (curves)  beta hb nest ema nema
        figures/data/05-momentum-thresholds-sim.dat (points)  beta hb nest ema nema
Deterministic; runs in about ten seconds.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def theory(beta):
    return {"hb": 2 * (1 + beta), "nest": 2 * (1 + beta) / (1 + 2 * beta),
            "ema": 2 * (1 + beta) / (1 - beta), "nema": 2 * (1 + beta) / ((1 - beta) * (1 + 2 * beta))}


def step(method, beta, eta, x, xprev, m):
    """One step of each method on L(x) = x^2/2 (gradient g = x), vectorized over eta."""
    g = x
    if method == "hb":        # x+ = x - eta g + beta (x - x_prev)
        return x - eta * g + beta * (x - xprev), x, m
    if method == "nest":      # b = beta b + g; x+ = x - eta (g + beta b)        (PyTorch, nesterov=True)
        m = beta * m + g
        return x - eta * (g + beta * m), x, m
    if method == "ema":       # m = beta m + (1-beta) g; x+ = x - eta m          (Adam's first moment)
        m = beta * m + (1 - beta) * g
        return x - eta * m, x, m
    if method == "nema":      # m = beta m + (1-beta) g; x+ = x - eta (beta m + (1-beta) g)   (Muon)
        m = beta * m + (1 - beta) * g
        return x - eta * (beta * m + (1 - beta) * g), x, m
    raise ValueError(method)


def measured_threshold(method, beta, T=8000, n=401, width=0.02):
    pred = theory(beta)[method]
    etas = pred * np.linspace(1 - width, 1 + width, n)
    x = np.ones(n); xprev = np.ones(n); m = np.zeros(n)
    growth = np.zeros(n)
    for t in range(T):
        x, xprev, m = step(method, beta, etas, x, xprev, m)
        s = np.sqrt(x * x + xprev * xprev + m * m)
        if t >= T // 2:
            growth += np.log(s)
        x /= s; xprev /= s; m /= s
    rate = growth / (T - T // 2)
    unstable = np.nonzero(rate >= 0)[0]
    return etas[unstable[0]] if unstable.size else np.nan


def main():
    betas = np.concatenate([np.linspace(0, 0.9, 91), np.linspace(0.905, 0.99, 18)])
    with open(os.path.join(DATA, "05-momentum-thresholds.dat"), "w") as f:
        f.write("beta hb nest ema nema\n")
        for b in betas:
            th = theory(b)
            f.write(f"{b:.4f} {th['hb']:.6f} {th['nest']:.6f} {th['ema']:.6f} {th['nema']:.6f}\n")
    sim_betas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99]
    worst = 0.0
    with open(os.path.join(DATA, "05-momentum-thresholds-sim.dat"), "w") as f:
        f.write("beta hb nest ema nema\n")
        for b in sim_betas:
            vals = {k: measured_threshold(k, b) for k in ["hb", "nest", "ema", "nema"]}
            th = theory(b)
            worst = max(worst, max(abs(vals[k] / th[k] - 1) for k in vals))
            f.write(f"{b:.4f} {vals['hb']:.6f} {vals['nest']:.6f} {vals['ema']:.6f} {vals['nema']:.6f}\n")
    print(f"largest relative deviation of the measured thresholds from the theory: {worst:.2e}")


if __name__ == "__main__":
    main()
