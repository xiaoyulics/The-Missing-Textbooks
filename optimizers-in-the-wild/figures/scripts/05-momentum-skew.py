"""Chapter 5 (momentum): normalized SGD with momentum on a one-dimensional quadratic with skewed noise.

Loss L(theta) = h theta^2/2 with h = 1; stochastic gradient g = h theta + zeta, where zeta = 3b with
probability 1/4 and -b with probability 3/4 (mean 0, standard deviation sigma = sqrt(3) b = 1).
In one dimension every normalized step is a sign step:
    m_t = beta m_{t-1} + (1-beta) g_t,   theta_{t+1} = theta_t - eta sign(m_t),   eta = 1e-4.
Theory (Section "Reading the theory"):
  bias   the stationary mean of grad L tends, as eta -> 0, to -med(zbar), the negative median of the
         stationary moving average zbar = (1-beta) sum_k beta^k zeta_k of the noise; computed exactly
         from the fixed-point equation of the distribution function of sum_k beta^k B_k, B_k ~ Bern(1/4)
         (for beta = 0 the value is b; for 0 < beta < 1/2 the law is a Cantor-type measure and the
         curve is not drawn);
  sd     heuristic (diffusion approximation with a Gaussian zbar), valid for beta near 1:
         Var(grad L) = h eta sqrt(2 pi k2) D / 4, k2 = sigma^2 (1-beta)/(1+beta),
         D = 1 + (4/pi) sum_{k>=1} arcsin(beta^k);
  bound  the stationary part of Theorem thm:momentum:normalized with L = h, chi = 1:
         L eta/2 + 2 beta L eta/(1-beta) + 2 sigma sqrt((1-beta)/(1+beta)).
Simulation: 2000 chains (4000 for beta >= 0.95) from theta = 0, burn-in 25000 steps, then the mean
(with a standard error from 10 time blocks) and the standard deviation of grad L = h theta over the
next 40000 steps, pooled over chains and time; seed 0.
Output: figures/data/05-momentum-skew.dat      x bias cf sd bound   (x = 1 - beta; nan where not drawn)
        figures/data/05-momentum-skew-sim.dat  x mean se sd
Runs in about half a minute (numpy and scipy).
"""
import os
import numpy as np
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
P = 0.25                      # probability of the large value
SIGMA = 1.0
B = SIGMA / np.sqrt(3.0)      # zeta in {3b, -b}
H = 1.0
ETA = 1e-4


def median_S(beta, n=8001, tol=1e-10):
    """Median of S = sum_k beta^k B_k, B_k i.i.d. Bernoulli(P): iterate the fixed-point equation
    F(x) = (1-P) F(x/beta) + P F((x-1)/beta) of its distribution function on a grid, starting from the
    Gaussian with the same mean and variance (relative accuracy of the median better than 1e-3)."""
    mu = P / (1 - beta)
    sd = np.sqrt(P * (1 - P) / (1 - beta ** 2))
    lo, hi = max(0.0, mu - 14 * sd), min(1 / (1 - beta), mu + 14 * sd)
    x = np.linspace(lo, hi, n)
    F = ndtr((x - mu) / sd)
    for _ in range(int(np.ceil(np.log(1e-12) / np.log(beta))) + 10):
        Fn = (1 - P) * np.interp(x / beta, x, F, left=0.0, right=1.0) + P * np.interp((x - 1) / beta, x, F, left=0.0, right=1.0)
        if np.max(np.abs(Fn - F)) < tol:
            F = Fn
            break
        F = Fn
    i = np.searchsorted(F, 0.5)
    return x[i - 1] + (0.5 - F[i - 1]) * (x[i] - x[i - 1]) / (F[i] - F[i - 1])


def bias_exact(beta):
    if beta == 0.0:
        return B
    return -(-B + 4 * B * (1 - beta) * median_S(beta))   # zbar = -b + 4b(1-beta) S


def bias_cf(beta):
    """Cornish-Fisher: -med(zbar) ~ k3/(6 k2), with E zeta^3 = gamma1 sigma^3, gamma1 = 2/sqrt(3)."""
    m3 = P * (3 * B) ** 3 - (1 - P) * B ** 3
    return m3 * (1 - beta) * (1 + beta) / (6 * SIGMA ** 2 * (1 + beta + beta ** 2))


def sd_heuristic(beta, eta=ETA):
    k = np.arange(1, int(60 / (1 - beta)) + 10)
    D = 1 + (4 / np.pi) * np.sum(np.arcsin(beta ** k))
    k2 = SIGMA ** 2 * (1 - beta) / (1 + beta)
    return np.sqrt(H * eta * np.sqrt(2 * np.pi * k2) * D / 4)


def bound(beta, eta=ETA):
    L = H
    return L * eta / 2 + 2 * beta * L * eta / (1 - beta) + 2 * SIGMA * np.sqrt((1 - beta) / (1 + beta))


def simulate(beta, rng, chains, burn=25000, T=40000, nblocks=10):
    th = np.zeros(chains); m = np.zeros(chains)
    s1 = s2 = 0.0
    blocks, acc = [], 0.0
    for t in range(burn + T):
        z = np.where(rng.random(chains) < P, 3 * B, -B)
        m = beta * m + (1 - beta) * (H * th + z)
        th = th - ETA * np.sign(m)
        if t >= burn:
            gr = H * th
            g1 = gr.mean()
            s1 += g1; s2 += (gr * gr).mean(); acc += g1
            if (t - burn + 1) % (T // nblocks) == 0:
                blocks.append(acc / (T // nblocks)); acc = 0.0
    mean = s1 / T
    se = np.std(blocks) / np.sqrt(len(blocks))
    return mean, se, np.sqrt(max(s2 / T - mean ** 2, 0.0))


def main():
    rng = np.random.default_rng(0)
    xs = np.geomspace(1.0, 0.003, 52)
    with open(os.path.join(DATA, "05-momentum-skew.dat"), "w") as f:
        f.write("x bias cf sd bound\n")
        for x in xs:
            beta = 1.0 - x
            if np.isclose(x, 1.0):
                bias = f"{bias_exact(0.0):.6e}"
            elif x <= 0.5 + 1e-12:
                bias = f"{bias_exact(beta):.6e}"
            else:
                bias = "nan"        # 0 < beta < 1/2: Cantor-type law of zbar, not drawn
            sd = f"{sd_heuristic(beta):.6e}" if x <= 0.25 else "nan"
            f.write(f"{x:.6f} {bias} {bias_cf(beta):.6e} {sd} {bound(beta):.6e}\n")
    with open(os.path.join(DATA, "05-momentum-skew-sim.dat"), "w") as f:
        f.write("x mean se sd\n")
        for beta in [0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995]:
            mean, se, sd = simulate(beta, rng, chains=4000 if beta >= 0.95 else 2000)
            f.write(f"{1 - beta:.4f} {mean:.6e} {se:.2e} {sd:.6e}\n")
            ref = bias_exact(beta)
            print(f"beta={beta}: mean {mean:.6f} +- {se:.6f} (exact eta->0 bias {ref:.6f}), sd {sd:.6f}"
                  + (f" (heuristic {sd_heuristic(beta):.6f})" if beta >= 0.75 else ""))


if __name__ == "__main__":
    main()
