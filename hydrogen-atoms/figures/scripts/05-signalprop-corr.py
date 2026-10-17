"""Figure 2 of Chapter 5 (Order and Chaos at Initialization): the correlation c^l
between the pre-activations of two inputs, layer by layer, in tanh networks with
sigma_b^2 = 0.05 in the three phases, infinite-width theory versus finite width.

Theory: iterate the correlation map C(c) at the fixed length qbar (the inputs are
placed at length qbar), with Gaussian expectations by the trapezoid rule.

Simulation: networks of width m = 1000 with Gaussian weights and biases, started
from layer-1 pre-activations with covariance qbar [[1, c1], [c1, 1]], c1 = 0.6. On
two inputs, a layer of the network is exactly, in law, the map
    H -> G^{1/2}-coloured Gaussian rows,  G = sigma_b^2 + sigma_w^2 phi(H)^T phi(H)/m
(Lemma "exact conditional Gaussianity" of the chapter), so we simulate this Markov
chain on m x 2 matrices instead of multiplying m x m weight matrices; the two
agree in law (checked against full weight matrices in development).
The empirical correlation <h, h'>/(|h| |h'|) is averaged over 50 networks.

Writes figures/data/05-signalprop-corr.dat. Deterministic (seed 20261017); about
5 s on a laptop.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
SB2 = 0.05
C1 = 0.6
M = 1000
NETS = 50
L = 50
Z1 = np.linspace(-9, 9, 3601)
W1 = np.exp(-Z1 ** 2 / 2); W1 /= W1.sum()
Z2 = np.linspace(-8, 8, 481)
W2 = np.exp(-Z2 ** 2 / 2); W2 /= W2.sum()
U, V = np.meshgrid(Z2, Z2, indexing="ij")
WW = np.outer(W2, W2)


def qbar(sw2):
    return optimize.brentq(lambda q: SB2 + sw2 * (W1 @ np.tanh(np.sqrt(q) * Z1) ** 2) - q,
                           1e-12, SB2 + sw2 + 1, xtol=1e-14)


def critical():
    def chi1(sw2):
        q = qbar(sw2)
        return sw2 * (W1 @ np.cosh(np.sqrt(q) * Z1) ** -4)
    return optimize.brentq(lambda s: chi1(s) - 1, 1.01, 10, xtol=1e-12)


def C(c, sw2, q):
    s = np.sqrt(q)
    return (SB2 + sw2 * np.sum(WW * np.tanh(s * U) * np.tanh(s * (c * U + np.sqrt(1 - c * c) * V)))) / q


def simulate(sw2, q, rng):
    out = np.zeros((NETS, L))
    K1 = q * np.array([[1, C1], [C1, 1]])
    R1 = np.linalg.cholesky(K1)
    for t in range(NETS):
        H = rng.standard_normal((M, 2)) @ R1.T
        for l in range(L):
            out[t, l] = H[:, 0] @ H[:, 1] / np.linalg.norm(H[:, 0]) / np.linalg.norm(H[:, 1])
            P = np.tanh(H)
            G = SB2 + sw2 * (P.T @ P) / M
            H = rng.standard_normal((M, 2)) @ np.linalg.cholesky(G).T
    return out.mean(0), out.std(0)


def main():
    rng = np.random.default_rng(20261017)
    swc = critical()
    phases = [("ord", 1.2), ("crit", swc), ("cha", 4.0)]
    cols = [np.arange(1, L + 1)]
    names = ["layer"]
    for name, sw2 in phases:
        q = qbar(sw2)
        th = [C1]
        for _ in range(L - 1):
            th.append(min(C(th[-1], sw2, q), 1.0))
        mc, sd = simulate(sw2, q, rng)
        cols += [np.array(th), mc, sd]
        names += [name + "th", name + "mc", name + "sd"]
        print(name, sw2, "qbar", q, "c^L theory", th[-1], "sim", mc[-1], "max|diff|",
              np.max(np.abs(np.array(th) - mc)))
    np.savetxt(os.path.join(DATA, "05-signalprop-corr.dat"), np.column_stack(cols),
               header=" ".join(names), comments="", fmt="%.6f")


if __name__ == "__main__":
    main()
