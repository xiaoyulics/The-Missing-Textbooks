"""Figure 3 of Chapter 5 (Order and Chaos at Initialization): ReLU networks at He
initialization (sigma_w^2 = 2, sigma_b^2 = 0).

Panel (a): l^2 (1 - c^l) for two orthogonal inputs (c^1 = 0). Theory: iterate the
arc-cosine map C(c) = (sqrt(1 - c^2) + (pi - arccos c) c)/pi; its limit is 9 pi^2/2.
Simulation: width m = 1000, 400 networks per point, through the exact two-input
Markov chain of the chapter's conditional-Gaussianity lemma (as in 05-signalprop-corr.py).

Panel (b): the variance of log(Y_L / Y_1), where Y_l = |D^l u^l|^2 is the squared
norm of the derivative of the post-activations phi(h^l) along a fixed input
direction (u^l the forward Jacobian-vector product, D^l the ReLU mask). The
chapter's finite-width theorem says Y_L/Y_1 is a product of L - 1 independent
factors (2/m) chi^2_B, B ~ Bin(m, 1/2), so Var log(Y_L/Y_1) = (L - 1) v_m with
v_m = Var log F computed exactly by summing over B. For deep linear networks
(sigma_w^2 = 1) the factors are chi^2_m/m and v_m = psi'(m/2). Simulation: full
m x m Gaussian weight matrices, m = 50, 2000 networks, input dimension d = m.

Writes figures/data/05-signalprop-relu.dat and figures/data/05-signalprop-fluct.dat.
Deterministic (seed 20261017); about 40 s on a laptop.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy import special, stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def C(c):
    return (np.sqrt(max(1 - c * c, 0.0)) + (np.pi - np.arccos(c)) * c) / np.pi


def chain_relu(m, L, rng):
    H = rng.standard_normal((m, 2))          # c^1 = 0, q^1 = 1
    cs = np.empty(L)
    for l in range(L):
        cs[l] = H[:, 0] @ H[:, 1] / np.linalg.norm(H[:, 0]) / np.linalg.norm(H[:, 1])
        P = np.maximum(H, 0)
        G = 2 * (P.T @ P) / m
        H = rng.standard_normal((m, 2)) @ np.linalg.cholesky(G).T
    return cs


def var_log_relu(m):
    """Var log F, F = (2/m) chi^2_B, B ~ Bin(m, 1/2), conditioned on B >= 1 (P(B = 0) = 2^-m)."""
    k = np.arange(1, m + 1)
    p = stats.binom.pmf(k, m, 0.5)
    p = p / p.sum()
    mean_k = special.digamma(k / 2)            # E log chi^2_k - log 2
    var_k = special.polygamma(1, k / 2)        # Var log chi^2_k
    mu = p @ mean_k
    return p @ var_k + p @ (mean_k - mu) ** 2


def fluct_sim(m, Ls, rng, nets, act):
    """Samples of log(Y_L/Y_1) for each L in Ls, from full m x m weight matrices.

    The tangent vector u is renormalized after every layer and the logarithms of
    the normalizations are accumulated, which avoids overflow and underflow."""
    Lmax = max(Ls)
    out = np.zeros((nets, len(Ls)))
    sw2 = 2.0 if act == "relu" else 1.0
    for t in range(nets):
        x, u = rng.standard_normal(m), rng.standard_normal(m)
        W = rng.standard_normal((m, m)) * np.sqrt(sw2 / m)
        h, u = W @ x, W @ u
        acc, logY = 0.0, []
        for l in range(1, Lmax + 1):
            D = (h > 0).astype(float) if act == "relu" else np.ones(m)
            a = h * D
            du = D * u
            logY.append(np.log(du @ du) + acc)
            W = rng.standard_normal((m, m)) * np.sqrt(sw2 / m)
            h, u = W @ a, W @ du
            s2 = u @ u
            u = u / np.sqrt(s2)
            acc += np.log(s2)
        logY = np.array(logY)
        out[t] = logY[np.array(Ls) - 1] - logY[0]
    return out


def main():
    rng = np.random.default_rng(20261017)
    # panel (a)
    L = 1000
    th = np.empty(L)
    c = 0.0
    for l in range(L):
        th[l] = c
        c = C(c)
    m, nets, Lsim = 1000, 400, 200
    sims = np.array([chain_relu(m, Lsim, rng) for _ in range(nets)])
    simells = (2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181)
    ells = np.union1d(np.unique(np.round(np.logspace(0.3, 3, 120)).astype(int)), simells)
    rows = []
    for l in ells:
        r = [l, l * l * (1 - th[l - 1])]
        if l in simells:
            e = 1 - sims[:, l - 1]
            r += [l * l * e.mean(), l * l * e.std() / np.sqrt(nets)]
        else:
            r += [np.nan, np.nan]
        rows.append(r)
    np.savetxt(os.path.join(DATA, "05-signalprop-relu.dat"), np.array(rows),
               header="layer theory mc mcse", comments="", fmt="%.6g")
    print("9 pi^2/2 =", 4.5 * np.pi ** 2, " theory at l=1000:", rows[-1][1])

    # panel (b)
    m = 50
    Ls = [2, 11, 26, 51, 76, 101, 126, 151]
    vr, vl = var_log_relu(m), special.polygamma(1, m / 2)
    relu = fluct_sim(m, Ls, rng, 2000, "relu")
    lin = fluct_sim(m, Ls, rng, 2000, "linear")
    rows = []
    for j, Lj in enumerate(Ls):
        rows.append([Lj / m, (Lj - 1) * vr, relu[:, j].var(ddof=1), (Lj - 1) * vl, lin[:, j].var(ddof=1)])
        print(rows[-1])
    np.savetxt(os.path.join(DATA, "05-signalprop-fluct.dat"), np.array(rows),
               header="Lm relutheory relumc lintheory linmc", comments="", fmt="%.6g")
    print("v_m relu", vr, "5/m", 5 / m, " v_m linear", vl, "2/m", 2 / m)


if __name__ == "__main__":
    main()
