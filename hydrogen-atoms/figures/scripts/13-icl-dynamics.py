"""Figure 2 of Chapter 13 (Learning in Context): gradient descent on the
parameters of one linear self-attention layer from a small, balanced
initialization, against the logistic solution of the gradient flow.

d = 8, N = 20, sigma^2 = 0.5, isotropic prompts as in 13-icl-risk.py.
Initialization: W11 = a0 I, u2 = sqrt(d) a0 (balanced), gamma0 = sqrt(d) a0^2,
for gamma0 in {1e-2, 1e-4}, plus independent N(0, 1e-8) perturbations of all
2(d+1)^2 parameters. Plain SGD with step eta = 0.005 on batches of 4096 fresh
prompts; time t = eta * step. Recorded: gammahat = tr(Gamma)/d.
Theory: the logistic solution of d gamma/dt = 4 sqrt(d) gamma (1 - gamma/g*),
g* = N/(N+d+1+s2), namely gamma(t) = g*/(1 + (g*/gamma0 - 1) exp(-4 sqrt(d) t)).

Deterministic (seeded); about 20 seconds on a laptop CPU.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import torch

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
D, N, S2 = 8, 20, 0.5
ETA, STEPS, B = 0.005, 400, 4096
GSTAR = N / (N + D + 1 + S2)
SEED = 1017


def logistic(t, g0):
    return GSTAR / (1 + (GSTAR / g0 - 1) * np.exp(-4 * np.sqrt(D) * t))


def prompts(g):
    w = torch.randn(B, D, generator=g)
    X = torch.randn(B, N, D, generator=g)
    y = torch.einsum("bnd,bd->bn", X, w) + S2 ** 0.5 * torch.randn(B, N, generator=g)
    xq = torch.randn(B, D, generator=g)
    yq = (xq * w).sum(1) + S2 ** 0.5 * torch.randn(B, generator=g)
    return X, y, xq, yq


def lsa(WKQ, WPV, X, y, xq):
    Zc = torch.cat([X, y[:, :, None]], 2)
    G = torch.einsum("bni,bnj->bij", Zc, Zc) / N
    zq = torch.cat([xq, torch.zeros(B, 1)], 1)
    return torch.einsum("i,bij,jk,bk->b", WPV[D], G, WKQ, zq)


def run(g0, g):
    a0 = np.sqrt(g0 / np.sqrt(D))
    WKQ = 1e-4 * torch.randn(D + 1, D + 1, generator=g)
    WPV = 1e-4 * torch.randn(D + 1, D + 1, generator=g)
    WKQ[:D, :D] += a0 * torch.eye(D)
    WPV[D, D] += np.sqrt(D) * a0
    WKQ.requires_grad_(); WPV.requires_grad_()
    out = []
    for k in range(STEPS + 1):
        if k % 8 == 0:
            Gam = WPV[D, D] * WKQ[:D, :D].T + torch.outer(WKQ[D, :D], WPV[D, :D])
            out.append((ETA * k, torch.trace(Gam).item() / D))
        X, y, xq, yq = prompts(g)
        loss = ((yq - lsa(WKQ, WPV, X, y, xq)) ** 2).mean()
        gK, gP = torch.autograd.grad(loss, [WKQ, WPV])
        with torch.no_grad():
            WKQ -= ETA * gK; WPV -= ETA * gP
    return out


def main():
    g = torch.Generator().manual_seed(SEED)
    sims = {g0: run(g0, g) for g0 in (1e-2, 1e-4)}
    with open(os.path.join(DATA, "13-icl-dynamics.dat"), "w") as f:
        f.write("t simA theoryA simB theoryB\n")
        for (t, a), (_, b) in zip(sims[1e-2], sims[1e-4]):
            f.write(f"{t:.4f} {a:.6f} {logistic(t, 1e-2):.6f} {b:.6f} {logistic(t, 1e-4):.6f}\n")
    for g0 in sims:
        dev = max(abs(v - logistic(t, g0)) for t, v in sims[g0])
        print(g0, "max |sim - theory| =", dev)


if __name__ == "__main__":
    main()
