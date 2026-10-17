"""Two features in one dimension (Chapter 11, superposition).

Writes
  figures/data/11-superposition-loss.dat   loss versus sparsity S at importance ratio r=0.5:
      exact losses of the dedicated and antipodal configurations, the numerically minimized loss,
      and the exact loss of parameters found by Adam on sampled data (three seeds);
  figures/data/11-superposition-phase.dat  the numerically computed phase boundary S_c(r);
  figures/data/11-superposition-phasecurves.dat  the two proven sufficient conditions
      r = kappa(S)/v(S) and S = (2+r)/(2+5r).
The expected loss is evaluated by exact integration over v (closed form) and Gauss-Legendre
quadrature over u. Runs in about a minute on one CPU thread.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from scipy.optimize import minimize, brentq
import torch
torch.set_num_threads(1)

xg, wg = np.polynomial.legendre.leggauss(16)
P = 24
U = ((np.arange(P)[:, None] + (xg[None, :] + 1) / 2) / P).ravel()
WU = np.tile(wg / 2 / P, P)
relu = lambda z: np.maximum(z, 0.0)


def Phi(u, z):  # antiderivative in z of (u - relu(z))^2, zero at z = 0
    return np.where(z <= 0, u * u * z, (u**3 - (u - z)**3) / 3)


def F(a, beta, b, S):
    """E (x1 - relu(a x1 + beta x2 + b))^2 for the sparse uniform features."""
    q = 1 - S
    none = relu(b)**2
    one = WU @ (U - relu(a * U + b))**2
    oth = WU @ relu(beta * U + b)**2
    c = a * U + b
    both = WU @ ((U - relu(c))**2 if abs(beta) < 1e-12 else (Phi(U, c + beta) - Phi(U, c)) / beta)
    return S * S * none + S * q * (one + oth) + q * q * both


def loss(th, S, r):
    w1, w2, b1, b2 = th
    return F(w1 * w1, w1 * w2, b1, S) + r * F(w2 * w2, w1 * w2, b2, S)


def vS(S):
    return (1 - S) * (1 + 3 * S) / 12


def kappa(S):
    q = 1 - S
    if S == 0:
        return 1 / 12
    tau = (-(S * S + q) + np.sqrt((S * S + q)**2 + S * q**3)) / (S * q)
    return S * (S * tau**2 + q * tau**3 / 3) + q * (tau**2 - q * tau + q / 3)


def gmin(S, r):
    q = 1 - S
    best = None
    for s0 in ([1, -1, 0, 0], [1, -0.3, 0.05, q / 2], [1, -0.05, 0.0, q / 2]):
        res = minimize(loss, s0, args=(S, r), method="L-BFGS-B",
                       options=dict(ftol=1e-15, gtol=1e-11, maxiter=400))
        if best is None or res.fun < best.fun:
            best = res
    return best


def train(S, r, seed, steps=1500, batch=2048):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(1, 2, generator=g) * 0.5).requires_grad_()
    b = torch.zeros(2, requires_grad=True)
    imp = torch.tensor([1.0, r])
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for t in range(steps):
        if t == 1000:
            for grp in opt.param_groups:
                grp["lr"] = 2e-3
        x = torch.rand(batch, 2, generator=g) * (torch.rand(batch, 2, generator=g) > S)
        xh = torch.relu(x @ W.T @ W + b)
        L = ((x - xh)**2 * imp).sum(1).mean()
        opt.zero_grad(); L.backward(); opt.step()
    w = W.detach().numpy().ravel(); bb = b.detach().numpy()
    return loss([w[0], w[1], bb[0], bb[1]], S, r)


r = 0.5
rows = []
for S in np.round(np.arange(0.0, 0.96, 0.05), 2):
    tr = [train(S, r, seed) for seed in range(3)]
    rows.append((S, r * vS(S), (1 + r) * (1 - S)**2 / 6, gmin(S, r).fun, *tr))
with open("figures/data/11-superposition-loss.dat", "w") as f:
    f.write("S dedicated antipodal optimum seed0 seed1 seed2\n")
    for row in rows:
        f.write(" ".join(f"{z:.6g}" for z in row) + "\n")

ratio = lambda S: kappa(S) / vS(S)
out = []
for r in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
    sup = lambda S: gmin(S, r).fun < r * vS(S) * (1 - 1e-6)
    lo, hi = 0.0, 0.995
    if sup(lo):
        Sc = 0.0
    else:
        for _ in range(9):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if sup(mid) else (mid, hi)
        Sc = (lo + hi) / 2
    Sk = 0.0 if r >= 1 else brentq(lambda S: ratio(S) - r, 1e-9, 0.9999)
    Sa = (2 + r) / (2 + 5 * r)
    out.append((r, Sc))
    print(r, Sc, flush=True)
with open("figures/data/11-superposition-phase.dat", "w") as f:
    f.write("r numeric\n")
    for row in out:
        f.write(" ".join(f"{z:.6g}" for z in row) + "\n")
with open("figures/data/11-superposition-phasecurves.dat", "w") as f:
    f.write("r kappa antipodal\n")
    for r in np.linspace(0.02, 1.0, 50):
        Sk = 0.0 if r >= 1 else brentq(lambda S: ratio(S) - r, 1e-9, 0.9999)
        f.write(f"{r:.4f} {Sk:.6f} {(2 + r) / (2 + 5 * r):.6f}\n")
