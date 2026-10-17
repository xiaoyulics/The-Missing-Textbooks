"""Figure 1 of Chapter 13 (Learning in Context): in-context risk versus the
context length N for d = 8 and noise variance sigma^2 = 0.5.

Prompts: x_i, x_q ~ N(0, I_d), w_star ~ N(0, I_d), y_i = <w_star, x_i> + eps_i,
eps_i ~ N(0, sigma^2); the query label has the same noise.

Curves (written to 13-icl-risk-theory.dat, excess risk per dimension,
(E(y_q - yhat)^2 - sigma^2)/d):
  onestep  optimal one-step learner, d(d+1+s2)/(N+d+1+s2) / d   (exact)
  bayes    posterior mean = ridge with N*lambda = s2              (Monte Carlo
           of the exact conditional risk tr(s2 (X^T X + s2 I)^{-1}), 4000 draws)
  ols      minimum-norm least squares (Chapter 1 formula with E r^2 = d)
Points (13-icl-risk-sim.dat): a single linear self-attention layer with all
2(d+1)^2 parameters trained by Adam on fresh prompts, one model per N; its risk
is estimated on 1.2*10^5 fresh prompts. Also the learned step size tr(Gamma)/d and
the size of the parts of the predictor that the theory says must vanish.

Deterministic (seeded); about a minute and a half on a laptop CPU.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import torch

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
D, S2 = 8, 0.5
SEED = 20261017


def onestep(N):
    return D * (D + 1 + S2) / (N + D + 1 + S2)


def ols(N):
    if N <= D - 2:
        return (D - N) + S2 * N / (D - N - 1)
    if N >= D + 2:
        return S2 * D / (N - D - 1)
    return np.nan


def bayes(N, rng, draws=4000):
    tot = 0.0
    for _ in range(draws):
        X = rng.standard_normal((N, D))
        ev = np.linalg.eigvalsh(X.T @ X)
        tot += np.sum(S2 / (ev + S2))
    return tot / draws


def prompts(B, N, g):
    w = torch.randn(B, D, generator=g)
    X = torch.randn(B, N, D, generator=g)
    y = torch.einsum("bnd,bd->bn", X, w) + S2 ** 0.5 * torch.randn(B, N, generator=g)
    xq = torch.randn(B, D, generator=g)
    yq = (xq * w).sum(1) + S2 ** 0.5 * torch.randn(B, generator=g)
    return X, y, xq, yq


def lsa(WKQ, WPV, X, y, xq):
    """Prediction of one linear self-attention layer, read from the query column."""
    Zc = torch.cat([X, y[:, :, None]], 2)               # context tokens z_i = (x_i, y_i)
    G = torch.einsum("bni,bnj->bij", Zc, Zc) / X.shape[1]  # (1/N) sum z_i z_i^T
    zq = torch.cat([xq, torch.zeros(xq.shape[0], 1)], 1)
    return torch.einsum("i,bij,jk,bk->b", WPV[D], G, WKQ, zq)


def train(N, g, steps=2000, B=1024):
    WKQ = (0.1 * torch.randn(D + 1, D + 1, generator=g)).requires_grad_()
    WPV = (0.1 * torch.randn(D + 1, D + 1, generator=g)).requires_grad_()
    opt = torch.optim.Adam([WKQ, WPV], lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        X, y, xq, yq = prompts(B, N, g)
        loss = ((yq - lsa(WKQ, WPV, X, y, xq)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return WKQ.detach(), WPV.detach()


def main():
    rng = np.random.default_rng(SEED)
    g = torch.Generator().manual_seed(SEED)
    Ns = sorted(set(np.unique(np.round(np.logspace(0, np.log10(256), 60)).astype(int))))
    with open(os.path.join(DATA, "13-icl-risk-theory.dat"), "w") as f:
        f.write("N onestep bayes ols gamma\n")
        for N in Ns:
            o = ols(N)
            f.write(f"{N} {onestep(N)/D:.6f} {bayes(N, rng)/D:.6f} "
                    f"{'nan' if np.isnan(o) else f'{o/D:.6f}'} {N/(N+D+1+S2):.6f}\n")
    with open(os.path.join(DATA, "13-icl-risk-sim.dat"), "w") as f:
        f.write("N risk se gammahat offdiag even theoryrisk\n")
        for N in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            WKQ, WPV = train(N, g)
            u1, u2 = WPV[D, :D], WPV[D, D]
            W11, w21 = WKQ[:D, :D], WKQ[D, :D]
            Gam = u2 * W11.T + torch.outer(w21, u1)
            gam = torch.trace(Gam).item() / D
            off = torch.linalg.norm(Gam - gam * torch.eye(D)).item() / np.sqrt(D)
            even = (torch.linalg.norm(W11.T @ u1) + torch.linalg.norm(u2 * w21)).item()
            errs = []
            with torch.no_grad():
                for _ in range(12):
                    X, y, xq, yq = prompts(10000, N, g)
                    errs.append(((yq - lsa(WKQ, WPV, X, y, xq)) ** 2).numpy())
            e = np.concatenate(errs) - S2
            f.write(f"{N} {e.mean()/D:.6f} {e.std()/np.sqrt(len(e))/D:.6f} {gam:.6f} "
                    f"{off:.2e} {even:.2e} {onestep(N)/D:.6f}\n")
            print(N, e.mean() / D, onestep(N) / D, gam, N / (N + D + 1 + S2), off, even, flush=True)


if __name__ == "__main__":
    main()
