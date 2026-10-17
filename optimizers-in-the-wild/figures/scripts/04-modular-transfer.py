"""Chapter 4 (modular): learning-rate transfer across width in a small MLP.

A ReLU network x -> W1 -> ReLU -> W2 -> ReLU -> W3 -> f with input dimension d0 = 32, two hidden layers
of width n in {64, 128, 256, 512, 1024} and one output, no biases, learns the multi-index target
y = sum_{j<=4} ((q_j . x)^2 - 1) / sqrt(8) (q_1..q_4 orthonormal, seed 5; unit variance) from fresh
standard Gaussian minibatches of B = 32 inputs, with the square loss (1/2B) sum_i (f(x_i) - y_i)^2, for
T = 150 steps at a constant learning rate. The target is a quadratic form in four directions: random
features of these widths fit it poorly, so the first layer has to learn the directions. The loss
reported is the loss after training on a fixed set of 4096 Gaussian inputs (seed 7).

Five configurations, each swept over a grid of base learning rates eta spaced by factors of 2:
  sgd_sp    SGD, standard parametrization: He initialization N(0, 2/fan_in) for W1, W2, N(0, 1/fan_in)
            for W3, one learning rate eta for every layer;
  sgd_mup   SGD, muP: as sgd_sp, but W3 initialized with standard deviation sqrt(n0)/n (n0 = 64), and
            per-layer learning rates eta*m, eta, eta/m for W1, W2, W3, with m = n/n0 (fan_out/fan_in rule);
  adam_sp   Adam (beta1 = 0.9, beta2 = 0.99, eps = 1e-8, bias correction), standard parametrization;
  adam_mup  Adam, muP: W3 as in sgd_mup, learning rates eta, eta/m, eta/m (1/fan_in rule);
  spec      normalized steepest descent in the modular max norm: every layer takes the step
            eta * lmo_{rms->rms}(G) = -eta * sqrt(fan_out/fan_in) * polar(G) of its minibatch gradient G
            (exact polar factor, computed from the rank-B factorization of G; Muon without momentum),
            initialization as in the muP configurations.
At the base width n0 = 64 the SP and muP versions of each optimizer coincide (same initial weights,
same per-layer learning rates). Every learning rate of a grid uses the same initial weights and the
same data stream (common random numbers), with seed 0.

Outputs (whitespace-separated, header row):
  figures/data/04-modular-transfer-curves.dat  column log2lr, then one column per configuration and
      width, named <cfg><n> with the underscore dropped (e.g. adamsp256): the final loss, nan if the run
      diverged or if the learning rate is outside that configuration's grid;
  figures/data/04-modular-transfer-optima.dat  one row per width: for each configuration the log2 of
      the optimal base learning rate (vertex of the parabola through the best grid point and its two
      neighbors in (log2 eta, log loss); the grid point itself if a neighbor is missing), the log2 of
      the largest grid learning rate below the first run that overflowed (columns <cfg>stable; Adam
      runs with too large a rate stop learning without overflowing, so for Adam and for the spectral
      steps this column is the top of the grid), the optimum relative to width 64
      (columns <cfg>rel), and the largest stable rate of sgd_sp relative to width 64.
Runs in about a minute on two CPU threads (longer on a loaded machine).
"""
import math
import os
import sys
import time

import torch

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = sys.argv[1] if len(sys.argv) > 1 else DATA

d0, dN, n0 = 32, 1, 64
B, T = 32, 150
WIDTHS = [64, 128, 256, 512, 1024]
GRIDS = {  # log2 of the base learning rates (placed by a pilot run to bracket every optimum)
    "sgd_sp": range(-8, 2),
    "sgd_mup": range(-6, 2),
    "adam_sp": range(-12, -4),
    "adam_mup": range(-10, -3),
    "spec": range(-7, 0),
}
CFGS = list(GRIDS)


def make_teacher(k=4, seed=5):
    """y = sum_j (z_j^2 - 1) / sqrt(2k), z = Q^T x for k orthonormal directions: unit variance."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d0, k, generator=g))

    def teacher(x):
        z = x @ Q
        return ((z ** 2 - 1).sum(-1) / math.sqrt(2 * k)).unsqueeze(-1)
    return teacher


def init(n, cfg, gen):
    W1 = torch.randn(n, d0, generator=gen) * math.sqrt(2 / d0)
    W2 = torch.randn(n, n, generator=gen) * math.sqrt(2 / n)
    if cfg.endswith("sp"):
        W3 = torch.randn(dN, n, generator=gen) * math.sqrt(1 / n)
    else:
        W3 = torch.randn(dN, n, generator=gen) * (math.sqrt(n0) / n)
    return [W1, W2, W3]


def forward_backward(Ws, x, y, need_grad=True, full=True):
    """Ws: list of (L, out, in) tensors (one network per learning rate); x: (B, d0); y: (B, 1)."""
    W1, W2, W3 = Ws
    u1 = torch.matmul(x, W1.transpose(1, 2))
    h1 = torch.relu(u1)
    u2 = torch.bmm(h1, W2.transpose(1, 2))
    h2 = torch.relu(u2)
    f = torch.bmm(h2, W3.transpose(1, 2))
    r = f - y
    loss = 0.5 * (r ** 2).sum(-1).mean(-1)
    if not need_grad:
        return loss
    d3 = r / x.shape[0]
    d2 = torch.bmm(d3, W3) * (u2 > 0)
    d1 = torch.bmm(d2, W2) * (u1 > 0)
    grads = [torch.matmul(d1.transpose(1, 2), x), torch.bmm(d2.transpose(1, 2), h1),
             torch.bmm(d3.transpose(1, 2), h2)] if full else None
    factors = [(d1, x.expand(d1.shape[0], *x.shape)), (d2, h1), (d3, h2)]
    return loss, grads, factors


def polar_from_factors(D, H):
    """polar(D^T H) for D: (L, B, m), H: (L, B, k), via thin QR factors and a B x B SVD."""
    Q1, R1 = torch.linalg.qr(D.transpose(1, 2))
    Q2, R2 = torch.linalg.qr(H.transpose(1, 2))
    U, s, Vh = torch.linalg.svd(torch.bmm(R1, R2.transpose(1, 2)), full_matrices=False)
    keep = (s > 1e-6 * s[:, :1]).to(U.dtype).unsqueeze(1)
    return torch.bmm(Q1, torch.bmm(U * keep, Vh).bmm(Q2.transpose(1, 2)))


def train(cfg, n, log2lrs, teacher, seed=0):
    L = len(log2lrs)
    gen = torch.Generator().manual_seed(seed)
    Ws = [W.unsqueeze(0).repeat(L, 1, 1).contiguous() for W in init(n, cfg, gen)]
    eta = [2.0 ** k for k in log2lrs]
    m = n / n0
    mult = {"sgd_sp": [1, 1, 1], "sgd_mup": [m, 1, 1 / m], "adam_sp": [1, 1, 1],
            "adam_mup": [1, 1 / m, 1 / m], "spec": [1, 1, 1]}[cfg]
    lr = [torch.tensor([e * mu for e in eta]).view(L, 1, 1) for mu in mult]
    if cfg.startswith("adam"):
        b1, b2, eps = 0.9, 0.99, 1e-8
        M = [torch.zeros_like(W) for W in Ws]
        V = [torch.zeros_like(W) for W in Ws]
    gdata = torch.Generator().manual_seed(1000 + seed)
    for t in range(T):
        x = torch.randn(B, d0, generator=gdata)
        _, G, F = forward_backward(Ws, x, teacher(x), full=(cfg != "spec"))
        for l in range(3):
            if cfg.startswith("sgd"):
                Ws[l].sub_(G[l].mul_(lr[l]))
            elif cfg.startswith("adam"):
                M[l].lerp_(G[l], 1 - b1)
                V[l].lerp_(G[l].mul_(G[l]), 1 - b2)
                den = V[l].div(1 - b2 ** (t + 1)).sqrt_().add_(eps)
                Ws[l].sub_(torch.div(M[l], den).mul_(lr[l] / (1 - b1 ** (t + 1))))
            else:
                fan_out, fan_in = Ws[l].shape[1], Ws[l].shape[2]
                P = polar_from_factors(*F[l])
                Ws[l].sub_(P.mul_(lr[l] * math.sqrt(fan_out / fan_in)))
    gtest = torch.Generator().manual_seed(7)
    xt = torch.randn(4096, d0, generator=gtest)
    loss = forward_backward(Ws, xt, teacher(xt), need_grad=False)
    return [v if math.isfinite(v) and v < 10 else float("nan") for v in loss.tolist()]


def optimum(log2lrs, losses):
    """Vertex of the parabola through the best grid point and its neighbors in (log2 eta, log loss)."""
    finite = [(k, v) for k, v in zip(log2lrs, losses) if math.isfinite(v)]
    i = min(range(len(finite)), key=lambda j: finite[j][1])
    k = finite[i][0]
    if 0 < i < len(finite) - 1 and finite[i - 1][0] == k - 1 and finite[i + 1][0] == k + 1:
        ym, y0, yp = (math.log(finite[i + j][1]) for j in (-1, 0, 1))
        curv = ym - 2 * y0 + yp
        if curv > 0:
            return k + max(-0.5, min(0.5, 0.5 * (ym - yp) / curv))
    return float(k)


def largest_stable(log2lrs, losses):
    """Largest grid learning rate below the first one whose run diverged (overflow or nan)."""
    best = None
    for k, v in zip(log2lrs, losses):
        if not math.isfinite(v):
            break
        best = k
    return float(best)


def main():
    teacher = make_teacher()
    t0 = time.time()
    table, optima, stable = {}, {n: {} for n in WIDTHS}, {n: {} for n in WIDTHS}
    for cfg in CFGS:
        grid = list(GRIDS[cfg])
        for n in WIDTHS:
            losses = train(cfg, n, grid, teacher)
            for k, v in zip(grid, losses):
                table[(cfg, n, k)] = v
            optima[n][cfg] = optimum(grid, losses)
            stable[n][cfg] = largest_stable(grid, losses)
            print(f"{cfg:8s} n={n:5d} opt log2 eta={optima[n][cfg]:6.2f} stable {stable[n][cfg]:5.1f} "
                  f"[{time.time() - t0:5.1f}s]  " +
                  " ".join("nan" if not math.isfinite(v) else f"{v:.3f}" for v in losses), file=sys.stderr)
    os.makedirs(OUT, exist_ok=True)
    keys = [c.replace("_", "") + str(n) for c in CFGS for n in WIDTHS]
    ks = sorted({k for (_, _, k) in table})
    with open(os.path.join(OUT, "04-modular-transfer-curves.dat"), "w") as fh:
        fh.write("log2lr " + " ".join(keys) + "\n")
        for k in ks:
            vals = []
            for c in CFGS:
                for n in WIDTHS:
                    v = table.get((c, n, k), float("nan"))
                    vals.append("nan" if not math.isfinite(v) else f"{v:.5f}")
            fh.write(f"{k} " + " ".join(vals) + "\n")
    with open(os.path.join(OUT, "04-modular-transfer-optima.dat"), "w") as fh:
        cols = [c.replace("_", "") for c in CFGS]
        fh.write("n " + " ".join(cols) + " " + " ".join(c + "stable" for c in cols) + " " +
                 " ".join(c + "rel" for c in cols) + " sgdspstablerel\n")
        for n in WIDTHS:
            base, sbase = optima[WIDTHS[0]], stable[WIDTHS[0]]
            fh.write(f"{n} " + " ".join(f"{optima[n][c]:.3f}" for c in CFGS) + " " +
                     " ".join(f"{stable[n][c]:.1f}" for c in CFGS) + " " +
                     " ".join(f"{optima[n][c] - base[c]:.3f}" for c in CFGS) +
                     f" {stable[n]['sgd_sp'] - sbase['sgd_sp']:.1f}\n")
    print(f"total {time.time() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
