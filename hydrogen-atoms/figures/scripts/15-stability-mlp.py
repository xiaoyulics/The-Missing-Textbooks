"""Chapter 15 (stability): progressive sharpening and the edge of stability in a small network.

A fully connected tanh network 10 -> 64 -> 64 -> 1 (PyTorch default initialization, seed 1) is
trained by full-batch gradient descent on the square loss (1/2n) sum_i (f(x_i) - y_i)^2 for
n = 500 Gaussian inputs x_i in R^10 (seed 0), with labels y_i = 2 * a^T tanh(W x_i) from a random
teacher (W in R^{32 x 10} with N(0, 1/10) entries, a in R^32 with N(0, 1/32) entries; seed 0).
Step sizes eta = 0.15, 0.25, 0.4 (2/eta = 13.3, 8, 5), each run until time eta*t = 900.
Every 5 units of time the sharpness (largest Hessian eigenvalue of the training loss) is estimated
by 20 steps of power iteration with Hessian-vector products, warm-started from the previous
eigenvector.
Output: figures/data/15-stability-mlp.dat with columns
  time  sA sB sC  lA lB lC
(time = eta*t; s = sharpness, l = training loss, for eta = 0.15, 0.25, 0.4).
Runs in well under a minute on a laptop CPU.
"""
import math
import os

import torch

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

n, d, m = 500, 10, 64
torch.manual_seed(0)
X = torch.randn(n, d)
W_teacher = torch.randn(d, 32) / math.sqrt(d)
a_teacher = torch.randn(32, 1) / math.sqrt(32)
Y = 2.0 * torch.tanh(X @ W_teacher) @ a_teacher


def make_model():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(d, m), torch.nn.Tanh(), torch.nn.Linear(m, m),
                               torch.nn.Tanh(), torch.nn.Linear(m, 1))


def loss_fn(model):
    return 0.5 * ((model(X) - Y) ** 2).mean()


def sharpness(model, v=None, iters=20):
    params = list(model.parameters())
    grads = torch.autograd.grad(loss_fn(model), params, create_graph=True)
    if v is None:
        g = torch.Generator().manual_seed(2)
        v = [torch.randn(p.shape, generator=g) for p in params]
    norm = math.sqrt(sum(float((x * x).sum()) for x in v))
    v = [x / norm for x in v]
    lam = 0.0
    for _ in range(iters):
        hv = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
        lam = sum(float((a * b).sum()) for a, b in zip(hv, v))
        norm = math.sqrt(sum(float((x * x).sum()) for x in hv))
        v = [x / norm for x in hv]
    return lam, v


def run(eta, horizon=900.0, every=5.0):
    model = make_model()
    steps = int(round(horizon / eta))
    stride = int(round(every / eta)) if every / eta >= 1 else 1
    times, sharp, losses, v = [], [], [], None
    for t in range(steps + 1):
        if t % stride == 0 or t == steps:
            lam, v = sharpness(model, v)
            times.append(eta * t)
            sharp.append(lam)
            with torch.no_grad():
                losses.append(loss_fn(model).item())
        model.zero_grad()
        loss_fn(model).backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= eta * p.grad
    return times, sharp, losses


def main():
    etas = (0.15, 0.25, 0.4)
    results = [run(eta) for eta in etas]
    grid = [5.0 * k for k in range(int(900 / 5) + 1)]
    cols = []
    for times, sharp, losses in results:
        lookup = {round(t, 6): (s, l) for t, s, l in zip(times, sharp, losses)}
        col = []
        for tt in grid:
            key = min(lookup, key=lambda k: abs(k - tt))
            col.append(lookup[key])
        cols.append(col)
    with open(os.path.join(DATA, "15-stability-mlp.dat"), "w") as f:
        f.write("time sA sB sC lA lB lC\n")
        for i, tt in enumerate(grid):
            s = [cols[j][i][0] for j in range(3)]
            l = [cols[j][i][1] for j in range(3)]
            f.write(f"{tt:.1f} " + " ".join(f"{x:.4f}" for x in s) + " " + " ".join(f"{x:.5e}" for x in l) + "\n")


if __name__ == "__main__":
    main()
