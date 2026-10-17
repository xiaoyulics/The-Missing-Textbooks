"""Chapter 4 (modular): the change of the features after one step, as a function of width.

The network of 04-modular-transfer.py (x -> W1 -> ReLU -> W2 -> ReLU -> W3 -> f, d0 = 32 inputs, two
hidden layers of width n, one output, no biases) at initialization: W1, W2 with N(0, 2/fan_in) entries;
W3 with N(0, 1/n) entries (standard parametrization, "sp") or standard deviation sqrt(n0)/n with n0 = 64
("mup", also used by "spec"). One example x, a standard Gaussian vector rescaled to ||x||_rms = 1,
with label y = f(x) - 1, so that the loss
(1/2)(f - y)^2 has derivative exactly 1 at the output. Each configuration takes one step from this
example with base learning rate eta = 1e-5 (small enough that the change is linear in eta):
  sgd_sp    W_l <- W_l - eta * G_l for every layer;
  sgd_mup   per-layer rates eta*m, eta, eta/m (m = n/n0) and the muP output layer;
  adam_sp   W_l <- W_l - eta * sign(G_l)  (the first step of Adam with bias correction and eps -> 0);
  adam_mup  per-layer rates eta, eta/m, eta/m and the muP output layer;
  spec      W_l <- W_l + eta * lmo_{rms->rms}(G_l) = W_l - eta sqrt(fan_out/fan_in) polar(G_l).
The update matrices are formed explicitly and the network is evaluated again on x. Recorded: the change
of the pre-activations of the second hidden layer, ||u2(after) - u2(before)||_rms / eta, averaged over
8 seeds, once when all three layers take their step (columns sgdsp ... spec) and once when only the
hidden matrix W2 does (columns with suffix "h"); and for "spec" the two first-order bounds of the
spectral-condition theorem,
  lower = ||h1||_rms        (no-cancellation bound) and
  upper = (a1 + a2) ||x||_rms with a_l = ||W_l||_{rms->rms},
averaged over the same seeds. Widths n = 64, 128, ..., 4096.
Output: figures/data/04-modular-onestep.dat with columns
  n  sgdsp sgdmup adamsp adammup spec  sgdsph sgdmuph adamsph adammuph spech  speclo spechi
Runs in well under a minute.
"""
import math
import os
import sys

import torch

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
dt = torch.float64
d0, n0, ETA, SEEDS = 32, 64, 1e-5, 8
WIDTHS = [64, 128, 256, 512, 1024, 2048, 4096]
CFGS = ["sgd_sp", "sgd_mup", "adam_sp", "adam_mup", "spec"]


def rms(v):
    return (v.norm() / math.sqrt(v.numel())).item()


def r2r(W):
    """||W||_{rms->rms} = sqrt(fan_in/fan_out) ||W||_op (power iteration on the smaller Gram matrix)."""
    if min(W.shape) <= 64:
        S = W.T @ W if W.shape[1] <= W.shape[0] else W @ W.T
        op = math.sqrt(torch.linalg.eigvalsh(S)[-1].item())
    else:
        v = torch.randn(W.shape[1], dtype=dt, generator=torch.Generator().manual_seed(99))
        for _ in range(60):
            v = W.T @ (W @ v)
            v = v / v.norm()
        op = (W @ v).norm().item()
    return math.sqrt(W.shape[1] / W.shape[0]) * op


def forward(Ws, x):
    u1 = Ws[0] @ x
    h1 = torch.relu(u1)
    u2 = Ws[1] @ h1
    h2 = torch.relu(u2)
    return u1, h1, u2, h2, Ws[2] @ h2


def gradients(Ws, x):
    """Gradients of (1/2)(f - y)^2 with f - y = 1, as rank-one factors (delta_l, input_l)."""
    u1, h1, u2, h2, f = forward(Ws, x)
    d3 = torch.ones(1, dtype=dt)
    d2 = (Ws[2].T @ d3) * (u2 > 0)
    d1 = (Ws[1].T @ d2) * (u1 > 0)
    return [(d1, x), (d2, h1), (d3, h2)]


def update(cfg, Ws, factors, n, layers=(0, 1, 2)):
    m = n / n0
    rates = {"sgd_sp": [1, 1, 1], "sgd_mup": [m, 1, 1 / m], "adam_sp": [1, 1, 1],
             "adam_mup": [1, 1 / m, 1 / m], "spec": [1, 1, 1]}[cfg]
    new = []
    for i, (W, (d, h), r) in enumerate(zip(Ws, factors, rates)):
        if i not in layers:
            new.append(W)
            continue
        G = torch.outer(d, h)
        if cfg.startswith("sgd"):
            D = -G
        elif cfg.startswith("adam"):
            D = -torch.sign(G)
        else:  # rank-one gradient: polar(G) = (d/|d|)(h/|h|)^T
            D = -math.sqrt(W.shape[0] / W.shape[1]) * torch.outer(d / d.norm(), h / h.norm())
        new.append(W + ETA * r * D)
    return new


def main():
    rows = []
    for n in WIDTHS:
        acc = {c: 0.0 for c in CFGS + [c + "h" for c in CFGS] + ["lo", "hi"]}
        for seed in range(SEEDS):
            g = torch.Generator().manual_seed(seed)
            W1 = torch.randn(n, d0, generator=g, dtype=dt) * math.sqrt(2 / d0)
            W2 = torch.randn(n, n, generator=g, dtype=dt) * math.sqrt(2 / n)
            z3 = torch.randn(1, n, generator=g, dtype=dt)
            x = torch.randn(d0, generator=g, dtype=dt)
            x = x / rms(x)
            for cfg in CFGS:
                W3 = z3 * (math.sqrt(1 / n) if cfg.endswith("sp") else math.sqrt(n0) / n)
                Ws = [W1, W2, W3]
                before = forward(Ws, x)
                grads = gradients(Ws, x)
                after = forward(update(cfg, Ws, grads, n), x)
                acc[cfg] += rms(after[2] - before[2]) / ETA / SEEDS
                after = forward(update(cfg, Ws, grads, n, layers=(1,)), x)
                acc[cfg + "h"] += rms(after[2] - before[2]) / ETA / SEEDS
            acc["lo"] += rms(torch.relu(W1 @ x)) / SEEDS
            acc["hi"] += (r2r(W1) + r2r(W2)) * rms(x) / SEEDS
        rows.append([n] + [acc[c] for c in CFGS] + [acc[c + "h"] for c in CFGS] + [acc["lo"], acc["hi"]])
        print(n, " ".join(f"{v:.4g}" for v in rows[-1][1:]), file=sys.stderr)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "04-modular-onestep.dat"), "w") as fh:
        names = [c.replace("_", "") for c in CFGS]
        fh.write("n " + " ".join(names) + " " + " ".join(c + "h" for c in names) + " speclo spechi\n")
        for r in rows:
            fh.write(f"{r[0]} " + " ".join(f"{v:.6g}" for v in r[1:]) + "\n")


if __name__ == "__main__":
    main()
