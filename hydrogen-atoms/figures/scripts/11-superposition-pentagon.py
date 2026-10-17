"""Five features in two dimensions (Chapter 11, superposition).

Trains x' = relu(W^T W x + b) with p=5, m=2, importances 0.8^(i-1), features that vanish with
probability S and are otherwise uniform on [0,1], by Adam on fresh batches; keeps the better of
two seeds. Writes the columns w_i of W for four sparsities to figures/data/11-superposition-pentagon.dat;
the column panel is the index of the sparsity, or -1 for a column of norm below 0.1.
Runs in about 40 seconds on one CPU thread.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
torch.set_num_threads(1)


def train(S, imp, m, seed, steps=3000, batch=4096):
    p = len(imp)
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(m, p, generator=g) * 0.3).requires_grad_()
    b = torch.zeros(p, requires_grad=True)
    I = torch.tensor(imp, dtype=torch.float32)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    sample = lambda n: torch.rand(n, p, generator=g) * (torch.rand(n, p, generator=g) > S)
    for t in range(steps):
        if t == 2000:
            for gr in opt.param_groups:
                gr["lr"] = 2e-3
        x = sample(batch)
        L = (((x - torch.relu(x @ W.T @ W + b))**2) * I).sum(1).mean()
        opt.zero_grad(); L.backward(); opt.step()
    x = sample(200000)
    with torch.no_grad():
        L = (((x - torch.relu(x @ W.T @ W + b))**2) * I).sum(1).mean().item()
    return W.detach().numpy(), b.detach().numpy(), L


imp = [0.8**i for i in range(5)]
with open("figures/data/11-superposition-pentagon.dat", "w") as f:
    f.write("S panel feature x y norm bias\n")
    for k, S in enumerate([0.0, 0.5, 0.8, 0.95]):
        W, b, L = min((train(S, imp, 2, seed) for seed in range(2)), key=lambda z: z[2])
        n = np.linalg.norm(W, axis=0)
        print(f"S={S} loss={L:.5f} norms={np.round(n, 3)} bias={np.round(b, 3)} "
              f"angles={np.round(np.degrees(np.arctan2(W[1], W[0])), 1)} sum|w|^2={np.sum(n**2):.2f}")
        for i in range(5):
            panel = k if n[i] > 0.1 else -1  # -1 marks a feature that is not represented
            f.write(f"{S} {panel} {i + 1} {W[0, i]:.5f} {W[1, i]:.5f} {n[i]:.5f} {b[i]:.5f}\n")
