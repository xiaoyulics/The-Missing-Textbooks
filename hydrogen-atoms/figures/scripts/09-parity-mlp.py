"""Abrupt learning and hidden progress in a two-layer network (Chapter 9, figure fig:parity:mlp).

Setting: x uniform on {-1,1}^50, y = x_1 x_2 x_3 (d = 50, k = 3). Two-layer ReLU network
f(x) = sum_j a_j relu(<W_j, x> + b_j) of width 256, standard Gaussian initialization
(W_ji ~ N(0, 1/d), b_j = 0, a_j ~ N(0, 1/m)), hinge loss max(0, 1 - y f(x)), plain SGD with step
0.02 and fresh minibatches of 32 samples, all parameters trained. Four runs (seeds 0..3).
Recorded every 250 steps on a fixed test set of 4000 samples: the hinge loss, the 0-1 error, and a
progress measure, the largest fraction of a neuron's squared input weight carried by the support,
max_j (W_j1^2 + W_j2^2 + W_j3^2)/|W_j|^2 (k/d = 0.06 for a uniformly spread neuron).
Output: figures/data/09-parity-mlp.dat with columns step, loss0..loss3, err0..err3, prog0..prog3.
Runtime: about 50 seconds with two threads.
"""
import os
import math
import torch

D, K, M, B, LR, STEPS, EVERY, RUNS = 50, 3, 256, 32, 0.02, 32000, 250, 4
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "09-parity-mlp.dat")


def run(seed):
    torch.manual_seed(seed)
    W = (torch.randn(M, D) / math.sqrt(D)).requires_grad_()
    b = torch.zeros(M).requires_grad_()
    a = (torch.randn(M) / math.sqrt(M)).requires_grad_()
    opt = torch.optim.SGD([W, b, a], lr=LR)
    Xt = torch.randint(0, 2, (4000, D)).float() * 2 - 1
    yt = Xt[:, :K].prod(1)
    out = []
    for t in range(STEPS + 1):
        if t % EVERY == 0:
            with torch.no_grad():
                o = torch.relu(Xt @ W.T + b) @ a
                loss = torch.clamp(1 - yt * o, min=0).mean().item()
                err = (torch.sign(o) != yt).float().mean().item()
                prog = ((W[:, :K] ** 2).sum(1) / (W ** 2).sum(1)).max().item()
            out.append((t, loss, err, prog))
        X = torch.randint(0, 2, (B, D)).float() * 2 - 1
        y = X[:, :K].prod(1)
        f = torch.relu(X @ W.T + b) @ a
        lossb = torch.clamp(1 - y * f, min=0).mean()
        opt.zero_grad()
        lossb.backward()
        opt.step()
    return out


def main():
    torch.set_num_threads(2)
    res = [run(s) for s in range(RUNS)]
    with open(OUT, "w") as f:
        f.write("step " + " ".join(f"loss{i}" for i in range(RUNS)) + " "
                + " ".join(f"err{i}" for i in range(RUNS)) + " "
                + " ".join(f"prog{i}" for i in range(RUNS)) + "\n")
        for j in range(len(res[0])):
            vals = [res[0][j][0]] + [r[j][1] for r in res] + [r[j][2] for r in res] + [r[j][3] for r in res]
            f.write(" ".join(f"{v:.4f}" if i else str(int(v)) for i, v in enumerate(vals)) + "\n")
    for i, r in enumerate(res):
        drop = next((s for s, l, e, p in r if e < 0.1), None)
        print("run", i, "error below 0.1 at step", drop, "initial prog", round(r[0][3], 3))


if __name__ == "__main__":
    main()
