"""Chapter 9 (warmup), Figure 2: the sharpness of a small network during a linear warmup.

A tanh network with three hidden layers of width 64 and an output multiplier alpha = 5
(a sharp initialization) is trained by full-batch gradient descent on the square loss,
with 256 Gaussian inputs in R^10 labelled by a random tanh teacher.  The learning rate is
warmed up linearly to eta_peak = 0.1 over Tw steps, eta_t = eta_peak min(1, (t+1)/Tw);
without warmup the same peak diverges.  The sharpness (top Hessian eigenvalue) is computed
every 20 steps by power iteration with Hessian-vector products.

Writes figures/data/09-warmup-mlp.dat and figures/data/09-warmup-mlp-nowarm.dat, and prints the
numbers quoted in the chapter.
torch with 2 threads, fixed seeds; under a minute on a laptop CPU.
"""
import math
import os

import torch

torch.set_num_threads(2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
N, DIM, WIDTH, DEPTH, ALPHA = 256, 10, 64, 3, 5.0
ETA_PEAK = 0.1

gen = torch.Generator().manual_seed(0)
X = torch.randn(N, DIM, generator=gen)
with torch.no_grad():
    Y = torch.tanh(X @ (torch.randn(DIM, 32, generator=gen) / DIM ** 0.5)) @ (
        torch.randn(32, 1, generator=gen) / 32 ** 0.5)
    Y = Y / Y.std()


def init_params(seed=1):
    g = torch.Generator().manual_seed(seed)
    dims = [DIM] + [WIDTH] * DEPTH + [1]
    return [(torch.randn(dims[i], dims[i + 1], generator=g) / dims[i] ** 0.5).requires_grad_(True)
            for i in range(len(dims) - 1)]


def loss_fn(ps):
    h = X
    for W in ps[:-1]:
        h = torch.tanh(h @ W)
    return 0.5 * ((ALPHA * (h @ ps[-1]) - Y) ** 2).mean()


def sharpness(ps, v, iters):
    """Top Hessian eigenvalue by power iteration (warm-started at v)."""
    grads = torch.autograd.grad(loss_fn(ps), ps, create_graph=True)
    if v is None:
        g = torch.Generator().manual_seed(2)
        v = [torch.randn(p.shape, generator=g) for p in ps]
    nrm = torch.sqrt(sum((a * a).sum() for a in v))
    v = [a / nrm for a in v]
    lam = float("nan")
    for _ in range(iters):
        hv = torch.autograd.grad(grads, ps, grad_outputs=v, retain_graph=True)
        lam = sum((a * b).sum() for a, b in zip(hv, v)).item()
        nrm = torch.sqrt(sum((a * a).sum() for a in hv))
        v = [a.detach() / nrm for a in hv]
    return lam, v


def train(Tw, T, every=20, measure=True):
    ps = init_params()
    v, rec = None, []
    for t in range(T):
        eta = ETA_PEAK if Tw == 0 else ETA_PEAK * min(1.0, (t + 1) / Tw)
        L = loss_fn(ps)
        Lv = L.item()
        if not math.isfinite(Lv) or Lv > 1e8:
            rec.append((t, eta, Lv, float("nan")))
            break
        if measure and t % every == 0:
            S, v = sharpness(ps, v, 100 if v is None else 20)
            rec.append((t, eta, Lv, S))
        elif not measure:
            rec.append((t, eta, Lv, float("nan")))
        g = torch.autograd.grad(L, ps)
        with torch.no_grad():
            for p, gp in zip(ps, g):
                p -= eta * gp
    return rec


def main():
    runs = {Tw: train(Tw, 2001) for Tw in (250, 1000)}
    with open(os.path.join(OUT, "09-warmup-mlp.dat"), "w") as f:
        f.write("t etaA edgeA SA LA etaB edgeB SB LB\n")
        A, B = runs[250], runs[1000]
        for (t, ea, la, sa), (_, eb, lb, sb) in zip(A, B):
            f.write(f"{t} {ea:.6g} {2 / ea:.6g} {sa:.6g} {la:.6g} {eb:.6g} {2 / eb:.6g} {sb:.6g} {lb:.6g}\n")
    for Tw, r in runs.items():
        last = r[-1]
        S_end = [s for (t, e, l, s) in r if t >= max(Tw, 1) and math.isfinite(s)]
        print(f"Tw = {Tw}: final loss {last[2]:.3e}; sharpness after warmup in"
              f" [{min(S_end):.2f}, {max(S_end):.2f}], 2/eta_peak = {2 / ETA_PEAK:.1f}")
        # numbers quoted in the chapter: where S meets 2/eta_t, the ratio S/(2/eta_t) during the rest
        # of the warmup (after the largest loss, the catapult), the mean of S over the last 500 steps
        pts = [(t, e, l, s) for (t, e, l, s) in r if math.isfinite(s)]
        meet = next(t for (t, e, l, s) in pts if s >= 2 / e)
        peak = max((l, t) for (t, e, l, s) in pts if 0 < t <= Tw)[1]
        rat = [s * e / 2 for (t, e, l, s) in pts if max(meet, peak) < t <= Tw]
        tail = [s for (t, e, l, s) in pts if t >= 1500]
        t01, s01 = next((t, s) for (t, e, l, s) in pts if l <= 0.0102)
        print(f"   S meets 2/eta_t between t = {meet - 20} and {meet}; largest loss of the warmup at t = {peak}"
              f" ({[(t, round(l, 3), round(s, 1)) for (t, e, l, s) in pts if abs(t - peak) <= 20]});"
              f" then S/(2/eta_t) in [{min(rat):.3f}, {max(rat):.3f}] until t = {Tw};"
              f" mean S over t >= 1500: {sum(tail) / len(tail):.2f}; first loss <= 0.0102 at t = {t01}, S = {s01:.1f}")
    nw = train(0, 30, every=1)
    with open(os.path.join(OUT, "09-warmup-mlp-nowarm.dat"), "w") as f:
        f.write("t loss\n")
        for (t, e, l, s) in nw:
            if math.isfinite(l):
                f.write(f"{t} {l:.6g}\n")
    print("no warmup: initial sharpness", nw[0][3], "losses", [f"{r[2]:.3g}" for r in nw])
    for Tw in (50, 100, 150, 200):
        r = train(Tw, 1000, measure=False)
        print(f"Tw = {Tw}: {'diverged at step ' + str(r[-1][0]) if not math.isfinite(r[-1][2]) or r[-1][2] > 1e8 else 'final loss %.3e' % r[-1][2]}")


if __name__ == "__main__":
    main()
