"""Figure of Chapter 6 (Lazy Training): a kernel needs about d^2/2 samples for a quadratic
single-index target; a trained network needs far fewer.

Inputs x ~ Unif(S^{d-1}), d = 30. Target: the normalized zonal harmonic of degree 2,
f_w(x) = sqrt(N) P_{d,2}(<w, x>) = sqrt(N) (d <w,x>^2 - 1)/(d - 1), with N = N_{d,2} = d(d+1)/2 - 1 = 464
and w uniform on the sphere, so that ||f_w||_{L^2} = 1. No label noise.
Reported: the relative test error ||f_hat - f_w||^2 / ||f_w||^2, estimated on 4000 fresh points,
averaged over 3 draws of (w, training set).

Kernel: kernel ridge regression with the tangent kernel of a two-layer ReLU network with a bias
coordinate (both layers), K(x, x') = u k0(u) + k1(u), u = (1 + <x, x'>)/2,
k0(u) = (pi - arccos u)/(2 pi), k1(u) = (sqrt(1 - u^2) + (pi - arccos u) u)/(2 pi);
ridge n*lam with lam chosen from a grid by test error (an oracle choice, favorable to the kernel).
Lower bound (chapter theorem): every estimator in the span of K(., x_i) has expected error
at least 1 - n/N.
Network: f(x) = m^{-1/2} sum_r a_r relu(sqrt(d) <w_r, x> + b_r), m = 256, all of (W, b, a)
trained by full-batch Adam (lr 0.02, 2000 steps) on the mean squared error plus 1e-4 times the
squared norm of the parameters; w_r ~ N(0, I_d/d), b_r = 0, a_r ~ N(0, 0.01) at initialization
(a small output scale, far from the lazy regime).

Writes figures/data/06-lazy-gap-kernel.dat and figures/data/06-lazy-gap-net.dat.
Deterministic (numpy seeds 100-102, torch seeds 0-2); about 90 seconds.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import torch
torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
D = 30
N2 = D * (D + 1) // 2 - 1
SEEDS = [0, 1, 2]
N_KERNEL = [25, 50, 100, 200, 300, 400, 600, 800, 1200, 1600, 2000]
N_NET = [30, 60, 90, 120, 180, 240, 360, 480]
RIDGES = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

def sphere(rng, k):
    x = rng.standard_normal((k, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def target(X, w):
    return np.sqrt(N2) * (D * (X @ w) ** 2 - 1) / (D - 1)

def ntk(A, B):
    u = np.clip((A @ B.T + 1) / 2, -1, 1)
    k0 = (np.pi - np.arccos(u)) / (2 * np.pi)
    k1 = (np.sqrt(1 - u * u) + (np.pi - np.arccos(u)) * u) / (2 * np.pi)
    return u * k0 + k1

def krr_error(Xtr, ytr, Xte, yte):
    n = len(ytr)
    K = ntk(Xtr, Xtr)
    Kte = ntk(Xte, Xtr)
    errs = []
    for lam in RIDGES:
        c = np.linalg.solve(K + lam * n * np.eye(n), ytr)
        errs.append(np.mean((Kte @ c - yte) ** 2) / np.mean(yte ** 2))
    return min(errs)

def net_error(Xtr, ytr, Xte, yte, seed, m=256, steps=2000, lr=0.02, wd=1e-4):
    g = torch.Generator().manual_seed(seed)
    X = torch.tensor(Xtr * np.sqrt(D), dtype=torch.float32)
    y = torch.tensor(ytr, dtype=torch.float32)
    W = (torch.randn(m, D, generator=g) / np.sqrt(D)).requires_grad_()
    b = torch.zeros(m).requires_grad_()
    a = (torch.randn(m, generator=g) * 0.1).requires_grad_()
    opt = torch.optim.Adam([W, b, a], lr=lr)
    for _ in range(steps):
        f = torch.relu(X @ W.T + b) @ a / np.sqrt(m)
        loss = 0.5 * ((f - y) ** 2).mean() + wd * (W.pow(2).sum() + a.pow(2).sum() + b.pow(2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = (torch.relu(torch.tensor(Xte * np.sqrt(D), dtype=torch.float32) @ W.T + b) @ a / np.sqrt(m)).numpy()
    return np.mean((p - yte) ** 2) / np.mean(yte ** 2)

def main():
    Xte = sphere(np.random.default_rng(0), 4000)
    draws = []
    for s in SEEDS:
        rng = np.random.default_rng(100 + s)
        w = sphere(rng, 1)[0]
        draws.append((w, sphere(rng, max(N_KERNEL))))
    with open(os.path.join(DATA, "06-lazy-gap-kernel.dat"), "w") as fh:
        fh.write("n krr bound\n")
        for n in N_KERNEL:
            e = [krr_error(X[:n], target(X[:n], w), Xte, target(Xte, w)) for (w, X) in draws]
            fh.write("%d %.5f %.5f\n" % (n, np.mean(e), max(1 - n / N2, 0.0)))
            print("kernel n=%d error %.3f (bound %.3f)" % (n, np.mean(e), max(1 - n / N2, 0)), flush=True)
    with open(os.path.join(DATA, "06-lazy-gap-net.dat"), "w") as fh:
        fh.write("n net\n")
        for n in N_NET:
            e = [net_error(X[:n], target(X[:n], w), Xte, target(Xte, w), s) for s, (w, X) in zip(SEEDS, draws)]
            fh.write("%d %.5f\n" % (n, np.mean(e)))
            print("network n=%d error %.3f" % (n, np.mean(e)), flush=True)

if __name__ == "__main__":
    main()
