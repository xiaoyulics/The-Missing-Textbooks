"""Chapter 8 (eos): gradient descent and Adam at their stability thresholds in a small network.

A fully connected tanh network 8 -> 32 -> 32 -> 1 (PyTorch default initialization, seed 1; 1377
parameters, float64) is trained full batch on the square loss (1/2n) sum_i (f(x_i) - y_i)^2 for
n = 256 Gaussian inputs x_i in R^8 (seed 0), labelled by a random teacher y = 3 a^T tanh(W x)
(W in R^{16 x 8} with N(0, 1/8) entries, a in R^16 with N(0, 1/16) entries; seed 0).

Runs, 4000 steps each:
  gradient descent with eta = 0.4 and eta = 0.25 (thresholds 2/eta = 5 and 8);
  Adam (beta1 = 0.9, beta2 = 0.999, eps = 1e-8, bias corrections) with eta = 0.01 and 0.003
  (thresholds 38/eta = 3800 and 12667).
Every 25 steps the sharpness S = lmax(H) is estimated with Hessian-vector products by block power
iteration (subspace iteration with three vectors and a Rayleigh-Ritz step: 6 iterations warm-started
from the previous vectors, 40 at the first evaluation; checked against the Lanczos method, relative
error below 3e-4 after the first evaluation). For Adam the preconditioned sharpness S_P = lmax(P^{-1} H)
of the preconditioner P = (1 - beta1^{t+1}) (sqrt(vhat) + eps) used in the update is estimated the
same way on P^{-1/2} H P^{-1/2}, and the sharpness of Adam's iterates every 100 steps. If the dominant
Ritz value is negative, the iteration is repeated on the shifted operator, so that the result is the
largest eigenvalue.
Outputs: figures/data/08-eos-network.dat with columns
  t  gdA gdB  lgdA lgdB  adA adB  ladA ladB
(gd = sharpness under gradient descent, ad = preconditioned sharpness under Adam, l = training loss;
A, B = the larger and the smaller step size), and figures/data/08-eos-network-raw.dat with columns
  t rawA rawB (the sharpness S along the two Adam runs).
One thread (the tensors are tiny, and a second thread only adds overhead); about 30 seconds on a
laptop CPU.
"""
import math
import os

import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

n, d, m = 256, 8, 32
torch.manual_seed(0)
X = torch.randn(n, d)
W_teacher = torch.randn(d, 16) / math.sqrt(d)
a_teacher = torch.randn(16, 1) / math.sqrt(16)
Y = 3.0 * torch.tanh(X @ W_teacher) @ a_teacher
STEPS, EVERY, RAW_EVERY = 4000, 25, 100
K, ITERS, COLD_ITERS = 3, 6, 40


def make_model():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(d, m), torch.nn.Tanh(), torch.nn.Linear(m, m),
                               torch.nn.Tanh(), torch.nn.Linear(m, 1))


def loss_fn(model):
    return 0.5 * ((model(X) - Y) ** 2).mean()


def top_eigenvalue(model, scale=None, V=None):
    """Largest eigenvalue of D H D (D = diag(scale), or the identity), by block power iteration."""
    params = list(model.parameters())
    grads = torch.autograd.grad(loss_fn(model), params, create_graph=True)
    sizes = [p.numel() for p in params]

    def matvec(vec):
        u = vec if scale is None else scale * vec
        parts = torch.split(u, sizes)
        hv = torch.autograd.grad(grads, params, grad_outputs=[q.reshape(p.shape) for q, p in zip(parts, params)],
                                 retain_graph=True)
        hv = torch.cat([h.reshape(-1) for h in hv])
        return hv if scale is None else scale * hv

    iters = ITERS
    if V is None:
        V = torch.randn(sum(sizes), K, generator=torch.Generator().manual_seed(2))
        iters = COLD_ITERS
    shift = 0.0
    for _ in range(2):
        Q, _ = torch.linalg.qr(V)
        for _ in range(iters):
            Q, _ = torch.linalg.qr(torch.stack([matvec(Q[:, j]) for j in range(K)], 1) - shift * Q)
        T = Q.T @ torch.stack([matvec(Q[:, j]) for j in range(K)], 1)
        evals, evecs = torch.linalg.eigh(0.5 * (T + T.T))
        V = Q @ evecs.flip(1)
        if evals[-1] >= -evals[0]:
            break
        shift, iters = float(evals[0]), COLD_ITERS  # the dominant Ritz value was negative
    return float(evals[-1]), V


def run_gd(eta):
    model = make_model()
    sharp, losses, V = [], [], None
    for t in range(STEPS + 1):
        if t % EVERY == 0:
            lam, V = top_eigenvalue(model, V=V)
            sharp.append(lam)
            with torch.no_grad():
                losses.append(loss_fn(model).item())
        model.zero_grad()
        loss_fn(model).backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= eta * p.grad
    return sharp, losses


def run_adam(eta, b1=0.9, b2=0.999, eps=1e-8):
    model = make_model()
    params = list(model.parameters())
    ms = [torch.zeros_like(p) for p in params]
    vs = [torch.zeros_like(p) for p in params]
    pre, raw, losses, vp, vr = [], [], [], None, None
    for t in range(STEPS + 1):
        model.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        with torch.no_grad():
            for p, mm, vv in zip(params, ms, vs):
                mm.mul_(b1).add_(p.grad, alpha=1 - b1)
                vv.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
        if t % EVERY == 0:
            P = torch.cat([((1 - b1 ** (t + 1)) * ((vv / (1 - b2 ** (t + 1))).sqrt() + eps)).reshape(-1) for vv in vs])
            lam_p, vp = top_eigenvalue(model, scale=P.rsqrt(), V=vp)
            pre.append(lam_p)
            losses.append(loss.item())
        if t % RAW_EVERY == 0:
            lam_r, vr = top_eigenvalue(model, V=vr)
            raw.append(lam_r)
        with torch.no_grad():
            for p, mm, vv in zip(params, ms, vs):
                p -= eta * (mm / (1 - b1 ** (t + 1))) / ((vv / (1 - b2 ** (t + 1))).sqrt() + eps)
    return pre, raw, losses


def main():
    gdA, lgdA = run_gd(0.4)
    gdB, lgdB = run_gd(0.25)
    adA, rawA, ladA = run_adam(0.01)
    adB, rawB, ladB = run_adam(0.003)
    with open(os.path.join(DATA, "08-eos-network.dat"), "w") as f:
        f.write("t gdA gdB lgdA lgdB adA adB ladA ladB\n")
        for i in range(len(gdA)):
            f.write(f"{i * EVERY} {gdA[i]:.4f} {gdB[i]:.4f} {lgdA[i]:.4e} {lgdB[i]:.4e} "
                    f"{adA[i]:.2f} {adB[i]:.2f} {ladA[i]:.4e} {ladB[i]:.4e}\n")
    with open(os.path.join(DATA, "08-eos-network-raw.dat"), "w") as f:
        f.write("t rawA rawB\n")
        for i in range(len(rawA)):
            f.write(f"{i * RAW_EVERY} {rawA[i]:.4f} {rawB[i]:.4f}\n")


if __name__ == "__main__":
    main()
