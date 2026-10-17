"""Chapter 12, figure fig:collapse:network.

A ReLU network trained past zero training error on a synthetic Gaussian
mixture: K = 4 classes, input dimension d = 32, class means drawn N(0, I_d/4)
(seed 0), unit isotropic noise, n = 64 training examples per class, and 2000
test examples.  Network 32 -> 256 -> 256 -> m = 32 -> K, ReLU, last layer with
bias.  Full-batch gradient descent with momentum 0.9, step 0.05, weight decay
5e-4 on all parameters, cross-entropy loss, 6000 steps (seed 0, CPU, 2 threads).
Writes figures/data/12-collapse-network.dat with the collapse metrics of the
penultimate features h (training set unless marked test):
  step     iteration
  trainerr training error,  testerr test error
  nc1      tr(Sigma_W Sigma_B^+)/K, within-class variability
  nc1test  the same on the test set
  nc2      || G/||G||_F - (I-J/K)/||I-J/K||_F ||_F with G the Gram of centered means
  nc3      || W^T/||W||_F - M/||M||_F ||_F with M the centered means
  nc4      fraction of training points where argmax_k <w_k,h>+b_k differs
           from the nearest centered class mean
"""
import os
import numpy as np
import torch

torch.set_num_threads(2)
torch.manual_seed(0)
K, d, n, ntest, m = 4, 32, 64, 2000, 32
g = torch.Generator().manual_seed(0)
mus = 0.5 * torch.randn(K, d, generator=g)
ytr = torch.arange(K).repeat_interleave(n)
yte = torch.randint(0, K, (ntest,), generator=g)
Xtr = mus[ytr] + torch.randn(K * n, d, generator=g)
Xte = mus[yte] + torch.randn(ntest, d, generator=g)

body = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(),
                           torch.nn.Linear(256, 256), torch.nn.ReLU(),
                           torch.nn.Linear(256, m), torch.nn.ReLU())
head = torch.nn.Linear(m, K)
params = list(body.parameters()) + list(head.parameters())
opt = torch.optim.SGD(params, lr=0.05, momentum=0.9, weight_decay=5e-3)


def nc1(H, y):
    mu_g = H.mean(0)
    M = torch.stack([H[y == k].mean(0) for k in range(K)]) - mu_g
    SB = M.T @ M / K
    SW = sum(((H[y == k] - H[y == k].mean(0)).T @ (H[y == k] - H[y == k].mean(0)))
             for k in range(K)) / H.shape[0]
    return (torch.trace(SW @ torch.linalg.pinv(SB, hermitian=True)) / K).item(), M, mu_g


def metrics():
    with torch.no_grad():
        H = body(Xtr)
        Ht = body(Xte)
        a, M, mu_g = nc1(H, ytr)
        at, _, _ = nc1(Ht, yte)
        G = M @ M.T
        E = torch.eye(K) - torch.ones(K, K) / K
        b2 = torch.linalg.norm(G / torch.linalg.norm(G) - E / torch.linalg.norm(E)).item()
        Wt = head.weight
        b3 = torch.linalg.norm(Wt / torch.linalg.norm(Wt) - M / torch.linalg.norm(M)).item()
        pred = (H @ Wt.T + head.bias).argmax(1)
        ncc = torch.cdist(H - mu_g, M).argmin(1)
        b4 = (pred != ncc).float().mean().item()
        tr = (pred != ytr).float().mean().item()
        te = ((Ht @ Wt.T + head.bias).argmax(1) != yte).float().mean().item()
    return tr, te, a, at, b2, b3, b4


rows = []
steps = 20000
checkpoints = sorted(set(np.unique(np.round(np.logspace(0, np.log10(steps), 40)).astype(int))))
for t in range(1, steps + 1):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(head(body(Xtr)), ytr)
    loss.backward()
    opt.step()
    if t in checkpoints:
        rows.append((t,) + metrics())
        print("%5d loss %.2e  train %.3f test %.3f nc1 %.2e nc1test %.2e nc2 %.3f nc3 %.3f nc4 %.3f"
              % ((t, loss.item()) + rows[-1][1:]))
out = os.path.join(os.path.dirname(__file__), "..", "data", "12-collapse-network.dat")
with open(out, "w") as f:
    f.write("step trainerr testerr nc1 nc1test nc2 nc3 nc4\n")
    for r in rows:
        f.write("%d %.5f %.5f %.5e %.5e %.5e %.5e %.5f\n" % r)
