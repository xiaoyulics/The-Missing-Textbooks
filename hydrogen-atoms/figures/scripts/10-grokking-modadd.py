"""Chapter 10 (grokking): a two-layer quadratic network learns modular addition.

Network: f_c(a, b) = sum_r V[c, r] * (U[r, a] + U'[r, b])^2, width m, inputs one-hot.
Task: c = a + b mod p, a random 30% of the p^2 pairs for training, the rest for testing.
Training: full-batch AdamW (lr 1e-3, betas (0.9, 0.98), decoupled weight decay 1.0),
cross-entropy loss, seed 0. About a minute on a laptop CPU with two threads.

Writes, in figures/data:
  10-grokking-modadd-curves.dat    step, train/test accuracy, Fourier concentration
  10-grokking-modadd-neurons.dat   per neuron: frequency, concentration, phase defect, norm
  10-grokking-modadd-spectrum.dat  share of first-layer energy at each frequency
and prints the normalized margin of the trained network next to the max-margin value.
"""
import os
import numpy as np
import torch

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
p, frac, m, steps = 59, 0.3, 256, 10000
lr, wd, seed = 1e-3, 1.0, 0
torch.manual_seed(seed)
np.random.seed(seed)

A, B = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
A, B = A.ravel(), B.ravel()
C = (A + B) % p
perm = np.random.permutation(p * p)
ntr = int(frac * p * p)
tr, te = perm[:ntr], perm[ntr:]
a_t, b_t, c_t = (torch.tensor(z) for z in (A, B, C))

U = torch.nn.Parameter(torch.randn(m, p) / np.sqrt(p))
Up = torch.nn.Parameter(torch.randn(m, p) / np.sqrt(p))
V = torch.nn.Parameter(torch.randn(p, m) / np.sqrt(m))
opt = torch.optim.AdamW([U, Up, V], lr=lr, weight_decay=wd, betas=(0.9, 0.98))


def logits(idx):
    return (V @ ((U[:, a_t[idx]] + Up[:, b_t[idx]]) ** 2)).T


def fourier_stats():
    """Dominant frequency, energy concentration and phase defect of every neuron."""
    u, up, v = U.detach().numpy(), Up.detach().numpy(), V.detach().numpy().T
    Fu, Fup, Fv = (np.fft.fft(z, axis=1) for z in (u, up, v))
    half = (p - 1) // 2
    E = np.abs(Fu[:, 1:half + 1]) ** 2 + np.abs(Fup[:, 1:half + 1]) ** 2
    tot = (np.abs(Fu) ** 2).sum(1) + (np.abs(Fup) ** 2).sum(1)
    j = E.argmax(1)
    conc = 2 * E.max(1) / tot                      # share of energy at +-zeta
    r = np.arange(m)
    defect = np.angle(Fu[r, j + 1]) + np.angle(Fup[r, j + 1]) - np.angle(Fv[r, j + 1])
    defect = (defect + np.pi) % (2 * np.pi) - np.pi  # chi - phi - psi, in (-pi, pi]
    norms = np.sqrt((u ** 2).sum(1) + (up ** 2).sum(1) + (v ** 2).sum(1))
    spec = E.sum(0) / E.sum()
    return j + 1, conc, defect, norms, spec


_, _, _, _, spec0 = fourier_stats()
evals = set(np.unique(np.round(np.logspace(0, np.log10(steps), 90)).astype(int)).tolist())
rows = []
for s in range(steps + 1):
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(logits(tr), c_t[tr])
    loss.backward()
    opt.step()
    if s in evals:
        with torch.no_grad():
            acc_tr = (logits(tr).argmax(1) == c_t[tr]).float().mean().item()
            acc_te = (logits(te).argmax(1) == c_t[te]).float().mean().item()
        _, conc, _, norms, _ = fourier_stats()
        wts = norms ** 3 / np.sum(norms ** 3)
        rows.append((s, acc_tr, acc_te, float(np.sum(wts * conc))))

with open(os.path.join(OUT, "10-grokking-modadd-curves.dat"), "w") as f:
    f.write("step train test conc\n")
    for r in rows:
        f.write("%d %.5f %.5f %.5f\n" % r)
zeta, conc, defect, norms, spec = fourier_stats()
with open(os.path.join(OUT, "10-grokking-modadd-neurons.dat"), "w") as f:
    f.write("neuron zeta conc defect norm relnorm\n")
    for r in range(m):
        f.write("%d %d %.5f %.5f %.5f %.5f\n" % (r, zeta[r], conc[r], defect[r], norms[r], norms[r] / norms.max()))
with open(os.path.join(OUT, "10-grokking-modadd-spectrum.dat"), "w") as f:
    f.write("zeta init final\n")
    for z in range(1, (p - 1) // 2 + 1):
        f.write("%d %.5f %.5f\n" % (z, spec0[z - 1], spec[z - 1]))

# summary numbers quoted in the text
first = lambda col, thr: next(r[0] for r in rows if r[col] >= thr)
big = norms > 0.3 * norms.max()
with torch.no_grad():
    F = logits(np.arange(p * p)).numpy()
correct = F[np.arange(p * p), C]
F2 = F.copy()
F2[np.arange(p * p), C] = -np.inf
margins = correct - F2.max(1)
n23 = np.sum(norms ** 3)
gamma_star = np.sqrt(2 / 27) / (np.sqrt(p) * (p - 1))
print("train acc >= 0.99 at step", first(1, 0.99), "; test acc >= 0.99 at step", first(2, 0.99))
print("large neurons (norm > 0.3 max): %d of %d; frequencies used: %d of %d; per frequency: min %d max %d"
      % (big.sum(), m, len(np.unique(zeta[big])), (p - 1) // 2,
         np.bincount(zeta[big], minlength=(p + 1) // 2)[1:].min(), np.bincount(zeta[big])[1:].max()))
print("concentration of large neurons: median %.3f; |phase defect|: median %.4f, max %.4f"
      % (np.median(conc[big]), np.median(np.abs(defect[big])), np.abs(defect[big]).max()))
print("margins on the full table: min/mean %.3f; normalized margin / gamma_star = %.3f"
      % (margins.min() / margins.mean(), margins.min() / n23 / gamma_star))
# share of the variance of the logits (centered over c) carried by the clocks, i.e. by the
# Fourier components at frequency vectors (k, k, -k); it equals 1 for every max-margin network
L = F.reshape(p, p, p)
L = L - L.mean(axis=2, keepdims=True)
E3 = np.abs(np.fft.fftn(L)) ** 2
k = np.arange(1, p)
print("share of logit variance in the clock components: %.3f" % (E3[k, k, (-k) % p].sum() / E3.sum()))
