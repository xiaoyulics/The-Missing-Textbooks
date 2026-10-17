"""Chapter 14: a two-layer attention-only transformer on in-context Markov chains.

K=3 states, Dirichlet(1) rows, uniform first token, context N=32, loss averaged over all
positions. Layer 1: one head whose logits are learned relative-position biases r[t-s] and
whose values are V1 e_x (output appended to the residual stream). Layer 2: one head with a
bilinear score on [e_x, h] and a sink slot with learned logit sigma and uniform value; its
output is a probability vector q, and the logits are gamma*log(q) (gamma learned, a stand-in
for the unembedding and MLP). Plain SGD, learning rate 1, batch 64, three initialization
scales. Baselines on the validation set: uniform (log 3), the best add-k in-context unigram
estimator, and the Bayes predictor of the chapter (in-context bigram, add-1).
Writes figures/data/14-induction-transformer.dat. Runs in about 40 s.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)
out = os.path.join(os.path.dirname(__file__), "..", "data")
K, N, B, STEPS, LR, EVERY = 3, 32, 64, 3000, 1.0, 20
E = torch.eye(K); idx = torch.arange(N)
rel = (idx[:, None] - idx[None, :]).clamp(min=0); mask = idx[:, None] >= idx[None, :]

def sample(rng, n):
    P = rng.dirichlet(np.ones(K), size=(n, K)); x = np.zeros((n, N + 1), int)
    x[:, 0] = rng.integers(K, size=n); u = rng.random((n, N + 1)); ar = np.arange(n)
    for t in range(1, N + 1):
        x[:, t] = (u[:, t, None] > np.cumsum(P[ar, x[:, t - 1]], 1)).sum(1)
    return x

xv_np = sample(np.random.default_rng(123), 4000); xv = torch.tensor(xv_np)
# baselines
ar = np.arange(len(xv_np)); cnt = np.zeros((len(xv_np), K, K)); n = np.zeros((len(xv_np), K))
Lb = []; Lu = {k: [] for k in [0.5, 1, 1.5, 2, 3, 4]}
for t in range(N):
    n[ar, xv_np[:, t]] += 1
    if t > 0: cnt[ar, xv_np[:, t - 1], xv_np[:, t]] += 1
    i, j = xv_np[:, t], xv_np[:, t + 1]
    Lb.append(-np.log((cnt[ar, i, j] + 1) / (cnt[ar, i].sum(1) + K)).mean())
    for k in Lu: Lu[k].append(-np.log((n[ar, j] + k) / (t + 1 + K * k)).mean())
Lbig = np.mean(Lb); Luni = min(np.mean(v) for v in Lu.values())
print(f"uniform {np.log(K):.4f}  unigram {Luni:.4f}  bigram Bayes {Lbig:.4f}")

def run(init, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    p = lambda *s: torch.nn.Parameter(init * torch.randn(*s))
    r, V1, W2 = p(N), p(K, K), p(2 * K, 2 * K)
    sig = torch.nn.Parameter(torch.zeros(1)); gam = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([r, V1, W2, sig, gam], lr=LR)
    def fwd(x):
        e = E[x]; a1 = r[rel].masked_fill(~mask, -1e9).softmax(-1)
        z = torch.cat([e, a1 @ (e @ V1)], -1)
        s = (z @ W2 @ z.transpose(1, 2)).masked_fill(~mask, -1e9)
        s = torch.cat([sig.expand(x.shape[0], N, 1), s], -1).softmax(-1)
        return gam * torch.log(s[:, :, 1:] @ e + s[:, :, :1] / K), a1
    hist = []
    for st in range(STEPS + 1):
        if st % EVERY == 0:
            with torch.no_grad():
                lo, a1 = fwd(xv[:, :N])
                hist.append((torch.nn.functional.cross_entropy(lo.reshape(-1, K), xv[:, 1:].reshape(-1)).item(),
                             a1[N - 1, N - 2].item()))
        x = torch.tensor(sample(rng, B)); lo, _ = fwd(x[:, :N])
        L = torch.nn.functional.cross_entropy(lo.reshape(-1, K), x[:, 1:].reshape(-1))
        opt.zero_grad(); L.backward(); opt.step()
    return np.array(hist)

runs = {init: run(init, 0) for init in [1e-1, 1e-2, 1e-3]}
with open(os.path.join(out, "14-induction-transformer.dat"), "w") as f:
    f.write("step l1 l2 l3 prev1 prev2 prev3 uniform unigram bigram\n")
    for k in range(STEPS // EVERY + 1):
        f.write(f"{k*EVERY} " + " ".join(f"{runs[i][k,0]:.5f}" for i in runs) + " "
                + " ".join(f"{runs[i][k,1]:.4f}" for i in runs)
                + f" {np.log(K):.5f} {Luni:.5f} {Lbig:.5f}\n")
for i, h in runs.items():
    mid = (Luni + Lbig) / 2; k = np.argmax(h[:, 0] < mid)
    print(f"init {i:g}: loss below midpoint at step {k*EVERY}; final {h[-1,0]:.4f}; prev-token weight {h[-1,1]:.3f}")
