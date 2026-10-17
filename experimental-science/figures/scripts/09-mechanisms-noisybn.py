"""Lab 9.1 of Deep Learning as an Experimental Science: knocking out the mediator.

A six-layer ReLU perceptron on the digits is trained in a 2 x 2 factorial design,
batch normalization (off/on) x injected covariate shift (off/on), 20 seeds per cell, all
at learning rate 0.3 with plain SGD; seeds are shared across arms. The shift is the
"noisy BatchNorm" of Santurkar et al. (2018, Algorithm 1): after each normalization (or,
without normalization, after each linear layer) every unit's pre-activation a is replaced
by s * a + m, where at every step each unit draws a shift mu ~ U(-0.5, 0.5) and a scale
sigma ~ U(1, 1.25), and every example draws m ~ U(mu - 0.1, mu + 0.1) and s ~ N(sigma, 0.1^2).
A dose d multiplies the four noise parameters n_mu = 0.5, n_sigma - 1 = 0.25, r_mu = 0.1 and
r_sigma = 0.1 (Santurkar et al. do not say whether r_sigma is a variance or a standard
deviation; we read it as a standard deviation); the batch-normalized network
is also trained at doses 1/8, 1/2, 2 and 4 (8 seeds each). A fifth arm (20 seeds) uses
batch normalization whose batch statistics are held constant in the backward pass: the
forward pass, and hence the stability of the distributions, is that of batch
normalization, but the gradient treats the batch mean and variance as constants.

For every run we record the training loss and test error every 50 steps (with batch
statistics recomputed on the training set and the noise switched off) and three measures
of covariate shift, on a fixed probe batch of 256 training images:
  shift_tot  the change, from one step to the next, of the distribution of each unit's
             input to its nonlinearity (Wasserstein-1 distance over the probe batch, in
             units of the unit's standard deviation; mean over units and layers), with the
             noise of each step: the shift a layer experiences;
  shift_upd  the same with the noise held fixed: the shift caused by the parameter update;
  grad_cos   the cosine between the gradient of a layer's weights before and after the
             update of all layers below it (Santurkar et al., Definition 2.1).
Writes figures/data/09-mechanisms-noisybn-*.dat. About four minutes with two threads.
"""
import copy
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.func import functional_call  # noqa: E402

WIDTH, DEPTH, BATCH, STEPS, EVAL_EVERY = 64, 6, 64, 600, 50
SHIFT_EVERY, GRAD_EVERY, PROBE = 30, 60, 256


class Shift(nn.Module):
    """Injected covariate shift (Santurkar et al. 2018, Algorithm 1), active in training only.

    `resample()` draws the per-unit shift and scale of a new step; each forward pass draws
    the per-example noise from the run's own generator, or from `ex_gen` when it is set
    (for probes, which must not disturb the run), or replays the last draw (`replay`)."""

    def __init__(self, units, seed, dose=1.0, n_mu=0.5, n_sigma=1.25, r_mu=0.1, r_sigma=0.1):
        super().__init__()
        self.units, self.gen, self.ex_gen, self.on = units, torch.Generator().manual_seed(seed), None, True
        self.replay, self.last = False, None
        self.n_mu, self.n_sigma = n_mu * dose, 1.0 + (n_sigma - 1.0) * dose
        self.r_mu, self.r_sigma = r_mu * dose, r_sigma * dose
        self.resample()

    def resample(self):
        g, J = self.gen, self.units
        self.mu = (2 * torch.rand(J, generator=g) - 1) * self.n_mu
        self.sigma = 1.0 + torch.rand(J, generator=g) * (self.n_sigma - 1.0)

    def forward(self, a):
        if not (self.training and self.on):
            return a
        if self.replay:
            m, s = self.last
        else:
            g = self.ex_gen if self.ex_gen is not None else self.gen
            n, J = a.shape
            m = self.mu + (2 * torch.rand(n, J, generator=g) - 1) * self.r_mu
            s = self.sigma + self.r_sigma * torch.randn(n, J, generator=g)
            if self.ex_gen is None:
                self.last = (m, s)
        return s * a + m


class DetachedBN(nn.BatchNorm1d):
    """Batch normalization whose batch mean and variance are treated as constants in the
    backward pass: the forward pass is that of nn.BatchNorm1d, the gradient is not."""

    def forward(self, x):
        if not self.training:
            return super().forward(x)
        mu, var = x.mean(0), x.var(0, unbiased=False)
        with torch.no_grad():
            n, mom = x.shape[0], self.momentum
            self.running_mean.mul_(1 - mom).add_(mom * mu)
            self.running_var.mul_(1 - mom).add_(mom * var * n / (n - 1))
        return self.weight * (x - mu.detach()) / torch.sqrt(var.detach() + self.eps) + self.bias


class Net(nn.Module):
    """DEPTH hidden layers of WIDTH ReLU units: Linear -> [BatchNorm] -> [Shift] -> ReLU.
    He-normal weights and zero biases in hidden layers; output weights N(0, 1/WIDTH).
    `bn` is False, True, or "detached" (statistics outside the gradient)."""

    def __init__(self, bn, noise, dose=1.0, init=0, noise_seed=0):
        super().__init__()
        torch.manual_seed(50_000 + init)
        self.bn = bool(bn)
        norm = DetachedBN if bn == "detached" else nn.BatchNorm1d
        self.lins, self.norms, self.shifts = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        d = 64
        for layer in range(DEPTH):
            lin = nn.Linear(d, WIDTH)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            self.lins.append(lin)
            self.norms.append(norm(WIDTH) if bn else nn.Identity())
            self.shifts.append(Shift(WIDTH, 70_000 + 1000 * noise_seed + layer, dose) if noise else nn.Identity())
            d = WIDTH
        self.out = nn.Linear(d, 10)
        nn.init.normal_(self.out.weight, std=1 / math.sqrt(d))
        nn.init.zeros_(self.out.bias)

    def forward(self, x, record=None):
        h = x
        for lin, norm, shift in zip(self.lins, self.norms, self.shifts):
            a = shift(norm(lin(h)))
            if record is not None:
                record.append(a.detach())
            h = F.relu(a)
        return self.out(h)

    def _shift_modules(self):
        return [s for s in self.shifts if isinstance(s, Shift)]

    def resample(self):
        for s in self._shift_modules():
            s.resample()

    def replay(self, flag):
        for s in self._shift_modules():
            s.replay = flag

    def share_noise(self, seed):
        """Draw per-example noise from fresh generators seeded `seed + layer` (None: the run's own)."""
        for layer, s in enumerate(self.shifts):
            if isinstance(s, Shift):
                s.ex_gen = None if seed is None else torch.Generator().manual_seed(seed + layer)

    def freeze_stats(self, frozen):
        for m in self.norms:
            if isinstance(m, nn.BatchNorm1d):
                m.momentum = 0.0 if frozen else 0.1


def w1_shift(A, B):
    """Mean over units of W1(A_j, B_j) / sd(A_j) for samples A, B of shape (n, units);
    A may be given as the pair (sorted A, sd of A)."""
    sa, sd = A if isinstance(A, tuple) else (torch.sort(A, 0)[0], A.std(0))
    sb, _ = torch.sort(B, 0)
    ok = sd > 1e-6
    return ((sa - sb).abs().mean(0)[ok] / sd[ok]).mean().item()


def presort(rec):
    return [(torch.sort(a, 0)[0], a.std(0)) for a in rec]


def probe(model, X, seed):
    """Inputs to every nonlinearity on the probe batch, in training mode, without
    touching the running statistics; per-example noise from `seed`."""
    model.freeze_stats(True)
    model.share_noise(seed)
    rec = []
    with torch.no_grad():
        model(X, record=rec)
    model.share_noise(None)
    model.freeze_stats(False)
    return rec


def grad_cosines(model, xb, yb, old, new):
    """Santurkar et al.'s Definition 2.1 with the injected noise held fixed: for layers
    i = 2..DEPTH+1, the cosine between the gradient of W_i at the old parameters and at
    the parameters in which the layers below i have been updated."""
    model.freeze_stats(True)
    model.replay(True)
    names = [f"lins.{i}.weight" for i in range(DEPTH)] + ["out.weight"]

    def layer_of(key):
        part = key.split(".")
        return int(part[1]) if part[0] in ("lins", "norms") else DEPTH

    def grads(params, wanted):
        p = {k: (v.clone().requires_grad_(True) if k in wanted else v) for k, v in params.items()}
        loss = F.cross_entropy(functional_call(model, p, (xb,)), yb)
        return torch.autograd.grad(loss, [p[k] for k in wanted])

    G = grads(old, names[1:])
    cos = []
    for i in range(1, DEPTH + 1):
        mix = {k: (new[k] if layer_of(k) < i else old[k]) for k in old}
        (Gi,) = grads(mix, [names[i]])
        cos.append(F.cosine_similarity(G[i - 1].flatten(), Gi.flatten(), 0).item())
    model.replay(False)
    model.freeze_stats(False)
    return float(np.mean(cos))


@torch.no_grad()
def evaluate(model, data):
    """Training loss and test error with the noise off and, for normalized networks, the
    batch statistics recomputed on the whole training set."""
    Xtr, ytr, Xte, yte = data
    m = copy.deepcopy(model)
    for s in m._shift_modules():
        s.on = False
    if m.bn:
        for b in m.norms:
            b.momentum = 1.0
        m.train()
        m(Xtr)
    m.eval()
    return F.cross_entropy(m(Xtr), ytr).item(), (m(Xte).argmax(1) != yte).float().mean().item()


def train(bn, noise, lr, seed, data, dose=1.0, grad_ics=True, trace=False):
    """One run. Returns None if the loss diverges."""
    t0 = time.time()
    Xtr, ytr, _, _ = data
    model = Net(bn, noise, dose, init=seed, noise_seed=seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    g = torch.Generator().manual_seed(60_000 + seed)
    perm, pos = torch.randperm(n, generator=g), 0
    Xp = Xtr[:PROBE]
    res = {"loss": [], "err": [], "tot": [], "upd": [], "cos": [], "trace": []}
    for step in range(STEPS):
        if pos + BATCH > n:
            perm, pos = torch.randperm(n, generator=g), 0
        idx = perm[pos:pos + BATCH]
        pos += BATCH
        xb, yb = Xtr[idx], ytr[idx]
        m_shift, m_grad = step % SHIFT_EVERY == 0, grad_ics and step % GRAD_EVERY == 0
        if m_shift:
            A = probe(model, Xp, 123)
            As = presort(A)
            if trace:
                a = A[-1][:, 0]
                res["trace"].append([step] + [torch.quantile(a, q).item() for q in (0.15, 0.5, 0.85)])
        if m_grad:
            old = {k: v.detach().clone() for k, v in model.named_parameters()}
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if not math.isfinite(loss.item()):
            return None
        if m_grad:
            new = {k: v.detach().clone() for k, v in model.named_parameters()}
            res["cos"].append(grad_cosines(model, xb, yb, old, new))
        if m_shift:
            B = probe(model, Xp, 123)            # same noise: the update alone
        model.resample()                         # the noise of the next step
        if m_shift:
            C = probe(model, Xp, 456)            # new noise as well
            res["upd"].append(np.mean([w1_shift(a, b) for a, b in zip(As, B)]))
            res["tot"].append(np.mean([w1_shift(a, c) for a, c in zip(As, C)]))
        if (step + 1) % EVAL_EVERY == 0:
            tl, te = evaluate(model, data)
            if not math.isfinite(tl):
                return None
            res["loss"].append(tl)
            res["err"].append(te)
    res["seconds"] = time.time() - t0
    return res


def summary(r):
    return dict(meanloss=float(np.mean(r["loss"])), final=r["loss"][-1], err=100 * r["err"][-1],
                tot=float(np.mean(r["tot"])), upd=float(np.mean(r["upd"])),
                cos=float(np.mean(r["cos"])) if r["cos"] else float("nan"))


def geo_band(rows):
    """Geometric mean of positive values across runs, with a 95% t-interval on the log scale."""
    L = np.log10(np.asarray(rows))
    out = []
    for j in range(L.shape[1]):
        m, h = lk.t_interval(L[:, j])
        out.append((10 ** m, 10 ** (m - h), 10 ** (m + h)))
    return np.array(out)


def main():
    t_start = time.time()
    data = lk.load(lk.Config())
    lr, n_main, n_dose = 0.3, 20, 8
    arms = [("plain", False, False), ("noisyplain", False, True), ("bn", True, False), ("noisybn", True, True),
            ("detached", "detached", False)]
    runs = {}
    for name, bn, noise in arms:
        runs[name] = [train(bn, noise, lr, s, data, grad_ics=(name != "detached"), trace=(s == 0))
                      for s in range(n_main)]
        ok = [r for r in runs[name] if r is not None]
        print(f"{name:10s} {len(ok)}/{n_main} runs trained, {np.mean([r['seconds'] for r in ok]):.2f} s per run", flush=True)
    doses = [0.125, 0.5, 2.0, 4.0]
    dose_runs = {d: [train(True, True, lr, s, data, dose=d, grad_ics=False) for s in range(n_dose)] for d in doses}
    print(f"dose arms done, {time.time() - t_start:.0f} s so far", flush=True)

    # ---- per-arm summaries --------------------------------------------------------------
    S = {k: [summary(r) if r is not None else None for r in v] for k, v in runs.items()}
    steps = np.arange(EVAL_EVERY, STEPS + 1, EVAL_EVERY)
    curves = {"step": steps}
    for name in runs:
        ok = [r for r in runs[name] if r is not None]
        band = geo_band([r["loss"] for r in ok])
        curves[f"{name}"], curves[f"{name}lo"], curves[f"{name}hi"] = band[:, 0], band[:, 1], band[:, 2]
    lk.write_dat("figures/data/09-mechanisms-noisybn-curves.dat", curves)

    xpos = {"plain": 1, "noisyplain": 2, "bn": 3, "noisybn": 4, "detached": 5}
    rng = np.random.default_rng(0)
    cols = {k: [] for k in ("x", "seed", "err", "meanloss", "tot", "upd", "cos")}
    means = {k: [] for k in ("x", "n", "err", "errhw", "meanloss", "meanlosshw", "tot", "tothw", "upd", "updhw",
                             "cos", "coshw")}
    print("\nper arm: mean and 95% t-interval half-width over trained runs")
    for name in runs:
        ok = [s for s in S[name] if s is not None]
        for seed, s in enumerate(S[name]):
            if s is None:
                continue
            cols["x"].append(xpos[name] + rng.uniform(-0.18, 0.18))
            cols["seed"].append(seed)
            for k in ("err", "meanloss", "tot", "upd", "cos"):
                cols[k].append(s[k])
        means["x"].append(xpos[name])
        means["n"].append(len(ok))
        line = f"  {name:10s} n={len(ok):2d}"
        for k in ("err", "meanloss", "tot", "upd", "cos"):
            m, h = lk.t_interval([s[k] for s in ok]) if k != "cos" or name != "detached" else (float("nan"),) * 2
            means[k].append(m)
            means[k + "hw"].append(h)
            line += f" | {k} {m:.4f} +- {h:.4f}"
        line += f" | final loss {np.mean([s['final'] for s in ok]):.5f}"
        print(line)
    lk.write_dat("figures/data/09-mechanisms-noisybn-runs.dat", cols)
    lk.write_dat("figures/data/09-mechanisms-noisybn-means.dat", means)

    # ---- paired comparisons (same seeds in every arm) -----------------------------------
    def paired(a, b, key, A=S, idx=range(n_main)):
        d = [A[a][i][key] - A[b][i][key] for i in idx if A[a][i] is not None and A[b][i] is not None]
        m, h = lk.t_interval(d)
        return m, h, len(d), sum(x > 0 for x in d)

    print("\npaired differences (first minus second), mean +- 95% half-width, n, number positive")
    for a, b in [("plain", "bn"), ("noisybn", "bn"), ("plain", "noisybn"), ("noisyplain", "plain"),
                 ("noisyplain", "noisybn"), ("detached", "bn"), ("detached", "plain")]:
        for key in ("meanloss", "err", "tot", "upd", "cos"):
            if key == "cos" and "detached" in (a, b):
                continue
            m, h, n, pos = paired(a, b, key)
            print(f"  {a:10s} - {b:8s} {key:8s} {m:+.5f} +- {h:.5f} (n={n}, positive {pos})")
    te = paired("plain", "bn", "meanloss")
    kb = paired("noisybn", "bn", "meanloss")
    print(f"\nknockout at dose 1: total effect {te[0]:.5f} +- {te[1]:.5f}; bound on indirect effect "
          f"{kb[0]:.5f} +- {kb[1]:.5f}; ratio {kb[0] / te[0]:.3f}")
    for key in ("tot", "upd"):
        dom = sum(S["noisybn"][i][key] > S["plain"][i][key] for i in range(n_main))
        print(f"dominance, seed by seed: shift_{key}(noisy BN) > shift_{key}(plain) on {dom} of {n_main} seeds")
    rt = np.mean([S["noisybn"][i]["tot"] / S["plain"][i]["tot"] for i in range(n_main)])
    print(f"mean ratio shift_tot(noisy BN) / shift_tot(plain) = {rt:.1f}")

    # ---- dose response -------------------------------------------------------------------
    D = {d: [summary(r) if r is not None else None for r in rr] for d, rr in dose_runs.items()}
    D[0.0], D[1.0] = S["bn"], S["noisybn"]
    D["plain"] = S["plain"]
    dose_tab = {k: [] for k in ("dose", "n", "tot", "tothw", "meanloss", "meanlosshw", "err", "errhw")}
    print(f"\ndose response (batch-normalized network with injected shift); bounds paired with BN on seeds 0-{n_dose - 1}")
    te10 = paired("plain", 0.0, "meanloss", A=D, idx=range(n_dose))
    print(f"  total effect on seeds 0-{n_dose - 1}: {te10[0]:.5f} +- {te10[1]:.5f}")
    for d in sorted(k for k in D if k != "plain"):
        ok = [s for s in D[d] if s is not None]
        dose_tab["dose"].append(d)
        dose_tab["n"].append(len(ok))
        line = f"  dose {d:6.3f} trained {len(ok)}/{len(D[d])}"
        for k in ("tot", "meanloss", "err"):
            m, h = lk.t_interval([s[k] for s in ok])
            dose_tab[k].append(m)
            dose_tab[k + "hw"].append(h)
            line += f" | {k} {m:.4f} +- {h:.4f}"
        if d > 0:
            b = paired(d, 0.0, "meanloss", A=D, idx=range(n_dose))
            dom = sum(D[d][i]["tot"] > D["plain"][i]["tot"] for i in range(n_dose) if D[d][i] is not None)
            line += f" | bound {b[0]:.5f} +- {b[1]:.5f} ({b[0] / te10[0]:.3f} of total) | dominates on {dom}/{n_dose}"
        print(line)
    lk.write_dat("figures/data/09-mechanisms-noisybn-dose.dat", dose_tab)

    # ---- the trace of one unit (seed 0), as in Ioffe and Szegedy's Figure 1 -------------
    tr = {"step": [row[0] for row in runs["plain"][0]["trace"]]}
    for name in ("plain", "bn", "noisybn"):
        T = np.array(runs[name][0]["trace"])
        tr[f"{name}15"], tr[f"{name}50"], tr[f"{name}85"] = T[:, 1], T[:, 2], T[:, 3]
        jumps = np.abs(np.diff(T[:, 2])).mean()
        print(f"trace {name}: median from {T[0, 2]:.3f} to {T[-1, 2]:.3f}, 15-85 range from "
              f"{T[0, 3] - T[0, 1]:.3f} to {T[-1, 3] - T[-1, 1]:.3f}, mean jump of the median {jumps:.3f}")
    lk.write_dat("figures/data/09-mechanisms-noisybn-trace.dat", tr)
    print(f"\ntotal time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
