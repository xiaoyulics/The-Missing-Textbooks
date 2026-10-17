"""labkit: the shared laboratory kit of *Deep Learning as an Experimental Science*
(The Missing Textbooks, volume 4).

Every lab in the book runs on a laptop CPU in minutes, with numpy, scipy, scikit-learn and
torch only. The unit of experiment is a *run*: one network trained once. A run is a
function of its configuration and of four separate seeds,

    init   the initial weights,
    order  the order in which minibatches are drawn,
    split  which examples are in the training and test sets,
    noise  label noise, dropout masks and augmentation,

so that each source of randomness can be varied while the others are held fixed.

Two datasets are built in. `digits` is the 8x8 handwritten-digit data shipped with
scikit-learn (1797 images, 10 classes): no download is needed. `teacher` is a synthetic
regression task, labels produced by a random two-layer network plus Gaussian noise, whose
size can be varied over several decades for scaling experiments.

Usage from a script in figures/scripts/:

    import labkit as lk
    cfg = lk.Config(width=64, optimizer="adam", lr=1e-3, epochs=30)
    r = lk.run(cfg, init=0, order=0)            # dict of measurements
    lo, hi = lk.t_interval([...])               # statistics helpers
    lk.write_dat("figures/data/01-seeds-spread.dat", {"seed": s, "acc": a})

Conventions: all accuracies and errors are fractions in [0, 1]; losses are mean
cross-entropy (classification) or mean squared error (regression).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, replace, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "data"))

# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------
_DIGITS = None


def _digits_raw():
    global _DIGITS
    if _DIGITS is None:
        from sklearn.datasets import load_digits
        d = load_digits()
        _DIGITS = (d.data.astype(np.float32) / 16.0, d.target.astype(np.int64))
    return _DIGITS


def digits(split=0, test_frac=0.3, train_size=None, label_noise=0.0, noise=0):
    """The scikit-learn digits, split into train and test by the seed `split`.

    Inputs are scaled to [0, 1] and then standardized with the training-set mean and
    standard deviation of each pixel. `train_size` subsamples the training set;
    `label_noise` replaces that fraction of training labels by uniformly random labels
    (seed `noise`). Returns float32/int64 tensors (Xtr, ytr, Xte, yte).
    """
    X, y = _digits_raw()
    rng = np.random.default_rng(10_000 + split)
    perm = rng.permutation(len(y))
    n_test = int(round(test_frac * len(y)))
    te, tr = perm[:n_test], perm[n_test:]
    if train_size is not None:
        tr = tr[:train_size]
    Xtr, ytr, Xte, yte = X[tr], y[tr].copy(), X[te], y[te]
    if label_noise > 0:
        nrng = np.random.default_rng(20_000 + noise)
        flip = nrng.random(len(ytr)) < label_noise
        ytr[flip] = nrng.integers(0, 10, size=flip.sum())
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-3
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    return t(Xtr), t(ytr), t(Xte), t(yte)


def teacher(n_train, n_test=4096, d=8, width=32, noise_sd=0.1, task=0, split=0):
    """A synthetic regression task: y = f*(x) + noise, with x ~ N(0, I_d) and f* a random
    two-layer tanh network of the given width (seed `task`), scaled so that f*(x) has unit
    variance. The irreducible test loss (mean squared error) is noise_sd**2, by default
    0.01. Training and test inputs are drawn with seed `split`. With d = 8 the test loss
    of the MLP student falls smoothly with both its width and n_train over about two
    decades each, which makes the task suitable for scaling experiments.
    Returns float32 tensors (Xtr, ytr, Xte, yte) with y of shape (n, 1).
    """
    g = torch.Generator().manual_seed(30_000 + task)
    W1 = torch.randn(d, width, generator=g) / math.sqrt(d)
    a = torch.randn(width, 1, generator=g) / math.sqrt(width)
    scale = (torch.tanh(torch.randn(20_000, d, generator=g) @ W1) @ a).std()
    f = lambda x: torch.tanh(x @ W1) @ a / scale
    gs = torch.Generator().manual_seed(40_000 + split)
    Xtr = torch.randn(n_train, d, generator=gs)
    Xte = torch.randn(n_test, d, generator=gs)
    ytr = f(Xtr) + noise_sd * torch.randn(n_train, 1, generator=gs)
    yte = f(Xte) + noise_sd * torch.randn(n_test, 1, generator=gs)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------
_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


class MLP(nn.Module):
    """A fully connected network: `depth` hidden layers of `width` units."""

    def __init__(self, d_in, d_out, width=64, depth=2, activation="relu",
                 batchnorm=False, dropout=0.0):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            if batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(_ACT[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------
@dataclass
class Config:
    # data
    data: str = "digits"            # "digits" or "teacher"
    train_size: int | None = None   # digits: subsample; teacher: number of examples (required)
    label_noise: float = 0.0        # digits only
    augment: bool = False           # digits only: random shifts by up to one pixel
    teacher_d: int = 8
    teacher_noise: float = 0.1
    # model
    width: int = 64
    depth: int = 2
    activation: str = "relu"
    batchnorm: bool = False
    dropout: float = 0.0
    # optimizer
    optimizer: str = "sgd"          # "sgd", "momentum", "nesterov", "adam", "adamw", "rmsprop", "adagrad"
    lr: float = 0.1
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    batch_size: int = 64
    # duration and schedule: give `epochs` or `steps`
    epochs: float | None = 20
    steps: int | None = None
    schedule: str = "constant"      # "constant", "cosine", "linear", "wsd" (decay over last 20%)
    warmup: int = 0                 # linear warm-up steps
    label_smoothing: float = 0.0
    # bookkeeping
    eval_every: int | None = None   # record test metrics every so many steps


def make_optimizer(cfg, params):
    o = cfg.optimizer
    if o == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if o in ("momentum", "nesterov"):
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                               nesterov=(o == "nesterov"), weight_decay=cfg.weight_decay)
    if o == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
                                weight_decay=cfg.weight_decay)
    if o == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
                                 weight_decay=cfg.weight_decay)
    if o == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.lr, alpha=cfg.beta2, eps=cfg.eps,
                                   momentum=cfg.momentum if cfg.momentum else 0.0,
                                   weight_decay=cfg.weight_decay)
    if o == "adagrad":
        return torch.optim.Adagrad(params, lr=cfg.lr, eps=cfg.eps, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {o}")


def lr_factor(cfg, step, total):
    """Multiplier of the base learning rate at a given step (0-based)."""
    if cfg.warmup and step < cfg.warmup:
        return (step + 1) / cfg.warmup
    s = cfg.schedule
    if s == "constant":
        return 1.0
    frac = (step - cfg.warmup) / max(1, total - cfg.warmup)
    if s == "cosine":
        return 0.5 * (1 + math.cos(math.pi * frac))
    if s == "linear":
        return 1.0 - frac
    if s == "wsd":
        return 1.0 if frac < 0.8 else max(0.0, (1.0 - frac) / 0.2)
    raise ValueError(f"unknown schedule {s}")


def _augment(x, gen):
    """Shift each 8x8 image by up to one pixel in each direction (zero padding)."""
    n = x.shape[0]
    img = F.pad(x.view(n, 1, 8, 8), (1, 1, 1, 1))
    dx = torch.randint(0, 3, (n,), generator=gen)
    dy = torch.randint(0, 3, (n,), generator=gen)
    out = torch.empty(n, 1, 8, 8)
    for i in range(3):
        for j in range(3):
            m = (dx == i) & (dy == j)
            if m.any():
                out[m] = img[m, :, j:j + 8, i:i + 8]
    return out.view(n, 64)


def load(cfg, split=0, noise=0):
    if cfg.data == "digits":
        return digits(split=split, train_size=cfg.train_size, label_noise=cfg.label_noise, noise=noise)
    if cfg.data == "teacher":
        if cfg.train_size is None:
            raise ValueError("teacher data needs cfg.train_size")
        return teacher(cfg.train_size, d=cfg.teacher_d, noise_sd=cfg.teacher_noise, split=split)
    raise ValueError(f"unknown data {cfg.data}")


@torch.no_grad()
def evaluate(model, X, y, regression):
    model.eval()
    out = model(X)
    model.train()
    if regression:
        return {"loss": F.mse_loss(out, y).item()}
    return {"loss": F.cross_entropy(out, y).item(),
            "acc": (out.argmax(1) == y).float().mean().item()}


def run(cfg: Config, init=0, order=0, split=0, noise=0, data=None, return_model=False):
    """Train one network and return its measurements.

    Returns a dict with keys `train_loss`, `test_loss` (and `train_acc`, `test_acc`,
    `test_err` for classification), `steps`, `params`, `seconds`, and, if
    `cfg.eval_every` is set, `curve`: a list of (step, test_loss[, test_acc]).
    `data` may pass a preloaded (Xtr, ytr, Xte, yte) to save time across many runs.
    """
    t0 = time.time()
    Xtr, ytr, Xte, yte = data if data is not None else load(cfg, split=split, noise=noise)
    regression = cfg.data == "teacher"
    d_out = 1 if regression else 10
    torch.manual_seed(50_000 + init)
    model = MLP(Xtr.shape[1], d_out, cfg.width, cfg.depth, cfg.activation, cfg.batchnorm, cfg.dropout)
    opt = make_optimizer(cfg, model.parameters())
    n = Xtr.shape[0]
    bs = min(cfg.batch_size, n)
    per_epoch = max(1, n // bs)
    total = cfg.steps if cfg.steps is not None else int(round(cfg.epochs * per_epoch))
    g_order = torch.Generator().manual_seed(60_000 + order)
    g_noise = torch.Generator().manual_seed(70_000 + noise)
    torch.manual_seed(80_000 + noise)     # dropout masks
    curve = []
    perm, pos = torch.randperm(n, generator=g_order), 0
    for step in range(total):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g_order), 0
        idx = perm[pos:pos + bs]
        pos += bs
        xb, yb = Xtr[idx], ytr[idx]
        if cfg.augment and not regression:
            xb = _augment(xb, g_noise)
        for group in opt.param_groups:
            group["lr"] = cfg.lr * lr_factor(cfg, step, total)
        out = model(xb)
        loss = F.mse_loss(out, yb) if regression else F.cross_entropy(out, yb, label_smoothing=cfg.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if cfg.eval_every and ((step + 1) % cfg.eval_every == 0 or step + 1 == total):
            ev = evaluate(model, Xte, yte, regression)
            curve.append((step + 1, ev["loss"]) if regression else (step + 1, ev["loss"], ev["acc"]))
    tr, te = evaluate(model, Xtr, ytr, regression), evaluate(model, Xte, yte, regression)
    res = {"train_loss": tr["loss"], "test_loss": te["loss"], "steps": total,
           "params": n_params(model), "seconds": time.time() - t0}
    if not regression:
        res.update(train_acc=tr["acc"], test_acc=te["acc"], test_err=1.0 - te["acc"])
        if not math.isfinite(res["test_loss"]):
            res["test_err"], res["test_acc"] = 0.9, 0.1
    if curve:
        res["curve"] = curve
    if return_model:
        res["model"] = model
    return res


# ---------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------
def t_interval(x, level=0.95):
    """Mean of x and the half-width of a two-sided t-interval for it."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(x.mean()), float("nan")
    h = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean()), float(h)


def bootstrap_ci(x, stat=np.mean, level=0.95, B=10_000, seed=0):
    """Percentile bootstrap interval for stat(x). Returns (estimate, lo, hi)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    reps = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.mean else x[idx].mean(1)
    a = (1 - level) / 2
    return float(stat(x)), float(np.quantile(reps, a)), float(np.quantile(reps, 1 - a))


def expected_best(scores, k, higher_is_better=True):
    """Unbiased estimate of E[best of k draws] from N >= k i.i.d. scores: the average of
    the best over all size-k subsets, computed exactly from the order statistics."""
    s = np.sort(np.asarray(scores, dtype=float))
    if not higher_is_better:
        s = -s[::-1]
    N = len(s)
    if k > N:
        raise ValueError("need k <= number of scores")
    # P(the i-th smallest, 1-based, is the max of a random k-subset) = C(i-1, k-1) / C(N, k)
    from scipy.special import comb
    w = np.array([comb(i - 1, k - 1, exact=False) for i in range(1, N + 1)]) / comb(N, k, exact=False)
    est = float(np.dot(w, s))
    return est if higher_is_better else -est


# ---------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------
def write_dat(path, columns: dict, fmt="{:.6g}"):
    """Write equal-length columns as a whitespace-separated table with a header row.
    `path` is relative to the volume root (e.g. figures/data/01-seeds-spread.dat) or absolute."""
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(HERE, "..", "..", path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(columns)
    cols = [list(columns[k]) for k in keys]
    n = len(cols[0])
    assert all(len(c) == n for c in cols), "columns must have equal length"
    with open(path, "w") as f:
        f.write(" ".join(keys) + "\n")
        for i in range(n):
            f.write(" ".join(v if isinstance(v, str) else fmt.format(v) for v in (c[i] for c in cols)) + "\n")
    return path


if __name__ == "__main__":
    # self-test: determinism, speed, and the statistics helpers
    cfg = Config(width=64, optimizer="momentum", lr=0.05, epochs=20)
    data = load(cfg)
    a = run(cfg, init=1, order=2, data=data)
    b = run(cfg, init=1, order=2, data=data)
    c = run(cfg, init=1, order=3, data=data)
    print(f"digits MLP: test_acc {a['test_acc']:.4f} ({a['seconds']:.2f}s, {a['steps']} steps, {a['params']} params)")
    print("bitwise repeatable:", a["test_loss"] == b["test_loss"], "| new order seed changes it:", a["test_loss"] != c["test_loss"])
    tcfg = Config(data="teacher", train_size=8192, width=64, optimizer="adam", lr=3e-3, steps=2000,
                  batch_size=128, schedule="cosine")
    r = run(tcfg)
    print(f"teacher MLP: test_loss {r['test_loss']:.4f} (floor {tcfg.teacher_noise**2:.4f}) in {r['seconds']:.2f}s")
    x = np.random.default_rng(0).normal(size=50)
    print("t_interval", np.round(t_interval(x), 4), "bootstrap", np.round(bootstrap_ci(x), 4))
    sc = np.random.default_rng(1).normal(size=20)
    brute = np.mean([max(c) for c in __import__("itertools").combinations(sc, 3)])
    print("expected_best unbiased (exact check):", np.isclose(expected_best(sc, 3), brute))
