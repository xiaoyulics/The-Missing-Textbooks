"""Lab 2.1 of Chapter 2 (Instability): what a fixed seed repeats, and the order of a sum.

Run from the volume root:

    python3 figures/scripts/02-instability-sums.py

Four checks on a laptop CPU, in binary32 (single precision, unit roundoff u = 2^-24):

  1. Three numbers whose sum depends on the order of addition.
  2. Sums of n standard normal numbers (20 vectors per n, 50 random orders each), by
     recursive (left-to-right) and by pairwise summation: how many different results the
     orders give, and how far apart they are, in units of u * sum |x_i|, against the
     worst-case bound of the chapter's summation proposition (2 * depth of the tree).
  3. Batch invariance: the logits of one trained network for the same test images, computed
     in batches of different sizes.
  4. Repeatability: two runs with the same seeds, and a run with a different thread count.

Writes figures/data/02-instability-sums.dat and prints the numbers quoted in the chapter.
Runtime: a few seconds.
"""
import os
import sys
import math
import time
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk  # noqa: E402  (sets two threads and deterministic algorithms)

import numpy as np  # noqa: E402
import torch  # noqa: E402

T0 = time.time()
U = 2.0 ** -24  # unit roundoff of binary32 with round-to-nearest


def recursive_sum(x):
    """Left-to-right summation with a binary32 accumulator."""
    return float(np.cumsum(x, dtype=np.float32)[-1])


def pairwise_sum(x):
    """Balanced binary-tree summation in binary32 (depth ceil(log2 n))."""
    x = np.asarray(x, dtype=np.float32)
    while len(x) > 1:
        if len(x) % 2:
            x = np.append(x, np.float32(0.0))  # adding 0 is exact
        x = (x[0::2] + x[1::2]).astype(np.float32)
    return float(x[0])


def weights_digest(model):
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------------------
# 1. Three numbers
# ---------------------------------------------------------------------------------------
one, e = np.float32(1.0), np.float32(2.0 ** -24)
left = (one + e) + e
right = one + (e + e)
print("1. Three numbers in binary32: x = (1, 2^-24, 2^-24)")
print(f"   (1 + 2^-24) + 2^-24 = {left!r}   1 + (2^-24 + 2^-24) = {right!r}")
print(f"   difference = {float(right) - float(left):.3e} = 2^{math.log2(float(right) - float(left)):.0f}"
      f" = {(float(right) - float(left)) / (U * (1 + 2 * 2.0 ** -24)):.3f} u sum|x|")

# ---------------------------------------------------------------------------------------
# 2. Reordered sums
# ---------------------------------------------------------------------------------------
rng = np.random.default_rng(20261017)
ns = [10, 100, 1000, 10000, 100000]
V, K = 20, 50          # vectors per n, random orders per vector
rows = {k: [] for k in ("n", "rec_distinct", "rec_spread", "rec_err", "pw_distinct", "pw_spread",
                        "pw_err", "bound_rec", "bound_pw")}
worst_partial = 0.0    # max of |error| / ((u/(1-u)) sum_k |s_k|): must not exceed 1
print(f"\n2. Sums of n standard normal binary32 numbers: {V} vectors per n, {K} random orders each.")
print("   Spread = (largest - smallest result over the orders) / (u sum|x_i|), averaged over vectors;")
print("   error = mean |computed - exact| / (u sum|x_i|); bound = 2 gamma_h / u for a tree of depth h.")
print("   n        recursive: distinct spread  error  bound | pairwise: distinct spread  error bound")
for n in ns:
    acc = {k: [] for k in ("rd", "rs", "re", "pd", "ps", "pe")}
    for _ in range(V):
        x = rng.standard_normal(n).astype(np.float32)
        x64 = x.astype(np.float64)             # exact copies of the binary32 inputs
        exact = math.fsum(x64)                 # correctly rounded binary64 sum
        scale = U * float(np.sum(np.abs(x64)))
        rec, pw = [], []
        for _ in range(K):
            p = rng.permutation(n)
            partial = np.cumsum(x[p], dtype=np.float32)
            rec.append(float(partial[-1]))
            pw.append(pairwise_sum(x[p]))
            bound = U / (1 - U) * float(np.sum(np.abs(partial[1:].astype(np.float64))))
            if bound > 0:
                worst_partial = max(worst_partial, abs(rec[-1] - exact) / bound)
        rec, pw = np.array(rec), np.array(pw)
        acc["rd"].append(len(set(rec.tolist())))
        acc["rs"].append((rec.max() - rec.min()) / scale)
        acc["re"].append(np.mean(np.abs(rec - exact)) / scale)
        acc["pd"].append(len(set(pw.tolist())))
        acc["ps"].append((pw.max() - pw.min()) / scale)
        acc["pe"].append(np.mean(np.abs(pw - exact)) / scale)
    h = math.ceil(math.log2(n))
    gam = lambda m: m * U / (1 - m * U)
    rows["n"].append(n)
    rows["rec_distinct"].append(float(np.mean(acc["rd"])))
    rows["rec_spread"].append(float(np.mean(acc["rs"])))
    rows["rec_err"].append(float(np.mean(acc["re"])))
    rows["pw_distinct"].append(float(np.mean(acc["pd"])))
    rows["pw_spread"].append(float(np.mean(acc["ps"])))
    rows["pw_err"].append(float(np.mean(acc["pe"])))
    rows["bound_rec"].append(2 * gam(n - 1) / U)
    rows["bound_pw"].append(2 * gam(h) / U)
    print(f"   {n:<7d}  {rows['rec_distinct'][-1]:15.1f} {rows['rec_spread'][-1]:6.2f} {rows['rec_err'][-1]:6.2f}"
          f" {rows['bound_rec'][-1]:7.0f} | {rows['pw_distinct'][-1]:15.1f} {rows['pw_spread'][-1]:6.2f}"
          f" {rows['pw_err'][-1]:6.3f} {rows['bound_pw'][-1]:4.0f}")
lk.write_dat("figures/data/02-instability-sums.dat", rows)
print(f"   largest |error| / partial-sum bound over all {len(ns) * V * K} recursive sums: {worst_partial:.3f}")

# ---------------------------------------------------------------------------------------
# 3. Batch invariance of a trained network
# ---------------------------------------------------------------------------------------
cfg = lk.Config(width=64, optimizer="momentum", lr=0.05, epochs=20)
data = lk.load(cfg)
Xte = data[2]
r = lk.run(cfg, init=1, order=2, data=data, return_model=True)
model = r["model"].eval()
with torch.no_grad():
    full = model(Xte)                                                   # one batch of 539
    single = torch.cat([model(Xte[i:i + 1]) for i in range(len(Xte))])  # batches of 1
    b64 = torch.cat([model(Xte[i:i + 64]) for i in range(0, len(Xte), 64)])
    again = model(Xte)
print("\n3. Batch invariance: logits of one trained MLP for the 539 test images")
for name, other in (("same batch, recomputed", again), ("batches of 64", b64), ("one image at a time", single)):
    differ = (other != full).any(1).float().mean().item()
    print(f"   {name:24s}: rows differing in some bit {100 * differ:5.1f}%,"
          f" max |difference| {(other - full).abs().max().item():.2e},"
          f" predictions changed {(other.argmax(1) != full.argmax(1)).sum().item()}")

# ---------------------------------------------------------------------------------------
# 4. Repeatability on this machine
# ---------------------------------------------------------------------------------------
a = lk.run(cfg, init=1, order=2, data=data, return_model=True)
b = lk.run(cfg, init=1, order=2, data=data, return_model=True)
torch.set_num_threads(1)
c = lk.run(cfg, init=1, order=2, data=data, return_model=True)
torch.set_num_threads(2)
d = lk.run(cfg, init=1, order=3, data=data, return_model=True)
print("\n4. Repeatability (SHA-256 of the final weights, first 16 hex digits)")
print(f"   seeds (1, 2), two threads : {weights_digest(a['model'])}  test acc {a['test_acc']:.4f}")
print(f"   seeds (1, 2), again       : {weights_digest(b['model'])}  test acc {b['test_acc']:.4f}")
print(f"   seeds (1, 2), one thread  : {weights_digest(c['model'])}  test acc {c['test_acc']:.4f}")
print(f"   seeds (1, 3), two threads : {weights_digest(d['model'])}  test acc {d['test_acc']:.4f}")
print(f"   torch {torch.__version__}, numpy {np.__version__}, BLAS: "
      f"{'accelerate' if 'BLAS_INFO=accelerate' in torch.__config__.show() else 'other'}")
print(f"\nruntime {time.time() - T0:.1f} s")
