"""Lab 9.2 of Deep Learning as an Experimental Science: the advantage of batch
normalization as a function of the learning rate, and the scale invariance behind it.

The six-layer perceptron of Lab 9.1 (09-mechanisms-noisybn.py), without and with batch
normalization, is trained with plain SGD for 600 steps at seven learning rates from 0.01
to 10, ten seeds per cell (seeds paired across the two arms). For every run we record the
mean training loss over twelve checkpoints, the final test error and the update-induced
covariate shift of Lab 9.1. For the normalized network we also check two predictions of
scale invariance on every step and every row w_j of a weight matrix that feeds a batch
normalization: the bias before the normalization receives a zero gradient, and
||w_j||^2 grows by exactly lr^2 ||grad w_j||^2 per step; and we report the effective
learning rate lr / ||w_j||^2 at the end of training. Finally, the batch-normalization
Jacobian identity is checked numerically, with the factor 1/m and with 1/sqrt(m).
Writes figures/data/09-mechanisms-lr-*.dat. About three minutes with two threads.
"""
import importlib.util
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labkit as lk  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_spec = importlib.util.spec_from_file_location("noisybn", os.path.join(HERE, "09-mechanisms-noisybn.py"))
nb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb)

LRS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
SEEDS = range(10)


def jacobian_check(m=8, trials=200, seed=0):
    """Largest relative error of the identity ||grad_y L||^2 = (gamma/sigma)^2 (||g||^2 -
    <1,g>^2/m - c <yhat,g>^2) over random draws, for c = 1/m and for c = 1/sqrt(m)."""
    g_ = torch.Generator().manual_seed(seed)
    worst = {"1/m": 0.0, "1/sqrt(m)": 0.0, "negative": 0}
    for _ in range(trials):
        y = torch.randn(m, generator=g_, dtype=torch.float64) * 3 + 1
        y.requires_grad_(True)
        gamma, beta = 0.7, 0.2
        g = torch.randn(m, generator=g_, dtype=torch.float64)
        mu = y.mean()
        sigma = ((y - mu) ** 2).mean().sqrt()
        yhat = (y - mu) / sigma
        z = gamma * yhat + beta
        (gy,) = torch.autograd.grad((g * z).sum(), y)    # a loss whose gradient in z is g
        lhs = (gy ** 2).sum().item()
        yh, s = yhat.detach(), sigma.item()
        for name, c in (("1/m", 1.0 / m), ("1/sqrt(m)", 1.0 / math.sqrt(m))):
            rhs = (gamma / s) ** 2 * ((g ** 2).sum() - g.sum() ** 2 / m - c * (yh @ g) ** 2).item()
            worst[name] = max(worst[name], abs(lhs - rhs) / lhs)
            worst["negative"] += int(name == "1/sqrt(m)" and rhs < 0)
    return worst


def run(bn, lr, seed, data):
    t0 = time.time()
    Xtr, ytr, _, _ = data
    model = nb.Net(bn, False, init=seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = Xtr.shape[0]
    g = torch.Generator().manual_seed(60_000 + seed)
    perm, pos = torch.randperm(n, generator=g), 0
    Xp = Xtr[:nb.PROBE]
    norm0 = [(lin.weight.detach() ** 2).sum(1) for lin in model.lins]
    grown = [torch.zeros_like(v) for v in norm0]       # running sum of lr^2 ||grad w_j||^2
    bias_grad = 0.0
    loss_curve, err_curve, upd = [], [], []
    for step in range(nb.STEPS):
        if pos + nb.BATCH > n:
            perm, pos = torch.randperm(n, generator=g), 0
        idx = perm[pos:pos + nb.BATCH]
        pos += nb.BATCH
        if step % nb.SHIFT_EVERY == 0:
            A = nb.probe(model, Xp, 123)
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if bn:
            for k, lin in enumerate(model.lins):
                grown[k] += lr ** 2 * (lin.weight.grad ** 2).sum(1)
                bias_grad = max(bias_grad, lin.bias.grad.abs().max().item())
        opt.step()
        if not math.isfinite(loss.item()):
            return None
        if step % nb.SHIFT_EVERY == 0:
            B = nb.probe(model, Xp, 123)
            upd.append(np.mean([nb.w1_shift(a, b) for a, b in zip(A, B)]))
        if (step + 1) % nb.EVAL_EVERY == 0:
            tl, te = nb.evaluate(model, data)
            if not math.isfinite(tl):
                return None
            loss_curve.append(tl)
            err_curve.append(te)
    out = dict(meanloss=float(np.mean(loss_curve)), final=loss_curve[-1], err=100 * err_curve[-1],
               upd=float(np.mean(upd)), seconds=time.time() - t0)
    normT = [(lin.weight.detach() ** 2).sum(1) for lin in model.lins]
    if bn:
        rel = [((nt - n0 - gr).abs() / nt).max().item() for nt, n0, gr in zip(normT, norm0, grown)]
        out.update(identity=max(rel), bias_grad=bias_grad,
                   eff0=float(torch.median(lr / torch.cat(norm0))), effT=float(torch.median(lr / torch.cat(normT))),
                   growth=float(torch.median(torch.cat(normT) / torch.cat(norm0))))
    with torch.no_grad():
        rec = []
        model.eval()
        model(Xp, record=rec)
        model.train()
        out["dead"] = float(np.mean([(a.max(0).values <= 0).float().mean().item() for a in rec]))
    return out


def main():
    t_start = time.time()
    worst = jacobian_check()
    print(f"Jacobian identity, worst relative error over 200 draws (m=8): with 1/m {worst['1/m']:.1e}, "
          f"with 1/sqrt(m) {worst['1/sqrt(m)']:.2f}; right side with 1/sqrt(m) negative in {worst['negative']} of 200")
    data = nb.lk.load(nb.lk.Config())
    R = {}
    for bn in (False, True):
        for lr in LRS:
            R[(bn, lr)] = [run(bn, lr, s, data) for s in SEEDS]
            ok = [r for r in R[(bn, lr)] if r is not None]
            msg = f"{'BN   ' if bn else 'plain'} lr={lr:<5} trained {len(ok)}/{len(SEEDS)}"
            if ok:
                msg += (f" | test {np.mean([r['err'] for r in ok]):.2f}% | mean loss {np.mean([r['meanloss'] for r in ok]):.4f}"
                        f" | final loss {np.mean([r['final'] for r in ok]):.5f} | shift_upd {np.mean([r['upd'] for r in ok]):.4f}"
                        f" | dead units {np.mean([r['dead'] for r in ok]):.2f}")
                if bn:
                    msg += (f" | eff lr {np.mean([r['eff0'] for r in ok]):.4f} -> {np.mean([r['effT'] for r in ok]):.4f}"
                            f" | norm growth x{np.mean([r['growth'] for r in ok]):.2f}"
                            f" | identity err {max(r['identity'] for r in ok):.1e} | bias grad {max(r['bias_grad'] for r in ok):.1e}")
            print(msg + f" | {time.time() - t_start:.0f} s", flush=True)

    # ---- data files ---------------------------------------------------------------------
    rng = np.random.default_rng(1)
    for bn, tag in ((False, "plain"), (True, "bn")):
        runs = {k: [] for k in ("lr", "seed", "err", "meanloss", "upd", "failed")}
        means = {k: [] for k in ("lr", "n", "err", "errhw", "meanloss", "meanlosshw", "upd", "updhw", "effT", "effThw", "eff0")}
        for lr in LRS:
            ok = []
            for s, r in zip(SEEDS, R[(bn, lr)]):
                runs["lr"].append(lr * 10 ** rng.uniform(-0.06, 0.06))
                runs["seed"].append(s)
                runs["failed"].append(0 if (r is not None and r["err"] < 50) else 1)
                runs["err"].append(r["err"] if r is not None else 90.0)
                runs["meanloss"].append(r["meanloss"] if r is not None else float("nan"))
                runs["upd"].append(r["upd"] if r is not None else float("nan"))
                if r is not None and r["err"] < 50:
                    ok.append(r)
            if len(ok) < 2:
                continue
            means["lr"].append(lr)
            means["n"].append(len(ok))
            for k in ("err", "meanloss", "upd"):
                m, h = lk.t_interval([r[k] for r in ok])
                means[k].append(m)
                means[k + "hw"].append(h)
            if bn:
                m, h = lk.t_interval([r["effT"] for r in ok])
                means["effT"].append(m)
                means["effThw"].append(h)
                means["eff0"].append(float(np.mean([r["eff0"] for r in ok])))
            else:
                means["effT"].append(float("nan"))
                means["effThw"].append(float("nan"))
                means["eff0"].append(float("nan"))
        lk.write_dat(f"figures/data/09-mechanisms-lr-runs-{tag}.dat", runs)
        lk.write_dat(f"figures/data/09-mechanisms-lr-means-{tag}.dat", means)

    # ---- the advantage, seed by seed, where both arms trained ---------------------------
    print("\nadvantage of batch normalization (plain minus BN), paired by seed")
    for lr in LRS:
        pairs = [(a, b) for a, b in zip(R[(False, lr)], R[(True, lr)])
                 if a is not None and b is not None and a["err"] < 50 and b["err"] < 50]
        if len(pairs) < 2:
            print(f"  lr={lr:<5}: fewer than two pairs (plain trained {sum(r is not None for r in R[(False, lr)])}, "
                  f"BN trained {sum(r is not None and r['err'] < 50 for r in R[(True, lr)])})")
            continue
        for k in ("err", "meanloss"):
            m, h = lk.t_interval([a[k] - b[k] for a, b in pairs])
            print(f"  lr={lr:<5} {k:8s} {m:+.4f} +- {h:.4f} (n={len(pairs)}, BN better on {sum(a[k] > b[k] for a, b in pairs)})")
    best = {}
    for bn in (False, True):
        cands = [(np.mean([r["err"] for r in R[(bn, lr)] if r is not None]), lr) for lr in LRS
                 if all(r is not None and r["err"] < 50 for r in R[(bn, lr)])]
        best[bn] = min(cands)
        print(f"best learning rate ({'BN' if bn else 'plain'}), every seed trained: lr={best[bn][1]}, test {best[bn][0]:.2f}%")
    a = [r["err"] for r in R[(False, best[False][1])]]
    b = [r["err"] for r in R[(True, best[True][1])]]
    m, h = lk.t_interval(np.array(a) - np.array(b))
    print(f"best against best, paired by seed: plain minus BN {m:+.2f} +- {h:.2f} points")
    print(f"\ntotal time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
