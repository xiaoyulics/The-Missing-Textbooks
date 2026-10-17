"""A numerical check of the inclusion proposition of Chapter 4: Adam with a large eps follows
momentum SGD.

On the task of Lab 4.1 (digits with 256 distractor features, split 0; MLP of width 64; batch
size 32; 900 steps, that is 100 epochs), momentum SGD with learning rate eta = 0.03 and
momentum mu = 0.9 is compared with Adam (beta1 = mu, beta2 = 0.999) for eps from 1e-8 to 1e8,
started from the same initial weights and fed the same minibatches. Two learning rates for
Adam are tried:
  scheduled  alpha_t = eps * eta * (1 - mu^t) / (1 - mu), which cancels Adam's bias correction
  constant   alpha   = eps * eta / (1 - mu), which matches momentum only once mu^t is small
The script reports the relative distance between the final weight vectors and the test error
of each run. Everything is computed in double precision, so that rounding does not hide the
1/eps convergence.

Run from the volume root:  python3 figures/scripts/04-budgets-inclusion.py
"""
import os, sys, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk
import numpy as np
import torch
import torch.nn.functional as F

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "protocols", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04-budgets-protocols.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

torch.set_default_dtype(torch.float64)
ETA, MU, BETA2, STEPS, SEED = 0.03, 0.9, 0.999, 900, 1


def train(kind, data, eps=None, schedule=None):
    Xtr, ytr, Xva, yva = [t.double() if t.is_floating_point() else t for t in data]
    torch.manual_seed(50_000 + SEED)
    model = lk.MLP(Xtr.shape[1], 10, P.WIDTH, 2)
    if kind == "momentum":
        opt = torch.optim.SGD(model.parameters(), lr=ETA, momentum=MU)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(MU, BETA2), eps=eps)
    n, bs = Xtr.shape[0], P.BATCH
    g = torch.Generator().manual_seed(60_000 + SEED)
    perm, pos = torch.randperm(n, generator=g), 0
    for step in range(1, STEPS + 1):
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g), 0
        idx = perm[pos:pos + bs]
        pos += bs
        if kind == "adam":
            a = eps * ETA / (1 - MU) * ((1 - MU ** step) if schedule == "scheduled" else 1.0)
            for group in opt.param_groups:
                group["lr"] = a
        loss = F.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def main():
    t0 = time.time()
    data, test = P.make_data(split=0)
    Xte, yte = test[0].double(), test[1]
    ref = train("momentum", data)
    w_ref = flat(ref)
    err_ref = 1 - lk.evaluate(ref, Xte, yte, False)["acc"]
    print(f"momentum (eta={ETA}, mu={MU}): test error {100 * err_ref:.2f}%")
    epss = [10.0 ** e for e in range(-8, 9, 2)]
    cols = {"eps": epss, "dsched": [], "dconst": [], "esched": [], "econst": []}
    for eps in epss:
        for sch, dk, ek in (("scheduled", "dsched", "esched"), ("constant", "dconst", "econst")):
            m = train("adam", data, eps=eps, schedule=sch)
            d = float(torch.norm(flat(m) - w_ref) / torch.norm(w_ref))
            ev = lk.evaluate(m, Xte, yte, False)
            e = 1 - ev["acc"] if math.isfinite(ev["loss"]) else 0.9
            cols[dk].append(d)
            cols[ek].append(100 * e)
        print(f"eps={eps:8.0e}: relative distance scheduled {cols['dsched'][-1]:.3e}, constant "
              f"{cols['dconst'][-1]:.3e}; test error scheduled {cols['esched'][-1]:.2f}%, "
              f"constant {cols['econst'][-1]:.2f}%", flush=True)
    lk.write_dat("figures/data/04-budgets-inclusion.dat", cols)
    for i in range(len(epss) - 1):
        print(f"distance ratio (scheduled) from eps={epss[i]:.0e} to eps={epss[i + 1]:.0e}: "
              f"{cols['dsched'][i] / cols['dsched'][i + 1]:.3g}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
