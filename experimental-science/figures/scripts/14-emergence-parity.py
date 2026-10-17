"""Lab 14.3: a transition that the metric does not make (a positive control).

Sparse parity after Barak et al. (2022): inputs x in {-1,+1}^50 drawn uniformly, label
y = x1 x2 x3, a network with one hidden layer of 256 ReLU units trained online by SGD
(learning rate 0.1, batches of 32 fresh examples, logistic loss) for 12,000 steps; ten runs,
run s with seed s for the initialization and the data stream. Every 100 steps, on 4,000
fixed test strings: the test loss (nats; a continuous metric), the test error (a threshold
metric), and a progress measure, the largest over hidden units of the share of the unit's
squared input weights that sits on the three relevant inputs, divided by the share 3/50 of a
unit that ignores relevance (its maximum is 50/3).

Each run's transition time t* is the first evaluation at which the test error is below 0.25;
the data files also give the steps relative to t*.

Run from the volume root: python3 figures/scripts/14-emergence-parity.py  (about 30 s)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import labkit as lk  # sets two threads and deterministic kernels

N_BITS, K, WIDTH, LR, BATCH, STEPS, EVERY = 50, 3, 256, 0.1, 32, 12000, 100
SEEDS = range(10)
OUT = "figures/data/14-emergence-parity"


def run(seed):
    g = torch.Generator().manual_seed(1000 + seed)
    torch.manual_seed(2000 + seed)
    model = nn.Sequential(nn.Linear(N_BITS, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 1))
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    Xte = torch.randint(0, 2, (4000, N_BITS), generator=g).float() * 2 - 1
    yte = Xte[:, :K].prod(1)
    rec = []
    for t in range(STEPS + 1):
        if t % EVERY == 0:
            with torch.no_grad():
                f = model(Xte).squeeze(1)
                W2 = model[0].weight ** 2
                conc = (W2[:, :K].sum(1) / W2.sum(1)) / (K / N_BITS)
                rec.append((t, F.softplus(-yte * f).mean().item(),
                            (torch.sign(f) != yte).float().mean().item(), conc.max().item()))
        X = torch.randint(0, 2, (BATCH, N_BITS), generator=g).float() * 2 - 1
        loss = F.softplus(-X[:, :K].prod(1) * model(X).squeeze(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return np.array(rec)


def main():
    t0 = time.time()
    cols = {k: [] for k in ("seed", "t", "rel", "loss", "err", "conc")}
    tstars, plateau, lead, drops = [], [], [], []
    for s in SEEDS:
        r = run(s)
        below = np.nonzero(r[:, 2] < 0.25)[0]
        if len(below) == 0:
            print(f"run {s}: no transition within {STEPS} steps")
            continue
        t_star = r[below[0], 0]
        tstars.append(t_star)
        for t, loss, err, conc in r:
            for k, v in zip(cols, (s, t, t - t_star, loss, err, conc)):
                cols[k].append(v)
        at = lambda dt, j: r[r[:, 0] == t_star + dt, j][0] if 0 <= t_star + dt <= STEPS else np.nan
        pre = r[r[:, 0] <= t_star - 1000]
        plateau.append((pre[:, 1].min(), pre[:, 1].max(), pre[:, 2].min()))
        lead.append((r[0, 3], at(-2000, 3), at(-1000, 3), at(-1000, 1), at(0, 1), at(1000, 1)))
        # the drop: from the first loss below 0.68 (the plateau left) to the first loss below 0.05
        t_leave = r[np.argmax(r[:, 1] < 0.68), 0]
        t_reach = r[np.argmax(r[:, 1] < 0.05), 0]
        drops.append((t_leave, t_reach - t_leave))
        print(f"run {s}: t* = {int(t_star)}; loss up to t*-1000 in [{pre[:, 1].min():.3f}, {pre[:, 1].max():.3f}]; "
              f"concentration {r[0, 3]:.2f} at 0, {at(-2000, 3):.2f} at t*-2000, {at(-1000, 3):.2f} at t*-1000; "
              f"loss {at(0, 1):.3f} at t*, {at(1000, 1):.3f} at t*+1000; "
              f"loss below 0.68 from step {int(t_leave)}, below 0.05 {int(t_reach - t_leave)} steps later")
    lk.write_dat(OUT + "-runs.dat", cols)
    tstars = np.array(tstars)
    lead = np.array(lead)
    print(f"\nt* from {tstars.min():.0f} to {tstars.max():.0f} steps (median {np.median(tstars):.0f})")
    print(f"loss during the plateaus: {min(p[0] for p in plateau):.3f} to {max(p[1] for p in plateau):.3f} "
          f"(ln 2 = {np.log(2):.3f}); smallest error there {min(p[2] for p in plateau):.3f}")
    print(f"concentration: start {np.nanmin(lead[:, 0]):.2f}-{np.nanmax(lead[:, 0]):.2f}; at t*-2000 "
          f"{np.nanmin(lead[:, 1]):.2f}-{np.nanmax(lead[:, 1]):.2f}; at t*-1000 {np.nanmin(lead[:, 2]):.2f}-{np.nanmax(lead[:, 2]):.2f}")
    print(f"loss at t*-1000: {np.nanmin(lead[:, 3]):.3f}-{np.nanmax(lead[:, 3]):.3f}; at t*: {np.nanmin(lead[:, 4]):.3f}-"
          f"{np.nanmax(lead[:, 4]):.3f}; at t*+1000: {np.nanmin(lead[:, 5]):.3f}-{np.nanmax(lead[:, 5]):.3f}")
    drops = np.array(drops)
    print(f"plateau left (loss < 0.68) after {drops[:, 0].min():.0f} to {drops[:, 0].max():.0f} steps; "
          f"loss from 0.68 to below 0.05 within {drops[:, 1].min():.0f} to {drops[:, 1].max():.0f} steps")
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
