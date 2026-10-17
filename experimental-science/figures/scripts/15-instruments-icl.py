"""Lab 15.2 of *Deep Learning as an Experimental Science*: do the labels of the
demonstrations matter? In-context classification of digits with a tiny transformer.

A sequence holds eight demonstrations, four images of each of two digit classes with
their labels, and a query image of one of the two classes; the network predicts the
query's label. Each token is an image (64 standardized pixels) concatenated with a
one-hot label (zeros for the query); a two-layer transformer without positional
encodings (72,458 parameters) reads the nine tokens, and the query's output gives the
label. Training sequences are built from the 1,258 training images of split 0. In a
fraction q of them (omega in the chapter) the ten labels are permuted at random for the whole sequence, so
that only the demonstrations tell which label goes with which digit; in the rest the
label is the digit itself, which the network can also learn in its weights.

Five values of q (0, 0.01, 0.05, 0.2, 1), three seeds each (initialization and
data stream), 1,500 steps of AdamW. Every network is evaluated on sequences built from
the 539 test images (never seen in training), with the query's true class as the answer:
  none    the query alone, no demonstrations;
  gold    demonstrations labelled with their digits;
  random  each demonstration labelled with one of the two classes, chosen by a fair
          coin (the label space is kept, the mapping destroyed);
  flipped the two classes' labels exchanged;
and, scored against the labels shown, `unrelated` (the two classes labelled with two
other digits, consistently). Evaluations use 1, 2, 4 and 8 demonstrations per class.

Run from the volume root:  python3 figures/scripts/15-instruments-icl.py
Writes figures/data/15-instruments-icl-{runs,means,shots}.dat and prints the numbers
quoted in Chapter 15. Three to five minutes on two threads, depending on the load of the
machine (fifteen networks of 1,500 steps each).
"""
import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labkit as lk

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

QS = [0.0, 0.01, 0.05, 0.2, 1.0]
SEEDS = [0, 1, 2]
STEPS, BATCH, LR, WARMUP = 1500, 128, 1e-3, 200
SHOTS_TRAIN = 4                      # demonstrations per class in training
SHOTS_EVAL = [1, 2, 4, 8]
N_EVAL = 2000                        # sequences per condition
CONDS = ["none", "gold", "random", "flipped", "unrelated"]

# ---- data: split 0 of the digits, standardized with the training set's global mean and sd
X, y = lk._digits_raw()
_perm = np.random.default_rng(10_000).permutation(len(y))
_nte = int(round(0.3 * len(y)))
TE, TR = _perm[:_nte], _perm[_nte:]
_m, _s = X[TR].mean(), X[TR].std()
XN = torch.from_numpy(((X - _m) / _s).astype(np.float32))
POOL_TR = [torch.from_numpy(TR[y[TR] == c]) for c in range(10)]
POOL_TE = [torch.from_numpy(TE[y[TE] == c]) for c in range(10)]


def draw_images(cls, count, pools, g):
    """For each row of `cls` (B,), `count` distinct images of that class: (B, count) indices."""
    out = torch.empty(len(cls), count, dtype=torch.long)
    for k in range(10):
        rows = (cls == k).nonzero().flatten()
        if len(rows):
            pool = pools[k]
            pick = torch.rand(len(rows), len(pool), generator=g).argsort(1)[:, :count]
            out[rows] = pool[pick]
    return out


def make_batch(B, g, pools, cond, shots):
    """Tokens (B, 2*shots+1, 74) [or (B,1,74) for `none`], the answer, the query's class and
    the other class."""
    two = torch.rand(B, 10, generator=g).argsort(1)[:, :2]               # two distinct classes
    c1, c2 = two[:, 0], two[:, 1]
    which = torch.randint(0, 2, (B,), generator=g)
    qc = torch.where(which == 0, c1, c2)                                  # query class
    oc = torch.where(which == 0, c2, c1)                                  # the other class
    img_q = draw_images(qc, shots + 1, pools, g)                          # shots + query, distinct
    img_o = draw_images(oc, shots, pools, g)
    ctx_idx = torch.cat([img_q[:, :shots], img_o], 1)
    ctx_cls = torch.cat([qc[:, None].expand(B, shots), oc[:, None].expand(B, shots)], 1)
    order = torch.rand(B, 2 * shots, generator=g).argsort(1)
    ctx_idx, ctx_cls = ctx_idx.gather(1, order), ctx_cls.gather(1, order)
    q_idx = img_q[:, shots]
    answer, lab = qc.clone(), ctx_cls.clone()
    if cond == "perm":                                                    # training only
        P = torch.rand(B, 10, generator=g).argsort(1)
        lab, answer = P.gather(1, ctx_cls), P.gather(1, qc[:, None]).squeeze(1)
    elif cond == "random":
        coin = torch.randint(0, 2, ctx_cls.shape, generator=g)
        lab = torch.where(coin == 0, c1[:, None].expand_as(ctx_cls), c2[:, None].expand_as(ctx_cls))
    elif cond == "flipped":
        lab = torch.where(ctx_cls == qc[:, None], oc[:, None].expand_as(ctx_cls), qc[:, None].expand_as(ctx_cls))
    elif cond == "unrelated":
        # two labels outside {c1, c2}: the classes' ranks among the other eight digits
        r = torch.rand(B, 10, generator=g)
        r[torch.arange(B), c1] = 2.0
        r[torch.arange(B), c2] = 2.0
        new = r.argsort(1)[:, :2]                                         # new label of qc, of oc
        lab = torch.where(ctx_cls == qc[:, None], new[:, :1].expand_as(ctx_cls), new[:, 1:].expand_as(ctx_cls))
        answer = new[:, 0]
    tq = torch.cat([XN[q_idx], torch.zeros(B, 10)], 1)[:, None]
    if cond == "none":
        return tq, answer, qc, oc
    tc = torch.cat([XN[ctx_idx], F.one_hot(lab, 10).float()], 2)
    return torch.cat([tc, tq], 1), answer, qc, oc


class TinyTransformer(nn.Module):
    """Two pre-norm encoder layers (width 64, 4 heads, feed-forward 128), no positional
    encodings; a learned flag marks the query token, whose output gives the label."""

    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Linear(74, d)
        self.qflag = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(d, 10)

    def forward(self, tok):
        h = self.inp(tok)
        h = torch.cat([h[:, :-1], h[:, -1:] + self.qflag], 1)
        return self.out(self.body(h)[:, -1])


def train(q, seed):
    torch.manual_seed(50_000 + seed)
    model = TinyTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    g = torch.Generator().manual_seed(60_000 + 1000 * seed + int(round(1000 * q)))
    n_perm = torch.distributions.Binomial(BATCH, torch.tensor(q))
    torch.manual_seed(70_000 + seed)                     # for the binomial draws below
    for step in range(STEPS):
        k = int(n_perm.sample().item()) if 0 < q < 1 else int(round(q * BATCH))
        parts = []
        if k > 0:
            parts.append(make_batch(k, g, POOL_TR, "perm", SHOTS_TRAIN))
        if k < BATCH:
            parts.append(make_batch(BATCH - k, g, POOL_TR, "gold", SHOTS_TRAIN))
        tok = torch.cat([p[0] for p in parts]); ans = torch.cat([p[1] for p in parts])
        lr = LR * min(1.0, (step + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * step / STEPS))
        for grp in opt.param_groups:
            grp["lr"] = lr
        loss = F.cross_entropy(model(tok), ans)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model, loss.item()


@torch.no_grad()
def evaluate(model, seed):
    """Accuracy (%) for every condition and number of demonstrations per class."""
    res = {}
    for shots in SHOTS_EVAL:
        for cond in CONDS:
            if cond == "none" and shots != SHOTS_EVAL[0]:
                res[(cond, shots)] = res[(cond, SHOTS_EVAL[0])]
                continue
            g = torch.Generator().manual_seed(90_000 + 100 * shots + CONDS.index(cond))
            tok, ans, qc, oc = make_batch(N_EVAL, g, POOL_TE, cond, shots)
            pred = model(tok).argmax(1)
            res[(cond, shots)] = 100.0 * (pred == ans).float().mean().item()
            if cond == "flipped":
                # how often the network outputs the label the demonstrations assign to its class
                res[("follows", shots)] = 100.0 * (pred == oc).float().mean().item()
    return res


def main():
    t0 = time.time()
    m0 = TinyTransformer()
    print(f"transformer parameters: {sum(p.numel() for p in m0.parameters()):,}")
    R = {}
    for q in QS:
        for seed in SEEDS:
            t = time.time()
            model, loss = train(q, seed)
            R[(q, seed)] = evaluate(model, seed)
            r = R[(q, seed)]
            print(f"q={q:<5} seed {seed}: final loss {loss:.3f}  " +
                  "  ".join(f"{c} {r[(c, 4)]:.1f}" for c in CONDS) +
                  f"  ({time.time() - t:.1f}s)", flush=True)

    def tint(v):
        return lk.t_interval(np.asarray(v, dtype=float))

    print("\naccuracy (%) with four demonstrations per class, mean and 95% t-half-width over seeds:")
    print("  q      " + "  ".join(f"{c:>14s}" for c in CONDS))
    for q in QS:
        cells = [tint([R[(q, s)][(c, 4)] for s in SEEDS]) for c in CONDS]
        print(f"  {q:<6}" + "  ".join(f"{m:8.1f} +- {h:4.1f}" for m, h in cells))
    print("\ncost of random labels (gold - random, points) by demonstrations per class:")
    for q in QS:
        cells = [tint([R[(q, s)][("gold", k)] - R[(q, s)][("random", k)] for s in SEEDS]) for k in SHOTS_EVAL]
        print(f"  q={q:<5} " + "  ".join(f"{k}: {m:5.1f} +- {h:4.1f}" for k, (m, h) in zip(SHOTS_EVAL, cells)))
    print("\nflipped labels: share of answers that follow the demonstrations (%), 4 per class:")
    for q in QS:
        m, h = tint([R[(q, s)][("follows", 4)] for s in SEEDS])
        print(f"  q={q:<5} {m:5.1f} +- {h:4.1f}")
    # the average over 'models' that a survey would report
    allq = [R[(q, s)][("gold", 4)] - R[(q, s)][("random", 4)] for q in QS for s in SEEDS]
    lowq = [R[(q, s)][("gold", 4)] - R[(q, s)][("random", 4)] for q in QS[:1] for s in SEEDS]
    print(f"\naverage cost of random labels over all 15 networks: {np.mean(allq):.1f} points "
          f"(range {np.min(allq):.1f} to {np.max(allq):.1f}); q = 0 only: {np.mean(lowq):.1f}")

    # ---- data for the figures -----------------------------------------------------------
    rng = np.random.default_rng(15)
    runs = {"q": [], "x": [], "seed": []}
    runs.update({c: [] for c in CONDS})
    for qi, q in enumerate(QS):
        for s in SEEDS:
            runs["q"].append(q); runs["x"].append(qi + 1 + rng.uniform(-0.12, 0.12)); runs["seed"].append(s)
            for c in CONDS:
                runs[c].append(R[(q, s)][(c, 4)])
    lk.write_dat("figures/data/15-instruments-icl-runs.dat", runs)
    means = {"x": [qi + 1 for qi in range(len(QS))]}
    for c in ["none", "gold", "random", "flipped", "unrelated"]:
        mh = [tint([R[(q, s)][(c, 4)] for s in SEEDS]) for q in QS]
        means[c] = [m for m, _ in mh]; means[c + "hw"] = [h for _, h in mh]
    lk.write_dat("figures/data/15-instruments-icl-means.dat", means)
    shots = {"shots": SHOTS_EVAL}
    for qi, q in enumerate(QS):
        mh = [tint([R[(q, s)][("gold", k)] - R[(q, s)][("random", k)] for s in SEEDS]) for k in SHOTS_EVAL]
        shots[f"cost{qi}"] = [m for m, _ in mh]; shots[f"hw{qi}"] = [h for _, h in mh]
    lk.write_dat("figures/data/15-instruments-icl-shots.dat", shots)
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
