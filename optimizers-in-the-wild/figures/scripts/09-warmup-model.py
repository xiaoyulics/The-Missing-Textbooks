"""Chapter 9 (warmup), Figure 1: gradient descent on the warmup model.

The warmup model is L(x, z) = q(x) + (1/2) (K0 + K1 q(x)) |z|^2 with q(x) = (mu/2) |x|^2:
a slow coordinate x that carries the loss and a stiff coordinate z whose curvature
S = K0 + K1 q is set by the loss.  Gradient descent started at (x0, z0), with z0 tiny, is
compared with the exact recursion on the invariant line {z = 0},
Delta_{t+1} = (1 - mu eta_t)^2 Delta_t, and with the guarantee of the warmup theorem.

Writes figures/data/09-warmup-model.dat (curves and points, every 25 steps),
figures/data/09-warmup-model-nowarm.dat (the run without warmup, every step), and
figures/data/09-warmup-lengths.dat (steps to the target against the warmup length).
Also prints the other step counts quoted in the chapter. Deterministic (no randomness); runs in
well under a minute.
"""
import math
import os

from scipy.special import wrightomega

K0, K1, MU = 1.0, 1.0, 0.02          # kappa = K0/mu = 50, and K0 >= 2 mu
DELTA0 = 50.0                        # initial gap; u0 = K1 Delta0 / K0 = 50
Z0 = 1e-6                            # perturbation of the stiff coordinate
EPS = 1e-8                           # target gap
X0 = math.sqrt(2 * DELTA0 / MU)
ETA_PEAK = 1.0 / K0                  # peak step of the linear warmups
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def gd(schedule, T, z0=Z0, stop=None):
    """Gradient descent on the warmup model; schedule(t, loss) -> eta.
    Returns lists (loss, eta, sharpness S_t = K0 + K1 q(x_t)); stops at divergence,
    or when the loss falls below `stop`."""
    x, z = X0, z0
    loss, eta, sharp = [], [], []
    for t in range(T):
        q = 0.5 * MU * x * x
        S = K0 + K1 * q
        L = q + 0.5 * S * z * z
        if not math.isfinite(L) or L > 1e30:
            break
        e = schedule(t, L)
        loss.append(L); eta.append(e); sharp.append(S)
        if stop is not None and L <= stop:
            break
        x, z = x - e * MU * x * (1.0 + 0.5 * K1 * z * z), z - e * S * z
    return loss, eta, sharp


def manifold(schedule, T, D0=DELTA0, stop=None):
    """The recursion on {z = 0}: Delta_{t+1} = (1 - mu eta_t)^2 Delta_t."""
    D = D0
    loss, eta, sharp = [], [], []
    for t in range(T):
        e = schedule(t, D)
        loss.append(D); eta.append(e); sharp.append(K0 + K1 * D)
        if stop is not None and D <= stop:
            break
        D = (1 - MU * e) ** 2 * D
    return loss, eta, sharp


def steps_to(loss, eps=EPS):
    for t, v in enumerate(loss):
        if v <= eps:
            return t
    return None


def linear(Tw):
    return lambda t, L: ETA_PEAK * min(1.0, (t + 1) / Tw)


def adaptive(t, L):
    return 1.0 / (K0 + K1 * L)


def constant(t, L):
    return 1.0 / (K0 + K1 * DELTA0)


def nowarm(t, L):
    return ETA_PEAK


def minimal_stable_Tw(D0):
    """Shortest linear warmup to ETA_PEAK whose recursion never has eta_t S_t > 2
    (after the warmup eta is constant and S decreases, so the maximum occurs by t = Tw)."""
    def ok(Tw):
        D = D0
        for t in range(Tw + 1):
            e = ETA_PEAK * min(1.0, (t + 1) / Tw)
            if e * (K0 + K1 * D) > 2.0:
                return False
            D *= (1 - MU * e) ** 2
        return True
    lo, hi = 1, 200000
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def main():
    T = 7001
    kappa, u0 = K0 / MU, K1 * DELTA0 / K0
    Twstar = minimal_stable_Tw(DELTA0)
    print(f"minimal stable linear warmup: Tw* = {Twstar};"
          f" heuristic (kappa/8)(1+u0/sqrt(e))^2 = {kappa / 8 * (1 + u0 / math.sqrt(math.e)) ** 2:.0f}")
    for D0 in [5.0, 10.0, 20.0, 100.0, 200.0]:
        uu = K1 * D0 / K0
        print(f"   u0 = {uu:5.0f}: Tw* = {minimal_stable_Tw(D0)},"
              f" heuristic {kappa / 8 * (1 + uu / math.sqrt(math.e)) ** 2:.0f}")

    runs = {"ad": adaptive, "co": constant, "ls": linear(Twstar), "lc": linear(2000)}
    sim = {k: gd(s, T) for k, s in runs.items()}
    rec = {k: manifold(s, T) for k, s in runs.items()}
    for k, s in runs.items():
        print(f"{k}: steps to {EPS:g}: simulation {steps_to(gd(s, 10 ** 5, stop=EPS)[0])},"
              f" recursion {steps_to(manifold(s, 10 ** 5, stop=EPS)[0])},"
              f" largest loss {max(sim[k][0]):.4g}")
    # guarantee of the theorem: D_{t+1} = D_t - mu D_t / (K0 + K1 D_t)
    D = [DELTA0]
    for t in range(T - 1):
        D.append(D[-1] - MU * D[-1] / (K0 + K1 * D[-1]))
    bound = (K0 * math.log(DELTA0 / EPS) + K1 * (DELTA0 - EPS)) / MU
    print(f"guarantee of the theorem: {bound:.0f} steps; with 2 mu in place of mu: {bound / 2:.0f}")
    # continuum profile of the loss-indexed warmup on the model (rate 2 mu):
    # u + log u = u0 + log u0 - 2 mu t / K0, eta = 1 / (K0 (1 + u))
    eta_cont = [1.0 / (K0 * (1.0 + wrightomega(u0 + math.log(u0) - 2 * MU * t / K0).real))
                for t in range(T)]
    half = (K0 / (2 * MU)) * (u0 + math.log(u0) - 1)
    t_half = next(t for t, e in enumerate(sim["ad"][1]) if e >= 0.5 / K0)
    print(f"eta reaches 1/(2K0) at t = {t_half} (continuum {half:.0f})")

    def fmt(v):
        return f"{v:.6g}" if (v is not None and math.isfinite(v)) else "nan"

    def get(lst, t):
        return lst[t] if t < len(lst) else None

    cols = ["t", "adsim", "adrec", "guarantee", "cosim", "corec", "lssim", "lsrec",
            "lcsim", "lcrec", "etaad", "etacont", "limad", "etals", "limls"]
    with open(os.path.join(OUT, "09-warmup-model.dat"), "w") as f:
        f.write(" ".join(cols) + "\n")
        for t in range(0, T, 25):
            vals = [t, get(sim["ad"][0], t), get(rec["ad"][0], t), D[t],
                    get(sim["co"][0], t), get(rec["co"][0], t),
                    get(sim["ls"][0], t), get(rec["ls"][0], t),
                    get(sim["lc"][0], t), get(rec["lc"][0], t),
                    get(sim["ad"][1], t), eta_cont[t], 2.0 / sim["ad"][2][t],
                    get(sim["ls"][1], t), 2.0 / sim["ls"][2][t]]
            f.write(" ".join(fmt(v) for v in vals) + "\n")

    loss_nw = gd(nowarm, 20)[0]
    with open(os.path.join(OUT, "09-warmup-model-nowarm.dat"), "w") as f:
        f.write("t loss\n")
        for t, v in enumerate(loss_nw):
            f.write(f"{t} {v:.6g}\n")
    print("no warmup, eta = 1/K0: loss", " ".join(f"{v:.3g}" for v in loss_nw))

    # steps to the target against the length of a linear warmup to 1/K0
    with open(os.path.join(OUT, "09-warmup-lengths.dat"), "w") as f:
        f.write("Tw simsteps recsteps maxloss\n")
        Tws = sorted({int(round(300 * (30000 / 300) ** (i / 30))) for i in range(31)})
        for Tw in Tws:
            ls = gd(linear(Tw), 10 ** 5, stop=EPS)[0]
            lr = manifold(linear(Tw), 10 ** 5, stop=EPS)[0]
            ss = steps_to(ls)
            f.write(f"{Tw} {fmt(ss)} {steps_to(lr)} {max(ls):.6g}\n")
            print(f"   Tw = {Tw:6d}: simulation {ss}, recursion {steps_to(lr)}, largest loss {max(ls):.3g}")
    quoted_numbers()


def quoted_numbers():
    """Print the other step counts quoted in the chapter (nothing is written to disk)."""
    S0 = K0 + K1 * DELTA0
    Phi = lambda D: K0 * math.log(D) + K1 * D
    dPhi = Phi(DELTA0) - Phi(EPS)
    # Proposition (warmup is optimal among stable schedules), eps = 1e-8
    for c in (1.0, 2.0):
        s = steps_to(gd(lambda t, L: c / (K0 + K1 * L), 10 ** 5, stop=EPS)[0])
        print(f"loss-indexed c = {c:g}: {s} steps; upper bound (ii) {dPhi / (c * MU * (2 - c * MU / K0)):.1f}")
    print(f"lower bound (i) for stable schedules: {(1 - 2 * MU / K0) ** 2 * dPhi / (4 * MU):.1f}")
    s = steps_to(gd(lambda t, L: 2.0 / S0, 10 ** 6, stop=EPS)[0])
    print(f"best stable constant step 2/S0: {s} steps;"
          f" lower bound (iii) {(1 - 2 * MU / S0) * S0 * math.log(DELTA0 / EPS) / (4 * MU):.1f}")
    Ds = [DELTA0]
    while len(Ds) < 10 ** 5:
        Ds.append(Ds[-1] - MU * Ds[-1] / (K0 + K1 * Ds[-1]))
    s = steps_to(gd(lambda t, L: 1.0 / (K0 + K1 * Ds[t]), 10 ** 5, stop=EPS)[0])
    print(f"scheduled warmup of the theorem, part (ii): {s} steps")
    # linear warmups to 1/K0 with spacing 10: divergence, catapults, step counts
    res = {}
    for Tw in range(10, 6200, 10):
        ls = gd(linear(Tw), 10 ** 5, stop=EPS)[0]
        res[Tw] = (steps_to(ls), max(ls), steps_to(manifold(linear(Tw), 10 ** 5, stop=EPS)[0]))
    short = [Tw for Tw in res if Tw < 1500]
    print(f"linear warmups shorter than 1500: {sum(res[Tw][0] is None for Tw in short)} of {len(short)} diverge")
    win = [Tw for Tw in res if 1500 <= Tw <= 3000 and res[Tw][0] is not None and res[Tw][1] > DELTA0 * (1 + 1e-9)]
    st = sorted(res[Tw][0] for Tw in win)
    med = st[len(st) // 2] if len(st) % 2 else (st[len(st) // 2 - 1] + st[len(st) // 2]) / 2
    print(f"catapults for Tw in [{min(win)}, {max(win)}] ({len(win)} runs): largest loss from"
          f" {min(res[Tw][1] for Tw in win):.3g} to {max(res[Tw][1] for Tw in win):.3g}; steps from {st[0]}"
          f" to {st[-1]}, median {med:g}; beat their recursion {sum(res[Tw][0] < res[Tw][2] for Tw in win)},"
          f" beat 1805: {sum(v < 1805 for v in st)}, beat 833: {sum(v < 833 for v in st)}")
    cat = [Tw for Tw in res if res[Tw][0] is not None and res[Tw][1] > DELTA0 * (1 + 1e-9)]
    print(f"all runs that survive a catapult: {len(cat)}, of which {sum(res[Tw][0] < 1805 for Tw in cat)}"
          f" beat the loss-indexed warmup")
    cross = [Tw for Tw in res if Tw >= 3000 and any(linear(Tw)(t, 0) * (K0 + K1 * D) > 2 for t, D in
                                                     enumerate(manifold(linear(Tw), Tw + 1)[0]))]
    print(f"Tw >= 3000 that cross the edge: up to {max(cross)}")
    # the instance of the proposition on constant steps: (mu/2) x^2 + L_c(z), eps = 1e-6
    eps, a = 1e-6, math.acosh(1 + K1 * DELTA0 / (2 * K0))
    etastar = 2 * a / (K0 * math.sinh(a))

    def cosh_gd(sched, T):
        x, z = math.sqrt(DELTA0 / MU), a / math.sqrt(K1)
        for t in range(T):
            if abs(math.sqrt(K1) * z) > 700:
                return None
            L = 0.5 * MU * x * x + K0 / K1 * (math.cosh(math.sqrt(K1) * z) - 1)
            if L <= eps:
                return t
            e = sched(t, L)
            x, z = x - e * MU * x, z - e * K0 / math.sqrt(K1) * math.sinh(math.sqrt(K1) * z)
        return None
    counts = [cosh_gd(lambda t, L: etastar * (1 - 10 ** (-k / 8)), 5000) for k in range(8, 49)]
    best = min(c for c in counts if c is not None)
    bound = ((K0 + K1 * DELTA0 / 2) * math.tanh(a) / (4 * MU * a) - 0.5) * math.log(DELTA0 / (2 * eps))
    print(f"constant-step instance: best constant step {best} steps (lower bound {bound:.1f}),"
          f" loss-indexed warmup {cosh_gd(lambda t, L: 1 / (K0 + K1 * L), 10 ** 5)},"
          f" theorem {(K0 * math.log(DELTA0 / eps) + K1 * (DELTA0 - eps)) / MU:.1f}, a coth a = {a / math.tanh(a):.3f}")


if __name__ == "__main__":
    main()
