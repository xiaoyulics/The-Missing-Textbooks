"""Chapter 13, figure "equilibrium", panel (c): the angular update under a decaying learning-rate
schedule, exact norm recursion (theory) against a training run (simulation).

Model: the normalized linear layer of 13-decay-relaxation.py (labels independent of the inputs), d = 2048,
batch 8, trained by AdamW without memory (sign updates, |u|^2 = d) with decoupled weight decay,
theta <- (1 - eta_t lam) theta - eta_t sign(g). Schedules over T = 20000 steps with eta_max*lam = 1e-3:
cosine decay to zero, and warmup-free WSD (constant, then linear decay to zero over the last 20%).
The run starts at the equilibrium norm of eta_max.
Theory (Proposition prop:decay:adamw, in expectation): u_{t+1} = (1 - eta_t lam)^2 u_t + eta_t^2 d, and
tan(omega_t) = eta_t sqrt(d) / ((1 - eta_t lam) sqrt(u_t)). Quasi-static value: arccos(1 - eta_t lam).
Writes figures/data/13-decay-schedule.dat: root mean square angles over windows of 200 steps.
Runtime: a few seconds.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")
d, B, s = 2048, 8, 1.0
lam, eta_max, T, WIN = 0.1, 0.01, 20000, 200


def schedule(kind):
    t = np.arange(T)
    if kind == "cos":
        return eta_max * 0.5 * (1 + np.cos(np.pi * t / T))
    e = np.full(T, eta_max)
    k = int(0.8 * T)
    e[k:] = eta_max * (1 - (t[k:] - k) / (T - k))
    return e


def theory(etas, u0):
    u = np.empty(T + 1)
    u[0] = u0
    for t in range(T):
        u[t + 1] = (1 - etas[t] * lam) ** 2 * u[t] + etas[t] ** 2 * d
    return np.arctan(etas * np.sqrt(d) / ((1 - etas * lam) * np.sqrt(u[:-1])))


def simulate(etas, u0, seed):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=d)
    th *= np.sqrt(u0) / np.linalg.norm(th)
    ang = np.empty(T)
    for t in range(T):
        nt = np.linalg.norm(th)
        X = rng.normal(size=(B, d))
        y = s * rng.normal(size=B)
        th_hat = th / nt
        gs = X.T @ (X @ th_hat - y) / B
        g = (gs - th_hat * (th_hat @ gs)) / nt
        new = (1 - etas[t] * lam) * th - etas[t] * np.sign(g)
        ang[t] = np.arccos(np.clip(th @ new / (nt * np.linalg.norm(new)), -1, 1))
        th = new
    return ang


def windows(x):
    return np.sqrt(np.mean(x.reshape(-1, WIN) ** 2, axis=1))


def main():
    u0 = eta_max ** 2 * d / (1 - (1 - eta_max * lam) ** 2)
    cols = {}
    for kind, seed in (("cos", 21), ("wsd", 22)):
        etas = schedule(kind)
        cols[f"{kind}_eta"] = windows(etas) / eta_max
        cols[f"{kind}_quasi"] = windows(np.arccos(1 - etas * lam))
        cols[f"{kind}_rec"] = windows(theory(etas, u0))
        cols[f"{kind}_sim"] = windows(simulate(etas, u0, seed))
    mid = (np.arange(T // WIN) + 0.5) * WIN / T
    with open(os.path.join(OUT, "13-decay-schedule.dat"), "w") as f:
        keys = list(cols)
        f.write("tT " + " ".join(keys) + "\n")
        for i in range(len(mid)):
            f.write(f"{mid[i]:.4f} " + " ".join(f"{cols[k][i]:.6g}" for k in keys) + "\n")
    for kind in ("cos", "wsd"):
        r = cols[f"{kind}_sim"] / cols[f"{kind}_rec"]
        q = cols[f"{kind}_rec"] / cols[f"{kind}_quasi"]
        print(f"{kind}: sim/recursion in [{r.min():.3f}, {r.max():.3f}]; recursion/quasi-static at "
              f"t/T=0.5: {q[49]:.3f}, 0.9: {q[89]:.3f}, 0.99: {q[98]:.3f}")


if __name__ == "__main__":
    main()
