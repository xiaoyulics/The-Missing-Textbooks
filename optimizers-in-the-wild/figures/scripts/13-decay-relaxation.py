"""Chapter 13, figure "equilibrium", panel (a): relaxation of the squared weight norm to the
rotational equilibrium, exact recursions (theory) against training runs (simulation).

Model: a normalized linear layer f(x) = <x, theta/|theta|> on Gaussian inputs x in R^d, trained on
labels that are independent of the inputs (y ~ N(0, s^2)), so that the stochastic gradient on the
unit sphere is pure noise with E|g|^2 |theta|^2 = G^2 = (d-1)(1+s^2)/B exactly, whatever the direction.
The loss is scale invariant, so every minibatch gradient is orthogonal to theta.

Optimizers, both with decoupled weight decay theta <- (1 - eta*lam) theta - eta*u:
  SGD:                         u = g,        theory u_{t+1} = a u_t + eta^2 G^2 / u_t   (Theorem thm:decay:equilibrium)
  AdamW without memory (sign): u = sign(g),  theory E u_{t+1} = a E u_t + eta^2 d       (Proposition prop:decay:adamw)
with a = (1 - eta*lam)^2 and u_t = |theta_t|^2, started at 16 u* and u*/16.
Writes figures/data/13-decay-relaxation-theory.dat (every 20 steps) and
figures/data/13-decay-relaxation-sim.dat (every 160 steps). Runtime: a few seconds.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data")

d, B, s = 1024, 8, 1.0
el = 1e-3                      # eta * lambda for both optimizers
T = 4000                       # = 4 / (eta * lambda)
a = (1 - el) ** 2
G2 = (d - 1) * (1 + s ** 2) / B
configs = {                    # (eta, lambda)
    "sgd": (0.1, el / 0.1),
    "sgn": (0.01, el / 0.01),
}


def ustar(opt):
    eta, lam = configs[opt]
    if opt == "sgd":
        return eta * np.sqrt(G2) / np.sqrt(1 - a)
    return eta ** 2 * d / (1 - a)


def theory(opt, ratio0):
    eta, lam = configs[opt]
    us = ustar(opt)
    u = np.empty(T + 1)
    u[0] = ratio0 * us
    for t in range(T):
        if opt == "sgd":
            u[t + 1] = a * u[t] + eta ** 2 * G2 / u[t]
        else:
            u[t + 1] = a * u[t] + eta ** 2 * d
    return u / us


def simulate(opt, ratio0, seed):
    eta, lam = configs[opt]
    rng = np.random.default_rng(seed)
    th = rng.normal(size=d)
    th *= np.sqrt(ratio0 * ustar(opt)) / np.linalg.norm(th)
    u = np.empty(T + 1)
    for t in range(T + 1):
        nt = np.linalg.norm(th)
        u[t] = nt ** 2
        if t == T:
            break
        X = rng.normal(size=(B, d))
        y = s * rng.normal(size=B)
        th_hat = th / nt
        gs = X.T @ (X @ th_hat - y) / B          # gradient with respect to the unit vector
        g = (gs - th_hat * (th_hat @ gs)) / nt    # exact gradient with respect to theta
        step = g if opt == "sgd" else np.sign(g)
        th = (1 - eta * lam) * th - eta * step
    return u / ustar(opt)


def main():
    cols_th, cols_sim = {}, {}
    for opt in ("sgd", "sgn"):
        for tag, r0 in (("hi", 16.0), ("lo", 1 / 16)):
            cols_th[f"{opt}_{tag}"] = theory(opt, r0)
            cols_sim[f"{opt}_{tag}"] = simulate(opt, r0, seed=11 if tag == "hi" else 12)
    t = np.arange(T + 1)
    with open(os.path.join(OUT, "13-decay-relaxation-theory.dat"), "w") as f:
        keys = list(cols_th)
        f.write("s " + " ".join(keys) + "\n")
        for i in range(0, T + 1, 20):
            f.write(f"{el * t[i]:.4f} " + " ".join(f"{cols_th[k][i]:.6g}" for k in keys) + "\n")
    with open(os.path.join(OUT, "13-decay-relaxation-sim.dat"), "w") as f:
        keys = list(cols_sim)
        f.write("s " + " ".join(keys) + "\n")
        for i in list(range(0, T + 1, 160)):
            f.write(f"{el * t[i]:.4f} " + " ".join(f"{cols_sim[k][i]:.6g}" for k in keys) + "\n")
    # report the agreement
    for k in cols_th:
        idx = np.arange(0, T + 1, 160)
        dev = np.max(np.abs(np.log(cols_sim[k][idx] / cols_th[k][idx])))
        print(f"{k}: max |log(sim/theory)| on plotted points = {dev:.4f}; final sim {cols_sim[k][-1]:.4f}")
    print("SGD relaxation rate near equilibrium 2a-1 =", 2 * a - 1, "; AdamW-model rate a =", a)


if __name__ == "__main__":
    main()
