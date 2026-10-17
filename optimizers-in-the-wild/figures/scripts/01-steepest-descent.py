"""Chapter 1 (steepest): the descent theorem against simulation (Figure 2).

Problem: the diagonal quadratic L(theta) = sum_i h_i theta_i^2 / 2 with d = 1000 and h_i = 1/i, so
that lambda_max(H) = 1 and tr H = H_1000 = 7.4855 (the l-infinity smoothness constant of a diagonal
quadratic); theta_0 has i.i.d. N(0,1) entries (numpy seed 9, a draw whose one-step ratio rho of
guaranteed decreases is the median of the ratios for seeds 0..199; the script checks this). L* = 0.

Output 1: figures/data/01-steepest-descent-onestep.dat  (one step from theta_0, g = H theta_0)
  eta       step length, in the norm of each method
  bdinf     guaranteed decrease of the sign step, eta |g|_1 - eta^2 trH / 2  (Theorem, one-step bound)
  acinf     actual decrease L(theta_0) - L(theta_0 - eta sign(g))
  bdtwo     guaranteed decrease of the normalized l2 step, eta |g|_2 - eta^2 lambda_max / 2
  actwo     actual decrease L(theta_0) - L(theta_0 - eta g/|g|_2)
Output 2: figures/data/01-steepest-descent-floor.dat  (sign descent theta <- theta - eta sign(g))
  T         number of steps (log-spaced, 1 ... 1e5)
  avgA, bdA running average (1/T) sum_{t<T} |grad L(theta_t)|_1 and the bound Delta_0/(eta T) + eta trH/2,
            for eta = 0.1;  avgB, bdB for eta = 0.01
  env       the bound at the best step for each horizon, sqrt(2 trH Delta_0 / T)
The script also prints the numbers quoted in the text. Deterministic; runs in about ten seconds.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
D = 1000


SEED = 9


def rho(h, th):
    g = h * th
    return (np.abs(g).sum() ** 2 / h.sum()) / (np.sum(g * g) / h.max())


def setup():
    h = 1.0 / np.arange(1, D + 1)
    ratios = [rho(h, np.random.default_rng(s).standard_normal(D)) for s in range(200)]
    theta0 = np.random.default_rng(SEED).standard_normal(D)
    print(f"rho over seeds 0..199: median {np.median(ratios):.3f}, quartiles {np.percentile(ratios, 25):.3f}, "
          f"{np.percentile(ratios, 75):.3f}; seed {SEED}: {rho(h, theta0):.3f}")
    return h, theta0


def loss(h, th):
    return 0.5 * np.sum(h * th * th)


def one_step(h, th0):
    g = h * th0
    trH, lmax = h.sum(), h.max()
    n1, n2 = np.abs(g).sum(), np.linalg.norm(g)
    q = np.sum(h * g * g) / n2 ** 2                  # curvature along g/|g|_2
    L0 = loss(h, th0)
    print(f"Delta_0 = {L0:.4f}, |g|_1 = {n1:.4f}, |g|_2 = {n2:.4f}, trH = {trH:.4f}, "
          f"density |g|_1^2/|g|_2^2 = {n1**2/n2**2:.2f}, effective rank trH/lmax = {trH/lmax:.3f}")
    print(f"guaranteed decreases: sign {n1**2/(2*trH):.4f} at eta = {n1/trH:.4f}; "
          f"l2 {n2**2/(2*lmax):.4f} at eta = {n2/lmax:.4f}; ratio rho = {(n1**2/trH)/(n2**2/lmax):.3f}")
    print(f"curvature along g/|g|_2: q = {q:.4f}; actual l2 decrease at eta = |g|_2: "
          f"{n2**2*(1 - q/2):.4f} (= {(1 - q/2)*2:.3f} x guaranteed); best l2 decrease {n2**2/(2*q):.4f}")
    etas = np.linspace(0.0, 2.4, 49)
    rows = []
    for e in etas:
        bdinf = e * n1 - 0.5 * e * e * trH
        acinf = L0 - loss(h, th0 - e * np.sign(g))
        bdtwo = e * n2 - 0.5 * e * e * lmax
        actwo = L0 - loss(h, th0 - e * g / n2)
        rows.append((e, bdinf, acinf, bdtwo, actwo))
    rows = np.array(rows)
    print("max |actual - bound| for the sign step:", np.abs(rows[:, 2] - rows[:, 1]).max())
    return rows


def floor(h, th0):
    trH = h.sum()
    D0 = loss(h, th0)
    Ts = np.unique(np.round(np.logspace(0, 5, 41)).astype(int))
    print(f"max_i |theta_0i| = {np.abs(th0).max():.4f}; crossover T = 2 Delta_0/(trH eta^2) for eta = 0.1, 0.01:",
          [round(2 * D0 / (trH * e * e)) for e in (0.1, 0.01)])
    cols = []
    for eta in (0.1, 0.01):
        th = th0.copy()
        S, k = 0.0, 0
        avg = np.empty(len(Ts))
        for t in range(1, Ts[-1] + 1):
            g = h * th
            S += np.abs(g).sum()
            th -= eta * np.sign(g)
            if t == Ts[k]:
                avg[k] = S / t
                k += 1
        bd = D0 / (eta * Ts) + eta * trH / 2
        gap = D0 / (eta * Ts[-1]) - loss(h, th) / (eta * Ts[-1])    # identity: avg = (D0 - D_T)/(eta T) + eta trH/2
        print(f"eta = {eta}: avg at T = 1e5 is {avg[-1]:.6g}, bound {bd[-1]:.6g}, floor eta trH/2 = {eta*trH/2:.6g}; "
              f"identity check {gap + eta*trH/2:.6g}; relative gap {(bd[-1]-avg[-1])/avg[-1]:.2e}; "
              f"avg/bound at T = 10, 100, 1000: {avg[Ts.searchsorted(10)]/bd[Ts.searchsorted(10)]:.3f}, "
              f"{avg[Ts.searchsorted(100)]/bd[Ts.searchsorted(100)]:.3f}, {avg[Ts.searchsorted(1000)]/bd[Ts.searchsorted(1000)]:.4f}")
        cols += [avg, bd]
    cols.append(np.sqrt(2 * trH * D0 / Ts))
    return Ts, cols


def main():
    os.makedirs(DATA, exist_ok=True)
    h, th0 = setup()
    rows = one_step(h, th0)
    path = os.path.join(DATA, "01-steepest-descent-onestep.dat")
    with open(path, "w") as f:
        f.write("eta bdinf acinf bdtwo actwo\n")
        for r in rows:
            f.write(" ".join(f"{v:.6f}" for v in r) + "\n")
    print("wrote", os.path.normpath(path))
    Ts, cols = floor(h, th0)
    path = os.path.join(DATA, "01-steepest-descent-floor.dat")
    with open(path, "w") as f:
        f.write("T avgA bdA avgB bdB env\n")
        for i, T in enumerate(Ts):
            f.write(f"{T} " + " ".join(f"{c[i]:.6e}" for c in cols) + "\n")
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
